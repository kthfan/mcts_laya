"""The AlphaZero / Expert-Iteration loop around a Laya agent.

    teacher warm start (optional)
    repeat:
        self-play with search  ->  replay buffer  ->  train Laya  ->  calibrate  ->  evaluate
        (optional gate: keep the new weights only if the gated metric did not drop)

With `parallel.workers > 0` (the default on a GPU) self-play and evaluation episodes are played by
actor processes that share one batched model (`mcts_laya.selfplay.parallel`); teacher data,
training and calibration stay in this process.

Every stage writes JSON lines to `<output_dir>/metrics.jsonl`; `summary.md` is rewritten after each
iteration with the learning curve and the Phase 0 milestone checks.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from . import progress
from .config import EvalSearch, ExperimentConfig, config_to_dict
from .evaluation import evaluate_searcher, format_table, success_moves, summarize_episodes
from .evaluators.base import Evaluator
from .evaluators.laya_evaluator import LayaEvaluator, load_agent
from .registry import ENVIRONMENTS, EVALUATORS, SEARCHERS, TEACHERS
from .runtime import available_cpus
from .selfplay.actor import Episode, SelfPlayConfig, episode_to_samples, play_episode
from .selfplay.parallel import ActorPool, episode_task, resolve_workers, search_spec
from .selfplay.targets import policy_weights, relabel_with_teacher, update_best_moves
from .training.sample import ReplayBuffer, save_samples
from .training.trainer import LayaTrainer

log = logging.getLogger("mcts_laya")


class AlphaZeroLoop:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / "config.yaml", "w") as f:
            yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False, allow_unicode=True)
        metrics = self.out / "metrics.jsonl"
        if metrics.exists():  # a run always starts from scratch; keep an interrupted attempt aside
            metrics.replace(self.out / "metrics.interrupted.jsonl")
        self.py_rng = random.Random(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)
        import torch

        torch.manual_seed(cfg.seed)  # training noise (dropout, RLCD sampling) must be reproducible
        self.env = ENVIRONMENTS.build(cfg.env.name, **cfg.env.params)

        m = cfg.model
        if m.build_tiny and not (Path(m.checkpoint) / "model.safetensors").exists():
            from .models.tiny import build_tiny_checkpoint

            build_tiny_checkpoint(m.checkpoint, **m.tiny_params)
        self.agent = load_agent(m.checkpoint, m.device, m.subfolder, m.revision, m.max_len, m.head_max_len)
        e = cfg.evaluator
        self.evaluator = LayaEvaluator(self.agent, max_rows=e.max_rows, cache_size=e.cache_size,
                                       prior_temperature=e.prior_temperature, option_order=e.option_order,
                                       seed=cfg.seed)
        self.trainer = LayaTrainer(self.agent, cfg.train.config)
        device = str(next(self.agent.model.parameters()).device)
        self.workers = resolve_workers(cfg.parallel.workers, device, available_cpus())
        self._actors: Optional[ActorPool] = None
        self.replay = ReplayBuffer(cfg.train.replay_capacity)
        eval_rng = random.Random(cfg.eval.seed)
        self.eval_problems = self.env.sample_problems(eval_rng, cfg.eval.problems, split="eval")
        self.val_problems = None
        if cfg.eval.gate_problems > 0:
            self.val_problems = self.env.sample_problems(random.Random(cfg.eval.gate_seed), cfg.eval.gate_problems,
                                                         split=cfg.eval.gate_split)
        if set(cfg.eval.extra_splits) & {"eval", "val"}:
            raise ValueError("eval.extra_splits names further test splits; 'eval' and 'val' are measured already")
        self.extra_problems = {split: self.env.sample_problems(random.Random(cfg.eval.seed), cfg.eval.problems,
                                                               split=split)
                               for split in cfg.eval.extra_splits}
        self._kept_test: Optional[Dict[str, Dict[str, float]]] = None  # test results of the kept weights
        self.history: List[Dict[str, Any]] = []
        self.best_metric: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.best_moves: Dict[Any, int] = {}  # fewest moves the agent has solved each training problem in
        self._relabel_teacher = None
        from .selfplay.targets import POLICY_FILTERS

        if cfg.selfplay.policy_filter not in POLICY_FILTERS:
            raise ValueError(f"selfplay.policy_filter must be one of {POLICY_FILTERS}")
        if cfg.selfplay.relabel not in ("none", "teacher"):
            raise ValueError("selfplay.relabel must be 'none' or 'teacher'")
        if cfg.selfplay.relabel == "teacher":
            self._relabel_teacher = TEACHERS.build(cfg.env.name, self.env)

    # --- helpers --------------------------------------------------------------------------
    def actors(self) -> Optional[ActorPool]:
        """The actor processes (started on first use), or None for in-process episodes."""
        if self.workers <= 0:
            return None
        if self._actors is None:
            p = self.cfg.parallel
            log.info("starting %d actor processes", self.workers)
            if not str(self.agent.device).startswith("cpu"):
                import torch

                # the actors own the CPU cores; with the model on a GPU this process only collates
                # rows and launches kernels, and a wide CPU thread pool would just compete with them
                torch.set_num_threads(min(torch.get_num_threads(), 2))
            self._actors = ActorPool(self.cfg.env.name, self.cfg.env.params, self.evaluator, self.workers,
                                     max_wait_ms=p.max_wait_ms, cache_size=p.cache_size,
                                     episode_timeout_s=p.episode_timeout_s, memory_gb=p.actor_memory_gb)
        return self._actors

    def close(self) -> None:
        if self._actors is not None:
            self._actors.close()
            self._actors = None

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record, time=time.time())
        self.history.append(record)
        with open(self.out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        log.info(json.dumps(record))

    def _tag(self, iteration: int) -> str:
        return f"it {iteration}/{self.cfg.iterations}"

    def _searcher(self, name: str, evaluator: Evaluator, params: Dict[str, Any]):
        return SEARCHERS.build(name, self.env, evaluator, **params)

    def _eval_evaluator(self, spec: EvalSearch) -> Evaluator:
        if spec.evaluator == "model":
            return self.evaluator
        return EVALUATORS.build(spec.evaluator, **spec.evaluator_params)

    def evaluate(self, iteration: int, stage: str, specs: List[EvalSearch], split: str = "eval") -> Dict[str, Dict[str, float]]:
        """Run `specs` on the test problems (`split="eval"`), the gate's validation problems ("val") or an
        extra test split (`eval.extra_splits`)."""
        if split == "val":
            problems, seed = self.val_problems, self.cfg.eval.gate_seed
        else:
            problems, seed = self.extra_problems.get(split, self.eval_problems), self.cfg.eval.seed
        results = {}
        for spec in specs:
            kind = {"val": "gate", "eval": "eval"}.get(split, f"eval {split}")
            desc = f"{self._tag(iteration)} {kind} {stage} {spec.label}"
            pool = self.actors()
            if pool is not None:
                cfg = SelfPlayConfig(max_moves=self.cfg.eval.max_moves, add_noise=False, action_selection="search")
                ss = search_spec(spec.name, spec.params, spec.evaluator, spec.evaluator_params)
                tasks = [episode_task(ss, p, np.random.default_rng([seed, i]).integers(2 ** 62), cfg)
                         for i, p in enumerate(problems)]
                t0 = time.time()
                played = pool.run(tasks, desc)
                failed = sum(ep is None for ep in played)
                # a failed episode (actor error / timeout, see ActorPool) counts as a loss at the move cap
                episodes = [ep if ep is not None else Episode(initial_state=p, final_value=-1.0)
                            for ep, p in zip(played, problems)]
                r = summarize_episodes(episodes, time.time() - t0, self.env)
                if failed:
                    r["failed_episodes"] = failed
                    r["moves"] = float(np.mean([ep.length if ep is not None else self.cfg.eval.max_moves
                                                for ep in played]))
            else:
                searcher = self._searcher(spec.name, self._eval_evaluator(spec), spec.params)
                r = evaluate_searcher(self.env, searcher, problems, seed=seed, max_moves=self.cfg.eval.max_moves,
                                      desc=desc)
            results[spec.label] = r
            self.log({"kind": "eval", "iteration": iteration, "stage": stage, "label": spec.label, "split": split, **r})
        return results

    def evaluate_extra(self, iteration: int, stage: str) -> None:
        """The current (kept) weights on each of `eval.extra_splits`."""
        for split in self.cfg.eval.extra_splits:
            self.evaluate(iteration, stage, self.cfg.eval.searches, split=split)

    def _gate_spec(self) -> EvalSearch:
        label = self.cfg.gate.metric.partition(".")[0]
        for spec in self.cfg.eval.searches:
            if spec.label == label:
                return spec
        raise KeyError(f"gate.metric {self.cfg.gate.metric!r} names no eval search label")

    def checkpoint_eval(self, iteration: int, stage: str, test: bool = True) -> None:
        """Gate the current weights, then measure the kept ones on the test problems.

        Legacy (no validation split): the test evaluation is also what the gate reads, so the logged
        result is the candidate's, even when the gate then reverts it.
        """
        cfg = self.cfg
        if self.val_problems is None:
            if test:
                res = self.evaluate(iteration, stage, cfg.eval.searches)
                if cfg.gate.enabled:
                    self._apply_gate(iteration, res)
            return
        accepted = True
        if cfg.gate.enabled:
            accepted = self._apply_gate(iteration, self.evaluate(iteration, stage, [self._gate_spec()], split="val"),
                                        split="val")
            if accepted:
                self._kept_test = None  # new weights: their test results are not known yet
        if not test:
            return
        if not accepted and self._kept_test is not None:
            # reverted to weights already measured: same model, same problems and seeds
            for label, r in self._kept_test.items():
                self.log({"kind": "eval", "iteration": iteration, "stage": stage, "label": label, "split": "eval",
                          **r, "reused": True})
            return
        self._kept_test = self.evaluate(iteration, stage, cfg.eval.searches)

    def _gate_value(self, results: Dict[str, Dict[str, float]]) -> Optional[float]:
        if not self.cfg.gate.metric:
            return None
        label, _, metric = self.cfg.gate.metric.partition(".")
        return results.get(label, {}).get(metric)

    def _apply_gate(self, iteration: int, results: Dict[str, Dict[str, float]], split: str = "eval") -> bool:
        value = self._gate_value(results)
        if value is None:
            return True
        accepted = (not self.cfg.gate.enabled or self.best_metric is None
                    or value >= self.best_metric - self.cfg.gate.tolerance)
        if accepted:
            if self.best_metric is None or value >= self.best_metric:
                self.best_metric = value
            if self.cfg.gate.enabled:
                self.best_state = copy.deepcopy({k: v.cpu() for k, v in self.agent.model.state_dict().items()})
                self.best_temps = list(self.agent.temperature)
        else:
            self.agent.model.load_state_dict(self.best_state)
            self.agent.temperature = list(self.best_temps)
            self.agent.cfg["temperature"] = list(self.best_temps)
            self.evaluator.clear_cache()
        self.log({"kind": "gate", "iteration": iteration, "value": value, "best": self.best_metric,
                  "accepted": accepted, "split": split})
        return accepted

    def _save(self, iteration: int, final: bool = False) -> None:
        mode = self.cfg.save_checkpoints
        if mode == "none" or (mode == "last" and not final):
            return
        name = "final" if final else f"iter_{iteration:03d}"
        self.trainer.save(str(self.out / "checkpoints" / name), half=self.cfg.save_half)

    # --- stages ---------------------------------------------------------------------------
    def warm_start(self) -> None:
        t = self.cfg.teacher
        teacher = TEACHERS.build(self.cfg.env.name, self.env, explore=t.explore)
        workers = t.workers
        if isinstance(workers, str) and workers == "auto":
            workers = min(available_cpus() - 1, 32)
        samples = teacher.generate(self.py_rng, t.problems, desc="it 0 teacher data", workers=int(workers or 0),
                                   env_spec=(self.cfg.env.name, dict(self.cfg.env.params)))
        save_samples(str(self.out / "teacher_samples.jsonl"), samples)
        metrics = self.trainer.train(samples, epochs=t.epochs, desc="it 0 warm-start train")
        self.evaluator.clear_cache()
        self.log({"kind": "train", "iteration": 0, "stage": "warmstart", "samples": len(samples), **metrics})
        if self.cfg.train.teacher_mix > 0:
            k = int(len(samples) * self.cfg.train.teacher_mix)
            self.replay.add(self.py_rng.sample(samples, min(k, len(samples))))

    def self_play(self, iteration: int) -> None:
        sp = self.cfg.selfplay
        t0, rows0 = time.time(), self.evaluator.rows_evaluated
        desc = f"{self._tag(iteration)} self-play"
        pool = self.actors()
        before = pool.snapshot() if pool is not None else None
        if pool is not None:
            spec = search_spec(self.cfg.search.name, self.cfg.search.params)
            tasks = [episode_task(spec, self.env.sample_problem(self.py_rng, "train"),
                                  self.np_rng.integers(2 ** 62), sp.config)
                     for _ in range(sp.episodes_per_iteration)]
            episodes = [ep for ep in pool.run(tasks, desc) if ep is not None]  # failed episodes are dropped
            if not episodes:
                log.warning("iteration %d: every self-play episode failed; nothing to learn from", iteration)
                return
        else:
            searcher = self._searcher(self.cfg.search.name, self.evaluator, self.cfg.search.params)
            episodes = []
            bar = progress.bar(sp.episodes_per_iteration, desc, "episodes")
            for _ in range(sp.episodes_per_iteration):
                episodes.append(play_episode(self.env, searcher, self.env.sample_problem(self.py_rng, "train"),
                                             self.np_rng, sp.config))
                bar.update(1)
                bar.set_postfix(success=float(np.mean([e.success for e in episodes])),
                                moves=float(np.mean([e.length for e in episodes])))
            bar.close()
        update_best_moves(self.env, episodes, self.best_moves)
        weights = policy_weights(self.env, episodes, sp.policy_filter, self.best_moves,
                                 failed_weight=sp.config.failed_policy_weight,
                                 best_known_ratio=sp.best_known_ratio, top_fraction=sp.top_fraction)
        n_samples = 0
        for ep, w in zip(episodes, weights):
            samples = episode_to_samples(self.env, ep, sp.config, meta={"iteration": iteration}, policy_weight=w)
            if self._relabel_teacher is not None:
                samples = relabel_with_teacher(self.env, self._relabel_teacher, ep, samples, sp.relabel_value)
            if sp.add_to_replay:
                self.replay.add(samples)
                n_samples += len(samples)
        dt = time.time() - t0
        kept = [ep for ep, w in zip(episodes, weights) if w > 0]
        self.log({"kind": "selfplay", "iteration": iteration, "episodes": len(episodes),
                  "success": float(np.mean([e.success for e in episodes])),
                  "moves": float(np.mean([e.length for e in episodes])),
                  "success_moves": success_moves(episodes),
                  "policy_episodes": len(kept) if self._relabel_teacher is None else len(episodes),
                  "policy_moves": float(np.mean([e.length for e in kept])) if kept else None,
                  "samples": n_samples, "replay": len(self.replay), "seconds": dt,
                  "rows_per_second": (self.evaluator.rows_evaluated - rows0) / max(dt, 1e-9),
                  **(pool.stats_since(before, dt) if pool is not None else {"workers": 0})})

    def train(self, iteration: int) -> None:
        batch = self.replay.sample(self.cfg.train.samples_per_iteration, self.py_rng)
        metrics = self.trainer.train(batch, desc=f"{self._tag(iteration)} train")
        self.evaluator.clear_cache()
        self.log({"kind": "train", "iteration": iteration, "stage": "selfplay", "samples": len(batch), **metrics})

    # --- driver ---------------------------------------------------------------------------
    def run(self) -> List[Dict[str, Any]]:
        try:
            return self._run()
        finally:
            self.close()

    def _run(self) -> List[Dict[str, Any]]:
        cfg = self.cfg
        if cfg.eval.baselines:
            self.evaluate(0, "baseline", cfg.eval.baselines)
        self.checkpoint_eval(0, "initial")
        if cfg.teacher.enabled:
            self.warm_start()
            self.best_metric = None  # the warm start defines the gate's reference
            self.checkpoint_eval(0, "warmstart")
            self.evaluate_extra(0, "warmstart")
        self.write_summary()
        for it in range(1, cfg.iterations + 1):
            self.self_play(it)
            self.train(it)
            self.checkpoint_eval(it, "selfplay", test=it % cfg.eval.every == 0 or it == cfg.iterations)
            self._save(it)
            self.write_summary()
        self._save(cfg.iterations, final=True)
        if cfg.iterations > 0:
            self.evaluate_extra(cfg.iterations, "final")
        self.write_summary()
        (self.out / "done.json").write_text(json.dumps({"finished": time.time(), "iterations": cfg.iterations}))
        return self.history

    # --- reporting ------------------------------------------------------------------------
    def curve(self) -> List[Dict[str, Any]]:
        """One row per (iteration, stage) with `<label>.<metric>` columns."""
        rows: Dict[tuple, Dict[str, Any]] = {}
        for r in self.history:
            if r.get("kind") != "eval" or r["stage"] == "baseline" or r.get("split", "eval") != "eval":
                continue
            key = (r["iteration"], r["stage"])
            row = rows.setdefault(key, {"iteration": r["iteration"], "stage": r["stage"]})
            row[f"{r['label']}.success"] = r["success"]
            row[f"{r['label']}.reward"] = r["reward"]
            row[f"{r['label']}.moves"] = r["moves"]
            if r.get("success_moves") is not None:
                row[f"{r['label']}.success_moves"] = r["success_moves"]
        return list(rows.values())

    def milestones(self) -> List[Dict[str, Any]]:
        m = self.cfg.milestone
        if not m.search_label:
            return []
        curve = [r for r in self.curve() if r["stage"] in ("warmstart", "selfplay")]
        key = f"{m.search_label}.{m.metric}"
        series = [r[key] for r in curve if key in r]
        last = curve[-1] if curve else {}
        baseline = {r["label"]: r for r in self.history if r.get("kind") == "eval" and r["stage"] == "baseline"}
        checks = []
        if len(series) >= 2:
            checks.append({"check": f"{key} improves over self-play iterations",
                           "value": f"{series[0]:.3f} -> {series[-1]:.3f}",
                           "pass": series[-1] > series[0]})
        if m.greedy_label and key in last and f"{m.greedy_label}.{m.metric}" in last:
            g = last[f"{m.greedy_label}.{m.metric}"]
            checks.append({"check": f"{m.search_label} beats {m.greedy_label} (no search)",
                           "value": f"{last[key]:.3f} vs {g:.3f}", "pass": last[key] > g})
        if m.baseline_label in baseline and key in last:
            b = baseline[m.baseline_label][m.metric]
            checks.append({"check": f"{m.search_label} beats {m.baseline_label} (same budget, no Laya)",
                           "value": f"{last[key]:.3f} vs {b:.3f}", "pass": last[key] > b})
        return checks

    def _category_table(self) -> List[str]:
        """Success per category (ALFWorld task type) of the latest evaluation of each split and label."""
        latest: Dict[tuple, Dict[str, Any]] = {}
        for r in self.history:
            if r.get("kind") == "eval" and r.get("by_category") and r.get("split", "eval") != "val":
                latest[(r.get("split", "eval"), r["label"])] = r
        if not latest:
            return []
        cats = sorted({c for r in latest.values() for c in r["by_category"]})
        rows = [{"split": split, "stage": f"{r['stage']} (it {r['iteration']})", "label": label, "all": r["success"],
                 **r["by_category"]} for (split, label), r in sorted(latest.items())]
        return ["## Success by category (latest evaluation)", "",
                format_table(rows, ["split", "stage", "label", "all"] + cats), ""]

    def write_summary(self) -> None:
        lines = [f"# {self.cfg.name}", "", f"- env: `{self.cfg.env.name}` {self.cfg.env.params}",
                 f"- checkpoint: `{self.cfg.model.checkpoint}`", f"- search: `{self.cfg.search.name}` "
                 f"{self.cfg.search.params}", ""]
        baseline = [r for r in self.history if r.get("kind") == "eval" and r["stage"] == "baseline"]
        if baseline:
            lines += ["## Baselines (no Laya)", "",
                      format_table(baseline, ["label", "success", "reward", "moves", "seconds"]), ""]
        curve = self.curve()
        if curve:
            cols = ["iteration", "stage"] + sorted({k for r in curve for k in r if "." in k})
            lines += ["## Learning curve", "", format_table(curve, cols), ""]
        extra = [dict(r, success_moves=r.get("success_moves") or "") for r in self.history
                 if r.get("kind") == "eval" and r.get("split") in self.cfg.eval.extra_splits]
        if extra:
            lines += ["## Other test splits", "", format_table(extra, ["iteration", "stage", "split", "label", "success",
                                                                       "reward", "moves", "success_moves"]), ""]
        lines += self._category_table()
        sp = [r for r in self.history if r.get("kind") == "selfplay"]
        if sp:
            lines += ["## Self-play", "",
                      format_table(sp, ["iteration", "episodes", "success", "moves", "success_moves", "policy_episodes",
                                        "policy_moves", "samples", "seconds", "rows_per_second", "workers",
                                        "rows_per_batch", "model_busy", "model_ms_per_batch",
                                        "actor_wait"]), ""]
        tr = [r for r in self.history if r.get("kind") == "train"]
        if tr:
            cols = ["iteration", "stage", "samples", "policy_ce", "value_ce", "heldout_policy_top1",
                    "heldout_value_brier", "temperature_choice", "temperature_noul"]
            lines += ["## Training", "", format_table(tr, cols), ""]
        gates = [r for r in self.history if r.get("kind") == "gate"]
        if gates and self.val_problems is not None:
            lines += [f"## Gate ({self.cfg.gate.metric} on {len(self.val_problems)} {self.cfg.eval.gate_split} problems)",
                      "", format_table(gates, ["iteration", "value", "best", "accepted"]), ""]
        checks = self.milestones()
        if checks:
            rows = [{"check": c["check"], "value": c["value"], "result": "PASS" if c["pass"] else "FAIL"}
                    for c in checks]
            lines += ["## Milestone checks", "", format_table(rows, ["check", "value", "result"]), ""]
        (self.out / "summary.md").write_text("\n".join(lines))
