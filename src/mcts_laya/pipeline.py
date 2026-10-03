"""The AlphaZero / Expert-Iteration loop around a Laya agent.

    teacher warm start (optional)
    repeat:
        self-play with search  ->  replay buffer  ->  train Laya  ->  calibrate  ->  evaluate
        (optional gate: keep the new weights only if the gated metric did not drop)

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

from .config import EvalSearch, ExperimentConfig, config_to_dict
from .evaluation import evaluate_searcher, format_table
from .evaluators.base import Evaluator
from .evaluators.laya_evaluator import LayaEvaluator, load_agent
from .registry import ENVIRONMENTS, EVALUATORS, SEARCHERS, TEACHERS
from .selfplay.actor import episode_to_samples, play_episode
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
        self.replay = ReplayBuffer(cfg.train.replay_capacity)
        eval_rng = random.Random(cfg.eval.seed)
        self.eval_problems = [self.env.sample_problem(eval_rng) for _ in range(cfg.eval.problems)]
        self.history: List[Dict[str, Any]] = []
        self.best_metric: Optional[float] = None
        self.best_state: Optional[dict] = None

    # --- helpers --------------------------------------------------------------------------
    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record, time=time.time())
        self.history.append(record)
        with open(self.out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        log.info(json.dumps(record))

    def _searcher(self, name: str, evaluator: Evaluator, params: Dict[str, Any]):
        return SEARCHERS.build(name, self.env, evaluator, **params)

    def _eval_evaluator(self, spec: EvalSearch) -> Evaluator:
        if spec.evaluator == "model":
            return self.evaluator
        return EVALUATORS.build(spec.evaluator, **spec.evaluator_params)

    def evaluate(self, iteration: int, stage: str, specs: List[EvalSearch]) -> Dict[str, Dict[str, float]]:
        results = {}
        for spec in specs:
            searcher = self._searcher(spec.name, self._eval_evaluator(spec), spec.params)
            r = evaluate_searcher(self.env, searcher, self.eval_problems, seed=self.cfg.eval.seed,
                                  max_moves=self.cfg.eval.max_moves)
            results[spec.label] = r
            self.log({"kind": "eval", "iteration": iteration, "stage": stage, "label": spec.label, **r})
        return results

    def _gate_value(self, results: Dict[str, Dict[str, float]]) -> Optional[float]:
        if not self.cfg.gate.metric:
            return None
        label, _, metric = self.cfg.gate.metric.partition(".")
        return results.get(label, {}).get(metric)

    def _apply_gate(self, iteration: int, results: Dict[str, Dict[str, float]]) -> bool:
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
                  "accepted": accepted})
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
        samples = teacher.generate(self.py_rng, t.problems)
        save_samples(str(self.out / "teacher_samples.jsonl"), samples)
        metrics = self.trainer.train(samples, epochs=t.epochs)
        self.evaluator.clear_cache()
        self.log({"kind": "train", "iteration": 0, "stage": "warmstart", "samples": len(samples), **metrics})
        if self.cfg.train.teacher_mix > 0:
            k = int(len(samples) * self.cfg.train.teacher_mix)
            self.replay.add(self.py_rng.sample(samples, min(k, len(samples))))

    def self_play(self, iteration: int) -> None:
        sp = self.cfg.selfplay
        searcher = self._searcher(self.cfg.search.name, self.evaluator, self.cfg.search.params)
        t0, rows0 = time.time(), self.evaluator.rows_evaluated
        successes, lengths, n_samples = [], [], 0
        for i in range(sp.episodes_per_iteration):
            ep = play_episode(self.env, searcher, self.env.sample_problem(self.py_rng), self.np_rng, sp.config)
            samples = episode_to_samples(self.env, ep, sp.config, meta={"iteration": iteration})
            self.replay.add(samples)
            successes.append(ep.success)
            lengths.append(ep.length)
            n_samples += len(samples)
        dt = time.time() - t0
        self.log({"kind": "selfplay", "iteration": iteration, "episodes": sp.episodes_per_iteration,
                  "success": float(np.mean(successes)), "moves": float(np.mean(lengths)),
                  "samples": n_samples, "replay": len(self.replay), "seconds": dt,
                  "rows_per_second": (self.evaluator.rows_evaluated - rows0) / max(dt, 1e-9)})

    def train(self, iteration: int) -> None:
        batch = self.replay.sample(self.cfg.train.samples_per_iteration, self.py_rng)
        metrics = self.trainer.train(batch)
        self.evaluator.clear_cache()
        self.log({"kind": "train", "iteration": iteration, "stage": "selfplay", "samples": len(batch), **metrics})

    # --- driver ---------------------------------------------------------------------------
    def run(self) -> List[Dict[str, Any]]:
        cfg = self.cfg
        if cfg.eval.baselines:
            self.evaluate(0, "baseline", cfg.eval.baselines)
        res = self.evaluate(0, "initial", cfg.eval.searches)
        self._apply_gate(0, res) if cfg.gate.enabled else None
        if cfg.teacher.enabled:
            self.warm_start()
            res = self.evaluate(0, "warmstart", cfg.eval.searches)
            if cfg.gate.enabled:
                self.best_metric = None  # the warm start defines the reference
                self._apply_gate(0, res)
        self.write_summary()
        for it in range(1, cfg.iterations + 1):
            self.self_play(it)
            self.train(it)
            if it % cfg.eval.every == 0 or it == cfg.iterations:
                res = self.evaluate(it, "selfplay", cfg.eval.searches)
                if cfg.gate.enabled:
                    self._apply_gate(it, res)
            self._save(it)
            self.write_summary()
        self._save(cfg.iterations, final=True)
        self.write_summary()
        return self.history

    # --- reporting ------------------------------------------------------------------------
    def curve(self) -> List[Dict[str, Any]]:
        """One row per (iteration, stage) with `<label>.<metric>` columns."""
        rows: Dict[tuple, Dict[str, Any]] = {}
        for r in self.history:
            if r.get("kind") != "eval" or r["stage"] == "baseline":
                continue
            key = (r["iteration"], r["stage"])
            row = rows.setdefault(key, {"iteration": r["iteration"], "stage": r["stage"]})
            row[f"{r['label']}.success"] = r["success"]
            row[f"{r['label']}.reward"] = r["reward"]
            row[f"{r['label']}.moves"] = r["moves"]
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
        sp = [r for r in self.history if r.get("kind") == "selfplay"]
        if sp:
            lines += ["## Self-play", "",
                      format_table(sp, ["iteration", "episodes", "success", "moves", "samples", "seconds",
                                        "rows_per_second"]), ""]
        tr = [r for r in self.history if r.get("kind") == "train"]
        if tr:
            cols = ["iteration", "stage", "samples", "policy_ce", "value_ce", "heldout_policy_top1",
                    "heldout_value_brier", "temperature_choice", "temperature_noul"]
            lines += ["## Training", "", format_table(tr, cols), ""]
        checks = self.milestones()
        if checks:
            rows = [{"check": c["check"], "value": c["value"], "result": "PASS" if c["pass"] else "FAIL"}
                    for c in checks]
            lines += ["## Phase 0 milestone checks", "", format_table(rows, ["check", "value", "result"]), ""]
        (self.out / "summary.md").write_text("\n".join(lines))
