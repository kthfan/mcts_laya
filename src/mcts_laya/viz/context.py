"""Everything the report builder and the live server share: env, model and the methods to compare."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import ExperimentConfig, load_config
from ..evaluators.base import Evaluator
from ..evaluators.laya_evaluator import LayaEvaluator, load_agent
from ..registry import ENVIRONMENTS, EVALUATORS, SEARCHERS, TEACHERS
from ..search.tree import Searcher
from ..selfplay.actor import SelfPlayConfig, play_episode
from .trace import curves_from_metrics, episode_to_json


@dataclass
class Method:
    label: str
    searcher: str
    evaluator: str = "model"  # "model" (Laya) or a registered evaluator name
    params: Dict[str, Any] = field(default_factory=dict)
    evaluator_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def uses_model(self) -> bool:
        return self.evaluator == "model" and self.searcher != "teacher"


class VizContext:
    def __init__(self, cfg: ExperimentConfig, run_dir: Optional[str] = None, checkpoint: Optional[str] = None,
                 device: Optional[str] = None, with_teacher: bool = True):
        self.cfg = cfg
        self.run_dir = Path(run_dir) if run_dir else None
        self.env = ENVIRONMENTS.build(cfg.env.name, **cfg.env.params)
        if checkpoint is None and self.run_dir and (self.run_dir / "checkpoints" / "final").exists():
            checkpoint = str(self.run_dir / "checkpoints" / "final")
        self.checkpoint = checkpoint or cfg.model.checkpoint
        m = cfg.model
        self.agent = load_agent(self.checkpoint, device or m.device, None if checkpoint else m.subfolder,
                                m.revision, m.max_len, m.head_max_len)
        self.evaluator = LayaEvaluator(self.agent, max_rows=cfg.evaluator.max_rows,
                                       prior_temperature=cfg.evaluator.prior_temperature)
        self.methods: List[Method] = [Method(s.label, s.name, s.evaluator, dict(s.params), dict(s.evaluator_params))
                                      for s in list(cfg.eval.searches) + list(cfg.eval.baselines)]
        has_teacher = cfg.env.name in TEACHERS.names()
        if with_teacher and has_teacher:
            self.methods.append(Method("teacher", "teacher", "none"))

    @classmethod
    def from_run(cls, run_dir: str, **kw) -> "VizContext":
        return cls(load_config(str(Path(run_dir) / "config.yaml")), run_dir=run_dir, **kw)

    @classmethod
    def from_config(cls, config: str, overrides: Optional[List[str]] = None, **kw) -> "VizContext":
        return cls(load_config(config, overrides), **kw)

    # --- methods ----------------------------------------------------------------------------
    def method(self, label: str) -> Method:
        for m in self.methods:
            if m.label == label:
                return m
        raise KeyError(f"unknown method {label!r}; known: {[m.label for m in self.methods]}")

    def _evaluator(self, m: Method) -> Optional[Evaluator]:
        if m.searcher == "teacher":
            return None
        if m.evaluator == "model":
            return self.evaluator
        return EVALUATORS.build(m.evaluator, **m.evaluator_params)

    def searcher(self, label: str, **overrides) -> Searcher:
        m = self.method(label)
        return SEARCHERS.build(m.searcher, self.env, self._evaluator(m), **{**m.params, **overrides})

    # --- data -------------------------------------------------------------------------------
    def problems(self, n: int, split: str = "eval", seed: Optional[int] = None) -> List[Any]:
        rng = random.Random(self.cfg.eval.seed if seed is None else seed)
        return self.env.sample_problems(rng, n, split)

    def play(self, problem: Any, label: str, with_tree: bool = True, seed: int = 0, **tree_kw) -> Dict[str, Any]:
        sp = SelfPlayConfig(max_moves=self.cfg.eval.max_moves, add_noise=False, action_selection="search")
        ep = play_episode(self.env, self.searcher(label), problem, np.random.default_rng(seed), sp)
        return episode_to_json(self.env, ep, with_tree=with_tree, **tree_kw)

    def curves(self) -> Optional[Dict[str, Any]]:
        if not self.run_dir or not (self.run_dir / "metrics.jsonl").exists():
            return None
        records = [json.loads(l) for l in open(self.run_dir / "metrics.jsonl") if l.strip()]
        return curves_from_metrics(records)

    def meta(self) -> Dict[str, Any]:
        return {
            "name": self.cfg.name, "env": self.cfg.env.name, "env_params": self.cfg.env.params,
            "checkpoint": self.checkpoint,
            "methods": [{"label": m.label, "searcher": m.searcher, "evaluator": m.evaluator,
                         "simulations": m.params.get("num_simulations", 0), "uses_model": m.uses_model}
                        for m in self.methods],
        }
