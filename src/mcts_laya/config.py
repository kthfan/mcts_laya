"""Experiment configuration: nested dataclasses loaded from YAML, with `a.b=c` overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .selfplay.actor import SelfPlayConfig
from .training.trainer import TrainConfig


@dataclass
class ComponentConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    checkpoint: str = "models/laya/multilingual"
    device: Optional[str] = None
    subfolder: Optional[str] = None
    revision: Optional[str] = None
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    # Build a tiny random Laya-compatible checkpoint at `checkpoint` if it does not exist
    # (tests / CPU development without Hub access).
    build_tiny: bool = False
    tiny_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorConfig:
    max_rows: int = 64
    cache_size: int = 200_000
    prior_temperature: float = 1.0
    option_order: str = "none"


@dataclass
class TeacherConfig:
    enabled: bool = True
    problems: int = 300
    explore: float = 0.3
    epochs: int = 2


@dataclass
class SelfPlaySection:
    episodes_per_iteration: int = 64
    config: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    # Which positions teach the policy (see mcts_laya.selfplay.targets):
    # "success" (default) | "efficient" (A) | "none" (B, value only) | "all"
    policy_filter: str = "success"
    best_known_ratio: Optional[float] = 1.3  # efficient: moves <= ratio x own best on that problem
    top_fraction: float = 0.5  # efficient: keep the top share of this iteration's successes by reward
    # (D) DAgger: "teacher" relabels every visited state with the teacher's targets
    relabel: str = "none"
    relabel_value: str = "teacher"  # "teacher" or "outcome" (keep the self-play value target)
    # False: self-play still runs (and is logged) but adds nothing to the replay buffer, so every
    # iteration trains on the teacher data alone - the control for "is it self-play or just more
    # training on the teacher data?"
    add_to_replay: bool = True


@dataclass
class TrainSection:
    replay_capacity: int = 50_000
    samples_per_iteration: Optional[int] = None  # None = whole buffer
    teacher_mix: float = 0.0  # fraction of teacher samples kept in the replay buffer
    config: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class EvalSearch:
    label: str
    name: str  # searcher name
    evaluator: str = "model"  # "model" (the Laya agent) or a registered evaluator name
    params: Dict[str, Any] = field(default_factory=dict)
    evaluator_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSection:
    problems: int = 100
    seed: int = 12345
    max_moves: int = 50
    every: int = 1
    # Gate on a separate validation split: `gate_problems` problems from `gate_split` (seeded by
    # `gate_seed`) decide whether an iteration's weights are kept, and the test problems above only
    # measure the kept model. 0 = legacy: the gate reads the test evaluation itself.
    gate_problems: int = 0
    gate_split: str = "val"
    gate_seed: int = 54321
    searches: List[EvalSearch] = field(default_factory=list)
    baselines: List[EvalSearch] = field(default_factory=list)  # evaluated once, not per iteration


@dataclass
class ParallelSection:
    """Many episodes at once (mcts_laya.selfplay.parallel): actor processes + batched model calls."""

    # actor processes for self-play and evaluation; "auto" = one per spare CPU core (max 32) when
    # the model is on a GPU, 0 (one episode at a time, in-process) on the CPU
    workers: Any = "auto"
    max_wait_ms: float = 2.0  # how long a model call waits for more actors' requests
    cache_size: int = 50_000  # per-actor evaluation cache


@dataclass
class GateSection:
    enabled: bool = False
    metric: str = ""  # "<label>.<metric>", e.g. "mcts.reward"
    tolerance: float = 0.0


@dataclass
class MilestoneSection:
    """Phase 0 acceptance checks, expressed on eval labels."""

    search_label: str = ""
    greedy_label: str = ""
    baseline_label: str = ""
    metric: str = "reward"


@dataclass
class ExperimentConfig:
    name: str
    env: ComponentConfig
    search: ComponentConfig
    output_dir: str = "runs/experiment"
    seed: int = 0
    iterations: int = 5
    save_checkpoints: str = "last"  # "last" | "all" | "none"
    save_half: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    selfplay: SelfPlaySection = field(default_factory=SelfPlaySection)
    train: TrainSection = field(default_factory=TrainSection)
    eval: EvalSection = field(default_factory=EvalSection)
    gate: GateSection = field(default_factory=GateSection)
    parallel: ParallelSection = field(default_factory=ParallelSection)
    milestone: MilestoneSection = field(default_factory=MilestoneSection)


def _build(cls, data: Any):
    if not dataclasses.is_dataclass(cls):
        return data
    if isinstance(data, cls):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} expects a mapping, got {data!r}")
    hints = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(hints)
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    import typing

    types = typing.get_type_hints(cls)
    for k, v in data.items():
        t = types[k]
        origin = typing.get_origin(t)
        if dataclasses.is_dataclass(t):
            kwargs[k] = _build(t, v)
        elif origin in (list, List) and dataclasses.is_dataclass(typing.get_args(t)[0]):
            kwargs[k] = [_build(typing.get_args(t)[0], x) for x in (v or [])]
        else:
            kwargs[k] = v
    return cls(**kwargs)


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"override must look like a.b=c, got {item!r}")
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return data


def load_config(path: str, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return config_from_dict(apply_overrides(data, overrides or []))


def config_from_dict(data: Dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, data)


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)
