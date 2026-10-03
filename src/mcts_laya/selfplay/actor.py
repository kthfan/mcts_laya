"""Self-play: run the search from the start state to a terminal state and record targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from ..envs.base import Environment
from ..search.tree import SearchResult, Searcher
from ..training.sample import Sample


@dataclass
class SelfPlayConfig:
    max_moves: int = 50
    add_noise: bool = True
    # "search": play the action the search selects (Gumbel already samples);
    # "sample": sample from the search policy with `temperature` for the first `temperature_moves`.
    action_selection: str = "search"
    temperature: float = 1.0
    temperature_moves: int = 0
    # Value target: "outcome" (z), "root" (search value) or "mix" ((1-lam) z + lam root).
    value_target: str = "outcome"
    value_mix: float = 0.5
    # Policy target = search policy ** (1 / policy_target_temperature), renormalised. Values < 1
    # sharpen the flat visit distributions that small simulation budgets produce.
    policy_target_temperature: float = 1.0
    # Policy-loss weight for positions from unsuccessful episodes (Expert-Iteration-style
    # filtering for single-agent tasks; the value is always trained).
    failed_policy_weight: float = 1.0


@dataclass
class StepRecord:
    state: Any
    player: int
    result: SearchResult
    action_index: int


@dataclass
class Episode:
    initial_state: Any
    steps: List[StepRecord] = field(default_factory=list)
    final_state: Any = None
    success: bool = False
    final_value: float = 0.0  # perspective of the player to move in the final state

    @property
    def length(self) -> int:
        return len(self.steps)


def _choose(result: SearchResult, cfg: SelfPlayConfig, move: int, rng: np.random.Generator) -> int:
    if cfg.action_selection == "search" or move >= cfg.temperature_moves:
        return result.selected if cfg.action_selection == "search" else int(np.argmax(result.policy))
    p = np.asarray(result.policy, dtype=np.float64) ** (1.0 / max(cfg.temperature, 1e-6))
    return int(rng.choice(len(p), p=p / p.sum()))


def play_episode(env: Environment, searcher: Searcher, state: Any, rng: np.random.Generator,
                 cfg: Optional[SelfPlayConfig] = None) -> Episode:
    cfg = cfg or SelfPlayConfig()
    ep = Episode(initial_state=state)
    for move in range(cfg.max_moves):
        if env.is_terminal(state):
            break
        result = searcher.search(state, rng, add_noise=cfg.add_noise)
        i = _choose(result, cfg, move, rng)
        ep.steps.append(StepRecord(state, env.current_player(state), result, i))
        state = env.step(state, result.actions[i])
    ep.final_state = state
    if env.is_terminal(state):
        ep.final_value = env.terminal_value(state)
        ep.success = env.is_success(state)
    else:  # move cap reached: score as a loss for the mover
        ep.final_value = -1.0
    return ep


def episode_to_samples(env: Environment, ep: Episode, cfg: Optional[SelfPlayConfig] = None,
                       meta: Optional[dict] = None) -> List[Sample]:
    cfg = cfg or SelfPlayConfig()
    final_player = env.current_player(ep.final_state)
    out = []
    pw = 1.0 if ep.success else cfg.failed_policy_weight
    for rec in ep.steps:
        pol = np.asarray(rec.result.policy, dtype=np.float64)
        if cfg.policy_target_temperature != 1.0 and pol.sum() > 0:
            pol = pol ** (1.0 / max(cfg.policy_target_temperature, 1e-6))
            pol = pol / pol.sum()
        z = ep.final_value if rec.player == final_player else -ep.final_value
        if cfg.value_target == "root":
            v = rec.result.root_value
        elif cfg.value_target == "mix":
            v = (1 - cfg.value_mix) * z + cfg.value_mix * rec.result.root_value
        else:
            v = z
        out.append(Sample(
            state_text=env.state_text(rec.state),
            action_texts=[env.action_text(rec.state, a) for a in rec.result.actions],
            policy=[float(p) for p in pol],
            value=float(v),
            policy_instruction=env.policy_instruction,
            value_instruction=env.value_instruction,
            value_criteria=dict(env.value_criteria),
            meta=dict(meta or {}, env=env.name, source="selfplay"),
            policy_weight=pw,
        ))
    return out
