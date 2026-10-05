"""Which self-play positions teach the policy, and with what targets.

Four strategies, selected in the `selfplay` config section (see docs/gpu_experiments.md):

* `policy_filter: success`   - policy loss on positions of successful episodes (the original setting)
* `policy_filter: efficient` - (A) only successful episodes that are efficient relative to the agent's
  own record: at most `best_known_ratio` x the fewest moves it has ever needed on that problem, and
  in the top `top_fraction` of this iteration's successful episodes by reward. No teacher involved.
* `policy_filter: none`      - (B) self-play trains the value head only; the policy keeps what it
  learned from the warm start.
* (C) is a search-budget change only: `search.params.num_simulations`.
* `relabel: teacher`         - (D) DAgger: the teacher labels every state self-play visited
  (policy, and optionally value). Imitation learning, kept as an upper-bound reference.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Sequence

import numpy as np

from ..envs.base import Environment
from ..training.sample import Sample
from .actor import Episode

POLICY_FILTERS = ("all", "success", "efficient", "none")


def episode_reward(ep: Episode) -> float:
    return (ep.final_value + 1.0) / 2.0


def update_best_moves(env: Environment, episodes: Sequence[Episode], best: Dict[Hashable, int]) -> None:
    """Record the fewest moves each problem has been solved in (by the agent itself)."""
    for ep in episodes:
        if ep.success:
            k = env.state_key(ep.initial_state)
            best[k] = min(best.get(k, ep.length), ep.length)


def policy_weights(env: Environment, episodes: Sequence[Episode], mode: str, best_moves: Dict[Hashable, int],
                   failed_weight: float = 0.0, best_known_ratio: Optional[float] = 1.3,
                   top_fraction: float = 0.5) -> List[float]:
    """Policy-loss weight per episode. `best_moves` must already include this batch."""
    if mode not in POLICY_FILTERS:
        raise ValueError(f"policy_filter must be one of {POLICY_FILTERS}, got {mode!r}")
    if mode == "none":
        return [0.0] * len(episodes)
    if mode == "all":
        return [1.0] * len(episodes)
    if mode == "success":
        return [1.0 if ep.success else failed_weight for ep in episodes]
    # efficient
    rewards = [episode_reward(ep) for ep in episodes if ep.success]
    cut = float(np.quantile(rewards, 1.0 - top_fraction)) if rewards and top_fraction < 1.0 else -np.inf
    out = []
    for ep in episodes:
        if not ep.success:
            out.append(failed_weight)
            continue
        ok = episode_reward(ep) >= cut - 1e-12
        if best_known_ratio is not None:
            ok = ok and ep.length <= best_known_ratio * best_moves[env.state_key(ep.initial_state)] + 1e-9
        out.append(1.0 if ok else 0.0)
    return out


def relabel_with_teacher(env: Environment, teacher: Any, ep: Episode, samples: List[Sample],
                         value: str = "teacher") -> List[Sample]:
    """DAgger: replace the policy (and, if `value == "teacher"`, the value) target of every position."""
    out = []
    for rec, s in zip(ep.steps, samples):
        actions, pi, v = teacher.label(rec.state)
        if [env.action_text(rec.state, a) for a in actions] != s.action_texts:
            raise RuntimeError("teacher and search disagree on the legal actions of a state")
        out.append(Sample(
            state_text=s.state_text, action_texts=s.action_texts, policy=[float(p) for p in pi],
            value=float(v) if value == "teacher" else s.value,
            policy_instruction=s.policy_instruction, value_instruction=s.value_instruction,
            value_criteria=s.value_criteria, meta=dict(s.meta, source="dagger"), policy_weight=1.0,
        ))
    return out
