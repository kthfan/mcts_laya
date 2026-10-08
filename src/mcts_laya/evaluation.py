"""Play a fixed problem set with a searcher and report success / reward / length."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import progress
from .envs.base import Environment
from .search.tree import Searcher
from .selfplay.actor import SelfPlayConfig, play_episode


def evaluate_searcher(env: Environment, searcher: Searcher, problems: Sequence[Any], seed: int = 0,
                      max_moves: int = 50, desc: str = "eval") -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    cfg = SelfPlayConfig(max_moves=max_moves, add_noise=False, action_selection="search")
    t0 = time.time()
    episodes = []
    bar = progress.bar(len(problems), desc, "problems")
    for p in problems:
        episodes.append(play_episode(env, searcher, p, rng, cfg))
        bar.update(1)
        bar.set_postfix(success=float(np.mean([e.success for e in episodes])))
    bar.close()
    return summarize_episodes(episodes, time.time() - t0, env)


def success_moves(episodes) -> Optional[float]:
    """Mean length of the successful episodes (None if none succeeded)."""
    won = [e.length for e in episodes if e.success]
    return float(np.mean(won)) if won else None


def success_by_category(env: Optional[Environment], episodes) -> Optional[Dict[str, float]]:
    """Success rate per problem category (ALFWorld: task type), for environments that define one."""
    category = getattr(env, "category", None)
    if category is None or not episodes:
        return None
    groups: Dict[str, List[bool]] = {}
    for e in episodes:
        groups.setdefault(category(e.initial_state), []).append(bool(e.success))
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def summarize_episodes(episodes, seconds: float, env: Optional[Environment] = None) -> Dict[str, Any]:
    r = {
        "success": float(np.mean([e.success for e in episodes])),
        # mean outcome mapped to [0, 1]; for single-agent envs this is the mean reward
        "reward": float(np.mean([(e.final_value + 1.0) / 2.0 for e in episodes])),
        # all episodes: a failed one counts with the moves it used (TextWorld: up to the step cap)
        "moves": float(np.mean([e.length for e in episodes])),
        "success_moves": success_moves(episodes),  # successful episodes only
        "seconds": seconds,
        "problems": len(episodes),
    }
    by_category = success_by_category(env, episodes)
    if by_category is not None:
        r["by_category"] = by_category
    return r


def format_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)
