"""Play a fixed problem set with a searcher and report success / reward / length."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np

from .envs.base import Environment
from .search.tree import Searcher
from .selfplay.actor import SelfPlayConfig, play_episode


def evaluate_searcher(env: Environment, searcher: Searcher, problems: Sequence[Any], seed: int = 0,
                      max_moves: int = 50) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    cfg = SelfPlayConfig(max_moves=max_moves, add_noise=False, action_selection="search")
    t0 = time.time()
    episodes = [play_episode(env, searcher, p, rng, cfg) for p in problems]
    return {
        "success": float(np.mean([e.success for e in episodes])),
        # mean outcome mapped to [0, 1]; for single-agent envs this is the mean reward
        "reward": float(np.mean([(e.final_value + 1.0) / 2.0 for e in episodes])),
        "moves": float(np.mean([e.length for e in episodes])),
        "seconds": time.time() - t0,
        "problems": len(problems),
    }


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
