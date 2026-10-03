"""Gumbel AlphaZero search (Danihelka et al., "Policy improvement by planning with Gumbel", 2022).

The root samples actions without replacement with the Gumbel-Top-k trick and allocates the
simulation budget by Sequential Halving; interior nodes select deterministically against the
improved policy. The improved policy softmax(logits + sigma(completed Q)) is the training target.
It is designed for small simulation budgets, which matters here because every Laya evaluation is
expensive compared with a small ResNet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from ..registry import SEARCHERS
from .tree import Node, SearchResult, Searcher, backup, expand_nodes, make_root


@dataclass
class GumbelConfig:
    num_simulations: int = 32
    max_considered_actions: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


@SEARCHERS.register("gumbel")
class GumbelSearch(Searcher):
    def __init__(self, env, evaluator, **config):
        super().__init__(env, evaluator)
        self.cfg = GumbelConfig(**config)

    # --- helpers ----------------------------------------------------------------------------
    def _sigma(self, q: np.ndarray, node: Node) -> np.ndarray:
        q01 = (q + 1.0) / 2.0  # values live in [-1, 1]
        max_visit = node.child_visits().max() if node.children else 0.0
        return (self.cfg.c_visit + max_visit) * self.cfg.c_scale * q01

    @staticmethod
    def _completed_q(node: Node) -> np.ndarray:
        n = node.child_visits()
        q = node.child_q(default=0.0)
        visited = n > 0
        if not visited.any():
            v_mix = node.nn_value
        else:
            p = node.priors
            weighted = (p[visited] * q[visited]).sum() / max(p[visited].sum(), 1e-12)
            v_mix = (node.nn_value + n.sum() * weighted) / (1.0 + n.sum())
        return np.where(visited, q, v_mix)

    def _improved_policy(self, node: Node) -> np.ndarray:
        return _softmax(node.logits + self._sigma(self._completed_q(node), node))

    def _interior_select(self, node: Node) -> int:
        pi = self._improved_policy(node)
        n = node.child_visits()
        return int(np.argmax(pi - n / (1.0 + n.sum())))

    def _simulate(self, roots_actions: List[Tuple[Node, int]]) -> None:
        """One simulation per (root, forced first action); leaves evaluated in one batch."""
        env = self.env
        pending = []
        for root, a in roots_actions:
            node = root.child(env, a)
            path = [root, node]
            while node.expanded and not node.terminal:
                node = node.child(env, self._interior_select(node))
                path.append(node)
            if node.terminal:
                backup(path, node.terminal_value, node.player)
            else:
                pending.append((node, path))
        values = expand_nodes(env, self.evaluator, [n for n, _ in pending])
        for (node, path), v in zip(pending, values):
            backup(path, v, node.player)

    # --- search -----------------------------------------------------------------------------
    def search(self, state: Any, rng: np.random.Generator, add_noise: bool = True) -> SearchResult:
        env, cfg = self.env, self.cfg
        root = make_root(env, state)
        if root.terminal:
            raise ValueError("cannot search from a terminal state")
        (v,) = expand_nodes(env, self.evaluator, [root])
        backup([root], v, root.player)
        k = len(root.actions)
        logits = root.logits
        g = rng.gumbel(size=k) if add_noise else np.zeros(k)
        m = min(cfg.max_considered_actions, k, max(cfg.num_simulations, 1))
        remaining = list(np.argsort(-(g + logits))[:m])

        budget, used = cfg.num_simulations, 0
        num_phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        for phase in range(num_phases):
            if used >= budget:
                break
            per_action = max(1, budget // (num_phases * len(remaining)))
            for _ in range(per_action):
                batch = remaining[: budget - used]
                if not batch:
                    break
                self._simulate([(root, int(a)) for a in batch])
                used += len(batch)
            if len(remaining) > 1:
                score = g + logits + self._sigma(self._completed_q(root), root)
                remaining = sorted(remaining, key=lambda a: -score[a])[: math.ceil(len(remaining) / 2)]
        while used < budget:  # leftover budget goes to the survivors
            batch = remaining[: budget - used]
            self._simulate([(root, int(a)) for a in batch])
            used += len(batch)

        completed = self._completed_q(root)
        score = g + logits + self._sigma(completed, root)
        selected = int(max(remaining, key=lambda a: score[a]))
        improved = self._improved_policy(root)
        return SearchResult(
            actions=root.actions,
            visit_counts=root.child_visits(),
            policy=improved,
            priors=root.priors.copy(),
            q_values=completed,
            # Sequential Halving spreads visits evenly, so the plain root mean is dragged down by
            # bad actions; report the value of the improved policy instead.
            root_value=float(improved @ completed),
            selected=selected,
            num_simulations=used,
        )


@SEARCHERS.register("greedy")
class PriorGreedySearch(Searcher):
    """No search: act on the evaluator's prior (the "raw network" baseline)."""

    def __init__(self, env, evaluator, **_):
        super().__init__(env, evaluator)

    def search(self, state: Any, rng: np.random.Generator, add_noise: bool = False) -> SearchResult:
        root = make_root(self.env, state)
        (v,) = expand_nodes(self.env, self.evaluator, [root])
        return SearchResult(
            actions=root.actions,
            visit_counts=np.zeros(len(root.actions)),
            policy=root.priors.copy(),
            priors=root.priors.copy(),
            q_values=np.zeros(len(root.actions)),
            root_value=v,
            selected=int(np.argmax(root.priors)),
            num_simulations=0,
        )
