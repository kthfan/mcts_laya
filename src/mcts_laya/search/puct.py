"""AlphaZero PUCT search with batched leaf evaluation (virtual loss)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from ..registry import SEARCHERS
from .tree import Node, SearchResult, Searcher, backup, expand_nodes, make_root


@dataclass
class PUCTConfig:
    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    batch_size: int = 8  # leaves per evaluator call
    virtual_loss: float = 1.0
    fpu_reduction: float = 0.0  # unvisited children start at parent value minus this


@SEARCHERS.register("puct")
class PUCTSearch(Searcher):
    def __init__(self, env, evaluator, **config):
        super().__init__(env, evaluator)
        self.cfg = PUCTConfig(**config)

    def _select(self, node: Node) -> int:
        n = node.child_visits()
        parent_v = node.q if node.chooser == node.player else -node.q
        q = node.child_q(default=parent_v - self.cfg.fpu_reduction)
        u = self.cfg.c_puct * node.priors * np.sqrt(max(n.sum(), 1.0)) / (1.0 + n)
        return int(np.argmax(q + u))

    def _apply_virtual_loss(self, path: List[Node], sign: float) -> None:
        vl = self.cfg.virtual_loss * sign
        for node in path[1:]:
            node.visit_count += vl
            node.value_sum -= vl

    def search(self, state: Any, rng: np.random.Generator, add_noise: bool = True) -> SearchResult:
        env, cfg = self.env, self.cfg
        root = make_root(env, state)
        if root.terminal:
            raise ValueError("cannot search from a terminal state")
        (v,) = expand_nodes(env, self.evaluator, [root])
        backup([root], v, root.player)
        raw_priors = root.priors.copy()
        if add_noise and cfg.dirichlet_fraction > 0 and len(root.actions) > 1:
            noise = rng.dirichlet([cfg.dirichlet_alpha] * len(root.actions))
            root.priors = (1 - cfg.dirichlet_fraction) * root.priors + cfg.dirichlet_fraction * noise

        sims = 0
        while sims < cfg.num_simulations:
            pending: List[Tuple[Node, List[Node]]] = []
            pending_ids = set()
            for _ in range(min(cfg.batch_size, cfg.num_simulations - sims)):
                path, node = [root], root
                while node.expanded and not node.terminal:
                    node = node.child(env, self._select(node))
                    path.append(node)
                    self._apply_virtual_loss([path[-2], node], +1)
                if node.terminal:
                    self._apply_virtual_loss(path, -1)
                    backup(path, node.terminal_value, node.player)
                    sims += 1
                elif id(node) in pending_ids:  # collision: evaluate what we have first
                    self._apply_virtual_loss(path, -1)
                    break
                else:
                    pending.append((node, path))
                    pending_ids.add(id(node))
                    sims += 1
            values = expand_nodes(env, self.evaluator, [n for n, _ in pending])
            for (node, path), value in zip(pending, values):
                self._apply_virtual_loss(path, -1)
                backup(path, value, node.player)

        visits = root.child_visits()
        policy = visits / visits.sum() if visits.sum() > 0 else raw_priors
        q = root.child_q(default=-np.inf)
        # most visits; ties (common with small budgets and wide batches) go to Q, then prior
        selected = max(range(len(visits)), key=lambda i: (visits[i], q[i], raw_priors[i]))
        return SearchResult(
            actions=root.actions,
            visit_counts=visits,
            policy=policy,
            priors=raw_priors,
            q_values=root.child_q(default=0.0),
            root_value=root.q,
            selected=int(selected),
            num_simulations=sims,
        )
