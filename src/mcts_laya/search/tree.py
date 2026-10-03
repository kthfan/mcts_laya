"""Search-tree machinery shared by the search algorithms.

Value bookkeeping: `Node.value_sum` is accumulated from the perspective of the player who chose
the action leading to the node (for the root: the player to move at the root), so a parent reads
`child.q` directly as "how good is this action for me".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np

from ..envs.base import Environment
from ..evaluators.base import Evaluator


class Node:
    __slots__ = (
        "state", "player", "chooser", "prior", "children", "actions", "priors", "logits",
        "visit_count", "value_sum", "terminal", "terminal_value", "expanded", "nn_value",
    )

    def __init__(self, state: Any, player: int, chooser: int, prior: float = 1.0):
        self.state = state
        self.player = player  # player to move in this state
        self.chooser = chooser  # player whose action led here
        self.prior = prior
        self.children: List[Optional["Node"]] = []
        self.actions: List[Any] = []
        self.priors: Optional[np.ndarray] = None
        self.logits: Optional[np.ndarray] = None
        self.visit_count = 0
        self.value_sum = 0.0
        self.terminal = False
        self.terminal_value = 0.0  # perspective of `player`
        self.expanded = False
        self.nn_value = 0.0  # evaluator value, perspective of `player`

    @property
    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def child(self, env: Environment, i: int) -> "Node":
        c = self.children[i]
        if c is None:
            s = env.step(self.state, self.actions[i])
            c = Node(s, env.current_player(s), self.player, float(self.priors[i]))
            if env.is_terminal(s):
                c.terminal = True
                c.terminal_value = env.terminal_value(s)
            self.children[i] = c
        return c

    def child_visits(self) -> np.ndarray:
        return np.array([c.visit_count if c is not None else 0 for c in self.children], dtype=np.float64)

    def child_q(self, default: float) -> np.ndarray:
        return np.array([c.q if c is not None and c.visit_count else default for c in self.children])


def make_root(env: Environment, state: Any) -> Node:
    p = env.current_player(state)
    root = Node(state, p, p)
    if env.is_terminal(state):
        root.terminal = True
        root.terminal_value = env.terminal_value(state)
    return root


def expand_nodes(env: Environment, evaluator: Evaluator, nodes: Sequence[Node]) -> List[float]:
    """Evaluate and expand non-terminal leaves in one batch; returns each leaf's value (own perspective)."""
    if not nodes:
        return []
    actions = [env.legal_actions(n.state) for n in nodes]
    results = evaluator.evaluate(env, [n.state for n in nodes], actions)
    values = []
    for n, acts, r in zip(nodes, actions, results):
        n.actions = acts
        n.priors = np.asarray(r.priors, dtype=np.float64)
        n.logits = np.log(np.clip(n.priors, 1e-12, None))
        n.children = [None] * len(acts)
        n.expanded = True
        n.nn_value = float(r.value)
        values.append(float(r.value))
    return values


def backup(path: Sequence[Node], value: float, leaf_player: int) -> None:
    """Propagate `value` (perspective of `leaf_player`) along `path` (root first)."""
    for node in path:
        node.visit_count += 1
        node.value_sum += value if node.chooser == leaf_player else -value


def leaf_value(node: Node) -> float:
    return node.terminal_value if node.terminal else node.nn_value


@dataclass
class SearchResult:
    actions: List[Any]
    visit_counts: np.ndarray
    policy: np.ndarray  # training target over `actions`
    priors: np.ndarray
    q_values: np.ndarray
    root_value: float  # perspective of the player to move at the root
    selected: int  # index into `actions` the search recommends
    num_simulations: int = 0
    extra: dict = field(default_factory=dict)


class Searcher(ABC):
    def __init__(self, env: Environment, evaluator: Evaluator):
        self.env = env
        self.evaluator = evaluator

    @abstractmethod
    def search(self, state: Any, rng: np.random.Generator, add_noise: bool = True) -> SearchResult:
        ...
