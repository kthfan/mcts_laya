"""Evaluator interface: the "network" queried by the search at leaf positions."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from ..envs.base import Environment
from ..registry import EVALUATORS


@dataclass
class EvalResult:
    priors: np.ndarray  # probability per legal action, sums to 1
    value: float  # in [-1, 1], perspective of the player to move


class Evaluator(ABC):
    """Maps positions to (priors over the given actions, value). Implementations batch."""

    @abstractmethod
    def evaluate(self, env: Environment, states: Sequence[Any], actions: Sequence[List[Any]]) -> List[EvalResult]:
        ...

    def clear_cache(self) -> None:
        """Drop cached evaluations (call after the underlying model changes)."""


@EVALUATORS.register("uniform")
class UniformEvaluator(Evaluator):
    """Uniform priors and a constant value: plain UCT-style search without knowledge."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def evaluate(self, env, states, actions):
        return [EvalResult(np.full(len(a), 1.0 / len(a)), self.value) for a in actions]


@EVALUATORS.register("rollout")
class RolloutEvaluator(Evaluator):
    """Uniform priors and the mean outcome of random playouts (classical MCTS)."""

    def __init__(self, n_rollouts: int = 1, max_depth: int = 100, seed: Optional[int] = 0):
        self.n_rollouts = n_rollouts
        self.max_depth = max_depth
        self.rng = random.Random(seed)

    def _rollout(self, env: Environment, state) -> float:
        player = env.current_player(state)
        for _ in range(self.max_depth):
            if env.is_terminal(state):
                v = env.terminal_value(state)
                return v if env.current_player(state) == player else -v
            state = env.step(state, self.rng.choice(env.legal_actions(state)))
        return 0.0

    def evaluate(self, env, states, actions):
        out = []
        for s, a in zip(states, actions):
            v = sum(self._rollout(env, s) for _ in range(self.n_rollouts)) / self.n_rollouts
            out.append(EvalResult(np.full(len(a), 1.0 / len(a)), v))
        return out
