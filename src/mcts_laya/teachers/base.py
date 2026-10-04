"""Cold-start teachers: exact solvers that label states with soft policy and value targets.

Laya's base checkpoints are near chance on a new decision family, so (like AlphaGo, unlike
AlphaZero) the loop starts from supervised data. A teacher walks solver-optimal trajectories,
with some random detours so the value head also sees states that are worse or lost.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Tuple

import numpy as np

from ..envs.base import Environment
from ..training.sample import Sample


class OracleTeacher(ABC):
    def __init__(self, env: Environment, explore: float = 0.3, max_moves: int = 50):
        self.env = env
        self.explore = explore
        self.max_moves = max_moves

    @abstractmethod
    def analyse(self, state: Any, actions: Sequence[Any]) -> Tuple[float, List[int]]:
        """(value in [-1, 1] for the player to move, indices of the optimal actions)."""

    def label(self, state: Any) -> Tuple[List[Any], np.ndarray, float]:
        actions = self.env.legal_actions(state)
        value, best = self.analyse(state, actions)
        pi = np.zeros(len(actions))
        if best:
            pi[best] = 1.0 / len(best)
        else:
            pi[:] = 1.0 / len(actions)
        return actions, pi, value

    def generate(self, rng: random.Random, n_problems: int) -> List[Sample]:
        env, out = self.env, []
        for _ in range(n_problems):
            state = env.sample_problem(rng, "train")
            for _ in range(self.max_moves):
                if env.is_terminal(state):
                    break
                actions, pi, value = self.label(state)
                out.append(Sample(
                    state_text=env.state_text(state),
                    action_texts=[env.action_text(state, a) for a in actions],
                    policy=pi.tolist(),
                    value=float(value),
                    policy_instruction=env.policy_instruction,
                    value_instruction=env.value_instruction,
                    value_criteria=dict(env.value_criteria),
                    meta={"env": env.name, "source": "teacher"},
                ))
                if rng.random() < self.explore:
                    i = rng.randrange(len(actions))
                else:
                    i = int(rng.choice(np.flatnonzero(pi == pi.max())))
                state = env.step(state, actions[i])
        return out

    def solve(self, state: Any) -> Tuple[bool, int]:
        """Follow the teacher greedily; (solved?, moves). Used as an upper-bound reference."""
        env = self.env
        for n in range(self.max_moves):
            if env.is_terminal(state):
                return env.is_success(state), n
            actions, pi, _ = self.label(state)
            state = env.step(state, actions[int(np.argmax(pi))])
        return False, self.max_moves
