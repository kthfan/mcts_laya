"""Cold-start teachers: exact solvers that label states with soft policy and value targets.

Laya's base checkpoints are near chance on a new decision family, so (like AlphaGo, unlike
AlphaZero) the loop starts from supervised data. A teacher walks solver-optimal trajectories,
with some random detours so the value head also sees states that are worse or lost.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import progress
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

    def generate(self, rng: random.Random, n_problems: int, desc: str = "teacher", workers: int = 0,
                 env_spec: Optional[Tuple[str, Dict[str, Any]]] = None) -> List[Sample]:
        """Teacher trajectories on `n_problems` training problems.

        With `workers` > 1 and `env_spec` = (env name, env params), the episodes run in that many
        processes (for slow teachers such as ALFWorld's planner); each episode then draws its
        problem and detours from its own seed, so the data differs from a serial run.
        """
        if workers > 1 and env_spec is not None:
            return self._generate_parallel(rng, n_problems, desc, workers, env_spec)
        out: List[Sample] = []
        for _ in progress.track(range(n_problems), desc=desc, unit="problems"):
            self.episode(self.env.sample_problem(rng, "train"), rng, out)
        return out

    def episode(self, state: Any, rng: random.Random, out: List[Sample]) -> List[Sample]:
        """Label the states of one teacher trajectory (with random detours) into `out`."""
        env = self.env
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

    def _generate_parallel(self, rng: random.Random, n_problems: int, desc: str, workers: int,
                           env_spec: Tuple[str, Dict[str, Any]]) -> List[Sample]:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        seeds = [rng.randrange(2 ** 62) for _ in range(n_problems)]
        name, params = env_spec
        spec = (name, dict(params), {"explore": self.explore, "max_moves": self.max_moves})
        results: List[Optional[List[Sample]]] = [None] * n_problems
        bar = progress.bar(n_problems, desc, "problems")
        with ProcessPoolExecutor(max_workers=min(workers, n_problems), mp_context=mp.get_context("spawn"),
                                 initializer=_worker_init, initargs=spec) as pool:
            futures = {pool.submit(_worker_episode, seed): i for i, seed in enumerate(seeds)}
            from concurrent.futures import as_completed

            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                bar.update(1)
        bar.close()
        return [s for r in results for s in (r or [])]

    def solve(self, state: Any) -> Tuple[bool, int]:
        """Follow the teacher greedily; (solved?, moves). Used as an upper-bound reference."""
        env = self.env
        for n in range(self.max_moves):
            if env.is_terminal(state):
                return env.is_success(state), n
            actions, pi, _ = self.label(state)
            state = env.step(state, actions[int(np.argmax(pi))])
        return False, self.max_moves


# --- teacher worker processes -----------------------------------------------------------------
_WORKER: Optional[OracleTeacher] = None


def _worker_init(env_name: str, env_params: Dict[str, Any], teacher_kwargs: Dict[str, Any]) -> None:
    global _WORKER
    from ..registry import ENVIRONMENTS, TEACHERS

    _WORKER = TEACHERS.build(env_name, ENVIRONMENTS.build(env_name, **env_params), **teacher_kwargs)


def _worker_episode(seed: int) -> List[Sample]:
    r = random.Random(seed)
    return _WORKER.episode(_WORKER.env.sample_problem(r, "train"), r, [])
