"""Exact teachers for the Phase 0 environments."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Sequence, Tuple

from ..envs.algebra import AlgebraState, LinearEquationEnv
from ..envs.countdown import CountdownEnv, CountdownState, apply_move, steps_to_solve
from ..registry import TEACHERS
from .base import OracleTeacher


@TEACHERS.register("countdown")
class CountdownTeacher(OracleTeacher):
    """Exhaustive search; the optimal actions are those reaching the target in the fewest moves."""

    env: CountdownEnv

    def analyse(self, state: CountdownState, actions: Sequence[Any]) -> Tuple[float, List[int]]:
        dists = [steps_to_solve(apply_move(state.numbers, a), state.target, self.env.use_all) for a in actions]
        solvable = [d for d in dists if d is not None]
        if not solvable:
            return -1.0, []
        best = min(solvable)
        return 1.0, [i for i, d in enumerate(dists) if d == best]


@TEACHERS.register("algebra")
class AlgebraTeacher(OracleTeacher):
    """Breadth-first search over order-insensitive equation keys, cached per problem."""

    env: LinearEquationEnv

    def __init__(self, env: LinearEquationEnv, explore: float = 0.3, max_moves: int = 50,
                 node_limit: int = 200_000):
        super().__init__(env, explore, max_moves)
        self.node_limit = node_limit
        self._dist: Dict[Any, int] = {}
        self._root_key = None

    def distances_from(self, root: AlgebraState) -> Dict[Any, int]:
        """Shortest moves-to-solved for every key reachable within the step budget."""
        env = self.env
        key = env.canonical_key
        rk = key(root)
        if rk == self._root_key or rk in self._dist:
            return self._dist
        budget = env.max_steps - root.steps
        depth = {rk: 0}
        parents: Dict[Any, List[Any]] = {}
        solved = []
        frontier = deque([root])
        while frontier and len(depth) < self.node_limit:
            s = frontier.popleft()
            ks = key(s)
            if env.is_solved(s):
                solved.append(ks)
                continue
            if env.is_terminal(s) or depth[ks] >= budget:
                continue
            for a in env.legal_actions(s):
                c = env.step(s, a)
                kc = key(c)
                parents.setdefault(kc, []).append(ks)
                if kc not in depth:
                    depth[kc] = depth[ks] + 1
                    frontier.append(c)
        dist = {k: 0 for k in solved}
        q = deque(solved)
        while q:  # reverse BFS over the explored graph
            k = q.popleft()
            for p in parents.get(k, ()):
                if p not in dist:
                    dist[p] = dist[k] + 1
                    q.append(p)
        self._dist, self._root_key = dist, rk
        return dist

    def analyse(self, state: AlgebraState, actions: Sequence[Any]) -> Tuple[float, List[int]]:
        env = self.env
        dist = self.distances_from(state)
        d = dist.get(env.canonical_key(state))
        if d is None or state.steps + d > env.max_steps:
            return -1.0, []
        child_d = [dist.get(env.canonical_key(env.step(state, a))) for a in actions]
        best = [i for i, cd in enumerate(child_d) if cd is not None and cd == d - 1]
        return 2.0 * env.solved_reward(state.steps + d) - 1.0, best
