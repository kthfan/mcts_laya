"""Cold-start teacher for ALFWorld: the PDDL planner's optimal plan from the current state.

The planner sees the whole world, hidden objects included, so its first move is often "go to" the
receptacle that holds the object, which the agent could not have known. Imitating it teaches the
model where objects usually are; searching the rest of the room is left to self-play (the same
set-up as the TextWorld goal levels, where self-play added +0.12 / +0.22 over the teacher alone).
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from ..envs.alfworld_env import ALFWorldEnv, AWState, PlanTimeout
from ..registry import TEACHERS
from .base import OracleTeacher


@TEACHERS.register("alfworld")
class ALFWorldTeacher(OracleTeacher):
    """`env.plan_timeout_s` bounds each planner call; an episode whose plan times out ends there
    (its samples so far are kept)."""

    env: ALFWorldEnv

    def episode(self, state, rng, out):
        try:
            return super().episode(state, rng, out)
        except PlanTimeout:
            return out

    def analyse(self, state: AWState, actions: Sequence[Any]) -> Tuple[float, List[int]]:
        plan = self.env.plan(state)
        if not plan or state.steps + len(plan) > self.env.max_steps:
            return -1.0, []
        best = [i for i, a in enumerate(actions) if a == plan[0]]
        return 2.0 * self.env.solved_reward(state.steps + len(plan)) - 1.0, best
