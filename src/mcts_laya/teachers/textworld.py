"""Cold-start teacher for TextWorld: the game's own optimal plan (`policy_commands`)."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from ..envs.textworld_env import TextWorldEnv, TWState
from ..registry import TEACHERS
from .base import OracleTeacher


@TEACHERS.register("textworld")
class TextWorldTeacher(OracleTeacher):
    env: TextWorldEnv

    def analyse(self, state: TWState, actions: Sequence[Any]) -> Tuple[float, List[int]]:
        plan = state.obs.policy
        if not plan or state.steps + len(plan) > self.env.max_steps:
            return -1.0, []
        best = [i for i, a in enumerate(actions) if a == plan[0]]
        return 2.0 * self.env.solved_reward(state.steps + len(plan)) - 1.0, best
