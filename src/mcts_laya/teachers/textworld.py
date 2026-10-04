"""Cold-start teacher for TextWorld: the game's own optimal plan (`policy_commands`)."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from ..envs.textworld_env import TextWorldEnv, TWState
from ..registry import TEACHERS
from .base import OracleTeacher


@TEACHERS.register("textworld")
class TextWorldTeacher(OracleTeacher):
    """Follows TextWorld's oracle plan.

    The oracle knows things the player has to look up (the recipe in cooking games), so its plan
    skips commands such as `examine cookbook`. Imitating it as is would teach the model to act on
    information it never saw; the teacher therefore issues every not-yet-used `remember_commands`
    entry that is available before following the plan.
    """

    env: TextWorldEnv

    def _plan(self, state: TWState, actions: Sequence[Any]) -> Tuple[str, ...]:
        seen = dict(state.notes)
        read_first = tuple(c for c in self.env.remember_commands if c in actions and c not in seen)
        return read_first + state.obs.policy

    def analyse(self, state: TWState, actions: Sequence[Any]) -> Tuple[float, List[int]]:
        plan = self._plan(state, actions) if state.obs.policy else ()
        if not plan or state.steps + len(plan) > self.env.max_steps:
            return -1.0, []
        best = [i for i, a in enumerate(actions) if a == plan[0]]
        return 2.0 * self.env.solved_reward(state.steps + len(plan)) - 1.0, best
