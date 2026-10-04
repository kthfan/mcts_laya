"""The cold-start teacher exposed as a `Searcher`, so it can be evaluated and visualised like one."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..registry import SEARCHERS, TEACHERS
from ..search.tree import SearchResult, Searcher


@SEARCHERS.register("teacher")
class TeacherSearch(Searcher):
    """Plays the environment's exact teacher (an upper-bound reference, not a learner)."""

    def __init__(self, env, evaluator=None, **_search_params):
        # search budgets (num_simulations, ...) do not apply to an exact teacher
        super().__init__(env, evaluator)
        self.teacher = TEACHERS.build(env.name, env)

    def search(self, state: Any, rng: np.random.Generator, add_noise: bool = False) -> SearchResult:
        actions, pi, value = self.teacher.label(state)
        return SearchResult(
            actions=actions, visit_counts=np.zeros(len(actions)), policy=pi, priors=pi,
            q_values=np.zeros(len(actions)), root_value=float(value), selected=int(np.argmax(pi)),
            num_simulations=0, extra={"teacher": True},
        )
