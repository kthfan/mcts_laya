"""Environment interface shared by search, self-play, teachers and the Laya evaluator.

Environments are *functional*: a state is an immutable value and `step` returns a new state, so
the search tree can branch from any node without snapshot/restore. A wrapper around a stateful
simulator (Jericho, a browser, ...) implements `step` with that simulator's own save/restore.

Value convention used everywhere in this package: a value is in [-1, 1] and is taken from the
perspective of the player to move in the state it describes. Single-agent environments report a
reward in [0, 1] and `SingleAgentEnvironment` maps it to that scale.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, Iterator, List


class Environment(ABC):
    """A deterministic, turn-based decision problem with text renderings for Laya."""

    name: str = "env"
    num_players: int = 1

    # Text shown to Laya. Subclasses override these to phrase the questions for their domain.
    policy_instruction: str = "Which action should be taken next?"
    value_instruction: str = "Will the player to move succeed from this state?"
    value_criteria: Dict[str, str] = {
        "false": "no, this state leads to failure",
        "true": "yes, this state leads to success",
    }

    # --- dynamics -------------------------------------------------------------------------
    @abstractmethod
    def sample_problem(self, rng: random.Random, split: str = "train") -> Any:
        """Draw an initial state from `split` ("train" or "eval").

        Procedurally generated environments may ignore the split (fresh problems are as good as
        held-out ones); environments with a fixed pool (TextWorld, ALFWorld) must keep the
        evaluation games disjoint from the training games.
        """

    def sample_problems(self, rng: random.Random, n: int, split: str = "train") -> List[Any]:
        """`n` initial states; pool-based environments override this to avoid repeats."""
        return [self.sample_problem(rng, split) for _ in range(n)]

    @abstractmethod
    def legal_actions(self, state: Any) -> List[Any]:
        """Actions available in a non-terminal state, in a deterministic order."""

    @abstractmethod
    def step(self, state: Any, action: Any) -> Any:
        """Return the successor state; must not mutate `state`."""

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        ...

    @abstractmethod
    def terminal_value(self, state: Any) -> float:
        """Outcome in [-1, 1] for the player to move in the terminal `state`."""

    def current_player(self, state: Any) -> int:
        return 0

    # --- text rendering for Laya ----------------------------------------------------------
    @abstractmethod
    def state_text(self, state: Any) -> str:
        ...

    @abstractmethod
    def action_text(self, state: Any, action: Any) -> str:
        ...

    def state_key(self, state: Any) -> Hashable:
        """Key for evaluator caching and transpositions; states are hashable by default."""
        return state

    def is_success(self, state: Any) -> bool:
        """Whether a terminal state counts as solved (for reporting)."""
        return self.terminal_value(state) > 0

    def render_data(self, state: Any) -> Dict[str, Any]:
        """JSON-serialisable description of `state` for the visualisation front end.

        `kind` selects the front-end panel; environments without a dedicated panel fall back to
        showing the state text.
        """
        return {"kind": "text", "text": self.state_text(state)}

    # --- utilities ------------------------------------------------------------------------
    def iter_texts(self, rng: random.Random, n_problems: int = 200) -> Iterator[str]:
        """Texts produced by random play; used to build tokenizers for local test models."""
        yield self.policy_instruction
        yield self.value_instruction
        yield from self.value_criteria.values()
        for _ in range(n_problems):
            state = self.sample_problem(rng)
            while True:
                yield self.state_text(state)
                if self.is_terminal(state):
                    break
                actions = self.legal_actions(state)
                for a in actions:
                    yield self.action_text(state, a)
                state = self.step(state, rng.choice(actions))


class SingleAgentEnvironment(Environment):
    """Single-agent problem whose terminal states carry a reward in [0, 1]."""

    num_players = 1

    @abstractmethod
    def terminal_reward(self, state: Any) -> float:
        """Reward in [0, 1] of a terminal state."""

    def terminal_value(self, state: Any) -> float:
        return 2.0 * self.terminal_reward(state) - 1.0

    def is_success(self, state: Any) -> bool:
        return self.terminal_reward(state) > 0
