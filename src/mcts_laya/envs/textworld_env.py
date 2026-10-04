"""TextWorld text-adventure games as a functional `Environment`.

A state is (game, action history) plus a snapshot of what the game reported after that history.
Successors are produced by replaying the history on a per-game `GameRunner`: TextWorld resets in
~5 ms and steps in ~5 ms, while `env.copy()` costs ~50 ms and ~2.5 MB, and restoring only the
z-machine state would desynchronise TextWorld's Python-side state tracking (which computes the
admissible and oracle commands). The runner keeps its current history and only replays the
missing suffix when the requested history extends it.
"""

from __future__ import annotations

import random
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..registry import ENVIRONMENTS
from .base import SingleAgentEnvironment
from .textworld_games import LEVELS, load_manifest


@dataclass(frozen=True)
class Observation:
    objective: str
    description: str
    inventory: str
    feedback: str
    admissible: Tuple[str, ...]
    policy: Tuple[str, ...]  # TextWorld's optimal remaining commands (teacher only)
    score: int
    max_score: int
    won: bool
    lost: bool


@dataclass(frozen=True)
class TWState:
    game: str
    history: Tuple[str, ...]
    obs: Observation = field(compare=False, hash=False)
    visited: Tuple[str, ...] = field(default=(), compare=False, hash=False)  # rooms, first-visit order

    @property
    def steps(self) -> int:
        return len(self.history)


class GameRunner:
    """One live TextWorld environment for one game, driven by replaying action histories."""

    def __init__(self, path: str):
        import textworld

        infos = textworld.EnvInfos(objective=True, description=True, inventory=True, feedback=True,
                                   admissible_commands=True, policy_commands=True, score=True,
                                   max_score=True, won=True, lost=True)
        self.env = textworld.start(path, infos)
        self.history: Optional[Tuple[str, ...]] = None
        self.gs = None
        self.replayed_steps = 0  # diagnostics

    def observe(self, history: Tuple[str, ...]) -> Observation:
        if self.history is not None and history[: len(self.history)] == self.history:
            todo = history[len(self.history):]
        else:
            self.gs = self.env.reset()
            self.history = ()
            todo = history
        for cmd in todo:
            self.gs, _, _ = self.env.step(cmd)
            self.history += (cmd,)
            self.replayed_steps += 1
        gs = self.gs
        return Observation(
            objective=gs["objective"] or "", description=gs["description"] or "",
            inventory=gs["inventory"] or "", feedback=gs["feedback"] or "",
            admissible=tuple(gs["admissible_commands"] or ()), policy=tuple(gs["policy_commands"] or ()),
            score=int(gs["score"] or 0), max_score=int(gs["max_score"] or 0),
            won=bool(gs["won"]), lost=bool(gs["lost"]),
        )

    def close(self) -> None:
        self.env.close()


# --- text cleaning ----------------------------------------------------------------------------
_BOILERPLATE = re.compile(r"TextWorld|^(Here is|You do|Got that|Good!|That's it)", re.I)


def clean_objective(text: str) -> str:
    """Drop TextWorld's randomised greeting sentences, keep the task."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return " ".join(s for s in sentences if s and not _BOILERPLATE.search(s))


def room_name(description: str) -> str:
    m = re.search(r"-=\s*(.*?)\s*=-", description)
    return m.group(1) if m else ""


def clean_room(text: str) -> str:
    text = re.sub(r"-=\s*(.*?)\s*=-", r"\1.", text)
    return " ".join(text.split())


def clean_feedback(text: str) -> str:
    text = text.split("\n>")[0]  # drop the prompt and status line
    return " ".join(text.split())


@ENVIRONMENTS.register("textworld")
class TextWorldEnv(SingleAgentEnvironment):
    name = "textworld"
    policy_instruction = "Which command brings you closest to completing the goal?"
    value_instruction = "Will the goal be completed within the remaining steps?"
    value_criteria = {
        "false": "no, the goal will not be completed in time",
        "true": "yes, the goal will be completed in time",
    }

    def __init__(
        self,
        game_dir: str = "data/textworld",
        level: str = "L1",
        max_steps: Optional[int] = None,
        gamma: float = 0.9,
        success_floor: float = 0.5,
        partial_weight: float = 0.0,
        drop_commands: Sequence[str] = ("look", "inventory", "examine"),
        history_len: int = 6,
        show_visited: bool = True,
        max_open_games: int = 64,
    ):
        self.level = level
        self.level_dir = Path(game_dir) / level
        manifest = load_manifest(str(self.level_dir))
        self.games: Dict[str, List[str]] = manifest["games"]
        spec = LEVELS.get(level)
        self.max_steps = int(max_steps if max_steps is not None else (spec.max_steps if spec else 20))
        self.gamma = gamma
        self.success_floor = success_floor
        self.partial_weight = partial_weight
        self.drop_commands = tuple(drop_commands)
        self.history_len = history_len
        self.show_visited = show_visited
        self.max_open_games = max_open_games
        self._runners: "OrderedDict[str, GameRunner]" = OrderedDict()

    # --- game access ------------------------------------------------------------------------
    def _runner(self, game: str) -> GameRunner:
        r = self._runners.get(game)
        if r is None:
            r = GameRunner(str(self.level_dir / game))
            self._runners[game] = r
            if len(self._runners) > self.max_open_games:
                _, old = self._runners.popitem(last=False)
                old.close()
        else:
            self._runners.move_to_end(game)
        return r

    def _state(self, game: str, history: Tuple[str, ...], visited: Tuple[str, ...] = ()) -> TWState:
        obs = self._runner(game).observe(history)
        room = room_name(obs.description)
        if room and room not in visited:
            visited = visited + (room,)
        return TWState(game, history, obs, visited)

    def initial_state(self, game: str) -> TWState:
        return self._state(game, ())

    # --- problems ---------------------------------------------------------------------------
    def _pool(self, split: str) -> List[str]:
        pool = self.games.get(split) or []
        if not pool:
            raise ValueError(f"level {self.level!r} has no {split!r} games in {self.level_dir}")
        return pool

    def sample_problem(self, rng: random.Random, split: str = "train") -> TWState:
        return self.initial_state(rng.choice(self._pool(split)))

    def sample_problems(self, rng: random.Random, n: int, split: str = "train") -> List[TWState]:
        pool = self._pool(split)
        games = rng.sample(pool, n) if n <= len(pool) else [rng.choice(pool) for _ in range(n)]
        return [self.initial_state(g) for g in games]

    # --- dynamics ---------------------------------------------------------------------------
    def legal_actions(self, state: TWState) -> List[str]:
        cmds = [c for c in state.obs.admissible if not c.startswith(self.drop_commands)]
        return cmds or list(state.obs.admissible)

    def step(self, state: TWState, action: str) -> TWState:
        return self._state(state.game, state.history + (action,), state.visited)

    def is_terminal(self, state: TWState) -> bool:
        o = state.obs
        return o.won or o.lost or state.steps >= self.max_steps

    def solved_reward(self, steps: int) -> float:
        return float(self.success_floor + (1.0 - self.success_floor) * self.gamma ** steps)

    def terminal_reward(self, state: TWState) -> float:
        o = state.obs
        if o.won:
            return self.solved_reward(state.steps)
        if o.lost or not o.max_score:
            return 0.0
        return self.partial_weight * o.score / o.max_score

    def is_success(self, state: TWState) -> bool:
        return state.obs.won

    # --- text -------------------------------------------------------------------------------
    def state_text(self, state: TWState) -> str:
        o = state.obs
        recent = state.history[-self.history_len:] if self.history_len else ()
        parts = [f"Goal: {clean_objective(o.objective)}"]
        parts.append("Recent actions: " + ("; ".join(recent) if recent else "none") + ".")
        if self.show_visited and len(state.visited) > 1:
            parts.append("Rooms visited: " + ", ".join(state.visited) + ".")
        parts.append(" ".join(o.inventory.split()))
        room = clean_room(o.description)
        parts.append(f"Location: {room}")
        fb = clean_feedback(o.feedback)
        if state.history and fb and not room.startswith(fb[: min(len(fb), 30)]):
            parts.append(f"Last result: {fb}")
        parts.append(f"Steps left: {self.max_steps - state.steps}.")
        return "\n".join(parts)

    def action_text(self, state: TWState, action: str) -> str:
        return action

    def state_key(self, state: TWState):
        return (state.game, state.history)
