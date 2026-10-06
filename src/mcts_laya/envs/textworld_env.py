"""TextWorld text-adventure games as a functional `Environment`.

A state is (game, action history) plus a snapshot of what the game reported after that history.
Successors are produced by replaying the history on a per-game `GameRunner`: TextWorld resets in
~5 ms and steps in ~5 ms, while `env.copy()` costs ~50 ms and ~2.5 MB, and restoring only the
z-machine state would desynchronise TextWorld's Python-side state tracking (which computes the
admissible and oracle commands). The runner keeps its current history and only replays the
missing suffix when the requested history extends it.

Replays dominate the cost of a search (a reset plus one engine step per action, so ~50 ms per
expansion mid-episode), so two things avoid them: each game keeps a few runners
(`runners_per_game`), and a search moving between branches extends whichever runner is already
parked on a prefix of the requested history; and observations are cached by (game, history)
(`obs_cache_size`), which serves the nodes the next move's search expands again. Games are
deterministic, so neither changes any result.
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
    notes: Tuple[Tuple[str, str], ...] = field(default=(), compare=False, hash=False)  # (command, what it showed)

    @property
    def steps(self) -> int:
        return len(self.history)


class GameRunner:
    """One live TextWorld environment for one game, driven by replaying action histories.

    `with_facts=True` also tracks the world facts TextWorld's renderer needs (visualisation only;
    search and training use runners without them).
    """

    def __init__(self, path: str, with_facts: bool = False):
        import textworld

        infos = textworld.EnvInfos(objective=True, description=True, inventory=True, feedback=True,
                                   admissible_commands=True, policy_commands=True, score=True,
                                   max_score=True, won=True, lost=True, facts=with_facts, game=with_facts,
                                   last_action=with_facts)
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

    def game_state(self, history: Tuple[str, ...]):
        """The raw TextWorld GameState after `history` (replays like `observe`)."""
        self.observe(history)
        return self.gs

    def close(self) -> None:
        self.env.close()


# --- text cleaning ----------------------------------------------------------------------------
# Greetings name the game at the end of the sentence ("... entered TextWorld!", "... round of
# TextWorld?") or as "the TextWorld today"; entity names ("the TextWorld chest") must survive.
_BOILERPLATE = re.compile(r"\bTextWorld( today\b[^.!?]*)?[.!?]*$|^(Here is|You do|Got that|Good!|That's it)", re.I)


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
        keep_commands: Sequence[str] = (),
        remember_commands: Sequence[str] = ("examine cookbook",),
        history_len: int = 6,
        show_visited: bool = True,
        max_open_games: int = 16,
        runners_per_game: int = 4,
        obs_cache_size: int = 20_000,
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
        self.keep_commands = tuple(keep_commands)  # exact commands exempt from drop_commands
        self.remember_commands = tuple(remember_commands)  # their output stays in the state as notes
        self.history_len = history_len
        self.show_visited = show_visited
        self.max_open_games = max_open_games
        self.runners_per_game = max(1, int(runners_per_game))
        self.obs_cache_size = int(obs_cache_size)
        self._runners: "OrderedDict[str, List[GameRunner]]" = OrderedDict()  # game -> runners, LRU first
        self._obs_cache: "OrderedDict[Tuple[str, Tuple[str, ...]], Observation]" = OrderedDict()
        self.cache_hits = 0  # diagnostics
        self._viz_runners: "OrderedDict[str, GameRunner]" = OrderedDict()

    # --- game access ------------------------------------------------------------------------
    def _runner(self, game: str, history: Tuple[str, ...] = ()) -> GameRunner:
        """The game's runner that needs the fewest replayed steps to reach `history`."""
        pool = self._runners.get(game)
        if pool is None:
            pool = self._runners[game] = []
            if len(self._runners) > self.max_open_games:
                for old in self._runners.popitem(last=False)[1]:
                    old.close()
        else:
            self._runners.move_to_end(game)
        best = None
        for r in pool:
            h = r.history
            if h is not None and len(h) <= len(history) and history[: len(h)] == h:
                if best is None or len(h) > len(best.history):
                    best = r
        if best is None:
            if len(pool) < self.runners_per_game:
                best = GameRunner(str(self.level_dir / game))
                pool.append(best)
            else:
                best = pool[0]  # least recently used: it will reset
        pool.remove(best)
        pool.append(best)
        return best

    def _observe(self, game: str, history: Tuple[str, ...]) -> Observation:
        key = (game, history)
        obs = self._obs_cache.get(key)
        if obs is not None:
            self._obs_cache.move_to_end(key)
            self.cache_hits += 1
            return obs
        obs = self._runner(game, history).observe(history)
        if self.obs_cache_size > 0:
            self._obs_cache[key] = obs
            if len(self._obs_cache) > self.obs_cache_size:
                self._obs_cache.popitem(last=False)
        return obs

    @property
    def replayed_steps(self) -> int:
        return sum(r.replayed_steps for pool in self._runners.values() for r in pool)

    def _state(self, game: str, history: Tuple[str, ...], visited: Tuple[str, ...] = (),
               notes: Tuple[Tuple[str, str], ...] = ()) -> TWState:
        obs = self._observe(game, history)
        room = room_name(obs.description)
        if room and room not in visited:
            visited = visited + (room,)
        if history and history[-1] in self.remember_commands and history[-1] not in dict(notes):
            notes = notes + ((history[-1], clean_feedback(obs.feedback)),)
        return TWState(game, history, obs, visited, notes)

    def initial_state(self, game: str) -> TWState:
        return self._state(game, ())

    # --- problems ---------------------------------------------------------------------------
    def _pool(self, split: str) -> List[str]:
        pool = self.games.get(split) or []
        if not pool:
            hint = f" (generate them: mcts-laya tw-games --level {self.level} --{split} N)" if split in ("val", "eval") else ""
            raise ValueError(f"level {self.level!r} has no {split!r} games in {self.level_dir}{hint}")
        return pool

    def sample_problem(self, rng: random.Random, split: str = "train") -> TWState:
        return self.initial_state(rng.choice(self._pool(split)))

    def sample_problems(self, rng: random.Random, n: int, split: str = "train") -> List[TWState]:
        pool = self._pool(split)
        games = rng.sample(pool, n) if n <= len(pool) else [rng.choice(pool) for _ in range(n)]
        return [self.initial_state(g) for g in games]

    # --- dynamics ---------------------------------------------------------------------------
    def legal_actions(self, state: TWState) -> List[str]:
        cmds = [c for c in state.obs.admissible if c in self.keep_commands or not c.startswith(self.drop_commands)]
        return cmds or list(state.obs.admissible)

    def step(self, state: TWState, action: str) -> TWState:
        return self._state(state.game, state.history + (action,), state.visited, state.notes)

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
        for _, note in state.notes:
            parts.append(f"Notes: {note}")
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

    # --- visualisation ----------------------------------------------------------------------
    def world_map(self, state: TWState) -> Dict:
        """Rooms, exits and items from TextWorld's own renderer (`textworld.render`).

        This is the full map for the human viewer; the agent only ever sees `state_text`.
        """
        from textworld.render import load_state_from_game_state

        r = self._viz_runners.get(state.game)
        if r is None:
            r = self._viz_runners[state.game] = GameRunner(str(self.level_dir / state.game), with_facts=True)
            if len(self._viz_runners) > 8:
                self._viz_runners.popitem(last=False)[1].close()
        rendered = load_state_from_game_state(r.game_state(state.history))
        here = room_name(state.obs.description).lower()
        visited = {v.lower() for v in state.visited}

        def names(items):
            out = []
            for it in items or []:
                if it.get("name"):
                    inner = names(it.get("contents"))
                    out.append(it["name"] + (f" ({', '.join(inner)})" if inner else ""))
            return out

        rooms = [{"name": rm["name"], "x": rm["position"][0], "y": rm["position"][1],
                  "player": rm["name"].lower() == here, "visited": rm["name"].lower() in visited,
                  "items": names(rm.get("items"))} for rm in rendered["rooms"]]
        edges = sorted({tuple(sorted((c["src"], c["dest"]))) for c in rendered["connections"]})
        return {"rooms": rooms, "edges": [list(e) for e in edges], "inventory": names(rendered.get("inventory"))}

    def render_data(self, state: TWState) -> Dict:
        d = {"kind": "textworld", "goal": clean_objective(state.obs.objective), "text": self.state_text(state),
             "steps": state.steps, "max_steps": self.max_steps, "won": state.obs.won}
        try:
            d["map"] = self.world_map(state)
        except Exception as e:  # the map is a convenience; never fail a trace because of it
            d["map_error"] = f"{type(e).__name__}: {e}"
        return d
