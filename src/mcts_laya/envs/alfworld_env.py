"""ALFWorld (text only): household tasks in TextWorld's PDDL engine.

A game is one ALFRED scene turned into a PDDL problem ("put a clean ladle in countertop"); the agent
moves between receptacles, opens them, takes / puts objects and heats, cools, cleans or switches
things on. Objects are hidden until the agent looks at their receptacle, so the state text carries
a memory built only from what the agent has observed (where it is, what it holds, what it has seen
in which receptacle, which receptacles it has not checked yet).

States are functional like `TextWorldEnv`: (game, action history) plus the observation after that
history. Unlike the z-machine games, the PDDL engine can jump to any state: a snapshot is the value
of every SAS variable of the planning task (~0.5-2 KB), and restoring one reloads the task with that
initial state (~20 ms, instead of ~0.35 s for a reset plus replaying the history). Each
observation keeps its snapshot, so the search can step from any node without replaying.

Patches to TextWorld make this practical (pure speed-ups, the text is unchanged; checked by
`tests/test_alfworld.py`): the PDDL text generator re-parses the same grammar templates with TatSu
and deep-copies its context for every symbol; `_speed_up_pddl_textgen` caches the parses and copies
the context's dicts only (a kitchen step: ~125 ms -> ~15 ms). And:
* snapshots / restores go through fast-downward's C state (`GameRunner.snapshot` / `restore`).

The teacher (`teachers/alfworld.py`) uses the PDDL planner from the current state. It knows where
hidden objects are; like the TextWorld teacher, it only provides the warm start.

Data: `mcts-laya alfworld-data` downloads the games (TextWorld-PDDL release of ALFWorld) and
writes `manifest.json` with the splits: train (3553 games minus `val`), val (held out from train
for the gate), valid_seen (140) and valid_unseen (134). The "eval" split is `eval_set`.
"""

from __future__ import annotations

import copy
import functools
import json
import random
import re
import zlib
from array import array
from collections import OrderedDict, defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..registry import ENVIRONMENTS
from .base import SingleAgentEnvironment

TASK_TYPES = ("pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place")
SPLITS = ("train", "val", "valid_seen", "valid_unseen")
MANIFEST = "manifest.json"
TW_PDDL_URL = "https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip"


def task_type(game: str) -> str:
    """'train/pick_two_obj_and_place-Book-None-Desk-301/trial_.../game.tw-pddl' -> 'pick_two_obj_and_place'."""
    parts = Path(game).parts
    return parts[-3].split("-")[0] if len(parts) >= 3 else ""


# --- data -------------------------------------------------------------------------------------
def build_manifest(data_dir: str, n_val: int = 60, seed: int = 0) -> Dict:
    """Index the games under `data_dir` (as ALFWorld does: solvable, no movable / sliced variants)."""
    root = Path(data_dir)
    found: Dict[str, List[str]] = {s: [] for s in ("train", "valid_seen", "valid_unseen")}
    for path in sorted(root.glob("**/game.tw-pddl")):
        rel = path.relative_to(root).as_posix()
        split = next((s for s in found if f"/{s}/" in f"/{rel}"), None)
        if split is None or "movable" in rel or "Sliced" in rel:
            continue
        if not json.loads(path.read_text()).get("solvable", False):
            continue
        found[split].append(rel)
    if not found["train"]:
        raise FileNotFoundError(f"no ALFWorld games under {root} (run: mcts-laya alfworld-data --out {data_dir})")
    train = list(found["train"])
    val = sorted(random.Random(seed).sample(train, min(n_val, len(train) // 2)))
    held = set(val)
    manifest = {"version": 1, "seed": seed,
                "splits": {"train": [g for g in train if g not in held], "val": val,
                           "valid_seen": found["valid_seen"], "valid_unseen": found["valid_unseen"]}}
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1))
    return manifest


def download_games(data_dir: str, url: str = TW_PDDL_URL, force: bool = False) -> Path:
    """Download and unpack ALFWorld's TextWorld-PDDL games (36 MB zip, ~350 MB unpacked)."""
    import io
    import urllib.request
    import zipfile

    root = Path(data_dir)
    if not force and any(root.glob("*/train/*/*/game.tw-pddl")):
        return root
    root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        data = r.read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(root)
    return root


# --- TextWorld speed-up -----------------------------------------------------------------------
_PATCHED = False


def _speed_up_pddl_textgen() -> None:
    """Cache TextWorld's grammar / query parses (each step re-parsed the same templates with TatSu)."""
    global _PATCHED
    if _PATCHED:
        return
    import textworld.envs.pddl.textgen as tg
    from textworld.logic import Rule

    parse = tg._parse_and_convert

    @functools.lru_cache(maxsize=8192)
    def cached(start, rule_name, trace=False):
        return parse(start, rule_name=rule_name, trace=trace)

    # the derivation mutates the parsed symbols (it sets their context), so hand out copies
    tg._parse_and_convert = lambda start, rule_name, trace=False: copy.deepcopy(cached(start, rule_name, trace))
    query = Rule.parse_conjunctive_query.__func__
    Rule.parse_conjunctive_query = classmethod(functools.lru_cache(maxsize=8192)(lambda cls, q: query(cls, q)))

    # Derivations copy their context for every symbol; the original deep-copies the variable and
    # mapping dicts (entity infos, PDDL variables), but they are only ever extended with `update`,
    # never changed in place, so copying the dicts themselves gives the same text ~10x faster.
    def copy_context(context):
        return {"state": context["state"], "facts": context["facts"], "variables": dict(context["variables"]),
                "mapping": dict(context["mapping"]), "entity_infos": context["entity_infos"]}

    tg.copy_context = copy_context
    _PATCHED = True


# --- one live game ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Observation:
    feedback: str
    admissible: Tuple[str, ...]
    won: bool
    snap: bytes = field(repr=False)  # engine state after this observation (see GameRunner)


class PlanTimeout(RuntimeError):
    """The planner did not finish in `plan_timeout_s`."""


class GameRunner:
    """One loaded game; can jump to any snapshot.

    A runner owns a private copy of fast-downward's library (TextWorld loads one per environment).
    Unloading it (`dlclose`) crashes the next game load, and keeping one per game would leak ~32 MB
    of address space per game, so runners are reused: `load` puts another game into this one.
    """

    def __init__(self, path: str):
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        _speed_up_pddl_textgen()
        infos = textworld.EnvInfos(won=True, admissible_commands=True)
        self.env = textworld.start(path, request_infos=infos, wrappers=[AlfredDemangler(), AlfredInfos])
        inner = self.env
        while hasattr(inner, "_wrapped_env"):
            inner = inner._wrapped_env
        self.inner = inner  # textworld.envs.pddl.PddlEnv
        self._lib = inner.downward_lib
        self.restores = 0  # diagnostics
        self._setup(path)

    def load(self, path: str) -> None:
        """Switch this runner to another game (reusing its engine)."""
        self.env.load(path)
        self._setup(path)

    def _setup(self, path: str) -> None:
        self.path = path
        self.gs = self.env.reset()
        self.history: Optional[Tuple[str, ...]] = ()
        ps = self.inner._pddl_state
        self._vars, self._sas_head, self._sas_tail = _split_sas(ps.sas)
        self._index = [{v: k for k, v in enumerate(vals)} for vals in self._vars]
        # snapshots start with a fingerprint of the variable layout: a state made in another process
        # (the actors) must come from the same translation of the same game
        self._layout = zlib.crc32(self._sas_head.encode()).to_bytes(4, "little")
        dynamic = {self._fact(v) for vals in self._vars for v in vals if v.startswith(("Atom ", "NegatedAtom "))}
        self._static = [p for p in ps.facts if p not in dynamic]  # facts no action changes

    def _fact(self, atom: str):
        from textworld.logic import Proposition, Variable

        kind, rest = atom.split(" ", 1)
        name, args = rest.split("(", 1)
        n2t = self.inner._pddl_state.name2type
        variables = [Variable(a.upper() if a.lower() in ("i", "p") else a, n2t[a]) for a in args[:-1].split(", ") if a]
        return Proposition(("not_" if kind == "NegatedAtom" else "") + name, variables)

    def _atoms(self) -> List[str]:
        from fast_downward import Atom

        n = self._lib.get_state_size()
        atoms = (Atom * n)()
        self._lib.get_state(atoms)
        return [a.name for a in atoms]

    def snapshot(self) -> bytes:
        return self._layout + array("H", [self._index[i][a] for i, a in enumerate(self._atoms())]).tobytes()

    def restore(self, snap: bytes, history: Tuple[str, ...]) -> None:
        if snap[:4] != self._layout:
            raise RuntimeError(f"snapshot of another game or PDDL translation cannot be restored in {self.path}")
        values = array("H")
        values.frombytes(snap[4:])
        self._lib.load_sas(f"{self._sas_head}\n{chr(10).join(map(str, values))}\n{self._sas_tail}".encode())
        ps = self.inner._pddl_state
        facts = list(self._static)
        facts += [self._fact(self._vars[i][v]) for i, v in enumerate(values) if self._vars[i][v].startswith("Atom ")]
        ps._facts, ps._vars_by_name, ps._vars_by_type, ps._var_counts = defaultdict(set), {}, defaultdict(set), Counter()
        ps.add_facts(facts)
        self.inner._gather_infos()
        self.history = history
        self.restores += 1

    def observe(self) -> Observation:
        gs = self.gs
        return Observation(feedback=gs["feedback"] or "", admissible=tuple(gs["admissible_commands"] or ()),
                           won=bool(gs["won"]), snap=self.snapshot())

    def reset(self) -> Observation:
        self.gs = self.env.reset()
        self.history = ()
        return self.observe()

    def step(self, command: str) -> Observation:
        self.gs, _, _ = self.env.step(command)
        self.history = self.history + (command,)
        return self.observe()

    def walkthrough(self) -> Tuple[str, ...]:
        """The solution stored in the game file (made by the same planner when the game was
        generated), in the game's displayed names; () if the file has none."""
        raw = self.inner.walkthrough or ()
        names = sorted(((k, info.name) for k, info in self.inner._entity_infos.items() if info.name),
                       key=lambda kv: -len(kv[0]))  # longest ids first: no id is replaced inside another
        out = []
        for cmd in raw:
            for key, name in names:
                cmd = cmd.replace(key, name)
            out.append(cmd)
        return tuple(out)

    def plan(self, timeout: Optional[float] = None) -> Tuple[str, ...]:
        """The planner's optimal remaining commands from the current state (knows hidden objects).

        With a `timeout`, the search runs in a thread (the C call releases the GIL); if it does not
        finish in time, `PlanTimeout` is raised and this runner must not be used again (its engine
        is still busy).
        """
        if timeout is None:
            return tuple(self.inner._pddl_state.replan(self.inner._entity_infos))
        import threading

        out: List[Tuple[str, ...]] = []
        thread = threading.Thread(target=lambda: out.append(self.plan()), daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive() or not out:
            raise PlanTimeout(f"no plan within {timeout:.0f} s for {self.path} after {len(self.history or ())} steps")
        return out[0]



def _split_sas(sas: str) -> Tuple[List[List[str]], str, str]:
    """(values of each SAS variable, text before the initial state, text after it)."""
    lines = sas.split("\n")
    variables, i = [], 0
    while lines[i] != "begin_state":
        if lines[i] == "begin_variable":
            n = int(lines[i + 3])
            variables.append(lines[i + 4: i + 4 + n])
            i += 4 + n
        else:
            i += 1
    start = i + 1
    return variables, "\n".join(lines[:start]), "\n".join(lines[start + len(variables):])


# --- what the agent remembers -----------------------------------------------------------------
@dataclass(frozen=True)
class Memory:
    location: str = ""                                          # "" = the middle of the room
    holding: Tuple[str, ...] = ()
    contents: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()      # receptacle -> objects seen, last update last
    closed: Tuple[str, ...] = ()                                # receptacles last seen closed
    status: Tuple[Tuple[str, str], ...] = ()                    # object -> "hot" / "cold" / "clean" / "on"


_ARRIVE = re.compile(r"You arrive at (?:loc \d+|(.+?))\.")
_SEE = re.compile(r"(?:On the (.+?)|The (.+?) is open\. In it), you see (.+?)\.")
_CLOSED = re.compile(r"The (.+?) is closed\.")
_OPEN = re.compile(r"You open the (.+?)\.")
_CLOSE = re.compile(r"You close the (.+?)\.")
_TAKE = re.compile(r"You pick up the (.+?) from the (.+?)\.")
_PUT = re.compile(r"You (?:move|put) the (.+?) (?:to|in|on|in/on) the (.+?)\.")
_CHANGE = re.compile(r"You (heat|cool|clean) the (.+?) using the (.+?)\.")
_TURN_ON = re.compile(r"You turn on the (.+?)\.")
_STATUS = {"heat": "hot", "cool": "cold", "clean": "clean"}


def parse_items(text: str) -> Tuple[str, ...]:
    """'a apple 2, a cup 2, and a tomato 1' -> ('apple 2', 'cup 2', 'tomato 1'); 'nothing' -> ()."""
    text = text.strip()
    if not text or text == "nothing":
        return ()
    items = []
    for part in re.split(r",\s*", text):
        part = re.sub(r"^(and\s+)?(an?|the)\s+", "", part.strip())
        if part:
            items.append(part)
    return tuple(items)


def update_memory(mem: Memory, action: str, feedback: str) -> Memory:
    fb = " ".join(feedback.split())
    location, holding = mem.location, list(mem.holding)
    contents = OrderedDict(mem.contents)
    closed, status = list(mem.closed), dict(mem.status)

    def saw(recep: str, items: Tuple[str, ...]) -> None:
        contents.pop(recep, None)
        contents[recep] = items
        if recep in closed:
            closed.remove(recep)

    m = _ARRIVE.search(fb)
    if m and action.startswith("go to "):
        location = m.group(1) or action[len("go to "):]
    for m in _SEE.finditer(fb):
        saw(m.group(1) or m.group(2), parse_items(m.group(3)))
    for m in _CLOSED.finditer(fb):
        if m.group(1) not in closed:
            closed.append(m.group(1))
    m = _CLOSE.search(fb)
    if m and m.group(1) not in closed:
        closed.append(m.group(1))
    m = _OPEN.search(fb)
    if m and not _SEE.search(fb):  # opened and empty is reported as "In it, you see nothing."
        saw(m.group(1), ())
    m = _TAKE.search(fb)
    if m:
        obj, recep = m.groups()
        holding.append(obj)
        if recep in contents:
            contents[recep] = tuple(o for o in contents[recep] if o != obj)
    m = _PUT.search(fb)
    if m:
        obj, recep = m.groups()
        if obj in holding:
            holding.remove(obj)
        items = contents.pop(recep, ())
        contents[recep] = items + (obj,) if obj not in items else items
    m = _CHANGE.search(fb)
    if m:
        status[m.group(2)] = _STATUS[m.group(1)]
    m = _TURN_ON.search(fb)
    if m:
        status[m.group(1)] = "on"
    return Memory(location, tuple(holding), tuple(contents.items()), tuple(closed), tuple(status.items()))


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


@dataclass(frozen=True)
class AWState:
    """(game, history) plus what the game showed. An initial state carries no observation until it
    is first used (`ALFWorldEnv.observe`): loading a game takes ~1 s, and the process that samples
    problems (the pipeline's main process) is usually not the one that plays them (the actors)."""

    game: str
    history: Tuple[str, ...]
    obs: Optional[Observation] = field(default=None, compare=False, hash=False)
    memory: Memory = field(default_factory=Memory, compare=False, hash=False)

    @property
    def steps(self) -> int:
        return len(self.history)


_INTRO = re.compile(r"Looking quickly around you, you see (.+?)\.\s*(?:\n|$)", re.S)
_TASK = re.compile(r"Your task is to:\s*(.+?)\s*$", re.S)


def parse_intro(feedback: str) -> Tuple[str, Tuple[str, ...]]:
    """(task, receptacles) from the opening text."""
    m = _TASK.search(feedback)
    task = " ".join(m.group(1).split()) if m else ""
    m = _INTRO.search(feedback)
    receptacles = tuple(sorted(parse_items(" ".join(m.group(1).split())), key=_natural_key)) if m else ()
    return task, receptacles


@ENVIRONMENTS.register("alfworld")
class ALFWorldEnv(SingleAgentEnvironment):
    name = "alfworld"
    policy_instruction = "Which command brings you closest to completing the task?"
    value_instruction = "Will the task be completed within the remaining steps?"
    value_criteria = {
        "false": "no, the task will not be completed in time",
        "true": "yes, the task will be completed in time",
    }

    def __init__(
        self,
        data_dir: str = "data/alfworld",
        eval_set: str = "valid_unseen",
        task_types: Optional[Sequence[str]] = None,
        max_steps: int = 50,
        gamma: float = 0.95,
        success_floor: float = 0.5,
        drop_commands: Sequence[str] = ("look", "inventory", "examine", "help"),
        history_len: int = 4,
        max_open_games: int = 8,
        obs_cache_size: int = 20_000,
        plan_cache_size: int = 5_000,
        plan_timeout_s: Optional[float] = 60.0,
    ):
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} not found (run: mcts-laya alfworld-data --out {data_dir})")
        self.splits: Dict[str, List[str]] = json.loads(manifest_path.read_text())["splits"]
        if eval_set not in self.splits:
            raise ValueError(f"eval_set must be one of {sorted(self.splits)}")
        self.eval_set = eval_set
        unknown = set(task_types or ()) - set(TASK_TYPES)
        if unknown:
            raise ValueError(f"unknown task types {sorted(unknown)}; choose from {TASK_TYPES}")
        self.task_types = tuple(task_types) if task_types else TASK_TYPES
        self.max_steps = int(max_steps)
        self.gamma = gamma
        self.success_floor = success_floor
        self.drop_commands = tuple(drop_commands)
        self.history_len = history_len
        self.max_open_games = max_open_games
        self.obs_cache_size = int(obs_cache_size)
        self.plan_cache_size = int(plan_cache_size)
        self.plan_timeout_s = plan_timeout_s  # teacher only; most plans take < 0.1 s, a few minutes
        self._runners: "OrderedDict[str, GameRunner]" = OrderedDict()
        self._obs_cache: "OrderedDict[Tuple[str, Tuple[str, ...]], Observation]" = OrderedDict()
        self._plan_cache: "OrderedDict[Tuple[str, Tuple[str, ...]], Tuple[str, ...]]" = OrderedDict()
        self._initial: "OrderedDict[str, Observation]" = OrderedDict()  # game -> initial observation
        self._intro: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
        self.cache_hits = 0  # diagnostics
        self.plans_computed = 0
        self.plan_timeouts = 0

    # --- game access ------------------------------------------------------------------------
    def _runner(self, game: str) -> GameRunner:
        r = self._runners.get(game)
        if r is not None:
            self._runners.move_to_end(game)
            return r
        path = str(self.data_dir / game)
        if len(self._runners) >= self.max_open_games:
            r = self._runners.popitem(last=False)[1]  # least recently used: load this game into it
            r.load(path)
        else:
            r = GameRunner(path)
        self._runners[game] = r
        return r

    def _at(self, state: AWState) -> GameRunner:
        """The game's runner, moved to `state`."""
        r = self._runner(state.game)
        if r.history != state.history:
            r.restore(self.observe(state).snap, state.history)
        return r

    def _cache(self, key, obs: Observation) -> None:
        if self.obs_cache_size > 0:
            self._obs_cache[key] = obs
            if len(self._obs_cache) > self.obs_cache_size:
                self._obs_cache.popitem(last=False)

    def _cached(self, key) -> Optional[Observation]:
        obs = self._obs_cache.get(key)
        if obs is not None:
            self._obs_cache.move_to_end(key)
            self.cache_hits += 1
        return obs

    @property
    def restores(self) -> int:
        return sum(r.restores for r in self._runners.values())

    def initial_state(self, game: str) -> AWState:
        return AWState(game, ())  # observed lazily

    def observe(self, state: AWState) -> Observation:
        """The observation of `state` (loads the game for an initial state seen the first time)."""
        if state.obs is not None:
            return state.obs
        key = (state.game, ())
        obs = self._initial.get(state.game) or self._cached(key)
        if obs is None:
            obs = self._runner(state.game).reset()
            self._cache(key, obs)
        if state.game not in self._initial:
            self._initial[state.game] = obs
            self._intro[state.game] = parse_intro(obs.feedback)
            if len(self._initial) > self.max_open_games * 64:
                old = next(iter(self._initial))
                self._initial.pop(old)
                self._intro.pop(old, None)
        return obs

    def intro(self, state: AWState) -> Tuple[str, Tuple[str, ...]]:
        """(task, receptacles) of the state's game."""
        if state.game not in self._intro:
            self._intro[state.game] = parse_intro(self.observe(AWState(state.game, ())).feedback)
        return self._intro[state.game]

    def plan(self, state: AWState) -> Tuple[str, ...]:
        """Optimal remaining commands (teacher only). The planner takes from milliseconds to minutes
        (mostly at the start of a game), so: the start uses the game file's walkthrough; plans are
        cached; and the rest of a plan is reused for the state its first command leads to (actions
        are deterministic, so the rest of an optimal plan stays optimal there). Measured on 48
        teacher episodes: 461 planner calls -> 171 with the reuse, most of the slow ones at the start."""
        key = (state.game, state.history)
        plan = self._plan_cache.get(key)
        if plan is None and not state.history:
            # at the start, the game file's own solution (planning from scratch is the slow case:
            # up to minutes in some games); checked to start with a legal command
            walk = self._at(state).walkthrough()
            if walk and walk[0] in self.observe(state).admissible:
                plan = walk
                self._remember_plan(key, plan)
        if plan is None:
            runner = self._at(state)
            try:
                plan = runner.plan(self.plan_timeout_s)
            except PlanTimeout:
                self._runners.pop(state.game, None)  # still planning: abandon it (never reuse or unload)
                self.plan_timeouts += 1
                raise
            self.plans_computed += 1
            self._remember_plan(key, plan)
        if plan:
            self._remember_plan((state.game, state.history + (plan[0],)), plan[1:])
        return plan

    def _remember_plan(self, key, plan: Tuple[str, ...]) -> None:
        self._plan_cache[key] = plan
        self._plan_cache.move_to_end(key)
        if len(self._plan_cache) > self.plan_cache_size:
            self._plan_cache.popitem(last=False)

    # --- problems ---------------------------------------------------------------------------
    def _pool(self, split: str) -> List[str]:
        name = self.eval_set if split == "eval" else split
        if name not in self.splits:
            raise ValueError(f"unknown ALFWorld split {split!r}; use one of eval, {', '.join(self.splits)}")
        return [g for g in self.splits[name] if task_type(g) in self.task_types]

    def sample_problem(self, rng: random.Random, split: str = "train") -> AWState:
        return self.initial_state(rng.choice(self._pool(split)))

    def sample_problems(self, rng: random.Random, n: int, split: str = "train") -> List[AWState]:
        """`n` distinct games; asking for at least the whole split returns all of it, in order."""
        pool = self._pool(split)
        games = pool if n >= len(pool) else rng.sample(pool, n)
        return [self.initial_state(g) for g in games]

    def category(self, state: AWState) -> str:
        return task_type(state.game)

    # --- dynamics ---------------------------------------------------------------------------
    def legal_actions(self, state: AWState) -> List[str]:
        admissible = self.observe(state).admissible
        cmds = [c for c in admissible if not c.startswith(self.drop_commands)]
        return cmds or list(admissible)

    def step(self, state: AWState, action: str) -> AWState:
        history = state.history + (action,)
        key = (state.game, history)
        obs = self._cached(key)
        if obs is None:
            obs = self._at(state).step(action)
            self._cache(key, obs)
        return AWState(state.game, history, obs, update_memory(state.memory, action, obs.feedback))

    def is_terminal(self, state: AWState) -> bool:
        return self.observe(state).won or state.steps >= self.max_steps

    def solved_reward(self, steps: int) -> float:
        return float(self.success_floor + (1.0 - self.success_floor) * self.gamma ** steps)

    def terminal_reward(self, state: AWState) -> float:
        return self.solved_reward(state.steps) if self.observe(state).won else 0.0

    def is_success(self, state: AWState) -> bool:
        return self.observe(state).won

    # --- text -------------------------------------------------------------------------------
    def state_text(self, state: AWState) -> str:
        m = state.memory
        status = dict(m.status)

        def name(obj: str) -> str:
            return f"{obj} ({status[obj]})" if obj in status else obj

        task, receptacles = self.intro(state)
        parts = [f"Task: {task}"]
        parts.append(f"You are at: {m.location}." if m.location else "You are in the middle of the room.")
        parts.append("Holding: " + (", ".join(map(name, m.holding)) if m.holding else "nothing") + ".")
        seen = []
        for recep, items in m.contents:
            tag = " (closed)" if recep in m.closed else ""
            seen.append(f"{name(recep)}{tag}: " + (", ".join(map(name, items)) if items else "nothing"))
        unopened = [r for r in m.closed if r not in dict(m.contents)]
        if seen:
            parts.append("Seen: " + "; ".join(seen) + ".")
        if unopened:
            parts.append("Closed, not opened yet: " + ", ".join(unopened) + ".")
        visited = set(dict(m.contents)) | set(m.closed)
        todo = [r for r in receptacles if r not in visited]
        if todo:
            parts.append("Not checked yet: " + ", ".join(todo) + ".")
        recent = state.history[-self.history_len:] if self.history_len else ()
        parts.append("Recent actions: " + ("; ".join(recent) if recent else "none") + ".")
        if state.history:
            parts.append("Last result: " + " ".join(self.observe(state).feedback.split()))
        parts.append(f"Steps left: {self.max_steps - state.steps}.")
        return "\n".join(parts)

    def action_text(self, state: AWState, action: str) -> str:
        return action

    def state_key(self, state: AWState):
        return (state.game, state.history)

    def render_data(self, state: AWState) -> Dict:
        return {"kind": "text", "text": self.state_text(state), "goal": self.intro(state)[0], "steps": state.steps,
                "max_steps": self.max_steps, "won": self.observe(state).won, "task_type": task_type(state.game)}
