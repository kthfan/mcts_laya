"""TextWorld curriculum levels and the on-disk game pools they are played from.

Compiling a TextWorld game takes a few seconds (Inform 7), so games are generated once into a
pool directory and sampled from there:

    <root>/<level>/manifest.json     {"level": ..., "options": {...}, "games": {"train": [...], "eval": [...]}}
    <root>/<level>/train/<seed>.z8   (+ .json / .ni written by tw-make)
    <root>/<level>/eval/<seed>.z8

Train and eval games come from disjoint seed ranges, so evaluation is always on unseen worlds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

EVAL_SEED_OFFSET = 1_000_000


@dataclass(frozen=True)
class LevelSpec:
    """Options of `tw-make custom`, plus the episode step limit used for this level."""

    world_size: int
    nb_objects: int
    quest_length: int
    only_last_action: bool = False  # objective names only the final step: explore and remember
    max_steps: int = 20

    def tw_make_args(self) -> List[str]:
        args = ["custom", "--world-size", str(self.world_size), "--nb-objects", str(self.nb_objects),
                "--quest-length", str(self.quest_length)]
        if self.only_last_action:
            args.append("--only-last-action")
        return args


LEVELS: Dict[str, LevelSpec] = {
    "L1": LevelSpec(world_size=3, nb_objects=6, quest_length=3, max_steps=10),
    "L2": LevelSpec(world_size=5, nb_objects=10, quest_length=5, max_steps=15),
    "L2-goal": LevelSpec(world_size=5, nb_objects=10, quest_length=5, only_last_action=True, max_steps=20),
    "L3-goal": LevelSpec(world_size=8, nb_objects=15, quest_length=8, only_last_action=True, max_steps=30),
}


def _tw_make() -> str:
    exe = Path(sys.executable).parent / "tw-make"
    return str(exe) if exe.exists() else "tw-make"


def _make_one(spec: LevelSpec, seed: int, out: Path) -> Optional[str]:
    cmd = [_tw_make(), *spec.tw_make_args(), "--seed", str(seed), "--output", str(out), "-f", "--silent"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out.exists():
        return None
    return out.name


def generate_pool(root: str, level: str, n_train: int, n_eval: int, workers: int = 4,
                  seed: int = 0, spec: Optional[LevelSpec] = None) -> Path:
    """Generate (or top up) the pool for `level`; returns the level directory."""
    spec = spec or LEVELS[level]
    level_dir = Path(root) / level
    games: Dict[str, List[str]] = {}
    jobs = []
    for split, n, offset in (("train", n_train, 0), ("eval", n_eval, EVAL_SEED_OFFSET)):
        (level_dir / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            s = seed + offset + i
            path = level_dir / split / f"{s}.z8"
            jobs.append((split, s, path))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda j: (j[0], j[2] if j[2].exists() else (_make_one(spec, j[1], j[2]) and j[2])),
                                jobs))
    for split, path in results:
        if path:
            games.setdefault(split, []).append(f"{split}/{Path(path).name}")
    manifest = {"level": level, "options": asdict(spec), "games": {k: sorted(v) for k, v in games.items()}}
    (level_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return level_dir


def load_manifest(level_dir: str) -> dict:
    path = Path(level_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; generate the pool first, e.g. `mcts-laya tw-games --level L1 --out data/textworld`")
    return json.loads(path.read_text())
