"""ALFWorld environment: snapshots, memory, teacher and a pipeline run, on four small games."""

import gzip
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("alfworld")
pytest.importorskip("fast_downward")

from mcts_laya.envs.alfworld_env import (ALFWorldEnv, Memory, build_manifest, parse_intro, parse_items,  # noqa: E402
                                         task_type, update_memory)

FIXTURES = Path(__file__).parent / "data" / "alfworld"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """The fixture games laid out like the ALFWorld release, with a manifest (1 validation game)."""
    root = tmp_path_factory.mktemp("alfworld")
    for gz in sorted(FIXTURES.glob("*.tw-pddl.gz")):
        split, name = gz.name[: -len(".tw-pddl.gz")].split("--")
        d = root / "json_2.1.1" / split / name / "trial_0"
        d.mkdir(parents=True)
        (d / "game.tw-pddl").write_bytes(gzip.decompress(gz.read_bytes()))
    build_manifest(str(root), n_val=1, seed=0)
    return str(root)


@pytest.fixture(scope="module")
def env(data_dir):
    return ALFWorldEnv(data_dir=data_dir, eval_set="valid_seen")


def test_manifest_and_splits(data_dir, env):
    m = json.loads((Path(data_dir) / "manifest.json").read_text())["splits"]
    assert len(m["train"]) == 2 and len(m["val"]) == 1 and len(m["valid_seen"]) == 1 and m["valid_unseen"] == []
    assert not set(m["train"]) & set(m["val"])
    (p,) = env.sample_problems(random.Random(0), 5, split="eval")
    assert env.category(p) == task_type(p.game) == "pick_heat_then_place_in_recep"
    assert len(env.sample_problems(random.Random(0), 1, split="train")) == 1
    with pytest.raises(ValueError):
        env.sample_problems(random.Random(0), 1, split="nope")


def test_intro_and_memory_parsing():
    task, receps = parse_intro("-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking "
                               "quickly around you, you see a drawer 2, a cabinet 10, a cabinet 2, and a fridge 1.\n\n"
                               "Your task is to: put a cool tomato in microwave.")
    assert task == "put a cool tomato in microwave." and receps == ("cabinet 2", "cabinet 10", "drawer 2", "fridge 1")
    assert parse_items("a apple 2, a cup 2, and a tomato 1") == ("apple 2", "cup 2", "tomato 1")
    assert parse_items("nothing") == ()
    m = Memory()
    m = update_memory(m, "go to fridge 1", "You arrive at fridge 1. The fridge 1 is closed.")
    assert m.location == "fridge 1" and m.closed == ("fridge 1",)
    m = update_memory(m, "open fridge 1", "You open the fridge 1. The fridge 1 is open. In it, you see a apple 2, "
                                          "and a tomato 1.")
    assert dict(m.contents)["fridge 1"] == ("apple 2", "tomato 1") and m.closed == ()
    m = update_memory(m, "take tomato 1 from fridge 1", "You pick up the tomato 1 from the fridge 1.")
    assert m.holding == ("tomato 1",) and dict(m.contents)["fridge 1"] == ("apple 2",)
    m = update_memory(m, "cool tomato 1 with fridge 1", "You cool the tomato 1 using the fridge 1.")
    m = update_memory(m, "close fridge 1", "You close the fridge 1.")
    m = update_memory(m, "go to microwave 1", "You arrive at microwave 1. The microwave 1 is open. In it, you see "
                                              "nothing.")
    m = update_memory(m, "move tomato 1 to microwave 1", "You move the tomato 1 to the microwave 1.")
    assert m.holding == () and dict(m.contents)["microwave 1"] == ("tomato 1",)
    assert dict(m.status) == {"tomato 1": "cold"} and m.closed == ("fridge 1",)


def test_snapshots_reproduce_the_game(env):
    """Stepping from a restored snapshot gives what stepping along the history gave."""
    rng = random.Random(1)
    state = env.sample_problems(random.Random(0), 1, split="eval")[0]
    path = [state]
    for _ in range(12):
        state = env.step(state, rng.choice(env.legal_actions(state)))
        path.append(state)
    fresh = ALFWorldEnv(data_dir=str(env.data_dir), eval_set="valid_seen", obs_cache_size=0)
    for k in rng.sample(range(12), 6):
        a = path[k + 1].history[-1]
        again = fresh.step(path[k], a)
        assert again.obs.feedback == path[k + 1].obs.feedback
        assert again.obs.admissible == path[k + 1].obs.admissible
        assert again.obs.snap == path[k + 1].obs.snap
    assert fresh.restores > 0  # (consecutive picks step on without one)
    text = env.state_text(path[-1])
    assert text.startswith("Task: ") and "Steps left: 38." in text


def test_runner_reused_for_another_game(data_dir):
    """With one engine for all games, switching games (re-loading the engine) changes nothing."""
    small = ALFWorldEnv(data_dir=data_dir, eval_set="valid_seen", max_open_games=1, obs_cache_size=0)
    big = ALFWorldEnv(data_dir=data_dir, eval_set="valid_seen", obs_cache_size=0)
    games = [s.game for s in small.sample_problems(random.Random(0), 2, split="train")]
    rng = random.Random(2)
    states = {g: (small.initial_state(g), big.initial_state(g)) for g in games}
    for _ in range(10):
        g = rng.choice(games)
        a, b = states[g]
        cmd = rng.choice(small.legal_actions(a))
        a, b = small.step(a, cmd), big.step(b, cmd)
        assert a.obs == b.obs and small.state_text(a) == big.state_text(b)
        states[g] = (a, b)
    assert len(small._runners) == 1


def _transcript_hash(patched: bool, folder: Path) -> str:
    code = f"""
import glob, hashlib, random, textworld
if {patched}:
    from mcts_laya.envs.alfworld_env import _speed_up_pddl_textgen; _speed_up_pddl_textgen()
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
h, rng = hashlib.md5(), random.Random(3)
for f in sorted(glob.glob({str(folder)!r} + "/*.tw-pddl")):
    env = textworld.start(f, request_infos=textworld.EnvInfos(won=True, admissible_commands=True),
                          wrappers=[AlfredDemangler(), AlfredInfos])
    st = env.reset(); h.update(st["feedback"].encode())
    for _ in range(15):
        a = rng.choice(st["admissible_commands"]); st, _, _ = env.step(a)
        h.update((a + st["feedback"] + "|".join(st["admissible_commands"])).encode())
print(h.hexdigest())
"""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()


def test_textgen_speed_up_keeps_the_text(tmp_path):
    for gz in FIXTURES.glob("*.tw-pddl.gz"):
        (tmp_path / gz.name[:-3]).write_bytes(gzip.decompress(gz.read_bytes()))
    assert _transcript_hash(True, tmp_path) == _transcript_hash(False, tmp_path)


def test_teacher_solves_and_labels(env):
    from mcts_laya.teachers.alfworld import ALFWorldTeacher

    teacher = ALFWorldTeacher(env, explore=0.0)
    for state in env.sample_problems(random.Random(0), 2, split="train"):
        solved, moves = teacher.solve(state)
        assert solved and moves <= 10
    samples = teacher.generate(random.Random(0), 2)
    assert samples and all(abs(sum(s.policy) - 1) < 1e-6 and 0 < s.value <= 1 for s in samples)


def test_plan_reuse_and_timeout(data_dir):
    from mcts_laya.envs.alfworld_env import PlanTimeout
    from mcts_laya.teachers.alfworld import ALFWorldTeacher

    env = ALFWorldEnv(data_dir=data_dir, eval_set="valid_seen")
    state = env.sample_problems(random.Random(0), 1, split="eval")[0]
    plan = env.plan(state)
    for cmd in plan[:-1]:  # following the plan needs no further planner call
        state = env.step(state, cmd)
        assert env.plan(state)[0] == plan[state.steps]
    assert env.plans_computed == 0  # the start uses the game's walkthrough
    assert env.is_success(env.step(state, plan[-1]))
    start = env.initial_state(state.game)
    detour = env.step(start, next(a for a in env.legal_actions(start) if a != plan[0]))
    assert env.plan(detour) and env.plans_computed == 1  # off the plan: the planner runs

    slow = ALFWorldEnv(data_dir=data_dir, eval_set="valid_seen", plan_timeout_s=0.0)
    first = slow.sample_problems(random.Random(0), 1, split="eval")[0]
    detour = slow.step(first, detour.history[0])
    with pytest.raises(PlanTimeout):
        slow.plan(detour)
    assert slow.plan_timeouts == 1 and first.game not in slow._runners  # the busy engine is dropped
    samples = ALFWorldTeacher(slow, explore=0.0).episode(detour, random.Random(0), [])
    assert samples == []  # the episode just ends


def test_pipeline_runs_on_alfworld(tiny_checkpoint, data_dir, tmp_path):
    from mcts_laya.config import load_config
    from mcts_laya.pipeline import AlphaZeroLoop

    cfg = load_config("configs/alfworld/base.yaml", [
        f"output_dir={tmp_path}", f"model.checkpoint={tiny_checkpoint}", "model.device=cpu", "model.max_len=null",
        "model.head_max_len=null", f"env.params.data_dir={data_dir}", "env.params.eval_set=valid_seen",
        "env.params.max_steps=8", "iterations=1", "teacher.problems=2", "teacher.epochs=1", "teacher.workers=2",
        "selfplay.episodes_per_iteration=2", "eval.problems=1", "eval.gate_problems=1", "eval.max_moves=8",
        "eval.extra_splits=[train]", "eval.baselines=[]", "search.params.num_simulations=4", "parallel.workers=2",
        "train.config.batch_size=2",
    ])
    AlphaZeroLoop(cfg).run()
    records = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    evals = [r for r in records if r["kind"] == "eval"]
    assert all(r["by_category"] for r in evals)
    assert {r["stage"] for r in evals if r["split"] == "train" and r["label"] == "puct16"} == {"warmstart", "final"}
    assert any(r["split"] == "val" for r in evals)  # the gate
    summary = (tmp_path / "summary.md").read_text()
    assert "Success by category" in summary and "Other test splits" in summary
    assert hashlib.md5(summary.encode()).hexdigest()  # written
