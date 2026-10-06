import json

from mcts_laya.config import config_from_dict, load_config
from mcts_laya.pipeline import AlphaZeroLoop


def test_phase0_configs_parse():
    for name in ("countdown", "countdown_tiny", "algebra", "algebra_tiny"):
        cfg = load_config(f"configs/phase0/{name}.yaml")
        assert cfg.env.name in ("countdown", "algebra")
        assert cfg.milestone.search_label in [s.label for s in cfg.eval.searches]


def test_overrides_and_unknown_keys():
    cfg = load_config("configs/phase0/countdown_tiny.yaml", ["iterations=2", "search.params.num_simulations=4"])
    assert cfg.iterations == 2 and cfg.search.params["num_simulations"] == 4
    try:
        config_from_dict({"name": "x", "env": {"name": "countdown"}, "search": {"name": "puct"}, "typo": 1})
    except KeyError as e:
        assert "typo" in str(e)
    else:
        raise AssertionError("unknown key accepted")


def test_loop_runs_end_to_end(tiny_checkpoint, tmp_path):
    cfg = load_config("configs/phase0/algebra_tiny.yaml", [
        f"output_dir={tmp_path}", f"model.checkpoint={tiny_checkpoint}", "iterations=1",
        "teacher.problems=10", "teacher.epochs=1", "selfplay.episodes_per_iteration=2",
        "eval.problems=3", "search.params.num_simulations=4", "save_checkpoints=last", "gate.enabled=true",
    ])
    loop = AlphaZeroLoop(cfg)
    loop.run()
    records = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    kinds = {r["kind"] for r in records}
    assert {"eval", "train", "selfplay", "gate"} <= kinds
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "checkpoints" / "final" / "model.safetensors").exists()


def test_gate_on_validation_split_and_test_reports_only_kept_weights(tiny_checkpoint, tmp_path):
    # tolerance -1: every self-play iteration is rejected, so its weights must never reach the test set
    cfg = load_config("configs/phase0/algebra_tiny.yaml", [
        f"output_dir={tmp_path}", f"model.checkpoint={tiny_checkpoint}", "iterations=2",
        "teacher.problems=10", "teacher.epochs=1", "selfplay.episodes_per_iteration=2",
        "eval.problems=3", "eval.gate_problems=2", "eval.baselines=[]", "search.params.num_simulations=4",
        "save_checkpoints=none", "gate.enabled=true", "gate.tolerance=-1",
    ])
    loop = AlphaZeroLoop(cfg)
    assert len(loop.val_problems) == 2
    loop.run()
    recs = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    val = [r for r in recs if r["kind"] == "eval" and r["split"] == "val"]
    assert {r["label"] for r in val} == {"puct16"} and {r["problems"] for r in val} == {2}
    gates = [r for r in recs if r["kind"] == "gate"]
    assert all(g["split"] == "val" for g in gates)
    assert [g["accepted"] for g in gates if g["iteration"] > 0] == [False, False]
    test = {(r["iteration"], r["label"]): r for r in recs if r["kind"] == "eval" and r["split"] == "eval"}
    for it in (1, 2):  # reverted to the warm-start weights: their test results, reused
        assert test[(it, "puct16")]["reused"] and test[(it, "puct16")]["reward"] == test[(0, "puct16")]["reward"]
    # the learning curve shows test results only: initial, warm start, 2 self-play iterations
    assert [(r["iteration"], r["stage"]) for r in loop.curve()] == [
        (0, "initial"), (0, "warmstart"), (1, "selfplay"), (2, "selfplay")]
    assert [r["puct16.reward"] for r in loop.curve()][1:] == [test[(0, "puct16")]["reward"]] * 3
    assert "## Gate" in (tmp_path / "summary.md").read_text()


def test_teacher_only_control_adds_no_selfplay_samples(tiny_checkpoint, tmp_path):
    cfg = load_config("configs/phase0/algebra_tiny.yaml", [
        f"output_dir={tmp_path}", f"model.checkpoint={tiny_checkpoint}", "iterations=1",
        "teacher.problems=10", "teacher.epochs=1", "train.teacher_mix=1.0", "selfplay.episodes_per_iteration=2",
        "selfplay.add_to_replay=false", "eval.problems=2", "eval.baselines=[]", "save_checkpoints=none",
        "search.params.num_simulations=4",
    ])
    loop = AlphaZeroLoop(cfg)
    loop.run()
    recs = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    warm = next(r for r in recs if r["kind"] == "train" and r["stage"] == "warmstart")
    sp = next(r for r in recs if r["kind"] == "selfplay")
    assert sp["samples"] == 0 and sp["replay"] == warm["samples"]
