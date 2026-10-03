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
