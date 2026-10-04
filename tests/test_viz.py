import json
import random

import numpy as np
import pytest

from mcts_laya.config import load_config
from mcts_laya.envs import CountdownEnv, LinearEquationEnv
from mcts_laya.evaluators import RolloutEvaluator
from mcts_laya.registry import SEARCHERS
from mcts_laya.selfplay import SelfPlayConfig, play_episode
from mcts_laya.viz.context import VizContext
from mcts_laya.viz.report import build_report, render_html
from mcts_laya.viz.server import LiveApp
from mcts_laya.viz.trace import curves_from_metrics, episode_to_json, tree_to_json


def _episode(env, name="puct", sims=24):
    s = SEARCHERS.build(name, env, RolloutEvaluator(), num_simulations=sims)
    return play_episode(env, s, env.sample_problem(random.Random(0)), np.random.default_rng(0), SelfPlayConfig(add_noise=False))


def test_tree_json_is_capped_and_consistent():
    env = LinearEquationEnv()
    ep = _episode(env)
    root = ep.steps[0].result.extra["root"]
    t = tree_to_json(env, root, max_nodes=10, max_children=3)
    assert t["size"] <= 10 and t["n"] == int(round(root.visit_count))
    def walk(n):
        assert len(n["children"]) <= 3
        assert sum(c["n"] for c in n["children"]) <= n["n"]
        for c in n["children"]:
            walk(c)
    walk(t)
    json.dumps(t)  # serialisable


@pytest.mark.parametrize("env", [CountdownEnv(), LinearEquationEnv()])
def test_episode_json_has_render_and_candidates(env):
    ep = _episode(env)
    d = episode_to_json(env, ep)
    assert len(d["steps"]) == ep.length and d["outcome"]["moves"] == ep.length
    st = d["steps"][0]
    assert st["render"]["kind"] == env.name and st["actions"] and "tree" in st
    assert abs(sum(a["policy"] for a in st["actions"]) - 1) < 1e-3
    json.dumps(d)


def test_teacher_searcher_and_curves():
    env = CountdownEnv()
    ep = _episode(env, "teacher")
    assert ep.success
    recs = [{"kind": "eval", "iteration": 0, "stage": "baseline", "label": "u", "success": 0.3, "reward": 0.3, "moves": 3},
            {"kind": "eval", "iteration": 0, "stage": "initial", "label": "g", "success": 0.1, "reward": 0.1, "moves": 3},
            {"kind": "eval", "iteration": 1, "stage": "selfplay", "label": "g", "success": 0.5, "reward": 0.5, "moves": 3},
            {"kind": "train", "iteration": 1, "stage": "selfplay", "policy_ce": 1.0, "value_ce": 0.5}]
    c = curves_from_metrics(recs)
    assert c["xs"] == ["0:initial", "1:selfplay"] and c["baselines"][0]["label"] == "u" and len(c["train"]) == 1


@pytest.fixture()
def ctx(tiny_checkpoint):
    cfg = load_config("configs/phase0/countdown_tiny.yaml", [f"model.checkpoint={tiny_checkpoint}", "model.device=cpu"])
    return VizContext(cfg)


def test_report_html_embeds_data(ctx):
    data = build_report(ctx, n_problems=1, methods=["greedy", "puct16", "teacher"], log=lambda *_: None)
    assert set(data["traces"]["p0"]) == {"greedy", "puct16", "teacher"}
    assert "tree" in data["traces"]["p0"]["puct16"]["steps"][0]
    assert "tree" not in data["traces"]["p0"]["teacher"]["steps"][0]
    html = render_html(data)
    assert "/*__DATA__*/" not in html and "/*__JS__*/" not in html and "window.VIZ_MODE = \"static\"" in html
    assert "</script>" not in html.split("window.VIZ_DATA = ", 1)[1].split(";\n</script>", 1)[0]


def test_report_roundtrip(tmp_path):
    from mcts_laya.viz.report import read_report_data, rerender, write_report

    data = {"meta": {"methods": [], "name": "x</script>y"}, "problems": [], "traces": {}, "curves": None}
    p = write_report(str(tmp_path / "r.html"), data)
    assert read_report_data(p) == data
    rerender(p)
    assert read_report_data(p) == data


def test_live_api_flow(ctx):
    app = LiveApp(ctx)
    r = app.new({"split": "eval", "index": 1})
    sid = r["session"]
    assert r["state"]["actions"] and not r["state"]["done"]
    s = app.search({"session": sid, "method": "puct16", "simulations": 8})["step"]
    assert s["simulations"] == 8 and s["tree"]["size"] >= 1
    done, guard = False, 0
    while not done and guard < 10:
        out = app.act({"session": sid, "index": 0})
        done, guard = out["state"]["done"], guard + 1
    assert done and "outcome" in out["state"]
    with pytest.raises(ValueError):
        app.search({"session": sid, "method": "puct16"})
    cmp = app.compare({"session": sid})
    assert "teacher" in cmp["traces"]
