"""Parallel actors: same episodes as in-process play, and the full loop runs with them."""

import json
import random

import numpy as np

from mcts_laya.config import load_config
from mcts_laya.envs.algebra import LinearEquationEnv
from mcts_laya.evaluators.laya_evaluator import LayaEvaluator
from mcts_laya.pipeline import AlphaZeroLoop
from mcts_laya.registry import SEARCHERS
from mcts_laya.selfplay import SelfPlayConfig, play_episode
from mcts_laya.selfplay.parallel import ActorPool, episode_task, resolve_workers, search_spec


def _moves(eps):
    return [[r.action_index for r in ep.steps] + [round(ep.final_value, 9)] for ep in eps]


def test_actor_pool_matches_in_process_play(tiny_agent):
    env = LinearEquationEnv()
    ev = LayaEvaluator(tiny_agent, max_rows=64)
    cfg = SelfPlayConfig(max_moves=8, add_noise=True)
    params = {"num_simulations": 8, "batch_size": 4}
    problems = [env.sample_problem(random.Random(i)) for i in range(6)]
    searcher = SEARCHERS.build("puct", env, ev, **params)
    serial = [play_episode(env, searcher, p, np.random.default_rng(i), cfg) for i, p in enumerate(problems)]
    with ActorPool("algebra", {}, ev, workers=2) as pool:
        tasks = [episode_task(search_spec("puct", params), p, i, cfg) for i, p in enumerate(problems)]
        parallel = pool.run(tasks)
        # a baseline evaluator runs inside the actors, without model calls
        batches = pool.batches
        pool.run([episode_task(search_spec("puct", params, "uniform"), problems[0], 0, cfg)])
        assert pool.batches == batches
    assert _moves(parallel) == _moves(serial)
    assert all("root" not in r.result.extra for ep in parallel for r in ep.steps)


def test_loop_runs_with_actors(tiny_checkpoint, tmp_path):
    cfg = load_config("configs/phase0/algebra_tiny.yaml", [
        f"output_dir={tmp_path}", f"model.checkpoint={tiny_checkpoint}", "iterations=1",
        "teacher.problems=10", "teacher.epochs=1", "selfplay.episodes_per_iteration=4",
        "eval.problems=3", "search.params.num_simulations=4", "parallel.workers=2",
    ])
    AlphaZeroLoop(cfg).run()
    records = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    sp = [r for r in records if r["kind"] == "selfplay"]
    assert sp and sp[0]["workers"] == 2 and sp[0]["episodes"] == 4 and sp[0]["rows_per_batch"] > 0
    assert 0 <= sp[0]["actor_wait"] <= 1 and sp[0]["model_ms_per_batch"] > 0
    assert {r["label"] for r in records if r["kind"] == "eval"} >= {"greedy", "puct16", "uniform-puct16"}


def test_restarted_run_starts_a_fresh_metrics_file(tiny_checkpoint, tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"kind": "eval", "stage": "stale"}\n')
    cfg = load_config("configs/phase0/algebra_tiny.yaml", [
        f"output_dir={tmp_path}", f"model.checkpoint={tiny_checkpoint}", "iterations=0", "teacher.enabled=false",
        "eval.problems=2", "eval.baselines=[]", "parallel.workers=0"])
    AlphaZeroLoop(cfg).run()
    assert "stale" not in (tmp_path / "metrics.jsonl").read_text()
    assert "stale" in (tmp_path / "metrics.interrupted.jsonl").read_text()


def test_resolve_workers():
    assert resolve_workers("auto", "cpu", 8) == 0
    assert resolve_workers("auto", "cuda:0", 8) == 7
    assert resolve_workers("auto", "cuda", 64) == 32
    assert resolve_workers(3, "cpu", 8) == 3 and resolve_workers(0, "cuda", 8) == 0
