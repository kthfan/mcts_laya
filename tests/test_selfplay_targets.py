import json
import random

import numpy as np
import pytest

from mcts_laya.config import load_config
from mcts_laya.envs import CountdownEnv, LinearEquationEnv
from mcts_laya.evaluators import RolloutEvaluator
from mcts_laya.pipeline import AlphaZeroLoop
from mcts_laya.registry import SEARCHERS
from mcts_laya.selfplay import SelfPlayConfig, episode_to_samples, play_episode
from mcts_laya.selfplay.actor import Episode
from mcts_laya.selfplay.targets import policy_weights, relabel_with_teacher, update_best_moves
from mcts_laya.teachers import AlgebraTeacher


class _Env:
    def state_key(self, s):
        return s


def _ep(problem, success, moves, reward):
    ep = Episode(initial_state=problem)
    ep.steps = [None] * moves
    ep.success = success
    ep.final_value = 2 * reward - 1
    return ep


def test_policy_filters():
    env = _Env()
    eps = [_ep("a", True, 5, 0.8), _ep("a", True, 9, 0.6), _ep("b", True, 4, 0.83), _ep("b", False, 20, 0.0)]
    best = {}
    update_best_moves(env, eps, best)
    assert best == {"a": 5, "b": 4}
    assert policy_weights(env, eps, "none", best) == [0, 0, 0, 0]
    assert policy_weights(env, eps, "all", best) == [1, 1, 1, 1]
    assert policy_weights(env, eps, "success", best) == [1, 1, 1, 0]
    # A: 9 moves > 1.3 x 5 is a detour; top half of successes by reward keeps 0.8 and 0.83
    assert policy_weights(env, eps, "efficient", best, best_known_ratio=1.3, top_fraction=0.5) == [1, 0, 1, 0]
    assert policy_weights(env, eps, "efficient", best, best_known_ratio=None, top_fraction=1.0) == [1, 1, 1, 0]
    with pytest.raises(ValueError):
        policy_weights(env, eps, "typo", best)


def test_policy_weight_override_reaches_samples():
    env = CountdownEnv()
    searcher = SEARCHERS.build("puct", env, RolloutEvaluator(), num_simulations=8)
    ep = play_episode(env, searcher, env.sample_problem(random.Random(0)), np.random.default_rng(0))
    assert all(s.policy_weight == 0.0 for s in episode_to_samples(env, ep, policy_weight=0.0))


def test_dagger_relabel_uses_teacher_targets():
    env = LinearEquationEnv()
    searcher = SEARCHERS.build("puct", env, RolloutEvaluator(), num_simulations=8)
    ep = play_episode(env, searcher, env.sample_problem(random.Random(1)), np.random.default_rng(0),
                      SelfPlayConfig(add_noise=False))
    teacher = AlgebraTeacher(env)
    samples = episode_to_samples(env, ep)
    out = relabel_with_teacher(env, teacher, ep, samples, value="teacher")
    for rec, s in zip(ep.steps, out):
        _, pi, v = teacher.label(rec.state)
        assert s.policy == pytest.approx(list(pi)) and s.value == pytest.approx(v)
        assert s.policy_weight == 1.0 and s.meta["source"] == "dagger"
    kept = relabel_with_teacher(env, teacher, ep, samples, value="outcome")
    assert [s.value for s in kept] == [s.value for s in samples]


@pytest.mark.parametrize("overrides", [
    ["selfplay.policy_filter=efficient"],
    ["selfplay.policy_filter=none"],
    ["selfplay.relabel=teacher"],
])
def test_pipeline_variants_run(overrides, tiny_checkpoint, tmp_path):
    cfg = load_config("configs/phase0/algebra_tiny.yaml", [
        f"output_dir={tmp_path}", f"model.checkpoint={tiny_checkpoint}", "iterations=1", "teacher.problems=3",
        "teacher.epochs=1", "selfplay.episodes_per_iteration=3", "eval.problems=2", "search.params.num_simulations=4",
        "eval.baselines=[]", "save_checkpoints=none", *overrides])
    AlphaZeroLoop(cfg).run()
    sp = [json.loads(l) for l in open(tmp_path / "metrics.jsonl") if '"selfplay"' in l]
    assert sp and "policy_episodes" in sp[0]
    if "selfplay.policy_filter=none" in overrides:
        assert sp[0]["policy_episodes"] == 0


def test_invalid_policy_filter_is_rejected(tiny_checkpoint, tmp_path):
    cfg = load_config("configs/phase0/algebra_tiny.yaml", [f"output_dir={tmp_path}", f"model.checkpoint={tiny_checkpoint}",
                                                          "selfplay.policy_filter=typo"])
    with pytest.raises(ValueError):
        AlphaZeroLoop(cfg)
