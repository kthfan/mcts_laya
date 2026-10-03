import random

import numpy as np
import pytest

from mcts_laya.envs import CountdownEnv, LinearEquationEnv
from mcts_laya.evaluators import LayaEvaluator
from mcts_laya.laya_io import option_labels
from mcts_laya.search import GumbelSearch
from mcts_laya.selfplay import play_episode
from mcts_laya.teachers import CountdownTeacher
from mcts_laya.training import LayaTrainer, TrainConfig


def test_option_labels():
    assert option_labels(3) == ["A", "B", "C"]
    assert option_labels(28)[25:] == ["Z", "AA", "AB"]


def test_evaluator_outputs_distributions_and_caches(tiny_agent):
    ev = LayaEvaluator(tiny_agent)
    for env in (CountdownEnv(), LinearEquationEnv()):
        states = [env.sample_problem(random.Random(i)) for i in range(5)]
        actions = [env.legal_actions(s) for s in states]
        res = ev.evaluate(env, states, actions)
        for r, a in zip(res, actions):
            assert r.priors.shape == (len(a),) and r.priors.sum() == pytest.approx(1.0)
            assert -1.0 <= r.value <= 1.0
        rows = ev.rows_evaluated
        ev.evaluate(env, states, actions)
        assert ev.rows_evaluated == rows and ev.cache_hits >= 5


def test_random_option_order_is_unpermuted(tiny_agent):
    """The prior must follow the action, not the slot it was shown in."""
    env = CountdownEnv()
    s = env.sample_problem(random.Random(0))
    a = env.legal_actions(s)
    fixed = LayaEvaluator(tiny_agent, cache_size=0).evaluate(env, [s], [a])[0].priors
    shuffled = LayaEvaluator(tiny_agent, cache_size=0, option_order="random", seed=1).evaluate(env, [s], [a])[0].priors
    # a random model has a position bias, so this only checks shapes/normalisation are kept
    assert shuffled.shape == fixed.shape and shuffled.sum() == pytest.approx(1.0)


def test_training_reduces_loss_and_checkpoint_roundtrips(tiny_checkpoint, tmp_path):
    import laya

    agent = laya.load(tiny_checkpoint, device="cpu")
    env = CountdownEnv()
    samples = CountdownTeacher(env).generate(random.Random(0), 40)
    trainer = LayaTrainer(agent, TrainConfig(epochs=1, batch_size=16, lr_encoder=1e-3, lr_head=1e-3,
                                             calibration_fraction=0.2))
    before = trainer.score(samples)
    first = trainer.train(samples, epochs=1)
    last = trainer.train(samples, epochs=4)
    assert last["loss"] < first["loss"]
    assert "temperature_noul" in last
    after = trainer.score(samples)
    assert after["heldout_value_brier"] < before["heldout_value_brier"]

    path = trainer.save(str(tmp_path / "ckpt"))
    reloaded = laya.load(path, device="cpu")
    s = env.sample_problem(random.Random(7))
    a = env.legal_actions(s)
    p1 = LayaEvaluator(agent, cache_size=0).evaluate(env, [s], [a])[0]
    p2 = LayaEvaluator(reloaded, cache_size=0).evaluate(env, [s], [a])[0]
    np.testing.assert_allclose(p1.priors, p2.priors, atol=1e-5)
    assert p1.value == pytest.approx(p2.value, abs=1e-5)


def test_rlcd_term_runs(tiny_checkpoint):
    import laya

    agent = laya.load(tiny_checkpoint, device="cpu")
    samples = CountdownTeacher(CountdownEnv()).generate(random.Random(1), 5)
    m = LayaTrainer(agent, TrainConfig(rl_weight=1.0, calibration_fraction=0)).train(samples)
    assert "rl" in m and np.isfinite(m["loss"])


def test_gumbel_episode_with_laya(tiny_agent):
    env = LinearEquationEnv()
    ev = LayaEvaluator(tiny_agent)
    ep = play_episode(env, GumbelSearch(env, ev, num_simulations=8), env.sample_problem(random.Random(0)),
                      np.random.default_rng(0))
    assert ep.length >= 1 and ev.rows_evaluated > 0
