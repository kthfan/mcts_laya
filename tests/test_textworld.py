import random

import numpy as np
import pytest

textworld = pytest.importorskip("textworld")

from mcts_laya.envs.textworld_env import TextWorldEnv, clean_feedback, clean_objective, clean_room  # noqa: E402
from mcts_laya.envs.textworld_games import LEVELS, generate_pool, load_manifest  # noqa: E402
from mcts_laya.registry import SEARCHERS  # noqa: E402
from mcts_laya.selfplay import SelfPlayConfig, play_episode  # noqa: E402
from mcts_laya.teachers import TextWorldTeacher  # noqa: E402


@pytest.fixture(scope="session")
def tw_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("tw")
    generate_pool(str(root), "L1", n_train=3, n_eval=2, workers=4)
    return str(root)


@pytest.fixture()
def env(tw_root):
    return TextWorldEnv(game_dir=tw_root, level="L1")


def test_pool_manifest_has_disjoint_splits(tw_root):
    m = load_manifest(f"{tw_root}/L1")
    assert len(m["games"]["train"]) == 3 and len(m["games"]["eval"]) == 2
    assert not set(m["games"]["train"]) & set(m["games"]["eval"])
    assert m["options"]["quest_length"] == LEVELS["L1"].quest_length


def test_text_cleaning():
    assert clean_objective("Welcome to TextWorld! Here is your task for today. First, go south. Got that? Good!") == \
        "First, go south."
    assert clean_room("-= Spare Room =-\nYou are   here.") == "Spare Room. You are here."
    assert clean_feedback("\nYou can't go that way.\n\n\n>    -= Cookhouse =-0/2") == "You can't go that way."


def test_replay_is_deterministic_and_states_are_immutable(env):
    s0 = env.sample_problem(random.Random(0))
    a = env.legal_actions(s0)
    s1 = env.step(s0, a[0])
    s2 = env.step(s0, a[-1])  # branching from the same parent must not disturb s1
    again = env.step(s0, a[0])
    assert s0.history == () and s1.history == (a[0],) and s2.history == (a[-1],)
    assert again == s1 and again.obs == s1.obs
    assert env.state_text(s1) == env.state_text(again)


def test_filtering_and_sampling(env):
    s = env.sample_problem(random.Random(1), "eval")
    assert s.game.startswith("eval/")
    acts = env.legal_actions(s)
    assert acts and not any(c.startswith(("look", "inventory", "examine")) for c in acts)
    probs = env.sample_problems(random.Random(2), 2, "eval")
    assert len({p.game for p in probs}) == 2  # no repeats while the pool allows it


def test_teacher_solves_and_rewards(env):
    teacher = TextWorldTeacher(env)
    for s in env.sample_problems(random.Random(3), 2, "eval"):
        solved, moves = teacher.solve(s)
        assert solved and moves == len(s.obs.policy)
    s = env.sample_problem(random.Random(4))
    for cmd in s.obs.policy:
        s = env.step(s, cmd)
    assert env.is_terminal(s) and env.is_success(s)
    assert env.terminal_reward(s) == pytest.approx(env.solved_reward(len(s.history)))
    samples = teacher.generate(random.Random(5), 2)
    assert samples and all(abs(sum(x.policy) - 1) < 1e-9 for x in samples)


def test_step_limit_ends_episode(tw_root):
    env = TextWorldEnv(game_dir=tw_root, level="L1", max_steps=2)
    s = env.sample_problem(random.Random(6))
    while not env.is_terminal(s):
        s = env.step(s, env.legal_actions(s)[0])
    assert len(s.history) <= 2
    if not env.is_success(s):
        assert env.terminal_value(s) == -1.0


def test_search_plays_an_episode(env):
    from mcts_laya.evaluators import RolloutEvaluator

    searcher = SEARCHERS.build("puct", env, RolloutEvaluator(), num_simulations=8)
    ep = play_episode(env, searcher, env.sample_problem(random.Random(7)), np.random.default_rng(0),
                      SelfPlayConfig(add_noise=False, max_moves=env.max_steps))
    assert 1 <= ep.length <= env.max_steps


def test_laya_evaluator_on_textworld(env, tiny_agent):
    from mcts_laya.evaluators import LayaEvaluator

    ev = LayaEvaluator(tiny_agent)
    s = env.sample_problem(random.Random(8))
    (r,) = ev.evaluate(env, [s], [env.legal_actions(s)])
    assert r.priors.sum() == pytest.approx(1.0) and -1 <= r.value <= 1
