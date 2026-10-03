import random

import numpy as np
import pytest

from mcts_laya.envs import CountdownEnv, LinearEquationEnv
from mcts_laya.envs.base import Environment
from mcts_laya.evaluators import EvalResult, Evaluator, RolloutEvaluator, UniformEvaluator
from mcts_laya.registry import SEARCHERS
from mcts_laya.selfplay import SelfPlayConfig, episode_to_samples, play_episode
from mcts_laya.teachers import AlgebraTeacher, CountdownTeacher


class TeacherEvaluator(Evaluator):
    """Perfect knowledge, for testing that search uses priors and values."""

    def __init__(self, teacher):
        self.teacher = teacher

    def evaluate(self, env, states, actions):
        out = []
        for s, a in zip(states, actions):
            value, best = self.teacher.analyse(s, a)
            p = np.full(len(a), 0.01)
            p[best] = 1.0
            out.append(EvalResult(p / p.sum(), value))
        return out


class Nim(Environment):
    """Take 1-2 stones; taking the last stone wins. Positions with n % 3 == 0 are lost."""

    name = "nim"
    num_players = 2

    def sample_problem(self, rng):
        return (rng.randint(4, 10), 0)

    def legal_actions(self, s):
        return [k for k in (1, 2) if k <= s[0]]

    def step(self, s, a):
        return (s[0] - a, 1 - s[1])

    def is_terminal(self, s):
        return s[0] == 0

    def terminal_value(self, s):
        return -1.0  # the previous player took the last stone

    def current_player(self, s):
        return s[1]

    def state_text(self, s):
        return f"{s[0]} stones"

    def action_text(self, s, a):
        return f"take {a}"


@pytest.mark.parametrize("name", ["puct", "gumbel"])
def test_two_player_backup_finds_winning_move(name):
    env = Nim()
    searcher = SEARCHERS.build(name, env, RolloutEvaluator(n_rollouts=4), num_simulations=200)
    for n in (4, 5, 7, 8):
        r = searcher.search((n, 0), np.random.default_rng(0), add_noise=False)
        assert (n - r.actions[r.selected]) % 3 == 0, (n, r.visit_counts, r.q_values)
        assert r.root_value > 0


@pytest.mark.parametrize("name", ["puct", "gumbel"])
def test_search_with_perfect_evaluator_solves_everything(name):
    rng = random.Random(0)
    for env, teacher in ((CountdownEnv(), CountdownTeacher), (LinearEquationEnv(), AlgebraTeacher)):
        searcher = SEARCHERS.build(name, env, TeacherEvaluator(teacher(env)), num_simulations=8)
        for _ in range(5):
            ep = play_episode(env, searcher, env.sample_problem(rng), np.random.default_rng(0),
                              SelfPlayConfig(add_noise=False))
            assert ep.success


def test_policy_targets_are_distributions_and_budget_respected():
    env = CountdownEnv()
    s = env.sample_problem(random.Random(1))
    for name, params in (("puct", {"num_simulations": 24, "batch_size": 4}), ("gumbel", {"num_simulations": 24})):
        r = SEARCHERS.build(name, env, UniformEvaluator(), **params).search(s, np.random.default_rng(0))
        assert r.policy.shape == (len(r.actions),)
        assert r.policy.sum() == pytest.approx(1.0) and (r.policy >= 0).all()
        assert r.num_simulations == 24
        assert r.visit_counts.sum() == 24


def test_virtual_loss_is_fully_reverted():
    env = LinearEquationEnv()
    s = env.sample_problem(random.Random(2))
    searcher = SEARCHERS.build("puct", env, UniformEvaluator(), num_simulations=40, batch_size=8)
    r = searcher.search(s, np.random.default_rng(0))
    assert all(float(v).is_integer() for v in r.visit_counts)


def test_episode_samples_carry_outcome_values():
    env = CountdownEnv()
    searcher = SEARCHERS.build("gumbel", env, TeacherEvaluator(CountdownTeacher(env)), num_simulations=8)
    ep = play_episode(env, searcher, env.sample_problem(random.Random(3)), np.random.default_rng(0))
    samples = episode_to_samples(env, ep)
    assert len(samples) == ep.length
    assert all(s.value == (1.0 if ep.success else -1.0) for s in samples)
    assert all(len(s.policy) == len(s.action_texts) for s in samples)


def test_teachers_label_optimal_moves():
    env = CountdownEnv()
    t = CountdownTeacher(env)
    samples = t.generate(random.Random(0), 10)
    assert samples and all(abs(sum(s.policy) - 1) < 1e-9 for s in samples)
    env2 = LinearEquationEnv()
    t2 = AlgebraTeacher(env2)
    s = env2.sample_problem(random.Random(5))
    solved, moves = t2.solve(s)
    assert solved and moves <= env2.max_steps
