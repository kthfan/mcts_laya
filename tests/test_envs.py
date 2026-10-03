import random

import pytest

from mcts_laya.envs import CountdownEnv, LinearEquationEnv
from mcts_laya.envs.algebra import C, G, X, AlgebraState, _has_x, render_equation, sympy_solutions
from mcts_laya.envs.countdown import CountdownState, legal_moves, steps_to_solve
from mcts_laya.registry import ENVIRONMENTS


def test_registry_builds_environments():
    assert {"countdown", "algebra"} <= set(ENVIRONMENTS.names())
    assert isinstance(ENVIRONMENTS.build("countdown", n_numbers=3), CountdownEnv)


def test_countdown_moves_are_exact_and_positive():
    for m in legal_moves((2, 3, 6, 7)):
        assert m.result > 0
        assert {"+": m.a + m.b, "-": m.a - m.b, "*": m.a * m.b, "/": m.a / m.b}[m.op] == m.result


def test_countdown_problems_are_solvable_and_terminal_rules():
    env = CountdownEnv()
    rng = random.Random(0)
    for _ in range(50):
        s = env.sample_problem(rng)
        assert not env.is_terminal(s)
        assert steps_to_solve(s.numbers, s.target, False) >= env.min_solution_steps
    win = CountdownState((3, 24), 24)
    assert env.is_terminal(win) and env.terminal_reward(win) == 1.0 and env.terminal_value(win) == 1.0
    lose = CountdownState((25,), 24)
    assert env.is_terminal(lose) and env.terminal_value(lose) == -1.0


def test_countdown_use_all_rule():
    env = CountdownEnv(n_numbers=4, use_all=True, target_range=(1, 500))
    s = env.sample_problem(random.Random(1))
    assert steps_to_solve(s.numbers, s.target, True) == 3
    assert not env.is_terminal(CountdownState((4, 24), 24))  # 24 present but numbers left over


def test_algebra_rendering():
    assert render_equation((G(3, X(1), C(2)), C(-4)), (X(1), C(10))) == "3(x + 2) - 4 = x + 10"
    assert render_equation((G("1/4", X(1), C(-3)),), ()) == "(x - 3)/4 = 0"
    assert render_equation((X(-1),), (C("3/2"),)) == "-x = 3/2"


@pytest.mark.parametrize("template", LinearEquationEnv.TEMPLATES)
def test_algebra_templates_have_the_stated_solution(template):
    env = LinearEquationEnv(templates=(template,))
    rng = random.Random(3)
    for _ in range(10):
        s = env.sample_problem(rng)
        assert sympy_solutions(s.lhs, s.rhs) == [s.solution]


def test_algebra_every_step_preserves_the_solution():
    env = LinearEquationEnv()
    rng = random.Random(4)
    for _ in range(30):
        s = env.sample_problem(rng)
        while not env.is_terminal(s):
            for a in env.legal_actions(s):
                c = env.step(s, a)
                if _has_x(c.lhs) or _has_x(c.rhs):
                    assert sympy_solutions(c.lhs, c.rhs) == [s.solution], (env.state_text(s), env.action_text(s, a))
            s = env.step(s, rng.choice(env.legal_actions(s)))


def test_algebra_solved_state_and_reward():
    env = LinearEquationEnv(gamma=0.9, success_floor=0.5)
    s = AlgebraState((X(2),), (C(6),), steps=2, solution=3, original="2x = 6")
    (div,) = [a for a in env.legal_actions(s) if a.kind == "divide"]
    t = env.step(s, div)
    assert env.is_solved(t) and env.is_terminal(t)
    assert env.terminal_reward(t) == pytest.approx(0.5 + 0.5 * 0.9 ** 3)
    assert env.action_text(s, div) == "divide both sides by 2"


def test_step_limit_is_terminal_failure():
    env = LinearEquationEnv(max_steps=3)
    s = AlgebraState((X(2), C(1)), (C(7),), steps=3, solution=3)
    assert env.is_terminal(s) and env.terminal_value(s) == -1.0
