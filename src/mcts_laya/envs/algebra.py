"""Linear equation solving by explicit rewrite steps.

An equation is two sides, each a sum of terms. A term is an x-term (`3x`), a constant (`-4`) or a
group (`3(x + 2)`, `(x + 1)/4`) holding x-terms and constants. Every action is one textbook step:
expand a group, combine like terms, move a term across the equals sign, clear fractions, divide by
the coefficient of x, or swap the sides. The episode is solved when it reads `x = c`.

Arithmetic is exact (`fractions.Fraction`); SymPy is used as an independent oracle to verify that
generated problems have the intended unique solution (and, in the tests, that every step keeps
the solution set).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from ..registry import ENVIRONMENTS
from .base import SingleAgentEnvironment

LEFT, RIGHT = "left", "right"


@dataclass(frozen=True)
class Term:
    kind: str  # "x" | "c" | "g"
    coef: Fraction  # x: coefficient, c: value, g: multiplier
    inner: Tuple["Term", ...] = ()  # g only: x-terms and constants

    def scaled(self, k: Fraction) -> "Term":
        return Term(self.kind, self.coef * k, self.inner)

    def sort_key(self):
        return (self.kind, self.coef, tuple(t.sort_key() for t in self.inner))


def X(c) -> Term:
    return Term("x", Fraction(c))


def C(c) -> Term:
    return Term("c", Fraction(c))


def G(m, *inner: Term) -> Term:
    return Term("g", Fraction(m), tuple(inner))


Side = Tuple[Term, ...]


@dataclass(frozen=True)
class AlgebraState:
    lhs: Side
    rhs: Side
    steps: int = 0
    solution: Fraction = field(default=Fraction(0), compare=False)
    original: str = field(default="", compare=False)


@dataclass(frozen=True)
class AlgebraAction:
    kind: str  # expand | combine | move | divide | multiply | swap
    side: str = ""
    index: int = -1
    value: Fraction = Fraction(0)


# --- rendering ------------------------------------------------------------------------------
def fmt_num(q: Fraction) -> str:
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def _body(t: Term) -> str:
    """Text of |t| (the sign is handled by the caller)."""
    a = abs(t.coef)
    if t.kind == "x":
        if a == 1:
            return "x"
        return f"{fmt_num(a)}x" if a.denominator == 1 else f"({fmt_num(a)})x"
    if t.kind == "c":
        return fmt_num(a)
    inner = render_side(t.inner)
    if a == 1:
        return f"({inner})"
    if a.denominator == 1:
        return f"{fmt_num(a)}({inner})"
    if a.numerator == 1:
        return f"({inner})/{a.denominator}"
    return f"({fmt_num(a)})({inner})"


def render_side(side: Sequence[Term]) -> str:
    if not side:
        return "0"
    out = []
    for i, t in enumerate(side):
        neg = t.coef < 0
        if i == 0:
            out.append(("-" if neg else "") + _body(t))
        else:
            out.append((" - " if neg else " + ") + _body(t))
    return "".join(out)


def render_equation(lhs: Sequence[Term], rhs: Sequence[Term]) -> str:
    return f"{render_side(lhs)} = {render_side(rhs)}"


# --- algebra helpers ------------------------------------------------------------------------
def _has_x(side: Sequence[Term]) -> bool:
    return any(t.kind == "x" or (t.kind == "g" and _has_x(t.inner)) for t in side)


def _combine(side: Side) -> Side:
    xs = sum((t.coef for t in side if t.kind == "x"), Fraction(0))
    cs = sum((t.coef for t in side if t.kind == "c"), Fraction(0))
    out, seen_x, seen_c = [], False, False
    for t in side:
        if t.kind == "x":
            if not seen_x and xs != 0:
                out.append(X(xs))
            seen_x = True
        elif t.kind == "c":
            if not seen_c and cs != 0:
                out.append(C(cs))
            seen_c = True
        else:
            out.append(t)
    return tuple(out)


def _can_combine(side: Side) -> bool:
    nx = sum(t.kind == "x" for t in side)
    nc = sum(t.kind == "c" for t in side)
    return nx >= 2 or nc >= 2 or any(t.kind != "g" and t.coef == 0 for t in side)


def _expand(side: Side, i: int) -> Side:
    g = side[i]
    return side[:i] + tuple(t.scaled(g.coef) for t in g.inner) + side[i + 1:]


def _denominators(side: Side):
    for t in side:
        yield t.coef.denominator


def to_sympy(side: Sequence[Term]):
    import sympy

    x = sympy.Symbol("x")
    expr = sympy.Integer(0)
    for t in side:
        q = sympy.Rational(t.coef.numerator, t.coef.denominator)
        if t.kind == "x":
            expr += q * x
        elif t.kind == "c":
            expr += q
        else:
            expr += q * to_sympy(t.inner)
    return expr


def sympy_solutions(lhs: Sequence[Term], rhs: Sequence[Term]):
    import sympy

    return sympy.solve(sympy.Eq(to_sympy(lhs), to_sympy(rhs)), sympy.Symbol("x"))


# --- environment ----------------------------------------------------------------------------
@ENVIRONMENTS.register("algebra")
class LinearEquationEnv(SingleAgentEnvironment):
    name = "algebra"
    policy_instruction = "Which algebra step should be applied next to solve the equation for x?"
    value_instruction = "Will the equation be solved for x within the remaining steps?"
    value_criteria = {
        "false": "no, it will not be solved in time",
        "true": "yes, it will be solved in time",
    }

    def __init__(
        self,
        templates: Sequence[str] = ("bracket", "fraction", "two_brackets", "two_fractions"),
        max_steps: int = 10,
        gamma: float = 0.9,
        coef_range: int = 9,
        shuffle_terms: bool = True,
        verify_with_sympy: bool = True,
    ):
        unknown = set(templates) - set(self.TEMPLATES)
        if unknown:
            raise ValueError(f"unknown templates {sorted(unknown)}; known: {self.TEMPLATES}")
        self.templates = tuple(templates)
        self.shuffle_terms = shuffle_terms
        self.max_steps = max_steps
        self.gamma = gamma
        self.coef_range = coef_range
        self.verify_with_sympy = verify_with_sympy

    # --- problems ---------------------------------------------------------------------------
    TEMPLATES = ("linear", "bracket", "fraction", "two_brackets", "two_fractions")

    def _nonzero(self, rng: random.Random, lo: int = 1) -> int:
        k = 0
        while abs(k) < lo:
            k = rng.randint(-self.coef_range, self.coef_range)
        return k

    def _template(self, rng: random.Random, name: str, s: int) -> Tuple[Side, Side]:
        """Equation of the given family whose unique solution is x = s (integer coefficients)."""
        nz = self._nonzero
        if name == "linear":  # ax + b = cx + d
            a, c, b = nz(rng), nz(rng), nz(rng)
            while c == a:
                c = nz(rng)
            return (X(a), C(b)), (X(c), C(a * s + b - c * s))
        if name == "bracket":  # a(x + b) + c = dx + e
            a, b, c, d = nz(rng, 2), nz(rng), nz(rng), nz(rng)
            while d == a:
                d = nz(rng)
            return (G(a, X(1), C(b)), C(c)), (X(d), C(a * (s + b) + c - d * s))
        if name == "fraction":  # (ax + b)/k = cx + d
            k = rng.choice([2, 3, 4, 5])
            a, c = nz(rng), nz(rng)
            while Fraction(a, k) == c:
                c = nz(rng)
            m = rng.randint(-self.coef_range, self.coef_range)
            b = k * m - a * s  # (a s + b) / k = m
            return (G(Fraction(1, k), X(a), C(b)),), (X(c), C(m - c * s))
        if name == "two_brackets":  # a(x + b) - c(x + d) = e
            a, b, c, d = nz(rng, 2), nz(rng), nz(rng, 2), nz(rng)
            while c == a:
                c = nz(rng, 2)
            return (G(a, X(1), C(b)), G(-c, X(1), C(d))), (C(a * (s + b) - c * (s + d)),)
        if name == "two_fractions":  # (x + a)/k + (x + b)/m = c
            k, m = rng.sample([2, 3, 4, 6], 2)
            r, t = rng.randint(-5, 5), rng.randint(-5, 5)
            a, b = k * r - s, m * t - s
            return (G(Fraction(1, k), X(1), C(a)), G(Fraction(1, m), X(1), C(b))), (C(r + t),)
        raise ValueError(f"unknown template {name!r}")

    def sample_problem(self, rng: random.Random) -> AlgebraState:
        for _ in range(1000):
            s = rng.randint(-self.coef_range, self.coef_range)
            lhs, rhs = self._template(rng, rng.choice(self.templates), s)
            lhs = tuple(t for t in lhs if not (t.kind == "c" and t.coef == 0))
            rhs = tuple(t for t in rhs if not (t.kind == "c" and t.coef == 0))
            if self.shuffle_terms:
                lhs, rhs = tuple(rng.sample(lhs, len(lhs))), tuple(rng.sample(rhs, len(rhs)))
                if rng.random() < 0.5:
                    lhs, rhs = rhs, lhs
            state = AlgebraState(lhs, rhs, 0, Fraction(s), render_equation(lhs, rhs))
            if self.is_terminal(state):
                continue
            if self.verify_with_sympy and sympy_solutions(lhs, rhs) != [s]:
                continue
            return state
        raise RuntimeError("could not sample an equation; loosen the generator settings")

    # --- dynamics -------------------------------------------------------------------------
    def legal_actions(self, state: AlgebraState) -> List[AlgebraAction]:
        acts: List[AlgebraAction] = []
        sides = ((LEFT, state.lhs), (RIGHT, state.rhs))
        for name, side in sides:
            for i, t in enumerate(side):
                if t.kind == "g":
                    acts.append(AlgebraAction("expand", name, i))
        for name, side in sides:
            if _can_combine(side):
                acts.append(AlgebraAction("combine", name))
        for name, side in sides:
            for i in range(len(side)):
                acts.append(AlgebraAction("move", name, i))
        k = 1
        for d in list(_denominators(state.lhs)) + list(_denominators(state.rhs)):
            k = k * d // math.gcd(k, d)
        if k > 1:
            acts.append(AlgebraAction("multiply", value=Fraction(k)))
        if (
            len(state.lhs) == 1
            and state.lhs[0].kind == "x"
            and state.lhs[0].coef not in (0, 1)
            and not _has_x(state.rhs)
            and all(t.kind == "c" for t in state.rhs)
        ):
            acts.append(AlgebraAction("divide", value=state.lhs[0].coef))
        if not _has_x(state.lhs) and _has_x(state.rhs):
            acts.append(AlgebraAction("swap"))
        return acts

    def step(self, state: AlgebraState, action: AlgebraAction) -> AlgebraState:
        lhs, rhs = state.lhs, state.rhs
        k = action.kind
        if k == "expand":
            if action.side == LEFT:
                lhs = _expand(lhs, action.index)
            else:
                rhs = _expand(rhs, action.index)
        elif k == "combine":
            if action.side == LEFT:
                lhs = _combine(lhs)
            else:
                rhs = _combine(rhs)
        elif k == "move":
            src, dst = (lhs, rhs) if action.side == LEFT else (rhs, lhs)
            t = src[action.index]
            src = src[: action.index] + src[action.index + 1:]
            dst = dst + (t.scaled(Fraction(-1)),)
            lhs, rhs = (src, dst) if action.side == LEFT else (dst, src)
        elif k == "multiply":
            lhs = tuple(t.scaled(action.value) for t in lhs)
            rhs = tuple(t.scaled(action.value) for t in rhs)
        elif k == "divide":
            inv = 1 / action.value
            lhs = tuple(t.scaled(inv) for t in lhs)
            rhs = tuple(t.scaled(inv) for t in rhs)
        elif k == "swap":
            lhs, rhs = rhs, lhs
        else:
            raise ValueError(f"unknown action {action}")
        return AlgebraState(lhs, rhs, state.steps + 1, state.solution, state.original)

    @staticmethod
    def is_solved(state: AlgebraState) -> bool:
        return (
            len(state.lhs) == 1
            and state.lhs[0] == X(1)
            and (len(state.rhs) == 0 or (len(state.rhs) == 1 and state.rhs[0].kind == "c"))
        )

    def is_terminal(self, state: AlgebraState) -> bool:
        if self.is_solved(state) or state.steps >= self.max_steps:
            return True
        return not (_has_x(state.lhs) or _has_x(state.rhs))

    def terminal_reward(self, state: AlgebraState) -> float:
        if not self.is_solved(state):
            return 0.0
        value = state.rhs[0].coef if state.rhs else Fraction(0)
        if state.original and value != state.solution:
            raise AssertionError(f"unsound rewrite: {state.original} solved as x = {value}")
        return float(self.gamma ** state.steps)

    # --- text -----------------------------------------------------------------------------
    def state_text(self, state: AlgebraState) -> str:
        left = self.max_steps - state.steps
        return f"Solve for x: {render_equation(state.lhs, state.rhs)}. Steps left: {left}."

    def action_text(self, state: AlgebraState, action: AlgebraAction) -> str:
        k = action.kind
        if k in ("expand", "move"):
            side = state.lhs if action.side == LEFT else state.rhs
            term = render_side((side[action.index],))
            if k == "expand":
                return f"expand {term} on the {action.side}"
            other = RIGHT if action.side == LEFT else LEFT
            return f"move {term} to the {other}"
        if k == "combine":
            return f"combine like terms on the {action.side}"
        if k == "multiply":
            return f"multiply both sides by {fmt_num(action.value)}"
        if k == "divide":
            return f"divide both sides by {fmt_num(action.value)}"
        return "swap the two sides"

    def state_key(self, state: AlgebraState):
        return (state.lhs, state.rhs, state.steps)

    def canonical_key(self, state: AlgebraState):
        """Order-insensitive key (term order does not change what is reachable)."""
        return (
            tuple(sorted(t.sort_key() for t in state.lhs)),
            tuple(sorted(t.sort_key() for t in state.rhs)),
        )
