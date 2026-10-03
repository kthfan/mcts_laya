"""Countdown numbers game.

Combine two of the available numbers with +, -, * or / (only exact, positive results) until the
target appears among the numbers. With `use_all=True` the target must be the last remaining
number (the "24 game" rule).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from ..registry import ENVIRONMENTS
from .base import SingleAgentEnvironment

Numbers = Tuple[int, ...]


@dataclass(frozen=True)
class CountdownState:
    numbers: Numbers  # sorted ascending
    target: int
    steps: int = 0
    history: Tuple[str, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class CountdownAction:
    a: int  # larger operand
    op: str
    b: int  # smaller operand
    result: int


def _apply(a: int, op: str, b: int) -> Optional[int]:
    """Result of `a op b` with a >= b, or None when the move is not allowed."""
    if op == "+":
        return a + b
    if op == "*":
        return a * b if b != 1 else None  # x * 1 wastes a number without changing anything
    if op == "-":
        return a - b if a > b else None
    if op == "/":
        return a // b if b > 1 and a % b == 0 else None
    raise ValueError(op)


OPS = ("+", "-", "*", "/")


def legal_moves(numbers: Numbers) -> List[CountdownAction]:
    moves, seen = [], set()
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            b, a = numbers[i], numbers[j]  # numbers is sorted, so a >= b
            for op in OPS:
                r = _apply(a, op, b)
                if r is None or (a, op, b) in seen:
                    continue
                seen.add((a, op, b))
                moves.append(CountdownAction(a, op, b, r))
    return moves


def apply_move(numbers: Numbers, move: CountdownAction) -> Numbers:
    rest = list(numbers)
    rest.remove(move.a)
    rest.remove(move.b)
    rest.append(move.result)
    return tuple(sorted(rest))


@lru_cache(maxsize=1_000_000)
def steps_to_solve(numbers: Numbers, target: int, use_all: bool) -> Optional[int]:
    """Minimum number of moves to reach the target, or None if it cannot be reached."""
    if use_all:
        if len(numbers) == 1:
            return 0 if numbers[0] == target else None
    else:
        if target in numbers:
            return 0
        if len(numbers) == 1:
            return None
    best = None
    for m in legal_moves(numbers):
        d = steps_to_solve(apply_move(numbers, m), target, use_all)
        if d is not None and (best is None or d + 1 < best):
            best = d + 1
    return best


@ENVIRONMENTS.register("countdown")
class CountdownEnv(SingleAgentEnvironment):
    name = "countdown"
    policy_instruction = "Which arithmetic step brings the numbers closer to the target?"
    value_instruction = "Can the target still be reached from these numbers?"
    value_criteria = {
        "false": "no, the target can no longer be reached",
        "true": "yes, the target can still be reached",
    }

    def __init__(
        self,
        n_numbers: int = 4,
        number_pool: Sequence[int] = tuple(range(1, 11)) + (25, 50, 75, 100),
        target_range: Tuple[int, int] = (10, 200),
        min_solution_steps: int = 2,
        use_all: bool = False,
        show_history: bool = False,
        action_detail: str = "outcome",
    ):
        if n_numbers < 2:
            raise ValueError("n_numbers must be at least 2")
        self.n_numbers = n_numbers
        self.number_pool = tuple(number_pool)
        self.target_range = tuple(target_range)
        self.min_solution_steps = min_solution_steps
        self.use_all = use_all
        self.show_history = show_history
        if action_detail not in ("move", "outcome"):
            raise ValueError("action_detail must be 'move' or 'outcome'")
        # "move": `50 * 2 = 100`; "outcome": `50 * 2 = 100 (gap 54)`, the distance from the
        # target to the closest number left, so the model compares outcomes instead of
        # redoing the arithmetic itself.
        self.action_detail = action_detail

    # --- problems -------------------------------------------------------------------------
    def sample_problem(self, rng: random.Random) -> CountdownState:
        lo, hi = self.target_range
        for _ in range(10_000):
            numbers = tuple(sorted(rng.choice(self.number_pool) for _ in range(self.n_numbers)))
            # Build a reachable target by playing random moves.
            cur, n_moves = numbers, 0
            depth = len(numbers) - 1 if self.use_all else rng.randint(self.min_solution_steps, len(numbers) - 1)
            for _ in range(depth):
                cur = apply_move(cur, rng.choice(legal_moves(cur)))
                n_moves += 1
            target = cur[-1] if self.use_all else rng.choice(cur)
            if not lo <= target <= hi or target in numbers:
                continue
            d = steps_to_solve(numbers, target, self.use_all)
            if d is not None and d >= self.min_solution_steps:
                return CountdownState(numbers, target)
        raise RuntimeError("could not sample a Countdown problem; loosen the generator settings")

    # --- dynamics -------------------------------------------------------------------------
    def legal_actions(self, state: CountdownState) -> List[CountdownAction]:
        return legal_moves(state.numbers)

    def step(self, state: CountdownState, action: CountdownAction) -> CountdownState:
        return CountdownState(
            apply_move(state.numbers, action),
            state.target,
            state.steps + 1,
            state.history + (self._move_text(action),),
        )

    def is_terminal(self, state: CountdownState) -> bool:
        if len(state.numbers) == 1:
            return True
        return (not self.use_all) and state.target in state.numbers

    def terminal_reward(self, state: CountdownState) -> float:
        if self.use_all:
            return 1.0 if state.numbers == (state.target,) else 0.0
        return 1.0 if state.target in state.numbers else 0.0

    # --- text -----------------------------------------------------------------------------
    @staticmethod
    def _move_text(m: CountdownAction) -> str:
        return f"{m.a} {m.op} {m.b} = {m.result}"

    def state_text(self, state: CountdownState) -> str:
        text = f"Target: {state.target}. Numbers: {', '.join(map(str, state.numbers))}."
        if self.show_history and state.history:
            text += " Done so far: " + "; ".join(state.history) + "."
        return text

    def action_text(self, state: CountdownState, action: CountdownAction) -> str:
        text = self._move_text(action)
        if self.action_detail == "outcome":
            left = apply_move(state.numbers, action)
            gap = min(abs(n - state.target) for n in left)
            text += f" (gap {gap})"
        return text

    def state_key(self, state: CountdownState):
        return (state.numbers, state.target)
