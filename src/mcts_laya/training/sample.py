"""Model-agnostic training sample: text in, soft targets out (JSONL-serialisable)."""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, Iterable, List, Optional


@dataclass
class Sample:
    state_text: str
    action_texts: List[str]
    policy: List[float]  # target distribution over action_texts
    value: float  # target in [-1, 1], perspective of the player to move
    policy_instruction: str
    value_instruction: str
    value_criteria: Dict[str, str]
    meta: Dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "Sample":
        return cls(**json.loads(line))


def save_samples(path: str, samples: Iterable[Sample]) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(s.to_json() + "\n")


def load_samples(path: str) -> List[Sample]:
    with open(path) as f:
        return [Sample.from_json(line) for line in f if line.strip()]


class ReplayBuffer:
    """FIFO buffer of samples (AlphaZero keeps the most recent self-play positions)."""

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self._data: Deque[Sample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, samples: Iterable[Sample]) -> None:
        self._data.extend(samples)

    def sample(self, n: Optional[int], rng: random.Random) -> List[Sample]:
        data = list(self._data)
        if n is None or n >= len(data):
            rng.shuffle(data)
            return data
        return rng.sample(data, n)

    def save(self, path: str) -> None:
        save_samples(path, self._data)
