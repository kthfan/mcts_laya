"""Laya as the policy/value network of the search.

`Agent.predict_batch` requires every state in a batch to share one question set, but every leaf
of a search tree has its own candidate actions, so this evaluator builds the rows itself (with
Laya's own `build_sequence`) and runs them through the agent's forward pass in shared batches.

The work is split so the parallel actors (`mcts_laya.selfplay.parallel`) can encode rows in their
own processes: `encode_rows` (tokenisation, CPU) -> `evaluate_encoded` (forward pass and
temperatures, where the model lives).
"""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
from laya.common import QTYPES, collate_items, temp_bucket

from ..envs.base import Environment
from ..laya_io import encode_question, policy_question, value_question
from ..registry import EVALUATORS
from .base import EvalResult, Evaluator


def load_agent(checkpoint: str, device: Optional[str] = None, subfolder: Optional[str] = None,
               revision: Optional[str] = None, max_len: Optional[int] = None,
               head_max_len: Optional[int] = None):
    import laya

    agent = laya.load(checkpoint, device=device, subfolder=subfolder, revision=revision)
    if max_len is not None:
        agent.cfg["max_len"] = int(max_len)
    if head_max_len is not None:
        agent.cfg["head_max_len"] = int(head_max_len)
    return agent


def encode_rows(tok, env: Environment, state, actions, max_len: int, head_max_len: int,
                rng: Optional[random.Random] = None) -> tuple:
    """The policy and value rows of one state: (policy item, value item, option order or None, #actions).

    With `rng`, the options are shown in a random order (`order[slot] = action index`).
    """
    text = env.state_text(state)
    texts = [env.action_text(state, a) for a in actions]
    order = None
    if rng is not None:
        order = list(range(len(texts)))
        rng.shuffle(order)
    pq = policy_question(env.policy_instruction, texts, order)
    vq = value_question(env.value_instruction, env.value_criteria)
    return (
        encode_question(tok, text, pq, max_len, head_max_len),
        encode_question(tok, text, vq, max_len, head_max_len),
        order,
        len(texts),
    )


@EVALUATORS.register("laya")
class LayaEvaluator(Evaluator):
    def __init__(
        self,
        agent,
        max_rows: int = 64,
        cache_size: int = 200_000,
        prior_temperature: float = 1.0,
        option_order: str = "none",  # "none" | "random"
        seed: int = 0,
    ):
        self.agent = agent
        self.max_rows = max_rows
        self.cache_size = cache_size
        self.prior_temperature = prior_temperature
        self.option_order = option_order
        self.rng = random.Random(seed)
        self._cache: "OrderedDict[Any, EvalResult]" = OrderedDict()
        self.rows_evaluated = 0
        self.cache_hits = 0
        self.version = 0  # bumped whenever the model changes (clear_cache); remote caches follow it

    @classmethod
    def from_checkpoint(cls, checkpoint: str, device: Optional[str] = None, subfolder: Optional[str] = None,
                        revision: Optional[str] = None, max_len: Optional[int] = None,
                        head_max_len: Optional[int] = None, **kwargs) -> "LayaEvaluator":
        agent = load_agent(checkpoint, device, subfolder, revision, max_len, head_max_len)
        return cls(agent, **kwargs)

    @property
    def max_len(self) -> int:
        return int(self.agent.cfg.get("max_len", 512))

    @property
    def head_max_len(self) -> int:
        return int(self.agent.cfg.get("head_max_len", 192))

    def clear_cache(self) -> None:
        self._cache.clear()
        self.version += 1

    def encoder_spec(self) -> dict:
        """What a remote process needs to call `encode_rows` exactly like this evaluator."""
        return {"tok": self.agent.tok, "max_len": self.max_len, "head_max_len": self.head_max_len,
                "option_order": self.option_order}

    def _temperature(self, qtype: int, k: int) -> float:
        a = self.agent
        return float(a.temperature_by_options.get(temp_bucket(qtype, k), a.temperature[qtype]))

    def _rows(self, env: Environment, state, actions) -> tuple:
        rng = self.rng if self.option_order == "random" else None
        return encode_rows(self.agent.tok, env, state, actions, self.max_len, self.head_max_len, rng)

    @torch.no_grad()
    def _forward(self, items: List[dict]) -> List[np.ndarray]:
        # Rows are grouped by length so short value rows are not padded to the long policy rows
        # they would otherwise share a batch with; results come back in the original order.
        order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
        out: List[Optional[np.ndarray]] = [None] * len(items)
        for start in range(0, len(order), self.max_rows):
            idx = order[start:start + self.max_rows]
            batch = collate_items([[items[i] for i in idx]], self.agent.tok.pad_token_id)
            logits, _ = self.agent._infer(batch)
            for i, row in zip(idx, logits.float().cpu().numpy()):
                out[i] = row
            self.rows_evaluated += len(idx)
        return out  # type: ignore[return-value]

    def evaluate(self, env: Environment, states: Sequence[Any], actions: Sequence[List[Any]]) -> List[EvalResult]:
        results: List[Optional[EvalResult]] = [None] * len(states)
        pending, rows = [], []
        for i, (s, a) in enumerate(zip(states, actions)):
            key = (env.name, env.state_key(s))
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                self.cache_hits += 1
                results[i] = hit
                continue
            pending.append((i, key))
            rows.append(self._rows(env, s, a))
        for (i, key), res in zip(pending, self.evaluate_encoded(rows)):
            results[i] = res
            self._cache[key] = res
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return results  # type: ignore[return-value]

    def evaluate_encoded(self, rows: Sequence[tuple]) -> List[EvalResult]:
        """Forward pass for rows from `encode_rows` (no cache)."""
        if not rows:
            return []
        logits = self._forward([it for p_item, v_item, _, _ in rows for it in (p_item, v_item)])
        out = []
        for j, (_, _, order, k) in enumerate(rows):
            pl, vl = logits[2 * j][:k], logits[2 * j + 1][:2]
            t_p = self._temperature(QTYPES["choice"], k) * self.prior_temperature
            z = pl / t_p
            p = np.exp(z - z.max())
            p /= p.sum()
            if order is not None:  # slot order -> action order
                unperm = np.empty_like(p)
                unperm[np.asarray(order)] = p
                p = unperm
            zv = vl / self._temperature(QTYPES["noul"], 2)
            pv = np.exp(zv - zv.max())
            p_true = float(pv[1] / pv.sum())
            out.append(EvalResult(p, 2.0 * p_true - 1.0))
        return out
