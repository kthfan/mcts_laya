"""Many episodes in parallel: AlphaZero-style actor processes around one batched inference loop.

One episode at a time leaves a GPU idle: every search step waits on the environment (a TextWorld
expansion replays the game, ~50 ms on one core) and then sends the network a handful of rows. Here
`workers` processes each play whole episodes - environment, search tree, tokenisation - and the main
process, which owns the model, only gathers their leaf requests and runs them as one large batch:

    worker 1..N:  search -> leaves -> encode rows --("eval")-->  main: forward pass of all requests
                  <---------------- priors, values ------------  (waits <= max_wait_ms to batch)

Workers cache evaluations themselves, keyed by `state_key` and invalidated by the evaluator's
`version` (bumped on every model change). Baseline evaluators (uniform, rollout) run inside the
workers. Each episode carries its own seed, so results do not depend on which worker plays it or
when; only batch composition can change the network's float rounding.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import time
import traceback
from collections import OrderedDict, deque
from multiprocessing.connection import wait
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import progress
from ..evaluators.base import EvalResult, Evaluator
from .actor import Episode, SelfPlayConfig, play_episode

WORKER_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS")


def search_spec(name: str, params: Optional[Dict[str, Any]] = None, evaluator: str = "model",
                evaluator_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """How a worker builds a searcher; `evaluator="model"` means the shared Laya model."""
    return {"name": name, "params": dict(params or {}), "evaluator": evaluator,
            "evaluator_params": dict(evaluator_params or {})}


def episode_task(spec: Dict[str, Any], problem: Any, seed: int, config: SelfPlayConfig) -> Dict[str, Any]:
    return {"search": spec, "problem": problem, "seed": int(seed), "config": config}


# --- worker side ----------------------------------------------------------------------------
class RemoteEvaluator(Evaluator):
    """Encodes rows locally, asks the main process for the forward pass, caches the answers."""

    def __init__(self, conn, encoder: Dict[str, Any], cache_size: int, seed: int):
        self.conn = conn
        self.enc = encoder
        self.rng = random.Random(seed) if encoder["option_order"] == "random" else None
        self.cache_size = cache_size
        self.version: Optional[int] = None
        self._cache: "OrderedDict[Any, EvalResult]" = OrderedDict()

    def set_version(self, version: int) -> None:
        if version != self.version:
            self._cache.clear()
            self.version = version

    def evaluate(self, env, states, actions):
        from ..evaluators.laya_evaluator import encode_rows

        results: List[Optional[EvalResult]] = [None] * len(states)
        pending, rows = [], []
        for i, (s, a) in enumerate(zip(states, actions)):
            key = env.state_key(s)
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                results[i] = hit
                continue
            pending.append((i, key))
            rows.append(encode_rows(self.enc["tok"], env, s, a, self.enc["max_len"], self.enc["head_max_len"],
                                    self.rng))
        if rows:
            self.conn.send(("eval", rows))
            kind, answer = self.conn.recv()
            if kind != "eval_result":
                raise RuntimeError(f"expected eval_result, got {kind!r}")
            for (i, key), res in zip(pending, answer):
                results[i] = res
                self._cache[key] = res
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return results


def _strip(ep: Episode) -> Episode:
    for rec in ep.steps:  # search trees are large and only the visualiser reads them
        rec.result.extra = {k: v for k, v in rec.result.extra.items() if k != "root"}
    return ep


def _worker_main(conn, index: int, env_name: str, env_params: Dict[str, Any], encoder: Dict[str, Any],
                 cache_size: int) -> None:
    for var in WORKER_THREAD_VARS:  # one core's worth of work per worker
        os.environ[var] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    from ..registry import ENVIRONMENTS, EVALUATORS, SEARCHERS

    try:
        env = ENVIRONMENTS.build(env_name, **env_params)
        remote = RemoteEvaluator(conn, encoder, cache_size, seed=index)
    except Exception:
        conn.send(("error", None, traceback.format_exc()))
        return
    searchers: Dict[str, Any] = {}
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "stop":
            return
        _, task_id, version, task = msg
        try:
            remote.set_version(version)
            spec = task["search"]
            key = json.dumps(spec, sort_keys=True, default=str)
            if key not in searchers:
                ev = remote if spec["evaluator"] == "model" else EVALUATORS.build(
                    spec["evaluator"], **spec["evaluator_params"])
                searchers[key] = SEARCHERS.build(spec["name"], env, ev, **spec["params"])
            ep = play_episode(env, searchers[key], task["problem"], np.random.default_rng(task["seed"]),
                              task["config"])
            conn.send(("done", task_id, _strip(ep)))
        except Exception:
            conn.send(("error", task_id, traceback.format_exc()))


# --- main side ------------------------------------------------------------------------------
class ActorPool:
    """`workers` episode-playing processes sharing `evaluator` (a LayaEvaluator) for model calls."""

    def __init__(self, env_name: str, env_params: Dict[str, Any], evaluator, workers: int,
                 max_wait_ms: float = 2.0, cache_size: int = 50_000):
        self.evaluator = evaluator
        self.max_wait = max_wait_ms / 1000.0
        self.batches = 0
        self.rows = 0
        self.model_seconds = 0.0  # time the main process spent in model calls (vs waiting on actors)
        ctx = mp.get_context("spawn")  # the parent holds CUDA state; fork would not be safe
        encoder = evaluator.encoder_spec()
        self.conns, self.procs = [], []
        for i in range(workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_main, name=f"mcts-laya-actor-{i}",
                            args=(child, i, env_name, env_params, encoder, cache_size), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)

    @property
    def workers(self) -> int:
        return len(self.conns)

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self.conns, self.procs = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _died(self, conn) -> RuntimeError:
        i = self.conns.index(conn)
        self.procs[i].join(timeout=1)
        return RuntimeError(f"actor process {i} died (exit code {self.procs[i].exitcode}); see its traceback above")

    def _send(self, conn, msg) -> None:
        try:
            conn.send(msg)
        except (BrokenPipeError, ConnectionResetError, EOFError):
            raise self._died(conn) from None

    def _recv(self, conn):
        try:
            return conn.recv()
        except (EOFError, ConnectionResetError):
            raise self._died(conn) from None

    def run(self, tasks: Sequence[Dict[str, Any]], desc: str = "episodes") -> List[Episode]:
        """Play every task; returns the episodes in task order."""
        results: List[Optional[Episode]] = [None] * len(tasks)
        queue = deque(range(len(tasks)))
        idle = list(self.conns)
        busy: Dict[Any, int] = {}
        version = self.evaluator.version
        bar = progress.bar(len(tasks), desc, "episodes")
        n_success = n_moves = n_done = 0
        try:
            while queue or busy:
                while queue and idle:
                    c = idle.pop()
                    tid = queue.popleft()
                    self._send(c, ("task", tid, version, tasks[tid]))
                    busy[c] = tid
                requests: List[tuple] = []  # (conn, rows)
                ready = wait(list(busy))
                deadline = time.perf_counter() + self.max_wait
                while ready:
                    for c in ready:
                        msg = self._recv(c)
                        if msg[0] == "eval":
                            requests.append((c, msg[1]))
                        elif msg[0] == "done":
                            ep = msg[2]
                            results[msg[1]] = ep
                            del busy[c]
                            idle.append(c)
                            n_done += 1
                            n_success += bool(ep.success)
                            n_moves += ep.length
                            bar.update(1)
                            bar.set_postfix(success=n_success / n_done, moves=n_moves / n_done)
                        else:
                            raise RuntimeError(f"actor failed on task {msg[1]}:\n{msg[2]}")
                    asking = {c for c, _ in requests}
                    others = [c for c in busy if c not in asking]
                    left = deadline - time.perf_counter()
                    if not requests or not others or left <= 0:
                        break  # everyone still playing is waiting on us, or time is up
                    ready = wait(others, timeout=left)
                if requests:
                    flat = [row for _, rows in requests for row in rows]
                    t = time.perf_counter()
                    answers = self.evaluator.evaluate_encoded(flat)
                    self.model_seconds += time.perf_counter() - t
                    self.batches += 1
                    self.rows += 2 * len(flat)
                    start = 0
                    for c, rows in requests:
                        self._send(c, ("eval_result", answers[start:start + len(rows)]))
                        start += len(rows)
        finally:
            bar.close()
        return results  # type: ignore[return-value]


def resolve_workers(setting: Any, device: str, cpus: int) -> int:
    """`auto`: one actor per spare core when the model is on a GPU, none (serial) on the CPU."""
    if isinstance(setting, str):
        if setting != "auto":
            return int(setting)
        return max(0, min(cpus - 1, 32)) if not device.startswith("cpu") else 0
    return max(0, int(setting or 0))
