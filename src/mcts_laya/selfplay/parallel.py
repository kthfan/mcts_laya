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
import logging
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

log = logging.getLogger(__name__)

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
        self.wait_seconds = 0.0  # time blocked on the main process's model calls

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
            t = time.perf_counter()
            self.conn.send(("eval", rows))
            kind, answer = self.conn.recv()
            self.wait_seconds += time.perf_counter() - t
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
                 cache_size: int, memory_gb: float = 0.0) -> None:
    for var in WORKER_THREAD_VARS:  # one core's worth of work per worker
        os.environ[var] = "1"
    if memory_gb > 0:
        try:  # a runaway inside an environment raises MemoryError here instead of exhausting the host
            import resource

            cap = int(memory_gb * 2 ** 30)
            hard = resource.getrlimit(resource.RLIMIT_AS)[1]
            resource.setrlimit(resource.RLIMIT_AS, (cap if hard < 0 else min(cap, hard), hard))
        except (ImportError, ValueError, OSError):
            pass
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
            t, w = time.perf_counter(), remote.wait_seconds
            ep = play_episode(env, searchers[key], task["problem"], np.random.default_rng(task["seed"]),
                              task["config"])
            stats = {"seconds": time.perf_counter() - t, "wait": remote.wait_seconds - w}
            conn.send(("done", task_id, _strip(ep), stats))
        except Exception:
            conn.send(("error", task_id, traceback.format_exc()))


# --- main side ------------------------------------------------------------------------------
class ActorPool:
    """`workers` episode-playing processes sharing `evaluator` (a LayaEvaluator) for model calls.

    An episode that raises in its actor (e.g. MemoryError under the `memory_gb` cap), whose actor dies,
    or that runs longer than `episode_timeout_s` is recorded as failed (`None` in the results) and the
    actor is replaced; the run goes on. Errors in setting an actor up still abort.
    """

    def __init__(self, env_name: str, env_params: Dict[str, Any], evaluator, workers: int,
                 max_wait_ms: float = 2.0, cache_size: int = 50_000, episode_timeout_s: float = 1800.0,
                 memory_gb: float = 16.0):
        self.evaluator = evaluator
        self.max_wait = max_wait_ms / 1000.0
        self.episode_timeout = episode_timeout_s
        self.batches = 0
        self.rows = 0
        self.model_seconds = 0.0  # time the main process spent in model calls (vs waiting on actors)
        self.actor_seconds = 0.0  # summed episode time over actors
        self.actor_wait = 0.0  # ... of which blocked waiting for model results
        self.failures: List[str] = []  # one line per failed episode, over the pool's lifetime
        self._ctx = mp.get_context("spawn")  # the parent holds CUDA state; fork would not be safe
        self._args = (env_name, env_params, evaluator.encoder_spec(), cache_size, memory_gb)
        self.conns: List[Any] = [None] * workers
        self.procs: List[Any] = [None] * workers
        for i in range(workers):
            self._start(i)

    def _start(self, i: int) -> None:
        env_name, env_params, encoder, cache_size, memory_gb = self._args
        parent, child = self._ctx.Pipe()
        p = self._ctx.Process(target=_worker_main, name=f"mcts-laya-actor-{i}",
                              args=(child, i, env_name, env_params, encoder, cache_size, memory_gb), daemon=True)
        p.start()
        child.close()
        self.conns[i], self.procs[i] = parent, p

    def _restart(self, conn) -> Any:
        """Replace the actor behind `conn`; returns the new connection."""
        i = self.conns.index(conn)
        p = self.procs[i]
        if p.is_alive():
            p.terminate()
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join(timeout=5)
        conn.close()
        self._start(i)
        return self.conns[i]

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

    def snapshot(self) -> Dict[str, float]:
        return {"batches": self.batches, "rows": self.rows, "model": self.model_seconds,
                "actor": self.actor_seconds, "wait": self.actor_wait, "failed": len(self.failures)}

    def stats_since(self, before: Dict[str, float], seconds: float) -> Dict[str, Any]:
        """Throughput diagnostics for the log, relative to an earlier `snapshot()`."""
        d = {k: v - before[k] for k, v in self.snapshot().items()}
        return {
            "workers": self.workers,
            "rows_per_batch": d["rows"] / max(d["batches"], 1),
            # share of the wall time the main process was in model calls; near 1 = model-bound
            "model_busy": d["model"] / max(seconds, 1e-9),
            "model_ms_per_batch": 1000 * d["model"] / max(d["batches"], 1),
            # share of the actors' time spent waiting for model results (the rest: env + search)
            "actor_wait": d["wait"] / max(d["actor"], 1e-9),
            "failed_episodes": int(d["failed"]),
        }

    def run(self, tasks: Sequence[Dict[str, Any]], desc: str = "episodes") -> List[Optional[Episode]]:
        """Play every task; returns the episodes in task order (`None` for a failed episode)."""
        results: List[Optional[Episode]] = [None] * len(tasks)
        queue = deque(range(len(tasks)))
        idle = list(self.conns)
        busy: Dict[Any, tuple] = {}  # conn -> (task id, start time)
        version = self.evaluator.version
        bar = progress.bar(len(tasks), desc, "episodes")
        n_success = n_moves = n_done = 0

        def finish(c, reason: Optional[str] = None) -> None:
            """Free `c` after its episode ended; with `reason`, the episode failed and the actor is replaced."""
            nonlocal n_done
            tid, _ = busy.pop(c)
            if reason is not None:
                line = f"{desc}: task {tid} failed: {reason}"
                self.failures.append(line)
                log.warning("%s; replacing the actor and going on", line)
                c = self._restart(c)
                n_done += 1
                bar.update(1)
            idle.append(c)

        try:
            while queue or busy:
                while queue and idle:
                    c = idle.pop()
                    tid = queue.popleft()
                    try:
                        c.send(("task", tid, version, tasks[tid]))
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        queue.appendleft(tid)  # never started: retry it on a fresh actor
                        idle.append(self._restart(c))
                        continue
                    busy[c] = (tid, time.time())
                requests: List[tuple] = []  # (conn, rows)
                ready = wait(list(busy), timeout=5.0)
                if not ready:
                    bar.heartbeat()  # no actor has asked or finished for 5 s: keep the log alive
                deadline = time.perf_counter() + self.max_wait
                while ready:
                    for c in ready:
                        try:
                            msg = c.recv()
                        except (EOFError, ConnectionResetError, OSError):
                            i = self.conns.index(c)
                            self.procs[i].join(timeout=1)
                            finish(c, f"actor process died (exit code {self.procs[i].exitcode})")
                            continue
                        if msg[0] == "eval":
                            requests.append((c, msg[1]))
                        elif msg[0] == "done":
                            ep = msg[2]
                            results[msg[1]] = ep
                            self.actor_seconds += msg[3]["seconds"]
                            self.actor_wait += msg[3]["wait"]
                            finish(c)
                            n_done += 1
                            n_success += bool(ep.success)
                            n_moves += ep.length
                            bar.update(1)
                            bar.set_postfix(success=n_success / n_done, moves=n_moves / n_done)
                        elif msg[1] is None:  # the actor could not even be set up
                            raise RuntimeError(f"actor failed to start:\n{msg[2]}")
                        else:
                            last = msg[2].strip().splitlines()[-1] if msg[2].strip() else "error"
                            log.debug("actor traceback:\n%s", msg[2])
                            finish(c, last)
                    asking = {c for c, _ in requests}
                    others = [c for c in busy if c not in asking]
                    left = deadline - time.perf_counter()
                    if not requests or not others or left <= 0:
                        break  # everyone still playing is waiting on us, or time is up
                    ready = wait(others, timeout=left)
                now = time.time()
                for c, (tid, t0) in list(busy.items()):
                    if now - t0 > self.episode_timeout:
                        requests = [(r, rows) for r, rows in requests if r is not c]  # drop its pending request
                        finish(c, f"timed out after {now - t0:.0f} s")
                if requests:
                    flat = [row for _, rows in requests for row in rows]
                    t = time.perf_counter()
                    answers = self.evaluator.evaluate_encoded(flat)
                    self.model_seconds += time.perf_counter() - t
                    self.batches += 1
                    self.rows += 2 * len(flat)
                    start = 0
                    for c, rows in requests:
                        try:
                            c.send(("eval_result", answers[start:start + len(rows)]))
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            finish(c, "actor process died")
                        start += len(rows)
        finally:
            bar.close()
        return results


def resolve_workers(setting: Any, device: str, cpus: int) -> int:
    """`auto`: one actor per spare core when the model is on a GPU, none (serial) on the CPU."""
    if isinstance(setting, str):
        if setting != "auto":
            return int(setting)
        return max(0, min(cpus - 1, 32)) if not device.startswith("cpu") else 0
    return max(0, int(setting or 0))
