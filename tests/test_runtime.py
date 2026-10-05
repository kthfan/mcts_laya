"""CPU budget: thread pools and (on Linux) core affinity follow `--cpus` / $MCTS_LAYA_CPUS."""

import json
import os
import subprocess
import sys

import pytest

from mcts_laya.runtime import available_cpus, resolve_cpus

PROBE = ("import json, os, mcts_laya.runtime as r, torch; n = r.limit_cpus({arg});"
         "print(json.dumps([n, torch.get_num_threads(), os.environ['OMP_NUM_THREADS'] if n else None,"
         " len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None]))")


def test_resolve(monkeypatch):
    monkeypatch.delenv("MCTS_LAYA_CPUS", raising=False)
    assert resolve_cpus() is None  # default: all cores
    assert resolve_cpus(1) == 1
    assert resolve_cpus(10_000) == available_cpus()
    assert resolve_cpus("all") is None and resolve_cpus(0) is None
    monkeypatch.setenv("MCTS_LAYA_CPUS", "1")
    assert resolve_cpus() == 1
    with pytest.raises(ValueError):
        resolve_cpus(-1)


def _probe(arg, env):
    out = subprocess.run([sys.executable, "-c", PROBE.format(arg=arg)], capture_output=True, text=True,
                         env=dict(os.environ, **env), check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_limit_applies_threads_and_affinity():
    n, torch_threads, omp, affinity = _probe(1, {})
    assert (n, torch_threads, omp) == (1, 1, "1")
    if affinity is not None:
        assert affinity == 1


def test_env_value_is_used_and_default_is_unlimited():
    assert _probe("None", {"MCTS_LAYA_CPUS": "1"})[0] == 1
    env = {k: v for k, v in os.environ.items() if k != "MCTS_LAYA_CPUS"}
    out = subprocess.run([sys.executable, "-c", PROBE.format(arg="None")], capture_output=True, text=True,
                         env=env, check=True)
    n, _, _, affinity = json.loads(out.stdout.strip().splitlines()[-1])
    assert n is None and affinity in (None, available_cpus())
