#!/usr/bin/env python
"""Run an ablation spec (levels x variants x seeds) one run after another, resumably.

A run whose output directory has `done.json` is skipped, so the same command can be re-issued after
an interruption. Each run's console output goes to <output_dir>/run.log.

`--jobs N` runs N experiments at the same time. One run is a single serial CPU stream (the
TextWorld engine and the search loop) that keeps a GPU mostly idle, so parallel runs sharing the GPU
give close to N times the throughput. Each job gets its own slice of the CPU cores.
"""

from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def plan(spec: dict, only_levels=None, only_variants=None, seeds=None, extra=(), root_override=None):
    for level, base in spec["levels"].items():
        if only_levels and level not in only_levels:
            continue
        for variant, overrides in spec["variants"].items():
            if only_variants and variant not in only_variants:
                continue
            for seed in seeds or spec.get("seeds", [0]):
                out = Path(root_override or spec["output_root"]) / level / f"{variant}-s{seed}"
                sets = list(spec.get("gpu_overrides", [])) + list(overrides or []) + list(extra) + [
                    f"seed={seed}", f"output_dir={out}", f"name={spec['name']}-{level}-{variant}-s{seed}"]
                yield level, variant, seed, base, out, sets


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("spec")
    p.add_argument("--only", nargs="*", default=None, help="level names to run")
    p.add_argument("--variants", nargs="*", default=None, help="variant names to run")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--extra", nargs="*", default=[], metavar="KEY=VALUE",
                   help="extra overrides for every run (e.g. a short smoke test: iterations=1 ...)")
    p.add_argument("--output-root", default=None, help="write runs here instead of the spec's output_root")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--cpus", default=None, metavar="N",
                   help="limit the CPU cores of the runs (default: $MCTS_LAYA_CPUS, else all cores; 0 = no limit)")
    p.add_argument("--jobs", type=int, default=1,
                   help="runs at the same time, sharing the GPU (each on its own slice of the CPU cores)")
    args = p.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    from mcts_laya.runtime import limit_cpus

    cpus = limit_cpus(args.cpus)  # inherited by every run (environment and, on Linux, core affinity)
    print(f"CPU limit: {cpus or 'none'}")
    spec = yaml.safe_load(open(args.spec))
    runs = list(plan(spec, args.only, args.variants, args.seeds, args.extra, args.output_root))
    print(f"{len(runs)} runs in {args.spec}")
    todo = []
    for i, (level, variant, seed, base, out, sets) in enumerate(runs, 1):
        cmd = [args.python, "-m", "mcts_laya", "-v", "run", base, "--set", *sets]
        tag = f"[{i}/{len(runs)}] {level} {variant} seed={seed}"
        if (ROOT / out / "done.json").exists():
            print(f"{tag}: done, skipping")
        elif args.dry_run:
            print(f"{tag}:\n  " + " ".join(shlex.quote(c) for c in cmd))
        else:
            todo.append((tag, out, cmd))
    return 1 if run_all(todo, max(1, args.jobs)) else 0


def core_slices(jobs: int):
    """Disjoint core sets, one per job (None where affinity is not supported or cores are too few)."""
    if jobs <= 1 or not hasattr(os, "sched_getaffinity"):
        return [None] * jobs
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) < jobs:
        return [None] * jobs
    k = len(cores) // jobs
    return [cores[j * k:(j + 1) * k] for j in range(jobs)]


def run_all(todo, jobs: int) -> int:
    """Run the commands, at most `jobs` at a time; returns the number of failed runs."""
    slices = core_slices(jobs)
    free = list(range(jobs))
    running = {}  # Popen -> (tag, t0, slot, log file)
    failures = 0
    queue = list(todo)
    try:
        while queue or running:
            while queue and free:
                tag, out, cmd = queue.pop(0)
                slot = free.pop(0)
                (ROOT / out).mkdir(parents=True, exist_ok=True)
                log = open(ROOT / out / "run.log", "w")
                env = dict(os.environ)
                cores = slices[slot]
                if cores is not None:
                    env["MCTS_LAYA_CPUS"] = str(len(cores))
                proc = subprocess.Popen(
                    cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                    preexec_fn=(lambda c=cores: os.sched_setaffinity(0, c)) if cores is not None else None)
                where = f" on cores {cores[0]}-{cores[-1]}" if cores is not None else ""
                print(f"{tag}: running -> {out}{where}", flush=True)
                running[proc] = (tag, time.time(), slot, log)
            time.sleep(2)
            for proc in [p for p in running if p.poll() is not None]:
                tag, t0, slot, log = running.pop(proc)
                log.close()
                free.append(slot)
                print(f"{tag}: exit {proc.returncode} after {(time.time() - t0) / 60:.1f} min", flush=True)
                failures += proc.returncode != 0
    except KeyboardInterrupt:
        for proc in running:
            proc.terminate()
        for proc in running:
            proc.wait()
        raise
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
