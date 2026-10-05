#!/usr/bin/env python
"""Run an ablation spec (levels x variants x seeds) one run after another, resumably.

A run whose output directory has `done.json` is skipped, so the same command can be re-issued after
an interruption. Each run's console output goes to <output_dir>/run.log.
"""

from __future__ import annotations

import argparse
import itertools
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
    args = p.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    from mcts_laya.runtime import limit_cpus

    cpus = limit_cpus(args.cpus)  # inherited by every run (environment and, on Linux, core affinity)
    print(f"CPU limit: {cpus or 'none'}")
    spec = yaml.safe_load(open(args.spec))
    runs = list(plan(spec, args.only, args.variants, args.seeds, args.extra, args.output_root))
    print(f"{len(runs)} runs in {args.spec}")
    failures = 0
    for i, (level, variant, seed, base, out, sets) in enumerate(runs, 1):
        cmd = [args.python, "-m", "mcts_laya", "-v", "run", base, "--set", *sets]
        tag = f"[{i}/{len(runs)}] {level} {variant} seed={seed}"
        if (ROOT / out / "done.json").exists():
            print(f"{tag}: done, skipping")
            continue
        if args.dry_run:
            print(f"{tag}:\n  " + " ".join(shlex.quote(c) for c in cmd))
            continue
        (ROOT / out).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"{tag}: running -> {out}", flush=True)
        with open(ROOT / out / "run.log", "w") as log:
            code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"{tag}: exit {code} after {(time.time() - t0) / 60:.1f} min", flush=True)
        failures += code != 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
