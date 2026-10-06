#!/usr/bin/env python
"""Summarise an ablation output root into one Markdown table (+ CSV).

Per run: warm-start and final evaluation of the main search (puct16 by default) and of greedy, the
best evaluation reached, how many self-play iterations the gate accepted, and self-play statistics.
Per variant: mean +- std over seeds of the final and best reward, the gain over the warm start, and
the paired difference to the reference variant (same seed, so seed-level noise cancels).

Only the last attempt in a `metrics.jsonl` is used (runs before the fresh-file fix appended a restarted
attempt to the old records), and runs without `done.json` are listed separately, not averaged.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def _sig(r):
    return r.get("kind"), r.get("stage"), r.get("label"), r.get("iteration")


def last_attempt(recs):
    """(records of the last run attempt, number of attempts): an attempt starts where the first record repeats."""
    if not recs:
        return recs, 0
    starts = [i for i, r in enumerate(recs) if _sig(r) == _sig(recs[0])]
    return recs[starts[-1]:], len(starts)


def load(run_dir: Path, label: str):
    all_recs = [json.loads(l) for l in open(run_dir / "metrics.jsonl") if l.strip()]
    recs, attempts = last_attempt(all_recs)
    # test-set evaluations only (the gate's validation evaluations carry split="val")
    ev = [r for r in recs if r.get("kind") == "eval" and r.get("stage") != "baseline" and r.get("split") != "val"]
    def at(stage, lab):
        xs = [r for r in ev if r["stage"] == stage and r["label"] == lab]
        return xs[-1] if xs else None
    warm = at("warmstart", label) or at("initial", label)
    sp = [r for r in ev if r["stage"] == "selfplay" and r["label"] == label]
    final = sp[-1] if sp else warm
    greedy_w, greedy_sp = at("warmstart", "greedy"), [r for r in ev if r["stage"] == "selfplay" and r["label"] == "greedy"]
    gates = [r for r in recs if r.get("kind") == "gate" and r["iteration"] > 0]
    sps = [r for r in recs if r.get("kind") == "selfplay"]
    base = {r["label"]: r for r in recs if r.get("kind") == "eval" and r.get("stage") == "baseline"}
    return {
        "warm_reward": warm and warm["reward"], "final_reward": final and final["reward"],
        "best_reward": max([r["reward"] for r in sp] + ([warm["reward"]] if warm else [])),
        "warm_success": warm and warm["success"], "final_success": final and final["success"],
        "final_moves": final and final["moves"],
        "greedy_warm": greedy_w and greedy_w["reward"], "greedy_final": greedy_sp[-1]["reward"] if greedy_sp else None,
        "gate_accepted": f"{sum(g['accepted'] for g in gates)}/{len(gates)}" if gates else "",
        "selfplay_success": st.mean(r["success"] for r in sps) if sps else None,
        "policy_episodes": st.mean(r.get("policy_episodes") or 0 for r in sps) if sps else None,
        "uniform_baseline": base.get("uniform-puct16", {}).get("reward"),
        "done": (run_dir / "done.json").exists(),
        "iterations": max([r["iteration"] for r in sps], default=0),
        "attempts": attempts,
    }


def fmt(x, d=3):
    return "" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("--label", default="puct16", help="eval label to compare (default puct16)")
    p.add_argument("--out", default=None, help="Markdown output (default <root>/summary.md)")
    p.add_argument("--reference", default="control", help="variant for the paired difference (default control)")
    p.add_argument("--include-incomplete", action="store_true", help="also average runs without done.json")
    args = p.parse_args(argv)
    root = Path(args.root)
    rows = []
    for mj in sorted(root.glob("*/*/metrics.jsonl")):
        run = mj.parent
        variant, _, seed = run.name.rpartition("-s")
        rows.append({"level": run.parent.name, "variant": variant, "seed": seed, **load(run, args.label)})
    if not rows:
        print(f"no runs under {root}")
        return 1
    cols = ["level", "variant", "seed", "done", "iterations", "attempts", "warm_reward", "final_reward", "best_reward", "warm_success",
            "final_success", "final_moves", "greedy_warm", "greedy_final", "gate_accepted", "selfplay_success",
            "policy_episodes", "uniform_baseline"]
    with open(root / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    used = [r for r in rows if r["done"] or args.include_incomplete]
    groups = defaultdict(list)
    for r in used:
        groups[(r["level"], r["variant"])].append(r)
    ref = {(r["level"], r["seed"]): r for r in used if r["variant"] == args.reference}
    lines = [f"# Ablation summary ({args.label})", "",
             f"| level | variant | seeds | final reward | best reward | gain vs warm start | "
             f"final − {args.reference} (paired) | greedy final | gate accepted |",
             "|---|---|---|---|---|---|---|---|---|"]
    def ms(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return ""
        return f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"
    for (level, variant), rs in sorted(groups.items()):
        gains = [r["final_reward"] - r["warm_reward"] for r in rs if r["final_reward"] is not None and r["warm_reward"] is not None]
        paired = [r["final_reward"] - ref[(level, r["seed"])]["final_reward"] for r in rs
                  if (level, r["seed"]) in ref and r["final_reward"] is not None
                  and ref[(level, r["seed"])]["final_reward"] is not None]
        lines.append(f"| {level} | {variant} | {len(rs)} | {ms([r['final_reward'] for r in rs])} | "
                     f"{ms([r['best_reward'] for r in rs])} | {ms(gains)} | "
                     f"{'' if variant == args.reference else ms(paired)} | {ms([r['greedy_final'] for r in rs])} | "
                     f"{', '.join(r['gate_accepted'] for r in rs)} |")
    skipped = [r for r in rows if r not in used]
    if skipped:
        lines += ["", "Not averaged (no `done.json`; rerun them with the same command, finished runs are skipped):",
                  ""] + [f"- {r['level']} / {r['variant']} seed {r['seed']}: {r['iterations']} self-play iterations logged"
                         for r in skipped]
    restarted = [r for r in used if r["attempts"] > 1]
    if restarted:
        lines += ["", "Restarted runs (only the last attempt is used): "
                  + ", ".join(f"{r['level']}/{r['variant']}-s{r['seed']} ({r['attempts']} attempts)" for r in restarted)]
    lines += ["", "Per-run details: `runs.csv`."]
    out = Path(args.out) if args.out else root / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
