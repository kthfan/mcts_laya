#!/usr/bin/env python
"""Inference speed of greedy (no search) vs PUCT with Laya, on the same problems.

One agent plays one game at a time, as it would when deployed (no parallel actors), so the numbers
are per-decision latency. For each method it reports quality (success, reward, moves) next to time
per game and per move, and where the time goes:

* model     - Laya calls: tokenisation + forward pass (`LayaEvaluator.evaluate`)
* env       - environment steps (for TextWorld: replaying the game in the engine)
* search    - everything else inside the search (tree bookkeeping, selection)

Each method gets a fresh environment and the model's evaluation cache is cleared before every game,
so no method profits from work done by another. The first `--warmup` games of each method are
played but not timed (CUDA kernels, tokenizer, game loading).

    python scripts/bench_inference.py --run runs/ablation/selfplay_textworld_v2/L2-goal/control-s0
    python scripts/bench_inference.py --config configs/textworld/l2-goal.yaml --problems 20 --sims 16 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcts_laya import progress  # noqa: E402
from mcts_laya.evaluation import success_moves  # noqa: E402
from mcts_laya.registry import ENVIRONMENTS, SEARCHERS  # noqa: E402
from mcts_laya.runtime import limit_cpus  # noqa: E402
from mcts_laya.selfplay.actor import SelfPlayConfig, play_episode  # noqa: E402
from mcts_laya.viz.context import VizContext  # noqa: E402


class Timer:
    """Wraps a method of an object and accumulates its call count and time."""

    def __init__(self, obj: Any, name: str):
        self.calls, self.seconds = 0, 0.0
        self.samples: List[float] = []
        original = getattr(obj, name)

        def timed(*a, **kw):
            t = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                dt = time.perf_counter() - t
                self.calls += 1
                self.seconds += dt
                self.samples.append(dt)

        setattr(obj, name, timed)


def bench_method(ctx: VizContext, label: str, searcher_name: str, params: Dict[str, Any], problems: List[Any],
                 warmup: int) -> Dict[str, Any]:
    cfg = ctx.cfg
    env = ENVIRONMENTS.build(cfg.env.name, **cfg.env.params)  # fresh: no observations cached by other methods
    ev = ctx.evaluator
    searcher = SEARCHERS.build(searcher_name, env, ev, **params)
    sp = SelfPlayConfig(max_moves=cfg.eval.max_moves, add_noise=False, action_selection="search")
    for p in problems[:warmup]:
        ev.clear_cache()
        play_episode(env, searcher, p, np.random.default_rng(0), sp)

    model, step, move = Timer(ev, "evaluate"), Timer(env, "step"), Timer(searcher, "search")
    rows0 = ev.rows_evaluated
    episodes, game_seconds = [], []
    bar = progress.bar(len(problems) - warmup, f"bench {label}", "games")
    for i, p in enumerate(problems[warmup:]):
        ev.clear_cache()
        t = time.perf_counter()
        episodes.append(play_episode(env, searcher, p, np.random.default_rng(i), sp))
        game_seconds.append(time.perf_counter() - t)
        bar.update(1)
    bar.close()
    for obj, name in ((ev, "evaluate"), (env, "step"), (searcher, "search")):  # unwrap
        delattr(obj, name)

    total = sum(game_seconds)
    moves = sum(ep.length for ep in episodes)
    lat = np.array(move.samples) * 1000
    return {
        "method": label,
        "simulations": params.get("num_simulations", 0),
        "games": len(episodes),
        "success": float(np.mean([ep.success for ep in episodes])),
        "reward": float(np.mean([(ep.final_value + 1) / 2 for ep in episodes])),
        "moves": moves / len(episodes),  # all games (a failed one counts the moves it used)
        "success_moves": success_moves(episodes),  # successful games only
        "s_per_game": total / len(episodes),
        "ms_per_move": float(lat.mean()) if len(lat) else 0.0,
        "ms_per_move_p50": float(np.percentile(lat, 50)) if len(lat) else 0.0,
        "ms_per_move_p95": float(np.percentile(lat, 95)) if len(lat) else 0.0,
        "model_calls_per_move": model.calls / max(moves, 1),
        "rows_per_move": (ev.rows_evaluated - rows0) / max(moves, 1),
        "model_share": model.seconds / max(total, 1e-9),
        "env_share": step.seconds / max(total, 1e-9),
    }


def to_markdown(rows: List[Dict[str, Any]], header: str) -> str:
    base = next((r for r in rows if r["method"] == "greedy"), rows[0])
    cols = ["method", "success", "reward", "moves", "success moves", "s/game", "ms/move", "p50", "p95", "× greedy",
            "model calls/move", "rows/move", "model %", "env %"]
    lines = [header, "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join([
            r["method"], f"{r['success']:.2f}", f"{r['reward']:.3f}", f"{r['moves']:.1f}",
            "" if r["success_moves"] is None else f"{r['success_moves']:.1f}",
            f"{r['s_per_game']:.2f}", f"{r['ms_per_move']:.0f}", f"{r['ms_per_move_p50']:.0f}",
            f"{r['ms_per_move_p95']:.0f}", f"{r['ms_per_move'] / max(base['ms_per_move'], 1e-9):.1f}",
            f"{r['model_calls_per_move']:.1f}", f"{r['rows_per_move']:.1f}",
            f"{100 * r['model_share']:.0f}", f"{100 * r['env_share']:.0f}"]) + " |")
    lines += ["", "moves: all games (a failed game counts the moves it used); success moves: successful games only. "
              "model % / env %: share of the game time spent in Laya calls (tokenisation + forward) / in "
              "environment steps; the rest is search bookkeeping. × greedy: latency per move relative to greedy."]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run directory (its config.yaml; checkpoints/final if present)")
    src.add_argument("--config", help="experiment config (uses model.checkpoint unless --checkpoint)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides (with --config)")
    p.add_argument("--checkpoint", default=None, help="Laya checkpoint to load instead")
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:1 (default: the config's, else auto)")
    p.add_argument("--problems", type=int, default=30, help="test games per method (eval split)")
    p.add_argument("--sims", nargs="*", type=int, default=[16, 64], help="PUCT simulation budgets")
    p.add_argument("--batch-size", type=int, default=None, help="PUCT leaf batch (default: the config's)")
    p.add_argument("--warmup", type=int, default=2, help="untimed games per method before measuring")
    p.add_argument("--out", default=None, help="write the table here (.md) and the numbers next to it (.json)")
    p.add_argument("--cpus", default=None, metavar="N", help="limit CPU cores (default: all)")
    args = p.parse_args(argv)
    limit_cpus(args.cpus)

    if args.run:
        ctx = VizContext.from_run(args.run, checkpoint=args.checkpoint, device=args.device, with_teacher=False)
    else:
        ctx = VizContext.from_config(args.config, args.set, checkpoint=args.checkpoint, device=args.device,
                                     with_teacher=False)
    cfg = ctx.cfg
    puct_params = dict(cfg.search.params)
    for s in cfg.eval.searches:  # evaluation settings (e.g. batch size) of the config's PUCT search
        if s.name == "puct":
            puct_params.update(s.params)
            break
    puct_params.pop("num_simulations", None)
    if args.batch_size is not None:
        puct_params["batch_size"] = args.batch_size

    problems = ctx.problems(args.problems + args.warmup, split="eval")
    warm, timed = problems[args.problems:], problems[:args.problems]
    rows = [bench_method(ctx, "greedy", "greedy", {}, warm + timed, len(warm))]
    for n in args.sims:
        rows.append(bench_method(ctx, f"puct{n}", "puct", {**puct_params, "num_simulations": n}, warm + timed,
                                 len(warm)))
    device = str(next(ctx.agent.model.parameters()).device)
    header = (f"# Inference speed: greedy vs PUCT\n\n- env: `{cfg.env.name}` {cfg.env.params.get('level', '')}\n"
              f"- checkpoint: `{ctx.checkpoint}`\n- device: `{device}`\n- games per method: {args.problems} "
              f"(eval split), warm-up {args.warmup}\n- PUCT: {puct_params}")
    md = to_markdown(rows, header)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n")
        out.with_suffix(".json").write_text(json.dumps({"device": device, "checkpoint": ctx.checkpoint,
                                                        "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
