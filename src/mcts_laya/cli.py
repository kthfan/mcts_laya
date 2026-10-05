"""Command line: `mcts-laya <command>` (or `python -m mcts_laya <command>`)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time


def _cmd_download(args) -> None:
    from .models.download import download_checkpoint

    for name in args.checkpoint:
        print(download_checkpoint(name, args.out, revision=args.revision, force=args.force))


def _cmd_make_tiny(args) -> None:
    from .models.tiny import build_tiny_checkpoint

    print(build_tiny_checkpoint(args.out, env_names=args.envs, hidden_size=args.hidden, num_layers=args.layers,
                                max_len=args.max_len, head_max_len=args.head_max_len))


def _cmd_tw_games(args) -> None:
    from .envs.textworld_games import LEVELS, generate_pool

    for level in args.level:
        d = generate_pool(args.out, level, args.train, args.eval, workers=args.workers, seed=args.seed)
        import json

        m = json.loads((d / "manifest.json").read_text())
        print(f"{level}: {len(m['games'].get('train', []))} train / {len(m['games'].get('eval', []))} eval -> {d}")


def _viz_context(args):
    from .viz.context import VizContext

    if args.run:
        return VizContext.from_run(args.run, checkpoint=args.checkpoint, device=args.device)
    return VizContext.from_config(args.config, args.set, checkpoint=args.checkpoint, device=args.device)


def _cmd_viz(args) -> None:
    from .viz.report import build_report, rerender, write_report

    if args.rerender:
        for path in args.rerender:
            print(rerender(path))
        return
    ctx = _viz_context(args)
    data = build_report(ctx, n_problems=args.problems, methods=args.methods or None, max_tree_nodes=args.max_tree_nodes)
    out = args.out or (str(ctx.run_dir / "viz.html") if ctx.run_dir else "viz.html")
    print(write_report(out, data))


def _cmd_serve(args) -> None:
    from .viz.server import serve

    serve(_viz_context(args), host=args.host, port=args.port)


def _cmd_run(args) -> None:
    from .config import load_config
    from .pipeline import AlphaZeroLoop

    cfg = load_config(args.config, args.set)
    loop = AlphaZeroLoop(cfg)
    loop.run()
    print((loop.out / "summary.md").read_text())


def _cmd_bench(args) -> None:
    """Rows/second of the batched Laya evaluator on random positions of an environment."""
    import numpy as np

    from .evaluators.laya_evaluator import LayaEvaluator
    from .registry import ENVIRONMENTS

    env = ENVIRONMENTS.build(args.env)
    ev = LayaEvaluator.from_checkpoint(args.checkpoint, device=args.device, max_rows=args.batch,
                                       max_len=args.max_len, head_max_len=args.head_max_len, cache_size=0)
    rng = random.Random(0)
    states = [env.sample_problem(rng) for _ in range(args.positions)]
    actions = [env.legal_actions(s) for s in states]
    ev.evaluate(env, states[:2], actions[:2])  # warm-up
    t0, rows0 = time.time(), ev.rows_evaluated
    ev.evaluate(env, states, actions)
    dt = time.time() - t0
    rows = ev.rows_evaluated - rows0
    print(json.dumps({"env": args.env, "positions": len(states), "rows": rows, "seconds": round(dt, 3),
                      "rows_per_second": round(rows / dt, 1),
                      "mean_actions": float(np.mean([len(a) for a in actions]))}))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="mcts-laya")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download Laya checkpoints from the Hugging Face Hub")
    d.add_argument("--checkpoint", nargs="+", default=["multilingual"],
                   choices=["english", "multilingual", "typed-decisions"])
    d.add_argument("--out", default="models/laya", help="each checkpoint goes to <out>/<name>")
    d.add_argument("--revision", default=None)
    d.add_argument("--force", action="store_true", help="download again even if <out>/<name> exists")
    d.set_defaults(func=_cmd_download)

    t = sub.add_parser("make-tiny", help="build a tiny random Laya-compatible checkpoint (tests, CPU dev)")
    t.add_argument("--out", default="models/tiny")
    t.add_argument("--envs", nargs="+", default=["countdown", "algebra"])
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--max-len", type=int, default=512)
    t.add_argument("--head-max-len", type=int, default=384)
    t.set_defaults(func=_cmd_make_tiny)

    g = sub.add_parser("tw-games", help="generate TextWorld game pools for curriculum levels")
    g.add_argument("--level", nargs="+", default=["L1"], help="level names, e.g. L1 L2 L2-goal L3-goal")
    g.add_argument("--out", default="data/textworld")
    g.add_argument("--train", type=int, default=200)
    g.add_argument("--eval", type=int, default=50)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=_cmd_tw_games)

    r = sub.add_parser("run", help="run an AlphaZero experiment from a YAML config")
    r.add_argument("config")
    r.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config values, e.g. --set iterations=2 search.params.num_simulations=8")
    r.set_defaults(func=_cmd_run)

    def viz_source(p, rerender=False):
        src = p.add_mutually_exclusive_group(required=True)
        if rerender:
            src.add_argument("--rerender", nargs="+", metavar="HTML",
                             help="re-wrap existing reports in the current front end (no search re-run)")
        src.add_argument("--run", help="experiment output dir (uses its config, metrics and final checkpoint)")
        src.add_argument("--config", help="experiment YAML (no learning curves)")
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
        p.add_argument("--checkpoint", default=None, help="override the checkpoint to load")
        p.add_argument("--device", default=None)

    v = sub.add_parser("viz", help="static HTML report: replays, search trees, comparisons, learning curves")
    viz_source(v, rerender=True)
    v.add_argument("--problems", type=int, default=4)
    v.add_argument("--methods", nargs="*", default=[], help="method labels (default: all eval searches, baselines, teacher)")
    v.add_argument("--max-tree-nodes", type=int, default=250)
    v.add_argument("--out", default=None, help="output .html (default: <run>/viz.html)")
    v.set_defaults(func=_cmd_viz)

    sv = sub.add_parser("serve", help="live visualiser in the browser (search step by step, play yourself)")
    viz_source(sv)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.set_defaults(func=_cmd_serve)

    b = sub.add_parser("bench", help="measure evaluator throughput")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--env", default="countdown")
    b.add_argument("--device", default=None)
    b.add_argument("--batch", type=int, default=64)
    b.add_argument("--positions", type=int, default=64)
    b.add_argument("--max-len", type=int, default=None)
    b.add_argument("--head-max-len", type=int, default=None)
    b.set_defaults(func=_cmd_bench)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    args.func(args)


if __name__ == "__main__":
    main()
