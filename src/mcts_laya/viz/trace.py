"""Turn episodes and search trees into JSON for the visualisation front end."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..envs.base import Environment
from ..search.tree import Node, SearchResult
from ..selfplay.actor import Episode


def _num(x: float, digits: int = 4) -> Optional[float]:
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, digits)


def tree_to_json(env: Environment, root: Node, max_nodes: int = 300, max_children: int = 8,
                 text_min_visits: int = 2, text_chars: int = 400) -> Dict[str, Any]:
    """Breadth-first serialisation of the visited part of a search tree, capped at `max_nodes`.

    Children are ordered by visits; at most `max_children` per node are kept (the rest are
    summarised as `hidden`). State text is attached to nodes with at least `text_min_visits`.
    """
    def node_json(n: Node, action: Optional[str]) -> Dict[str, Any]:
        d: Dict[str, Any] = {"a": action, "p": _num(n.prior), "n": int(round(n.visit_count)),
                             "q": _num(n.q) if n.visit_count else None, "children": []}
        if n.expanded:
            d["v"] = _num(n.nn_value)
        if n.terminal:
            d["t"] = True
            d["tv"] = _num(n.terminal_value)
            d["ok"] = bool(env.is_success(n.state))
        if n.visit_count >= text_min_visits or n.terminal:
            d["s"] = env.state_text(n.state)[:text_chars]
        return d

    out = node_json(root, None)
    count, queue = 1, [(root, out)]
    while queue:
        nxt = []
        for node, js in queue:
            kids = [(i, c) for i, c in enumerate(node.children) if c is not None and c.visit_count > 0]
            kids.sort(key=lambda ic: -ic[1].visit_count)
            js["hidden"] = max(0, len(kids) - max_children)
            for i, c in kids[:max_children]:
                if count >= max_nodes:
                    js["hidden"] += 1
                    continue
                cj = node_json(c, env.action_text(node.state, node.actions[i]))
                js["children"].append(cj)
                nxt.append((c, cj))
                count += 1
        queue = nxt
    out["size"] = count
    return out


def step_to_json(env: Environment, state: Any, result: SearchResult, chosen: int,
                 with_tree: bool = True, **tree_kw) -> Dict[str, Any]:
    visits = np.asarray(result.visit_counts, dtype=float)
    q_known = visits > 0
    actions = []
    for i, a in enumerate(result.actions):
        actions.append({
            "text": env.action_text(state, a),
            "prior": _num(result.priors[i]),
            "policy": _num(result.policy[i]),
            "visits": int(round(visits[i])),
            "q": _num(result.q_values[i]) if q_known[i] or result.extra.get("root") is None else None,
        })
    root = result.extra.get("root")
    d = {
        "render": env.render_data(state),
        "text": env.state_text(state),
        "actions": actions,
        "selected": int(result.selected),
        "chosen": int(chosen),
        "nn_value": _num(root.nn_value) if root is not None and root.expanded else None,
        "root_value": _num(result.root_value),
        "simulations": int(result.num_simulations),
    }
    if with_tree and root is not None and result.num_simulations > 0:
        d["tree"] = tree_to_json(env, root, **tree_kw)
    return d


def episode_to_json(env: Environment, ep: Episode, with_tree: bool = True, **tree_kw) -> Dict[str, Any]:
    final = ep.final_state
    return {
        "steps": [step_to_json(env, r.state, r.result, r.action_index, with_tree, **tree_kw) for r in ep.steps],
        "final": {"render": env.render_data(final), "text": env.state_text(final)},
        "outcome": {"success": bool(ep.success), "reward": _num((ep.final_value + 1.0) / 2.0),
                    "moves": ep.length},
    }


def curves_from_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Learning curves, baselines and training diagnostics from `metrics.jsonl` records."""
    points, baselines, train, selfplay = [], [], [], []
    order: List[str] = []
    for r in records:
        kind = r.get("kind")
        if kind == "eval":
            if r["stage"] == "baseline":
                baselines.append({k: r.get(k) for k in ("label", "success", "reward", "moves")})
                continue
            key = f"{r['iteration']}:{r['stage']}"
            if key not in order:
                order.append(key)
            points.append({"x": key, "label": r["label"], "success": r["success"], "reward": r["reward"],
                           "moves": r["moves"]})
        elif kind == "train":
            train.append({k: r.get(k) for k in ("iteration", "stage", "policy_ce", "value_ce",
                                                 "heldout_policy_top1", "heldout_value_brier",
                                                 "temperature_choice", "temperature_noul", "samples")})
        elif kind == "selfplay":
            selfplay.append({k: r.get(k) for k in ("iteration", "success", "moves", "samples", "seconds",
                                                    "rows_per_second")})
    return {"xs": order, "points": points, "baselines": baselines, "train": train, "selfplay": selfplay}
