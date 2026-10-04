"""Static HTML report: `mcts-laya viz <run_dir>` -> one self-contained file."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .context import VizContext

STATIC = Path(__file__).parent / "static"


def render_html(data: Dict[str, Any], mode: str = "static") -> str:
    """Inline the front end and the data into one HTML document."""
    html = (STATIC / "index.html").read_text()
    css = (STATIC / "app.css").read_text()
    js = (STATIC / "app.js").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (html.replace("/*__CSS__*/", css)
                .replace("/*__JS__*/", js)
                .replace("\"__MODE__\"", json.dumps(mode))
                .replace("null/*__DATA__*/", payload))


def build_report(ctx: VizContext, n_problems: int = 4, methods: Optional[List[str]] = None,
                 tree_methods: Optional[List[str]] = None, max_tree_nodes: int = 250,
                 log=print) -> Dict[str, Any]:
    methods = methods or [m.label for m in ctx.methods]
    if tree_methods is None:  # trees only where the search used Laya
        tree_methods = [m.label for m in ctx.methods if m.uses_model and m.label in methods]
    problems = ctx.problems(n_problems)
    data: Dict[str, Any] = {"meta": ctx.meta(), "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                            "problems": [], "traces": {}, "curves": ctx.curves()}
    data["meta"]["shown_methods"] = methods
    for i, p in enumerate(problems):
        pid = f"p{i}"
        data["problems"].append({"id": pid, "render": ctx.env.render_data(p), "text": ctx.env.state_text(p)})
        data["traces"][pid] = {}
        for label in methods:
            t0 = time.time()
            data["traces"][pid][label] = ctx.play(p, label, with_tree=label in tree_methods,
                                                  max_nodes=max_tree_nodes)
            o = data["traces"][pid][label]["outcome"]
            log(f"[viz] problem {i + 1}/{len(problems)} {label}: success={o['success']} moves={o['moves']} "
                f"({time.time() - t0:.1f}s)")
    return data


def read_report_data(path: str) -> Dict[str, Any]:
    """The data embedded in a report written by `write_report`."""
    html = Path(path).read_text()
    start = html.index("window.VIZ_DATA = ") + len("window.VIZ_DATA = ")
    end = html.index(";\n</script>", start)
    return json.loads(html[start:end].replace("<\\/", "</"))


def rerender(path: str, out: Optional[str] = None) -> str:
    """Re-wrap an existing report's data in the current front end (no search is re-run)."""
    return write_report(out or path, read_report_data(path))


def write_report(path: str, data: Dict[str, Any]) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_html(data, "static"))
    return path
