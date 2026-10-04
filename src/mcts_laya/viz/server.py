"""Live mode: `mcts-laya serve` - pick a problem, search step by step, or play moves yourself.

Standard-library HTTP server, bound to localhost by default. The model and the environments are
not thread-safe, so every API call runs under one lock.
"""

from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

import numpy as np

from .context import VizContext
from .report import render_html
from .trace import step_to_json


class LiveSession:
    def __init__(self, problem: Any):
        self.problem = problem
        self.state = problem
        self.history = []


class LiveApp:
    def __init__(self, ctx: VizContext, max_sessions: int = 32):
        self.ctx = ctx
        self.lock = threading.Lock()
        self.sessions: Dict[str, LiveSession] = {}
        self.max_sessions = max_sessions
        self.rng = np.random.default_rng(0)

    def _state_json(self, s: LiveSession) -> Dict[str, Any]:
        env, st = self.ctx.env, s.state
        done = env.is_terminal(st)
        out = {"render": env.render_data(st), "text": env.state_text(st), "done": done,
               "actions": [] if done else [env.action_text(st, a) for a in env.legal_actions(st)]}
        if done:
            out["outcome"] = {"success": bool(env.is_success(st)),
                              "reward": round((env.terminal_value(st) + 1) / 2, 4), "moves": len(s.history)}
        return out

    def _session(self, body) -> LiveSession:
        s = self.sessions.get(body.get("session"))
        if s is None:
            raise KeyError("unknown or expired session; load a problem again")
        return s

    # --- endpoints --------------------------------------------------------------------------
    def state(self, _body):
        return {"meta": self.ctx.meta(), "curves": self.ctx.curves()}

    def new(self, body):
        split, index = body.get("split", "eval"), int(body.get("index", 0))
        problems = self.ctx.problems(index + 1, split=split)
        sid = uuid.uuid4().hex[:12]
        if len(self.sessions) >= self.max_sessions:
            self.sessions.pop(next(iter(self.sessions)))
        self.sessions[sid] = LiveSession(problems[index])
        return {"session": sid, "state": self._state_json(self.sessions[sid])}

    def search(self, body):
        s = self._session(body)
        if self.ctx.env.is_terminal(s.state):
            raise ValueError("the episode is over")
        overrides = {}
        if body.get("simulations") is not None and self.ctx.method(body["method"]).searcher in ("puct", "gumbel"):
            overrides["num_simulations"] = int(body["simulations"])
        result = self.ctx.searcher(body["method"], **overrides).search(s.state, self.rng, add_noise=False)
        return {"step": step_to_json(self.ctx.env, s.state, result, result.selected, with_tree=True)}

    def act(self, body):
        s = self._session(body)
        env = self.ctx.env
        actions = env.legal_actions(s.state)
        i = int(body["index"])
        if not 0 <= i < len(actions):
            raise ValueError(f"action index {i} out of range")
        s.state = env.step(s.state, actions[i])
        s.history.append(i)
        return {"state": self._state_json(s)}

    def compare(self, body):
        s = self._session(body)
        traces = {m.label: self.ctx.play(s.problem, m.label, with_tree=m.uses_model, max_nodes=200)
                  for m in self.ctx.methods}
        return {"problem": {"render": self.ctx.env.render_data(s.problem), "text": self.ctx.env.state_text(s.problem)},
                "traces": traces}

    ROUTES = {"/api/state": "state", "/api/new": "new", "/api/search": "search", "/api/act": "act",
              "/api/compare": "compare"}

    def handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # quieter console
                pass

            def _send(self, code: int, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, obj):
                self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

            def _dispatch(self, body):
                name = app.ROUTES.get(self.path.split("?")[0])
                if name is None:
                    return self._json(404, {"error": "not found"})
                try:
                    with app.lock:
                        out = getattr(app, name)(body)
                    self._json(200, out)
                except Exception as e:  # report to the page instead of dropping the connection
                    self._json(400, {"error": f"{type(e).__name__}: {e}"})

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    html = render_html({"meta": app.ctx.meta(), "problems": [], "traces": {}, "curves": None}, "live")
                    return self._send(200, html.encode(), "text/html; charset=utf-8")
                self._dispatch({})

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    return self._json(400, {"error": "invalid JSON"})
                self._dispatch(body)

        return Handler


def serve(ctx: VizContext, host: str = "127.0.0.1", port: int = 8765) -> None:
    app = LiveApp(ctx)
    httpd = ThreadingHTTPServer((host, port), app.handler())
    print(f"Live visualiser on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
