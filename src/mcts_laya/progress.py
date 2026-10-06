"""Progress display for the long stages (teacher data, self-play, training, evaluation).

Modes, from `set_mode()` (the CLI's `--progress`) or `$MCTS_LAYA_PROGRESS`:

* `bar`  - tqdm progress bars on stderr;
* `log`  - one plain line every `$MCTS_LAYA_PROGRESS_INTERVAL` seconds (default 30) and one when a
  stage ends, readable in a log file (`scripts/run_ablation.py` sends each run to `run.log`);
* `off`  - nothing;
* `auto` - (default) `bar` when stderr is a terminal, otherwise `log`.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Any, Iterable, Iterator, Optional

MODES = ("auto", "bar", "log", "off")
ENV_VAR = "MCTS_LAYA_PROGRESS"
_mode: Optional[str] = None


def set_mode(mode: Optional[str]) -> None:
    global _mode
    if mode is not None and mode not in MODES:
        raise ValueError(f"progress mode must be one of {MODES}, got {mode!r}")
    _mode = mode
    if mode is not None:
        os.environ[ENV_VAR] = mode  # child processes (run_ablation.py runs) follow


def mode() -> str:
    m = _mode or os.environ.get(ENV_VAR, "auto")
    if m not in MODES:
        m = "auto"
    if m == "auto":
        return "bar" if sys.stderr.isatty() else "log"
    return m


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    return f"{h}:{rest // 60:02d}:{rest % 60:02d}" if h else f"{rest // 60}:{rest % 60:02d}"


class _LogProgress:
    """tqdm-like object that prints a line now and then instead of redrawing a bar."""

    def __init__(self, total: Optional[int], desc: str, unit: str, interval: float):
        self.total, self.desc, self.unit, self.interval = total, desc, unit, interval
        self.n, self.postfix = 0, ""
        self.t0 = self.last = time.time()

    def update(self, n: int = 1) -> None:
        self.n += n
        if time.time() - self.last >= self.interval:
            self._emit()

    def heartbeat(self) -> None:
        """Print the periodic line even if nothing finished (a stuck stage stays visible in the log)."""
        if time.time() - self.last >= self.interval:
            self._emit()

    def set_postfix(self, **kw: Any) -> None:
        self.postfix = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items())

    def _emit(self, done: bool = False) -> None:
        self.last = now = time.time()
        dt = now - self.t0
        rate = self.n / dt if dt > 0 else 0.0
        count = f"{self.n}/{self.total}" if self.total else str(self.n)
        pct = f" ({100 * self.n / self.total:.0f}%)" if self.total else ""
        eta = ""
        if not done and self.total and rate > 0:
            eta = f", eta {_fmt_time((self.total - self.n) / rate)}"
        tail = f" | {self.postfix}" if self.postfix else ""
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp} [{self.desc}] {count}{pct} {self.unit}, {rate:.2f} {self.unit}/s, "
              f"elapsed {_fmt_time(dt)}{eta}{' - done' if done else ''}{tail}", file=sys.stderr, flush=True)

    def close(self) -> None:
        self._emit(done=True)


class _NoProgress:
    def update(self, n: int = 1) -> None:
        pass

    def heartbeat(self) -> None:
        pass

    def set_postfix(self, **kw: Any) -> None:
        pass

    def close(self) -> None:
        pass


def bar(total: Optional[int] = None, desc: str = "", unit: str = "it"):
    """A progress object with `update(n)`, `set_postfix(**kw)` and `close()`."""
    m = mode()
    if m == "off":
        return _NoProgress()
    if m == "log":
        return _LogProgress(total, desc, unit, float(os.environ.get(ENV_VAR + "_INTERVAL", 30)))
    from tqdm.auto import tqdm

    class _Bar(tqdm):
        def heartbeat(self) -> None:
            self.refresh()  # keeps elapsed time moving while nothing finishes

    return _Bar(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=True, file=sys.stderr)


def track(iterable: Iterable, total: Optional[int] = None, desc: str = "", unit: str = "it") -> Iterator:
    """Iterate with progress; `total` defaults to `len(iterable)`."""
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)  # type: ignore[arg-type]
    p = bar(total, desc, unit)
    try:
        for x in iterable:
            yield x
            p.update(1)
    finally:
        p.close()


@contextlib.contextmanager
def logging_compatible():
    """Route log records through tqdm while bars are shown, so log lines do not break them."""
    if mode() != "bar":
        yield
        return
    from tqdm.contrib.logging import logging_redirect_tqdm

    with logging_redirect_tqdm():
        yield
