"""Keep the project from taking every CPU core of the machine.

`limit_cpus(n)` is called once at start-up (the CLI, the scripts and the test suite do it):

* thread pools - torch intra-/inter-op threads, OpenMP / MKL / OpenBLAS, and the Rust thread pool
  of `tokenizers` - are sized to `n`;
* on Linux the process is also pinned to `n` cores (`sched_setaffinity`). Child processes inherit
  that, so TextWorld's game compiler and engines, and the runs started by `scripts/run_ablation.py`,
  stay inside the same cores.

`n` comes from the argument, else the `MCTS_LAYA_CPUS` environment variable, else half of the cores
available to the process. `0` or `all` means no limit. The chosen value is written back to
`MCTS_LAYA_CPUS` so child processes use the same budget instead of halving it again.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

ENV_VAR = "MCTS_LAYA_CPUS"
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")

log = logging.getLogger(__name__)
_applied: Optional[int] = None


def available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def resolve_cpus(value: Union[int, str, None] = None) -> Optional[int]:
    """Number of cores to use, or None for no limit."""
    if value is None:
        value = os.environ.get(ENV_VAR)
    if value is None or str(value).strip() == "":
        return max(1, available_cpus() // 2)
    if str(value).strip().lower() in ("0", "all", "none"):
        return None
    n = int(value)
    if n < 0:
        raise ValueError(f"cpus must be >= 0, got {n}")
    return min(n, available_cpus())


def limit_cpus(value: Union[int, str, None] = None) -> Optional[int]:
    """Apply the CPU budget to this process (and its future children). Returns the core count, or None."""
    global _applied
    n = resolve_cpus(value)
    os.environ[ENV_VAR] = str(n or 0)
    if n is None:
        return None
    for var in THREAD_VARS:
        os.environ[var] = str(n)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true" if n > 1 else "false")
    if hasattr(os, "sched_setaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        if len(cores) > n:
            os.sched_setaffinity(0, cores[:n])
    try:
        import torch

        torch.set_num_threads(n)
        if _applied is None:
            try:
                torch.set_num_interop_threads(max(1, min(n, 4)))
            except RuntimeError:  # only allowed before torch has run any parallel work
                pass
    except ImportError:
        pass
    if _applied != n:
        log.info("CPU limit: %d of %d cores (set %s=0 or --cpus 0 for no limit)", n, os.cpu_count() or n, ENV_VAR)
    _applied = n
    return n
