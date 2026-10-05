"""Download Laya checkpoints from the Hugging Face Hub into a local directory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ID = "convaiinnovations/laya"
# checkpoint name -> subfolder inside the bundled repo ("" = repo root)
CHECKPOINTS = {"english": "", "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")


def download_checkpoint(name: str = "multilingual", out_dir: str = "models/laya",
                        revision: Optional[str] = None, token: Optional[str] = None) -> str:
    """Fetch one checkpoint and return the local directory that `laya.load` accepts."""
    from huggingface_hub import snapshot_download

    if name not in CHECKPOINTS:
        raise KeyError(f"unknown checkpoint {name!r}; choose from {sorted(CHECKPOINTS)}")
    sub = CHECKPOINTS[name]
    prefix = f"{sub}/" if sub else ""
    snapshot_download(
        REPO_ID,
        local_dir=out_dir,
        revision=revision,
        token=token or os.environ.get("HF_TOKEN"),
        allow_patterns=[prefix + f for f in FILES],
    )
    path = Path(out_dir) / sub if sub else Path(out_dir)
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError(f"download finished but {path / 'model.safetensors'} is missing")
    return str(path)
