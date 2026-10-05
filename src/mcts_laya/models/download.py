"""Download Laya checkpoints from the Hugging Face Hub into a local directory.

All checkpoints live in one Hub repo (english at its root, the others in subfolders). Locally each
checkpoint gets its own directory, `<out_dir>/<name>`, whatever its place in the repo.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

REPO_ID = "convaiinnovations/laya"
# checkpoint name -> subfolder inside the bundled repo ("" = repo root)
CHECKPOINTS = {"english": "", "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
# top-level entries of one checkpoint, used to move files out of the staging directory
ENTRIES = ("rl_agent_config.json", "model.safetensors", "tokenizer", "encoder")


def _migrate_root_english(out: Path, target: Path) -> bool:
    """Earlier versions kept english directly in `out_dir`; move it into `out_dir/english`."""
    if not (out / "model.safetensors").exists() or (target / "model.safetensors").exists():
        return False
    target.mkdir(parents=True, exist_ok=True)
    for entry in ENTRIES:
        if (out / entry).exists():
            shutil.move(str(out / entry), str(target / entry))
    print(f"moved the english checkpoint from {out} to {target}")
    return True


def download_checkpoint(name: str = "multilingual", out_dir: str = "models/laya",
                        revision: Optional[str] = None, token: Optional[str] = None,
                        force: bool = False) -> str:
    """Fetch one checkpoint into `<out_dir>/<name>` and return that directory (what `laya.load` accepts)."""
    from huggingface_hub import snapshot_download

    if name not in CHECKPOINTS:
        raise KeyError(f"unknown checkpoint {name!r}; choose from {sorted(CHECKPOINTS)}")
    out, target = Path(out_dir), Path(out_dir) / name
    if name == "english" and not force:
        _migrate_root_english(out, target)
    if (target / "model.safetensors").exists() and not force:
        return str(target)

    sub = CHECKPOINTS[name]
    prefix = f"{sub}/" if sub else ""
    staging = out / ".staging" / name
    snapshot_download(
        REPO_ID,
        local_dir=staging,
        revision=revision,
        token=token or os.environ.get("HF_TOKEN"),
        allow_patterns=[prefix + f for f in FILES],
    )
    src = staging / sub if sub else staging
    if not (src / "model.safetensors").exists():
        raise FileNotFoundError(f"download finished but {src / 'model.safetensors'} is missing")
    target.mkdir(parents=True, exist_ok=True)
    for entry in ENTRIES:
        if (src / entry).exists():
            if (target / entry).exists():
                shutil.rmtree(target / entry) if (target / entry).is_dir() else (target / entry).unlink()
            shutil.move(str(src / entry), str(target / entry))
    shutil.rmtree(staging)
    if not any((out / ".staging").iterdir()):
        (out / ".staging").rmdir()
    return str(target)
