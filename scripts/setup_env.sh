#!/usr/bin/env bash
# Create the project environment and (optionally) download Laya checkpoints.
#
#   scripts/setup_env.sh                 # venv + package + dev tools
#   scripts/setup_env.sh --download      # ... and the multilingual checkpoint
#   CHECKPOINTS="multilingual english" scripts/setup_env.sh --download
#
# Needs Python 3.10+. Uses uv when available, otherwise venv + pip.
# The checkpoints come from huggingface.co (convaiinnovations/laya); set HF_TOKEN if needed.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
VENV=${VENV:-.venv}
CHECKPOINTS=${CHECKPOINTS:-multilingual}

if command -v uv >/dev/null 2>&1; then
  [ -d "$VENV" ] || uv venv --python "${PYTHON_VERSION:-3.11}" "$VENV"
  uv pip install --python "$VENV/bin/python" -e ".[dev]"
else
  [ -d "$VENV" ] || "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -e ".[dev]"
fi

"$VENV/bin/python" -c "import laya, torch; print('laya', laya.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

if [ "${1:-}" = "--download" ]; then
  # shellcheck disable=SC2086
  "$VENV/bin/mcts-laya" download --checkpoint $CHECKPOINTS --out models/laya
fi
