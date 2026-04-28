#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv-labelstudio"
PYTHON_BIN="$VENV_DIR/bin/python"
LABEL_STUDIO_BIN="$VENV_DIR/bin/label-studio"
PORT="${1:-8080}"

if [[ ! -x "$PYTHON_BIN" || ! -x "$LABEL_STUDIO_BIN" ]]; then
  echo "Label Studio environment not found: $VENV_DIR"
  echo "Please install it first:"
  echo "  python3 -m venv .venv-labelstudio"
  echo "  .venv-labelstudio/bin/python -m pip install label-studio"
  exit 1
fi

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$SCRIPT_DIR"
export LABEL_STUDIO_BASE_DATA_DIR="$SCRIPT_DIR/.labelstudio"
export DEBUG=0

exec "$LABEL_STUDIO_BIN" start \
  --no-browser \
  --port "$PORT"
