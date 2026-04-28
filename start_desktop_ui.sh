#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

pick_python() {
  local -a candidates=()
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    candidates+=("$VIRTUAL_ENV/bin/python")
  fi
  candidates+=(
    "$PROJECT_ROOT/.venv/bin/python"
    "$PROJECT_ROOT/venv/bin/python"
  )
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi

  for candidate in "${candidates[@]}"; do
    [[ -x "$candidate" ]] || continue
    if "$candidate" -c "import ultralytics, cv2; from PIL import Image" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "No usable Python runtime found for desktop UI."
  echo "Please install dependencies first: pip install -r requirements.txt"
  exit 1
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/desktop_ui.py" "$@"
