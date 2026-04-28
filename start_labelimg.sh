#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CLASS_FILE="$PROJECT_ROOT/configs/labelimg_classes.txt"
DATA_ROOT="$PROJECT_ROOT/data"
LABEL_ROOT="$PROJECT_ROOT/labels"
ARG_ONE="${1:-}"
ARG_TWO="${2:-}"

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
    if "$candidate" -c "import labelImg" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

resolve_dir() {
  local input_path="$1"
  if [[ "$input_path" = /* ]]; then
    printf '%s\n' "$input_path"
  else
    printf '%s\n' "$PROJECT_ROOT/$input_path"
  fi
}

if [[ -n "$ARG_ONE" && -n "$ARG_TWO" ]]; then
  IMAGE_DIR="$(resolve_dir "$ARG_ONE")"
  SAVE_DIR="$(resolve_dir "$ARG_TWO")"
elif [[ -n "$ARG_ONE" ]]; then
  IMAGE_DIR="$DATA_ROOT/$ARG_ONE"
  SAVE_DIR="$LABEL_ROOT/$ARG_ONE"
else
  IMAGE_DIR="$DATA_ROOT"
  SAVE_DIR="$LABEL_ROOT"
fi

if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Image directory not found: $IMAGE_DIR"
  echo "Usage 1 (single class): bash $PROJECT_ROOT/start_labelimg.sh 车刀"
  echo "Usage 2 (mixed scenes): bash $PROJECT_ROOT/start_labelimg.sh data/mixed_scenes labels/mixed_scenes"
  exit 1
fi

PYTHON_BIN="$(pick_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "No usable Python runtime found for LabelImg."
  echo "Please install dependencies first: pip install pyqt5 lxml labelImg"
  exit 1
fi

mkdir -p "$SAVE_DIR"
exec "$PYTHON_BIN" -m labelImg.labelImg "$IMAGE_DIR" "$CLASS_FILE" "$SAVE_DIR"
