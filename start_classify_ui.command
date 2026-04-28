#!/bin/zsh
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
exec bash "$SCRIPT_DIR/start_classify_ui.sh"
