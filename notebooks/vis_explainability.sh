#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PORT="${PORT:-2719}"
cd "$PROJECT_ROOT"
exec uv run marimo run notebooks/vis_explainability.py --host 0.0.0.0 --port "$PORT"
