#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MATRIX.yaml [SESSION] [sslx matrix-run options...]" >&2
    exit 2
fi

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX="$1"
SESSION="${2:-ssl_matrix}"
shift
if [[ $# -gt 0 ]]; then
    shift
fi

mkdir -p "$PROJECT/tmux_logs"
EXTRA=""
if [[ $# -gt 0 ]]; then
    printf -v EXTRA ' %q' "$@"
fi
printf -v COMMAND 'cd %q && set -o pipefail && uv run sslx matrix-run %q --analyze%s 2>&1 | tee %q' \
    "$PROJECT" "$MATRIX" "$EXTRA" "$PROJECT/tmux_logs/$SESSION.log"

tmux new-session -d -s "$SESSION" "$COMMAND"
echo "started tmux session: $SESSION"
echo "log: $PROJECT/tmux_logs/$SESSION.log"
