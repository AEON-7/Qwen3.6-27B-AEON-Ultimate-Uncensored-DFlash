#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/qwen36-aeon-dflash-spark-vllm"
MODE="${1:-dflash}"

if [[ $# -gt 0 ]]; then
  shift
fi

case "$MODE" in
  baseline)
    exec bash "$ROOT/scripts/serve_qwen36_baseline.sh" "$@"
    ;;
  dflash)
    exec bash "$ROOT/scripts/serve_qwen36_dflash.sh" "$@"
    ;;
  bench)
    exec python3 "$ROOT/scripts/bench_categories_stream.py" "$@"
    ;;
  bash|shell)
    exec bash "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
