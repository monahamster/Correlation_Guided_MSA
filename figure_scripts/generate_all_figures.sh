#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:-.}"
OUT_DIR="${2:-revision/overleaf_source/figures}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/generate_fig_robustness_curves.py" \
  --experiments-root "$DATA_ROOT/experiments" \
  --out-dir "$OUT_DIR"
python "$SCRIPT_DIR/generate_fig_five_seed_effects.py" \
  --experiments-root "$DATA_ROOT/experiments" \
  --out-dir "$OUT_DIR"
