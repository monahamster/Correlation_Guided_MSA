#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:-.}"
OUT_DIR="${2:-figures}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/generate_fig_confusion_binary.py" --data-root "$DATA_ROOT" --out-dir "$OUT_DIR"
python "$SCRIPT_DIR/generate_fig_confusion_mosi_7class.py" --data-root "$DATA_ROOT" --out-dir "$OUT_DIR"
python "$SCRIPT_DIR/generate_fig_silhouette_selected.py" --data-root "$DATA_ROOT" --out-dir "$OUT_DIR"
python "$SCRIPT_DIR/generate_fig_representation_pca.py" --data-root "$DATA_ROOT" --out-dir "$OUT_DIR"
