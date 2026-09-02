#!/usr/bin/env bash
# Copy the author-provided framework artwork into the Overleaf figure directory.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$PROJECT_ROOT/revision/overleaf_source/figures}"
mkdir -p "$OUT_DIR"
cp "$PROJECT_ROOT/framework.png" "$OUT_DIR/fig_framework.png"
