#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
python "$ROOT/code/generate_results.py" --outdir "$ROOT/regenerated_primary"
python "$ROOT/code/verify_and_extend.py" --primary-dir "$ROOT/regenerated_primary" --outdir "$ROOT/regenerated_extended"
python "$ROOT/code/validate_results.py"
python "$ROOT/code/make_figures.py"
echo "Numerical results, verification outputs, and figures regenerated."
