#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
fi

"$PYTHON" artifact/tools/verify_claims.py claim3 \
  --baseline artifact/paper_results/strict_dpt/results.json \
  --hard-benign artifact/paper_results/strict_dpt/results_strict_dpt_clean_hardbenign_fdroid24_lowbyte025.json
