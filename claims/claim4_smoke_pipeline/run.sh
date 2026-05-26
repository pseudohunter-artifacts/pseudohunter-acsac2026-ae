#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
fi

export PYTHONPATH="$PWD/artifact/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" artifact/tools/verify_claims.py claim4
