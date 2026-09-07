#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

PYTHON="python"
if [ -f .venv/Scripts/python.exe ]; then
  PYTHON=".venv/Scripts/python.exe"
elif [ -f .venv/bin/python ]; then
  PYTHON=".venv/bin/python"
fi

if [ ! -f artifacts/target/target_classifier.joblib ]; then
  echo "== Training target model =="
  "$PYTHON" -m modelvault.model.train
  echo
fi

echo "== Running undefended-vs-defended attack comparison =="
echo "(attack_sim.evaluate exercises the full gateway in-process -- no separate server needed)"
"$PYTHON" -m attack_sim.evaluate
