#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${GATEWAY_PORT:-8000}"
if [ -f .env ]; then
  set -a
  source .env
  set +a
  PORT="${GATEWAY_PORT:-8000}"
fi

PYTHON="python"
if [ -f .venv/Scripts/python.exe ]; then
  PYTHON=".venv/Scripts/python.exe"
elif [ -f .venv/bin/python ]; then
  PYTHON=".venv/bin/python"
fi

if [ ! -f artifacts/target/target_classifier.joblib ]; then
  echo "No trained model found -- running training first..."
  "$PYTHON" -m modelvault.model.train
fi

exec "$PYTHON" -m uvicorn modelvault.gateway.api:app --host 0.0.0.0 --port "$PORT"
