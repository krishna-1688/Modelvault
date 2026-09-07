#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${DASHBOARD_PORT:-8501}"
if [ -f .env ]; then
  set -a
  source .env
  set +a
  PORT="${DASHBOARD_PORT:-8501}"
fi

STREAMLIT="streamlit"
if [ -f .venv/Scripts/streamlit.exe ]; then
  STREAMLIT=".venv/Scripts/streamlit.exe"
elif [ -f .venv/bin/streamlit ]; then
  STREAMLIT=".venv/bin/streamlit"
fi

exec "$STREAMLIT" run dashboard/app.py --server.port "$PORT"
