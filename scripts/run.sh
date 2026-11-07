#!/usr/bin/env bash
# Start Personal Content Cloud: set up whatever is missing, then run the server.
#   PORT (default 41010) and HOST (default 127.0.0.1) choose where it listens.
#   DATA_DIR (default ./data) holds the database and every uploaded file.
#   MODELS_DIR (default ./models) holds the models (downloaded on first start).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-41010}"
HOST="${HOST:-127.0.0.1}"

cd "$ROOT/backend"
echo "[run] syncing Python environment"
uv sync --frozen
.venv/bin/python "$ROOT/scripts/download_models.py" || echo "[run] model download failed; stages without their model are skipped until it succeeds"

cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  echo "[run] installing frontend dependencies"
  npm ci --no-audit --no-fund
fi
if [ ! -f dist/index.html ] || [ -n "$(find src public index.html vite.config.ts package.json -newer dist/index.html -print -quit 2>/dev/null)" ]; then
  echo "[run] building frontend"
  npm run build
fi

cd "$ROOT/backend"
echo "[run] starting on http://$HOST:$PORT"
# --timeout-keep-alive: the portal's proxy keeps idle connections to this server open and reuses them.
# With uvicorn's default of 5 s the server closes them first, and the next request on a reused
# connection (a click, a POST) fails at the proxy with "This review server is not running".
exec .venv/bin/uvicorn --factory app.main:create_app --host "$HOST" --port "$PORT" \
  --proxy-headers --forwarded-allow-ips 127.0.0.1 --timeout-graceful-shutdown 10 --timeout-keep-alive 600
