#!/usr/bin/env bash
# Install the backend into a virtual environment on a Mac, for the iOS tests on the CI machine.
#   install_backend.sh <venv folder>
# The lock (backend/uv.lock) is resolved for the Linux server with a GPU: its onnxruntime-gpu and the
# nvidia-* libraries have no Mac builds. The same versions of everything else are installed, with the
# CPU onnxruntime of the same version in place of the GPU one (the models are off in the tests anyway).
set -euo pipefail
VENV="${1:?usage: install_backend.sh <venv folder>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ="$(mktemp)"
cd "$ROOT/backend"
uv export --frozen --no-hashes --no-dev --no-emit-project --format requirements-txt > "$REQ.all"
ORT="$(sed -n 's/^onnxruntime-gpu==\([^ ;]*\).*/\1/p' "$REQ.all")"
grep -v -E '^(nvidia-|onnxruntime-gpu)' "$REQ.all" > "$REQ"
uv venv --quiet --python 3.12 "$VENV"
if ! uv pip install --quiet --python "$VENV/bin/python" -r "$REQ" "onnxruntime==${ORT}"; then
  echo "onnxruntime ${ORT} has no build here; installing the newest onnxruntime instead"
  uv pip install --quiet --python "$VENV/bin/python" -r "$REQ" onnxruntime
fi
rm -f "$REQ" "$REQ.all"
"$VENV/bin/python" -c "import fastapi, cv2, onnxruntime, PIL; print('backend packages ready, onnxruntime', onnxruntime.__version__)"
