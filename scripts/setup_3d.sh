#!/usr/bin/env bash
# Install the 3D head generator (docs/ARCHITECTURE.md section 36). It is a separate program with its own Python
# environment (PyTorch), the pinned TripoSR code and the model files: about 8.5 GB in all, and 0.9 GB more for the
# two optional models that read the face's relief. The app itself never imports any of it; it starts the generator
# only while a person's 3D heads are being built.
#
#   scripts/setup_3d.sh            install into MODELS_DIR/gen3d (models/gen3d), or into $GEN3D_DIR when that is set
#   scripts/setup_3d.sh --check    say what is installed and stop
#
# Safe to run again: finished steps are skipped, and every downloaded file is checked against its checksum.
# To keep the large files on another disk, make models/gen3d a link to a folder there before running this.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi
DIR="${GEN3D_DIR:-${MODELS_DIR:-$ROOT/models}/gen3d}"
VENV="$DIR/venv"
WEIGHTS="$DIR/weights"
SRC="$DIR/src"

# TripoSR: code MIT (Tripo AI, Stability AI), weights MIT (huggingface.co/stabilityai/TripoSR)
TRIPOSR_REPO="https://github.com/VAST-AI-Research/TripoSR.git"
TRIPOSR_COMMIT="107cefdc244c39106fa830359024f6a2f1c78871"
TRIPOSR_REV="5b521936b01fbe1890f6f9baed0254ab6351c04a"
TRIPOSR_BASE="https://huggingface.co/stabilityai/TripoSR/resolve/$TRIPOSR_REV"
CKPT_SHA="429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"
CONFIG_SHA="74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62"
# The image encoder's settings file (facebook/dino-vitb16, Apache 2.0): TripoSR asks the Hugging Face cache for it
DINO_REV="f205d5d8e640a89a2b8ef0369670dfc37cc07fc2"
DINO_URL="https://huggingface.co/facebook/dino-vitb16/resolve/$DINO_REV/config.json"
DINO_SHA="b87c0270b97db085fd82cf114a761fd0f62ae7914fbd407c752a2260646b689c"
# U2-Net (Apache 2.0), the ONNX export the rembg project publishes: cuts the background out of a look's picture
U2NET_URL="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
U2NET_SHA="8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491"
# DAViD (Microsoft, MIT licence), plain ONNX files, 449 MB each: depth and surface tilt of a person from a photo.
# gen3d/relief.py corrects the face of a head with them. Optional: without the two files heads are built without
# that stage, so a failed download does not stop this script.
DAVID_BASE="https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models"
DAVID_DEPTH="depth-model-vitb16_384.onnx"
DAVID_DEPTH_SHA="c8dc49821c95f7ab56a30150e58eba989b1d67ce988d11f083dc7350c54b883b"
DAVID_NORMAL="normal-model-vitb16_384.onnx"
DAVID_NORMAL_SHA="3d37ecc812b6b50a09c514f3c8c8ad95da5cd6ae9b68c425bc52f1c9cd78c344"

say() { echo "[3d] $*"; }

check() {
  if [ -x "$VENV/bin/python" ]; then
    (cd "$ROOT" && "$VENV/bin/python" -m gen3d --check --device --weights-dir "$WEIGHTS")
  else
    say "not installed: $VENV has no Python"
    return 1
  fi
}

fetch() {  # fetch URL FILE SHA256
  local url="$1" file="$2" sha="$3"
  if [ -f "$file" ] && echo "$sha  $file" | sha256sum --check --status; then
    return 0
  fi
  mkdir -p "$(dirname "$file")"
  say "downloading $(basename "$file")"
  curl --fail --location --silent --show-error --retry 3 --output "$file.part" "$url"
  if ! echo "$sha  $file.part" | sha256sum --check --status; then
    rm -f "$file.part"
    say "checksum mismatch for $url"
    return 1
  fi
  mv "$file.part" "$file"
}

fetch_optional() {  # fetch_optional URL FILE SHA256: as fetch, but a failure is said and the script goes on
  local url="$1" file="$2" sha="$3"
  if [ -f "$file" ] && echo "$sha  $file" | sha256sum --check --status; then
    return 0
  fi
  mkdir -p "$(dirname "$file")"
  say "downloading $(basename "$file") (optional)"
  if ! curl --fail --location --silent --show-error --retry 3 --output "$file.part" "$url"; then
    rm -f "$file.part"
    say "could not download $url"
    return 1
  fi
  if ! echo "$sha  $file.part" | sha256sum --check --status; then
    rm -f "$file.part"
    say "checksum mismatch for $url"
    return 1
  fi
  mv "$file.part" "$file"
}

if [ "${1:-}" = "--check" ]; then
  check
  exit $?
fi

mkdir -p "$DIR"
say "installing into $DIR"

if [ ! -x "$VENV/bin/python" ]; then
  say "creating the Python environment"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$VENV"
  else
    python3.12 -m venv "$VENV"
  fi
fi
say "installing the Python packages (PyTorch is about 5 GB)"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" -r "$ROOT/gen3d/requirements.txt"
else
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r "$ROOT/gen3d/requirements.txt"
fi

if [ "$(git -C "$SRC/TripoSR" rev-parse HEAD 2>/dev/null || true)" != "$TRIPOSR_COMMIT" ]; then
  say "fetching the TripoSR code at $TRIPOSR_COMMIT"
  rm -rf "$SRC/TripoSR"
  mkdir -p "$SRC"
  git clone --quiet "$TRIPOSR_REPO" "$SRC/TripoSR"
  git -C "$SRC/TripoSR" checkout --quiet "$TRIPOSR_COMMIT"
fi

fetch "$TRIPOSR_BASE/config.yaml" "$WEIGHTS/TripoSR/config.yaml" "$CONFIG_SHA"
fetch "$TRIPOSR_BASE/model.ckpt" "$WEIGHTS/TripoSR/model.ckpt" "$CKPT_SHA"
fetch "$U2NET_URL" "$WEIGHTS/u2net/u2net.onnx" "$U2NET_SHA"
HUB="$WEIGHTS/hf/hub/models--facebook--dino-vitb16"
fetch "$DINO_URL" "$HUB/snapshots/$DINO_REV/config.json" "$DINO_SHA"
mkdir -p "$HUB/refs"
printf '%s' "$DINO_REV" > "$HUB/refs/main"

relief=yes
fetch_optional "$DAVID_BASE/$DAVID_DEPTH" "$WEIGHTS/david/$DAVID_DEPTH" "$DAVID_DEPTH_SHA" || relief=no
fetch_optional "$DAVID_BASE/$DAVID_NORMAL" "$WEIGHTS/david/$DAVID_NORMAL" "$DAVID_NORMAL_SHA" || relief=no
if [ "$relief" = no ]; then
  say "the face relief models are not both there: heads are built without that stage (run this script again to retry)"
fi

say "done"
check
