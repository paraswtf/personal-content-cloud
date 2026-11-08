"""API build id: a short hash of the backend source the site depends on (schemas, routes, URL
helpers), computed once at import so it describes the code this process runs even when the files
change later. The frontend bakes the same hash in at build time (frontend/scripts/api-build.mjs,
read by vite.config.ts into __API_BUILD__); GET /api/auth/state reports this one, and the shell shows
a banner when the two differ, which happens when a server keeps running an older checkout than the
site it serves (docs/ARCHITECTURE.md section 14)."""
from __future__ import annotations

import hashlib
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
TOP_FILES = ("schemas.py", "urls.py")


def api_files(app_dir: Path = APP_DIR) -> list[Path]:
    """The files that define the API surface: schemas.py, urls.py and every api/*.py, sorted by path."""
    files = [app_dir / name for name in TOP_FILES]
    files += (app_dir / "api").glob("*.py")
    present = [f for f in files if f.is_file()]
    return sorted(present, key=lambda p: p.relative_to(app_dir).as_posix())


def compute(app_dir: Path = APP_DIR) -> str:
    """sha256 over (relative posix path, NUL, bytes, NUL) of each file, first 12 hex digits."""
    h = hashlib.sha256()
    for f in api_files(app_dir):
        h.update(f.relative_to(app_dir).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


API_BUILD = compute()
