"""Learned model files under DATA_DIR/learned (docs/ARCHITECTURE.md section 18).

Small JSON models (and .npy matrices next to them) written atomically and read through an mtime
cache, so the pipeline pays one stat per call and one parse per change. `enabled()` is the switch:
the LEARNING_ENABLED setting and the kv flag `learning_enabled` (the Settings switch) must both be on;
without a connection the flag is read through a short-lived one and remembered for 30 seconds.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings

DIRNAME = "learned"
ENABLED_KEY = "learning_enabled"
FLAG_TTL_S = 30.0

_lock = threading.Lock()
_json_cache: dict[str, tuple[float, int, Any]] = {}
_array_cache: dict[str, tuple[float, int, np.ndarray]] = {}
_flag: tuple[float, bool] | None = None


def learned_dir() -> Path:
    return Path(settings.data_dir) / DIRNAME


def path(name: str, ext: str = "json") -> Path:
    return learned_dir() / f"{name}.{ext}"


# ------------------------------------------------------------------ switch
def flag_from_db(conn: sqlite3.Connection) -> bool:
    from app.db import kv_get

    return kv_get(conn, ENABLED_KEY, "1") != "0"


def enabled(conn: sqlite3.Connection | None = None) -> bool:
    """Both switches on: the setting and the kv flag."""
    global _flag
    if not settings.learning_enabled:
        return False
    if conn is not None:
        value = flag_from_db(conn)
        _flag = (time.monotonic(), value)
        return value
    if _flag is not None and time.monotonic() - _flag[0] < FLAG_TTL_S:
        return _flag[1]
    try:
        from app.db import connect

        c = connect()
        try:
            value = flag_from_db(c)
        finally:
            c.close()
    except Exception:  # noqa: BLE001 - no database yet: treat as on, the setting still rules
        value = True
    _flag = (time.monotonic(), value)
    return value


def set_enabled(conn: sqlite3.Connection, value: bool) -> None:
    global _flag
    from app.db import kv_set, tx

    with tx(conn):
        kv_set(conn, ENABLED_KEY, "1" if value else "0")
    _flag = (time.monotonic(), bool(value))


def forget_flag() -> None:
    global _flag
    _flag = None


# ------------------------------------------------------------------ files
def _stamp(p: Path) -> tuple[float, int] | None:
    try:
        st = p.stat()
    except OSError:
        return None
    return st.st_mtime, st.st_size


def load(name: str) -> dict[str, Any] | None:
    """The learned model <name>.json, or None when it does not exist. Re-read only when the file changed."""
    p = path(name)
    stamp = _stamp(p)
    if stamp is None:
        _json_cache.pop(name, None)
        return None
    with _lock:
        hit = _json_cache.get(name)
        if hit is not None and hit[0] == stamp[0] and hit[1] == stamp[1]:
            return hit[2]
    try:
        model = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    with _lock:
        _json_cache[name] = (stamp[0], stamp[1], model)
    return model


def save(name: str, model: dict[str, Any]) -> Path:
    """Write <name>.json atomically (tmp file in the same directory, then rename)."""
    d = learned_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = path(name)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, p)
    with _lock:
        _json_cache.pop(name, None)
    return p


def load_array(name: str) -> np.ndarray | None:
    p = path(name, "npy")
    stamp = _stamp(p)
    if stamp is None:
        _array_cache.pop(name, None)
        return None
    with _lock:
        hit = _array_cache.get(name)
        if hit is not None and hit[0] == stamp[0] and hit[1] == stamp[1]:
            return hit[2]
    try:
        arr = np.load(p, allow_pickle=False)
    except (OSError, ValueError):
        return None
    with _lock:
        _array_cache[name] = (stamp[0], stamp[1], arr)
    return arr


def save_array(name: str, arr: np.ndarray) -> Path:
    d = learned_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = path(name, "npy")
    tmp = p.with_name(p.name + ".tmp.npy")
    np.save(tmp, np.ascontiguousarray(arr), allow_pickle=False)
    os.replace(tmp, p)
    with _lock:
        _array_cache.pop(name, None)
    return p


def remove(name: str) -> None:
    for p in (path(name), path(name, "npy")):
        try:
            p.unlink()
        except OSError:
            pass
    with _lock:
        _json_cache.pop(name, None)
        _array_cache.pop(name, None)


def clear() -> None:
    """Delete every learned file (the choices themselves stay in the database)."""
    d = learned_dir()
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    with _lock:
        _json_cache.clear()
        _array_cache.clear()
