"""Storage tiers (docs/ARCHITECTURE.md section 31): moving the files of the earlier single-folder
layout from DATA_DIR to ORIGINALS_DIR.

CONTRACT.
- move_staged() -> int. Runs at startup, before recovery and the workers (app.main.lifespan): every
  DATA_DIR/staging/*.part and DATA_DIR/failed/*.part goes to the same name under ORIGINALS_DIR (copy
  to a scratch file beside the destination, size check, fsync, rename, then the old file is removed).
  Synchronous, because chunk writes and failed-upload moves must never race it. A destination that
  already exists is kept (the newer layout wins) and the old file dropped. Returns the files moved.
- migrate_originals(stop=None) -> dict. Moves DATA_DIR/originals/<sh>/<file> to ORIGINALS_DIR/originals
  one file at a time; a destination that already exists with the same size and mtime (a copy made by
  hand before the switch) is kept and only the old file removed. Empty shard folders and
  DATA_DIR/originals itself go at the end. Returns status(). Stops early when the threading.Event
  stop is set; what is left is picked up at the next start. A file that cannot be moved is logged,
  skipped and tried again next time.
- start() runs migrate_originals on a nice-19 daemon thread when ORIGINALS_DIR is set and legacy
  originals exist (returns True when it did); stop() ends it. status() -> {active, running,
  pending_files, pending_bytes, moved_files, moved_bytes, error, started_at, finished_at}: active is
  True while legacy originals remain or the move runs; pending counts what was still to move when the
  run started, moved what this run has done.
- While a legacy file remains, storage.original_path returns it, so reads keep working during the move.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from app import storage
from app.util import now_iso

log = logging.getLogger(__name__)

_lock = threading.Lock()
_stop = threading.Event()
_thread: threading.Thread | None = None
_state: dict[str, Any] = {
    "active": False, "running": False, "pending_files": 0, "pending_bytes": 0, "moved_files": 0,
    "moved_bytes": 0, "error": None, "started_at": None, "finished_at": None,
}


def _set(**values: Any) -> None:
    with _lock:
        _state.update(values)


def status() -> dict[str, Any]:
    with _lock:
        out = dict(_state)
    if not out["running"]:
        legacy = storage.legacy_dir("originals")
        out["active"] = bool(legacy is not None and legacy.is_dir() and any(_files(legacy)))
    return out


def _files(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file() and not p.is_symlink())


def _same(a: os.stat_result, b: os.stat_result) -> bool:
    return a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)


def _move_file(src: Path, dst: Path, prefer_existing: bool) -> int:
    """Move src to dst (copy, check, fsync, rename, unlink). Returns the bytes now at dst. When dst
    exists: prefer_existing keeps it as it is; otherwise it is kept only when size and mtime match
    src, else replaced by a fresh copy."""
    st = src.stat()
    if dst.exists():
        if prefer_existing or _same(st, dst.stat()):
            src.unlink()
            return dst.stat().st_size
    tmp = storage.scratch_beside(dst)
    try:
        shutil.copy2(src, tmp)
        if tmp.stat().st_size != st.st_size:
            raise OSError(f"short copy of {src.name}")
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    except BaseException:
        storage.safe_unlink(tmp)
        raise
    src.unlink()
    return st.st_size


def _remove_empty_dirs(base: Path) -> None:
    """rmdir every empty folder under base, deepest first, then base itself when empty."""
    if not base.is_dir():
        return
    for p in sorted((d for d in base.rglob("*") if d.is_dir() and not d.is_symlink()), key=lambda d: -len(d.parts)):
        try:
            p.rmdir()
        except OSError:
            pass
    try:
        base.rmdir()
    except OSError:
        pass


def move_staged() -> int:
    if not storage.tiered():
        return 0
    moved = 0
    for top in ("staging", "failed"):
        legacy = storage.legacy_dir(top)
        if legacy is None or not legacy.is_dir():
            continue
        for p in _files(legacy):
            dst = storage.originals_root() / top / p.relative_to(legacy)
            try:
                _move_file(p, dst, prefer_existing=True)
                moved += 1
            except OSError:
                log.exception("tiers: could not move %s to %s", p, dst)
        _remove_empty_dirs(legacy)
    if moved:
        log.info("tiers: moved %d staged or failed upload files to %s", moved, storage.originals_root())
    return moved


def migrate_originals(stop: threading.Event | None = None) -> dict[str, Any]:
    legacy = storage.legacy_dir("originals")
    if legacy is None or not legacy.is_dir():
        _set(active=False, running=False)
        return status()
    files = _files(legacy)
    sizes = {}
    for p in files:
        try:
            sizes[p] = p.stat().st_size
        except OSError:
            sizes[p] = 0
    _set(active=bool(files), running=True, pending_files=len(files), pending_bytes=sum(sizes.values()),
         moved_files=0, moved_bytes=0, error=None, started_at=now_iso(), finished_at=None)
    if files:
        log.info("tiers: moving %d originals (%.1f GB) from %s to %s", len(files), sum(sizes.values()) / 1e9,
                 legacy, storage.originals_root() / "originals")
    target = storage.originals_root() / "originals"
    stopped = False
    try:
        for p in files:
            if stop is not None and stop.is_set():
                stopped = True
                break
            dst = target / p.relative_to(legacy)
            try:
                n = _move_file(p, dst, prefer_existing=False)
            except OSError as e:
                log.warning("tiers: could not move %s (%s); it is tried again next time", p, e)
                _set(error=str(e))
                continue
            with _lock:
                _state["moved_files"] += 1
                _state["moved_bytes"] += n
        if not stopped:
            _remove_empty_dirs(legacy)
    except Exception as e:  # noqa: BLE001 - the move must never take the server down
        log.exception("tiers: the move of the originals failed")
        _set(error=str(e))
    finally:
        left = len(_files(legacy)) if legacy.is_dir() else 0
        _set(running=False, active=left > 0, finished_at=now_iso())
        with _lock:
            done = _state["moved_files"]
        if done or left:
            log.info("tiers: moved %d originals; %d left%s", done, left, " (stopped)" if stopped else "")
    return status()


def _run() -> None:
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    except (AttributeError, OSError):
        pass
    migrate_originals(_stop)


def start() -> bool:
    """Start the background move when there is something to move; True when a thread was started."""
    if not storage.tiered():
        return False
    legacy = storage.legacy_dir("originals")
    if legacy is None or not legacy.is_dir():
        return False
    if not _files(legacy):
        _remove_empty_dirs(legacy)
        return False
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return True
        _stop.clear()
        _thread = threading.Thread(target=_run, name="pcc-tiers-move", daemon=True)
        _thread.start()
    return True


def stop(timeout: float = 10.0) -> None:
    with _lock:
        t = _thread
    if t is None or not t.is_alive():
        return
    _stop.set()
    t.join(timeout)
