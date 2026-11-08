"""Nightly backups to BACKUP_DIR (docs/ARCHITECTURE.md section 31).

CONTRACT.
Layout under BACKUP_DIR:
  snapshots/<YYYY-MM-DD>T<HHMM>/  one complete snapshot: cloud.db (an online copy made with SQLite's
                                  backup API that passed PRAGMA quick_check, taken first), secret.key,
                                  learned/, wallet/ and originals/<sh>/<file>: exactly the originals the
                                  copied database references (trashed items included), so a restore
                                  never has a row without its file. An original whose size and mtime
                                  match the previous snapshot's copy is a hard link to that copy, so a
                                  snapshot costs only what changed since the night before; one deleted
                                  while the run goes is linked from the previous snapshot, or counted as
                                  missing.
  latest -> snapshots/<newest>    a symlink moved when a snapshot completes
  snapshots/.partial-<name>       the snapshot being written; a leftover one is deleted at the next run
Not in a snapshot: derived files (rebuilt from the originals, section 3), staging and failed uploads,
tmp/, site/, and .env (it lives outside DATA_DIR).
Retention: after each run the newest BACKUP_KEEP complete snapshots are kept and older ones deleted.
Schedule: start() runs a nice-19 daemon thread that takes one snapshot a day at BACKUP_HOUR local time
(settings.local_timezone). A missed night (the server was off) is made up CATCH_UP_DELAY_S after start
when the last complete snapshot is older than a day; a first snapshot waits for the hour (or "Back up
now"). A run never starts while app.tiers is still moving originals (checked again every WAIT_POLL_S)
and a failed run is tried again after RETRY_DELAY_S. request_now() (POST /api/system/backup) runs one
as soon as the thread is free, on a one-off thread when the scheduler is not running (workers off);
a run already going is reported, not doubled.
State: kv backup_last holds JSON {started_at, finished_at, ok, error, snapshot, files, missing, bytes_copied,
bytes_linked, duration_s, reason} written when a run ends. status(conn) -> {enabled, dir, hour, keep,
running, progress{phase, files_total, files_done, bytes_copied, bytes_linked} | None, last, snapshots
(names, newest first), next_run_at, free_bytes, total_bytes}. run(conn, reason) takes one snapshot
synchronously and returns the record it stored (tests and the thread use it).
"""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app import storage
from app.config import settings
from app.db import connect, kv_get, kv_set, tx
from app.util import json_dumps, json_loads, now_iso, parse_iso

log = logging.getLogger(__name__)

LAST_KEY = "backup_last"
CATCH_UP_DELAY_S = 300.0
RETRY_DELAY_S = 3600.0
WAIT_POLL_S = 600.0
SNAPSHOT_FMT = "%Y-%m-%dT%H%M"
SMALL_FILES = ("secret.key",)
SMALL_DIRS = ("learned",)

_lock = threading.Lock()
_stop = threading.Event()
_wake = threading.Event()
_thread: threading.Thread | None = None
_running = False
_requested = False
_retry_at: datetime | None = None
_process_started: datetime | None = None
_progress: dict[str, Any] = {}


class _Stopped(Exception):
    pass


# ------------------------------------------------------------------ time
def _tz():
    try:
        return ZoneInfo(settings.local_timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc


def _now() -> datetime:
    return datetime.now(_tz())


def enabled() -> bool:
    return settings.backup_dir is not None


def next_run(now: datetime, last_ok_at: datetime | None, process_started: datetime | None) -> datetime:
    """When the schedule runs next: today's BACKUP_HOUR when still ahead, else tomorrow's; a missed
    night (the last complete snapshot is more than a day old) is made up CATCH_UP_DELAY_S after the
    process started."""
    today = now.replace(hour=settings.backup_hour, minute=0, second=0, microsecond=0)
    if now < today:
        return today
    if last_ok_at is not None and now - last_ok_at > timedelta(hours=24) and process_started is not None:
        return max(process_started + timedelta(seconds=CATCH_UP_DELAY_S), now)
    return today + timedelta(days=1)


# ------------------------------------------------------------------ state
def last_record(conn: sqlite3.Connection) -> dict[str, Any] | None:
    return json_loads(kv_get(conn, LAST_KEY), None)


def _last_ok_at(last: dict[str, Any] | None) -> datetime | None:
    if not last or not last.get("ok"):
        return None
    return parse_iso(last.get("finished_at"))


def _snapshot_dirs() -> list[Path]:
    base = settings.backup_dir / "snapshots" if settings.backup_dir else None
    if base is None or not base.is_dir():
        return []
    return sorted((p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name)


def status(conn: sqlite3.Connection) -> dict[str, Any]:
    with _lock:
        running = _running
        progress = dict(_progress) if _running else None
        retry_at = _retry_at
    last = last_record(conn)
    out: dict[str, Any] = {
        "enabled": enabled(), "dir": str(settings.backup_dir) if enabled() else None,
        "hour": settings.backup_hour, "keep": settings.backup_keep, "running": running, "progress": progress,
        "last": last, "snapshots": [], "next_run_at": None, "free_bytes": None, "total_bytes": None,
    }
    if not enabled():
        return out
    out["snapshots"] = [p.name for p in reversed(_snapshot_dirs())]
    try:
        free, total = storage.disk_usage(settings.backup_dir)
        out["free_bytes"], out["total_bytes"] = free, total
    except OSError:
        pass
    due = next_run(_now(), _last_ok_at(last), _process_started)
    if retry_at is not None and retry_at < due:
        due = retry_at
    out["next_run_at"] = due.isoformat(timespec="seconds")
    return out


def _set_progress(**values: Any) -> None:
    with _lock:
        _progress.update(values)


# ------------------------------------------------------------------ one snapshot
def _snapshot_db(dst: Path) -> None:
    src = sqlite3.connect(settings.db_path, timeout=30)
    try:
        out = sqlite3.connect(dst)
        try:
            src.backup(out)
            row = out.execute("PRAGMA quick_check").fetchone()
            if row is None or row[0] != "ok":
                raise RuntimeError(f"the database copy failed its check: {row[0] if row else 'no result'}")
        finally:
            out.close()
    finally:
        src.close()


def _originals(db: Path) -> dict[Path, Path]:
    """Relative path (<sh>/<uid>.v<n><ext>) -> where the original is now, for every item in the copied
    database db (storage.original_path also finds a file not yet moved from DATA_DIR/originals)."""
    found: dict[Path, Path] = {}
    conn = sqlite3.connect(db)
    try:
        for uid, version, ext in conn.execute("SELECT uid, version, ext FROM items"):
            name = f"{uid}.v{int(version)}{ext}"
            found[Path(uid[:2]) / name] = storage.original_path(uid, int(version), ext)
    finally:
        conn.close()
    return found


def _copy_originals(db: Path, dst: Path, prev: Path | None, stop: threading.Event | None) -> tuple[int, int, int, int]:
    """(files, missing, bytes copied, bytes linked)."""
    files = _originals(db)
    _set_progress(phase="originals", files_total=len(files), files_done=0, bytes_copied=0, bytes_linked=0)
    copied = linked = done = missing = 0
    for rel in sorted(files):
        if stop is not None and stop.is_set():
            raise _Stopped()
        src = files[rel]
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        old = prev / rel if prev is not None else None
        try:
            st = src.stat()
        except FileNotFoundError:
            st = None
        if old is not None and old.is_file():
            ost = old.stat()
            if st is None or (ost.st_size == st.st_size and int(ost.st_mtime) == int(st.st_mtime)):
                os.link(old, out)  # unchanged, or deleted since the database was copied
                linked += ost.st_size
                done += 1
                _set_progress(files_done=done, bytes_linked=linked)
                continue
        if st is None:
            missing += 1
            log.warning("backup: the original %s is missing", rel)
            continue
        try:
            shutil.copy2(src, out)
        except FileNotFoundError:
            missing += 1
            continue
        copied += st.st_size
        done += 1
        _set_progress(files_done=done, bytes_copied=copied)
    return done, missing, copied, linked


def _copy_small(dst: Path) -> None:
    for name in SMALL_FILES:
        p = settings.data_dir / name
        if p.is_file():
            shutil.copy2(p, dst / name)
    for name in SMALL_DIRS:
        p = settings.data_dir / name
        if p.is_dir():
            shutil.copytree(p, dst / name, dirs_exist_ok=True)
    wallet = storage.wallet_dir()
    if wallet.is_dir():
        shutil.copytree(wallet, dst / "wallet", dirs_exist_ok=True)


def _prune(base: Path) -> int:
    snaps = _snapshot_dirs()
    old = snaps[:-settings.backup_keep] if len(snaps) > settings.backup_keep else []
    for p in old:
        shutil.rmtree(p, ignore_errors=True)
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith(".partial-"):
            shutil.rmtree(p, ignore_errors=True)
    return len(old)


def _point_latest(root: Path, name: str) -> None:
    tmp = root / ".latest.tmp"
    storage.safe_unlink(tmp)
    os.symlink(f"snapshots/{name}", tmp)
    os.replace(tmp, root / "latest")


def _snapshot_name(base: Path) -> str:
    name = _now().strftime(SNAPSHOT_FMT)
    candidate, n = name, 2
    while (base / candidate).exists():
        candidate = f"{name}-{n}"
        n += 1
    return candidate


def run(conn: sqlite3.Connection, reason: str = "scheduled", stop: threading.Event | None = None) -> dict[str, Any]:
    """Take one snapshot now (the caller makes sure no other run is going) and store the record."""
    global _running
    if not enabled():
        raise RuntimeError("BACKUP_DIR is not set")
    root = settings.backup_dir
    base = root / "snapshots"
    started = time.monotonic()
    record: dict[str, Any] = {"started_at": now_iso(), "finished_at": None, "ok": False, "error": None,
                              "snapshot": None, "files": 0, "missing": 0, "bytes_copied": 0, "bytes_linked": 0,
                              "duration_s": 0.0, "reason": reason}
    with _lock:
        _running = True
        _progress.clear()
        _progress.update(phase="starting", files_total=0, files_done=0, bytes_copied=0, bytes_linked=0)
    partial: Path | None = None
    try:
        base.mkdir(parents=True, exist_ok=True)
        for p in base.iterdir():  # a run that died or was stopped
            if p.is_dir() and p.name.startswith(".partial-"):
                shutil.rmtree(p, ignore_errors=True)
        name = _snapshot_name(base)
        record["snapshot"] = name
        partial = base / f".partial-{name}"
        partial.mkdir()
        prev: Path | None = None
        latest = root / "latest"
        if latest.is_symlink() or latest.is_dir():
            cand = latest.resolve() / "originals"
            prev = cand if cand.is_dir() else None
        _set_progress(phase="database")
        _snapshot_db(partial / "cloud.db")  # first: the originals copied below are the ones it references
        _copy_small(partial)
        files, missing, copied, linked = _copy_originals(partial / "cloud.db", partial / "originals", prev, stop)
        record.update(files=files, missing=missing, bytes_copied=copied, bytes_linked=linked)
        os.rename(partial, base / name)
        partial = None
        _point_latest(root, name)
        pruned = _prune(base)
        record["ok"] = True
        log.info("backup: snapshot %s done, %d files, %.1f MB copied, %.1f GB linked, %d old snapshot(s) removed",
                 name, files, copied / 1e6, linked / 1e9, pruned)
    except _Stopped:
        record["error"] = "stopped at shutdown"
        log.warning("backup: snapshot stopped at shutdown; it is taken again at the next run")
    except Exception as e:  # noqa: BLE001 - a backup failure is reported, never raised into the caller
        record["error"] = f"{type(e).__name__}: {e}"
        log.exception("backup: snapshot failed")
    finally:
        if partial is not None:
            shutil.rmtree(partial, ignore_errors=True)
        record["finished_at"] = now_iso()
        record["duration_s"] = round(time.monotonic() - started, 1)
        try:
            with tx(conn):
                kv_set(conn, LAST_KEY, json_dumps(record))
        except sqlite3.Error:
            log.exception("backup: could not store the run's record")
        with _lock:
            _running = False
    return record


# ------------------------------------------------------------------ the thread
def _lower_priority() -> None:
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    except (AttributeError, OSError):
        pass


def _tiers_busy() -> bool:
    try:
        from app import tiers

        return bool(tiers.status().get("running"))
    except Exception:  # noqa: BLE001
        return False


def _run_once(reason: str) -> dict[str, Any] | None:
    conn = connect()
    try:
        return run(conn, reason, stop=_stop)
    finally:
        conn.close()


def _loop(one_shot: bool) -> None:
    global _requested, _retry_at
    _lower_priority()
    while not _stop.is_set():
        with _lock:
            requested = _requested
            retry_at = _retry_at
        conn = connect()
        try:
            last = last_record(conn)
        finally:
            conn.close()
        now = _now()
        due = next_run(now, _last_ok_at(last), _process_started)
        if retry_at is not None and retry_at < due:
            due = retry_at
        if requested or now >= due:
            if _tiers_busy():
                log.info("backup: waiting for the move of the originals to finish")
                if _stop.wait(WAIT_POLL_S):
                    return
                continue
            with _lock:
                _requested = False
            record = _run_once("manual" if requested else "scheduled")
            with _lock:
                _retry_at = None if (record is None or record.get("ok")) else _now() + timedelta(seconds=RETRY_DELAY_S)
            if one_shot:
                return
            continue
        wait = min(max(1.0, (due - now).total_seconds()), 300.0)
        _wake.wait(wait)
        _wake.clear()
    return


def start() -> bool:
    """Start the scheduler thread when BACKUP_DIR is set; True when it runs."""
    global _thread, _process_started
    if not enabled():
        return False
    with _lock:
        if _thread is not None and _thread.is_alive():
            return True
        _stop.clear()
        _wake.clear()
        _process_started = _now()
        _thread = threading.Thread(target=_loop, args=(False,), name="pcc-backup", daemon=True)
        _thread.start()
    return True


def stop(timeout: float = 10.0) -> None:
    with _lock:
        t = _thread
    if t is None or not t.is_alive():
        return
    _stop.set()
    _wake.set()
    t.join(timeout)


def request_now() -> dict[str, Any]:
    """Run a snapshot as soon as possible. {"started": bool, "running": bool}: started is False when
    one is already going or backups are off."""
    global _requested, _thread
    if not enabled():
        return {"started": False, "running": False}
    with _lock:
        if _running:
            return {"started": False, "running": True}
        _requested = True
        if _thread is None or not _thread.is_alive():
            _stop.clear()
            _thread = threading.Thread(target=_loop, args=(True,), name="pcc-backup-now", daemon=True)
            _thread.start()
        else:
            _wake.set()
    return {"started": True, "running": True}


def wait(timeout: float | None = None) -> None:
    """Join the backup thread (tests)."""
    with _lock:
        t = _thread
    if t is not None:
        t.join(timeout)
