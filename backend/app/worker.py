"""Background worker threads for the durable job queue (app.jobs).

CONTRACT (implemented by the ingest agent).
- Thread 'main' runs ingest/restore/faces/recluster jobs one at a time (serialised ingest means
  dedup decisions never race). Thread 'video' runs video_preview jobs.
- Each thread owns its own DB connection, loops: claim_next -> run_job -> complete / fail,
  waits on app.jobs.wake[queue] with a 1 s timeout when idle.
- After the main queue drains, enqueue one 'recluster' job (unique) if any faces were added
  since the last recluster. Every 10 minutes the main thread runs ingest.cleanup_stale_uploads.
- A handler exception: log it with traceback, app.jobs.fail(...). A full disk (ENOSPC/EDQUOT or
  SQLite 'disk is full' anywhere in the exception chain) re-queues the job after
  DISK_FULL_DELAY_S without using up an attempt (app.jobs.defer); when an ingest job finally
  fails, set the upload to status 'error' with a short message and move its file to
  data_dir/failed (ingest.mark_upload_failed) so an admin can retry it.
- jobs.complete/jobs.fail are retried a few times (a lock timeout must not strand the job). If
  they still fail, the job stays 'running'; the main thread's periodic sweep re-queues 'running'
  jobs older than STALE_RUNNING_S that no live thread of this process holds.
- start()/stop() are called from the FastAPI lifespan; nothing starts when
  settings.workers_enabled is False (tests call drain() instead).
- While items stored before capture positions were read are waiting (ingest.gps_backfill_pending),
  a third thread 'gps-backfill' reads their positions in small batches (ingest.backfill_gps_batch)
  after a short delay, pausing between batches, and exits when the backfill is done.
"""
from __future__ import annotations

import errno
import json
import logging
import sqlite3
import threading
import time
import traceback

from app.changelog import actors
from app.changelog import context as changelog_context
from app.changelog import retention
from app import ingest, jobs
from app.config import settings
from app.db import connect, kv_get, kv_set, tx

log = logging.getLogger(__name__)

IDLE_POLL_S = 1.0
CLEANUP_INTERVAL_S = 600.0
RECLUSTER_MARK = "recluster_max_face_id"  # kv: highest face id when the last recluster ran
BOOKKEEPING_DELAYS_S = (2.0, 4.0, 8.0, 16.0)  # waits between tries of jobs.complete / jobs.fail
STALE_RUNNING_S = 3600.0  # 'running' jobs older than this, not held by a live thread, are re-queued
DISK_FULL_DELAY_S = 120.0  # a job that failed because the disk is full waits this long, attempt not counted
GPS_BACKFILL_START_DELAY_S = 15.0  # let startup and the first queued jobs go first
GPS_BACKFILL_PAUSE_S = 0.5  # between batches, so file reads never crowd out serving and ingest
GPS_BACKFILL_RETRY_S = 300.0  # after ffprobe could not run, or an unexpected error

_threads: list[threading.Thread] = []
_stop = threading.Event()
_lock = threading.Lock()
_held: dict[int, int] = {}  # thread ident -> id of the job that thread is running
_held_lock = threading.Lock()


def held_job_ids() -> set[int]:
    with _held_lock:
        return set(_held.values())


def run_job(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    """Dispatch one claimed job to its handler (payload JSON: ingest {upload_id}, faces
    {media_id, version, people?: 1 for a newly stored item}, video_preview {media_id, version},
    restore {superseded_id}, recluster {})."""
    payload = json.loads(job["payload"] or "{}")
    kind = job["type"]
    if kind == "ingest":
        ingest.process_upload(conn, int(payload["upload_id"]))
    elif kind == "faces":
        ingest.process_faces(conn, int(payload["media_id"]), int(payload["version"]),
                             people=bool(payload.get("people")))
    elif kind == "video_preview":
        ingest.process_video_preview(conn, int(payload["media_id"]), int(payload["version"]))
    elif kind == "restore":
        ingest.restore_superseded(conn, int(payload["superseded_id"]))
    elif kind == "recluster":
        _recluster(conn)
    else:
        raise ValueError(f"unknown job type {kind!r}")


def _recluster(conn: sqlite3.Connection) -> None:
    from app.faces import cluster

    max_id = _max_face_id(conn)
    cluster.recluster_unassigned(conn)
    with tx(conn):
        kv_set(conn, RECLUSTER_MARK, max_id)


def _max_face_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(id) FROM faces").fetchone()
    return int(row[0] or 0)


def _maybe_enqueue_recluster(conn: sqlite3.Connection) -> bool:
    """After the main queue drains: queue one recluster if faces were added since the last one and
    some automatic faces are still unassigned. Returns True if a job was queued."""
    max_id = _max_face_id(conn)
    try:
        mark = int(kv_get(conn, RECLUSTER_MARK, "0") or 0)
    except ValueError:
        mark = 0
    if max_id <= mark:
        return False
    if conn.execute("SELECT 1 FROM faces WHERE person_id IS NULL AND assigned_by = 'auto' LIMIT 1").fetchone() is None:
        with tx(conn):
            kv_set(conn, RECLUSTER_MARK, max_id)  # nothing to retry; remember we looked
        return False
    return jobs.enqueue(conn, "recluster", {}, unique=True) is not None


def _final_failure(conn: sqlite3.Connection, job: sqlite3.Row, exc: BaseException) -> None:
    """The job used up its attempts: record the outcome on the affected row."""
    try:
        payload = json.loads(job["payload"] or "{}")
        kind = job["type"]
        if kind == "ingest":
            ingest.mark_upload_failed(conn, int(payload["upload_id"]), f"Processing failed: {ingest.short_error(exc)}")
        elif kind == "faces":
            ingest.mark_faces_failed(conn, int(payload["media_id"]), int(payload["version"]))
        elif kind == "video_preview":
            ingest.mark_preview_failed(conn, int(payload["media_id"]), int(payload["version"]))
        elif kind == "recluster":
            with tx(conn):  # do not queue it again for the same faces
                kv_set(conn, RECLUSTER_MARK, _max_face_id(conn))
    except Exception:
        log.exception("could not record the final failure of job %s", job["id"])


def _bookkeep(conn: sqlite3.Connection, what: str, fn, *args, **kwargs):
    """Call jobs.complete/jobs.fail, retrying on database errors (busy, locked, disk full).
    Re-raises the last error when every try failed; the stale-job sweep then recovers the job."""
    for delay in (*BOOKKEEPING_DELAYS_S, None):
        try:
            return fn(conn, *args, **kwargs)
        except sqlite3.Error as e:
            if conn.in_transaction:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            if delay is None or _stop.is_set():
                raise
            log.warning("could not %s (%s), trying again in %.0fs", what, e, delay)
            _stop.wait(delay)
    return None


def _is_disk_full(exc: BaseException) -> bool:
    """True if the exception or anything in its cause/context chain says the disk is full."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, OSError) and cur.errno in (errno.ENOSPC, errno.EDQUOT):
            return True
        if isinstance(cur, sqlite3.OperationalError) and "disk is full" in str(cur).lower():
            return True
        cur = cur.__cause__ or (None if cur.__suppress_context__ else cur.__context__)
    return False


def _declare_job_change(conn: sqlite3.Connection, job: sqlite3.Row) -> bool:
    """The job's writes join the change that queued it (an upload, an admin action), or a change of
    its own (automatic regrouping, upgrade re-runs). False when that change was undone meanwhile:
    the job then has nothing left to do."""
    cid = job["change_id"] if "change_id" in job.keys() else None
    if cid is not None:
        row = conn.execute("SELECT state FROM changes WHERE id = ?", (cid,)).fetchone()
        if row is not None:
            if row["state"] != "active":
                return False
            changelog_context.resume(conn, int(cid), job_id=int(job["id"]))
            return True
    kind = job["type"]
    # A streaming copy nobody asked for (an upgrade backfill) is upkeep, not a change to undo.
    changelog_context.declare(
        conn, actors.system_spec(f"job.{kind}", maintenance=kind == "video_preview"), job_id=int(job["id"])
    )
    return True


def _handle(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    """Run one claimed job and complete or fail it. Never raises for handler errors."""
    started = time.monotonic()
    me = threading.get_ident()
    with _held_lock:
        _held[me] = int(job["id"])
    try:
        if not _declare_job_change(conn, job):
            log.info("job %s (%s): its change was undone, skipping", job["id"], job["type"])
            _bookkeep(conn, f"complete job {job['id']}", jobs.complete, int(job["id"]))
            return
        try:
            run_job(conn, job)
        except Exception as exc:
            if conn.in_transaction:  # a handler bug left a transaction open
                conn.execute("ROLLBACK")
            log.exception("job %s (%s %s) failed on attempt %s", job["id"], job["type"], job["payload"],
                          job["attempts"])
            tb = traceback.format_exc()
            if _is_disk_full(exc):
                # Transient: wait for space without using up attempts, so a finished upload is never
                # given up on just because the disk was full for a while.
                log.warning("job %s waits %.0fs for disk space", job["id"], DISK_FULL_DELAY_S)
                _bookkeep(conn, f"defer job {job['id']}", jobs.defer, int(job["id"]), tb, DISK_FULL_DELAY_S)
                return
            retried = _bookkeep(conn, f"record the failure of job {job['id']}", jobs.fail, int(job["id"]), tb)
            if not retried:
                log.error("job %s (%s) failed for good", job["id"], job["type"])
                _final_failure(conn, job, exc)
            return
        _bookkeep(conn, f"complete job {job['id']}", jobs.complete, int(job["id"]))
        log.debug("job %s (%s) done in %.2fs", job["id"], job["type"], time.monotonic() - started)
    finally:
        # Describe the change (or drop it if the job left it empty), then forget it: housekeeping
        # on this connection must never join the last job's change.
        changelog_context.finish(conn)
        with _held_lock:
            _held.pop(me, None)


def _housekeeping(conn: sqlite3.Connection) -> None:
    """Every CLEANUP_INTERVAL_S on the main worker: abandoned uploads, orphaned files, the trash.
    Writes here belong to an upkeep change (listed in History only if it changed content)."""
    changelog_context.declare(conn, actors.system_spec("maintenance.cleanup", "Cleaned up abandoned uploads",
                                                       maintenance=True))
    try:
        ingest.cleanup_stale_uploads(conn)
    finally:
        changelog_context.finish(conn)
    try:
        retention.expire(conn)
        retention.purge_for_space(conn)
    except Exception:
        log.exception("history: trash housekeeping failed")

def drain(queue: str | None = None, max_jobs: int = 10_000) -> int:
    """Run queued jobs synchronously in the calling thread until none are runnable (jobs delayed by
    a retry backoff are left for later). Both queues when queue is None, main first.
    Returns the number of jobs run. Used by tests and scripts."""
    queues = ["main", "video"] if queue is None else [queue]
    conn = connect()
    ran = 0
    try:
        while ran < max_jobs:
            job = None
            for q in queues:
                job = jobs.claim_next(conn, q)
                if job is None and q == "main" and _maybe_enqueue_recluster(conn):
                    job = jobs.claim_next(conn, q)
                if job is not None:
                    break
            if job is None:
                break
            _handle(conn, job)
            ran += 1
    finally:
        conn.close()
    return ran


def _loop(queue: str) -> None:
    conn = connect()
    wake = jobs.wake[queue]
    last_cleanup = float("-inf")
    busy = True  # check for a recluster once at startup too
    try:
        while not _stop.is_set():
            try:
                if queue == "main" and time.monotonic() - last_cleanup >= CLEANUP_INTERVAL_S:
                    last_cleanup = time.monotonic()
                    for q in ("main", "video"):
                        n = jobs.requeue_stale(conn, q, STALE_RUNNING_S, held_job_ids())
                        if n:
                            log.warning("re-queued %d stranded '%s' jobs", n, q)
                    _housekeeping(conn)
                wake.clear()
                job = jobs.claim_next(conn, queue)
                if job is None and queue == "main" and busy:
                    busy = False
                    if _maybe_enqueue_recluster(conn):
                        continue
                if job is None:
                    wake.wait(IDLE_POLL_S)
                    continue
                busy = True
                _handle(conn, job)
            except Exception:
                log.exception("worker %s loop error", queue)
                if conn.in_transaction:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                _stop.wait(IDLE_POLL_S)
    finally:
        conn.close()
        log.info("worker %s stopped", queue)


def _gps_backfill_loop() -> None:
    """Background reading of capture positions for items stored before they were read."""
    if _stop.wait(GPS_BACKFILL_START_DELAY_S):
        return
    conn = connect()
    total = 0
    try:
        while not _stop.is_set():
            changelog_context.declare(conn, actors.system_spec(
                "maintenance.gps", "Read the capture positions of older items", maintenance=True))
            try:
                n = ingest.backfill_gps_batch(conn)
            except Exception:
                log.exception("gps backfill batch failed")
                if conn.in_transaction:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                _stop.wait(GPS_BACKFILL_RETRY_S)
                continue
            finally:
                changelog_context.finish(conn)
            if n is None:
                log.info("gps backfill finished (%d items checked in this run)", total)
                return
            total += n
            _stop.wait(GPS_BACKFILL_PAUSE_S if n else GPS_BACKFILL_RETRY_S)
    finally:
        conn.close()


def _gps_backfill_needed() -> bool:
    try:
        conn = connect()
        try:
            return ingest.gps_backfill_pending(conn)
        finally:
            conn.close()
    except Exception:
        log.exception("could not check whether capture positions need a backfill")
        return False


def start() -> None:
    if not settings.workers_enabled:
        return
    with _lock:
        if any(t.is_alive() for t in _threads):
            return
        _threads.clear()
        _stop.clear()
        for queue in ("main", "video"):
            t = threading.Thread(target=_loop, args=(queue,), name=f"hub-worker-{queue}", daemon=True)
            t.start()
            _threads.append(t)
        if _gps_backfill_needed():
            t = threading.Thread(target=_gps_backfill_loop, name="hub-gps-backfill", daemon=True)
            t.start()
            _threads.append(t)
    log.info("workers started")


def stop(timeout: float = 10.0) -> None:
    with _lock:
        if not _threads:
            return
        _stop.set()
        for ev in jobs.wake.values():
            ev.set()
        deadline = time.monotonic() + timeout
        for t in _threads:
            t.join(max(0.0, deadline - time.monotonic()))
            if t.is_alive():
                log.warning("worker thread %s still busy at shutdown; its job is re-queued on next start", t.name)
        # A thread still finishing a job stays listed, so start() cannot run a second one beside it.
        _threads[:] = [t for t in _threads if t.is_alive()]
