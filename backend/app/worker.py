"""Background worker threads for the durable job queue (app.jobs).

CONTRACT (docs/ARCHITECTURE.md section 7).
- One thread per queue in app.jobs.QUEUES: 'main' (ingest, analyze, guess, recluster: serial, so
  dedup decisions never race and two GPU models never contend), 'text' (OCR and document parsing),
  'video' (transcodes). Each thread owns its DB connection and loops claim_next -> run_job ->
  complete / fail, waiting on app.jobs.wake[queue] (1 s timeout) when idle.
- run_job dispatches through HANDLERS: ingest -> app.ingest.process_upload(upload_id); analyze ->
  app.analysis.pipeline.analyze_item(item_id, version); text -> app.documents.extract.extract_item
  (item_id, version); video_preview -> app.ingest.process_video_preview(item_id, version); guess ->
  app.analysis.guess.propagate(person_id, item_id); recluster -> app.analysis.cluster
  .recluster_unassigned(). Handler modules are imported lazily. A handler whose module (or
  function) cannot be imported raises HandlerMissing: the job is marked failed at once, without
  retries, and the item's status is left as it is (recover_on_startup queues it again once the
  module exists).
- A handler exception is logged with its traceback and the job retried with backoff
  (app.jobs.fail). A full disk (ENOSPC/EDQUOT or SQLite 'disk is full' anywhere in the exception
  chain) re-queues the job after DISK_FULL_DELAY_S without using up an attempt. A ToolFailure
  (ffmpeg/ffprobe timeout, signal or missing tool) anywhere in the chain re-queues it without using
  an attempt either, after 1, 5, 30 minutes and then hourly (jobs.defer_tool_failure); after about
  48 hours of such failures in a row the job fails for good. When a job fails for good,
  _final_failure marks the item's analysis_status / text_status / preview_status 'error' or the
  upload 'error' (its bytes kept in data_dir/failed).
- jobs.complete / jobs.fail are retried a few times; a job left 'running' is re-queued by the
  main thread's periodic sweep when no live thread holds it (STALE_RUNNING_S). A job claimed
  jobs.MAX_ATTEMPTS times that is found running after a crash or by that sweep fails for good
  (finish_exhausted), so a job that kills the process cannot loop the server. stop() gives back
  the attempt of jobs still running at a clean shutdown (jobs.refund_attempt).
- Housekeeping every CLEANUP_INTERVAL_S on the main thread: stale running jobs, abandoned uploads
  (ingest.cleanup_stale_uploads), failed uploads kept longer than settings.trash_days
  (ingest.expire_failed), the trash (ingest.expire_trash) and expired sessions
  (app.auth.sessions.purge_expired). housekeeping() holds housekeeping_lock, so the manual
  endpoint and the worker loop never run it at once.
- The orphan file sweep (ingest.sweep_orphan_files) walks every stored file, so it runs on its own
  low-priority thread: first SWEEP_FIRST_DELAY_S after start, then every SWEEP_INTERVAL_S.
- run_in_background(key, fn, *args): one-off work that should not hold a request thread (page
  renders of a document reprocess) runs on a short-lived thread owned by this module; one per key
  at a time. drain() waits for these threads first.
- After the main queue drains, one unique 'recluster' job is queued when faces were added since
  the last run and some automatic faces are still unassigned.
- start()/stop() are called from the FastAPI lifespan; nothing starts when
  settings.workers_enabled is False. drain() runs jobs synchronously for tests: main first, then
  text, then video.
"""
from __future__ import annotations

import errno
import importlib
import json
import logging
import sqlite3
import threading
import time
import traceback
from typing import Any, Callable

from app import jobs
from app.config import settings
from app.db import connect, kv_get, kv_set, tx

log = logging.getLogger(__name__)

IDLE_POLL_S = 1.0
CLEANUP_INTERVAL_S = 600.0
RECLUSTER_MARK = "recluster_max_face_id"  # kv: highest face id when the last recluster ran
BOOKKEEPING_DELAYS_S = (2.0, 4.0, 8.0, 16.0)  # waits between tries of jobs.complete / jobs.fail
STALE_RUNNING_S = 3600.0  # 'running' jobs older than this, not held by a live thread, are re-queued
DISK_FULL_DELAY_S = 120.0  # a job that failed because the disk is full waits this long, attempt not counted
SWEEP_FIRST_DELAY_S = 3600.0  # the orphan sweep first runs this long after start
SWEEP_INTERVAL_S = 86400.0  # and then once a day


def _item_args(p: dict[str, Any]) -> tuple:
    return int(p["item_id"]), int(p["version"])


# job type -> (module, function, payload -> positional args after conn)
HANDLERS: dict[str, tuple[str, str, Callable[[dict[str, Any]], tuple]]] = {
    "ingest": ("app.ingest", "process_upload", lambda p: (int(p["upload_id"]),)),
    "analyze": ("app.analysis.pipeline", "analyze_item", _item_args),
    "text": ("app.documents.extract", "extract_item", _item_args),
    "video_preview": ("app.ingest", "process_video_preview", _item_args),
    "guess": ("app.analysis.guess", "propagate",
              lambda p: (int(p["person_id"]), int(p["item_id"]) if p.get("item_id") is not None else None)),
    "recluster": ("app.worker", "_recluster", lambda p: ()),
}

_threads: list[threading.Thread] = []
_stop = threading.Event()
_lock = threading.Lock()
_held: dict[int, int] = {}  # thread ident -> id of the job that thread is running
_held_lock = threading.Lock()
housekeeping_lock = threading.Lock()  # held by housekeeping() and by POST /api/system/retry-failed
_background: dict[str, threading.Thread] = {}
_background_lock = threading.Lock()


class HandlerMissing(RuntimeError):
    """The module or function that handles a job type cannot be imported (not installed yet)."""


def held_job_ids() -> set[int]:
    with _held_lock:
        return set(_held.values())


def _resolve(kind: str) -> tuple[Callable[..., Any], Callable[[dict[str, Any]], tuple]]:
    entry = HANDLERS.get(kind)
    if entry is None:
        raise ValueError(f"unknown job type {kind!r}")
    module_name, func_name, args = entry
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise HandlerMissing(f"handler module {module_name} for {kind!r} jobs is not available: {e}") from e
    func = getattr(module, func_name, None)
    if func is None:
        raise HandlerMissing(f"{module_name}.{func_name} for {kind!r} jobs does not exist")
    return func, args


def run_job(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    """Dispatch one claimed job to its handler. Raises HandlerMissing when the handler cannot be
    imported, ValueError for an unknown type, and whatever the handler raises."""
    payload = json.loads(job["payload"] or "{}")
    func, args = _resolve(job["type"])
    func(conn, *args(payload))


def _max_face_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(id) FROM faces").fetchone()
    return int(row[0] or 0)


def _recluster(conn: sqlite3.Connection) -> None:
    try:
        from app.analysis import cluster
    except ImportError as e:
        raise HandlerMissing(f"app.analysis.cluster for 'recluster' jobs is not available: {e}") from e

    max_id = _max_face_id(conn)
    cluster.recluster_unassigned(conn)
    with tx(conn):
        kv_set(conn, RECLUSTER_MARK, max_id)


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
    from app import ingest

    try:
        payload = json.loads(job["payload"] or "{}")
        kind = job["type"]
        if kind == "ingest":
            ingest.mark_upload_failed(conn, int(payload["upload_id"]), f"Processing failed: {ingest.short_error(exc)}")
        elif kind == "analyze":
            ingest.mark_analysis_failed(conn, *_item_args(payload))
        elif kind == "text":
            ingest.mark_text_failed(conn, *_item_args(payload))
        elif kind == "video_preview":
            ingest.mark_preview_failed(conn, *_item_args(payload))
        elif kind == "recluster":
            with tx(conn):  # do not queue it again for the same faces
                kv_set(conn, RECLUSTER_MARK, _max_face_id(conn))
    except Exception:
        log.exception("could not record the final failure of job %s", job["id"])


def finish_exhausted(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> None:
    """Record the final failure of jobs that jobs.requeue_running / requeue_stale moved to 'error'
    because they were interrupted MAX_ATTEMPTS times (call after their transaction committed)."""
    for row in rows:
        log.error("job %s (%s %s) was interrupted %s times; giving up", row["id"], row["type"], row["payload"],
                  row["attempts"])
        _final_failure(conn, row, RuntimeError(f"interrupted {int(row['attempts'])} times"))


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


def _is_tool_failure(exc: BaseException) -> bool:
    """True if the exception or anything in its cause/context chain is a media.probe.ToolFailure."""
    from app.media.probe import ToolFailure

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ToolFailure):
            return True
        cur = cur.__cause__ or (None if cur.__suppress_context__ else cur.__context__)
    return False


def _handle(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    """Run one claimed job and complete or fail it. Never raises for handler errors."""
    started = time.monotonic()
    me = threading.get_ident()
    with _held_lock:
        _held[me] = int(job["id"])
    try:
        try:
            run_job(conn, job)
        except HandlerMissing as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            log.error("job %s (%s): %s", job["id"], job["type"], exc)
            _bookkeep(conn, f"record the failure of job {job['id']}", jobs.fail, int(job["id"]), str(exc),
                      retry=False)
            return
        except Exception as exc:
            if conn.in_transaction:  # a handler bug left a transaction open
                conn.execute("ROLLBACK")
            log.exception("job %s (%s %s) failed on attempt %s", job["id"], job["type"], job["payload"],
                          job["attempts"])
            tb = traceback.format_exc()
            if _is_disk_full(exc):
                log.warning("job %s waits %.0fs for disk space", job["id"], DISK_FULL_DELAY_S)
                _bookkeep(conn, f"defer job {job['id']}", jobs.defer, int(job["id"]), tb, DISK_FULL_DELAY_S)
                return
            if _is_tool_failure(exc):
                if _bookkeep(conn, f"defer job {job['id']}", jobs.defer_tool_failure, int(job["id"]), tb):
                    log.warning("job %s waits for ffmpeg/ffprobe to work again", job["id"])
                    return
                retried = _bookkeep(conn, f"record the failure of job {job['id']}", jobs.fail, int(job["id"]), tb,
                                    retry=False)
            else:
                retried = _bookkeep(conn, f"record the failure of job {job['id']}", jobs.fail, int(job["id"]), tb)
            if not retried:
                log.error("job %s (%s) failed for good", job["id"], job["type"])
                _final_failure(conn, job, exc)
            return
        _bookkeep(conn, f"complete job {job['id']}", jobs.complete, int(job["id"]))
        log.debug("job %s (%s) done in %.2fs", job["id"], job["type"], time.monotonic() - started)
    finally:
        with _held_lock:
            _held.pop(me, None)


def housekeeping(conn: sqlite3.Connection) -> dict[str, int]:
    """Abandoned uploads, failed uploads past their retention, the trash, expired sessions (the
    orphan file sweep runs on its own thread, see _sweep_loop). Each part is guarded so one failure
    does not stop the others. Holds housekeeping_lock, so two callers never run it at once. Returns
    counts per part."""
    with housekeeping_lock:
        return _housekeeping(conn)


def _housekeeping(conn: sqlite3.Connection) -> dict[str, int]:
    from app import ingest

    out = {"cleaned": 0, "failed": 0, "purged": 0, "sessions": 0}
    try:
        out["cleaned"] = ingest.cleanup_stale_uploads(conn)
    except Exception:
        log.exception("housekeeping: upload cleanup failed")
    try:
        out["failed"] = ingest.expire_failed(conn)
    except Exception:
        log.exception("housekeeping: failed upload expiry failed")
    try:
        out["purged"] = ingest.expire_trash(conn)
    except Exception:
        log.exception("housekeeping: trash expiry failed")
    try:
        from app.auth.sessions import purge_expired

        out["sessions"] = purge_expired(conn)
    except Exception:
        log.exception("housekeeping: session purge failed")
    try:
        from app.learning import train as learning_train

        if learning_train.maybe_train(conn):
            out["trained"] = 1
    except Exception:  # noqa: BLE001 - learning never blocks housekeeping
        log.exception("learning: training failed")
    try:
        from app.geo import places as geo_places

        out["geo"] = geo_places.housekeeping(conn)
    except Exception:  # noqa: BLE001 - places never block housekeeping
        log.exception("housekeeping: places failed")

    return out


def run_in_background(key: str, fn: Callable[..., Any], *args: Any) -> threading.Thread | None:
    """Run fn(*args) on a daemon thread owned by this module. Only one thread per key runs at a
    time: returns None (and does nothing) while one for the same key is still running."""
    def target() -> None:
        try:
            fn(*args)
        except Exception:
            log.exception("background task %s failed", key)
        finally:
            with _background_lock:
                if _background.get(key) is threading.current_thread():
                    del _background[key]

    with _background_lock:
        running = _background.get(key)
        if running is not None and running.is_alive():
            return None
        t = threading.Thread(target=target, name=f"pcc-bg-{key}"[:60], daemon=True)
        _background[key] = t
        t.start()
    return t


def wait_background(timeout: float | None = None) -> None:
    """Wait for the threads started by run_in_background (tests and drain)."""
    with _background_lock:
        threads = list(_background.values())
    for t in threads:
        t.join(timeout)


def _lower_priority() -> None:
    """Make the calling thread nice 19 (Linux sets niceness per thread); ignored where unsupported."""
    try:
        import os

        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    except (AttributeError, OSError):
        pass


def _sweep_loop() -> None:
    """Orphan file sweep: first SWEEP_FIRST_DELAY_S after start, then every SWEEP_INTERVAL_S."""
    from app import ingest

    _lower_priority()
    if _stop.wait(SWEEP_FIRST_DELAY_S):
        return
    while True:
        conn = None
        try:
            conn = connect()
            n = ingest.sweep_orphan_files(conn, stop=_stop)
            if n:
                log.info("orphan sweep: removed %d files", n)
        except Exception:
            log.exception("orphan sweep failed")
        finally:
            if conn is not None:
                conn.close()
        if _stop.wait(SWEEP_INTERVAL_S):
            return


def drain(queue: str | None = None, max_jobs: int = 10_000) -> int:
    """Run queued jobs synchronously in the calling thread until none are runnable (jobs delayed by
    a retry backoff are left for later). All queues when queue is None: main first, then text, then
    video. Waits for run_in_background threads first. Returns the number of jobs run. Used by tests
    and scripts."""
    wait_background()
    queues = list(jobs.QUEUES) if queue is None else [queue]
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
                    for q in jobs.QUEUES:
                        n, exhausted = jobs.requeue_stale(conn, q, STALE_RUNNING_S, held_job_ids())
                        if n:
                            log.warning("re-queued %d stranded '%s' jobs", n, q)
                        finish_exhausted(conn, exhausted)
                    housekeeping(conn)
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


def start() -> None:
    if not settings.workers_enabled:
        return
    with _lock:
        if any(t.is_alive() for t in _threads):
            return
        _threads.clear()
        _stop.clear()
        for queue in jobs.QUEUES:
            t = threading.Thread(target=_loop, args=(queue,), name=f"pcc-worker-{queue}", daemon=True)
            t.start()
            _threads.append(t)
        t = threading.Thread(target=_sweep_loop, name="pcc-worker-sweep", daemon=True)
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
        busy = held_job_ids()
        if busy:  # interrupted by a clean shutdown, not by the job: that claim does not count
            try:
                conn = connect()
                try:
                    jobs.refund_attempt(conn, busy)
                finally:
                    conn.close()
            except sqlite3.Error as e:
                log.warning("could not give back the attempts of interrupted jobs %s (%s)", sorted(busy), e)
        # A thread still finishing a job stays listed, so start() cannot run a second one beside it.
        _threads[:] = [t for t in _threads if t.is_alive()]
