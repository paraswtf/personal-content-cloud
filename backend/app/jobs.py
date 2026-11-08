"""Durable job queue on the `jobs` table (docs/ARCHITECTURE.md section 7).

Queues: 'main' (ingest, analyze, guess, recluster; one thread, so ingest decisions and the GPU
models are serialised), 'text' (OCR and document parsing), 'video' (transcodes).
Priorities (lower first): ingest 10, analyze 50, video_preview 50, text 60, guess 70, recluster 90.
Handlers must be idempotent: after a crash a 'running' job is re-queued on startup, unless it has
already been claimed MAX_ATTEMPTS times: then it goes to 'error' (requeue_running and requeue_stale
return those rows so the caller can record the final failure), so a job that kills the process
cannot loop the server. claim_next never takes a job that reached MAX_ATTEMPTS.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from typing import Any, Iterable

from app.db import tx
from app.util import json_dumps, now_iso

MAX_ATTEMPTS = 3

# A ToolFailure (ffmpeg or ffprobe timed out, was killed or is missing) says nothing about the file:
# the job waits and tries again without using an attempt, 1, 5 and 30 minutes, then hourly, and gives
# up TOOL_GIVE_UP_S after the first such failure in a row.
TOOL_DELAYS_S = (60.0, 300.0, 1800.0)
TOOL_LATER_DELAY_S = 3600.0
TOOL_GIVE_UP_S = 48 * 3600.0
_TOOL_MARK = re.compile(r"^\[tool failure (?P<n>\d+) since (?P<since>\S+)\]\n")

PRIORITY = {"ingest": 10, "analyze": 50, "video_preview": 50, "text": 60, "guess": 70, "recluster": 90}
QUEUE_OF = {"ingest": "main", "analyze": "main", "guess": "main", "recluster": "main", "text": "text",
            "video_preview": "video"}
QUEUES = ("main", "text", "video")

# Set when a job is enqueued in this process so the worker wakes without waiting a poll cycle.
wake = {q: threading.Event() for q in QUEUES}


def _dup_sql(type_: str) -> str:
    """Duplicate check for a unique enqueue. The literal status IN ('queued', 'running') term lets
    SQLite use the partial index jobs_dedup; the second term narrows it to queued jobs for every type
    but recluster."""
    narrow = "" if type_ == "recluster" else " AND status = 'queued'"
    return f"SELECT id FROM jobs WHERE type = ? AND payload = ? AND status IN ('queued', 'running'){narrow}"


def _insert(conn: sqlite3.Connection, type_: str, body: str, unique: bool, now: str,
            priority: int | None = None) -> int | None:
    if unique and conn.execute(_dup_sql(type_), (type_, body)).fetchone():
        return None
    cur = conn.execute(
        "INSERT INTO jobs(type, queue, payload, status, priority, created_at, updated_at) "
        "VALUES (?, ?, ?, 'queued', ?, ?, ?)",
        (type_, QUEUE_OF[type_], body, PRIORITY[type_] if priority is None else int(priority), now, now),
    )
    return int(cur.lastrowid)


def enqueue(conn: sqlite3.Connection, type_: str, payload: dict[str, Any] | None = None, *,
            unique: bool = False, priority: int | None = None) -> int | None:
    """Add a job (priority: instead of PRIORITY[type_], for background work such as a re-analysis).
    With unique=True, skip if an identical queued job exists (returns None). A recluster also counts
    a running one as a duplicate."""
    with tx(conn):
        job_id = _insert(conn, type_, json_dumps(payload or {}), unique, now_iso(), priority)
    if job_id is not None:
        wake[QUEUE_OF[type_]].set()
    return job_id


def enqueue_many(conn: sqlite3.Connection, type_: str, payloads: Iterable[dict[str, Any]], *,
                 unique: bool = False, priority: int | None = None) -> int:
    """Add one job per payload in a single transaction (same unique and priority rules as enqueue).
    Returns the number of jobs added."""
    added = 0
    now = now_iso()
    with tx(conn):
        for payload in payloads:
            added += _insert(conn, type_, json_dumps(payload or {}), unique, now, priority) is not None
    if added:
        wake[QUEUE_OF[type_]].set()
    return added


def claim_next(conn: sqlite3.Connection, queue: str) -> sqlite3.Row | None:
    """Atomically take the next runnable job of a queue and mark it running."""
    now = now_iso()
    with tx(conn):
        row = conn.execute(
            "SELECT * FROM jobs WHERE queue = ? AND status = 'queued' AND (run_after IS NULL OR run_after <= ?) "
            "AND attempts < ? ORDER BY priority, id LIMIT 1",
            (queue, now, MAX_ATTEMPTS),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE jobs SET status = 'running', attempts = attempts + 1, updated_at = ? WHERE id = ?",
            (now, row["id"]),
        )
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()


def complete(conn: sqlite3.Connection, job_id: int) -> None:
    with tx(conn):
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def fail(conn: sqlite3.Connection, job_id: int, error: str, *, retry: bool = True) -> bool:
    """Record a failure. Re-queues with backoff while attempts remain; returns True if retried."""
    with tx(conn):
        row = conn.execute("SELECT attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return False
        if retry and row["attempts"] < MAX_ATTEMPTS:
            delay = 15 * (2 ** (row["attempts"] - 1))
            conn.execute(
                "UPDATE jobs SET status = 'queued', last_error = ?, run_after = ?, updated_at = ? WHERE id = ?",
                (error[:4000], now_iso(delay), now_iso(), job_id),
            )
            return True
        conn.execute(
            "UPDATE jobs SET status = 'error', last_error = ?, updated_at = ? WHERE id = ?",
            (error[:4000], now_iso(), job_id),
        )
        return False


def defer(conn: sqlite3.Connection, job_id: int, error: str, delay_s: float) -> bool:
    """Put a running job back after a transient condition (a full disk) without using up an attempt."""
    with tx(conn):
        cur = conn.execute(
            "UPDATE jobs SET status = 'queued', attempts = MAX(attempts - 1, 0), last_error = ?, run_after = ?, "
            "updated_at = ? WHERE id = ?",
            (error[:4000], now_iso(delay_s), now_iso(), job_id),
        )
        return cur.rowcount == 1


def defer_tool_failure(conn: sqlite3.Connection, job_id: int, error: str) -> bool:
    """Put a running job back after a ToolFailure without using up an attempt, waiting longer each
    time (TOOL_DELAYS_S, then TOOL_LATER_DELAY_S). The first line of last_error counts the failures
    in a row and when they started; any other failure overwrites it and so starts the count again.
    Returns False (job left running) once the failures have lasted TOOL_GIVE_UP_S: the caller then
    fails the job for good."""
    with tx(conn):
        row = conn.execute("SELECT last_error FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return False
        m = _TOOL_MARK.match(row["last_error"] or "")
        n, since = (int(m["n"]), m["since"]) if m else (0, now_iso())
        if since <= now_iso(-TOOL_GIVE_UP_S):
            return False
        delay = TOOL_DELAYS_S[n] if n < len(TOOL_DELAYS_S) else TOOL_LATER_DELAY_S
        text = f"[tool failure {n + 1} since {since}]\n{error}"
        conn.execute(
            "UPDATE jobs SET status = 'queued', attempts = MAX(attempts - 1, 0), last_error = ?, run_after = ?, "
            "updated_at = ? WHERE id = ?",
            (text[:4000], now_iso(delay), now_iso(), job_id),
        )
        return True


def retry_failed(conn: sqlite3.Connection, job_id: int | None = None) -> int:
    """Put errored jobs (one, or all) back on their queues."""
    with tx(conn):
        if job_id is None:
            cur = conn.execute(
                "UPDATE jobs SET status = 'queued', attempts = 0, run_after = NULL, last_error = NULL, updated_at = ? "
                "WHERE status = 'error'",
                (now_iso(),),
            )
        else:
            cur = conn.execute(
                "UPDATE jobs SET status = 'queued', attempts = 0, run_after = NULL, last_error = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'error'",
                (now_iso(), job_id),
            )
        n = cur.rowcount
    for q in QUEUES:
        wake[q].set()
    return n


INTERRUPTED_ERROR = "The job was interrupted {n} times (the server stopped or crashed while it ran); giving up."


def _give_up(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'error', last_error = ?, updated_at = ? WHERE id = ?",
        (INTERRUPTED_ERROR.format(n=int(row["attempts"])), now_iso(), int(row["id"])),
    )


def requeue_running(conn: sqlite3.Connection) -> tuple[int, list[sqlite3.Row]]:
    """Startup recovery: jobs left 'running' by a crash or restart go back to the queue, except jobs
    already claimed MAX_ATTEMPTS times (and queued ones at the cap, which claim_next never takes):
    those go to 'error'. Returns (re-queued count, rows moved to 'error', as they were before) so
    the caller can record their final failure once this transaction is committed."""
    with tx(conn):
        exhausted = conn.execute(
            "SELECT * FROM jobs WHERE status IN ('queued', 'running') AND attempts >= ?", (MAX_ATTEMPTS,)
        ).fetchall()
        for row in exhausted:
            _give_up(conn, row)
        cur = conn.execute("UPDATE jobs SET status = 'queued', updated_at = ? WHERE status = 'running'", (now_iso(),))
        return cur.rowcount, exhausted


def refund_attempt(conn: sqlite3.Connection, job_ids: Iterable[int]) -> int:
    """A clean shutdown interrupted these running jobs: give back the attempt their claim used, so
    restarts during a long job do not count towards MAX_ATTEMPTS."""
    ids = [int(i) for i in job_ids]
    if not ids:
        return 0
    with tx(conn):
        cur = conn.execute(
            f"UPDATE jobs SET attempts = MAX(attempts - 1, 0) WHERE status = 'running' AND id IN "
            f"({', '.join('?' * len(ids))})", ids,
        )
        return cur.rowcount


def requeue_stale(conn: sqlite3.Connection, queue: str, older_than_s: float,
                  exclude_ids: set[int]) -> tuple[int, list[sqlite3.Row]]:
    """Put 'running' jobs of a queue back that were claimed more than older_than_s ago and are not
    held by a live worker thread (exclude_ids). Jobs already claimed MAX_ATTEMPTS times go to
    'error' instead. Returns (re-queued count, rows moved to 'error') like requeue_running."""
    cutoff = now_iso(-older_than_s)
    requeued = 0
    exhausted: list[sqlite3.Row] = []
    with tx(conn):
        rows = conn.execute(
            "SELECT * FROM jobs WHERE queue = ? AND status = 'running' AND updated_at < ?", (queue, cutoff)
        ).fetchall()
        for row in rows:
            if int(row["id"]) in exclude_ids:
                continue
            if int(row["attempts"]) >= MAX_ATTEMPTS:
                _give_up(conn, row)
                exhausted.append(row)
                continue
            requeued += conn.execute(
                "UPDATE jobs SET status = 'queued', run_after = NULL, updated_at = ? WHERE id = ? AND status = 'running'",
                (now_iso(), int(row["id"])),
            ).rowcount
    if requeued:
        wake[queue].set()
    return requeued, exhausted


def counts(conn: sqlite3.Connection) -> dict[str, Any]:
    rows = conn.execute("SELECT queue, status, COUNT(*) AS n FROM jobs GROUP BY queue, status").fetchall()
    out: dict[str, Any] = {"queued": 0, "running": 0, "failed": 0, "by_queue": {q: {"queued": 0, "running": 0} for q in QUEUES}}
    for r in rows:
        if r["status"] == "error":
            out["failed"] += r["n"]
        elif r["status"] in ("queued", "running"):
            out[r["status"]] += r["n"]
            out["by_queue"].setdefault(r["queue"], {"queued": 0, "running": 0})[r["status"]] += r["n"]
    return out
