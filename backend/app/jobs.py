"""Durable job queue on the `jobs` table (docs/ARCHITECTURE.md section 7).

Queues: 'main' (ingest, analyze, guess, recluster; one thread, so ingest decisions and the GPU
models are serialised), 'text' (OCR and document parsing), 'video' (transcodes).
Priorities (lower first): ingest 10, analyze 50, video_preview 50, text 60, guess 70, recluster 90.
Handlers must be idempotent: after a crash a 'running' job is re-queued on startup.
"""
from __future__ import annotations

import sqlite3
import threading
from typing import Any

from app.db import tx
from app.util import json_dumps, now_iso

MAX_ATTEMPTS = 3

PRIORITY = {"ingest": 10, "analyze": 50, "video_preview": 50, "text": 60, "guess": 70, "recluster": 90}
QUEUE_OF = {"ingest": "main", "analyze": "main", "guess": "main", "recluster": "main", "text": "text",
            "video_preview": "video"}
QUEUES = ("main", "text", "video")

# Set when a job is enqueued in this process so the worker wakes without waiting a poll cycle.
wake = {q: threading.Event() for q in QUEUES}


def enqueue(conn: sqlite3.Connection, type_: str, payload: dict[str, Any] | None = None, *,
            unique: bool = False) -> int | None:
    """Add a job. With unique=True, skip if an identical queued job exists (returns None). A
    recluster also counts a running one as a duplicate."""
    queue = QUEUE_OF[type_]
    body = json_dumps(payload or {})
    now = now_iso()
    with tx(conn):
        if unique:
            statuses = "('queued', 'running')" if type_ == "recluster" else "('queued')"
            dup = conn.execute(
                f"SELECT id FROM jobs WHERE type = ? AND payload = ? AND status IN {statuses}", (type_, body)
            ).fetchone()
            if dup:
                return None
        cur = conn.execute(
            "INSERT INTO jobs(type, queue, payload, status, priority, created_at, updated_at) "
            "VALUES (?, ?, ?, 'queued', ?, ?, ?)",
            (type_, queue, body, PRIORITY[type_], now, now),
        )
        job_id = int(cur.lastrowid)
    wake[queue].set()
    return job_id


def claim_next(conn: sqlite3.Connection, queue: str) -> sqlite3.Row | None:
    """Atomically take the next runnable job of a queue and mark it running."""
    now = now_iso()
    with tx(conn):
        row = conn.execute(
            "SELECT * FROM jobs WHERE queue = ? AND status = 'queued' AND (run_after IS NULL OR run_after <= ?) "
            "ORDER BY priority, id LIMIT 1",
            (queue, now),
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


def retry_failed(conn: sqlite3.Connection, job_id: int | None = None) -> int:
    """Put errored jobs (one, or all) back on their queues."""
    with tx(conn):
        if job_id is None:
            cur = conn.execute(
                "UPDATE jobs SET status = 'queued', attempts = 0, run_after = NULL, updated_at = ? WHERE status = 'error'",
                (now_iso(),),
            )
        else:
            cur = conn.execute(
                "UPDATE jobs SET status = 'queued', attempts = 0, run_after = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'error'",
                (now_iso(), job_id),
            )
        n = cur.rowcount
    for q in QUEUES:
        wake[q].set()
    return n


def requeue_running(conn: sqlite3.Connection) -> int:
    """Startup recovery: jobs left 'running' by a crash or restart go back to the queue."""
    with tx(conn):
        cur = conn.execute("UPDATE jobs SET status = 'queued', updated_at = ? WHERE status = 'running'", (now_iso(),))
        return cur.rowcount


def requeue_stale(conn: sqlite3.Connection, queue: str, older_than_s: float, exclude_ids: set[int]) -> int:
    """Put 'running' jobs of a queue back that were claimed more than older_than_s ago and are not
    held by a live worker thread (exclude_ids)."""
    cutoff = now_iso(-older_than_s)
    with tx(conn):
        rows = conn.execute(
            "SELECT id FROM jobs WHERE queue = ? AND status = 'running' AND updated_at < ?", (queue, cutoff)
        ).fetchall()
        ids = [int(r["id"]) for r in rows if int(r["id"]) not in exclude_ids]
        for job_id in ids:
            conn.execute(
                "UPDATE jobs SET status = 'queued', run_after = NULL, updated_at = ? WHERE id = ? AND status = 'running'",
                (now_iso(), job_id),
            )
    if ids:
        wake[queue].set()
    return len(ids)


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
