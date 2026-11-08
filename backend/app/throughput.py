"""Processing speed (docs/ARCHITECTURE.md section 32): one row per finished job, summed into per-minute
or per-hour buckets for the Settings > System charts.

CONTRACT.
- record(conn, job_type, queue, ok, duration_s): one job_runs row with finished_at = now. The worker
  calls it after jobs.complete and after a failure that used an attempt; a job deferred because the
  disk is full or ffmpeg could not run did not run, so nothing is recorded. Never raises.
- prune(conn) -> int: rows older than KEEP_DAYS go (housekeeping).
- summary(conn, range) for GET /api/system/throughput?range=hour|day|week: buckets aligned to the
  local clock (settings.local_timezone), the last one holding now (so it is partial):
    hour: 60 buckets of one minute    day: 24 buckets of one hour    week: 28 buckets of six hours
  {range, bucket_s, start, end, buckets: [UTC ISO start of each], types: [{type, label, total, failed,
  avg_ms, counts: [...], failed_counts: [...]}] in TYPES order (every type present, zeros when idle),
  per_minute: {type: jobs finished per minute over the last ten minutes}}.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import settings
from app.db import tx
from app.util import now_iso, parse_iso

log = logging.getLogger(__name__)

KEEP_DAYS = 14
RATE_WINDOW_S = 600
TYPES: tuple[tuple[str, str], ...] = (
    ("ingest", "Files stored"),
    ("analyze", "Photos and videos analysed"),
    ("text", "Documents read"),
    ("video_preview", "Video web copies"),
    ("guess", "Person guesses"),
    ("recluster", "Face clustering runs"),
)
RANGES: dict[str, tuple[int, int]] = {"hour": (60, 60), "day": (3600, 24), "week": (6 * 3600, 28)}


def record(conn: sqlite3.Connection, job_type: str, queue: str, ok: bool, duration_s: float) -> None:
    try:
        with tx(conn):
            conn.execute(
                "INSERT INTO job_runs(type, queue, finished_at, duration_ms, ok) VALUES (?, ?, ?, ?, ?)",
                (str(job_type), str(queue), now_iso(), int(max(0.0, duration_s) * 1000), 1 if ok else 0),
            )
    except sqlite3.Error:
        log.warning("throughput: could not record a %s run", job_type, exc_info=True)


def prune(conn: sqlite3.Connection) -> int:
    cutoff = now_iso(-KEEP_DAYS * 86400)
    with tx(conn):
        cur = conn.execute("DELETE FROM job_runs WHERE finished_at < ?", (cutoff,))
    return int(cur.rowcount or 0)


def _tz():
    try:
        return ZoneInfo(settings.local_timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc


def _floor(now: datetime, bucket_s: int) -> datetime:
    """Start of the bucket that holds now, on the local clock."""
    local = now.astimezone(_tz())
    if bucket_s < 3600:
        base = local.replace(second=0, microsecond=0)
        return base - timedelta(minutes=base.minute % max(1, bucket_s // 60))
    hours = bucket_s // 3600
    base = local.replace(minute=0, second=0, microsecond=0)
    return base - timedelta(hours=base.hour % hours)


def _iso(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def summary(conn: sqlite3.Connection, range_: str, now: datetime | None = None) -> dict[str, Any]:
    if range_ not in RANGES:
        raise ValueError(f"unknown range {range_!r}")
    bucket_s, n = RANGES[range_]
    now = now or datetime.now(timezone.utc)
    last = _floor(now, bucket_s)
    start = last - timedelta(seconds=bucket_s * (n - 1))
    end = last + timedelta(seconds=bucket_s)
    start_epoch = start.timestamp()
    counts = {t: [0] * n for t, _ in TYPES}
    failed = {t: [0] * n for t, _ in TYPES}
    dur_sum = {t: 0 for t, _ in TYPES}
    dur_n = {t: 0 for t, _ in TYPES}
    for r in conn.execute("SELECT type, finished_at, duration_ms, ok FROM job_runs WHERE finished_at >= ? AND finished_at < ?",
                          (_iso(start), _iso(end))):
        t = r["type"]
        if t not in counts:
            continue
        at = parse_iso(r["finished_at"])
        if at is None:
            continue
        i = int((at.timestamp() - start_epoch) // bucket_s)
        if not 0 <= i < n:
            continue
        counts[t][i] += 1
        dur_sum[t] += int(r["duration_ms"] or 0)
        dur_n[t] += 1
        if not r["ok"]:
            failed[t][i] += 1
    rate_since = _iso(now - timedelta(seconds=RATE_WINDOW_S))
    per_minute = {t: 0.0 for t, _ in TYPES}
    for r in conn.execute("SELECT type, COUNT(*) AS n FROM job_runs WHERE finished_at >= ? AND ok = 1 GROUP BY type",
                          (rate_since,)):
        if r["type"] in per_minute:
            per_minute[r["type"]] = round(int(r["n"]) / (RATE_WINDOW_S / 60), 2)
    return {
        "range": range_, "bucket_s": bucket_s, "start": _iso(start), "end": _iso(end),
        "buckets": [_iso(start + timedelta(seconds=bucket_s * i)) for i in range(n)],
        "types": [
            {"type": t, "label": label, "total": sum(counts[t]), "failed": sum(failed[t]),
             "avg_ms": round(dur_sum[t] / dur_n[t]) if dur_n[t] else None,
             "counts": counts[t], "failed_counts": failed[t]}
            for t, label in TYPES
        ],
        "per_minute": per_minute,
    }
