"""Processing speed (docs/ARCHITECTURE.md section 32): finished jobs are recorded and summed per bucket."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.helpers import jpeg_bytes, noise_image, run_jobs, upload_bytes


def _insert(conn, type_, at: datetime, ok=True, ms=1200):
    from app.db import tx

    with tx(conn):
        conn.execute("INSERT INTO job_runs(type, queue, finished_at, duration_ms, ok) VALUES (?, 'main', ?, ?, ?)",
                     (type_, at.strftime("%Y-%m-%dT%H:%M:%S.000Z"), ms, 1 if ok else 0))


def test_summary_buckets_the_last_hour(conn):
    from app import throughput

    now = datetime(2026, 11, 8, 10, 30, 20, tzinfo=timezone.utc)
    _insert(conn, "ingest", now - timedelta(seconds=5))
    _insert(conn, "ingest", now - timedelta(minutes=5), ms=800)
    _insert(conn, "ingest", now - timedelta(minutes=5, seconds=30), ok=False, ms=400)
    _insert(conn, "analyze", now - timedelta(minutes=59))  # 09:31:20 UTC: the first bucket starts 09:31:00
    _insert(conn, "analyze", now - timedelta(minutes=70))  # outside
    _insert(conn, "bogus", now)  # unknown type: ignored

    s = throughput.summary(conn, "hour", now=now)
    assert s["range"] == "hour" and s["bucket_s"] == 60 and len(s["buckets"]) == 60
    assert s["buckets"][-1] == "2026-11-08T10:30:00.000Z" and s["buckets"][0] == "2026-11-08T09:31:00.000Z"
    assert s["end"] == "2026-11-08T10:31:00.000Z"
    by = {t["type"]: t for t in s["types"]}
    assert [t["type"] for t in s["types"]] == [t for t, _ in throughput.TYPES]
    ingest = by["ingest"]
    assert ingest["total"] == 3 and ingest["failed"] == 1 and ingest["avg_ms"] == 800
    assert ingest["counts"][59] == 1 and ingest["counts"][54] == 1 and ingest["counts"][53] == 1
    assert ingest["failed_counts"][53] == 1 and ingest["failed_counts"][54] == 0
    assert sum(ingest["counts"]) == 3
    assert by["analyze"]["counts"][0] == 1 and by["analyze"]["total"] == 1
    assert by["text"]["total"] == 0 and by["text"]["avg_ms"] is None and len(by["text"]["counts"]) == 60
    assert s["per_minute"]["ingest"] == 0.2 and s["per_minute"]["analyze"] == 0.0  # two ok runs in ten minutes


def test_day_and_week_align_to_local_hours(conn):
    from app import throughput
    from app.config import settings

    settings.local_timezone = "Asia/Kolkata"  # UTC+5:30: hour buckets start at :30 UTC
    now = datetime(2026, 11, 8, 10, 10, 0, tzinfo=timezone.utc)  # 15:40 IST
    _insert(conn, "text", now - timedelta(minutes=5))
    _insert(conn, "text", now - timedelta(hours=23, minutes=20))  # 16:20 IST yesterday: first bucket
    s = throughput.summary(conn, "day", now=now)
    assert s["bucket_s"] == 3600 and len(s["buckets"]) == 24
    assert s["buckets"][-1] == "2026-11-08T09:30:00.000Z"  # 15:00 IST
    text = next(t for t in s["types"] if t["type"] == "text")
    assert text["counts"][-1] == 1 and text["counts"][0] == 1 and text["total"] == 2
    w = throughput.summary(conn, "week", now=now)
    assert w["bucket_s"] == 21600 and len(w["buckets"]) == 28
    assert w["buckets"][-1] == "2026-11-08T06:30:00.000Z"  # 12:00 IST, the six-hour block holding 15:40
    assert next(t for t in w["types"] if t["type"] == "text")["total"] == 2
    with pytest.raises(ValueError):
        throughput.summary(conn, "month", now=now)


def test_prune_keeps_two_weeks(conn):
    from app import throughput

    now = datetime.now(timezone.utc)
    _insert(conn, "ingest", now - timedelta(days=throughput.KEEP_DAYS + 1))
    _insert(conn, "ingest", now - timedelta(days=1))
    assert throughput.prune(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM job_runs").fetchone()[0] == 1


def test_the_worker_records_runs_and_the_api_reports_them(client, conn, monkeypatch):
    from app import jobs, throughput, worker

    up = upload_bytes(client, jpeg_bytes(noise_image(320, 240, seed=5)), "a.jpg")
    assert up["status"] == "queued", up
    run_jobs()
    rows = conn.execute("SELECT type, ok, duration_ms FROM job_runs ORDER BY id").fetchall()
    assert [r["type"] for r in rows][:1] == ["ingest"] and all(r["ok"] == 1 for r in rows)
    assert all(r["duration_ms"] >= 0 for r in rows)

    # a job that fails records a run with ok = 0; a deferral (tool failure) records nothing
    def boom(conn, job):
        raise RuntimeError("handler broke")

    monkeypatch.setattr(worker, "run_job", boom)
    from app.db import tx

    with tx(conn):
        jobs.enqueue(conn, "guess", {"item_id": 1, "version": 1})
    job = jobs.claim_next(conn, "main")
    assert job is not None
    worker._handle(conn, job)
    failed = conn.execute("SELECT type, ok FROM job_runs WHERE ok = 0").fetchall()
    assert [(r["type"], r["ok"]) for r in failed] == [("guess", 0)]

    r = client.get("/api/system/throughput", params={"range": "day"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["range"] == "day" and len(body["buckets"]) == 24
    ingest = next(t for t in body["types"] if t["type"] == "ingest")
    assert ingest["total"] >= 1 and ingest["label"] == "Files stored"
    guess = next(t for t in body["types"] if t["type"] == "guess")
    assert guess["failed"] == 1
    assert body["per_minute"]["ingest"] >= 0.1
    assert client.get("/api/system/throughput", params={"range": "month"}).status_code == 422
    assert isinstance(throughput.summary(conn, "hour")["per_minute"], dict)
