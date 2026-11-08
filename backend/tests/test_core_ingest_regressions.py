"""Regression tests for the ingest and jobs review findings C14, C16, C17, C18, C19, C24 and C28."""
from __future__ import annotations

import json
import os
import threading
import time

from tests.conftest import CSRF
from tests.helpers import jpeg_bytes, make_item, noise_image, pdf_bytes, run_jobs
from tests.test_core_ingest import item_of, quiet_jobs, stage, upload_row  # noqa: F401  (quiet_jobs is a fixture)


def _set_job(conn, job_id: int, **cols) -> None:
    from app.db import tx

    sets = ", ".join(f"{k} = ?" for k in cols)
    with tx(conn):
        conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", (*cols.values(), job_id))


def _only_job(conn, type_: str | None = None):
    rows = conn.execute("SELECT * FROM jobs" + (" WHERE type = ?" if type_ else "") + " ORDER BY id",
                        (type_,) if type_ else ()).fetchall()
    assert len(rows) == 1, [dict(r) for r in rows]
    return rows[0]


def _age(path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


# ------------------------------------------------------------------ C14: poison jobs
def test_c14_startup_gives_up_on_a_job_that_crashed_max_attempts_times(conn, quiet_jobs):
    from app import ingest, jobs, storage

    up = stage(conn, jpeg_bytes(noise_image(120, 90, seed=3)), "crash.jpg")
    job = _only_job(conn, "ingest")
    uid = upload_row(conn, up)["uid"]
    # The process died while running this job on every attempt.
    _set_job(conn, job["id"], status="running", attempts=jobs.MAX_ATTEMPTS)
    with conn:
        conn.execute("UPDATE uploads SET status = 'processing' WHERE id = ?", (up,))

    ingest.recover_on_startup(conn)

    job = conn.execute("SELECT status, last_error FROM jobs WHERE type = 'ingest'").fetchone()
    assert job["status"] == "error" and "interrupted" in job["last_error"]
    row = upload_row(conn, up)
    assert row["status"] == "error"
    assert storage.failed_path(uid).is_file() and not storage.staging_path(uid).exists()
    # Nothing is left for the worker to claim, so the next start does not crash again.
    assert jobs.claim_next(conn, "main") is None


def test_c14_claim_next_skips_jobs_at_the_cap_and_requeue_stale_fails_them(conn):
    from app import jobs, worker

    item = make_item(conn, analysis_status="pending")
    job_id = jobs.enqueue(conn, "analyze", {"item_id": item["id"], "version": 1})
    _set_job(conn, job_id, attempts=jobs.MAX_ATTEMPTS)
    assert jobs.claim_next(conn, "main") is None

    from app.util import now_iso

    _set_job(conn, job_id, status="running", updated_at=now_iso(-7200))
    n, exhausted = jobs.requeue_stale(conn, "main", 3600, set())
    assert n == 0 and [r["id"] for r in exhausted] == [job_id]
    worker.finish_exhausted(conn, exhausted)
    assert conn.execute("SELECT status FROM jobs").fetchone()[0] == "error"
    assert conn.execute("SELECT analysis_status FROM items").fetchone()[0] == "error"


def test_c14_clean_shutdown_gives_back_the_attempt_of_a_busy_job(conn, monkeypatch):
    from app import jobs, worker

    item = make_item(conn, analysis_status="pending")
    jobs.enqueue(conn, "analyze", {"item_id": item["id"], "version": 1})
    job = jobs.claim_next(conn, "main")
    assert job["attempts"] == 1

    class Busy:
        name = "pcc-worker-main"

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

    monkeypatch.setattr(worker, "_threads", [Busy()])
    monkeypatch.setitem(worker._held, -1, int(job["id"]))
    worker.stop(timeout=0.1)
    assert conn.execute("SELECT attempts FROM jobs").fetchone()[0] == 0


def test_c14_running_job_below_the_cap_is_still_requeued(conn):
    from app import ingest, jobs

    item = make_item(conn, analysis_status="pending")
    job_id = jobs.enqueue(conn, "analyze", {"item_id": item["id"], "version": 1})
    _set_job(conn, job_id, status="running", attempts=jobs.MAX_ATTEMPTS - 1)
    ingest.recover_on_startup(conn)
    job = _only_job(conn)
    assert job["status"] == "queued"
    assert jobs.claim_next(conn, "main")["id"] == job_id


# ------------------------------------------------------------------ C16: restore re-queues pending stages
def test_c16_restore_requeues_a_text_job_that_ran_while_trashed(conn):
    from app import ingest, jobs

    item = make_item(conn, kind="pdf", ext=".pdf", mime="application/pdf", text_status="pending",
                     write_files=False)
    jobs.enqueue(conn, "text", ingest.item_payload(item["id"], 1), unique=True)
    ingest.soft_delete(conn, item["id"])
    assert run_jobs("text") == 1  # the real extract_item skips the trashed item and the job is gone
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert conn.execute("SELECT text_status FROM items").fetchone()[0] == "pending"

    ingest.restore(conn, item["id"])
    job = _only_job(conn)
    assert (job["type"], job["status"]) == ("text", "queued")
    assert json.loads(job["payload"]) == {"item_id": item["id"], "version": 1}
    # Unique: trashing and restoring again does not add a second job.
    ingest.soft_delete(conn, item["id"])
    ingest.restore(conn, item["id"])
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_c16_restore_queues_nothing_for_finished_items(conn):
    from app import ingest

    item = make_item(conn, analysis_status="done", text_status="none")
    ingest.soft_delete(conn, item["id"])
    ingest.restore(conn, item["id"])
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


# ------------------------------------------------------------------ C17: tool failures, retry, failed/ retention
def tool_fail(c, item_id, version):
    from app.media.probe import ToolFailure

    try:
        raise ToolFailure("ffmpeg timed out")
    except ToolFailure as e:
        raise RuntimeError("frame extraction failed") from e  # wrapped: found through the cause chain


def test_c17_tool_failure_waits_longer_each_time_without_using_attempts(conn, monkeypatch):
    from app import jobs, worker
    from app.util import now_iso, parse_iso

    monkeypatch.setitem(worker.HANDLERS, "analyze", (__name__, "tool_fail", worker.HANDLERS["analyze"][2]))
    item = make_item(conn, analysis_status="pending")
    job_id = jobs.enqueue(conn, "analyze", {"item_id": item["id"], "version": 1})
    expected = [60, 300, 1800, 3600, 3600]
    for n, delay in enumerate(expected, 1):
        before = parse_iso(now_iso())
        assert worker.drain() == 1
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        assert job["status"] == "queued" and job["attempts"] == 0
        assert job["last_error"].startswith(f"[tool failure {n} since ")
        wait = (parse_iso(job["run_after"]) - before).total_seconds()
        assert delay - 5 < wait < delay + 5
        _set_job(conn, job_id, run_after=None)
    assert conn.execute("SELECT analysis_status FROM items").fetchone()[0] == "pending"

    # About 48 hours of tool failures in a row: the job fails for good and the item is marked.
    job = conn.execute("SELECT last_error FROM jobs WHERE id = ?", (job_id,)).fetchone()
    old = now_iso(-(jobs.TOOL_GIVE_UP_S + 60))
    rest = job["last_error"].split("\n", 1)[1]
    _set_job(conn, job_id, last_error=f"[tool failure 40 since {old}]\n{rest}")
    assert worker.drain() == 1
    assert conn.execute("SELECT status FROM jobs").fetchone()[0] == "error"
    assert conn.execute("SELECT analysis_status FROM items").fetchone()[0] == "error"


def test_c17_retry_failed_endpoint(client):
    from app import jobs, storage
    from app.db import connect, tx
    from app.util import now_iso

    conn = connect()
    try:
        now = now_iso()
        with tx(conn):
            # An upload whose ingest failed for good, bytes kept in failed/.
            cur = conn.execute(
                "INSERT INTO uploads(uid, original_filename, size_bytes, status, message, created_at, updated_at) "
                "VALUES ('failedup2345', 'v.mp4', 3, 'error', 'Processing failed', ?, ?)", (now, now))
            failed_upload = int(cur.lastrowid)
            conn.execute("INSERT INTO jobs(type, queue, payload, status, attempts, created_at, updated_at) "
                         "VALUES ('ingest', 'main', ?, 'error', 3, ?, ?)",
                         (json.dumps({"upload_id": failed_upload}, separators=(",", ":")), now, now))
            # An upload in error whose bytes are gone: its errored ingest job is dropped.
            cur = conn.execute(
                "INSERT INTO uploads(uid, original_filename, size_bytes, status, created_at, updated_at) "
                "VALUES ('gone23456789', 'g.jpg', 3, 'error', ?, ?)", (now, now))
            conn.execute("INSERT INTO jobs(type, queue, payload, status, attempts, created_at, updated_at) "
                         "VALUES ('ingest', 'main', ?, 'error', 3, ?, ?)",
                         (json.dumps({"upload_id": int(cur.lastrowid)}, separators=(",", ":")), now, now))
        storage.failed_path("failedup2345").parent.mkdir(parents=True, exist_ok=True)
        storage.failed_path("failedup2345").write_bytes(b"abc")
        item = make_item(conn, kind="video", ext=".mp4", mime="video/mp4", preview_status="error",
                         write_files=False)
        with tx(conn):
            conn.execute("INSERT INTO jobs(type, queue, payload, status, attempts, last_error, created_at, "
                         "updated_at) VALUES ('video_preview', 'video', ?, 'error', 3, 'boom', ?, ?)",
                         (json.dumps({"item_id": item["id"], "version": 1}, separators=(",", ":")), now, now))

        assert client.post("/api/system/retry-failed", headers={"X-Requested-With": ""}).status_code == 403
        r = client.post("/api/system/retry-failed", headers=CSRF)
        assert r.status_code == 200, r.text
        assert r.json() == {"requeued": 2}

        assert conn.execute("SELECT status FROM uploads WHERE id = ?", (failed_upload,)).fetchone()[0] == "queued"
        assert storage.staging_path("failedup2345").read_bytes() == b"abc"
        assert not storage.failed_path("failedup2345").exists()
        rows = conn.execute("SELECT type, status, attempts FROM jobs ORDER BY type").fetchall()
        assert [tuple(r) for r in rows] == [("ingest", "queued", 0), ("video_preview", "queued", 0)]
        # The handler only runs pending items, so the status is set back too.
        assert conn.execute("SELECT preview_status FROM items WHERE id = ?",
                            (item["id"],)).fetchone()[0] == "pending"
        assert jobs.counts(conn)["failed"] == 0
    finally:
        conn.close()


def test_c17_housekeeping_deletes_failed_files_after_retention(conn):
    from app import storage, worker
    from app.config import settings

    old, fresh = storage.failed_path("oldfail23456"), storage.failed_path("newfail23456")
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(b"x")
    fresh.write_bytes(b"y")
    _age(old, settings.trash_days * 86400 + 3600)
    _age(fresh, (settings.trash_days - 1) * 86400)
    out = worker.housekeeping(conn)
    assert out["failed"] == 1
    assert not old.exists() and fresh.is_file()


# ------------------------------------------------------------------ C18: failed uploads keep their bytes
def _error_upload(conn, uid: str) -> int:
    from app.db import tx
    from app.util import now_iso

    now = now_iso()
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO uploads(uid, original_filename, size_bytes, status, created_at, updated_at) "
            "VALUES (?, 'a.jpg', 1, 'error', ?, ?)", (uid, now, now))
    return int(cur.lastrowid)


def test_c18_cleanup_keeps_the_staged_file_of_a_failed_upload(conn):
    from app import ingest, storage

    _error_upload(conn, "errstaged234")
    staged = storage.staging_path("errstaged234")
    staged.write_bytes(b"only copy")
    _age(staged, 7200)
    ingest.cleanup_stale_uploads(conn)
    assert not staged.exists()
    assert storage.failed_path("errstaged234").read_bytes() == b"only copy"


def test_c18_mark_upload_failed_moves_bytes_before_the_status_commit(conn, monkeypatch):
    from app import ingest, storage
    from app.db import connect

    up = stage(conn, b"payload", "a.jpg")
    with conn:
        conn.execute("UPDATE uploads SET status = 'processing' WHERE id = ?", (up,))
    uid = upload_row(conn, up)["uid"]
    seen: list[str] = []
    real_move = storage.move_into_place

    def move(src, dst):
        other = connect()
        try:
            seen.append(other.execute("SELECT status FROM uploads WHERE id = ?", (up,)).fetchone()[0])
        finally:
            other.close()
        real_move(src, dst)

    monkeypatch.setattr(storage, "move_into_place", move)
    ingest.mark_upload_failed(conn, up, "Processing failed: x")
    assert seen == ["processing"]
    assert upload_row(conn, up)["status"] == "error"
    assert storage.failed_path(uid).read_bytes() == b"payload"


def test_c18_failed_move_leaves_bytes_that_housekeeping_keeps(conn, monkeypatch):
    from app import ingest, storage, worker

    up = stage(conn, b"payload", "a.jpg")
    uid = upload_row(conn, up)["uid"]

    def broken(src, dst):
        raise OSError(18, "Invalid cross-device link")

    with monkeypatch.context() as m:
        m.setattr(storage, "move_into_place", broken)
        ingest.mark_upload_failed(conn, up, "Processing failed: x")
    assert upload_row(conn, up)["status"] == "error"
    assert storage.staging_path(uid).read_bytes() == b"payload"
    # An ingest job for the errored upload must not delete the staged copy either.
    ingest.process_upload(conn, up)
    assert storage.staging_path(uid).is_file()
    worker.housekeeping(conn)
    assert storage.failed_path(uid).read_bytes() == b"payload"
    assert ingest.requeue_failed_upload(conn, up) is True
    assert storage.staging_path(uid).read_bytes() == b"payload"


def test_c18_ingest_resumes_from_failed_dir_after_a_crash_between_move_and_commit(conn, quiet_jobs):
    from app import ingest, storage

    up = stage(conn, jpeg_bytes(noise_image(160, 120, seed=21)), "resume.jpg")
    uid = upload_row(conn, up)["uid"]
    # mark_upload_failed moved the bytes, then the process died before the status commit.
    storage.move_into_place(storage.staging_path(uid), storage.failed_path(uid))
    ingest.process_upload(conn, up)
    assert upload_row(conn, up)["status"] == "stored"
    assert not storage.failed_path(uid).exists()


def test_c18_housekeeping_callers_share_one_lock(conn):
    from app import worker

    done = threading.Event()

    def run():
        from app.db import connect

        c = connect()
        try:
            worker.housekeeping(c)
        finally:
            c.close()
        done.set()

    with worker.housekeeping_lock:
        t = threading.Thread(target=run)
        t.start()
        assert not done.wait(0.3)  # blocked while another caller holds the lock
    t.join(10)
    assert done.is_set()


# ------------------------------------------------------------------ C19: document reprocess renders off the request thread
def test_c19_document_preview_reprocess_swaps_pages_in_the_background(conn, quiet_jobs, monkeypatch):
    from app import ingest, storage, worker
    from app.media import documents

    up = stage(conn, pdf_bytes([["page one"], ["page two"], ["page three"]]), "doc.pdf")
    assert run_jobs("main") >= 1
    item = item_of(conn, up)
    pages = storage.pages_dir(item["uid"], 1)
    before = {p.name: p.stat().st_ino for p in pages.iterdir()}
    assert sorted(before) == ["1.webp", "2.webp", "3.webp"]

    gate = threading.Event()
    checks: list[list[str]] = []
    real_render = documents.render_pages

    def slow_render(*args, **kwargs):
        out = real_render(*args, **kwargs)
        checks.append(sorted(p.name for p in pages.iterdir()))  # what a text job sees meanwhile
        gate.wait(10)
        return out

    monkeypatch.setattr(documents, "render_pages", slow_render)
    started = time.monotonic()
    ingest.enqueue_reprocess(conn, item["id"], ["preview"])
    assert time.monotonic() - started < 5  # the request does not wait for the render
    assert sorted(p.name for p in pages.iterdir()) == ["1.webp", "2.webp", "3.webp"]
    gate.set()
    worker.wait_background(30)

    assert checks == [["1.webp", "2.webp", "3.webp"]]
    after = {p.name: p.stat().st_ino for p in pages.iterdir()}
    assert sorted(after) == ["1.webp", "2.webp", "3.webp"]
    assert all(after[n] != before[n] for n in after)  # replaced by new renders
    # No scratch directory is left next to the real one.
    assert [d.name for d in pages.parent.iterdir()] == [pages.name]


def test_c19_swap_pages_never_leaves_a_page_missing(tmp_path, monkeypatch):
    from app import ingest

    final, scratch = tmp_path / "final", tmp_path / "scratch"
    final.mkdir()
    scratch.mkdir()
    for n in (1, 2, 3, 4):
        (final / f"{n}.webp").write_bytes(b"old")
    for n in (1, 2, 3):
        (scratch / f"{n}.webp").write_bytes(b"new")
    seen: list[set[str]] = []
    real_replace = os.replace

    def replace(src, dst):
        seen.append({p.name for p in final.iterdir()})
        real_replace(src, dst)

    monkeypatch.setattr(ingest.os, "replace", replace)
    assert ingest._swap_pages(scratch, final) == 3
    assert all(s >= {"1.webp", "2.webp", "3.webp", "4.webp"} for s in seen)
    assert [(final / f"{n}.webp").read_bytes() for n in (1, 2, 3, 4)] == [b"new", b"new", b"new", b"old"]


# ------------------------------------------------------------------ C24: startup recovery in one transaction
def test_c24_recovery_enqueues_in_one_transaction_without_duplicates(conn):
    from app import ingest, jobs
    from app.db import tx
    from app.util import new_uid, now_iso

    now = now_iso()
    with tx(conn):
        for i in range(300):
            uid = new_uid()
            conn.execute(
                "INSERT INTO items(uid, kind, version, original_filename, ext, mime, sha256, size_bytes, category, "
                "analysis_status, text_status, preview_status, created_at, updated_at) "
                "VALUES (?, 'image', 1, 'a.jpg', '.jpg', 'image/jpeg', ?, 1, 'misc', 'pending', 'none', 'none', ?, ?)",
                (uid, uid, now, now))
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        ingest.recover_on_startup(conn)
        first = sum(s.strip().upper().startswith("BEGIN") for s in statements)
        statements.clear()
        ingest.recover_on_startup(conn)
        second = sum(s.strip().upper().startswith("BEGIN") for s in statements)
    finally:
        conn.set_trace_callback(None)
    # requeue_running, the enqueue batch and (first start only) the guess window record of section 10:
    # never one per item
    assert first <= 3 and second <= 2
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE type = 'analyze'").fetchone()[0] == 300
    assert conn.execute("SELECT COUNT(DISTINCT payload) FROM jobs").fetchone()[0] == 300
    plan = " ".join(str(r[-1]) for r in conn.execute("EXPLAIN QUERY PLAN " + jobs._dup_sql("analyze"), ("a", "b")))
    assert "jobs_dedup" in plan


def test_c24_enqueue_unique_rules_are_unchanged(conn):
    from app import jobs

    assert jobs.enqueue(conn, "analyze", {"item_id": 1, "version": 1}, unique=True) is not None
    assert jobs.enqueue(conn, "analyze", {"item_id": 1, "version": 1}, unique=True) is None
    claimed = jobs.claim_next(conn, "main")
    assert claimed is not None
    # A running analyze job does not block a new one; a running recluster does.
    assert jobs.enqueue(conn, "analyze", {"item_id": 1, "version": 1}, unique=True) is not None
    assert jobs.enqueue_many(conn, "analyze", [{"item_id": 1, "version": 1}, {"item_id": 2, "version": 1}],
                             unique=True) == 1
    jobs.enqueue(conn, "recluster", {}, unique=True)
    with conn:
        conn.execute("UPDATE jobs SET status = 'running' WHERE type = 'recluster'")
    assert jobs.enqueue(conn, "recluster", {}, unique=True) is None


# ------------------------------------------------------------------ C28: the orphan sweep runs daily on its own thread
def test_c28_housekeeping_no_longer_walks_every_file(conn, monkeypatch):
    from app import ingest, storage, worker

    calls: list[int] = []
    monkeypatch.setattr(ingest, "sweep_orphan_files", lambda *a, **k: calls.append(1) or 0)
    orphan = storage.thumb_path("zzorphan2345", 1)
    storage.write_bytes_atomic(orphan, b"x")
    _age(orphan, 7200)
    worker.housekeeping(conn)
    assert calls == [] and orphan.is_file()


def test_c28_sweep_thread_waits_an_hour_then_runs_daily(data_dir, monkeypatch):
    from app import ingest, worker

    class FakeStop:
        def __init__(self, answers):
            self.answers = list(answers)
            self.waits: list[float] = []

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return self.answers.pop(0)

        def is_set(self):
            return False

    fake = FakeStop([False, False, True])
    sweeps: list[int] = []
    monkeypatch.setattr(worker, "_lower_priority", lambda: None)  # would renice the test process
    monkeypatch.setattr(worker, "_stop", fake)
    monkeypatch.setattr(ingest, "sweep_orphan_files", lambda conn, stop=None: sweeps.append(1) or 0)
    worker._sweep_loop()
    assert fake.waits == [worker.SWEEP_FIRST_DELAY_S, worker.SWEEP_INTERVAL_S, worker.SWEEP_INTERVAL_S]
    assert worker.SWEEP_FIRST_DELAY_S >= 3600 and worker.SWEEP_INTERVAL_S >= 86400
    assert sweeps == [1, 1]


def test_c28_sweep_stops_early_when_asked(conn):
    from app import ingest, storage

    orphan = storage.thumb_path("zzorphan2345", 1)
    storage.write_bytes_atomic(orphan, b"x")
    _age(orphan, 7200)
    stop = threading.Event()
    stop.set()
    assert ingest.sweep_orphan_files(conn, stop=stop) == 0
    assert orphan.is_file()
    assert ingest.sweep_orphan_files(conn) == 1


def test_c28_sweep_rechecks_live_uploads_before_deleting(conn, monkeypatch):
    """An upload that becomes active after the sweep's snapshot keeps its placed original."""
    from app import ingest, storage
    from app.db import tx
    from app.util import now_iso

    original = storage.original_path("lateup234567", 1, ".jpg")
    storage.write_bytes_atomic(original, b"x")
    _age(original, 7200)
    real_is_referenced = ingest._is_referenced

    def late_upload(c, path):
        # The upload row appears after the snapshot of active uploads was taken.
        if not c.execute("SELECT 1 FROM uploads WHERE uid = 'lateup234567'").fetchone():
            with tx(c):
                c.execute("INSERT INTO uploads(uid, original_filename, size_bytes, status, created_at, updated_at) "
                          "VALUES ('lateup234567', 'a.jpg', 1, 'processing', ?, ?)", (now_iso(), now_iso()))
        return real_is_referenced(c, path)

    monkeypatch.setattr(ingest, "_is_referenced", late_upload)
    assert ingest.sweep_orphan_files(conn) == 0
    assert original.is_file()
