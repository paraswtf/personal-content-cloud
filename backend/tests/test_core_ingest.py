"""Core pipeline: ingest of every kind, exact and near duplicates, trash and purge, housekeeping, worker."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from PIL import Image

from tests.helpers import jpeg_bytes, make_face, make_item, make_person, make_video, noise_image, pdf_bytes, run_jobs


def noop(conn, *args) -> None:
    """Stand-in handler for jobs of other areas (analysis, documents)."""


@pytest.fixture()
def quiet_jobs(monkeypatch):
    """Run only core's own handlers: analysis and documents jobs complete without doing anything."""
    from app import worker

    for kind in ("analyze", "text", "guess"):
        monkeypatch.setitem(worker.HANDLERS, kind, (__name__, "noop", worker.HANDLERS[kind][2]))
    monkeypatch.setitem(worker.HANDLERS, "recluster", (__name__, "noop", lambda p: ()))
    return worker


def stage(conn, data: bytes, filename: str, source: str = "web") -> int:
    """An upload whose bytes are complete and whose ingest job is queued."""
    from app import jobs, storage
    from app.db import tx
    from app.util import new_uid, now_iso

    uid = new_uid()
    storage.staging_path(uid).write_bytes(data)
    now = now_iso()
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO uploads(uid, source, original_filename, size_bytes, received_bytes, status, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
            (uid, source, filename, len(data), len(data), now, now),
        )
        upload_id = int(cur.lastrowid)
        jobs.enqueue(conn, "ingest", {"upload_id": upload_id})
    return upload_id


def upload_row(conn, upload_id: int):
    return conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()


def item_of(conn, upload_id: int):
    return conn.execute("SELECT i.* FROM items i JOIN uploads u ON u.item_id = i.id WHERE u.id = ?",
                        (upload_id,)).fetchone()


def job_rows(conn):
    return conn.execute("SELECT type, queue, payload, status, attempts FROM jobs ORDER BY id").fetchall()


# ------------------------------------------------------------------ ingest
def test_ingest_photo(conn, quiet_jobs):
    from app import storage

    data = jpeg_bytes(noise_image(800, 600))
    up = stage(conn, data, "holiday.jpg")
    from app.worker import drain

    # Stop after the ingest job so the follow-up jobs can be inspected.
    assert drain("main", max_jobs=1) == 1
    row = upload_row(conn, up)
    assert row["status"] == "stored"
    item = item_of(conn, up)
    assert item["uid"] == row["uid"]
    assert (item["kind"], item["ext"], item["width"], item["height"]) == ("image", ".jpg", 800, 600)
    assert (item["analysis_status"], item["text_status"], item["preview_status"]) == ("pending", "none", "none")
    assert item["phash"] and item["sig"]
    original = storage.original_path(item["uid"], 1, ".jpg")
    assert original.read_bytes() == data
    for p in (storage.thumb_path(item["uid"], 1), storage.preview_path(item["uid"], 1)):
        assert p.is_file()
    # 800 px already fits the small size: no small preview is stored, its URL serves the large file (section 3)
    assert not storage.preview_small_path(item["uid"], 1).exists()
    assert not storage.staging_path(row["uid"]).exists()
    payload = f'{{"item_id":{item["id"]},"version":1}}'
    assert [(j["type"], j["payload"]) for j in job_rows(conn)] == [("analyze", payload)]


def test_ingest_pdf_and_document(conn, quiet_jobs):
    from app import storage

    up = stage(conn, pdf_bytes([["Passport"], ["Page two"]]), "scan.pdf")
    assert run_jobs("main") >= 1
    item = item_of(conn, up)
    assert (item["kind"], item["page_count"], item["category"]) == ("pdf", 2, "documents")
    assert (item["analysis_status"], item["text_status"]) == ("pending", "pending")
    assert storage.page_path(item["uid"], 1, 1).is_file() and storage.page_path(item["uid"], 1, 2).is_file()
    assert storage.thumb_path(item["uid"], 1).is_file()
    types = [(j["type"], j["queue"]) for j in job_rows(conn)]
    assert ("text", "text") in types

    up2 = stage(conn, b"shopping list\nmilk\neggs\n", "list.txt")
    run_jobs()
    doc = item_of(conn, up2)
    assert (doc["kind"], doc["mime"], doc["ext"], doc["page_count"]) == ("document", "text/plain", ".txt", 1)
    assert storage.preview_path(doc["uid"], 1).is_file()


def test_ingest_unknown_file_is_kept(conn, quiet_jobs):
    from app import storage

    data = os.urandom(3000)
    up = stage(conn, data, "backup.tar.zst")
    run_jobs()
    item = item_of(conn, up)
    assert (item["kind"], item["mime"], item["ext"]) == ("other", "application/octet-stream", ".zst")
    assert item["analysis_status"] == "done" and item["text_status"] == "none"
    assert storage.original_path(item["uid"], 1, ".zst").read_bytes() == data
    assert not storage.thumb_path(item["uid"], 1).exists()


def test_document_that_fails_to_render_is_kept(conn, quiet_jobs, monkeypatch):
    from app import storage
    from app.media import documents

    def broken(*args, **kwargs):
        raise RuntimeError("pdfium could not render this page")

    monkeypatch.setattr(documents, "page_image", broken)
    monkeypatch.setattr(documents, "render_pages", broken)
    up = stage(conn, pdf_bytes([["one"]]), "odd.pdf")
    run_jobs()
    assert upload_row(conn, up)["status"] == "stored"
    item = item_of(conn, up)
    assert item["kind"] == "pdf" and not storage.thumb_path(item["uid"], 1).exists()
    assert storage.original_path(item["uid"], 1, ".pdf").is_file()


def test_ingest_empty_file_is_rejected(conn, quiet_jobs):
    up = stage(conn, b"", "empty.jpg")
    run_jobs()
    assert upload_row(conn, up)["status"] == "rejected"
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0


def test_ingest_video_and_streaming_copy(conn, quiet_jobs, tmp_path):
    from app import storage

    # High-bitrate 1080p: needs a compressed streaming copy.
    path = make_video(tmp_path / "big.mp4", size="1920x1080", seconds=2, crf=10)
    up = stage(conn, path.read_bytes(), "big.mp4")
    run_jobs()
    item = item_of(conn, up)
    assert item["kind"] == "video"
    assert item["frames"] and item["duration_s"] == pytest.approx(2.0, abs=0.2)
    assert storage.thumb_path(item["uid"], 1).is_file()
    assert item["preview_status"] == "done"
    assert storage.video_preview_path(item["uid"], 1).is_file()


def test_exact_duplicate_points_at_existing_item(conn, quiet_jobs):
    data = jpeg_bytes(noise_image(300, 200, seed=4))
    first = stage(conn, data, "a.jpg")
    run_jobs()
    item = item_of(conn, first)
    second = stage(conn, data, "b.jpg")
    run_jobs()
    row = upload_row(conn, second)
    assert row["status"] == "duplicate" and row["item_id"] == item["id"]
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1

    from app import ingest

    ingest.soft_delete(conn, item["id"])
    third = stage(conn, data, "c.jpg")
    run_jobs()
    row = upload_row(conn, third)
    assert row["status"] == "duplicate" and row["item_id"] == item["id"]
    assert "trash" in row["message"]


def test_near_duplicate_is_linked_not_replaced(conn, quiet_jobs, fixture_file):
    src = fixture_file("armstrong.jpg")
    img = Image.open(src).convert("RGB")
    first = stage(conn, src.read_bytes(), "armstrong.jpg")
    run_jobs()
    copy = stage(conn, jpeg_bytes(img.resize((396, 495)), quality=60), "armstrong-small.jpg")
    run_jobs()
    best, small = item_of(conn, first), item_of(conn, copy)
    assert small is not None and upload_row(conn, copy)["status"] == "stored"
    assert (best["dup_group_id"], best["dup_role"]) == (best["id"], "best")
    assert (small["dup_group_id"], small["dup_role"]) == (best["id"], "copy")
    assert "phash" in small["dup_detail"]

    # A bigger copy arriving later becomes the best.
    bigger = stage(conn, jpeg_bytes(img.resize((1080, 1350)), quality=95), "armstrong-big.jpg")
    run_jobs()
    big = item_of(conn, bigger)
    roles = {r["id"]: (r["dup_group_id"], r["dup_role"]) for r in conn.execute("SELECT * FROM items")}
    assert roles[big["id"]] == (big["id"], "best")
    assert roles[best["id"]] == (big["id"], "copy") and roles[small["id"]] == (big["id"], "copy")

    # A different person is not linked.
    other = stage(conn, fixture_file("aldrin.jpg").read_bytes(), "aldrin.jpg")
    run_jobs()
    assert item_of(conn, other)["dup_group_id"] is None


# ------------------------------------------------------------------ trash and purge
def test_soft_delete_restore_and_group_best(conn, quiet_jobs, fixture_file):
    from app import ingest

    src = fixture_file("armstrong.jpg")
    img = Image.open(src).convert("RGB")
    a_up = stage(conn, src.read_bytes(), "a.jpg")
    run_jobs()
    b_up = stage(conn, jpeg_bytes(img.resize((396, 495)), quality=60), "b.jpg")
    run_jobs()
    a, b = item_of(conn, a_up), item_of(conn, b_up)
    ingest.soft_delete(conn, a["id"])
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM items")}
    assert rows[a["id"]]["deleted_at"] is not None
    assert (rows[b["id"]]["dup_group_id"], rows[b["id"]]["dup_role"]) == (b["id"], "best")
    ingest.restore(conn, a["id"])
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM items")}
    assert rows[a["id"]]["deleted_at"] is None
    assert rows[a["id"]]["dup_role"] == "best" and rows[b["id"]]["dup_group_id"] == a["id"]


def test_purge_removes_rows_and_every_file(conn, quiet_jobs):
    from app import ingest, storage

    up = stage(conn, pdf_bytes([["one"], ["two"]]), "x.pdf")
    run_jobs()
    item = item_of(conn, up)
    uid = item["uid"]
    storage.write_bytes_atomic(storage.page_path(uid, 1, 2), b"page")
    storage.write_bytes_atomic(storage.doc_photo_path(uid, 1), b"photo")
    unnamed = make_person(conn)
    named = make_person(conn, "Asha")
    f1 = make_face(conn, item["id"], unnamed)
    f2 = make_face(conn, item["id"], named)
    from app.db import tx
    from app.util import now_iso

    with tx(conn):
        cur = conn.execute("INSERT INTO bodies(item_id, x, y, w, h, det_score, created_at) VALUES (?, 0, 0, 1, 1, 0.9, ?)",
                           (item["id"], now_iso()))
        body_id = int(cur.lastrowid)
        if not conn.execute("SELECT 1 FROM items_fts WHERE rowid = ?", (item["id"],)).fetchone():
            conn.execute("INSERT INTO items_fts(rowid, filename, text, doc, people, labels) "
                         "VALUES (?, 'x.pdf', '', '', '', '')", (item["id"],))
    storage.write_bytes_atomic(storage.body_crop_path(body_id), b"body")
    files = [storage.original_path(uid, 1, ".pdf"), storage.thumb_path(uid, 1), storage.preview_path(uid, 1),
             storage.preview_small_path(uid, 1), storage.page_path(uid, 1, 1), storage.page_path(uid, 1, 2),
             storage.doc_photo_path(uid, 1), storage.face_crop_path(f1), storage.face_crop_path(f2),
             storage.body_crop_path(body_id)]
    assert all(p.is_file() for p in files)
    fv = conn.execute("SELECT value FROM kv WHERE key = 'faces_version'").fetchone()[0]
    ingest.soft_delete(conn, item["id"])
    ingest.purge(conn, item["id"])
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM items_fts WHERE rowid = ?", (item["id"],)).fetchone()[0] == 0
    assert [r[0] for r in conn.execute("SELECT id FROM persons")] == [named]
    assert upload_row(conn, up)["item_id"] is None
    assert not any(p.exists() for p in files)
    assert not storage.pages_dir(uid, 1).exists()
    assert conn.execute("SELECT value FROM kv WHERE key = 'faces_version'").fetchone()[0] != fv
    # The id is never handed out again.
    new = make_item(conn)
    assert new["id"] > item["id"]


def test_expire_trash(conn):
    from app import ingest
    from app.config import settings
    from app.db import tx
    from app.util import now_iso

    old = make_item(conn)
    recent = make_item(conn)
    with tx(conn):
        conn.execute("UPDATE items SET deleted_at = ? WHERE id = ?", (now_iso(-(settings.trash_days + 1) * 86400), old["id"]))
        conn.execute("UPDATE items SET deleted_at = ? WHERE id = ?", (now_iso(-3600), recent["id"]))
    assert ingest.expire_trash(conn) == 1
    assert [r[0] for r in conn.execute("SELECT id FROM items")] == [recent["id"]]


# ------------------------------------------------------------------ housekeeping
def _age(path: Path, seconds: float = 7200) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


def test_sweep_keeps_referenced_files(conn):
    from app import ingest, storage

    item = make_item(conn)
    uid = item["uid"]
    face = make_face(conn, item["id"])
    kept = [storage.original_path(uid, 1, ".jpg"), storage.thumb_path(uid, 1), storage.preview_path(uid, 1),
            storage.preview_small_path(uid, 1), storage.face_crop_path(face)]
    storage.write_bytes_atomic(storage.page_path(uid, 1, 1), b"p")
    kept.append(storage.page_path(uid, 1, 1))
    orphans = [storage.original_path("zzorphan2345", 1, ".jpg"), storage.thumb_path("zzorphan2345", 1),
               storage.original_path(uid, 2, ".jpg"), storage.original_path(uid, 1, ".png"),
               storage.page_path("zzorphan2345", 1, 3), storage.face_crop_path(999999),
               storage.body_crop_path(5), storage.root() / "videos" / "zz" / "junk.tmp"]
    for p in orphans:
        storage.write_bytes_atomic(p, b"x")
    fresh = storage.thumb_path("zzfresh23456", 1)
    storage.write_bytes_atomic(fresh, b"x")
    for p in kept + orphans:
        _age(p)
    removed = ingest.sweep_orphan_files(conn)
    assert removed == len(orphans)
    assert all(p.is_file() for p in kept)
    assert not any(p.exists() for p in orphans)
    assert fresh.is_file()  # younger than an hour


def test_cleanup_stale_uploads(conn):
    from app import ingest, storage
    from app.config import settings
    from app.db import tx
    from app.util import now_iso

    old = now_iso(-(settings.staging_ttl_hours + 1) * 3600)
    with tx(conn):
        conn.execute("INSERT INTO uploads(uid, original_filename, size_bytes, status, created_at, updated_at) "
                     "VALUES ('stale2345678', 'a.jpg', 10, 'uploading', ?, ?)", (old, old))
        conn.execute("INSERT INTO uploads(uid, original_filename, size_bytes, status, created_at, updated_at) "
                     "VALUES ('live23456789', 'b.jpg', 10, 'uploading', ?, ?)", (now_iso(), now_iso()))
    for uid in ("stale2345678", "live23456789", "norow2345678"):
        storage.staging_path(uid).write_bytes(b"x")
    _age(storage.staging_path("norow2345678"))
    ingest.cleanup_stale_uploads(conn)
    assert conn.execute("SELECT status FROM uploads WHERE uid = 'stale2345678'").fetchone()[0] == "cancelled"
    assert not storage.staging_path("stale2345678").exists()
    assert not storage.staging_path("norow2345678").exists()
    assert storage.staging_path("live23456789").exists()


def test_recover_on_startup(conn):
    from app import ingest, storage
    from app.db import tx
    from app.util import now_iso

    pending = make_item(conn, analysis_status="pending")
    make_item(conn, kind="pdf", ext=".pdf", mime="application/pdf", text_status="pending", write_files=False)
    with tx(conn):
        conn.execute("INSERT INTO jobs(type, queue, payload, status, created_at, updated_at) "
                     "VALUES ('ingest', 'main', '{\"upload_id\":99}', 'running', ?, ?)", (now_iso(), now_iso()))
    (storage.root() / "tmp" / "leftover").write_bytes(b"x")
    ingest.recover_on_startup(conn)
    rows = job_rows(conn)
    assert all(r["status"] == "queued" for r in rows)
    types = sorted(r["type"] for r in rows)
    assert types == ["analyze", "ingest", "text"]
    assert any(f'"item_id":{pending["id"]}' in r["payload"] for r in rows)
    assert not (storage.root() / "tmp" / "leftover").exists()


def test_enqueue_reprocess_and_text(conn, quiet_jobs):
    from app import ingest, storage

    up = stage(conn, jpeg_bytes(noise_image(400, 300, seed=9)), "p.jpg")
    run_jobs()
    item = item_of(conn, up)
    storage.safe_unlink(storage.thumb_path(item["uid"], 1))
    ingest.enqueue_reprocess(conn, item["id"], ["analyze", "text", "preview", "bogus"])
    item = item_of(conn, up)
    assert item["analysis_status"] == "pending" and item["text_status"] == "pending"
    assert storage.thumb_path(item["uid"], 1).is_file()
    assert sorted(r["type"] for r in job_rows(conn)) == ["analyze", "text"]


# ------------------------------------------------------------------ worker
def test_missing_handler_fails_without_retry(conn, monkeypatch):
    from app import jobs, worker

    monkeypatch.setitem(worker.HANDLERS, "analyze",
                        ("app.module_that_does_not_exist", "analyze_item", worker.HANDLERS["analyze"][2]))
    item = make_item(conn, analysis_status="pending")
    jobs.enqueue(conn, "analyze", {"item_id": item["id"], "version": 1})
    assert worker.drain() == 1
    job = conn.execute("SELECT status, attempts, last_error FROM jobs").fetchone()
    assert job["status"] == "error" and job["attempts"] == 1
    assert "not available" in job["last_error"]
    # The item stays pending so a later start can queue it again.
    assert conn.execute("SELECT analysis_status FROM items").fetchone()[0] == "pending"


def test_handler_error_retries_then_marks_item(conn, monkeypatch):
    from app import jobs, worker
    from app.db import tx

    calls = []

    def boom(c, item_id, version):
        calls.append(item_id)
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(__import__(__name__, fromlist=["x"]), "boom", boom, raising=False)
    monkeypatch.setitem(worker.HANDLERS, "analyze", (__name__, "boom", worker.HANDLERS["analyze"][2]))
    item = make_item(conn, analysis_status="pending")
    jobs.enqueue(conn, "analyze", {"item_id": item["id"], "version": 1})
    for _ in range(jobs.MAX_ATTEMPTS):
        worker.drain()
        with tx(conn):
            conn.execute("UPDATE jobs SET run_after = NULL")
    assert len(calls) == jobs.MAX_ATTEMPTS
    assert conn.execute("SELECT status FROM jobs").fetchone()[0] == "error"
    assert conn.execute("SELECT analysis_status FROM items").fetchone()[0] == "error"


def test_disk_full_defers_without_using_an_attempt(conn, monkeypatch):
    import errno

    from app import jobs, worker

    def full(c, item_id, version):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(__import__(__name__, fromlist=["x"]), "full", full, raising=False)
    monkeypatch.setitem(worker.HANDLERS, "analyze", (__name__, "full", worker.HANDLERS["analyze"][2]))
    item = make_item(conn, analysis_status="pending")
    jobs.enqueue(conn, "analyze", {"item_id": item["id"], "version": 1})
    worker.drain()
    job = conn.execute("SELECT status, attempts, run_after FROM jobs").fetchone()
    assert job["status"] == "queued" and job["attempts"] == 0 and job["run_after"]


def test_drain_order_and_guess_payload(conn, monkeypatch):
    from app import jobs, worker

    seen: list[tuple] = []

    def record(kind):
        def handler(c, *args):
            seen.append((kind, *args))
        return handler

    module = __import__(__name__, fromlist=["x"])
    for kind in ("analyze", "text", "guess"):
        monkeypatch.setattr(module, f"rec_{kind}", record(kind), raising=False)
        monkeypatch.setitem(worker.HANDLERS, kind, (__name__, f"rec_{kind}", worker.HANDLERS[kind][2]))
    jobs.enqueue(conn, "text", {"item_id": 1, "version": 1})
    jobs.enqueue(conn, "guess", {"person_id": 7, "item_id": None})
    jobs.enqueue(conn, "analyze", {"item_id": 2, "version": 3})
    assert worker.drain() == 3
    assert seen == [("analyze", 2, 3), ("guess", 7, None), ("text", 1, 1)]
