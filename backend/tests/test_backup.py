"""Nightly backups (docs/ARCHITECTURE.md section 31): dated snapshots under BACKUP_DIR with the database
first, the originals it references (hard links for unchanged ones), retention, the schedule and the API."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.helpers import make_item


@pytest.fixture()
def backup_dir(data_dir, tmp_path):
    from app import storage
    from app.config import reload_settings, settings

    b = tmp_path / "hdd"
    reload_settings(data_dir=settings.data_dir, frontend_dist=settings.frontend_dist, workers_enabled=False,
                    min_free_disk_bytes=0, chunk_size=settings.chunk_size, session_days=30,
                    originals_dir=tmp_path / "fast", backup_dir=b, backup_keep=2, backup_hour=3)
    storage.ensure_dirs()
    return b


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _rel(item) -> Path:
    return Path(item["uid"][:2]) / f"{item['uid']}.v1{item['ext']}"


def _item(conn, data: bytes, **kw):
    """An item whose original holds these bytes (kept as a plain file: nothing is decoded)."""
    kw.setdefault("ext", ".jpg")
    kw.setdefault("mime", "image/jpeg")
    return make_item(conn, kind=kw.pop("kind", "other"), data=data, analysis_status="done", **kw)


def test_snapshot_has_everything_and_links_unchanged_files(backup_dir, conn):
    from app import backup, storage
    from app.config import settings
    from app.db import kv_set, tx

    a = _item(conn, b"A" * 1000)
    b = _item(conn, b"B" * 2000, ext=".mov", mime="video/quicktime")
    _write(storage.originals_root() / "originals" / "zz" / "zzorphan.v1.jpg", b"not an item")  # not referenced
    _write(settings.data_dir / "secret.key", b"k" * 32)
    _write(settings.data_dir / "learned" / "model.bin", b"m")
    _write(storage.wallet_dir() / "cert.pem", b"c")
    with tx(conn):
        kv_set(conn, "marker", "1")

    rec = backup.run(conn, "manual")
    assert rec["ok"] is True, rec
    snap = backup_dir / "snapshots" / rec["snapshot"]
    assert snap.is_dir() and (backup_dir / "latest").resolve() == snap.resolve()
    assert (snap / "originals" / _rel(a)).read_bytes() == b"A" * 1000
    assert (snap / "originals" / _rel(b)).read_bytes() == b"B" * 2000
    assert not (snap / "originals" / "zz").exists()
    assert (snap / "secret.key").read_bytes() == b"k" * 32
    assert (snap / "learned" / "model.bin").is_file() and (snap / "wallet" / "cert.pem").is_file()
    assert not (snap / "thumbs").exists()  # derived files are rebuilt, not backed up
    copy = sqlite3.connect(snap / "cloud.db")
    try:
        assert copy.execute("SELECT value FROM kv WHERE key = 'marker'").fetchone()[0] == "1"
        assert copy.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert copy.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2
    finally:
        copy.close()
    assert rec["files"] == 2 and rec["missing"] == 0 and rec["bytes_copied"] == 3000 and rec["bytes_linked"] == 0
    assert rec["reason"] == "manual" and rec["duration_s"] >= 0
    assert backup.last_record(conn)["snapshot"] == rec["snapshot"]

    c = _item(conn, b"C" * 300)
    rec2 = backup.run(conn, "scheduled")
    assert rec2["ok"] and rec2["snapshot"] != rec["snapshot"]
    snap2 = backup_dir / "snapshots" / rec2["snapshot"]
    assert rec2["files"] == 3 and rec2["bytes_copied"] == 300 and rec2["bytes_linked"] == 3000
    a1, a2 = snap / "originals" / _rel(a), snap2 / "originals" / _rel(a)
    assert a1.stat().st_ino == a2.stat().st_ino and a2.stat().st_nlink == 2
    assert (snap2 / "originals" / _rel(c)).stat().st_nlink == 1
    assert (backup_dir / "latest").resolve() == snap2.resolve()


def test_an_original_deleted_during_the_run_comes_from_the_previous_snapshot(backup_dir, conn):
    from app import backup, storage

    a = _item(conn, b"A" * 10)
    b = _item(conn, b"B" * 10)
    first = backup.run(conn)
    assert first["ok"] and first["files"] == 2
    os.unlink(storage.original_path(a["uid"], 1, ".jpg"))  # the row is still in the database copy
    storage.write_bytes_atomic(storage.original_path(b["uid"], 1, ".jpg"), b"B" * 10)  # rewritten: newer mtime
    os.utime(storage.original_path(b["uid"], 1, ".jpg"), (1, 1))
    rec = backup.run(conn)
    assert rec["ok"] and rec["files"] == 2 and rec["missing"] == 0
    snap = backup_dir / "snapshots" / rec["snapshot"]
    assert (snap / "originals" / _rel(a)).read_bytes() == b"A" * 10
    c = _item(conn, b"C")
    os.unlink(storage.original_path(c["uid"], 1, ".jpg"))
    rec = backup.run(conn)
    assert rec["ok"] and rec["missing"] == 1 and rec["files"] == 2


def test_retention_keeps_the_newest_snapshots(backup_dir, conn):
    from app import backup

    names = [backup.run(conn)["snapshot"] for _ in range(3)]
    kept = sorted(p.name for p in (backup_dir / "snapshots").iterdir())
    assert kept == sorted(names[1:])  # backup_keep = 2
    st = backup.status(conn)
    assert st["snapshots"] == [names[2], names[1]] and st["enabled"] and st["running"] is False
    assert st["last"]["snapshot"] == names[2] and st["next_run_at"] and st["total_bytes"] > 0


def test_a_failed_run_leaves_no_partial_snapshot(backup_dir, conn, monkeypatch):
    from app import backup

    (backup_dir / "snapshots" / ".partial-old").mkdir(parents=True)

    def boom(dst):
        raise RuntimeError("boom")

    monkeypatch.setattr(backup, "_snapshot_db", boom)
    rec = backup.run(conn)
    assert rec["ok"] is False and "boom" in rec["error"] and rec["snapshot"]
    assert list((backup_dir / "snapshots").iterdir()) == []
    assert not (backup_dir / "latest").exists()
    assert backup.last_record(conn)["ok"] is False
    assert backup.status(conn)["last"]["error"] == rec["error"]


def test_legacy_originals_not_yet_moved_are_included(backup_dir, conn):
    from app import backup, storage
    from app.config import settings

    d = _item(conn, b"D")
    tier = storage.original_path(d["uid"], 1, ".jpg")
    legacy = settings.data_dir / "originals" / d["uid"][:2] / tier.name
    legacy.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tier, legacy)
    e = _item(conn, b"E")
    rec = backup.run(conn)
    assert rec["ok"] and rec["files"] == 2 and rec["missing"] == 0
    snap = backup_dir / "snapshots" / rec["snapshot"]
    assert (snap / "originals" / _rel(d)).read_bytes() == b"D"
    assert (snap / "originals" / _rel(e)).read_bytes() == b"E"


def test_next_run_schedule(backup_dir):
    from app import backup

    def at(h, m=0):
        return datetime(2026, 11, 8, h, m, tzinfo=timezone.utc)

    assert backup.next_run(at(1), None, at(0)) == at(3)  # ahead today
    assert backup.next_run(at(5), at(3, 10), at(0)) == at(3) + timedelta(days=1)  # done this morning
    assert backup.next_run(at(5), None, at(0)) == at(3) + timedelta(days=1)  # never ran: tonight
    missed = at(5) - timedelta(hours=30)
    delay = timedelta(seconds=backup.CATCH_UP_DELAY_S)
    assert backup.next_run(at(5), missed, at(4, 58)) == at(4, 58) + delay  # a missed night: soon after start
    assert backup.next_run(at(5), missed, at(4)) == at(5)  # the delay has passed: now


def test_back_up_now_runs_on_its_own_thread_and_the_api_reports_it(backup_dir, client, conn):
    from app import backup

    _item(conn, b"F")
    r = client.post("/api/system/backup")
    assert r.status_code == 200 and r.json()["started"] is True, r.text
    backup.wait(60)
    s = client.get("/api/system").json()["backup"]
    assert s["enabled"] and s["running"] is False and s["last"]["ok"] is True, s
    assert s["last"]["reason"] == "manual" and s["last"]["files"] == 1 and len(s["snapshots"]) == 1
    assert s["dir"] == str(backup_dir) and s["keep"] == 2 and s["hour"] == 3 and s["next_run_at"]
    assert s["total_bytes"] > 0 and s["progress"] is None
