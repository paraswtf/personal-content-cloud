"""Storage tiers (docs/ARCHITECTURE.md section 31): ORIGINALS_DIR holds the originals, staging and failed
uploads; DATA_DIR the database and the derived files; files of the earlier single-folder layout are still
found and are moved over."""
from __future__ import annotations

import errno
import os
import threading
import time
from pathlib import Path

import pytest

from tests.helpers import jpeg_bytes, make_item, noise_image, run_jobs, upload_bytes


def _retier(originals_dir: Path | None, **extra):
    """reload_settings with the data_dir fixture's overrides plus a second root."""
    from app import storage
    from app.config import reload_settings, settings

    reload_settings(data_dir=settings.data_dir, frontend_dist=settings.frontend_dist, workers_enabled=False,
                    min_free_disk_bytes=0, chunk_size=settings.chunk_size, session_days=30,
                    originals_dir=originals_dir, **extra)
    storage.ensure_dirs()


@pytest.fixture()
def tiered(data_dir, tmp_path):
    fast = tmp_path / "fast"
    _retier(fast)
    return fast


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _age(path: Path, seconds: float = 7200) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


def test_paths_split_between_the_roots(tiered, tmp_path):
    from app import storage
    from app.config import settings

    data = settings.data_dir
    assert storage.tiered() and storage.originals_root() == tiered and storage.root() == data
    assert storage.original_path("abcdef", 1, ".jpg") == tiered / "originals" / "ab" / "abcdef.v1.jpg"
    assert storage.staging_path("u1") == tiered / "staging" / "u1.part"
    assert storage.failed_path("u1") == tiered / "failed" / "u1.part"
    assert storage.thumb_path("abcdef", 1) == data / "thumbs" / "ab" / "abcdef.v1.webp"
    assert storage.video_preview_path("abcdef", 1) == data / "videos" / "ab" / "abcdef.v1.mp4"
    assert all((tiered / d).is_dir() for d in storage.TIER_DIRS)
    assert all((data / d).is_dir() for d in storage.DERIVED_DIRS)
    assert storage.rel(tiered / "originals" / "ab" / "x.jpg") == "originals/ab/x.jpg"
    assert storage.rel(data / "thumbs" / "ab" / "x.webp") == "thumbs/ab/x.webp"
    assert storage.from_rel("originals/ab/x.jpg") == (tiered / "originals" / "ab" / "x.jpg").resolve()
    assert storage.from_rel("thumbs/ab/x.webp") == (data / "thumbs" / "ab" / "x.webp").resolve()
    with pytest.raises(ValueError):
        storage.rel(tmp_path / "elsewhere" / "x.jpg")
    tiers = storage.tiers()
    assert [t["key"] for t in tiers] == ["originals", "data"]
    assert tiers[0]["path"] == str(tiered) and tiers[0]["total_bytes"] > 0
    assert tiers[1]["shared_with"] == "originals"  # one disk in the test, counted once
    assert [(top, base) for top, base in storage.swept_bases() if top == "originals"] == [
        ("originals", tiered / "originals"), ("originals", data / "originals")]


def test_the_same_folder_is_one_tier(data_dir):
    from app import storage
    from app.config import settings

    _retier(data_dir)
    assert settings.originals_dir is None and not storage.tiered()
    assert storage.original_path("abcdef", 1, ".jpg") == data_dir / "originals" / "ab" / "abcdef.v1.jpg"
    assert [t["key"] for t in storage.tiers()] == ["originals"]
    assert storage.legacy_dir("originals") is None
    assert storage.swept_bases() == [(top, data_dir / top) for top in storage.SWEPT_DIRS]


def test_original_path_falls_back_to_a_legacy_file(tiered):
    from app import storage
    from app.config import settings

    legacy = _write(settings.data_dir / "originals" / "ab" / "abcdef.v1.jpg", b"legacy")
    assert storage.original_path("abcdef", 1, ".jpg") == legacy
    tier = _write(tiered / "originals" / "ab" / "abcdef.v1.jpg", b"tier")
    assert storage.original_path("abcdef", 1, ".jpg") == tier
    assert storage.original_path("abcdef", 2, ".jpg") == tiered / "originals" / "ab" / "abcdef.v2.jpg"


def test_move_into_place_copies_across_filesystems(tiered, monkeypatch):
    from app import storage
    from app.config import settings

    src = _write(settings.data_dir / "tmp" / "a.bin", b"hello" * 1000)
    dst = tiered / "originals" / "ab" / "abc.v1.bin"
    real = os.replace
    first = {"done": False}

    def cross_device(a, b):
        if not first["done"]:
            first["done"] = True
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real(a, b)

    monkeypatch.setattr(storage.os, "replace", cross_device)
    storage.move_into_place(src, dst)
    assert dst.read_bytes() == b"hello" * 1000 and not src.exists()
    assert [p.name for p in dst.parent.iterdir()] == [dst.name]  # no scratch file left behind


def test_an_upload_lands_on_the_originals_root(tiered, client):
    from app import storage
    from app.config import settings
    from app.db import connect

    data = jpeg_bytes(noise_image(320, 240, seed=3))
    up = upload_bytes(client, data, "a.jpg")
    assert up["status"] == "queued", up
    assert not any((settings.data_dir / "staging").glob("*")) if (settings.data_dir / "staging").exists() else True
    run_jobs()
    conn = connect()
    try:
        row = conn.execute("SELECT uid, version, ext FROM items").fetchone()
    finally:
        conn.close()
    assert row is not None
    original = storage.original_path(row["uid"], int(row["version"]), row["ext"])
    assert original.is_file() and original.is_relative_to(tiered) and original.read_bytes() == data
    thumb = storage.thumb_path(row["uid"], int(row["version"]))
    assert thumb.is_file() and thumb.is_relative_to(settings.data_dir)
    legacy = settings.data_dir / "originals"
    assert not legacy.exists() or not any(legacy.rglob("*"))
    assert not any((tiered / "staging").iterdir())
    r = client.get(f"/files/original/{row['uid']}")
    assert r.status_code == 200 and r.content == data


def test_move_staged_and_failed_uploads_at_startup(tiered):
    from app import tiers
    from app.config import settings

    legacy_staging = settings.data_dir / "staging"
    legacy_failed = settings.data_dir / "failed"
    _write(legacy_staging / "u1.part", b"a" * 10)
    _write(legacy_staging / "u2.part", b"old")
    _write(tiered / "staging" / "u2.part", b"newer")  # the new layout wins
    _write(legacy_failed / "f1.part", b"f")
    assert tiers.move_staged() == 3
    assert (tiered / "staging" / "u1.part").read_bytes() == b"a" * 10
    assert (tiered / "staging" / "u2.part").read_bytes() == b"newer"
    assert (tiered / "failed" / "f1.part").read_bytes() == b"f"
    assert not legacy_staging.exists() and not legacy_failed.exists()
    assert tiers.move_staged() == 0


def test_migrate_originals_moves_copies_and_removes_the_legacy_folder(tiered):
    from app import tiers
    from app.config import settings

    legacy = settings.data_dir / "originals"
    a = _write(legacy / "aa" / "aaaaaaaa.v1.jpg", b"A" * 100)
    b = _write(legacy / "bb" / "bbbbbbbb.v1.jpg", b"B" * 50)
    tier_b = _write(tiered / "originals" / "bb" / "bbbbbbbb.v1.jpg", b"B" * 50)  # copied by hand before the switch
    os.utime(tier_b, (b.stat().st_atime, b.stat().st_mtime))
    c = _write(legacy / "cc" / "cccccccc.v1.jpg", b"C" * 70)
    tier_c = _write(tiered / "originals" / "cc" / "cccccccc.v1.jpg", b"short")  # a damaged copy: replaced
    ino_b = tier_b.stat().st_ino
    assert tiers.status()["active"] is True

    st = tiers.migrate_originals()
    assert st["running"] is False and st["active"] is False and st["error"] is None
    assert st["pending_files"] == 3 and st["moved_files"] == 3 and st["moved_bytes"] == 100 + 50 + 70
    assert (tiered / "originals" / "aa" / "aaaaaaaa.v1.jpg").read_bytes() == b"A" * 100
    assert tier_b.stat().st_ino == ino_b and tier_b.read_bytes() == b"B" * 50
    assert tier_c.read_bytes() == b"C" * 70
    assert not a.exists() and not b.exists() and not c.exists() and not legacy.exists()
    assert tiers.start() is False  # nothing left to move
    assert tiers.status()["active"] is False


def test_migration_stops_early_and_resumes(tiered):
    from app import tiers
    from app.config import settings

    legacy = settings.data_dir / "originals"
    _write(legacy / "aa" / "aaaaaaaa.v1.jpg", b"A")
    stop = threading.Event()
    stop.set()
    st = tiers.migrate_originals(stop)
    assert st["moved_files"] == 0 and st["active"] is True and legacy.is_dir()
    st = tiers.migrate_originals()
    assert st["moved_files"] == 1 and st["active"] is False and not legacy.exists()


def test_the_sweeper_walks_both_roots(tiered, conn):
    from app import ingest, storage
    from app.config import settings

    item = make_item(conn)
    uid = item["uid"]
    kept = [storage.original_path(uid, 1, ".jpg"),  # on the tier
            _write(settings.data_dir / "originals" / uid[:2] / f"{uid}.v1.jpg", b"legacy copy of a live item")]
    orphans = [_write(settings.data_dir / "originals" / "zz" / "zzorphan2345.v1.jpg", b"x"),
               _write(storage.original_path("zzorphan2345", 1, ".png"), b"x")]
    for p in kept + orphans:
        _age(p)
    assert ingest.sweep_orphan_files(conn) == 2
    assert all(p.is_file() for p in kept)
    assert not any(p.exists() for p in orphans)


def test_has_room_checks_both_disks(tiered, monkeypatch):
    from app import storage
    from app.config import settings

    settings.min_free_disk_bytes = 1000
    free = {"tier": 5000, "data": 5000}

    def usage(path=None):
        base = path if path is not None else storage.originals_root()
        return (free["tier"] if base == tiered else free["data"]), 10_000

    monkeypatch.setattr(storage, "disk_usage", usage)
    assert storage.has_room_for(3000) and not storage.has_room_for(4500)
    free["data"] = 500  # the data disk is too full for the derived files
    assert not storage.has_room_for(10)


def test_system_reports_the_tiers(tiered, client):
    r = client.get("/api/system")
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["storage"]["tiered"] is True
    assert [t["key"] for t in s["storage"]["tiers"]] == ["originals", "data"]
    assert s["storage"]["migration"]["active"] is False and s["storage"]["migration"]["running"] is False
    assert s["backup"]["enabled"] is False and s["backup"]["snapshots"] == []
    r = client.post("/api/system/backup")
    assert r.status_code == 200 and r.json() == {"started": False, "running": False}
