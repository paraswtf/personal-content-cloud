"""Derived files the app no longer stores twice (docs/ARCHITECTURE.md sections 3 and 20): no small preview
when the large one already fits the small size, no thumb or previews for Live Photo clips (their pictures
are the still's), pet crops removed with their item and swept as orphans, and the one-time tidy pass."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest
from PIL import Image

from tests.helpers import jpeg_bytes, make_item, noise_image, run_jobs, upload_bytes, upload_status


def test_no_small_preview_when_the_large_one_already_fits(data_dir, tmp_path):
    from app.media import images

    small_img = noise_image(1000, 700, seed=1)
    thumb, large, small = tmp_path / "t.webp", tmp_path / "p.webp", tmp_path / "p.sm.webp"
    small.write_bytes(b"stale")
    images.make_image_derivatives(small_img, thumb, large, small)
    assert large.is_file() and thumb.is_file() and not small.exists()
    with Image.open(large) as im:
        assert im.size == (1000, 700)

    big = noise_image(3000, 2000, seed=2)
    images.make_image_derivatives(big, thumb, large, small)
    with Image.open(large) as im, Image.open(small) as sm:
        assert max(im.size) == 2048 and max(sm.size) == 1280
    assert images.small_is_large((1280, 900)) and not images.small_is_large((1281, 900))


def test_the_small_url_serves_the_large_file_when_none_is_stored(client, conn):
    from app import storage

    item = make_item(conn, width=900, height=600)
    uid = item["uid"]
    storage.preview_small_path(uid, 1).unlink()
    r = client.get(f"/files/preview-small/{uid}", params={"v": "1"})
    assert r.status_code == 200
    assert r.content == storage.preview_path(uid, 1).read_bytes()
    assert "immutable" in r.headers["cache-control"]
    assert not storage.preview_small_path(uid, 1).exists()  # nothing made on the side

    big = make_item(conn, width=1600, height=1200)
    storage.preview_small_path(big["uid"], 1).unlink()
    r = client.get(f"/files/preview-small/{big['uid']}", params={"v": "1"})
    assert r.status_code == 200
    with Image.open(storage.preview_small_path(big["uid"], 1)) as sm:  # a large picture still gets one, made now
        assert max(sm.size) == 1280


def _clip(path: Path, seconds: float = 2.0) -> bytes:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=640x480:rate=25", "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(path)], check=True, timeout=120)
    return path.read_bytes()


@pytest.mark.parametrize("clip_first", [False, True])
def test_a_live_photo_clip_keeps_no_pictures_of_its_own(client, conn, tmp_path, clip_first):
    from app import storage

    still_bytes = jpeg_bytes(noise_image(1600, 1200, seed=5))
    clip_bytes = _clip(tmp_path / "IMG_0005.mov")
    uploads = [("IMG_0005.MOV", clip_bytes), ("IMG_0005.JPG", still_bytes)]
    if not clip_first:
        uploads.reverse()
    ids = {}
    for name, data in uploads:
        up = upload_bytes(client, data, name)
        assert up["status"] == "queued", up
        ids[name] = up["id"]
        run_jobs()
    still = conn.execute("SELECT * FROM items WHERE original_filename = 'IMG_0005.JPG'").fetchone()
    clip = conn.execute("SELECT * FROM items WHERE original_filename = 'IMG_0005.MOV'").fetchone()
    assert clip["companion_of"] == still["id"] and still["live_video_item_id"] == clip["id"]
    for p in (storage.thumb_path(clip["uid"], 1), storage.preview_path(clip["uid"], 1),
              storage.preview_small_path(clip["uid"], 1)):
        assert not p.exists(), p
    assert storage.thumb_path(still["uid"], 1).is_file()
    assert storage.video_preview_path(clip["uid"], 1).is_file() or clip["preview_status"] == "none"  # the clip itself stays
    # every picture URL of the clip shows the still
    for route, path in (("thumb", storage.thumb_path), ("preview", storage.preview_path)):
        r = client.get(f"/files/{route}/{clip['uid']}", params={"v": "1"})
        assert r.status_code == 200 and r.content == path(still["uid"], 1).read_bytes()
    assert client.get(f"/files/preview-small/{clip['uid']}").status_code == 200
    # the upload panel shows the still for the clip's upload
    clip_up = upload_status(client, ids["IMG_0005.MOV"])
    assert clip_up["item_uid"] == clip["uid"]
    assert still["uid"] in clip_up["item_thumb_url"] and clip["uid"] not in clip_up["item_thumb_url"]
    # a reprocess never brings them back
    from app import ingest

    ingest.enqueue_reprocess(conn, int(clip["id"]), ["preview"])
    run_jobs()
    assert not storage.thumb_path(clip["uid"], 1).exists()


def _animal(conn, item_id: int) -> int:
    from app.db import tx
    from app.util import now_iso

    with tx(conn):
        return int(conn.execute(
            "INSERT INTO animals(item_id, assigned_by, x, y, w, h, det_score, species, created_at) "
            "VALUES (?, 'auto', 0.1, 0.1, 0.4, 0.4, 0.9, 'cat', ?)", (item_id, now_iso())).lastrowid)


def test_purge_removes_pet_crops_and_the_sweeper_sweeps_orphans(conn):
    from app import ingest, storage

    item = make_item(conn)
    aid = _animal(conn, item["id"])
    crop = storage.animal_crop_path(aid)
    storage.write_bytes_atomic(crop, b"crop")
    ingest.purge(conn, int(item["id"]))
    assert not crop.exists()

    keep_item = make_item(conn)
    kept = storage.animal_crop_path(_animal(conn, keep_item["id"]))
    storage.write_bytes_atomic(kept, b"kept")
    orphan = storage.animal_crop_path(999_999)
    storage.write_bytes_atomic(orphan, b"orphan")
    old = time.time() - 7200
    for p in (kept, orphan):
        os.utime(p, (old, old))
    assert ingest.sweep_orphan_files(conn) == 1
    assert kept.is_file() and not orphan.exists()


def test_tidy_removes_what_the_app_no_longer_keeps_once(conn):
    from app import ingest, storage, worker
    from app.db import kv_get, tx

    small = make_item(conn, width=640, height=480)  # make_item writes all three pictures at full size
    big = make_item(conn, width=1600, height=1200)
    still = make_item(conn, width=1600, height=1200)
    clip = make_item(conn, kind="video", ext=".mov", mime="video/quicktime", data=b"x" * 100, width=640, height=480)
    with tx(conn):
        conn.execute("UPDATE items SET companion_of = ? WHERE id = ?", (still["id"], clip["id"]))
        conn.execute("UPDATE items SET live = 1, live_video_item_id = ? WHERE id = ?", (clip["id"], still["id"]))
    img = noise_image(640, 480, seed=9)
    for p in (storage.thumb_path(clip["uid"], 1), storage.preview_path(clip["uid"], 1),
              storage.preview_small_path(clip["uid"], 1)):
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p, "WEBP")

    removed = worker.housekeeping(conn)["tidied"]
    assert removed == 4  # the clip's three pictures and the small image's repeated small preview
    assert not storage.preview_small_path(small["uid"], 1).exists()
    assert storage.preview_path(small["uid"], 1).is_file() and storage.thumb_path(small["uid"], 1).is_file()
    assert storage.preview_small_path(big["uid"], 1).is_file()  # 1600 px: a real small preview, kept
    assert not storage.thumb_path(clip["uid"], 1).exists()
    assert storage.thumb_path(still["uid"], 1).is_file()
    assert kv_get(conn, ingest.TIDY_KEY) == str(ingest.TIDY_VERSION)
    assert ingest.tidy_derivatives(conn) == 0  # once per version
    assert ingest.tidy_derivatives(conn, force=True) == 0
