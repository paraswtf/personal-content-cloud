"""Bringing stored faces up to the current face analysis (section 10).

CONTRACT.
- retrack_videos(conn) -> int: once per app.analysis.faces.TRACKER_VERSION (kv video_tracker_version),
  every analysed video (not a Live Photo's clip, not in the trash) is analysed again at
  REANALYSE_PRIORITY, below new uploads, so the current tracker follows its faces through the whole clip
  and measures them; its faces, names and decisions stay until the job runs (the analysis carries them
  over). Waits, kv untouched, while the face models are off or missing. One transaction for the sweep and
  the kv write. Returns the number queued.
- measure_photo_faces(conn, budget_s=None) -> {"items", "faces", "complete"}: faces of analysed photos
  without clarity get it in place. The photo as the analysis saw it is detected again and each stored face
  takes the clarity and sharpness of the detection over it (IoU >= MATCH_IOU); the stored embedding stays.
  Item by item from kv face_measure_cursor, one short transaction each; faces_version moves every
  BUMP_EVERY items and at the end, so the face index weighs them anew. Nothing while the face models are
  off.
- start(conn): on startup (app.ingest.recover_on_startup): retrack_videos, and measure_photo_faces on a
  background thread when photos are left to measure.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from app.config import settings
from app.db import bump_faces_version, kv_get, kv_set, tx
from app.util import now_iso

log = logging.getLogger(__name__)

VIDEO_TRACKER_KEY = "video_tracker_version"
REANALYSE_PRIORITY = 80  # after new uploads' analysis (50), text (60) and guesses (70)
CURSOR_KEY = "face_measure_cursor"
MATCH_IOU = 0.5
BUMP_EVERY = 25


def _faces_ready() -> bool:
    from app.analysis.pipeline import _model

    return _model("faces") is not None


def retrack_videos(conn: sqlite3.Connection) -> int:
    from app import jobs
    from app.analysis.faces import TRACKER_VERSION
    from app.ingest import item_payload

    if int(kv_get(conn, VIDEO_TRACKER_KEY, "1") or 1) >= TRACKER_VERSION or not _faces_ready():
        return 0
    with tx(conn):
        rows = conn.execute(
            "SELECT id, version FROM items WHERE kind = 'video' AND companion_of IS NULL AND deleted_at IS NULL "
            "AND analysis_status = 'done' ORDER BY id").fetchall()
        now = now_iso()
        for r in rows:
            conn.execute("UPDATE items SET analysis_status = 'pending', updated_at = ? WHERE id = ?", (now, r["id"]))
        queued = jobs.enqueue_many(conn, "analyze", (item_payload(r["id"], r["version"]) for r in rows), unique=True,
                                   priority=REANALYSE_PRIORITY)
        kv_set(conn, VIDEO_TRACKER_KEY, str(TRACKER_VERSION))
    if rows:
        log.info("faces: %d analysed video(s) queued for the tracker of version %d", len(rows), TRACKER_VERSION)
    return queued


def _photos_left(conn: sqlite3.Connection, after: int, limit: int = 200) -> list[int]:
    return [int(r[0]) for r in conn.execute(
        "SELECT DISTINCT f.item_id FROM faces f JOIN items i ON i.id = f.item_id WHERE f.clarity IS NULL "
        "AND i.kind = 'image' AND i.analysis_status = 'done' AND i.deleted_at IS NULL AND f.item_id > ? "
        "ORDER BY f.item_id LIMIT ?", (int(after), int(limit)))]


def _measure_item(conn: sqlite3.Connection, engine: Any, item: sqlite3.Row) -> list[tuple[float, float, int]]:
    """(clarity, sharpness, face id) for the unmeasured faces of one photo that a detection sits over."""
    from app.analysis.animals import open_photo
    from app.analysis.cluster import iou
    from app.analysis.faces import _norm_box

    rows = conn.execute("SELECT id, x, y, w, h FROM faces WHERE item_id = ? AND clarity IS NULL", (item["id"],)).fetchall()
    if not rows:
        return []
    _img, bgr = open_photo(item)
    H, W = bgr.shape[:2]
    dets = engine.detect_multiscale(bgr, tuple(settings.face_det_sizes), settings.face_det_threshold)
    if not dets:
        return []
    engine.embed(bgr, dets)
    boxes = [_norm_box(d.bbox, W, H) for d in dets]
    out: list[tuple[float, float, int]] = []
    for r in rows:
        mine = (r["x"], r["y"], r["w"], r["h"])
        best = max(range(len(dets)), key=lambda k: iou(mine, boxes[k]))
        if iou(mine, boxes[best]) >= MATCH_IOU and dets[best].norm is not None:
            out.append((float(dets[best].norm), float(dets[best].sharpness or 0.0), int(r["id"])))
    return out


def measure_photo_faces(conn: sqlite3.Connection, budget_s: float | None = None) -> dict[str, Any]:
    if not _faces_ready():
        return {"items": 0, "faces": 0, "complete": False}
    from app.ml.faces import get_engine

    engine = get_engine()
    deadline = None if budget_s is None else time.monotonic() + float(budget_s)
    cursor = int(kv_get(conn, CURSOR_KEY, "0") or 0)
    done_items = done_faces = 0
    complete = False
    while True:
        ids = _photos_left(conn, cursor)
        if not ids:
            complete = True
            break
        for item_id in ids:
            if deadline is not None and time.monotonic() > deadline:
                break
            item = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
            updates: list[tuple[float, float, int]] = []
            if item is not None:
                try:
                    updates = _measure_item(conn, engine, item)
                except Exception as e:  # noqa: BLE001 - one unreadable photo must not stop the pass
                    log.warning("faces: measuring the faces of item %s failed: %s", item_id, e)
            with tx(conn):
                for clarity, sharpness, fid in updates:
                    conn.execute("UPDATE faces SET clarity = ?, sharpness = ? WHERE id = ? AND clarity IS NULL",
                                 (clarity, sharpness, fid))
                kv_set(conn, CURSOR_KEY, str(item_id))
                done_items += 1
                done_faces += len(updates)
                if done_items % BUMP_EVERY == 0:
                    bump_faces_version(conn)
            cursor = item_id
        else:
            continue
        break
    if done_items:
        with tx(conn):
            bump_faces_version(conn)
        log.info("faces: measured %d face(s) on %d photo(s)%s", done_faces, done_items, "" if complete else " so far")
    return {"items": done_items, "faces": done_faces, "complete": complete}


def _measure_in_background() -> dict[str, Any]:
    from app.db import connect

    conn = connect()
    try:
        return measure_photo_faces(conn)
    finally:
        conn.close()


def start(conn: sqlite3.Connection) -> None:
    try:
        retrack_videos(conn)
    except Exception:
        log.exception("startup: queueing analysed videos for the face tracker failed")
    try:
        if _faces_ready() and _photos_left(conn, int(kv_get(conn, CURSOR_KEY, "0") or 0), limit=1):
            from app import worker

            worker.run_in_background("face-measure", _measure_in_background)
    except Exception:
        log.exception("startup: could not start measuring the faces of analysed photos")
