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
- drop_nested_twins(conn) -> int: once (kv nested_faces_v1): on a photo, two face boxes from the same analysis
  (same created_at, both found by the detector and not moved) with at least app.ml.faces.CONTAIN_THRESH of the
  smaller inside the other are one face seen twice (the detector's merge drops such twins since 2026-11-09).
  One goes: a face the owner tagged stays over one they did not, then the better detector score; two faces the
  owner tagged as different people both stay. A body around the one that goes takes the one that stays, or
  goes too when that one has a body; an unnamed group left empty goes. Not an owner action (no undo). Returns
  the faces removed.
- start(conn): on startup (app.ingest.recover_on_startup): drop_nested_twins, retrack_videos, and
  measure_photo_faces on a background thread when photos are left to measure; app.analysis.ages
  measure_stored_faces on another when the age model is present and a face has no age yet.
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


NESTED_KEY = "nested_faces_v1"


def drop_nested_twins(conn: sqlite3.Connection) -> int:
    from app import storage
    from app.analysis import cluster, guess
    from app.ml.faces import CONTAIN_THRESH

    if kv_get(conn, NESTED_KEY) == "1":
        return 0
    gone: list[int] = []
    gone_bodies: list[int] = []
    items: set[int] = set()
    with tx(conn):
        groups: dict[tuple[int, str], list[sqlite3.Row]] = {}
        for r in conn.execute(
            "SELECT id, item_id, person_id, assigned_by, det_score, created_at, x, y, w, h FROM faces "
            "WHERE frame_time_s IS NULL AND source = 'detector' AND moved = 0 ORDER BY item_id, created_at, id"
        ):
            groups.setdefault((int(r["item_id"]), str(r["created_at"])), []).append(r)
        persons: set[int] = set()
        for faces in groups.values():
            for i, a in enumerate(faces):
                for b in faces[i + 1:]:
                    if int(a["id"]) in gone or int(b["id"]) in gone:
                        continue
                    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
                    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
                    smaller = min(a["w"] * a["h"], b["w"] * b["h"])
                    if smaller <= 0 or ix * iy / smaller < CONTAIN_THRESH:
                        continue
                    a_mine = a["assigned_by"] == "user" and a["person_id"] is not None
                    b_mine = b["assigned_by"] == "user" and b["person_id"] is not None
                    if a_mine and b_mine and a["person_id"] != b["person_id"]:
                        continue  # the owner said these are two people
                    keep, drop = (a, b) if (a_mine, float(a["det_score"])) >= (b_mine, float(b["det_score"])) else (b, a)
                    gone.append(int(drop["id"]))
                    items.add(int(drop["item_id"]))
                    if drop["person_id"] is not None:
                        persons.add(int(drop["person_id"]))
                    held = conn.execute("SELECT 1 FROM bodies WHERE face_id = ? LIMIT 1", (int(keep["id"]),)).fetchone()
                    for body in conn.execute("SELECT id FROM bodies WHERE face_id = ?", (int(drop["id"]),)).fetchall():
                        if held:
                            conn.execute("DELETE FROM bodies WHERE id = ?", (int(body["id"]),))
                            gone_bodies.append(int(body["id"]))
                        else:
                            conn.execute("UPDATE bodies SET face_id = ? WHERE id = ?", (int(keep["id"]), int(body["id"])))
                            held = True
        if gone:
            cluster.remove_faces(conn, gone)
            if gone_bodies:
                cluster.drop_rejections(conn, "body", gone_bodies)
            for item_id in sorted(items):
                conn.execute(
                    "UPDATE items SET face_count = (SELECT COUNT(*) FROM faces WHERE item_id = ?), "
                    "body_count = (SELECT COUNT(*) FROM bodies WHERE item_id = ?) WHERE id = ?", (item_id, item_id, item_id))
            cluster.sync_bodies_to_faces(conn, sorted(items))
            cluster.prune_empty_persons(conn, persons)
            bump_faces_version(conn)
        kv_set(conn, NESTED_KEY, "1")
    for fid in gone:
        storage.safe_unlink(storage.face_crop_path(fid))
    for bid in gone_bodies:
        storage.safe_unlink(storage.body_crop_path(bid))
    if items:
        guess.refresh_fts(conn, items)
        log.info("faces: removed %d nested duplicate face box(es) on %d photo(s)", len(gone), len(items))
    return len(gone)


def _measure_in_background() -> dict[str, Any]:
    from app.db import connect

    conn = connect()
    try:
        return measure_photo_faces(conn)
    finally:
        conn.close()


def _ages_in_background() -> dict[str, Any]:
    from app.analysis import ages
    from app.db import connect

    conn = connect()
    try:
        out = ages.measure_stored_faces(conn)
        log.info("face ages: %d stored faces measured (complete: %s)", out["faces"], out["complete"])
        return out
    finally:
        conn.close()


def start(conn: sqlite3.Connection) -> None:
    try:
        drop_nested_twins(conn)
    except Exception:
        log.exception("startup: removing nested duplicate face boxes failed")
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
    try:
        from app.analysis import ages

        if ages.available():
            if conn.execute("SELECT 1 FROM faces WHERE age_groups IS NULL LIMIT 1").fetchone():
                from app import worker

                worker.run_in_background("face-ages", _ages_in_background)
            elif not ages.ready(conn):
                with tx(conn):
                    ages.mark_ready(conn)  # nothing to read (an empty library, or every face measured)
    except Exception:
        log.exception("startup: could not start reading the ages of stored faces")
