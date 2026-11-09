"""Person boxes the owner draws (section 10; the pet boxes of section 22 work the same way).

CONTRACT.
- add_box(conn, item, box, person_id, *, t=None) -> DrawnBox(body_id, face_id, existing_face), in two steps:
  prepare_box(item, box, t=) (decode, SigLIP and the face search, before any write) and store_box(conn,
  prepared, person_id) (one transaction). A box the
  owner drew around a person on a photo, or on a video's frame at t seconds (x, y, w, h normalised, inside
  the frame, at least animals.MIN_BOX_SHARE of it). A bodies row holds the person for that occasion:
  source 'user', assigned_by 'user', det_score 1.0, the SigLIP vector of its upper part as the clothes the
  guesses compare within the hour (section 10), frame_time_s = t on a video, and its crop file. The face
  detector then looks inside the box (padded by FACE_PAD): the best face it finds becomes a face row of the
  same person (assigned_by 'user', with clarity and sharpness, so its face weight decides how much it
  teaches about their face; a video's gets spans t +- VIDEO_SPAN_S), and the body points at it. On a photo
  a stored face already there (IoU >= EXISTING_IOU) is used instead and returned as existing_face, for the
  caller to assign through the people API. No face (a helmet, a back, someone far away): the box alone,
  which never counts as their face. items.face_count and body_count follow. Raises animals.FrameTimeNeeded,
  animals.NoImage or ValueError. Opens its own transaction.
- Face boxes the owner draws (2026-11-09): prepare_face(item, box, t=) then store_face(conn, prepared,
  person_id) -> DrawnFace(face_id, existing). A box drawn around a face the detector missed, on a photo or a
  video frame at t (inside the frame, at least box_edit.MIN_SIDE_PX a side in the analysed frame). The face
  detector looks inside it (padded by FACE_PAD; the region is enlarged, so a small or soft face it missed in
  the whole picture is often found): a face whose centre lies in the box gives the vector, score, size,
  quality, clarity and sharpness. None: NoFace (a box where no face shows belongs to a person box); no
  detector loaded: animals.ModelsUnavailable. The row keeps the owner's box with source 'user' (a re-analysis
  keeps it on a photo), assigned_by 'user' for the person, frame_time_s = t and spans t +- VIDEO_SPAN_S on a
  video, the head-and-torso vector when SigLIP is loaded, and its crop; on a photo a stored face over the found
  one (IoU >= EXISTING_IOU) is used instead and returned as existing, for the caller to assign through the
  people API; a new face inside a detector's faceless person box (its centre in the box's upper half) becomes that
  box's face, as a detected one would, and the box follows its person. items.face_count follows. Opens its own
  transaction (nested in the caller's when one is open).
- drawn_body_rows(rows) / protected_face_ids(rows): a re-analysis keeps the owner's drawn boxes and the
  faces they point at (app.analysis.pipeline).
- drop_over_drawn(conn, table, new_ids, drawn_rows) -> dropped ids: new detector boxes over a kept drawn
  one (IoU >= EXISTING_IOU, photos) are the same person and are dropped. Call inside a tx.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np

from app import storage
from app.config import settings
from app.db import tx
from app.util import json_dumps, now_iso

log = logging.getLogger(__name__)

FACE_PAD = 0.15  # the face search looks this far (share of the box) around the drawn box
EXISTING_IOU = 0.4  # a stored face over the found one by this much is the same face
VIDEO_SPAN_S = 0.5


@dataclass
class DrawnBox:
    body_id: int
    face_id: int | None
    existing_face: bool


def _check(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    from app.analysis.animals import MIN_BOX_SHARE

    x, y, w, h = (float(v) for v in box)
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0
            and x + w <= 1.0005 and y + h <= 1.0005 and w * h >= MIN_BOX_SHARE):
        raise ValueError("The box must lie inside the photo and cover at least half a percent of it")
    return x, y, min(w, 1.0 - x), min(h, 1.0 - y)


def _clothes(bgr: np.ndarray, px: tuple[float, float, float, float]) -> np.ndarray | None:
    from app.analysis import bodies as bodies_mod
    from app.analysis.animals import look_engine

    engine = look_engine()
    torso = bodies_mod.upper_crop(bgr, px)
    if engine is None or torso is None or not torso.size:
        return None
    try:
        v = np.asarray(engine.embed_images([torso]), dtype=np.float32)[0]
    except Exception:  # noqa: BLE001 - the box stores without its clothes vector
        log.exception("person box: clothes vector failed")
        return None
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


def _faces_in(bgr: np.ndarray, px: tuple[float, float, float, float]) -> list[Any]:
    """Every face the detector finds in the padded box (with a vector; coordinates of bgr), or none."""
    from app.analysis.pipeline import _model

    if _model("faces") is None:
        return []
    from app.ml.faces import get_engine

    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = px
    pw, ph = (x2 - x1) * FACE_PAD, (y2 - y1) * FACE_PAD
    l, t = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    r, b = min(W, int(round(x2 + pw))), min(H, int(round(y2 + ph)))
    if r - l < 8 or b - t < 8:
        return []
    region = np.ascontiguousarray(bgr[t:b, l:r])
    try:
        faces = get_engine().analyze(region, sizes=(640,), threshold=settings.face_det_threshold, min_size=0)
    except Exception:  # noqa: BLE001 - the box stores without a face
        log.exception("person box: face search failed")
        return []
    out = [f for f in faces if f.embedding is not None]
    for f in out:
        f.bbox = f.bbox + np.array([l, t, l, t], dtype=np.float32)
        f.kps = f.kps + np.array([l, t], dtype=np.float32)
    return out


def _face_in(bgr: np.ndarray, px: tuple[float, float, float, float]) -> Any | None:
    """The best face the detector finds in the padded box (coordinates of bgr), or None."""
    faces = _faces_in(bgr, px)
    return max(faces, key=lambda f: f.det_score * min(f.size_px, 160)) if faces else None


def pick_face(faces: list[Any], px: tuple[float, float, float, float]) -> Any | None:
    """The face a box the owner drew or moved means: among those whose centre lies inside it, the one most inside
    it, then the one it overlaps most (a box drawn around a face that takes in part of a neighbour's still means
    the face it surrounds); None when no centre is inside."""
    from app.analysis.cluster import iou

    def norm(b: Any) -> tuple[float, float, float, float]:
        return float(b[0]), float(b[1]), float(b[2]) - float(b[0]), float(b[3]) - float(b[1])

    box = (px[0], px[1], px[2] - px[0], px[3] - px[1])

    def inside_share(f: Any) -> float:
        x1, y1, x2, y2 = (float(v) for v in f.bbox)
        area = max(x2 - x1, 1e-6) * max(y2 - y1, 1e-6)
        ix = max(0.0, min(x2, px[2]) - max(x1, px[0]))
        iy = max(0.0, min(y2, px[3]) - max(y1, px[1]))
        return round(ix * iy / area, 2)

    inside = [f for f in faces
              if px[0] <= (float(f.bbox[0]) + float(f.bbox[2])) / 2.0 <= px[2]
              and px[1] <= (float(f.bbox[1]) + float(f.bbox[3])) / 2.0 <= px[3]]
    return max(inside, key=lambda f: (inside_share(f), iou(norm(f.bbox), box))) if inside else None


@dataclass
class Prepared:
    """The slow part of a drawn person box (decode, SigLIP, face search), done before any write."""

    item_id: int
    video: bool
    t: float | None
    box: tuple[float, float, float, float]
    size: tuple[int, int]  # analysed frame W, H
    scale: float  # original long side / analysed long side
    clothes: np.ndarray | None
    body_crop: bytes
    face: Any | None
    face_crop: bytes | None
    duration_s: float


def prepare_box(item: sqlite3.Row, box: tuple[float, float, float, float], *, t: float | None = None) -> Prepared:
    """Check the box, open the photo or the frame at t, and find the clothes vector and the face inside.
    Raises animals.FrameTimeNeeded, animals.NoImage or ValueError."""
    from app.analysis import bodies as bodies_mod
    from app.analysis import faces as faces_mod
    from app.analysis.animals import FrameTimeNeeded, open_frame, open_photo
    from app.analysis.pipeline import _encode_face_crop

    x, y, w, h = _check(box)
    video = item["kind"] == "video"
    if video:
        if t is None:
            raise FrameTimeNeeded()
        t = round(max(0.0, float(t)), 3)
        img, bgr = open_frame(item, t)
    else:
        t = None
        img, bgr = open_photo(item)
    H, W = bgr.shape[:2]
    px = (x * W, y * H, min(float(W), (x + w) * W), min(float(H), (y + h) * H))
    found = _face_in(bgr, px)
    face_crop = None
    if found is not None:
        face_crop = _encode_face_crop(faces_mod.crop_face(img, tuple(float(v) for v in found.bbox)))  # type: ignore[arg-type]
    longest = max(int(item["width"] or 0), int(item["height"] or 0))
    return Prepared(
        item_id=int(item["id"]), video=video, t=t, box=(x, y, w, h), size=(W, H),
        scale=max(1.0, longest / max(W, H)) if longest else 1.0, clothes=_clothes(bgr, px),
        body_crop=bodies_mod.encode_crop(bodies_mod._crop_rgb(img, px)), face=found, face_crop=face_crop,
        duration_s=float(item["duration_s"] or 0.0) if video else 0.0,
    )


def store_box(conn: sqlite3.Connection, prep: Prepared, person_id: int) -> DrawnBox:
    """Write a prepared box for person_id in one transaction (nested in the caller's when one is open)."""
    from app.analysis import faces as faces_mod
    from app.analysis.cluster import iou

    item_id, t = prep.item_id, prep.t
    x, y, w, h = prep.box
    W, H = prep.size
    now = now_iso()
    written: list[Any] = []
    face_id: int | None = None
    existing = False
    clothes = prep.clothes.tobytes() if prep.clothes is not None else None
    try:
        with tx(conn):
            found = prep.face
            if found is not None:
                fbox = faces_mod._norm_box(found.bbox, W, H)
                if not prep.video:
                    for r in conn.execute("SELECT id, x, y, w, h FROM faces WHERE item_id = ? ORDER BY id", (item_id,)):
                        if iou(fbox, (r["x"], r["y"], r["w"], r["h"])) >= EXISTING_IOU:
                            face_id, existing = int(r["id"]), True
                            break
                if face_id is None:
                    size_px = int(round(found.size_px * prep.scale))
                    spans = None
                    if prep.video and t is not None:
                        end = prep.duration_s or (t + VIDEO_SPAN_S)
                        spans = [[round(max(0.0, t - VIDEO_SPAN_S), 3), round(min(end, t + VIDEO_SPAN_S), 3)]]
                    face_id = int(conn.execute(
                        "INSERT INTO faces(item_id, person_id, assigned_by, x, y, w, h, det_score, size_px, quality, "
                        "frame_time_s, spans, embedding, torso_embedding, created_at, clarity, sharpness) "
                        "VALUES (?, ?, 'user', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (item_id, int(person_id), *fbox, float(found.det_score), size_px,
                         faces_mod.face_quality(found.det_score, size_px), t,
                         json_dumps(spans) if spans is not None else None,
                         np.asarray(found.embedding, dtype=np.float32).tobytes(), clothes, now, found.norm,
                         found.sharpness),
                    ).lastrowid)
                    path = storage.face_crop_path(face_id)
                    storage.write_bytes_atomic(path, prep.face_crop or b"")
                    written.append(path)
            body_id = int(conn.execute(
                "INSERT INTO bodies(item_id, face_id, person_id, assigned_by, x, y, w, h, det_score, torso_embedding, "
                "frame_time_s, created_at, source) VALUES (?, ?, ?, 'user', ?, ?, ?, ?, 1.0, ?, ?, ?, 'user')",
                (item_id, face_id, int(person_id), x, y, w, h, clothes, t, now),
            ).lastrowid)
            path = storage.body_crop_path(body_id)
            storage.write_bytes_atomic(path, prep.body_crop)
            written.append(path)
            conn.execute(
                "UPDATE items SET face_count = (SELECT COUNT(*) FROM faces WHERE item_id = ?), "
                "body_count = (SELECT COUNT(*) FROM bodies WHERE item_id = ?), updated_at = ? WHERE id = ?",
                (item_id, item_id, now, item_id))
    except BaseException:
        for p in written:
            storage.safe_unlink(p)
        raise
    return DrawnBox(body_id=body_id, face_id=face_id, existing_face=existing)


def add_box(conn: sqlite3.Connection, item: sqlite3.Row, box: tuple[float, float, float, float], person_id: int, *,
            t: float | None = None) -> DrawnBox:
    """prepare_box then store_box."""
    return store_box(conn, prepare_box(item, box, t=t), person_id)


class NoFace(Exception):
    """No face shows inside the drawn face box."""


@dataclass
class DrawnFace:
    face_id: int
    existing: bool


@dataclass
class PreparedFace:
    """The slow part of a drawn face box (decode, face search, SigLIP), done before any write."""

    item_id: int
    video: bool
    t: float | None
    box: tuple[float, float, float, float]
    found_box: tuple[float, float, float, float]  # the detector's box, normalised (to find a stored twin)
    det_score: float
    size_px: int
    clarity: float | None
    sharpness: float | None
    embedding: bytes
    torso: bytes | None
    crop: bytes
    duration_s: float


def prepare_face(item: sqlite3.Row, box: tuple[float, float, float, float], *, t: float | None = None) -> PreparedFace:
    """Check the box, open the photo or the frame at t and find the face inside. Raises animals.FrameTimeNeeded,
    animals.NoImage, animals.ModelsUnavailable, NoFace or ValueError."""
    from app.analysis import box_edit
    from app.analysis import faces as faces_mod
    from app.analysis.animals import FrameTimeNeeded, ModelsUnavailable, open_frame, open_photo
    from app.analysis.pipeline import _encode_face_crop, _model

    x, y, w, h = (float(v) for v in box)
    if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0 and w > 0.0 and h > 0.0 and x + w <= 1.0005 and y + h <= 1.0005):
        raise ValueError("The box must lie inside the picture")
    w, h = min(w, 1.0 - x), min(h, 1.0 - y)
    video = item["kind"] == "video"
    if video:
        if t is None:
            raise FrameTimeNeeded()
        t = round(max(0.0, float(t)), 3)
        img, bgr = open_frame(item, t)
    else:
        t = None
        img, bgr = open_photo(item)
    H, W = bgr.shape[:2]
    if w * W < box_edit.MIN_SIDE_PX or h * H < box_edit.MIN_SIDE_PX:
        raise ValueError(f"The box must be at least {box_edit.MIN_SIDE_PX} pixels on each side")
    if _model("faces") is None:
        raise ModelsUnavailable()
    px = (x * W, y * H, min(float(W), (x + w) * W), min(float(H), (y + h) * H))
    found = pick_face(_faces_in(bgr, px), px)
    if found is None:
        raise NoFace()
    longest = max(int(item["width"] or 0), int(item["height"] or 0))
    scale = max(1.0, longest / max(W, H)) if longest else 1.0
    torso = box_edit._siglip(faces_mod.torso_crop(bgr, px))
    return PreparedFace(
        item_id=int(item["id"]), video=video, t=t, box=(x, y, w, h),
        found_box=faces_mod._norm_box(found.bbox, W, H), det_score=float(found.det_score),
        size_px=int(round(float(found.size_px) * scale)), clarity=getattr(found, "norm", None),
        sharpness=getattr(found, "sharpness", None), embedding=np.asarray(found.embedding, dtype=np.float32).tobytes(),
        torso=torso.tobytes() if torso is not None else None,
        crop=_encode_face_crop(faces_mod.crop_face(img, px)),
        duration_s=float(item["duration_s"] or 0.0) if video else 0.0,
    )


def store_face(conn: sqlite3.Connection, prep: PreparedFace, person_id: int) -> DrawnFace:
    """Write a prepared face box for person_id in one transaction (nested in the caller's when one is open)."""
    from app.analysis import faces as faces_mod
    from app.analysis.cluster import iou

    item_id, t = prep.item_id, prep.t
    now = now_iso()
    path = None
    try:
        with tx(conn):
            if not prep.video:
                for r in conn.execute("SELECT id, x, y, w, h FROM faces WHERE item_id = ? ORDER BY id", (item_id,)):
                    if iou(prep.found_box, (r["x"], r["y"], r["w"], r["h"])) >= EXISTING_IOU:
                        return DrawnFace(face_id=int(r["id"]), existing=True)
            spans = None
            if prep.video and t is not None:
                end = prep.duration_s or (t + VIDEO_SPAN_S)
                spans = [[round(max(0.0, t - VIDEO_SPAN_S), 3), round(min(end, t + VIDEO_SPAN_S), 3)]]
            face_id = int(conn.execute(
                "INSERT INTO faces(item_id, person_id, assigned_by, x, y, w, h, det_score, size_px, quality, "
                "frame_time_s, spans, embedding, torso_embedding, created_at, clarity, sharpness, source) "
                "VALUES (?, ?, 'user', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'user')",
                (item_id, int(person_id), *prep.box, prep.det_score, prep.size_px,
                 faces_mod.face_quality(prep.det_score, prep.size_px), t,
                 json_dumps(spans) if spans is not None else None, prep.embedding, prep.torso, now,
                 prep.clarity, prep.sharpness),
            ).lastrowid)
            path = storage.face_crop_path(face_id)
            storage.write_bytes_atomic(path, prep.crop)
            if not prep.video:
                _link_body(conn, item_id, face_id, prep.box)
            conn.execute("UPDATE items SET face_count = (SELECT COUNT(*) FROM faces WHERE item_id = ?), updated_at = ? "
                         "WHERE id = ?", (item_id, now, item_id))
    except BaseException:
        if path is not None:
            storage.safe_unlink(path)
        raise
    return DrawnFace(face_id=face_id, existing=False)


def _link_body(conn: sqlite3.Connection, item_id: int, face_id: int, box: tuple[float, float, float, float]) -> None:
    """A face drawn inside a person box the detector found (with no face of its own, not the owner's) belongs to it,
    as a detected face would: the smallest such box with the face's centre in its upper half points at the face and
    follows its person; the unnamed person the box had goes when nothing else holds them."""
    from app.analysis import cluster

    cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
    rows = [r for r in conn.execute(
        "SELECT id, person_id, x, y, w, h FROM bodies WHERE item_id = ? AND face_id IS NULL AND frame_time_s IS NULL "
        "AND source = 'detector' AND assigned_by != 'user'", (item_id,))
        if r["x"] <= cx <= r["x"] + r["w"] and r["y"] <= cy <= r["y"] + r["h"] / 2.0]
    if not rows:
        return
    body = min(rows, key=lambda r: r["w"] * r["h"])
    conn.execute("UPDATE bodies SET face_id = ? WHERE id = ?", (face_id, int(body["id"])))
    cluster.sync_bodies_to_faces(conn, [item_id])
    if body["person_id"] is not None:
        cluster.prune_empty_persons(conn, {int(body["person_id"])})


def relink_twin_bodies(conn: sqlite3.Connection, new_face_ids: list[int], kept: list[sqlite3.Row],
                       new_body_ids: list[int]) -> list[int]:
    """Before the twins of kept faces go (drop_over_drawn): a new body around a twin takes the kept face instead,
    as dropping the twin would clear its face_id and leave a faceless person box around the kept face; when a
    body already holds the kept face, the new one is the same person again and is deleted. Returns the deleted
    body ids (their crop files are the caller's). Call inside a tx."""
    from app.analysis.animals import EXISTING_IOU as SAME
    from app.analysis.cluster import drop_rejections, iou

    if not new_face_ids or not kept or not new_body_ids:
        return []
    twin_of: dict[int, int] = {}
    marks = ",".join("?" * len(new_face_ids))
    for r in conn.execute(f"SELECT id, x, y, w, h FROM faces WHERE id IN ({marks})", new_face_ids):
        box = (r["x"], r["y"], r["w"], r["h"])
        best = max(kept, key=lambda k: iou(box, (k["x"], k["y"], k["w"], k["h"])))
        if iou(box, (best["x"], best["y"], best["w"], best["h"])) >= SAME:
            twin_of[int(r["id"])] = int(best["id"])
    if not twin_of:
        return []
    deleted: list[int] = []
    marks = ",".join("?" * len(new_body_ids))
    rows = conn.execute(f"SELECT id, face_id FROM bodies WHERE id IN ({marks}) AND face_id IS NOT NULL", new_body_ids).fetchall()
    for b in rows:
        kept_id = twin_of.get(int(b["face_id"]))
        if kept_id is None:
            continue
        if conn.execute("SELECT 1 FROM bodies WHERE face_id = ? AND id != ? LIMIT 1", (kept_id, int(b["id"]))).fetchone():
            conn.execute("DELETE FROM bodies WHERE id = ?", (int(b["id"]),))
            drop_rejections(conn, "body", [int(b["id"])])
            deleted.append(int(b["id"]))
        else:
            conn.execute("UPDATE bodies SET face_id = ? WHERE id = ?", (kept_id, int(b["id"])))
    return deleted


def drawn_body_rows(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    """The owner-drawn person boxes (source 'user') among body rows."""
    return [r for r in rows if "source" in r.keys() and r["source"] == "user"]


def protected_face_ids(drawn: list[sqlite3.Row]) -> set[int]:
    """The faces the owner's drawn person boxes point at: a re-analysis keeps them."""
    return {int(r["face_id"]) for r in drawn if r["face_id"] is not None}


def drop_over_drawn(conn: sqlite3.Connection, table: str, new_ids: list[int],
                    drawn: list[tuple[float, float, float, float]]) -> list[int]:
    """New detector rows of `table` ('faces' or 'bodies') that sit over a kept drawn box or face
    (IoU >= animals.EXISTING_IOU) are the same person: deleted, ids returned (crop files are the caller's)."""
    from app.analysis.animals import EXISTING_IOU as SAME
    from app.analysis.cluster import iou

    if not new_ids or not drawn or table not in ("faces", "bodies"):
        return []
    marks = ",".join("?" * len(new_ids))
    gone = [int(r["id"]) for r in conn.execute(f"SELECT id, x, y, w, h FROM {table} WHERE id IN ({marks})", new_ids)
            if any(iou((r["x"], r["y"], r["w"], r["h"]), d) >= SAME for d in drawn)]
    if gone:
        from app.analysis import cluster

        if table == "faces":
            cluster.remove_faces(conn, gone)  # with their rejections and any cover they became
        else:
            marks = ",".join("?" * len(gone))
            conn.execute(f"DELETE FROM bodies WHERE id IN ({marks})", gone)
            cluster.drop_rejections(conn, "body", gone)
    return gone
