"""Moving or resizing a face, person or pet box (section 10): the owner's box takes the old one's place.

CONTRACT.
- prepare(conn, table, box_id, box) -> Prepared, with no writes. table is 'faces', 'bodies' or 'animals'; box is
  (x, y, w, h) normalised to the picture. Reads the box and its item, opens the photo or the video frame at the
  box's time (animals.open_photo / open_frame, the frame the analysis saw), checks the new box (inside the
  picture; at least MIN_SIDE_PX a side in that frame, unless it is no smaller than the old box, so a nudge on a
  small detector box always works) and does the slow part:
  - every box: its crop, made as the pipeline makes it (a face: faces.crop_face around the box; a person or
    pet box: the box itself);
  - a face: the face detector inside the new box (person_boxes._face_in, padded by FACE_PAD). A face whose
    centre lies inside the box gives the new face vector, detector score, size (original pixels), quality,
    clarity and sharpness; of several, the one most inside the box (person_boxes.pick_face). None found: the
    old vector stays while the new box is still around the old face (near_old: overlapping, covering most of
    it, or holding its centre), or when the detector is not loaded; otherwise NoFace (the box went where there
    is no face: that is a person box, not a face). With SigLIP, the head-and-torso vector of the new box;
  - a person box: its clothes vector (SigLIP of the upper part) again; a pet box: its look vector again.
  Vectors stay as they were when SigLIP is not loaded.
  Raises LookupError (no such box), ValueError (a bad box), NoFace, animals.FrameTimeNeeded, animals.NoImage.
- store(conn, prepared) -> bool, inside the caller's transaction: writes the owner's box (not the detector's)
  with moved 1 (a re-analysis keeps it like a drawn box; a video's faces are tracked again, section 10; who put
  the person there does not change: source stays) and the new vectors. Names and guesses stay: the owner may be fixing the box before answering a guess. A pet box nobody
  is on is matched to pets again. False when the box is gone.
- write_crop(prepared) after the commit: the old crop is copied to the action's undo stash (undo.keep_copy),
  and the new one written in its place (a new file time, so its URL changes, app.urls).
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app import storage
from app.analysis import ages as ages_mod

log = logging.getLogger(__name__)

TABLES = ("faces", "bodies", "animals")
MIN_SIDE_PX = 12  # a moved box's smallest side, in pixels of the analysed frame
KEEP_IOU = 0.3  # a face box moved no further than this keeps its face vector when no face is found inside
CROP_PATHS = {"faces": storage.face_crop_path, "bodies": storage.body_crop_path, "animals": storage.animal_crop_path}


class NoFace(Exception):
    """The face box moved to where the detector finds no face."""


@dataclass
class Prepared:
    table: str
    box_id: int
    item_id: int
    box: tuple[float, float, float, float]
    crop: bytes
    updates: dict[str, Any] = field(default_factory=dict)  # columns besides x, y, w, h


def check_box(box: tuple[float, float, float, float], old: tuple[float, float, float, float],
              width: int, height: int) -> tuple[float, float, float, float]:
    """The new box, clipped to the picture, or ValueError."""
    x, y, w, h = (float(v) for v in box)
    if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0 and w > 0.0 and h > 0.0 and x + w <= 1.0005 and y + h <= 1.0005):
        raise ValueError("The box must lie inside the picture")
    w, h = min(w, 1.0 - x), min(h, 1.0 - y)
    small = w * width < MIN_SIDE_PX or h * height < MIN_SIDE_PX
    if small and not (w >= float(old[2]) - 1e-9 and h >= float(old[3]) - 1e-9):
        raise ValueError(f"The box must be at least {MIN_SIDE_PX} pixels on each side")
    return x, y, w, h


def near_old(new: tuple[float, float, float, float], old: tuple[float, float, float, float]) -> bool:
    """Still around the old face: the boxes overlap (IoU >= KEEP_IOU), the new one covers most of the old one
    (an enlarged box), or the old one's centre lies inside the new one."""
    from app.analysis.cluster import iou

    if iou(new, old) >= KEEP_IOU:
        return True
    ix = max(0.0, min(new[0] + new[2], old[0] + old[2]) - max(new[0], old[0]))
    iy = max(0.0, min(new[1] + new[3], old[1] + old[3]) - max(new[1], old[1]))
    if old[2] * old[3] > 0 and ix * iy / (old[2] * old[3]) >= 0.5:
        return True
    cx, cy = old[0] + old[2] / 2.0, old[1] + old[3] / 2.0
    return new[0] <= cx <= new[0] + new[2] and new[1] <= cy <= new[1] + new[3]


def _siglip(crop: np.ndarray | None) -> np.ndarray | None:
    """The SigLIP vector of a BGR crop (normalised), or None without the model."""
    from app.analysis.animals import look_engine

    engine = look_engine()
    if engine is None or crop is None or not crop.size:
        return None
    try:
        v = np.asarray(engine.embed_images([crop]), dtype=np.float32)[0]
    except Exception:  # noqa: BLE001 - the box moves without the new vector
        log.exception("box move: SigLIP vector failed")
        return None
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


def prepare(conn: sqlite3.Connection, table: str, box_id: int, box: tuple[float, float, float, float]) -> Prepared:
    from app.analysis import animals as animals_mod
    from app.analysis import bodies as bodies_mod
    from app.analysis import faces as faces_mod
    from app.analysis import person_boxes
    from app.analysis.pipeline import _encode_face_crop, _model

    if table not in TABLES:
        raise ValueError(f"Unknown box kind {table!r}")
    row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (int(box_id),)).fetchone()
    if row is None:
        raise LookupError(box_id)
    item = conn.execute("SELECT * FROM items WHERE id = ?", (int(row["item_id"]),)).fetchone()
    if item is None:
        raise LookupError(box_id)
    if item["kind"] == "video":
        if row["frame_time_s"] is None:
            raise animals_mod.FrameTimeNeeded()
        img, bgr = animals_mod.open_frame(item, float(row["frame_time_s"]))
    else:
        img, bgr = animals_mod.open_photo(item)
    H, W = bgr.shape[:2]
    old = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
    x, y, w, h = check_box(box, old, W, H)
    px = (x * W, y * H, min(float(W), (x + w) * W), min(float(H), (y + h) * H))
    updates: dict[str, Any] = {"moved": 1}
    if table == "faces":
        # the face the box means: centre inside, most inside it (not a neighbour's it takes in part of)
        found = person_boxes.pick_face(person_boxes._faces_in(bgr, px), px)
        if found is not None:
            longest = max(int(item["width"] or 0), int(item["height"] or 0))
            scale = max(1.0, longest / max(W, H)) if longest else 1.0
            size_px = int(round(float(found.size_px) * scale))
            det = float(found.det_score)
            updates.update(
                embedding=np.asarray(found.embedding, dtype=np.float32).tobytes(), det_score=det, size_px=size_px,
                quality=faces_mod.face_quality(det, size_px), clarity=getattr(found, "norm", None),
                sharpness=getattr(found, "sharpness", None),
            )
        elif _model("faces") is not None and not near_old((x, y, w, h), old):
            raise NoFace()
        torso = _siglip(faces_mod.torso_crop(bgr, px))
        if torso is not None:
            updates["torso_embedding"] = torso.tobytes()
        face_img = faces_mod.crop_face(img, px)
        crop = _encode_face_crop(face_img)
        # section 10: the face's age from its new crop (none without the model: never the old crop's)
        updates["age_groups"] = ages_mod.measure([face_img])[0]
    elif table == "bodies":
        clothes = person_boxes._clothes(bgr, px)
        if clothes is not None:
            updates["torso_embedding"] = clothes.tobytes()
        crop = bodies_mod.encode_crop(bodies_mod._crop_rgb(img, px))
    else:
        look = animals_mod.padded_crop(bgr, px)
        emb = animals_mod.embed_look(animals_mod.look_engine(), look)
        if emb is not None:
            updates["embedding"] = np.asarray(emb, dtype=np.float32).tobytes()
        crop = animals_mod.encode_crop(bodies_mod._crop_rgb(img, px))
    return Prepared(table=table, box_id=int(box_id), item_id=int(item["id"]), box=(x, y, w, h), crop=crop,
                    updates=updates)


def store(conn: sqlite3.Connection, p: Prepared) -> bool:
    from app.db import bump_faces_version

    cols: dict[str, Any] = {"x": p.box[0], "y": p.box[1], "w": p.box[2], "h": p.box[3], **p.updates}
    sets = ", ".join(f"{k} = ?" for k in cols)
    n = conn.execute(f"UPDATE {p.table} SET {sets} WHERE id = ? AND item_id = ?",
                     [*cols.values(), p.box_id, p.item_id]).rowcount
    if not n:
        return False
    if p.table in ("faces", "bodies"):
        bump_faces_version(conn)
    elif "embedding" in p.updates:
        from app.analysis import pets

        row = conn.execute("SELECT person_id FROM animals WHERE id = ?", (p.box_id,)).fetchone()
        if row is not None and row["person_id"] is None:
            try:
                pets.assign_new_animals(conn, p.item_id)
            except Exception:  # noqa: BLE001 - the box moves even when matching fails
                log.exception("pets: matching a moved box of item %s failed", p.item_id)
    return True


def write_crop(p: Prepared) -> None:
    from app import undo

    path = CROP_PATHS[p.table](p.box_id)
    undo.keep_copy(path)
    storage.write_bytes_atomic(path, p.crop)
