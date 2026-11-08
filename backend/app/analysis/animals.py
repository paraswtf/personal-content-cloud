"""Animal boxes (section 22): the YOLOX detector's COCO animal classes, with crops for the pet
identities and for storage.

CONTRACT.
- AnimalRecord: normalised box, det_score, species, frame_time_s, look (BGR crop padded by
  PAD_SHARE for SigLIP), crop (RGB for storage.animal_crop_path, at most CROP_SIDE px).
- available() -> bool: settings.pets_enabled, models allowed (PCC_NO_MODELS unset), app.ml.persons present.
- thresholds() -> {class_id: threshold}: what to ask the detector for.
- from_boxes(img, img_bgr, found, frame_time_s=None) -> list[AnimalRecord]: records from a
  persons.detect_classes result (best score first per species, at most MAX_PER_SPECIES each); boxes of
  different species over one animal (IoU >= CROSS_SPECIES_IOU) are one record, the better-scored one.
- detect_animals(img, img_bgr, frame_time_s=None) -> list[AnimalRecord]: one detector run.

Species corrections (section 22):
- SPECIES_PROMPTS: one zero-shot prompt group per species. species_scores_of(engine, emb) -> {species: p}.
- species_check(embedding, detected, *, engine=None, scores=None) -> (species, scores, changed): the
  zero-shot answer wins over the detector's class when it is at least SPECIES_SWAP_MIN and the detector's
  species gets at most SPECIES_DETECTED_MAX; `scores` (a table, for tests and the backfill) or `engine`
  (SigLIP) supply the probabilities; without either the detector's class stands.
- settle_species(embedding, detected, *, engine=None) -> (species, scores): the look-alike vote of the
  owner's corrections (learning.images.species_vote) first, then species_check.
- animal_faces(faces, animals, bodies) -> list[int]: indices of face boxes that sit inside an animal box
  (FACE_IN_ANIMAL of their area) with no person box over them: an animal's face, dropped before clustering.
- housekeeping(conn, budget_s) -> int and status(conn): the backfill for stored crops without species_scores
- add_box(conn, item, (x, y, w, h), *, species=None, engine=None) -> int: a box the owner drew on a photo
  (section 22, boxes the owner adds): crop, look embedding, species (the owner's, else the zero-shot's, else
  SpeciesNeeded), the row with source 'user', the pet match. redetect(conn, item, threshold) -> list[int]:
  the detector once more at a lower threshold; boxes over a stored one are left out; ModelsUnavailable
  without the detector. drawn_ids(rows) and drop_over_drawn(conn, item_id, new_ids): what a re-analysis
  keeps and what it must not duplicate. look_engine() -> the SigLIP engine or None (tests patch it).
  (never a user row), the cross-species duplicates and the faces inside animal boxes of stored items.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from app import storage
from app.analysis import bodies as bodies_mod
from app.config import settings
from app.db import tx
from app.util import json_dumps, json_loads, now_iso

log = logging.getLogger(__name__)

PAD_SHARE = 0.10
MIN_BOX_PX = 24
MAX_PER_SPECIES = 6
MAX_ANIMALS = 12
SPECIES = ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")
SPECIES_PROMPTS: dict[str, list[str]] = {
    "bird": ["a photo of a bird"], "cat": ["a photo of a cat"], "dog": ["a photo of a dog"],
    "horse": ["a photo of a horse"], "sheep": ["a photo of a sheep"], "cow": ["a photo of a cow"],
    "elephant": ["a photo of an elephant"], "bear": ["a photo of a bear"], "zebra": ["a photo of a zebra"],
    "giraffe": ["a photo of a giraffe"],
}
SPECIES_SWAP_MIN = 0.6      # the zero-shot's species replaces the detector's from this probability
SPECIES_DETECTED_MAX = 0.25  # and only when the detector's species gets at most this
CROSS_SPECIES_IOU = 0.6     # boxes of different species this overlapping are one animal
FACE_IN_ANIMAL = 0.8        # a face with this share of its area inside an animal box is the animal's face
BODY_OVER_FACE = 0.5        # a person box covering this share of the face says a person is there
BACKFILL_BATCH = 500
SWEEP_KEY = "animals_sweep_version"  # kv: the stored-box sweeps (duplicates, animal faces) done
SWEEP_VERSION = 1

_lock = threading.Lock()
_running = False


@dataclass
class AnimalRecord:
    x: float
    y: float
    w: float
    h: float
    det_score: float
    species: str
    look: np.ndarray | None
    crop: Image.Image
    frame_time_s: float | None = None
    box_px: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    species_scores: dict[str, float] | None = None  # the zero-shot check's table, once it ran


def available() -> bool:
    if not settings.pets_enabled or os.environ.get("PCC_NO_MODELS") == "1":
        return False
    try:
        from app.ml import persons
    except ImportError:
        return False
    try:
        return bool(persons.available())
    except Exception as e:  # a broken model must not fail the analysis
        log.warning("animal detector unavailable: %s", e)
        return False


def thresholds() -> dict[int, float]:
    from app.ml import persons

    return {c: float(settings.animal_det_threshold) for c in persons.ANIMAL_CLASSES}


def padded_crop(img_bgr: np.ndarray, box: tuple[float, float, float, float], share: float = PAD_SHARE) -> np.ndarray | None:
    """The box grown by `share` on every side (clipped), as a BGR array for the look embedding."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * share, (y2 - y1) * share
    l, t = max(0, int(round(x1 - px))), max(0, int(round(y1 - py)))
    r, b = min(w, int(round(x2 + px))), min(h, int(round(y2 + py)))
    if r - l < 4 or b - t < 4:
        return None
    return np.ascontiguousarray(img_bgr[t:b, l:r]).copy()


def from_boxes(img: Image.Image, img_bgr: np.ndarray, found: dict[int, list[Any]],
               frame_time_s: float | None = None) -> list[AnimalRecord]:
    from app.ml import persons

    H, W = img_bgr.shape[:2]
    out: list[AnimalRecord] = []
    for cls, boxes in found.items():
        species = persons.ANIMAL_CLASSES.get(int(cls))
        if species is None:
            continue
        for x1, y1, x2, y2, score in sorted(boxes, key=lambda b: -float(b[4]))[:MAX_PER_SPECIES]:
            x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
            x2, y2 = min(float(W), float(x2)), min(float(H), float(y2))
            if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                continue
            box = (x1, y1, x2, y2)
            out.append(AnimalRecord(
                x=x1 / W, y=y1 / H, w=(x2 - x1) / W, h=(y2 - y1) / H, det_score=float(score), species=species,
                look=padded_crop(img_bgr, box), crop=bodies_mod._crop_rgb(img, box), frame_time_s=frame_time_s,
                box_px=box,
            ))
    out.sort(key=lambda a: -a.det_score)
    return one_per_animal(out)[:MAX_ANIMALS]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two (x, y, w, h) boxes in any one unit."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def one_per_animal(records: list[AnimalRecord]) -> list[AnimalRecord]:
    """Boxes of different species over one animal (IoU >= CROSS_SPECIES_IOU) collapse to the
    better-scored one; `records` best score first. Boxes of one species were already NMS'd."""
    kept: list[AnimalRecord] = []
    for rec in records:
        if any(k.frame_time_s == rec.frame_time_s and k.species != rec.species
               and iou((k.x, k.y, k.w, k.h), (rec.x, rec.y, rec.w, rec.h)) >= CROSS_SPECIES_IOU for k in kept):
            continue
        kept.append(rec)
    return kept


def detect_animals(img: Image.Image, img_bgr: np.ndarray, frame_time_s: float | None = None) -> list[AnimalRecord]:
    """Animal boxes from one detector run. Callers check available()."""
    from app.ml import persons

    return from_boxes(img, img_bgr, persons.detect_classes(img_bgr, thresholds()), frame_time_s)


def encode_crop(img: Image.Image) -> bytes:
    return bodies_mod.encode_crop(img)


# ------------------------------------------------------------------ species (section 22)
def species_scores_of(engine: Any, emb: np.ndarray | None) -> dict[str, float] | None:
    """{species: probability} from one zero-shot over SPECIES_PROMPTS, or None without an embedding."""
    if emb is None or engine is None:
        return None
    probs = np.asarray(engine.zero_shot(np.asarray(emb, dtype=np.float32)[None, :], SPECIES_PROMPTS), dtype=np.float64)[0]
    return {sp: round(float(p), 4) for sp, p in zip(SPECIES_PROMPTS, probs)}


def species_check(embedding: np.ndarray | None, detected: str, *, engine: Any = None,
                  scores: dict[str, float] | None = None) -> tuple[str, dict[str, float] | None, bool]:
    """The detector's species, or the zero-shot's when it is sure enough (section 22). Pure when `scores`
    is given; otherwise the engine computes them from the embedding; without either nothing changes."""
    if scores is None:
        try:
            scores = species_scores_of(engine, embedding)
        except Exception:  # noqa: BLE001 - a model failure never loses the box
            log.exception("species check failed")
            scores = None
    if not scores:
        return detected, scores, False
    best = max(scores, key=lambda k: float(scores[k]))
    if best != detected and best in SPECIES and float(scores[best]) >= SPECIES_SWAP_MIN \
            and float(scores.get(detected, 0.0)) <= SPECIES_DETECTED_MAX:
        return best, scores, True
    return detected, scores, False


def settle_species(embedding: np.ndarray | None, detected: str, *, engine: Any = None,
                   scores: dict[str, float] | None = None) -> tuple[str, dict[str, float] | None]:
    """The owner's look-alike vote first (learning.images.species_vote), then species_check."""
    vote: str | None = None
    try:
        from app.learning import images as learned_images

        vote = learned_images.species_vote(embedding)
    except Exception:  # noqa: BLE001 - learning never blocks the analysis
        log.exception("species vote failed")
    if scores is None and engine is not None:
        try:
            scores = species_scores_of(engine, embedding)
        except Exception:  # noqa: BLE001
            log.exception("species scores failed")
            scores = None
    if vote in SPECIES:
        return vote, scores
    species, scores, _changed = species_check(embedding, detected, scores=scores)
    return species, scores


def _face_share_inside(face: tuple[float, float, float, float], box: tuple[float, float, float, float]) -> float:
    fx2, fy2, bx2, by2 = face[0] + face[2], face[1] + face[3], box[0] + box[2], box[1] + box[3]
    iw, ih = max(0.0, min(fx2, bx2) - max(face[0], box[0])), max(0.0, min(fy2, by2) - max(face[1], box[1]))
    area = face[2] * face[3]
    return (iw * ih) / area if area > 0 else 0.0


def animal_faces(faces: list[tuple[float, float, float, float]], animals: list[tuple[float, float, float, float]],
                 bodies: list[tuple[float, float, float, float]]) -> list[int]:
    """Indices of the faces (x, y, w, h, normalised) that lie inside an animal box by at least FACE_IN_ANIMAL
    of their area while no person box covers BODY_OVER_FACE of them: the animal's face, not a person's."""
    out: list[int] = []
    for i, f in enumerate(faces):
        if not any(_face_share_inside(f, a) >= FACE_IN_ANIMAL for a in animals):
            continue
        if any(_face_share_inside(f, b) >= BODY_OVER_FACE for b in bodies):
            continue
        out.append(i)
    return out


# ------------------------------------------------------------------ the backfill
def _engine() -> Any | None:
    if os.environ.get("PCC_NO_MODELS") == "1":
        return None
    try:
        from app.ml import siglip

        if not siglip.available():
            return None
        return siglip.get_engine()
    except Exception:  # noqa: BLE001 - no model, no check
        log.debug("siglip unavailable for the species backfill", exc_info=True)
        return None


def _vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    v = np.frombuffer(blob, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


def running() -> bool:
    return _running


def recheck_crop(conn: sqlite3.Connection, row: sqlite3.Row, *, engine: Any = None,
                 scores: dict[str, float] | None = None) -> bool:
    """One stored crop through settle_species: writes species_scores, and when the species changes, the
    new species; a crop in a pet of another species leaves it and is offered to the new species' pets.
    Never a crop whose species the owner set. Call inside a tx. Returns True when the species changed."""
    from app.analysis import cluster, pets

    if row["species_source"] == "user":
        return False
    emb = _vec(row["embedding"])
    species, table = settle_species(emb, str(row["species"]), engine=engine, scores=scores)
    conn.execute("UPDATE animals SET species_scores = ? WHERE id = ?", (json_dumps(table or {}), int(row["id"])))
    if species == row["species"]:
        return False
    conn.execute("UPDATE animals SET species = ?, species_source = 'auto' WHERE id = ?", (species, int(row["id"])))
    pid = row["person_id"]
    if pid is not None:
        pet = conn.execute("SELECT species FROM persons WHERE id = ?", (pid,)).fetchone()
        if pet is None or pet["species"] != species:
            conn.execute("UPDATE animals SET person_id = NULL, assigned_by = 'auto' WHERE id = ?", (int(row["id"]),))
            cluster.prune_empty_persons(conn, {int(pid)})
    if row["assigned_by"] != "user" or pid is None:
        pets.assign_new_animals(conn, int(row["item_id"]))
    return True


def dedup_item(conn: sqlite3.Connection, item_id: int) -> list[int]:
    """Stored boxes of one item: cross-species duplicates (IoU >= CROSS_SPECIES_IOU, same frame) collapse
    to the better-scored box; the dropped box's pet moves to the kept box when it has none. Call inside
    a tx. Returns the dropped ids (their crop files are the caller's to remove)."""
    rows = conn.execute(
        "SELECT id, person_id, assigned_by, species, species_source, det_score, frame_time_s, x, y, w, h FROM animals "
        "WHERE item_id = ? ORDER BY CASE WHEN source = 'user' THEN 0 ELSE 1 END, "
        "CASE WHEN species_source = 'user' THEN 0 ELSE 1 END, det_score DESC, id",
        (item_id,),
    ).fetchall()
    kept: list[sqlite3.Row] = []
    dropped: list[int] = []
    for r in rows:
        twin = next((k for k in kept if k["frame_time_s"] == r["frame_time_s"] and k["species"] != r["species"]
                     and iou((k["x"], k["y"], k["w"], k["h"]), (r["x"], r["y"], r["w"], r["h"])) >= CROSS_SPECIES_IOU), None)
        if twin is None:
            kept.append(r)
            continue
        if twin["person_id"] is None and r["person_id"] is not None:
            pet = conn.execute("SELECT species FROM persons WHERE id = ?", (r["person_id"],)).fetchone()
            if pet is not None and pet["species"] == twin["species"]:
                conn.execute("UPDATE animals SET person_id = ?, assigned_by = ? WHERE id = ?",
                             (r["person_id"], r["assigned_by"], int(twin["id"])))
        conn.execute("DELETE FROM animals WHERE id = ?", (int(r["id"]),))
        dropped.append(int(r["id"]))
    if dropped:
        conn.execute("UPDATE items SET animal_count = (SELECT COUNT(*) FROM animals WHERE item_id = ?) WHERE id = ?",
                     (item_id, item_id))
    return dropped


def drop_animal_faces(conn: sqlite3.Connection, item_id: int) -> list[int]:
    """Stored faces of one item that sit inside an animal box with no person box over them are removed
    (the way DELETE /api/faces/{id} removes a face: the row, its rejections, the cover it may have been,
    the count). Call inside a tx. Returns the removed face ids (their crop files are the caller's)."""
    from app.analysis import cluster

    faces = conn.execute("SELECT id, person_id, x, y, w, h FROM faces WHERE item_id = ? AND frame_time_s IS NULL", (item_id,)).fetchall()
    if not faces:
        return []
    animals = [(r["x"], r["y"], r["w"], r["h"]) for r in conn.execute(
        "SELECT x, y, w, h FROM animals WHERE item_id = ? AND frame_time_s IS NULL", (item_id,))]
    if not animals:
        return []
    bodies = [(r["x"], r["y"], r["w"], r["h"]) for r in conn.execute("SELECT x, y, w, h FROM bodies WHERE item_id = ?", (item_id,))]
    hits = animal_faces([(f["x"], f["y"], f["w"], f["h"]) for f in faces], animals, bodies)
    removed: list[int] = []
    people: set[int] = set()
    for i in hits:
        f = faces[i]
        removed.append(int(f["id"]))
        if f["person_id"] is not None:
            people.add(int(f["person_id"]))
    if not removed:
        return []
    cluster.remove_faces(conn, removed)
    conn.execute("UPDATE items SET face_count = (SELECT COUNT(*) FROM faces WHERE item_id = ?) WHERE id = ?", (item_id, item_id))
    if people:
        cluster.prune_empty_persons(conn, people)
    return removed


def housekeeping(conn: sqlite3.Connection, budget_s: float = 5.0) -> int:
    """The worker's entry (section 22): re-check the species of stored crops without species_scores
    (BACKFILL_BATCH per run, within the budget, never a user row), and once per SWEEP_VERSION collapse the
    cross-species duplicates and remove the faces inside animal boxes of stored items. Returns how many
    crops were checked."""
    global _running
    if not settings.pets_enabled:
        return 0
    if not _lock.acquire(blocking=False):
        return 0
    from app import storage
    from app.db import bump_faces_version, kv_get, kv_set, tx

    deadline = time.monotonic() + float(budget_s)
    done = 0
    _running = True
    try:
        if (kv_get(conn, SWEEP_KEY, "0") or "0") != str(SWEEP_VERSION):
            trash: list[int] = []
            trash_faces: list[int] = []
            with tx(conn):
                items = [int(r[0]) for r in conn.execute(
                    "SELECT item_id FROM animals GROUP BY item_id HAVING COUNT(DISTINCT species) > 1")]
                for item_id in items:
                    trash.extend(dedup_item(conn, item_id))
                for r in conn.execute(
                    "SELECT DISTINCT a.item_id FROM animals a JOIN faces f ON f.item_id = a.item_id "
                    "JOIN items i ON i.id = a.item_id WHERE i.kind = 'image'"
                ).fetchall():
                    trash_faces.extend(drop_animal_faces(conn, int(r[0])))
                if trash or trash_faces:
                    bump_faces_version(conn)
                kv_set(conn, SWEEP_KEY, SWEEP_VERSION)
            for aid in trash:
                storage.safe_unlink(storage.animal_crop_path(aid))
            for fid in trash_faces:
                storage.safe_unlink(storage.face_crop_path(fid))
            if trash or trash_faces:
                log.info("pets: dropped %d duplicate box(es) and %d face(s) inside animal boxes", len(trash), len(trash_faces))
        engine = _engine()
        if engine is None:
            return done
        while time.monotonic() < deadline:
            rows = conn.execute(
                "SELECT id, item_id, person_id, assigned_by, species, species_source, embedding FROM animals "
                "WHERE species_scores IS NULL AND species_source = 'auto' AND embedding IS NOT NULL ORDER BY id LIMIT ?",
                (BACKFILL_BATCH,),
            ).fetchall()
            if not rows:
                break
            changed = 0
            with tx(conn):
                for r in rows:
                    if time.monotonic() >= deadline:
                        break
                    try:
                        if recheck_crop(conn, r, engine=engine):
                            changed += 1
                    except Exception:  # noqa: BLE001 - one bad crop never stalls the backfill
                        log.exception("pets: species backfill failed for crop %s", r["id"])
                        conn.execute("UPDATE animals SET species_scores = '{}' WHERE id = ?", (int(r["id"]),))
                    done += 1
                if changed:
                    bump_faces_version(conn)
            if changed:
                log.info("pets: species backfill changed %d of %d crop(s)", changed, len(rows))
    finally:
        _running = False
        _lock.release()
    return done


def status(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        "SELECT COALESCE(SUM(CASE WHEN species_scores IS NULL AND species_source = 'auto' AND embedding IS NOT NULL "
        "THEN 1 ELSE 0 END), 0) AS unchecked, COALESCE(SUM(CASE WHEN species_source = 'user' THEN 1 ELSE 0 END), 0) AS corrected "
        "FROM animals"
    ).fetchone()
    return {"enabled": bool(settings.pets_enabled), "unchecked": int(row["unchecked"]),
            "corrected": int(row["corrected"]), "running": running()}


# ------------------------------------------------------------------ boxes the owner adds (section 22)
MIN_BOX_SHARE = 0.005   # an owner-drawn box covers at least this share of the frame
REDETECT_DEFAULT = 0.2  # the second look's detector threshold
REDETECT_FLOOR = 0.1
EXISTING_IOU = 0.5      # a box over a stored one by this much is the same animal
SOURCES = ("detector", "user")
REDETECT_VIDEO_FRAMES = 12  # a video's second look: frames sampled evenly besides the poster
SAMPLE_GAP_S = 0.5          # the second look samples a video no denser than this


class SpeciesNeeded(Exception):
    """No species given and no model to tell one: the owner has to pick."""


class NoImage(Exception):
    """The item is not a photo, or its original cannot be opened."""


class FrameTimeNeeded(Exception):
    """A box on a video needs the time of its frame."""


class ModelsUnavailable(Exception):
    """The detector is not loaded on this server."""


def look_engine() -> Any | None:
    """The SigLIP engine when its model is present and allowed, else None."""
    return _engine()


def open_photo(item: sqlite3.Row) -> tuple[Image.Image, np.ndarray]:
    """The photo as the analyze stage sees it (oriented, at most ANALYSIS_SIDE px): (PIL RGB, BGR array)."""
    from app.analysis.pipeline import ANALYSIS_SIDE, _to_bgr
    from app.media.images import load_oriented

    if item["kind"] != "image":
        raise NoImage("photos only")
    try:
        img = load_oriented(storage.original_path(item["uid"], int(item["version"]), item["ext"]), max_side=ANALYSIS_SIDE)
    except Exception as e:  # noqa: BLE001 - a missing or broken original
        raise NoImage(str(e)) from e
    if img is None:
        raise NoImage("no image")
    img = img.convert("RGB")
    return img, _to_bgr(img)


def _video_paths(item: sqlite3.Row) -> list[Any]:
    """Where a video's frames come from: the original, else the web copy (section 22)."""
    paths = [storage.original_path(item["uid"], int(item["version"]), item["ext"]),
             storage.video_preview_path(item["uid"], int(item["version"]))]
    return [p for p in paths if p.is_file()]


def open_frame(item: sqlite3.Row, t: float) -> tuple[Image.Image, np.ndarray]:
    """A video's frame at t seconds as the analyze stage would see it (display-oriented, at most
    ANALYSIS_SIDE px): from the original, else from the web copy. Raises NoImage when nothing decodes there."""
    from app.analysis.pipeline import ANALYSIS_SIDE, _to_bgr
    from app.media import video

    if item["kind"] != "video":
        raise NoImage("videos only")
    img = None
    for path in _video_paths(item):
        try:
            img = video.extract_frame_image(path, float(t), max_side=ANALYSIS_SIDE)
        except Exception as e:  # noqa: BLE001 - ffmpeg could not run
            log.warning("frame at %.2fs of %s failed: %s", t, path, e)
            img = None
        if img is not None:
            break
    if img is None:
        raise NoImage(f"no frame at {float(t):.2f}s")
    img = img.convert("RGB")
    return img, _to_bgr(img)


def video_info(item: sqlite3.Row) -> Any:
    """A MediaInfo from the item row (duration, fps, size): what video.frames_at and poster_time need."""
    from app.media.probe import MediaInfo

    return MediaInfo(kind="video", mime=item["mime"], ext=item["ext"], width=item["width"], height=item["height"],
                     duration_s=item["duration_s"], fps=item["fps"])


def sample_times(info: Any, k: int = REDETECT_VIDEO_FRAMES) -> list[float]:
    """The poster time and up to k times spread evenly over the clip (at least SAMPLE_GAP_S apart; one within
    that of the poster is dropped), sorted (section 22, the second look on a video)."""
    from app.media import video

    d = max(0.0, float(info.duration_s or 0.0))
    poster = float(video.poster_time(info))
    times = [poster]
    n = min(int(k), int(d / SAMPLE_GAP_S))
    for i in range(n):
        t = round(d * (i + 0.5) / n, 3)
        if abs(t - poster) > SAMPLE_GAP_S:
            times.append(t)
    return sorted(times)


def video_frames(item: sqlite3.Row) -> list[tuple[float, Image.Image, np.ndarray]]:
    """(time, PIL RGB, BGR) of a video's poster frame and up to REDETECT_VIDEO_FRAMES sampled frames, from
    the original, else the web copy. Raises NoImage when no frame decodes."""
    from app.analysis.pipeline import ANALYSIS_SIDE, _to_bgr
    from app.media import video

    info = video_info(item)
    times = sample_times(info)
    out: list[tuple[float, Image.Image, np.ndarray]] = []
    for path in _video_paths(item):
        try:
            frames = video.frames_at(path, info, times, max_side=ANALYSIS_SIDE)
        except Exception as e:  # noqa: BLE001 - ffmpeg could not run
            log.warning("frames of %s failed: %s", path, e)
            continue
        for t, f in zip(times, frames):
            if f is not None:
                f = f.convert("RGB")
                out.append((t, f, _to_bgr(f)))
        if out:
            break
    if not out:
        raise NoImage("no frame decoded")
    return out


def embed_look(engine: Any | None, look: np.ndarray | None) -> np.ndarray | None:
    """The L2-normalised SigLIP vector of a padded crop, or None without an engine or a crop."""
    if engine is None or look is None or not getattr(look, "size", 0):
        return None
    v = np.asarray(engine.embed_images([look])[0], dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


def drawn_ids(rows: list[sqlite3.Row]) -> list[int]:
    """The owner-drawn boxes (source 'user') among animal rows: a re-analysis keeps them untouched."""
    out: list[int] = []
    for r in rows:
        try:
            src = r["source"]
        except (IndexError, KeyError):
            src = "detector"
        if src == "user":
            out.append(int(r["id"]))
    return out


def _store(conn: sqlite3.Connection, item_id: int, *, box: tuple[float, float, float, float], det_score: float,
           species: str, species_source: str, scores: dict[str, float] | None, frame_time_s: float | None,
           crop: Image.Image, embedding: np.ndarray | None, source: str, now: str) -> int:
    aid = int(conn.execute(
        "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, species_source, "
        "species_scores, frame_time_s, embedding, source, created_at) "
        "VALUES (?, NULL, 'auto', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (item_id, *box, float(det_score), species, species_source, json_dumps(scores) if scores else None,
         frame_time_s, None if embedding is None else embedding.astype(np.float32).tobytes(), source, now),
    ).lastrowid)
    storage.write_bytes_atomic(storage.animal_crop_path(aid), encode_crop(crop))
    return aid


def _count_animals(conn: sqlite3.Connection, item_id: int) -> None:
    conn.execute("UPDATE items SET animal_count = (SELECT COUNT(*) FROM animals WHERE item_id = ?) WHERE id = ?",
                 (item_id, item_id))


def add_box(conn: sqlite3.Connection, item: sqlite3.Row, box: tuple[float, float, float, float], *,
            species: str | None = None, engine: Any | None = None, t: float | None = None) -> int:
    """A box the owner drew on a photo, or on a video's frame at `t` seconds (x, y, w, h normalised, inside
    the frame, at least MIN_BOX_SHARE of it): the padded crop is embedded when SigLIP is loaded, the species
    is the owner's (species_source 'user') or the zero-shot's best answer, else SpeciesNeeded; the row
    carries det_score 1.0, source 'user' and, for a video, frame_time_s = t; items.animal_count follows and
    pets.assign_new_animals matches the box. Returns the box id. Raises FrameTimeNeeded for a video without
    t and NoImage for an unreadable original or frame. Opens its own transaction."""
    from app.analysis import pets

    x, y, w, h = (float(v) for v in box)
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0
            and x + w <= 1.0005 and y + h <= 1.0005 and w * h >= MIN_BOX_SHARE):
        raise ValueError("The box must lie inside the photo and cover at least half a percent of it")
    if species is not None and species not in SPECIES:
        raise ValueError(f"Unknown species {species!r}")
    if item["kind"] == "video":
        if t is None:
            raise FrameTimeNeeded()
        t = round(max(0.0, float(t)), 3)
        img, bgr = open_frame(item, t)
    else:
        t = None
        img, bgr = open_photo(item)
    H, W = bgr.shape[:2]
    px = (x * W, y * H, min(float(W), (x + w) * W), min(float(H), (y + h) * H))
    look = padded_crop(bgr, px)
    crop = bodies_mod._crop_rgb(img, px)
    engine = engine if engine is not None else look_engine()
    emb = embed_look(engine, look)
    scores: dict[str, float] | None = None
    try:
        scores = species_scores_of(engine, emb) if emb is not None else None
    except Exception:  # noqa: BLE001 - the box still stores without the table
        log.exception("species scores for a drawn box failed")
        scores = None
    if species is None:
        if not scores:
            raise SpeciesNeeded()
        species = max(scores, key=lambda k: float(scores[k]))
        species_source = "auto"
    else:
        species_source = "user"
    item_id = int(item["id"])
    now = now_iso()
    with tx(conn):
        aid = _store(conn, item_id, box=(x, y, w, h), det_score=1.0, species=species, species_source=species_source,
                     scores=scores, frame_time_s=t, crop=crop, embedding=emb, source="user", now=now)
        _count_animals(conn, item_id)
        try:
            pets.assign_new_animals(conn, item_id)
        except Exception:  # noqa: BLE001 - the box stays even when matching fails
            log.exception("pets: matching a drawn box of item %s failed", item_id)
    return aid


def redetect(conn: sqlite3.Connection, item: sqlite3.Row, threshold: float = REDETECT_DEFAULT, *,
             engine: Any | None = None) -> list[int]:
    """The detector once more, with `threshold` (floor REDETECT_FLOOR) for the animal classes: on a photo, or
    on a video's poster frame and up to REDETECT_VIDEO_FRAMES frames sampled evenly (video_frames). Boxes over
    a stored box of the same frame (IoU >= EXISTING_IOU; a video's stored box belongs to the sampled frame
    nearest its time) are left out; the rest, at most MAX_ANIMALS per run (highest scores
    first), are stored like the pipeline stores them (source 'detector', the detector's species settled by
    the zero-shot check and the look-alike vote, the look embedding, a video frame's time) and matched to
    pets. Returns the new ids. Raises ModelsUnavailable without the detector and NoImage for an unreadable
    original or video. Opens its own transaction."""
    from app.analysis import pets
    from app.ml import persons

    if not available():
        raise ModelsUnavailable()
    frames: list[tuple[float | None, Image.Image, np.ndarray]]
    if item["kind"] == "video":
        frames = list(video_frames(item))
    else:
        img, bgr = open_photo(item)
        frames = [(None, img, bgr)]
    thr = max(REDETECT_FLOOR, min(0.9, float(threshold)))
    item_id = int(item["id"])
    times = [t for t, _, _ in frames]

    def frame_of(stored_t: float | None) -> float | None:
        """The sampled frame a stored box belongs to: the nearest in time (a photo has one frame)."""
        if stored_t is None or len(times) == 1:
            return times[0]
        return min(times, key=lambda x: abs(float(x) - float(stored_t)))  # type: ignore[arg-type]

    stored: list[tuple[float | None, tuple[float, float, float, float]]] = [
        (frame_of(r["frame_time_s"]), (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])))
        for r in conn.execute("SELECT x, y, w, h, frame_time_s FROM animals WHERE item_id = ?", (item_id,))]
    recs: list[AnimalRecord] = []
    for t, img, bgr in frames:
        found = persons.detect_classes(bgr, {c: thr for c in persons.ANIMAL_CLASSES})
        for rec in from_boxes(img, bgr, found, frame_time_s=t):
            box = (rec.x, rec.y, rec.w, rec.h)
            if any(ft == t and iou(box, e) >= EXISTING_IOU for ft, e in stored):
                continue
            stored.append((t, box))
            recs.append(rec)
    recs = sorted(recs, key=lambda r: -r.det_score)[:MAX_ANIMALS]
    engine = engine if engine is not None else look_engine()
    new_ids: list[int] = []
    now = now_iso()
    with tx(conn):
        for rec in recs:
            emb = embed_look(engine, rec.look)
            species, scores = settle_species(emb, rec.species, engine=engine)
            aid = _store(conn, item_id, box=(rec.x, rec.y, rec.w, rec.h), det_score=rec.det_score, species=species,
                         species_source="auto", scores=scores, frame_time_s=rec.frame_time_s, crop=rec.crop,
                         embedding=emb, source="detector", now=now)
            new_ids.append(aid)
        if new_ids:
            _count_animals(conn, item_id)
            try:
                pets.assign_new_animals(conn, item_id)
            except Exception:  # noqa: BLE001
                log.exception("pets: matching re-detected boxes of item %s failed", item_id)
    return new_ids


def drop_over_drawn(conn: sqlite3.Connection, item_id: int, new_ids: list[int]) -> list[int]:
    """After a re-analysis: new detector boxes that sit over an owner-drawn box (IoU >= EXISTING_IOU) are
    the same animal and are dropped (the drawn box stays). Call inside a tx; returns the dropped ids
    (their crop files are the caller's to remove)."""
    if not new_ids:
        return []
    drawn = conn.execute("SELECT x, y, w, h FROM animals WHERE item_id = ? AND source = 'user'", (item_id,)).fetchall()
    if not drawn:
        return []
    marks = ",".join("?" * len(new_ids))
    dropped: list[int] = []
    for r in conn.execute(f"SELECT id, x, y, w, h FROM animals WHERE id IN ({marks}) AND source = 'detector'", new_ids):
        if any(iou((r["x"], r["y"], r["w"], r["h"]), (d["x"], d["y"], d["w"], d["h"])) >= EXISTING_IOU for d in drawn):
            conn.execute("DELETE FROM animals WHERE id = ?", (int(r["id"]),))
            dropped.append(int(r["id"]))
    if dropped:
        _count_animals(conn, item_id)
    return dropped
