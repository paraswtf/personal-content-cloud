"""The 'analyze' job: stage 2 of the processing pipeline for one item (docs/ARCHITECTURE.md sections 7, 9, 10).

CONTRACT.
- analyze_item(conn, item_id, version) -> None. A no-op when the item is gone, its version differs
  or analysis_status is already 'done' (reprocess resets it to 'pending' first). Otherwise:
  1. loads the analysis input: the oriented original at most ANALYSIS_SIDE px (images, RAW included,
     falling back to the large preview), the page 1 render (pdf, document), or sampled frames plus the
     poster (videos);
  2. runs every model that is available (each one optional; a missing one is listed under
     quality_scores['skipped'] and category_scores['skipped'], and the rows it produced on an earlier
     run are kept: embeddings and labels without SigLIP, faces and bodies without the face engine,
     bodies without the person detector): SigLIP embeddings (frame 0 plus up to
     settings.search_video_frames video frames) into item_embeddings, zero-shot category scores and
     the top 5 labels into item_labels, the NSFW classifier, faces (SCRFD + ArcFace, video tracks),
     person boxes (YOLOX) with torso crops embedded by SigLIP into torso_embedding;
  3. measures blur and exposure, finds bursts;
  4. in one short write transaction: embeddings, labels, faces (+ crops, carry-over of names on a
     reprocess), bodies (+ crops), clustering (cluster.assign_new_faces), category, nsfw and mis-click
     decisions (never overriding *_source='user'), burst flags of the neighbours, face_count,
     body_count, analysis_status 'done'; faces_version and search_version are bumped;
  5. afterwards: guess.guess_item for the open boxes, the FTS row (labels, people), and the text job for
     images with doc_likeness >= settings.doc_likeness_min (app.ingest.enqueue_text).
- mark_failed(conn, item_id, version): analysis_status 'error' (the worker calls it after the last retry).
Heavy work (decode, inference, ffmpeg) runs outside the write transaction.
"""
from __future__ import annotations

import dataclasses
import importlib
import io
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app import storage
from app.analysis import animals as animals_mod
from app.analysis import bodies as bodies_mod
from app.analysis import categories, cluster, quality
from app.analysis import faces as faces_mod
from app.analysis.index import get_index
from app.config import settings
from app.db import bump_faces_version, bump_search_version, faces_version, tx, kv_get
from app.util import json_dumps, json_loads, now_iso

log = logging.getLogger(__name__)

ANALYSIS_SIDE = 2048
FRAME_KEEP_SIDE = 448  # video frames kept in memory for SigLIP / NSFW / exposure
NSFW_VIDEO_FRAMES = 16
ANIMAL_VIDEO_FRAMES = 6  # sampled frames the animal detector looks at (plus the poster)
ANIMAL_MIN_AREA = 0.05   # an animal box this large (share of the frame) makes the photo a pet photo
LABEL_COUNT = 5


class _Stale(Exception):
    """The item changed (or vanished) while it was being analysed."""


@dataclass
class Visual:
    """What the models look at."""

    img: Image.Image | None = None  # oriented RGB analysis image (images, document page 1, video poster)
    bgr: np.ndarray | None = None  # the same as BGR
    source_scale: float = 1.0  # original long side / analysis image long side
    frames: list[tuple[float, np.ndarray]] = field(default_factory=list)  # videos: (time, small BGR)
    poster_time: float | None = None
    faces: list[faces_mod.FaceRecord] = field(default_factory=list)
    faces_done: bool = False  # faces already found while decoding (videos)


@dataclass
class Result:
    embeddings: list[tuple[int, float | None, np.ndarray]] = field(default_factory=list)  # (frame_index, time, emb)
    model: str | None = None
    labels: list[tuple[str, float]] = field(default_factory=list)
    category_scores: dict[str, Any] = field(default_factory=dict)
    quality_scores: dict[str, Any] = field(default_factory=dict)
    nsfw_score: float | None = None
    explicit_p: float | None = None
    suggestive_p: float | None = None
    safe_p: float | None = None
    screenshot_p: float | None = None
    doc_p: float | None = None
    meme_p: float | None = None  # section 29
    density: float | None = None
    faces: list[faces_mod.FaceRecord] = field(default_factory=list)
    face_torsos: list[np.ndarray | None] = field(default_factory=list)  # SigLIP vectors per face
    bodies: list[bodies_mod.BodyRecord] = field(default_factory=list)
    body_torsos: list[np.ndarray | None] = field(default_factory=list)
    animals: list[animals_mod.AnimalRecord] = field(default_factory=list)  # section 22
    animal_looks: list[np.ndarray | None] = field(default_factory=list)  # SigLIP vectors per animal crop
    skipped: list[str] = field(default_factory=list)
    faces_ran: bool = False  # the face engine looked at this item (r.faces is the real answer)
    bodies_ran: bool = False  # the person detector looked at this item
    animals_ran: bool = False  # the animal classes were asked for


# ------------------------------------------------------------------ model access
def models_disabled() -> bool:
    """PCC_NO_MODELS=1 (tests, parallel agents) turns every model off, as if its files were missing."""
    return os.environ.get("PCC_NO_MODELS") == "1"


def _model(name: str) -> ModuleType | None:
    """app.ml.<name> when it is importable and its model files are present, else None."""
    if models_disabled():
        return None
    try:
        mod = importlib.import_module(f"app.ml.{name}")
    except ImportError as e:
        log.debug("app.ml.%s not importable: %s", name, e)
        return None
    try:
        return mod if mod.available() else None
    except Exception as e:  # a broken model install must not fail the item
        log.warning("app.ml.%s availability check failed: %s", name, e)
        return None


def _resize_max(bgr: np.ndarray, side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= side:
        return np.ascontiguousarray(bgr)
    s = side / max(h, w)
    return cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def _to_bgr(img: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8)[:, :, ::-1])


def _open_rgb(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def _media_info(item: sqlite3.Row) -> Any:
    """A MediaInfo for the video helpers built from the item row (fields the class has)."""
    from app.media.probe import MediaInfo

    values = {
        "kind": item["kind"], "mime": item["mime"], "ext": item["ext"], "width": int(item["width"] or 0),
        "height": int(item["height"] or 0), "duration_s": item["duration_s"], "bitrate": item["bitrate"],
        "video_codec": item["video_codec"], "fps": item["fps"],
        "has_audio": None if item["has_audio"] is None else bool(item["has_audio"]),
        "page_count": item["page_count"],
    }
    names = {f.name for f in dataclasses.fields(MediaInfo)}
    return MediaInfo(**{k: v for k, v in values.items() if k in names})


# ------------------------------------------------------------------ inputs
def _load_visual(item: sqlite3.Row, faces_engine: ModuleType | None, need_frames: bool) -> Visual:
    uid, version, kind = item["uid"], int(item["version"]), item["kind"]
    v = Visual()
    if kind == "image":
        original = storage.original_path(uid, version, item["ext"])
        img: Image.Image | None = None
        try:
            from app.media.images import load_oriented

            img = load_oriented(original, max_side=ANALYSIS_SIDE)
        except Exception as e:
            log.warning("analyze: %s: original not decodable (%s), using the preview", uid, e)
            img = _open_rgb(storage.preview_path(uid, version))
        if img is not None:
            v.img = img.convert("RGB")
            v.bgr = _to_bgr(v.img)
            longest = max(int(item["width"] or 0), int(item["height"] or 0))
            v.source_scale = max(1.0, longest / max(v.img.size)) if longest else 1.0
    elif kind in ("pdf", "document"):
        img = _open_rgb(storage.page_path(uid, version, 1)) or _open_rgb(storage.preview_path(uid, version))
        if img is not None:
            v.img = img
            v.bgr = _to_bgr(img)
    elif kind == "video":
        original = storage.original_path(uid, version, item["ext"])
        info = _media_info(item)
        try:
            from app.media.video import poster_time

            v.poster_time = float(poster_time(info))
        except Exception:
            v.poster_time = 0.0
        poster = _open_rgb(storage.preview_path(uid, version))
        if poster is None:
            try:
                from app.media.video import extract_frame

                arr = extract_frame(original, v.poster_time or 0.0, max_side=1280)
                if arr is not None:
                    poster = Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]))
            except Exception as e:
                log.warning("analyze: %s: no poster frame (%s)", uid, e)
        if poster is not None:
            v.img = poster
            v.bgr = _to_bgr(poster)

        def keep(t: float, bgr: np.ndarray) -> None:
            v.frames.append((float(t), _resize_max(bgr, FRAME_KEEP_SIDE)))

        if faces_engine is not None:
            v.faces = faces_mod.analyze_video(original, info, on_frame=keep)
            v.faces_done = True
        elif need_frames:
            from app.media.video import iter_frames

            frames = iter_frames(original, info, settings.video_sample_interval_s, settings.video_max_samples,
                                 faces_mod.VIDEO_MAX_SIDE)
            try:
                for t, bgr in frames:
                    keep(t, bgr)
            finally:
                close = getattr(frames, "close", None)
                if close is not None:
                    close()
    return v


def _spread(frames: list[tuple[float, np.ndarray]], k: int) -> list[tuple[float, np.ndarray]]:
    if not frames or k <= 0:
        return []
    if len(frames) <= k:
        return list(frames)
    idx = sorted({int(round(i)) for i in np.linspace(0, len(frames) - 1, k)})
    return [frames[i] for i in idx]


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v


# ------------------------------------------------------------------ the models
def _run_models(item: sqlite3.Row, v: Visual, siglip: ModuleType | None, nsfw: ModuleType | None,
                faces_engine: ModuleType | None) -> Result:
    kind = item["kind"]
    r = Result()
    visual = v.bgr is not None
    if not visual and kind != "other":
        r.skipped.append("image")

    # Faces (images; videos found them while decoding).
    if kind == "image" and visual:
        if faces_engine is not None:
            r.faces = faces_mod.analyze_image(v.img, source_scale=v.source_scale, img_bgr=v.bgr)  # type: ignore[arg-type]
            r.faces_ran = True
        else:
            r.skipped.append("faces")
    elif kind == "video":
        if v.faces_done:
            r.faces = v.faces
            r.faces_ran = True
        else:
            r.skipped.append("faces")

    # Person boxes (images only) and animal boxes (images and sampled video frames, section 22):
    # one YOLOX run per frame answers both.
    if kind == "image" and visual:
        want_bodies, want_animals = bodies_mod.available(), animals_mod.available()
        if want_bodies or want_animals:
            from app.ml import persons as persons_ml

            asks: dict[int, float] = {}
            if want_bodies:
                asks[persons_ml.PERSON_CLASS] = float(settings.body_det_threshold)
            if want_animals:
                asks.update(animals_mod.thresholds())
            found = persons_ml.detect_classes(v.bgr, asks)  # type: ignore[arg-type]
            if want_bodies:
                r.bodies = bodies_mod.detect_bodies(v.img, v.bgr, r.faces,  # type: ignore[arg-type]
                                                    raw=found.get(persons_ml.PERSON_CLASS, []))
                r.bodies_ran = True
            if want_animals:
                r.animals = animals_mod.from_boxes(v.img, v.bgr, found)  # type: ignore[arg-type]
                r.animals_ran = True
        if not want_bodies:
            r.skipped.append("bodies")
        if not want_animals:
            r.skipped.append("animals")
    elif kind == "video" and visual:
        if animals_mod.available():
            frames: list[tuple[float, Image.Image | None, np.ndarray]] = [(float(v.poster_time or 0.0), v.img, v.bgr)]  # type: ignore[list-item]
            frames += [(float(t), None, f) for t, f in _spread(v.frames, ANIMAL_VIDEO_FRAMES)]
            for t, img, bgr in frames:
                pil = img if img is not None else Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                r.animals.extend(animals_mod.detect_animals(pil, bgr, frame_time_s=t))
            r.animals = sorted(r.animals, key=lambda a: -a.det_score)[:animals_mod.MAX_ANIMALS]
            r.animals_ran = True
        else:
            r.skipped.append("animals")

    # SigLIP: frame 0, video frames, torso crops, zero-shot scores.
    if siglip is not None and visual:
        engine = siglip.get_engine()
        r.model = str(getattr(siglip, "MODEL_ID", "siglip"))
        sample = _spread(v.frames, settings.search_video_frames) if kind == "video" else []
        embs = np.asarray(engine.embed_images([v.bgr, *[f for _, f in sample]]), dtype=np.float32)
        r.embeddings.append((0, v.poster_time if kind == "video" else None, _norm(embs[0])))
        for k, ((t, _), e) in enumerate(zip(sample, embs[1:]), start=1):
            r.embeddings.append((k, float(t), _norm(e)))
        scene = np.stack([e for _, _, e in r.embeddings])
        frame0 = scene[:1]
        r.category_scores = categories.category_scores(engine, scene)
        r.labels = categories.top_labels(engine, scene, k=LABEL_COUNT)
        if kind in ("image", "video"):
            r.explicit_p = categories.binary_p(engine, scene, categories.EXPLICIT_PROMPTS,
                                               categories.NOT_EXPLICIT_PROMPTS, reduce="max")
            r.category_scores["explicit_p"] = round(r.explicit_p, 4)
            r.suggestive_p = categories.binary_p(engine, scene, categories.SUGGESTIVE_PROMPTS,
                                                 categories.NOT_SUGGESTIVE_PROMPTS, reduce="max")
            r.safe_p = categories.binary_p(engine, scene, categories.SAFE_SKIN_PROMPTS,
                                           categories.NOT_SAFE_SKIN_PROMPTS, reduce="mean")
            r.category_scores["suggestive_p"] = round(r.suggestive_p, 4)
            r.category_scores["safe_p"] = round(r.safe_p, 4)
        if kind == "image":
            r.doc_p = categories.binary_p(engine, frame0, categories.DOCUMENT_PROMPTS, categories.NOT_DOCUMENT_PROMPTS)
            r.screenshot_p = categories.binary_p(engine, frame0, categories.SCREENSHOT_PROMPTS,
                                                 categories.NOT_SCREENSHOT_PROMPTS)
            r.category_scores["doc_p"] = round(r.doc_p, 4)
            r.category_scores["screenshot_p"] = round(r.screenshot_p, 4)
            r.quality_scores["accident_p"] = round(float(quality.accident_probability(engine, frame0)[0]), 4)
            r.quality_scores["motion_blur_p"] = round(float(quality.motion_blur_probability(engine, frame0)[0]), 4)
        if kind in ("image", "video"):
            r.meme_p = categories.binary_p(engine, frame0, categories.MEME_PROMPTS, categories.NOT_MEME_PROMPTS)
            r.category_scores["meme_p"] = round(r.meme_p, 4)
        crops = [f.torso for f in r.faces] + [b.torso for b in r.bodies] + [a.look for a in r.animals]
        valid = [i for i, c in enumerate(crops) if c is not None and c.size]
        vecs: list[np.ndarray | None] = [None] * len(crops)
        if valid:
            out = np.asarray(engine.embed_images([crops[i] for i in valid]), dtype=np.float32)
            for i, e in zip(valid, out):
                vecs[i] = _norm(e)
        nf, nb = len(r.faces), len(r.bodies)
        r.face_torsos = vecs[:nf]
        r.body_torsos = vecs[nf:nf + nb]
        r.animal_looks = vecs[nf + nb:]
    else:
        if visual:
            r.skipped.append("siglip")
        r.face_torsos = [None] * len(r.faces)
        r.body_torsos = [None] * len(r.bodies)
        r.animal_looks = [None] * len(r.animals)

    # NSFW classifier (images and videos).
    if kind in ("image", "video") and visual:
        if nsfw is not None:
            imgs = [v.bgr] + [f for _, f in _spread(v.frames, NSFW_VIDEO_FRAMES)]
            scores = [float(s) for s in nsfw.score(imgs)]
            r.nsfw_score = round(max(scores), 4) if scores else None
        else:
            r.skipped.append("nsfw")

    # Pixel measures.
    if kind == "image" and visual:
        r.quality_scores.update(quality.measure(v.bgr))  # type: ignore[arg-type]
        r.density = categories.text_density(v.bgr)  # type: ignore[arg-type]
        r.category_scores["text_density"] = r.density
    elif kind == "video" and (v.frames or visual):
        frames = [f for _, f in v.frames] or [v.bgr]
        mean, p95, clipped = quality.exposure_many(frames)  # type: ignore[arg-type]
        r.quality_scores.update({"mean_lum": round(mean, 3), "p95_lum": round(p95, 3), "clipped_frac": round(clipped, 4)})
    return r


# ------------------------------------------------------------------ writes
def _drop_animal_faces(r: Result) -> int:
    """Section 22: a face box inside an animal box with no person box over it is the animal's face.
    Drops those faces (with their torso vectors) and re-points the bodies' face indexes."""
    hits = animals_mod.animal_faces(
        [(f.x, f.y, f.w, f.h) for f in r.faces], [(a.x, a.y, a.w, a.h) for a in r.animals],
        [(b.x, b.y, b.w, b.h) for b in r.bodies],
    )
    if not hits:
        return 0
    gone = set(hits)
    keep = [i for i in range(len(r.faces)) if i not in gone]
    new_index = {old: new for new, old in enumerate(keep)}
    r.faces = [r.faces[i] for i in keep]
    if len(r.face_torsos) > len(keep):
        r.face_torsos = [r.face_torsos[i] for i in keep]
    for b in r.bodies:
        if b.face_index is not None:
            b.face_index = new_index.get(b.face_index)
    return len(hits)


def _settle_species(r: Result, siglip: ModuleType | None) -> int:
    """Section 22: each animal record's species after the owner's look-alike vote and the zero-shot
    check; the scores are kept on the record. Returns how many records changed species."""
    if not r.animals:
        return 0
    engine = None
    if siglip is not None:
        try:
            engine = siglip.get_engine()
        except Exception:  # noqa: BLE001 - no engine, the detector's class stands
            engine = None
    changed = 0
    looks = list(r.animal_looks) + [None] * (len(r.animals) - len(r.animal_looks))
    for rec, look in zip(r.animals, looks):
        species, scores = animals_mod.settle_species(look, rec.species, engine=engine)
        rec.species_scores = scores
        if species != rec.species:
            rec.species = species
            changed += 1
    return changed


def _encode_face_crop(img: Image.Image) -> bytes:
    if img.size != (faces_mod.CROP_SIZE, faces_mod.CROP_SIZE):
        img = img.resize((faces_mod.CROP_SIZE, faces_mod.CROP_SIZE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "WEBP", quality=faces_mod.CROP_QUALITY, method=4)
    return buf.getvalue()


def _blob(v: np.ndarray | None) -> bytes | None:
    return None if v is None else np.asarray(v, dtype=np.float32).tobytes()


def _repoint_covers(conn: sqlite3.Connection, item_id: int, old_ids: list[int], new_ids: list[int], now: str) -> None:
    """Covers that point at a re-detected face move to its carried-over successor."""
    marks = ",".join("?" * len(old_ids))
    for p in conn.execute(f"SELECT id FROM persons WHERE cover_face_id IN ({marks})", old_ids).fetchall():
        succ = None
        if new_ids:
            nmarks = ",".join("?" * len(new_ids))
            succ = conn.execute(
                f"SELECT id FROM faces WHERE item_id = ? AND person_id = ? AND id IN ({nmarks}) "
                "ORDER BY quality DESC, id LIMIT 1",
                (item_id, p["id"], *new_ids),
            ).fetchone()
        conn.execute("UPDATE persons SET cover_face_id = ?, updated_at = ? WHERE id = ?",
                     (succ["id"] if succ else None, now, p["id"]))


def _carry_bodies(conn: sqlite3.Connection, old_bodies: list[sqlite3.Row], new_ids: list[int]) -> None:
    """User-assigned faceless bodies keep their person on the best-overlapping new body (IoU >= 0.5);
    faceless bodies with rejected guesses pass those rejections on the same way."""
    rejected = cluster.rejected_ids(conn, "body", [int(b["id"]) for b in old_bodies])
    olds = [b for b in old_bodies
            if b["face_id"] is None and (b["assigned_by"] == "user" or int(b["id"]) in rejected)]
    if not olds or not new_ids:
        return
    marks = ",".join("?" * len(new_ids))
    news = conn.execute(f"SELECT id, x, y, w, h FROM bodies WHERE id IN ({marks}) AND face_id IS NULL", new_ids).fetchall()
    used: set[int] = set()
    for o in olds:
        best: tuple[float, int] | None = None
        for n in news:
            if n["id"] in used:
                continue
            s = cluster.iou((o["x"], o["y"], o["w"], o["h"]), (n["x"], n["y"], n["w"], n["h"]))
            if s >= 0.5 and (best is None or s > best[0]):
                best = (s, int(n["id"]))
        if best is not None:
            used.add(best[1])
            if o["assigned_by"] == "user":
                conn.execute("UPDATE bodies SET person_id = ?, assigned_by = 'user' WHERE id = ?",
                             (o["person_id"], best[1]))
            if int(o["id"]) in rejected:
                cluster.move_rejections(conn, "body", int(o["id"]), best[1])


def _burst_entry(roles: dict[int, str | None], item_id: int, size: int) -> dict[str, Any] | None:
    if item_id not in roles:
        return None
    best_uid = roles[item_id]
    if best_uid is None:
        return {"burst_best": {"size": size}}
    return {"burst": {"best_uid": best_uid, "size": size}}


def _analyze_requested_again(conn: sqlite3.Connection, item_id: int) -> bool:
    """True when another analyze job for the item is queued (a reprocess requested during this run)."""
    row = conn.execute(
        "SELECT 1 FROM jobs WHERE type = 'analyze' AND status = 'queued' "
        "AND json_extract(payload, '$.item_id') = ? LIMIT 1",
        (int(item_id),),
    ).fetchone()
    return row is not None


def analyze_item(conn: sqlite3.Connection, item_id: int, version: int) -> None:
    item = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if item is None or int(item["version"]) != int(version):
        log.info("analyze: item %s version %s is gone or outdated, skipping", item_id, version)
        return
    if item["analysis_status"] == "done":
        return
    if item["companion_of"]:
        # A Live Photo's clip: the still carries the analysis (section 20).
        with tx(conn):
            conn.execute("UPDATE items SET analysis_status = 'done', updated_at = ? WHERE id = ?", (now_iso(), item_id))
        return
    uid, kind = item["uid"], item["kind"]

    siglip = _model("siglip")
    nsfw = _model("nsfw")
    faces_engine = _model("faces") if kind in ("image", "video") else None
    v = _load_visual(item, faces_engine, need_frames=(siglip is not None or nsfw is not None))
    r = _run_models(item, v, siglip, nsfw, faces_engine)
    del v

    # Decisions that need only this item.
    good_face = any(cluster.is_good_face(f.det_score, f.size_px) for f in r.faces)
    # Video frames are sampled at 640 px, so their faces rarely pass the good-face size: any tracked face counts.
    has_person = good_face or bool(r.bodies) or (kind == "video" and bool(r.faces))
    doc_like = categories.doc_likeness(r.doc_p, r.density) if kind == "image" else (1.0 if kind in ("pdf", "document") else None)
    screen_heur = kind == "image" and categories.screenshot_heuristic(
        item["ext"], item["camera"], item["width"], item["height"], item["exif_datetime"])
    if r.skipped:
        r.quality_scores["skipped"] = sorted(set(r.skipped))
        r.category_scores["skipped"] = sorted(set(r.skipped))
    # A model that did not run must not wipe what it found before (user-named faces, labels...).
    keep_embeddings = r.model is None
    keep_faces = kind in ("image", "video") and not r.faces_ran
    # Without faces, new person boxes would all look faceless: keep the old ones as well.
    keep_bodies = keep_faces or (kind == "image" and not r.bodies_ran)
    if keep_faces:
        r.faces, r.face_torsos = [], []
    if keep_bodies:
        r.bodies, r.body_torsos = [], []
    keep_animals = kind in ("image", "video") and not r.animals_ran
    if keep_animals:
        r.animals, r.animal_looks = [], []
    if kind == "image" and r.faces and r.animals:
        _drop_animal_faces(r)
    _settle_species(r, siglip)
    face_crops = [_encode_face_crop(f.crop) for f in r.faces]
    body_crops = [bodies_mod.encode_crop(b.crop) for b in r.bodies]
    animal_crops = [animals_mod.encode_crop(a.crop) for a in r.animals]
    has_animal = any(a.w * a.h >= ANIMAL_MIN_AREA for a in r.animals)

    from app.analysis import pets as pets_mod

    idx = get_index()
    idx.sync(conn)
    written: list[Path] = []
    trash: list[Path] = []
    roles: dict[int, str | None] = {}
    members: list[quality.BurstMember] = []
    try:
        with tx(conn):
            cur = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
            if cur is None or int(cur["version"]) != int(version) or cur["analysis_status"] == "done":
                raise _Stale()
            base_version = faces_version(conn)
            now = now_iso()

            # Embeddings and labels.
            if keep_embeddings:
                conn.execute("DELETE FROM item_embeddings WHERE item_id = ? AND version != ?", (item_id, int(version)))
            else:
                conn.execute("DELETE FROM item_embeddings WHERE item_id = ?", (item_id,))
            for frame_index, t, emb in r.embeddings:
                conn.execute(
                    "INSERT INTO item_embeddings(item_id, version, model, frame_index, frame_time_s, embedding, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (item_id, int(version), r.model, frame_index, t, _blob(emb), now),
                )
            if not keep_embeddings:
                conn.execute("DELETE FROM item_labels WHERE item_id = ?", (item_id,))
            # SigLIP missing now but it ran before: its scene scores still stand.
            prior_scene = keep_embeddings and conn.execute(
                "SELECT 1 FROM item_embeddings WHERE item_id = ? LIMIT 1", (item_id,)).fetchone() is not None
            for label, score in r.labels:
                conn.execute("INSERT OR REPLACE INTO item_labels(item_id, label, score) VALUES (?, ?, ?)",
                             (item_id, label, float(score)))

            # Faces.
            old_faces = conn.execute("SELECT * FROM faces WHERE item_id = ? ORDER BY id", (item_id,)).fetchall()
            if keep_faces:
                # Kept faces still say whether a person is in the picture.
                good_face = any(cluster.is_good_face(f["det_score"], f["size_px"]) for f in old_faces)
                has_person = has_person or good_face or (kind == "video" and bool(old_faces))
                old_faces = []
            old_ids = [int(f["id"]) for f in old_faces]
            old_people = {int(f["person_id"]) for f in old_faces if f["person_id"] is not None}
            new_face_ids: list[int] = []
            for rec, crop, torso in zip(r.faces, face_crops, r.face_torsos):
                fid = int(conn.execute(
                    "INSERT INTO faces(item_id, person_id, assigned_by, x, y, w, h, det_score, size_px, quality, "
                    "frame_time_s, spans, embedding, torso_embedding, created_at) "
                    "VALUES (?, NULL, 'auto', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (item_id, rec.x, rec.y, rec.w, rec.h, rec.det_score, int(rec.size_px), rec.quality,
                     rec.frame_time_s, json_dumps(rec.spans) if rec.spans is not None else None,
                     np.asarray(rec.embedding, dtype=np.float32).tobytes(), _blob(torso), now),
                ).lastrowid)
                new_face_ids.append(fid)
                path = storage.face_crop_path(fid)
                written.append(path)
                storage.write_bytes_atomic(path, crop)
            bump_faces_version(conn)
            if old_faces and new_face_ids:
                cluster.carry_over(conn, old_faces, new_face_ids)

            # Bodies (after the faces, so face_id can point at the new rows).
            old_bodies = conn.execute("SELECT * FROM bodies WHERE item_id = ? ORDER BY id", (item_id,)).fetchall()
            if keep_bodies:
                # Faceless bodies stay; bodies around a face go only when that face is re-detected.
                kept = [b for b in old_bodies if keep_faces or b["face_id"] is None]
                has_person = has_person or bool(kept)
                old_bodies = [b for b in old_bodies if not (keep_faces or b["face_id"] is None)]
            old_body_ids = [int(b["id"]) for b in old_bodies]
            old_people |= {int(b["person_id"]) for b in old_bodies if b["person_id"] is not None}
            new_body_ids: list[int] = []
            for rec, crop, torso in zip(r.bodies, body_crops, r.body_torsos):
                face_id = new_face_ids[rec.face_index] if rec.face_index is not None else None
                bid = int(conn.execute(
                    "INSERT INTO bodies(item_id, face_id, person_id, assigned_by, x, y, w, h, det_score, "
                    "torso_embedding, frame_time_s, created_at) VALUES (?, ?, NULL, 'auto', ?, ?, ?, ?, ?, ?, NULL, ?)",
                    (item_id, face_id, rec.x, rec.y, rec.w, rec.h, rec.det_score, _blob(torso), now),
                ).lastrowid)
                new_body_ids.append(bid)
                path = storage.body_crop_path(bid)
                written.append(path)
                storage.write_bytes_atomic(path, crop)
            if old_bodies and new_body_ids:
                _carry_bodies(conn, old_bodies, new_body_ids)
            if old_body_ids:
                marks = ",".join("?" * len(old_body_ids))
                conn.execute(f"DELETE FROM bodies WHERE id IN ({marks})", old_body_ids)
                cluster.drop_rejections(conn, "body", old_body_ids)
                trash.extend(storage.body_crop_path(b) for b in old_body_ids)
            if old_ids:
                _repoint_covers(conn, item_id, old_ids, new_face_ids, now)
                marks = ",".join("?" * len(old_ids))
                conn.execute(f"DELETE FROM faces WHERE id IN ({marks})", old_ids)
                cluster.drop_rejections(conn, "face", old_ids)
                trash.extend(storage.face_crop_path(f) for f in old_ids)
            idx.sync_item(conn, item_id, base_version)
            if new_face_ids:
                marks = ",".join("?" * len(new_face_ids))
                remaining = [int(x[0]) for x in conn.execute(
                    f"SELECT id FROM faces WHERE id IN ({marks}) AND person_id IS NULL AND assigned_by = 'auto' "
                    "ORDER BY id", new_face_ids,
                )]
                if remaining:
                    cluster.assign_new_faces(conn, remaining, exclude_face_ids=set(old_ids))
            cluster.sync_bodies_to_faces(conn, [item_id])

            # Animals (section 22): boxes, crops, look embeddings, then the pet identities.
            old_animals = [] if keep_animals else conn.execute(
                "SELECT * FROM animals WHERE item_id = ? ORDER BY id", (item_id,)).fetchall()
            if keep_animals:
                has_animal = has_animal or conn.execute(
                    "SELECT 1 FROM animals WHERE item_id = ? AND w * h >= ? LIMIT 1", (item_id, ANIMAL_MIN_AREA)
                ).fetchone() is not None
            old_animal_ids = [int(a["id"]) for a in old_animals]
            old_people |= {int(a["person_id"]) for a in old_animals if a["person_id"] is not None}
            new_animal_ids: list[int] = []
            for rec, crop, look in zip(r.animals, animal_crops, r.animal_looks):
                aid = int(conn.execute(
                    "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, "
                    "species_source, species_scores, frame_time_s, embedding, created_at) "
                    "VALUES (?, NULL, 'auto', ?, ?, ?, ?, ?, ?, 'auto', ?, ?, ?, ?)",
                    (item_id, rec.x, rec.y, rec.w, rec.h, rec.det_score, rec.species,
                     json_dumps(rec.species_scores) if rec.species_scores is not None else None, rec.frame_time_s,
                     _blob(look), now),
                ).lastrowid)
                new_animal_ids.append(aid)
                path = storage.animal_crop_path(aid)
                written.append(path)
                storage.write_bytes_atomic(path, crop)
            if old_animals and new_animal_ids:
                pets_mod.carry_over(conn, old_animals, new_animal_ids)
            if old_animal_ids:
                marks = ",".join("?" * len(old_animal_ids))
                conn.execute(f"DELETE FROM animals WHERE id IN ({marks})", old_animal_ids)
                trash.extend(storage.animal_crop_path(a) for a in old_animal_ids)
            if new_animal_ids:
                try:
                    pets_mod.assign_new_animals(conn, item_id)
                except Exception:  # noqa: BLE001 - an identity failure must not lose the analysis
                    log.exception("pets: assigning animals of item %s failed", item_id)
            if old_people:
                cluster.prune_empty_persons(conn, old_people)
            # items_fts.people lists the names on the item: refresh it when they may have changed.
            people_changed = bool(old_people) or conn.execute(
                "SELECT 1 FROM faces WHERE item_id = ? AND person_id IS NOT NULL UNION ALL "
                "SELECT 1 FROM bodies WHERE item_id = ? AND person_id IS NOT NULL UNION ALL "
                "SELECT 1 FROM animals WHERE item_id = ? AND person_id IS NOT NULL LIMIT 1",
                (item_id, item_id, item_id),
            ).fetchone() is not None

            # Category and nsfw.
            nsfw_override = bool(cur["nsfw"]) if cur["nsfw_source"] == "user" else None
            nsfw_score = r.nsfw_score
            if nsfw_override is None and r.nsfw_score is None and kind in ("image", "video"):
                # No classifier this time: an item the classifier put in the vault stays there.
                nsfw_override, nsfw_score = bool(cur["nsfw"]), cur["nsfw_score"]
            if prior_scene:
                old_scores = json_loads(cur["category_scores"], {}) or {}
                r.category_scores = {**{k: v for k, v in old_scores.items() if k != "skipped"}, **r.category_scores}
            # Section 18: the owner's labelled look-alikes decide before the rules do.
            emb0 = next((e for fi, _t, e in r.embeddings if fi == 0), None)
            sensitivity = kv_get(conn, "vault_sensitivity", settings.vault_sensitivity) or "normal"
            learned_cat = learned_nsfw = None
            if emb0 is not None and kind in ("image", "video"):
                try:
                    from app.learning import images as learned_images

                    learned_cat = learned_images.predict_category(emb0)
                    learned_nsfw = learned_images.predict_nsfw(
                        emb0, {"cls": nsfw_score, "explicit": r.explicit_p, "suggestive": r.suggestive_p, "safe": r.safe_p},
                        sensitivity=sensitivity)
                except Exception:  # noqa: BLE001 - learning never blocks analysis
                    log.exception("learning: image learner failed")
            dec = categories.decide(
                kind, nsfw_score=nsfw_score, explicit_p=r.explicit_p, nsfw_override=nsfw_override,
                screenshot_heur=screen_heur, screenshot_p=r.screenshot_p, doc_like=doc_like,
                has_person=has_person, scores=r.category_scores, learned=learned_cat, nsfw_learned=learned_nsfw,
                suggestive_p=r.suggestive_p, safe_p=r.safe_p, sensitivity=sensitivity, has_animal=has_animal,
            )
            if cur["category_source"] == "user":
                category = cur["category"]
            elif prior_scene and not dec.nsfw and cur["category"] and cur["category"] != "nsfw":
                category = cur["category"]  # decided with SigLIP before; nothing better to go on now
            else:
                category = dec.category
            new_nsfw = int(cur["nsfw"]) if cur["nsfw_source"] == "user" else int(dec.nsfw)

            # Mis-clicks and bursts.
            if kind == "image" and category not in ("documents", "screenshots") and "blur_var" in r.quality_scores:
                members = quality.burst_group(conn, cur, float(r.quality_scores["blur_var"]), nsfw=new_nsfw)
                roles = quality.burst_roles(members)
            burst = _burst_entry(roles, item_id, len(members))
            mis, reasons, _ = quality.decide(kind, category, r.quality_scores, good_face=good_face, burst=burst)
            if cur["misclick_source"] == "user":
                mis, reasons = bool(cur["misclick"]), json_loads(cur["misclick_reasons"], {}) or {}
            for m in members:
                if m.id == item_id or m.misclick_source == "user":
                    continue
                row = conn.execute("SELECT misclick_reasons, misclick_source FROM items WHERE id = ?", (m.id,)).fetchone()
                if row is None or row["misclick_source"] == "user":
                    continue
                other = quality.with_burst(json_loads(row["misclick_reasons"], {}) or {},
                                           _burst_entry(roles, m.id, len(members)))
                conn.execute(
                    "UPDATE items SET misclick = ?, misclick_reasons = ?, updated_at = ? WHERE id = ?",
                    (int(quality.is_misclick(other)), json_dumps(other) if other else None, now, m.id),
                )

            face_count = conn.execute("SELECT COUNT(*) FROM faces WHERE item_id = ?", (item_id,)).fetchone()[0]
            body_count = conn.execute("SELECT COUNT(*) FROM bodies WHERE item_id = ?", (item_id,)).fetchone()[0]
            animal_count = conn.execute("SELECT COUNT(*) FROM animals WHERE item_id = ?", (item_id,)).fetchone()[0]
            # A reprocess asked for while this run was going queued another analyze job (unique enqueue
            # only dedupes against queued jobs). Keep the status pending so that job runs in full
            # instead of returning at its 'done' check; the results of this run stay visible meanwhile.
            status = "pending" if _analyze_requested_again(conn, item_id) else "done"
            conn.execute(
                "UPDATE items SET category = ?, category_scores = ?, doc_likeness = ?, nsfw = ?, nsfw_score = ?, "
                "misclick = ?, misclick_reasons = ?, quality_scores = ?, face_count = ?, body_count = ?, "
                "animal_count = ?, analysis_status = ?, updated_at = ? WHERE id = ?",
                (category, json_dumps(r.category_scores) if r.category_scores else None, doc_like, new_nsfw,
                 nsfw_score, int(bool(mis)), json_dumps(reasons) if reasons else None,
                 json_dumps(r.quality_scores) if r.quality_scores else None, face_count, body_count, animal_count,
                 status, now, item_id),
            )
            # Forwards (section 29): the file's signals (from ingest, or now for older items) plus
            # the model's; the owner's decision stays.
            try:
                _decide_forward(conn, cur, kind, r.category_scores, doc_like, has_person, emb0)
            except Exception:  # noqa: BLE001 - never lose the analysis over the forward decision
                log.exception("forwards: decision failed for item %s", item_id)
            if cur["live_video_item_id"]:
                from app.media import live as live_media

                live_media.sync_companion(conn, item_id)
            tags_added = 0
            try:
                from app.learning import tags as learned_tags

                tags_added = learned_tags.apply_item(conn, item_id, emb=emb0, text=None, kind=kind)
            except Exception:  # noqa: BLE001 - learning never blocks analysis
                log.exception("learning: tag suggestions failed for item %s", item_id)
            bump_faces_version(conn)
            bump_search_version(conn)
            final_version = faces_version(conn)
    except _Stale:
        for p in written:
            storage.safe_unlink(p)
        log.info("analyze: item %s changed during analysis, skipping", uid)
        return
    except BaseException:
        idx.invalidate()
        for p in written:
            storage.safe_unlink(p)
        raise
    # The index already holds what this transaction committed: no full re-read on the next sync.
    idx.settle(conn, final_version)
    for p in trash:
        storage.safe_unlink(p)
    log.info("analyze: %s %s v%s -> %s, %d faces, %d bodies, %d animals%s", kind, uid, version, category,
             len(r.faces), len(r.bodies), len(r.animals), f", skipped {sorted(set(r.skipped))}" if r.skipped else "")
    _after(conn, item_id, int(version), kind, doc_like, bool(r.labels) or people_changed or tags_added > 0)


def _decide_forward(conn: sqlite3.Connection, cur: sqlite3.Row, kind: str, scores: dict[str, Any],
                    doc_like: float | None, has_person: bool, emb0: np.ndarray | None) -> None:
    """Section 29 inside the analysis transaction: the file-level signals ingest stored (computed now
    for an item from before the section), the model-level ones, then forwards.apply. Documents and
    other files only get received_via: the forward rule is for images and videos."""
    from app.analysis import forwards

    sig = json_loads(cur["forward_signals"], None) if cur["forward_signals"] else None
    if not isinstance(sig, dict) or "stripped" not in sig:
        path = None
        if kind == "image" and (cur["ext"] or "").lower() in (".jpg", ".jpeg"):
            p = storage.original_path(cur["uid"], int(cur["version"]), cur["ext"])
            path = p if p.exists() else None
        sig = forwards.file_signals(
            kind=kind, ext=cur["ext"], width=cur["width"], height=cur["height"], camera=cur["camera"],
            exif_datetime=cur["exif_datetime"], gps_lat=cur["gps_lat"], bitrate=cur["bitrate"],
            filename=cur["original_filename"], path=path)
    if kind in ("image", "video"):
        sig.update(forwards.model_signals(meme_p=scores.get("meme_p"), doc_likeness=doc_like,
                                          screenshot_p=scores.get("screenshot_p"), people=has_person))
    else:
        sig.update(forwards.model_signals(meme_p=None, doc_likeness=None, screenshot_p=None, people=False))
    forwards.apply(conn, int(cur["id"]), sig, emb=emb0)


def _after(conn: sqlite3.Connection, item_id: int, version: int, kind: str, doc_like: float | None,
           fts_changed: bool) -> None:
    """Follow-ups outside the analysis transaction; failures are logged, never raised."""
    from app.analysis import guess

    try:
        guess.guess_item(conn, item_id)
    except Exception as e:
        log.warning("analyze: guessing failed for item %s: %s", item_id, e)
    if fts_changed:
        guess.refresh_fts(conn, [item_id])
    if kind == "image" and doc_like is not None and doc_like >= settings.doc_likeness_min:
        row = conn.execute("SELECT text_status, version FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is not None and row["text_status"] == "none" and int(row["version"]) == version:
            try:
                from app.ingest import enqueue_text
            except ImportError:
                return
            try:
                enqueue_text(conn, item_id, version)
            except Exception as e:
                log.warning("analyze: could not queue text extraction for item %s: %s", item_id, e)


def mark_failed(conn: sqlite3.Connection, item_id: int, version: int) -> None:
    """After the last retry: analysis_status 'error' when the version still matches."""
    with tx(conn):
        conn.execute(
            "UPDATE items SET analysis_status = 'error', updated_at = ? WHERE id = ? AND version = ? "
            "AND analysis_status = 'pending'",
            (now_iso(), item_id, version),
        )
