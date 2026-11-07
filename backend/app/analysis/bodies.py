"""Person boxes (YOLOX through app.ml.persons), matched to faces, with torso crops for clothes matching
(docs/ARCHITECTURE.md sections 7 and 10).

CONTRACT.
- BodyRecord: normalised box, det_score, face_index (index into the item's FaceRecord list, or None),
  torso (BGR crop for SigLIP: the upper TORSO_SHARE of the box), crop (RGB, at most CROP_SIDE px).
- detect_bodies(img, img_bgr, faces) -> list[BodyRecord] (empty when app.ml.persons is unavailable).
- match_faces(boxes_px, face_boxes_px) -> list[int | None]: a body takes the face whose centre lies
  inside its upper half, largest face first; each face goes to at most one body.
- upper_crop(img_bgr, box) -> BGR crop of the upper TORSO_SHARE of a body box.
- encode_crop(img) -> WebP bytes for storage.body_crop_path (CROP_SIDE px long side).
"""
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings

log = logging.getLogger(__name__)

CROP_SIDE = 256
CROP_QUALITY = 82
TORSO_SHARE = 0.6
MIN_BOX_PX = 16
MAX_BODIES = 40


@dataclass
class BodyRecord:
    x: float
    y: float
    w: float
    h: float
    det_score: float
    face_index: int | None
    torso: np.ndarray | None
    crop: Image.Image
    box_px: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def match_faces(boxes: list[tuple[float, float, float, float]],
                face_boxes: list[tuple[float, float, float, float]]) -> list[int | None]:
    """Pair body boxes with face boxes (both x1, y1, x2, y2 pixels). A face can belong to a body when
    its centre lies inside the body box and in the body's upper half. Larger faces are placed first;
    among candidate bodies, the one whose horizontal centre is nearest the face centre wins."""
    result: list[int | None] = [None] * len(boxes)
    order = sorted(range(len(face_boxes)),
                   key=lambda i: -((face_boxes[i][2] - face_boxes[i][0]) * (face_boxes[i][3] - face_boxes[i][1])))
    for fi in order:
        fx1, fy1, fx2, fy2 = face_boxes[fi]
        cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        best: tuple[float, int] | None = None
        for bi, (x1, y1, x2, y2) in enumerate(boxes):
            if result[bi] is not None:
                continue
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            if cy > y1 + (y2 - y1) * 0.5:
                continue
            d = abs((x1 + x2) / 2.0 - cx) / max(x2 - x1, 1.0)
            if best is None or d < best[0]:
                best = (d, bi)
        if best is not None:
            result[best[1]] = fi
    return result


def upper_crop(img_bgr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    l, t = max(0, int(round(x1))), max(0, int(round(y1)))
    r = min(w, int(round(x2)))
    b = min(h, int(round(y1 + (y2 - y1) * TORSO_SHARE)))
    if r - l < MIN_BOX_PX or b - t < MIN_BOX_PX:
        return None
    return np.ascontiguousarray(img_bgr[t:b, l:r]).copy()


def _crop_rgb(img: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    W, H = img.size
    x1, y1, x2, y2 = box
    c = img.crop((max(0, int(x1)), max(0, int(y1)), min(W, int(round(x2))), min(H, int(round(y2)))))
    c = c.convert("RGB")
    if max(c.size) > CROP_SIDE:
        c.thumbnail((CROP_SIDE, CROP_SIDE), Image.Resampling.LANCZOS)
    return c


def encode_crop(img: Image.Image) -> bytes:
    if max(img.size) > CROP_SIDE:
        img = img.copy()
        img.thumbnail((CROP_SIDE, CROP_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "WEBP", quality=CROP_QUALITY, method=4)
    return buf.getvalue()


def available() -> bool:
    """settings.bodies_enabled, models allowed (PCC_NO_MODELS unset) and app.ml.persons present."""
    if not settings.bodies_enabled or os.environ.get("PCC_NO_MODELS") == "1":
        return False
    try:
        from app.ml import persons
    except ImportError:
        return False
    try:
        return bool(persons.available())
    except Exception as e:  # a broken model must not fail the analysis
        log.warning("person detector unavailable: %s", e)
        return False


def detect_bodies(img: Image.Image, img_bgr: np.ndarray, faces: list[Any]) -> list[BodyRecord]:
    """Person boxes in the analysed image (score >= settings.body_det_threshold), matched to the faces
    (FaceRecord.bbox_px). Raises ImportError when app.ml.persons is missing; callers check available()."""
    from app.ml import persons

    H, W = img_bgr.shape[:2]
    raw = persons.detect(img_bgr, threshold=settings.body_det_threshold)
    boxes: list[tuple[float, float, float, float]] = []
    scores: list[float] = []
    for x1, y1, x2, y2, score in sorted(raw, key=lambda b: -float(b[4]))[:MAX_BODIES]:
        x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
        x2, y2 = min(float(W), float(x2)), min(float(H), float(y2))
        if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
            continue
        boxes.append((x1, y1, x2, y2))
        scores.append(float(score))
    face_boxes = [tuple(f.bbox_px) if getattr(f, "bbox_px", None) else
                  (f.x * W, f.y * H, (f.x + f.w) * W, (f.y + f.h) * H) for f in faces]
    matched = match_faces(boxes, face_boxes)  # type: ignore[arg-type]
    out: list[BodyRecord] = []
    for box, score, fi in zip(boxes, scores, matched):
        x1, y1, x2, y2 = box
        out.append(BodyRecord(
            x=x1 / W, y=y1 / H, w=(x2 - x1) / W, h=(y2 - y1) / H, det_score=score, face_index=fi,
            torso=upper_crop(img_bgr, box), crop=_crop_rgb(img, box), box_px=box,
        ))
    return out
