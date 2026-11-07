"""Animal boxes (section 22): the YOLOX detector's COCO animal classes, with crops for the pet
identities and for storage.

CONTRACT.
- AnimalRecord: normalised box, det_score, species, frame_time_s, look (BGR crop padded by
  PAD_SHARE for SigLIP), crop (RGB for storage.animal_crop_path, at most CROP_SIDE px).
- available() -> bool: settings.pets_enabled, models allowed (PCC_NO_MODELS unset), app.ml.persons present.
- thresholds() -> {class_id: threshold}: what to ask the detector for.
- from_boxes(img, img_bgr, found, frame_time_s=None) -> list[AnimalRecord]: records from a
  persons.detect_classes result (best score first per species, at most MAX_PER_SPECIES each).
- detect_animals(img, img_bgr, frame_time_s=None) -> list[AnimalRecord]: one detector run.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from app.analysis import bodies as bodies_mod
from app.config import settings

log = logging.getLogger(__name__)

PAD_SHARE = 0.10
MIN_BOX_PX = 24
MAX_PER_SPECIES = 6
MAX_ANIMALS = 12
SPECIES = ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")


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
    return out[:MAX_ANIMALS]


def detect_animals(img: Image.Image, img_bgr: np.ndarray, frame_time_s: float | None = None) -> list[AnimalRecord]:
    """Animal boxes from one detector run. Callers check available()."""
    from app.ml import persons

    return from_boxes(img, img_bgr, persons.detect_classes(img_bgr, thresholds()), frame_time_s)


def encode_crop(img: Image.Image) -> bytes:
    return bodies_mod.encode_crop(img)
