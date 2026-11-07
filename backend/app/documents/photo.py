"""ID document photo crop.

Contract: crop_id_photo(img_bgr, dest) -> bool. Finds faces with app.ml.faces (detection only), takes
the largest, crops a square around it with a 25 percent margin on every side (white padding where the
square leaves the page), resizes it to SIZE px and writes a WebP to dest (atomically). False when no
face is found or the face model is unavailable.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings

log = logging.getLogger(__name__)

SIZE = 320
MARGIN = 0.25
MIN_FACE_PX = 24


def _faces(img_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
    from app.ml import faces as ml_faces

    if not ml_faces.available():
        return []
    eng = ml_faces.get_engine()
    threshold = settings.face_det_threshold
    if hasattr(eng, "detect_multiscale"):
        found = eng.detect_multiscale(img_bgr, (640, 1280), threshold)
    else:
        found = eng.analyze(img_bgr, (640, 1280), threshold, MIN_FACE_PX)
    out = []
    for f in found:
        x0, y0, x1, y1 = (float(v) for v in f.bbox[:4])
        if min(x1 - x0, y1 - y0) >= MIN_FACE_PX:
            out.append((x0, y0, x1, y1))
    return out


def square_crop(img_bgr: np.ndarray, box: tuple[float, float, float, float], margin: float = MARGIN) -> Image.Image:
    """Square crop centred on box, side = longer box side * (1 + 2 * margin), padded with white."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = max(1, int(round(side)))
    h, w = img_bgr.shape[:2]
    rgb = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    src = rgb.crop((max(0, left), max(0, top), min(w, left + s), min(h, top + s)))
    canvas.paste(src, (max(0, -left), max(0, -top)))
    return canvas


def crop_id_photo(img_bgr: np.ndarray, dest: Path) -> bool:
    try:
        boxes = _faces(img_bgr)
    except Exception:
        log.warning("face detection for the ID photo failed", exc_info=True)
        return False
    if not boxes:
        return False
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    crop = square_crop(img_bgr, box).resize((SIZE, SIZE), Image.LANCZOS)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".{os.getpid()}.tmp")
    crop.save(tmp, "WEBP", quality=settings.preview_quality)
    os.replace(tmp, dest)
    return True
