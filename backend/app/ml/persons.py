"""Person detector: YOLOX-S (Megvii release 0.1.1rc0, COCO, class 0 = person), onnxruntime.

Contract (docs/ARCHITECTURE.md sections 8 and 10):
  available() -> bool                         yolox_s.onnx present (app.ml.models 'persons')
  detect(img_bgr, threshold=0.5) -> list[(x0, y0, x1, y1, score)]   pixel boxes in img_bgr, clipped,
                                              best score first, after NMS (NMS_IOU)
  get_engine() -> PersonEngine; reset(); loaded(); provider()

Preprocessing follows YOLOX's demo (yolox/data/data_augment.py preproc, legacy=False): the BGR image
is resized by r = min(640 / h, 640 / w) with bilinear interpolation, pasted at the top-left of a
640x640 canvas filled with 114, and fed as float32 CHW BGR without normalisation. The exported model
returns raw grid outputs (1, 8400, 85): x, y, w, h relative to the grid cell, then the objectness and
the 80 class probabilities, already passed through a sigmoid. Boxes are decoded with strides 8, 16 and
32 as in yolox/utils/demo_utils.demo_postprocess, the person score is obj * cls[0], and the boxes are
divided by r to get back to image pixels.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_FILE = "yolox_s.onnx"
INPUT_SIZE = 640
STRIDES = (8, 16, 32)
PAD_VALUE = 114
NMS_IOU = 0.45
PERSON_CLASS = 0
MIN_BOX_PX = 2.0

Box = tuple[float, float, float, float, float]


def letterbox(img_bgr: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float]:
    """Resize into a size x size canvas (top-left, pad 114). Returns (CHW float32 blob, r)."""
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    blob = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)
    return blob, r


def _grids(size: int = INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for stride in STRIDES:
        n = size // stride
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((grid.shape[0], 1), stride))
    return np.concatenate(grids, 0).astype(np.float32), np.concatenate(strides, 0).astype(np.float32)


_GRIDS, _STRIDE_COL = _grids()


def decode(raw: np.ndarray, r: float, width: int, height: int, threshold: float,
           nms_iou: float = NMS_IOU) -> list[Box]:
    """Raw (8400, 85) or (1, 8400, 85) output -> person boxes in original image pixels."""
    out = np.asarray(raw, dtype=np.float32).reshape(-1, raw.shape[-1])
    if out.shape[0] != _GRIDS.shape[0]:
        raise ValueError(f"unexpected YOLOX output shape {raw.shape}")
    scores = out[:, 4] * out[:, 5 + PERSON_CLASS]
    keep = np.where(scores >= threshold)[0]
    if keep.size == 0:
        return []
    o = out[keep]
    xy = (o[:, :2] + _GRIDS[keep]) * _STRIDE_COL[keep]
    wh = np.exp(np.clip(o[:, 2:4], -20, 20)) * _STRIDE_COL[keep]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1) / r
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    s = scores[keep]
    ok = ((boxes[:, 2] - boxes[:, 0]) >= MIN_BOX_PX) & ((boxes[:, 3] - boxes[:, 1]) >= MIN_BOX_PX)
    boxes, s = boxes[ok], s[ok]
    if len(s) == 0:
        return []
    order = nms(boxes, s, nms_iou)
    return [(float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3]), float(s[i]))
            for i in order]


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy NMS; indices of kept boxes, best score first."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ovr = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[ovr <= iou]
    return keep


def _as_bgr(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return np.ascontiguousarray(arr[:, :, :3])
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got shape {arr.shape}")
    return arr


class PersonEngine:
    """One YOLOX-S session. Thread-safe; runs are serialised."""

    def __init__(self, model_dir: Path, threads: int | None = None):
        from app.ml import runtime

        model = Path(model_dir) / MODEL_FILE
        if not model.is_file():
            raise FileNotFoundError(f"person model missing: {model} (run scripts/download_models.py)")
        self._sess = runtime.session(model, threads=threads)
        self.provider = runtime.session_provider(self._sess)
        self._in = self._sess.get_inputs()[0].name
        self._lock = threading.Lock()

    def detect(self, img_bgr: np.ndarray, threshold: float = 0.5) -> list[Box]:
        img = _as_bgr(img_bgr)
        h, w = img.shape[:2]
        if h < 8 or w < 8:
            return []
        blob, r = letterbox(img)
        with self._lock:
            raw = self._sess.run(None, {self._in: blob[None]})[0]
        return decode(raw, r, w, h, threshold)


_engine: PersonEngine | None = None
_engine_lock = threading.Lock()


def available() -> bool:
    from app.ml import models

    return models.present("persons")


def get_engine() -> PersonEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from app.config import settings
                from app.ml import models

                _engine = PersonEngine(models.model_dir("persons"), settings.ort_threads)
                log.info("person detector loaded on %s", _engine.provider)
    return _engine


def detect(img_bgr: np.ndarray, threshold: float = 0.5) -> list[Box]:
    """Person boxes (x0, y0, x1, y1, score) in img_bgr pixels, best first."""
    return get_engine().detect(img_bgr, threshold)


def reset() -> None:
    global _engine
    with _engine_lock:
        _engine = None


def loaded() -> bool:
    return _engine is not None


def provider() -> str | None:
    e = _engine
    return e.provider if e is not None else None
