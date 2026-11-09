"""Whole-body pose: DWPose-m (RTMPose-m, 133 COCO-WholeBody points), used to find limbs and hands on people boxes.

Contract (docs/ARCHITECTURE.md sections 8 and 34):
  available() -> bool                 pose/end2end.onnx present (app.ml.models 'pose')
  estimate(bgr, boxes_px) -> list     one (133, 3) float32 array per box: x, y in image pixels, score
                                      bgr: HxWx3 uint8 image in BGR order (OpenCV); boxes_px: (x0, y0, x1, y1) in px
  get_engine() -> PoseEngine; reset(); loaded(); provider()

Point order is COCO-WholeBody: 0-16 body (5/6 shoulders, 7/8 elbows, 9/10 wrists, 13/14 knees, 15/16 ankles),
17-22 feet, 23-90 face, 91-111 left hand, 112-132 right hand.

Preprocessing follows the model's config: the box grows by 1.25, is widened or heightened to the 192:256 aspect,
and is cut with an affine warp to 192 x 256 (RGB, mean 123.675 / 116.28 / 103.53, std 58.395 / 57.12 / 57.375, NCHW).
The two outputs are SimCC vectors (x: 384 bins, y: 512 bins, two bins per input pixel): the point is the argmax / 2
and its score is 0.5 * (max_x + max_y). Scores are not probabilities; 0.4 to 0.5 is the usual "seen" cut.
The boxes of one image run in batches of 8.

Licence: the code is Apache 2.0, the weights are trained on COCO-WholeBody (research and non-commercial use).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_FILE = "end2end.onnx"
INPUT_W, INPUT_H = 192, 256
PAD = 1.25
BATCH = 8
POINTS = 133
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class PoseEngine:
    """One DWPose session. Thread-safe; runs are serialised."""

    def __init__(self, model_dir: Path, threads: int | None = None):
        from app.ml import runtime

        model = Path(model_dir) / MODEL_FILE
        if not model.is_file():
            raise FileNotFoundError(f"pose model missing: {model} (run scripts/download_models.py)")
        self._sess = runtime.session(model, threads=threads)
        self.provider = runtime.session_provider(self._sess)
        self._in = self._sess.get_inputs()[0].name
        self._lock = threading.Lock()

    @staticmethod
    def _warp(rgb: np.ndarray, box: tuple[float, float, float, float]) -> tuple[np.ndarray, float, float, float, float]:
        x0, y0, x1, y1 = (float(v) for v in box)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = max(x1 - x0, 1.0) * PAD, max(y1 - y0, 1.0) * PAD
        if w > h * INPUT_W / INPUT_H:
            h = w * INPUT_H / INPUT_W
        else:
            w = h * INPUT_W / INPUT_H
        src = np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx - w / 2, cy + h / 2]])
        dst = np.float32([[0, 0], [INPUT_W, 0], [0, INPUT_H]])
        crop = cv2.warpAffine(rgb, cv2.getAffineTransform(src, dst), (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
        return crop, cx, cy, w, h

    def estimate(self, bgr: np.ndarray, boxes_px: list[tuple[float, float, float, float]]) -> list[np.ndarray]:
        if not boxes_px:
            return []
        img = np.asarray(bgr)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        out: list[np.ndarray] = []
        for i in range(0, len(boxes_px), BATCH):
            chunk = boxes_px[i:i + BATCH]
            warped = [self._warp(rgb, b) for b in chunk]
            blob = np.stack([((c.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1) for c, *_ in warped])
            with self._lock:
                sx, sy = self._sess.run(None, {self._in: np.ascontiguousarray(blob, dtype=np.float32)})[:2]
            for k, (_, cx, cy, w, h) in enumerate(warped):
                xs = sx[k].argmax(1) / 2.0
                ys = sy[k].argmax(1) / 2.0
                score = 0.5 * (sx[k].max(1) + sy[k].max(1))
                out.append(np.stack([xs / INPUT_W * w + cx - w / 2, ys / INPUT_H * h + cy - h / 2, score],
                                    axis=1).astype(np.float32))
        return out


_engine: PoseEngine | None = None
_engine_lock = threading.Lock()


def available() -> bool:
    from app.ml import models

    return models.present("pose")


def get_engine() -> PoseEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from app.config import settings
                from app.ml import models

                _engine = PoseEngine(models.model_dir("pose"), getattr(settings, "traits_threads", 2))
                log.info("pose model loaded on %s", _engine.provider)
    return _engine


def estimate(bgr: np.ndarray, boxes_px: list[tuple[float, float, float, float]]) -> list[np.ndarray]:
    """133 points per box (see module docstring). An empty box list returns [] without loading the model."""
    if not boxes_px:
        return []
    return get_engine().estimate(bgr, boxes_px)


def reset() -> None:
    global _engine
    with _engine_lock:
        _engine = None


def loaded() -> bool:
    return _engine is not None


def provider() -> str | None:
    e = _engine
    return e.provider if e is not None else None
