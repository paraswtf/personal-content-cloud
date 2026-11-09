"""Hand landmarks: MediaPipe handpose (OpenCV Zoo 2023feb), run on a crop around a hand found by the pose model.

Contract (docs/ARCHITECTURE.md sections 8 and 34):
  available() -> bool                 hands/handpose.onnx present (app.ml.models 'hands')
  landmarks(crops_rgb224) -> list     per crop: ((21, 2) float32 landmark x, y in crop pixels, confidence, right_p)
                                      crops: 224 x 224 x 3 uint8 RGB arrays (other sizes are resized to 224)
  get_engine() -> HandEngine; reset(); loaded(); provider()

The model takes one NHWC float image in 0..1 (batch fixed at 1, so crops run one by one, 2.7 ms each). The graph has
four outputs, named Identity to Identity_3 by the TensorFlow export, told apart by shape at load time:
  - the first (1, 63) output: 21 x (x, y, z) landmarks, x and y in input pixels (0..224);
  - the second (1, 63) output: the same landmarks in metres (world), not used;
  - the first (1, 1) output: hand presence confidence, 0..1 (near 0 on clothing, prints and plain backgrounds);
  - the second (1, 1) output: handedness, 0..1, read as the probability of a right hand (the MediaPipe convention:
    values above 0.5 are the right hand of the person, for an unmirrored image). Not verified against labelled
    hands; callers should treat it as a weak hint.
If the graph has other shapes the engine refuses to load. Outputs of equal shape keep their graph order.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_FILE = "handpose.onnx"
SIZE = 224


class HandEngine:
    """One handpose session. Thread-safe; runs are serialised."""

    def __init__(self, model_dir: Path, threads: int | None = None):
        from app.ml import runtime

        model = Path(model_dir) / MODEL_FILE
        if not model.is_file():
            raise FileNotFoundError(f"hand model missing: {model} (run scripts/download_models.py)")
        self._sess = runtime.session(model, threads=threads)
        self.provider = runtime.session_provider(self._sess)
        self._in = self._sess.get_inputs()[0].name
        outs = self._sess.get_outputs()
        wide = [i for i, o in enumerate(outs) if _last_dim(o.shape) == 63]
        single = [i for i, o in enumerate(outs) if _last_dim(o.shape) == 1]
        if len(wide) < 1 or len(single) < 2:
            raise RuntimeError(f"unexpected handpose outputs: {[(o.name, o.shape) for o in outs]}")
        self._lm, self._conf, self._hand = wide[0], single[0], single[1]
        self._lock = threading.Lock()

    def landmarks(self, crops_rgb224: list[np.ndarray]) -> list[tuple[np.ndarray, float, float]]:
        res: list[tuple[np.ndarray, float, float]] = []
        for crop in crops_rgb224:
            img = np.asarray(crop)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=2)
            img = img[:, :, :3]
            if img.shape[0] != SIZE or img.shape[1] != SIZE:
                img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA if img.shape[0] > SIZE else cv2.INTER_LINEAR)
            blob = (img.astype(np.float32) / 255.0)[None]
            with self._lock:
                outs = self._sess.run(None, {self._in: np.ascontiguousarray(blob)})
            lm = np.asarray(outs[self._lm], dtype=np.float32).reshape(21, 3)[:, :2].copy()
            res.append((lm, float(np.ravel(outs[self._conf])[0]), float(np.ravel(outs[self._hand])[0])))
        return res


def _last_dim(shape: list) -> int | None:
    return shape[-1] if shape and isinstance(shape[-1], int) else None


_engine: HandEngine | None = None
_engine_lock = threading.Lock()


def available() -> bool:
    from app.ml import models

    return models.present("hands")


def get_engine() -> HandEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from app.config import settings
                from app.ml import models

                _engine = HandEngine(models.model_dir("hands"), getattr(settings, "traits_threads", 2))
                log.info("hand model loaded on %s", _engine.provider)
    return _engine


def landmarks(crops_rgb224: list[np.ndarray]) -> list[tuple[np.ndarray, float, float]]:
    """Landmarks, confidence and right-hand probability per crop (see module docstring)."""
    if not crops_rgb224:
        return []
    return get_engine().landmarks(crops_rgb224)


def reset() -> None:
    global _engine
    with _engine_lock:
        _engine = None


def loaded() -> bool:
    return _engine is not None


def provider() -> str | None:
    e = _engine
    return e.provider if e is not None else None
