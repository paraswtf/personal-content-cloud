"""Image descriptors: DINOv2 small (fp32 ONNX), the CLS token of a 224 px crop, used to shortlist tattoo and mehndi
crops of the same design before SIFT verifies them.

Contract (docs/ARCHITECTURE.md sections 8 and 34):
  available() -> bool                 dino/dinov2_small.onnx present (app.ml.models 'dino')
  embed(crops_rgb) -> (n, 384)        L2-normalised float32 CLS embeddings; crops are HxWx3 uint8 RGB arrays of any
                                      size, resized to 224 x 224 with INTER_AREA (INTER_LINEAR when enlarging)
  get_engine() -> DinoEngine; reset(); loaded(); provider()

Preprocessing: RGB, x / 255, ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), NCHW. The model
output is last_hidden_state (batch, 257, 384): token 0 is the CLS token. Batches of 8 (77 ms a crop on 4 CPU threads).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_FILE = "dinov2_small.onnx"
SIZE = 224
BATCH = 8
DIM = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DinoEngine:
    """One DINOv2 session. Thread-safe; runs are serialised."""

    def __init__(self, model_dir: Path, threads: int | None = None):
        from app.ml import runtime

        model = Path(model_dir) / MODEL_FILE
        if not model.is_file():
            raise FileNotFoundError(f"dino model missing: {model} (run scripts/download_models.py)")
        self._sess = runtime.session(model, threads=threads)
        self.provider = runtime.session_provider(self._sess)
        self._in = self._sess.get_inputs()[0].name
        self._lock = threading.Lock()

    @staticmethod
    def preprocess(crop: np.ndarray) -> np.ndarray:
        img = np.asarray(crop)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=2)
        img = img[:, :, :3]
        if img.shape[0] != SIZE or img.shape[1] != SIZE:
            interp = cv2.INTER_AREA if img.shape[0] >= SIZE and img.shape[1] >= SIZE else cv2.INTER_LINEAR
            img = cv2.resize(img, (SIZE, SIZE), interpolation=interp)
        return ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)

    def embed(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        n = len(crops_rgb)
        out = np.zeros((n, DIM), dtype=np.float32)
        for i in range(0, n, BATCH):
            blob = np.stack([self.preprocess(c) for c in crops_rgb[i:i + BATCH]])
            with self._lock:
                hidden = self._sess.run(None, {self._in: np.ascontiguousarray(blob, dtype=np.float32)})[0]
            cls = np.asarray(hidden, dtype=np.float32)[:, 0]
            out[i:i + len(cls)] = cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)
        return out


_engine: DinoEngine | None = None
_engine_lock = threading.Lock()


def available() -> bool:
    from app.ml import models

    return models.present("dino")


def get_engine() -> DinoEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from app.config import settings
                from app.ml import models

                _engine = DinoEngine(models.model_dir("dino"), getattr(settings, "traits_threads", 2))
                log.info("dino model loaded on %s", _engine.provider)
    return _engine


def embed(crops_rgb: list[np.ndarray]) -> np.ndarray:
    """(n, 384) L2-normalised CLS embeddings (see module docstring)."""
    if crops_rgb is None or len(crops_rgb) == 0:
        return np.zeros((0, DIM), dtype=np.float32)
    return get_engine().embed(crops_rgb)


def reset() -> None:
    global _engine
    with _engine_lock:
        _engine = None


def loaded() -> bool:
    return _engine is not None


def provider() -> str | None:
    e = _engine
    return e.provider if e is not None else None
