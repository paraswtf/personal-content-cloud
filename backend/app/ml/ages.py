"""Face age groups: the FairFace ViT classifier (dima806/fairface_age_image_detection), int8 ONNX export by
onnx-community.

Contract (docs/ARCHITECTURE.md sections 8 and 10):
  available() -> bool                  onnx/model_int8.onnx and config.json present (app.ml.models 'ages')
  GROUPS                               the nine FairFace age groups, youngest first
  predict(crops) -> (n, 9) float32     group probabilities per face crop, rows sum to 1, columns in GROUPS order.
                                       A crop is a BGR HxWx3 uint8 array (grey and BGRA accepted) or a PIL image.
  get_engine() -> AgeEngine; reset(); loaded(); provider()

The crop is the stored face crop (app.analysis.faces.crop_face: a square of 1.6 x the longer face side around the
face), close to FairFace's own crops (faces with a margin). Preprocessing follows preprocessor_config.json
(ViTFeatureExtractor): RGB, 224 x 224 bilinear, x / 255, mean 0.5 and std 0.5, NCHW. The labels come from
config.json id2label and must be the nine groups below in that order, or the engine refuses to load.

Measured on the owner's library (2026-11-09): babies and toddlers read 0.7 to 0.9 under 10, children of about 10 to
12 split between 3-9 and 10-19, adults stay in the adult groups (405 of 418 sampled), young adults of 18 to 22 may
read 10-19; about 70 ms a face on a 4-thread CPU. Reading ages is the analysis's business (app.analysis.ages).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

MODEL_FILE = "onnx/model_int8.onnx"
GROUPS: tuple[str, ...] = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+")
_LABELS = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70")
BATCH = 16


def _to_pil_rgb(img: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(img, Image.Image):
        return img if img.mode == "RGB" else img.convert("RGB")
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got shape {arr.shape}")
    return Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]), "RGB")


class AgeEngine:
    """One ViT classifier session. Thread-safe; runs are serialised."""

    def __init__(self, model_dir: Path, threads: int | None = None):
        from app.ml import runtime

        model_dir = Path(model_dir)
        model = model_dir / MODEL_FILE
        if not model.is_file():
            raise FileNotFoundError(f"age model missing: {model} (run scripts/download_models.py)")
        cfg = json.loads((model_dir / "config.json").read_text()) if (model_dir / "config.json").is_file() else {}
        labels = tuple(str(v) for _, v in sorted((int(k), v) for k, v in (cfg.get("id2label") or {}).items()))
        if labels and labels != _LABELS:
            raise RuntimeError(f"unexpected age labels in {model_dir / 'config.json'}: {labels}")
        pre = {}
        if (model_dir / "preprocessor_config.json").is_file():
            pre = json.loads((model_dir / "preprocessor_config.json").read_text())
        size = pre.get("size") or {"height": 224, "width": 224}
        if isinstance(size, int):
            size = {"height": size, "width": size}
        self.size = (int(size.get("width", 224)), int(size.get("height", 224)))
        self.mean = np.array(pre.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(pre.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        self.rescale = float(pre.get("rescale_factor", 1 / 255))
        self._sess = runtime.session(model, threads=threads)
        self.provider = runtime.session_provider(self._sess)
        self._in = self._sess.get_inputs()[0].name
        self._lock = threading.Lock()

    def preprocess(self, img: np.ndarray | Image.Image) -> np.ndarray:
        pil = _to_pil_rgb(img).resize(self.size, Image.Resampling.BILINEAR)
        x = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) * self.rescale
        return np.ascontiguousarray((x - self.mean) / self.std, dtype=np.float32)

    def predict(self, crops: list[np.ndarray] | list[Image.Image]) -> np.ndarray:
        """(n, 9) group probabilities, in order."""
        n = len(crops) if crops is not None else 0
        out = np.zeros((n, len(GROUPS)), dtype=np.float32)
        for i in range(0, n, BATCH):
            blob = np.stack([self.preprocess(c) for c in crops[i:i + BATCH]])
            with self._lock:
                logits = self._sess.run(None, {self._in: blob})[0]
            logits = np.asarray(logits, dtype=np.float64).reshape(len(blob), -1)
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            out[i:i + len(blob)] = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        return out


_engine: AgeEngine | None = None
_engine_lock = threading.Lock()


def available() -> bool:
    from app.ml import models

    return models.present("ages")


def get_engine() -> AgeEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from app.config import settings
                from app.ml import models

                _engine = AgeEngine(models.model_dir("ages"), settings.ort_threads)
                log.info("age classifier loaded on %s", _engine.provider)
    return _engine


def predict(crops: list[np.ndarray] | list[Image.Image]) -> np.ndarray:
    """Group probabilities per face crop (see module docstring)."""
    if crops is None or len(crops) == 0:
        return np.zeros((0, len(GROUPS)), dtype=np.float32)
    return get_engine().predict(crops)


def reset() -> None:
    global _engine
    with _engine_lock:
        _engine = None


def loaded() -> bool:
    return _engine is not None


def provider() -> str | None:
    e = _engine
    return e.provider if e is not None else None
