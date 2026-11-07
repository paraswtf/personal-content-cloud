"""NSFW image classifier: Falconsai nsfw_image_detection (ViT-B/16), ONNX export by onnx-community.

Contract (docs/ARCHITECTURE.md sections 8 and 9):
  available() -> bool                       model.onnx and config.json present (app.ml.models 'nsfw')
  score(images) -> list[float]              p(nsfw) in [0, 1] per BGR image (HxWx3 uint8; grey and BGRA
                                            are accepted), run in batches of BATCH
  get_engine() -> NsfwEngine; reset(); loaded(); provider()

Preprocessing follows preprocessor_config.json (ViTFeatureExtractor): RGB, resize to 224x224 with PIL
bilinear (resample 2), scale by 1/255, normalise with mean 0.5 and std 0.5, NCHW. The model returns
two logits; a softmax gives the class probabilities and the index of the "nsfw" label is read from
config.json id2label (0 'normal', 1 'nsfw' for the pinned commit). The classifier only judges
explicit nudity and sexual content; the decision rules live in app.analysis.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

MODEL_FILE = "model.onnx"
BATCH = 8


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


class NsfwEngine:
    """One ViT classifier session. Thread-safe; runs are serialised."""

    def __init__(self, model_dir: Path, threads: int | None = None):
        from app.ml import runtime

        model_dir = Path(model_dir)
        model = model_dir / MODEL_FILE
        if not model.is_file():
            raise FileNotFoundError(f"NSFW model missing: {model} (run scripts/download_models.py)")
        cfg = {}
        if (model_dir / "config.json").is_file():
            cfg = json.loads((model_dir / "config.json").read_text())
        pre = {}
        if (model_dir / "preprocessor_config.json").is_file():
            pre = json.loads((model_dir / "preprocessor_config.json").read_text())

        labels = {int(k): str(v).lower() for k, v in (cfg.get("id2label") or {"0": "normal", "1": "nsfw"}).items()}
        nsfw_idx = [i for i, name in labels.items() if name == "nsfw"]
        if not nsfw_idx:
            raise RuntimeError(f"no 'nsfw' label in {model_dir / 'config.json'}: {labels}")
        self.nsfw_index = nsfw_idx[0]
        self.num_labels = len(labels)

        size = pre.get("size") or {"height": 224, "width": 224}
        if isinstance(size, int):
            size = {"height": size, "width": size}
        self.size = (int(size.get("width", 224)), int(size.get("height", 224)))
        self.mean = np.array(pre.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(pre.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        self.rescale = float(pre.get("rescale_factor", 1 / 255))
        resample = int(pre.get("resample", 2))
        self.resample = {0: Image.Resampling.NEAREST, 1: Image.Resampling.LANCZOS, 2: Image.Resampling.BILINEAR,
                         3: Image.Resampling.BICUBIC}.get(resample, Image.Resampling.BILINEAR)

        self._sess = runtime.session(model, threads=threads)
        self.provider = runtime.session_provider(self._sess)
        self._in = self._sess.get_inputs()[0].name
        self._lock = threading.Lock()

    def preprocess(self, img: np.ndarray | Image.Image) -> np.ndarray:
        pil = _to_pil_rgb(img).resize(self.size, self.resample)
        x = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) * self.rescale
        return np.ascontiguousarray((x - self.mean) / self.std, dtype=np.float32)

    def score(self, images: list[np.ndarray]) -> list[float]:
        """p(nsfw) for each image, in order."""
        out: list[float] = []
        if images is None or len(images) == 0:
            return out
        for i in range(0, len(images), BATCH):
            blob = np.stack([self.preprocess(im) for im in images[i:i + BATCH]])
            with self._lock:
                logits = self._sess.run(None, {self._in: blob})[0]
            logits = np.asarray(logits, dtype=np.float64).reshape(len(blob), -1)
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            p = e / e.sum(axis=1, keepdims=True)
            out.extend(float(v) for v in p[:, self.nsfw_index])
        return out


_engine: NsfwEngine | None = None
_engine_lock = threading.Lock()


def available() -> bool:
    from app.ml import models

    return models.present("nsfw")


def get_engine() -> NsfwEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from app.config import settings
                from app.ml import models

                _engine = NsfwEngine(models.model_dir("nsfw"), settings.ort_threads)
                log.info("NSFW classifier loaded on %s", _engine.provider)
    return _engine


def score(images: list[np.ndarray]) -> list[float]:
    """p(nsfw) per BGR image (see module docstring)."""
    if images is None or len(images) == 0:
        return []
    return get_engine().score(images)


def reset() -> None:
    global _engine
    with _engine_lock:
        _engine = None


def loaded() -> bool:
    return _engine is not None


def provider() -> str | None:
    e = _engine
    return e.provider if e is not None else None
