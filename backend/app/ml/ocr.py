"""Text recognition with RapidOCR (onnxruntime engine, models bundled with the rapidocr package).

Contract (docs/ARCHITECTURE.md sections 8 and 11):
  available() -> bool                    rapidocr and its bundled models are installed (app.ml.models 'ocr')
  ocr(img_bgr) -> list[OcrBox]           text lines in reading order (top to bottom, then left to right),
                                         pixel boxes in img_bgr; [] for blank or tiny images
  OcrBox(x0, y0, x1, y1, text, score)    axis-aligned box around the detected (possibly tilted) quad
  get_engine() -> RapidOCR; reset(); loaded(); provider()

rapidocr 3.10 ships PP-OCRv6 small detection and recognition models (multilingual Latin + Chinese
character set; English reads fine) plus the PP-OCR v2 mobile direction classifier inside the package;
the default config points at those files, so nothing is downloaded at run time (the character list is
stored in the recognition model's metadata). Do not change the language or model version params: any
model that is not bundled would be downloaded on first use.

GPU: app.ml.runtime.prepare() loads the CUDA libraries first; RapidOCR then gets use_cuda with the same
CUDA options as the other models (heuristic cuDNN algorithm search, because every page has a new input
shape). Threads follow settings.ort_threads. Calls are serialised by one lock (RapidOCR keeps per-call
state on the instance).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

MIN_SIDE = 8


@dataclass
class OcrBox:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


_engine: Any = None
_engine_lock = threading.Lock()
_run_lock = threading.Lock()
_provider: str | None = None


def available() -> bool:
    from app.ml import models

    return models.present("ocr")


def _params(use_cuda: bool, threads: int) -> dict[str, Any]:
    from app.ml import runtime

    params: dict[str, Any] = {
        "Global.log_level": "error",
        "EngineConfig.onnxruntime.intra_op_num_threads": int(max(1, threads)),
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": bool(use_cuda),
    }
    if use_cuda:
        opts = runtime.cuda_provider_options()
        for k in ("device_id", "arena_extend_strategy", "cudnn_conv_algo_search", "do_copy_in_default_stream"):
            params[f"EngineConfig.onnxruntime.cuda_ep_cfg.{k}"] = opts[k]
    return params


def _det_provider(engine: Any) -> str | None:
    try:
        return engine.text_det.session.session.get_providers()[0]
    except Exception:  # noqa: BLE001 - internal layout differs between rapidocr versions
        return None


def get_engine() -> Any:
    """The shared RapidOCR instance. The first call loads the three models (one tiny warm-up run)."""
    global _engine, _provider
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from rapidocr import RapidOCR

                from app.config import settings
                from app.ml import runtime

                use_cuda = runtime.cuda_usable()
                eng = RapidOCR(params=_params(use_cuda, settings.ort_threads))
                logging.getLogger("RapidOCR").setLevel(logging.ERROR)
                # RapidOCR loads its models lazily: warm up so the provider is known and load errors surface here.
                warm = np.full((120, 480, 3), 255, dtype=np.uint8)
                cv2.putText(warm, "OCR 2026", (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
                with _run_lock:
                    eng(warm)
                _provider = _det_provider(eng) or (runtime.CUDA if use_cuda else runtime.CPU)
                if use_cuda and _provider != runtime.CUDA:
                    log.warning("OCR asked for CUDA but runs on %s", _provider)
                log.info("OCR loaded on %s", _provider)
                _engine = eng
    return _engine


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
    return np.ascontiguousarray(arr)


def reading_order(boxes: list[OcrBox]) -> list[OcrBox]:
    """Sort into lines by vertical position, then each line left to right.

    A box joins the current line when its vertical centre lies within the line's band (half the
    median height of the line's boxes around the line's centre), so words of one line with slightly
    different baselines stay together.
    """
    if not boxes:
        return []
    rest = sorted(boxes, key=lambda b: (b.cy, b.x0))
    lines: list[list[OcrBox]] = []
    for b in rest:
        if lines:
            line = lines[-1]
            cy = sum(x.cy for x in line) / len(line)
            hs = sorted(x.height for x in line)
            band = max(hs[len(hs) // 2], b.height) / 2.0
            if abs(b.cy - cy) <= band:
                line.append(b)
                continue
        lines.append([b])
    out: list[OcrBox] = []
    for line in lines:
        out.extend(sorted(line, key=lambda x: x.x0))
    return out


def ocr(img_bgr: np.ndarray) -> list[OcrBox]:
    """Recognise text in a BGR image; lines in reading order. Empty list for blank images."""
    img = _as_bgr(img_bgr)
    h, w = img.shape[:2]
    if h < MIN_SIDE or w < MIN_SIDE:
        return []
    eng = get_engine()
    with _run_lock:
        res = eng(img)
    boxes = getattr(res, "boxes", None)
    txts = getattr(res, "txts", None)
    scores = getattr(res, "scores", None)
    if boxes is None or txts is None:
        return []
    if scores is None:
        scores = [1.0] * len(txts)
    out: list[OcrBox] = []
    for quad, text, score in zip(boxes, txts, scores):
        text = str(text or "").strip()
        if not text:
            continue
        q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
        x0, y0 = float(np.clip(q[:, 0].min(), 0, w)), float(np.clip(q[:, 1].min(), 0, h))
        x1, y1 = float(np.clip(q[:, 0].max(), 0, w)), float(np.clip(q[:, 1].max(), 0, h))
        out.append(OcrBox(x0, y0, x1, y1, text, float(score)))
    return reading_order(out)


def text_of(boxes: list[OcrBox]) -> str:
    """Join OCR boxes into text: one output line per reading-order line, words separated by spaces."""
    if not boxes:
        return ""
    lines: list[list[str]] = [[boxes[0].text]]
    prev = boxes[0]
    for b in boxes[1:]:
        if b.x0 >= prev.x0 and abs(b.cy - prev.cy) <= max(prev.height, b.height) / 2.0:
            lines[-1].append(b.text)
        else:
            lines.append([b.text])
        prev = b
    return "\n".join(" ".join(parts) for parts in lines)


def reset() -> None:
    global _engine, _provider
    with _engine_lock:
        _engine = None
        _provider = None


def loaded() -> bool:
    return _engine is not None


def provider() -> str | None:
    return _provider if _engine is not None else None
