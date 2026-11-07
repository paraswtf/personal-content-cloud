"""Text recognition with RapidOCR (onnxruntime engine): Latin script from the models bundled with the
rapidocr package, plus Hindi and Marathi (Devanagari) when the second recogniser is on disk.

Contract (docs/ARCHITECTURE.md sections 8 and 11):
  available() -> bool                    rapidocr and its bundled models are installed (app.ml.models 'ocr')
  ocr(img_bgr) -> list[OcrBox]           text lines in reading order (top to bottom, then left to right),
                                         pixel boxes in img_bgr; [] for blank or tiny images
  OcrBox(x0, y0, x1, y1, text, score)    axis-aligned box around the detected (possibly tilted) quad
  get_engine() -> RapidOCR; reset(); loaded(); provider()
  devanagari_active() -> bool            the second recogniser is loaded and used by ocr()

rapidocr 3.10 ships PP-OCRv6 small detection and recognition models (multilingual Latin + Chinese
character set; English reads fine) plus the PP-OCR v2 mobile direction classifier inside the package;
the default config points at those files, so nothing is downloaded at run time (the character list is
stored in the recognition model's metadata). Do not change the language or model version params: any
model that is not bundled would be downloaded on first use.

Devanagari: the bundled recogniser cannot read Hindi or Marathi (it gives Latin garbage, a truncated
reading such as "/MALE" for "पुरुष / MALE", or an empty string, and rapidocr then drops the line). When
the pinned PP-OCRv5 mobile Devanagari recogniser is on disk (app.ml.models 'ocr_devanagari', engine
state in app.ml.ocr_devanagari), ocr() runs the rapidocr steps itself: one detection and one direction
classification with the bundled engine, then every crop is read by both recognisers and
ocr_devanagari.choose() keeps one reading per crop (the Devanagari reading wins when it has a Devanagari
letter and scores at least max(0.6, bundled score - 0.35); otherwise the higher score wins, ties go to
the bundled reading; two guards keep Latin-only lines as they were). The direction classifier is not
trained on Devanagari and often turns an upright Hindi line upside down, so the Devanagari recogniser
also reads every crop the classifier turned in its upright form and keeps the better of the two. Empty readings and readings under rapidocr's text_score (0.5) are dropped after
the choice, as rapidocr does. The second recogniser costs about one more recognition pass per page
(see docs/integration-notes/ml.md). When the model is absent, or fails to load (logged once), ocr()
calls the bundled engine exactly as before.

GPU: app.ml.runtime.prepare() loads the CUDA libraries first; RapidOCR then gets use_cuda with the same
CUDA options as the other models (heuristic cuDNN algorithm search, because every page has a new input
shape). Threads follow settings.ort_threads. Calls are serialised by one lock (RapidOCR keeps per-call
state on the instance); the Devanagari recogniser runs under the same lock.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.ml import ocr_devanagari

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


def shrink_sessions(engine: Any, parts: tuple[str, ...] = ("text_det", "text_cls", "text_rec")) -> int:
    """Wrap the loaded rapidocr sessions with runtime.wrap_session, so a CUDA session frees its
    arena's unused memory after every run (see app.ml.runtime). Returns how many were wrapped."""
    from app.ml import runtime

    n = 0
    for part in parts:
        holder = getattr(getattr(engine, part, None), "session", None)
        sess = getattr(holder, "session", None)
        if sess is None:
            continue
        wrapped = runtime.wrap_session(sess)
        if wrapped is not sess:
            holder.session = wrapped
            n += 1
    return n


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
                shrink_sessions(eng)
                log.info("OCR loaded on %s", _provider)
                # the Devanagari recogniser loads with the bundled engine; None (and no error) when absent
                ocr_devanagari.get_engine()
                _engine = eng
    return _engine


def devanagari_active() -> bool:
    """True when ocr() reads every crop with the Devanagari recogniser as well."""
    return ocr_devanagari.loaded()


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
    # loads the Devanagari recogniser when its model appeared after the bundled engine was loaded (a
    # download from Settings); a file check per call when absent, None after a logged load failure
    if ocr_devanagari.get_engine() is not None:
        return _ocr_two_recognisers(eng, img)
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


def _flipped(eng: Any, cls_res: Any) -> list[int]:
    """Indexes of the crops the direction classifier turned by 180 degrees (rapidocr's own test)."""
    thresh = float(getattr(getattr(eng, "text_cls", None), "cls_thresh", 0.9))
    out = []
    for i, item in enumerate(getattr(cls_res, "cls_res", None) or ()):
        try:
            label, score = item
        except (TypeError, ValueError):
            continue
        if "180" in str(label) and float(score) > thresh:
            out.append(i)
    return out


def _ocr_two_recognisers(eng: Any, img: np.ndarray) -> list[OcrBox]:
    """The rapidocr steps of RapidOCR.__call__ with a second recogniser.

    rapidocr's own build_final_output drops every crop whose (bundled) reading is empty, which loses
    Marathi lines, so the steps run here: preprocess, detect and crop, classify and rotate, recognise
    with both recognisers, choose per crop, map the boxes back to the input image, then drop empty
    readings and readings under text_score. cls never drops crops and each recogniser returns its
    readings in input order (it sorts by width internally and restores the order), so index i is crop i
    everywhere. The bundled recogniser reads the classified crops as rapidocr does; the Devanagari one
    reads them too, plus the crops the classifier turned upside down in their upright form.
    """
    import time

    from rapidocr.ch_ppocr_rec import TextRecInput
    from rapidocr.main import RapidOCRError
    from rapidocr.utils.process_img import map_boxes_to_original

    h, w = img.shape[:2]
    t0 = time.perf_counter()
    with _run_lock:
        pre, op_record = eng.preprocess_img(img)
        try:
            crops, det_res = eng.detect_and_crop(pre, op_record)
        except RapidOCRError:
            return []
        if det_res.boxes is None or not len(crops):
            return []
        t_det = time.perf_counter()
        rotated, flipped = crops, []
        if eng.use_cls:
            try:
                rotated, cls_res = eng.cls_and_rotate(crops)
                flipped = _flipped(eng, cls_res)
            except RapidOCRError:
                rotated = crops
            if len(rotated) != len(crops):
                rotated, flipped = crops, []
        t_cls = time.perf_counter()
        rec = eng._load_rec_model()
        latin = rec(TextRecInput(img=rotated))
        t_rec = time.perf_counter()
        deva = ocr_devanagari.recognise(rotated)
        if flipped and len(deva) == len(crops):
            # the direction classifier is not trained on Devanagari and often turns an upright Hindi or
            # Marathi line upside down (the headline stroke looks like a Latin baseline); read those
            # crops upright as well and keep the better Devanagari reading
            upright = ocr_devanagari.recognise([crops[i] for i in flipped])
            for i, alt in zip(flipped, upright):
                if alt[1] > deva[i][1]:
                    deva[i] = alt
        t_deva = time.perf_counter()
        quads = map_boxes_to_original(np.array(det_res.boxes, dtype=np.float32, copy=True), op_record, h, w)
        text_score = float(getattr(eng, "text_score", 0.5))
    log.debug("OCR %d crops: det %.3f s, cls %.3f s, rec %.3f s, devanagari rec %.3f s",
              len(crops), t_det - t0, t_cls - t_det, t_rec - t_cls, t_deva - t_rec)

    l_txts = list(latin.txts or ())
    l_scores = list(latin.scores or ())
    n = len(quads)
    out: list[OcrBox] = []
    for i in range(n):
        lt = (l_txts[i], l_scores[i]) if i < len(l_txts) and i < len(l_scores) else ("", 0.0)
        dv = deva[i] if i < len(deva) else ("", 0.0)
        text, score, _which = ocr_devanagari.choose(lt, dv)
        if not text or score < text_score:
            continue
        q = np.asarray(quads[i], dtype=np.float32).reshape(-1, 2)
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
        ocr_devanagari.reset()


def loaded() -> bool:
    return _engine is not None


def provider() -> str | None:
    return _provider if _engine is not None else None
