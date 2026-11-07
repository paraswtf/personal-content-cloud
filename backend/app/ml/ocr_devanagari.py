"""Second text recogniser for Devanagari (Hindi, Marathi): PP-OCRv5 mobile, used by app.ml.ocr.

Contract:
  available() -> bool          the recognition model is on disk with its pinned size (app.ml.models 'ocr_devanagari')
  get_engine() -> RapidOCR | None
                               a RapidOCR instance whose recogniser is the Devanagari model; only its
                               recogniser is loaded (no second detector or classifier). None when the
                               model is absent or failed to load (the failure is logged once, until reset())
  recognise(crops) -> list[(text, score)]
                               one reading per crop, in input order; the caller holds app.ml.ocr's run lock
  is_devanagari(text) -> bool  the text has at least one Devanagari letter (digits and danda do not count)
  load_error() -> str | None   why the engine failed to load (kept until reset()), else None
  choose(latin, deva) -> (text, score, which)
                               the per-crop rule: the Devanagari reading wins when it has a Devanagari
                               letter and its score is at least max(0.6, latin_score - 0.35); otherwise the
                               higher score wins, ties go to the bundled (Latin) reading; two guards keep
                               Latin-only lines unchanged (see choose()); a Devanagari reading comes
                               back in NFC
  reset(); loaded(); provider()

The model's 568-entry character list (Devanagari plus ASCII letters, digits and punctuation) is stored
in the ONNX metadata, so rapidocr needs no dictionary file and downloads nothing. The language, version
and model type params must be rapidocr enums (plain strings raise "must be Enum Type").
"""
from __future__ import annotations

import logging
import math
import threading
import unicodedata
from typing import Any, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_REL = "ocr/devanagari_PP-OCRv5_rec_mobile.onnx"
MIN_SCORE = 0.6       # the Devanagari reading needs at least this score
MARGIN = 0.35         # ... and may trail the bundled reading's score by at most this much

_engine: Any = None
_failed: str | None = None
_provider: str | None = None
_lock = threading.Lock()


def model_path():
    from app.ml import models

    return models.path("ocr_devanagari", MODEL_REL)


def available() -> bool:
    """The recognition model file is on disk with its pinned size (the font is not needed for OCR)."""
    from app.ml import models

    try:
        spec = models.REGISTRY["ocr_devanagari"]
    except KeyError:
        return False
    pinned = next((f.size for f in spec.files if f.rel == MODEL_REL), None)
    if pinned is None:
        return False
    p = model_path()
    try:
        return p.is_file() and p.stat().st_size == pinned
    except OSError:
        return False


def _is_deva_letter(ch: str) -> bool:
    cp = ord(ch)
    if not (0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF):
        return False
    return unicodedata.category(ch).startswith("L")


def is_devanagari(text: str) -> bool:
    return any(_is_deva_letter(ch) for ch in text or "")


def _score(x: Any) -> float:
    try:
        s = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(s) else s


def _letter_counts(text: str) -> tuple[int, int]:
    """(Devanagari letters, other letters) in text. Vowel signs, virama, digits and danda are not letters."""
    deva = other = 0
    for ch in text:
        if _is_deva_letter(ch):
            deva += 1
        elif unicodedata.category(ch).startswith("L"):
            other += 1
    return deva, other


def choose(latin: tuple[str, float], deva: tuple[str, float]) -> tuple[str, float, str]:
    """Pick one reading of a crop. Returns (text, score, 'latin' or 'deva').

    The Devanagari reading wins when it has a Devanagari letter and its score is at least
    max(0.6, bundled score - 0.35); otherwise the higher score wins and ties go to the bundled reading.
    An empty reading counts as score 0, so a crop the bundled recogniser cannot read at all (a Marathi
    line comes back empty) still gets the Devanagari reading.

    Two guards keep Latin-only lines as they were (measured on English pages, see
    docs/integration-notes/ml.md):
    - a Devanagari reading with no Devanagari letter never replaces a non-empty bundled reading (on
      Latin text the two recognisers often tie within 0.01, and the Devanagari model then dropped the
      rupee sign or read "AI" as "Al");
    - the score margin applies only when the Devanagari reading has at least two Devanagari letters or
      no other letters (the Devanagari model reads the rupee sign in "Price ₹ 499" as the letter RA, and
      that reading would otherwise win at 0.93 against 0.96); a reading with one Devanagari letter among
      Latin letters still wins on a higher score.
    """
    lt, ls = str(latin[0] or "").strip(), _score(latin[1])
    # NFC: the model's list has the precomposed nukta letters U+0958 to U+095F (क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़),
    # which NFC turns into the letter plus U+093C, the form typed text and search queries use
    dt, ds = unicodedata.normalize("NFC", str(deva[0] or "").strip()), _score(deva[1])
    if not lt:
        ls = 0.0
    if not dt:
        return lt, ls, "latin"
    n_deva, n_other = _letter_counts(dt)
    if n_deva:
        if (n_deva >= 2 or n_other == 0) and ds >= max(MIN_SCORE, ls - MARGIN):
            return dt, ds, "deva"
        if ds > ls:
            return dt, ds, "deva"
        return lt, ls, "latin"
    if not lt and ds > 0:
        return dt, ds, "deva"
    return lt, ls, "latin"


def _rec_provider(engine: Any) -> str | None:
    try:
        return engine.text_rec.session.session.get_providers()[0]
    except Exception:  # noqa: BLE001 - internal layout differs between rapidocr versions
        return None


def _warm_crop() -> np.ndarray:
    crop = np.full((48, 240, 3), 255, dtype=np.uint8)
    cv2.putText(crop, "OCR 2026", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    return crop


def get_engine() -> Any:
    """The Devanagari RapidOCR instance, loaded on first use; None when absent or broken.

    Built with the same CUDA and thread params as the bundled engine (app.ml.ocr._params). The instance
    is warmed up before it is published, so no other thread can use it during the warm-up.
    """
    global _engine, _failed, _provider
    if _engine is not None or _failed is not None:
        return _engine
    if not available():
        return None
    with _lock:
        if _engine is not None or _failed is not None:
            return _engine
        try:
            from rapidocr import RapidOCR
            from rapidocr.ch_ppocr_rec import TextRecInput
            from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

            from app.config import settings
            from app.ml import runtime
            from app.ml.ocr import _params

            use_cuda = runtime.cuda_usable()
            params = _params(use_cuda, settings.ort_threads)
            params.update({
                "Rec.model_path": str(model_path()),
                "Rec.lang_type": LangRec.DEVANAGARI,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
            })
            eng = RapidOCR(params=params)
            logging.getLogger("RapidOCR").setLevel(logging.ERROR)
            rec = eng._load_rec_model()
            rec(TextRecInput(img=[_warm_crop()]))
            _provider = _rec_provider(eng) or (runtime.CUDA if use_cuda else runtime.CPU)
            if use_cuda and _provider != runtime.CUDA:
                log.warning("Devanagari OCR asked for CUDA but runs on %s", _provider)
            from app.ml.ocr import shrink_sessions

            shrink_sessions(eng, ("text_rec",))
            log.info("Devanagari OCR loaded on %s", _provider)
            _engine = eng
        except Exception as e:  # noqa: BLE001 - English OCR keeps working without it
            _failed = str(e) or e.__class__.__name__
            _provider = None
            log.warning("Devanagari OCR could not be loaded, reading Latin script only: %s", _failed)
    return _engine


def recognise(crops: Sequence[np.ndarray]) -> list[tuple[str, float]]:
    """Read each crop with the Devanagari recogniser; [] when the engine is not loaded."""
    eng = _engine
    if eng is None or not crops:
        return []
    from rapidocr.ch_ppocr_rec import TextRecInput

    res = eng.text_rec(TextRecInput(img=list(crops)))
    txts = list(res.txts or ())
    scores = list(res.scores or ())
    return [(str(t or ""), _score(s)) for t, s in zip(txts, scores)]


def load_error() -> str | None:
    return _failed


def reset() -> None:
    global _engine, _failed, _provider
    with _lock:
        _engine = None
        _failed = None
        _provider = None


def loaded() -> bool:
    return _engine is not None


def provider() -> str | None:
    return _provider if _engine is not None else None
