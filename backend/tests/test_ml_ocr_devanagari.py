"""app.ml.ocr with the Devanagari recogniser (Hindi, Marathi) and the 'ocr_devanagari' registry entry.

The choice rule and the registry tests need no models. The page tests render text with Pillow (Noto Sans
Devanagari through raqm for Hindi and Marathi, FreeSans and FreeMono for Latin) and skip when the models,
the fonts or raqm are missing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont, features

from tests.conftest import skip_without_models

FREESANS = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
FREEMONO = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"

HINDI_1 = "भारत सरकार"
HINDI_2 = "राहुल कुमार शर्मा"
HINDI_DATE = "जन्म तिथि: 15/08/1988"
MIXED = "पुरुष / MALE"
MARATHI = "माझे नाव प्रिया नायर आहे"
MARATHI_AMOUNT = "एकूण रक्कम रु. 25,000"
ENGLISH_1 = "Government of India"
ENGLISH_2 = "Permanent Account Number ABCPS1234K"


# ------------------------------------------------------------------ choice rule (no models)

def test_is_devanagari_letters_only():
    from app.ml.ocr_devanagari import is_devanagari

    assert is_devanagari("भारत")
    assert is_devanagari("पुरुष / MALE")
    assert not is_devanagari("Government of India")
    assert not is_devanagari("")
    # Devanagari digits, danda and a lone vowel sign are not letters
    assert not is_devanagari("१२३ ।")
    assert not is_devanagari("ा")


@pytest.mark.parametrize("latin, deva, want, which", [
    # Hindi line read as Latin garbage: the Devanagari reading wins within the margin
    (("HRARCR", 0.87), ("भारत सरकार", 0.96), "भारत सरकार", "deva"),
    # truncated but confident bundled reading: 0.97 >= max(0.6, 1.00 - 0.35)
    (("/MALE", 1.00), ("पुरुष / MALE", 0.97), "पुरुष / MALE", "deva"),
    # exactly at the margin
    (("abc", 0.95), ("नमस्ते", 0.60), "नमस्ते", "deva"),
    # just below the margin, and the bundled score is higher
    (("abc", 0.99), ("नमस्ते", 0.63), "abc", "latin"),
    # the 0.6 floor applies when the bundled score is low
    (("x", 0.70), ("नमस्ते", 0.59), "x", "latin"),
    (("x", 0.50), ("नमस्ते", 0.60), "नमस्ते", "deva"),
    # Marathi line the bundled recogniser cannot read: empty counts as score 0
    (("", 0.0), ("माझे नाव प्रिया नायर आहे", 0.95), "माझे नाव प्रिया नायर आहे", "deva"),
    (("   ", 0.9), ("आहे", 0.55), "आहे", "deva"),
    # a lone Devanagari word (one letter plus vowel signs, no Latin letters) still uses the margin
    (("a", 0.95), ("है", 0.80), "है", "deva"),
    # ties go to the bundled reading
    (("Hello", 0.90), ("Hel1o", 0.90), "Hello", "latin"),
    # both empty
    (("", 0.0), ("", 0.0), "", "latin"),
    # NaN score counts as 0
    (("Hello", float("nan")), ("नमस्ते", 0.70), "नमस्ते", "deva"),
])
def test_choose_rule(latin, deva, want, which):
    from app.ml.ocr_devanagari import choose

    text, score, got = choose(latin, deva)
    assert (text, got) == (want, which)
    assert 0.0 <= score <= 1.0


def test_choose_returns_devanagari_in_nfc():
    import unicodedata

    from app.ml.ocr_devanagari import choose

    # U+095B (precomposed ZA) is excluded from composition: NFC gives JA plus nukta
    text, _, which = choose(("", 0.0), ("\u095bमीन", 0.9))
    assert which == "deva" and text == "\u091c\u093cमीन" == unicodedata.normalize("NFC", "\u095bमीन")
    # the bundled reading is returned as it is
    assert choose(("\u212b", 0.9), ("", 0.0))[0] == "\u212b"


def test_choose_keeps_latin_only_lines():
    from app.ml.ocr_devanagari import choose

    # the Devanagari model reads the rupee sign as the letter RA: one Devanagari letter among Latin
    # letters wins only on a higher score, not through the margin
    assert choose(("Price ₹ 499", 0.958), ("Price र 499", 0.932))[2] == "latin"
    assert choose(("Price ₹ 499", 0.90), ("Price र 499", 0.95))[2] == "deva"
    # two Devanagari letters among Latin letters use the margin again
    assert choose(("Rent Agreement", 0.99), ("किराया - Rent Agreement", 0.80))[2] == "deva"
    # a Devanagari reading without a Devanagari letter never replaces a non-empty bundled reading
    assert choose(("Price ₹ 499", 0.948), ("Price  499", 0.950))[2] == "latin"
    assert choose(("Flight AI 865", 0.962), ("Flight Al 865", 0.967))[2] == "latin"
    # ... but it fills in a crop the bundled recogniser read as empty
    assert choose(("", 0.0), ("2026", 0.9)) == ("2026", 0.9, "deva")


# ------------------------------------------------------------------ registry (no models)

def test_registry_entry_pins_and_engine():
    from app.ml import models

    spec = models.REGISTRY["ocr_devanagari"]
    assert spec.key == "ocr_devanagari" and spec.subdir == "" and spec.package is None
    assert "Hindi" in spec.label and "Marathi" in spec.label
    assert "Apache 2.0" in spec.license and "Open Font License" in spec.license
    files = {f.rel: f for f in spec.files}
    model = files["ocr/devanagari_PP-OCRv5_rec_mobile.onnx"]
    font = files["fonts/NotoSansDevanagari-Regular.ttf"]
    assert model.size == 7_940_361
    assert model.sha256 == "d6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e"
    assert model.url == ("https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/"
                         "devanagari_PP-OCRv5_rec_mobile.onnx")
    assert font.size == 277_680
    assert font.sha256 == "573e525d9f4923d34645fe752475419f45f8dec5a8e43f3e83c5c5f43d1f2ff9"
    assert font.url.startswith("https://raw.githubusercontent.com/notofonts/") and font.url.endswith(".ttf")
    assert models._ENGINES["ocr_devanagari"] == "app.ml.ocr_devanagari"


def test_statuses_report_the_devanagari_engine(monkeypatch):
    from app.ml import models, ocr_devanagari

    monkeypatch.setattr(ocr_devanagari, "loaded", lambda: True)
    monkeypatch.setattr(ocr_devanagari, "provider", lambda: "CPUExecutionProvider")
    row = {r["key"]: r for r in models.statuses()}["ocr_devanagari"]
    assert row["loaded"] is True and row["provider"] == "CPUExecutionProvider"
    assert row["size_bytes"] == 7_940_361 + 277_680


def test_absent_model_and_font(tmp_path, monkeypatch):
    from app.config import reload_settings
    from app.ml import models, ocr_devanagari

    monkeypatch.setattr(ocr_devanagari, "_engine", None)
    monkeypatch.setattr(ocr_devanagari, "_failed", None)
    reload_settings(models_dir=tmp_path / "models")
    try:
        assert not models.present("ocr_devanagari")
        assert models.devanagari_font() is None
        assert not ocr_devanagari.available()
        # absent model: no engine and no error
        assert ocr_devanagari.get_engine() is None
        assert ocr_devanagari.load_error() is None
        assert ocr_devanagari.recognise([np.zeros((48, 100, 3), np.uint8)]) == []

        # a font with the wrong size does not count
        font = tmp_path / "models" / "fonts" / "NotoSansDevanagari-Regular.ttf"
        font.parent.mkdir(parents=True)
        font.write_bytes(b"x" * 10)
        assert models.devanagari_font() is None
    finally:
        reload_settings()


def test_devanagari_font_when_present():
    from app.ml import models

    if not models.path("ocr_devanagari", "fonts/NotoSansDevanagari-Regular.ttf").is_file():
        pytest.skip("Noto Sans Devanagari is not on disk")
    p = models.devanagari_font()
    assert p is not None and p.name == "NotoSansDevanagari-Regular.ttf" and p.stat().st_size == 277_680


# ------------------------------------------------------------------ rendered pages (models)

def _needs_models_and_fonts() -> Path:
    skip_without_models("ocr", "ocr_devanagari")
    if not features.check("raqm"):
        pytest.skip("Pillow has no raqm: Devanagari would not be shaped")
    from app.ml import models

    noto = models.devanagari_font()
    if noto is None or not Path(FREESANS).is_file() or not Path(FREEMONO).is_file():
        pytest.skip("fonts missing")
    return noto


def _page(lines: list[tuple[str, str]], size: int = 30, w: int = 1000, min_h: int = 800) -> np.ndarray:
    """A page-sized render. Pages shorter than the detector's 736 px input side are scaled up before
    detection, which on a short strip can split a Devanagari line into overlapping word boxes."""
    step = int(size * 2.3)
    img = Image.new("RGB", (w, max(min_h, 40 + step * len(lines))), "white")
    d = ImageDraw.Draw(img)
    for i, (font_path, text) in enumerate(lines):
        font = ImageFont.truetype(font_path, size, layout_engine=ImageFont.Layout.RAQM)
        d.text((40, 30 + step * i), text, fill="black", font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def test_reads_hindi_marathi_english_and_mixed_lines():
    noto = str(_needs_models_and_fonts())
    from app.ml import ocr

    lines = [(noto, HINDI_1), (FREESANS, ENGLISH_1), (noto, HINDI_2), (noto, HINDI_DATE), (noto, MIXED),
             (noto, MARATHI), (FREESANS, ENGLISH_2), (noto, MARATHI_AMOUNT)]
    # at 30 and 32 px the recogniser drops the subscript RA of "प्रिया"; 26, 28 and 34 to 40 px read exactly
    boxes = ocr.ocr(_page(lines, size=36))
    assert ocr.devanagari_active()
    got = ocr.text_of(boxes).splitlines()
    assert got == [t for _, t in lines], got
    assert all(0.5 <= b.score <= 1 and b.x0 < b.x1 and b.y0 < b.y1 for b in boxes)


def test_marathi_line_dropped_by_bundled_engine_is_kept():
    noto = str(_needs_models_and_fonts())
    from app.ml import ocr

    page = _page([(FREESANS, "Name"), (noto, MARATHI), (FREESANS, "End of page")], size=36)
    lines = ocr.text_of(ocr.ocr(page)).splitlines()
    assert lines == ["Name", MARATHI, "End of page"], lines


def test_latin_pages_unchanged_by_second_recogniser(monkeypatch):
    _needs_models_and_fonts()
    from app.ml import ocr, ocr_devanagari

    lines = [(FREESANS, "Invoice No. INV-2024/0091 dated 12 March 2024"),
             (FREESANS, "Total due: Rs 1,25,000.00 (incl. GST 18%)"),
             (FREEMONO, "Price ₹ 499 / € 12.50 / $ 9.99"),
             (FREESANS, "email: owner@example.com #42 & co"),
             (FREESANS, "Flight AI 865 Seat 12A Gate 23")]
    pages = [_page(lines, size=28), _page(lines, size=19)]
    with_both = [[(b.text, b.x0, b.y0) for b in ocr.ocr(p)] for p in pages]
    assert ocr.devanagari_active()
    monkeypatch.setattr(ocr_devanagari, "get_engine", lambda: None)
    bundled_only = [[(b.text, b.x0, b.y0) for b in ocr.ocr(p)] for p in pages]
    assert [[t for t, *_ in page] for page in with_both] == [[t for t, *_ in page] for page in bundled_only]
    for a, b in zip(with_both, bundled_only):
        for (_, ax, ay), (_, bx, by) in zip(a, b):
            assert abs(ax - bx) < 1e-3 and abs(ay - by) < 1e-3
    assert any("₹ 499" in t for t, *_ in with_both[0])


def test_without_devanagari_model_no_second_engine(monkeypatch):
    skip_without_models("ocr")
    from app.ml import ocr, ocr_devanagari

    monkeypatch.setattr(ocr_devanagari, "available", lambda: False)
    ocr.reset()
    try:
        ocr.get_engine()
        assert ocr.loaded() and not ocr_devanagari.loaded() and not ocr.devanagari_active()
        assert ocr_devanagari.load_error() is None
        font = ImageFont.truetype(FREESANS, 28) if Path(FREESANS).is_file() else ImageFont.load_default(28)
        img = Image.new("RGB", (700, 120), "white")
        ImageDraw.Draw(img).text((20, 40), "Bundled only 2026", fill="black", font=font)
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        assert "Bundled only 2026" in ocr.text_of(ocr.ocr(bgr))
    finally:
        ocr.reset()


def test_broken_devanagari_model_falls_back(monkeypatch, tmp_path):
    skip_without_models("ocr")
    from app.ml import ocr, ocr_devanagari

    bad = tmp_path / "broken.onnx"
    bad.write_bytes(b"not an onnx model")
    monkeypatch.setattr(ocr_devanagari, "available", lambda: True)
    monkeypatch.setattr(ocr_devanagari, "model_path", lambda: bad)
    ocr.reset()
    try:
        ocr.get_engine()
        assert ocr.loaded() and not ocr_devanagari.loaded()
        assert ocr_devanagari.load_error()
        from app.ml import models

        row = {r["key"]: r for r in models.statuses()}["ocr_devanagari"]
        assert row["loaded"] is False and row["error"].startswith("could not be loaded")
        assert ocr.ocr(np.full((200, 200, 3), 255, np.uint8)) == []
    finally:
        ocr.reset()


def test_devanagari_engine_downloads_nothing(monkeypatch):
    _needs_models_and_fonts()
    import requests
    from rapidocr.utils.download_file import DownloadFile

    from app.ml import models, ocr, ocr_devanagari

    def no_network(*a, **kw):
        raise AssertionError("rapidocr tried to download a model")

    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(DownloadFile, "_make_http_request", classmethod(lambda cls, url, logger: no_network()))
    ocr.reset()
    try:
        ocr.get_engine()
        assert ocr_devanagari.loaded() and ocr_devanagari.load_error() is None
        assert ocr_devanagari.provider() in ("CUDAExecutionProvider", "CPUExecutionProvider")
        eng = ocr_devanagari.get_engine()
        # only the recogniser of the second instance is loaded
        assert eng.text_rec is not None and eng.text_det is None and eng.text_cls is None
        row = {r["key"]: r for r in models.statuses()}["ocr_devanagari"]
        assert row["present"] and row["loaded"] and row["provider"] == ocr_devanagari.provider()
        page = _page([(str(models.devanagari_font()), HINDI_1)])
        assert ocr.text_of(ocr.ocr(page)) == HINDI_1
    finally:
        ocr.reset()
