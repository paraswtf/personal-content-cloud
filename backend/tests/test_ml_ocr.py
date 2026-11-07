"""app.ml.ocr: reads rendered text, empty for blank images, no downloads, reading order."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tests.conftest import skip_without_models
from tests.helpers import text_image

FREESANS = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"


def _bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _font_fallback(monkeypatch):
    """helpers.text_image asks for DejaVuSans and otherwise falls back to Pillow's 10 px default font,
    which the detector misses as a lone line on a blank page. DejaVu is not installed on this machine,
    so point that request at a font that exists (FreeSans, else Pillow's scalable default)."""
    real = ImageFont.truetype

    def truetype(font=None, size=10, *a, **kw):
        try:
            return real(font, size, *a, **kw)
        except OSError:
            try:
                return real(FREESANS, size)
            except OSError:
                return ImageFont.load_default(size)

    monkeypatch.setattr(ImageFont, "truetype", truetype)


def test_ocr_reads_hello(monkeypatch):
    skip_without_models("ocr")
    from app.ml import ocr

    _font_fallback(monkeypatch)

    assert ocr.available()
    boxes = ocr.ocr(_bgr(text_image(["Hello World 12345"])))
    text = " ".join(b.text for b in boxes)
    assert "Hello" in text
    assert all(0 <= b.score <= 1 and b.x0 < b.x1 and b.y0 < b.y1 for b in boxes)


def test_ocr_lines_in_reading_order():
    skip_without_models("ocr")
    from app.ml import ocr

    try:
        font = ImageFont.truetype(FREESANS, 28)
    except OSError:
        font = ImageFont.load_default(28)
    img = Image.new("RGB", (1000, 400), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 30), "First line here", fill="black", font=font)
    d.text((600, 32), "RIGHT", fill="black", font=font)
    d.text((40, 120), "Second line 2024", fill="black", font=font)
    boxes = ocr.ocr(_bgr(img))
    texts = [b.text for b in boxes]
    joined = " | ".join(texts)
    assert joined.index("First") < joined.index("RIGHT") < joined.index("Second"), texts
    assert "Second line 2024" in ocr.text_of(boxes).splitlines()[-1]


def test_ocr_blank_and_tiny_images():
    skip_without_models("ocr")
    from app.ml import ocr

    assert ocr.ocr(np.full((300, 300, 3), 255, np.uint8)) == []
    assert ocr.ocr(np.zeros((4, 4, 3), np.uint8)) == []
    assert ocr.ocr(np.full((200, 200), 128, np.uint8)) == []


def test_ocr_engine_downloads_nothing(monkeypatch):
    skip_without_models("ocr")
    import requests
    from rapidocr.utils.download_file import DownloadFile

    from app.ml import ocr

    def no_network(*a, **kw):
        raise AssertionError("rapidocr tried to download a model")

    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(DownloadFile, "_make_http_request", classmethod(lambda cls, url, logger: no_network()))
    ocr.reset()
    try:
        eng = ocr.get_engine()
        assert ocr.loaded()
        assert ocr.provider() in ("CUDAExecutionProvider", "CPUExecutionProvider")
        # det, cls and rec are loaded lazily; the warm-up must have reached all three
        assert eng.text_det is not None and eng.text_cls is not None and eng.text_rec is not None
        font = ImageFont.truetype(FREESANS, 28) if Path(FREESANS).is_file() else ImageFont.load_default(28)
        img = Image.new("RGB", (600, 120), "white")
        ImageDraw.Draw(img).text((20, 40), "Offline check 42", fill="black", font=font)
        assert "Offline" in " ".join(b.text for b in ocr.ocr(_bgr(img)))
    finally:
        ocr.reset()


def test_reading_order_pure():
    from app.ml.ocr import OcrBox, reading_order, text_of

    b = [
        OcrBox(300, 102, 400, 122, "world", 1),
        OcrBox(10, 200, 100, 220, "next", 1),
        OcrBox(10, 100, 120, 120, "hello", 1),
    ]
    out = reading_order(b)
    assert [x.text for x in out] == ["hello", "world", "next"]
    assert text_of(out) == "hello world\nnext"
    assert reading_order([]) == [] and text_of([]) == ""
