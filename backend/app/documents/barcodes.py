"""Barcode reading with zxing-cpp (QR, PDF417, Aztec, Code 128, Data Matrix, EAN and the rest).

Contract: available() -> bool; decode(img) -> list[(format, text)] for a numpy image (BGR, BGRA or grey,
uint8), formats named as zxing-cpp's BarcodeFormat names ('QRCode', 'PDF417', 'Aztec', 'Code128', ...),
duplicates removed, in reading order. A second pass on an Otsu-binarised, 2x upscaled copy runs when
the first finds nothing on a small image. Never raises: errors are logged and give [].
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

MAX_SIDE = 3000
SMALL_SIDE = 1200


def available() -> bool:
    try:
        import zxingcpp  # noqa: F401
    except Exception:
        return False
    return True


def _grey(img: np.ndarray) -> np.ndarray:
    import cv2

    if img.ndim == 2:
        g = img
    elif img.shape[2] == 4:
        g = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    h, w = g.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        g = cv2.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(g)


def format_name(fmt) -> str:
    name = getattr(fmt, "name", None)
    if name:
        return str(name)
    return str(fmt).split(".")[-1].replace(" ", "")


def _read(g: np.ndarray) -> list[tuple[str, str]]:
    import zxingcpp

    out: list[tuple[str, str]] = []
    for b in zxingcpp.read_barcodes(g):
        if getattr(b, "valid", True) is False:
            continue
        text = (b.text or "").strip("\x00")
        if not text:
            continue
        item = (format_name(b.format), text)
        if item not in out:
            out.append(item)
    return out


def decode(img: np.ndarray | None) -> list[tuple[str, str]]:
    if img is None or not isinstance(img, np.ndarray) or img.size == 0 or not available():
        return []
    try:
        g = _grey(img)
        found = _read(g)
        if not found and max(g.shape[:2]) <= SMALL_SIDE:
            import cv2

            big = cv2.resize(g, (g.shape[1] * 2, g.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
            _t, bw = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            found = _read(np.ascontiguousarray(bw))
        return found
    except Exception:
        log.warning("barcode decode failed", exc_info=True)
        return []
