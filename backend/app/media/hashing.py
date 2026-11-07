"""Content hashes and perceptual fingerprints.

CONTRACT.
- sha256 identifies byte-identical files (exact duplicates).
- pHash/dHash (64-bit, 16 hex chars) find visually similar candidates cheaply.
- The 64x64 grayscale signature lets dedup check pixel-level sameness, which separates a
  recompressed/resized copy (same pixels) from a burst shot (someone moved).
All fingerprints are computed from the display-oriented image (EXIF orientation applied).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.media.probe import MediaInfo

log = logging.getLogger(__name__)

SIG_SIZE = 64  # signature is SIG_SIZE x SIG_SIZE uint8 grayscale = 4096 bytes
VIDEO_FP_POSITIONS = tuple(round(0.05 + i * 0.9 / 11, 4) for i in range(12))  # 12 points, 5%..95%


@dataclass
class Fingerprint:
    phash: str  # 16 hex chars
    dhash: str  # 16 hex chars
    sig: bytes  # SIG_SIZE*SIG_SIZE grayscale bytes, image resized to exactly SIG_SIZE x SIG_SIZE (aspect ignored)
    frames: list[str] | None = None  # video: phash of the frame at each VIDEO_FP_POSITIONS (None entries -> "")


def sha256_file(path: Path) -> str:
    """Hex sha256, streamed in 1 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def phash(img: Image.Image) -> str:
    """DCT perceptual hash: grayscale 32x32 (LANCZOS), 2-D DCT (cv2.dct on float32), top-left 8x8,
    bit = coefficient > median of those 64 coefficients excluding [0,0]. 16 hex chars."""
    small = _gray(img).resize((32, 32), Image.Resampling.LANCZOS, reducing_gap=3.0)
    pixels = np.asarray(small, dtype=np.float32)
    coeffs = cv2.dct(pixels)[:8, :8].flatten()
    median = float(np.median(coeffs[1:]))
    return _bits_to_hex(coeffs > median)


def dhash(img: Image.Image) -> str:
    """Difference hash: grayscale 9x8, bit = left pixel > right pixel. 16 hex chars."""
    small = _gray(img).resize((9, 8), Image.Resampling.LANCZOS, reducing_gap=3.0)
    px = np.asarray(small, dtype=np.int16)
    return _bits_to_hex((px[:, :-1] > px[:, 1:]).flatten())


def signature(img: Image.Image) -> bytes:
    """Grayscale, resized to SIG_SIZE x SIG_SIZE with LANCZOS, as raw bytes."""
    small = _gray(img).resize((SIG_SIZE, SIG_SIZE), Image.Resampling.LANCZOS, reducing_gap=3.0)
    return small.tobytes()


def image_fingerprint(img: Image.Image) -> Fingerprint:
    gray = _gray(img)
    return Fingerprint(phash=phash(gray), dhash=dhash(gray), sig=signature(gray))


def video_fingerprint(path: Path, info: MediaInfo) -> Fingerprint:
    """frames = phash of frames at VIDEO_FP_POSITIONS * duration; phash/dhash/sig from the middle
    frame (50%). Raises ValueError if no frame decodes."""
    from app.media import video  # local import: video imports probe, hashing is imported by dedup

    duration = info.duration_s or 0.0
    times = [p * duration for p in VIDEO_FP_POSITIONS]
    frames = video.frames_at(path, info, times, max_side=256)
    frame_hashes = [phash(f) if f is not None else "" for f in frames]
    middle = video.frames_at(path, info, [0.5 * duration], max_side=256)[0]
    if middle is None:
        # nearest decoded fingerprint frame to the middle
        order = sorted(range(len(frames)), key=lambda i: abs(VIDEO_FP_POSITIONS[i] - 0.5))
        middle = next((frames[i] for i in order if frames[i] is not None), None)
    if middle is None:
        raise ValueError("no video frame could be decoded")
    gray = _gray(middle)
    return Fingerprint(phash=phash(gray), dhash=dhash(gray), sig=signature(gray), frames=frame_hashes)


def hamming(a: str, b: str) -> int:
    """Bit distance between two equal-length hex hashes (64 if either is empty/invalid)."""
    if not a or not b or len(a) != len(b):
        return 64
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except ValueError:
        return 64


def sig_diff(a: bytes, b: bytes) -> tuple[float, float]:
    """(mean absolute difference, max mean-abs-difference over the 8x8 grid of 8x8-pixel blocks),
    both on the 0..255 scale, after equalising the two signatures' mean brightness."""
    n = SIG_SIZE * SIG_SIZE
    if not a or not b or len(a) != n or len(b) != n:
        return 255.0, 255.0
    x = np.frombuffer(a, dtype=np.uint8).astype(np.float32).reshape(SIG_SIZE, SIG_SIZE)
    y = np.frombuffer(b, dtype=np.uint8).astype(np.float32).reshape(SIG_SIZE, SIG_SIZE)
    y = y + (x.mean() - y.mean())
    diff = np.abs(x - y)
    grid = SIG_SIZE // 8
    blocks = diff.reshape(grid, 8, grid, 8).mean(axis=(1, 3))
    return float(diff.mean()), float(blocks.max())


def _gray(img: Image.Image) -> Image.Image:
    return img if img.mode == "L" else img.convert("L")


def _bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.tolist():
        value = (value << 1) | int(bool(bit))
    return f"{value:0{(len(bits) + 3) // 4}x}"
