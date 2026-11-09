"""Photo decoding and derivative images (thumbnails, large and small previews).

CONTRACT. Originals are never modified; everything here reads them and writes WebP derivatives
atomically (temp file in the same dir + os.replace).
- load_oriented(path, max_side=None, *, raw=None) -> RGB PIL image, EXIF orientation applied.
  Camera RAW files (by extension, or raw=True for files without one) go through app.media.raw.
- make_image_derivatives(img, thumb_path, preview_path, preview_small_path) (no small file when the
  large preview already fits the small size; small_is_large(size) says when)
- save_webp(img, path, max_side, quality); to_bgr(img) -> contiguous HxWx3 uint8 BGR array.
"""
from __future__ import annotations

import logging
import os
import uuid
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.config import settings

log = logging.getLogger(__name__)

# Decompression-bomb guard shared by probe and decoding: Pillow raises DecompressionBombError
# above 2x this value (250 MP); between 1x and 2x it only warns, which we silence.
Image.MAX_IMAGE_PIXELS = 125_000_000
MAX_PIXELS = 2 * Image.MAX_IMAGE_PIXELS
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

_heif_registered = False


def register_heif() -> None:
    """Register pillow-heif's HEIF/HEIC opener with Pillow (idempotent). Called at import time."""
    global _heif_registered
    if _heif_registered:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _heif_registered = True
    except ImportError:  # pragma: no cover - dependency is declared, but stay importable without it
        log.warning("pillow-heif is not installed; HEIC/HEIF photos cannot be decoded")


register_heif()

_jxl_registered = False


def register_jxl() -> None:
    """Register pillow-jxl-plugin's JPEG XL opener with Pillow (idempotent). Called at import time."""
    global _jxl_registered
    if _jxl_registered:
        return
    try:
        import pillow_jxl  # noqa: F401 - importing it registers the opener

        _jxl_registered = True
    except ImportError:  # pragma: no cover - dependency is declared, but stay importable without it
        log.warning("pillow-jxl-plugin is not installed; JPEG XL photos cannot be decoded")


register_jxl()


def _flatten_rgb(im: Image.Image) -> Image.Image:
    """Any mode -> RGB, compositing transparency onto white."""
    if im.mode == "RGB":
        return im
    if im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    elif im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I"):
        arr = np.asarray(im, dtype=np.uint32)
        if arr.max(initial=0) > 255:
            arr = arr >> 8
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")
    if im.mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return im.convert("RGB")


def load_oriented(path: Path, max_side: int | None = None, *, raw: bool | None = None) -> Image.Image:
    """Open a photo as RGB with EXIF orientation applied (ImageOps.exif_transpose).

    Animated images: first frame. Transparency is flattened onto white. If max_side is given,
    downscale (use JPEG draft mode when possible) so the longer side <= max_side.
    Raises ValueError if the file cannot be decoded. RAW files are demosaiced at half size.
    """
    from app.media import raw as raw_media

    if raw or (raw is None and raw_media.is_raw_path(Path(path))):
        return raw_media.load_raw(Path(path), max_side)
    register_heif()
    try:
        with Image.open(path) as im:
            if im.width * im.height > MAX_PIXELS:
                raise ValueError(f"image too large ({im.width}x{im.height})")
            if max_side and im.format == "JPEG":
                longer = max(im.size)
                if longer > max_side:
                    s = max_side / longer
                    im.draft("RGB", (max(1, int(im.width * s)), max(1, int(im.height * s))))
            im.load()
            # JPEG XL carries its orientation in the codestream, applied by the decoder; an Exif box's
            # orientation must be ignored (the format's rule), or the picture turns twice.
            out = im if im.format == "JXL" else ImageOps.exif_transpose(im)
            out = _flatten_rgb(out)
            if out is im:
                out = im.copy()
    except ValueError:
        raise
    except Exception as e:  # OSError, SyntaxError, DecompressionBombError, decoder errors
        raise ValueError(f"cannot decode image: {e}") from e
    if max_side and max(out.size) > max_side:
        out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS, reducing_gap=3.0)
    return out


def _fit(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale (LANCZOS, never upscale) so the longer side <= max_side. May return img itself."""
    w, h = img.size
    longer = max(w, h)
    if longer <= max_side:
        return img
    s = max_side / longer
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return img.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def save_webp(img: Image.Image, path: Path, max_side: int, quality: int) -> None:
    """Resize (LANCZOS, never upscale) so the longer side <= max_side and save as WebP atomically."""
    out = _fit(img, max_side)
    if out.mode not in ("RGB", "RGBA"):
        out = _flatten_rgb(out)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        out.save(tmp, "WEBP", quality=quality, method=4)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def make_image_derivatives(img: Image.Image, thumb_path: Path, preview_path: Path,
                           preview_small_path: Path) -> None:
    """Write the large preview (settings.preview_size), the small preview (settings.preview_small_size,
    resized from the large one) and the grid thumbnail (settings.thumb_size, quality 80) from an
    already-oriented RGB image. Both previews use settings.preview_quality. When the large preview
    already fits the small size, the small one would be the same bytes: none is written (a stale one is
    removed) and the small preview URL serves the large file (api/files.py, docs section 3)."""
    quality = settings.preview_quality
    preview = _fit(img, settings.preview_size)
    save_webp(preview, preview_path, settings.preview_size, quality)
    if small_is_large(preview.size):
        small = preview
        Path(preview_small_path).unlink(missing_ok=True)
    else:
        small = _fit(preview, settings.preview_small_size)
        save_webp(small, preview_small_path, settings.preview_small_size, quality)
    save_webp(small, thumb_path, settings.thumb_size, 80)


def small_is_large(size: tuple[int, int]) -> bool:
    """True when a large preview of this pixel size is no bigger than the small preview size, so the
    small preview would repeat it."""
    return max(int(size[0]), int(size[1])) <= int(settings.preview_small_size)


def to_bgr(img: Image.Image) -> np.ndarray:
    """RGB PIL image -> contiguous uint8 BGR array for OpenCV / the face engine."""
    if img.mode != "RGB":
        img = _flatten_rgb(img)
    arr = np.asarray(img, dtype=np.uint8)
    return np.ascontiguousarray(arr[:, :, ::-1])
