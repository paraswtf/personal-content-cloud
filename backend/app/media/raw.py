"""Camera RAW files (DNG, CR2, CR3, NEF, ARW, RAF, ORF, RW2, PEF, SRW).

CONTRACT.
- sniff_raw(data, filename) -> (kind, mime, ext) | None: recognises RAW files from their leading
  bytes, with the file name's extension deciding between TIFF-based formats that share a header
  (NEF, ARW, PEF, SRW). Must run BEFORE the plain TIFF and ftyp sniffers: CR2, NEF, ARW and DNG
  start with a TIFF header and CR3 is an ISO media file with the 'crx ' brand.
- is_raw_path(path): True for a stored original whose extension is a RAW extension.
- probe_raw(path, kind_mime_ext) -> (width, height, exif): display dimensions (camera flip
  applied) and the Pillow EXIF of the file or of its embedded preview (may be empty). ValueError
  when LibRaw cannot open the file.
- load_raw(path, max_side=None) -> PIL RGB image: half-size demosaic (fast, enough for previews
  and analysis), camera white balance, flip applied; falls back to the embedded JPEG preview when
  demosaicing fails. ValueError when nothing decodes.
- A TIFF-based RAW that LibRaw cannot open at all (an iPhone ProRAW DNG whose raw image is JPEG XL
  compressed, DNG 1.7, which this LibRaw build cannot decode) is read from its largest embedded JPEG
  preview instead (embedded_preview: IFD0, the chained IFDs and their SubIFDs; an iPhone keeps a
  full-size one), turned upright once by its own EXIF orientation or else the file's; probe_raw then
  takes the size from that preview and the EXIF from the file (2026-11-09).
The original bytes are never modified.
"""
from __future__ import annotations

import io
import logging
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

RAW_MIME = {
    ".dng": "image/x-adobe-dng",
    ".cr2": "image/x-canon-cr2",
    ".cr3": "image/x-canon-cr3",
    ".nef": "image/x-nikon-nef",
    ".arw": "image/x-sony-arw",
    ".raf": "image/x-fuji-raf",
    ".orf": "image/x-olympus-orf",
    ".rw2": "image/x-panasonic-rw2",
    ".pef": "image/x-pentax-pef",
    ".srw": "image/x-samsung-srw",
}
RAW_EXTS = frozenset(RAW_MIME)
# TIFF-based RAW formats that only the extension tells apart from a plain TIFF.
_TIFF_RAW_BY_EXT = frozenset({".nef", ".arw", ".pef", ".srw", ".dng", ".cr2"})
_DNG_VERSION_TAG = 0xC612


def _raw(ext: str) -> tuple[str, str, str]:
    return "image", RAW_MIME[ext], ext


def _ext_of(filename: str | None) -> str:
    if not filename:
        return ""
    name = filename.rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def _tiff_ifd0_tags(data: bytes) -> set[int]:
    """Tag numbers of the first IFD of a TIFF header (empty when it lies beyond `data`)."""
    if len(data) < 8:
        return set()
    endian = "<" if data[:2] == b"II" else ">"
    try:
        offset = struct.unpack(endian + "I", data[4:8])[0]
        if offset + 2 > len(data):
            return set()
        count = struct.unpack(endian + "H", data[offset:offset + 2])[0]
        tags = set()
        for i in range(min(count, 512)):
            start = offset + 2 + i * 12
            if start + 2 > len(data):
                break
            tags.add(struct.unpack(endian + "H", data[start:start + 2])[0])
        return tags
    except struct.error:
        return set()


def sniff_raw(data: bytes, filename: str | None = None) -> tuple[str, str, str] | None:
    """(kind, mime, ext) for a camera RAW file, or None."""
    if len(data) < 16:
        return None
    ext = _ext_of(filename)
    if data[:15] == b"FUJIFILMCCD-RAW":
        return _raw(".raf")
    if data[:4] in (b"IIRO", b"IIRS", b"MMOR"):
        return _raw(".orf")
    if data[:4] == b"IIU\x00":
        return _raw(".rw2")
    if data[4:8] == b"ftyp" and data[8:12] == b"crx ":
        return _raw(".cr3")
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        if data[8:10] == b"CR":
            return _raw(".cr2")
        if _DNG_VERSION_TAG in _tiff_ifd0_tags(data):
            return _raw(".dng")
        if ext in _TIFF_RAW_BY_EXT:
            return _raw(ext)
    return None


def is_raw_path(path: Path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTS


def _exif_from(path: Path, raw_obj) -> Image.Exif:
    """EXIF of the RAW file itself (TIFF-based formats) or of its embedded JPEG preview."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if exif:
                return exif
    except Exception:  # not a container Pillow reads (CR3, RAF, ORF, RW2)
        pass
    try:
        thumb = raw_obj.extract_thumb()
        import rawpy

        if thumb.format == rawpy.ThumbFormat.JPEG:
            with Image.open(io.BytesIO(thumb.data)) as im:
                return im.getexif()
    except Exception:  # no embedded preview
        pass
    return Image.Exif()


_JPEG_COMPRESSION = (6, 7)  # old-style and new-style JPEG in TIFF
_MIN_PREVIEW_SIDE = 640  # a smaller embedded image is a thumbnail, not a picture to show or analyse


def _ifds(data: bytes) -> list[dict[int, tuple[int, int, int]]]:
    """Every IFD of a TIFF file (IFD0 and its chain, and their SubIFDs one level down) as {tag: (type, count,
    value or offset)}. Empty on a malformed header."""
    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        return []
    e = "<" if data[:2] == b"II" else ">"
    out: list[dict[int, tuple[int, int, int]]] = []
    seen: set[int] = set()

    def read(off: int, depth: int) -> int:
        if off in seen or off + 2 > len(data) or len(out) > 64:
            return 0
        seen.add(off)
        n = struct.unpack(e + "H", data[off:off + 2])[0]
        tags: dict[int, tuple[int, int, int]] = {}
        for i in range(min(n, 512)):
            at = off + 2 + 12 * i
            if at + 12 > len(data):
                break
            tag, typ, cnt = struct.unpack(e + "HHI", data[at:at + 8])
            if typ == 3 and cnt == 1:
                val = struct.unpack(e + "H", data[at + 8:at + 10])[0]
            else:
                val = struct.unpack(e + "I", data[at + 8:at + 12])[0]
            tags[tag] = (typ, cnt, val)
        out.append(tags)
        if 330 in tags and depth == 0:  # SubIFDs
            typ, cnt, val = tags[330]
            offs = [val] if cnt == 1 else [struct.unpack(e + "I", data[val + 4 * k:val + 4 * k + 4])[0]
                                           for k in range(min(cnt, 16)) if val + 4 * k + 4 <= len(data)]
            for o in offs:
                read(o, depth + 1)
        end = off + 2 + 12 * n
        return struct.unpack(e + "I", data[end:end + 4])[0] if end + 4 <= len(data) else 0

    nxt = struct.unpack(e + "I", data[4:8])[0]
    while nxt and len(out) < 64:
        nxt = read(nxt, 0)
    return out


def embedded_preview(path: Path) -> Image.Image | None:
    """The largest JPEG picture embedded in a TIFF-based RAW file, upright (its own EXIF orientation, else the
    file's), in RGB; None when there is none at least _MIN_PREVIEW_SIDE pixels on its long side."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    ifds = _ifds(data)
    if not ifds:
        return None
    best: tuple[int, bytes] | None = None
    for tags in ifds:
        comp = tags.get(259, (0, 0, 0))[2]
        width, height = tags.get(256, (0, 0, 0))[2], tags.get(257, (0, 0, 0))[2]
        chunk = None
        if 513 in tags and 514 in tags:  # JPEGInterchangeFormat and its length
            chunk = (tags[513][2], tags[514][2])
        elif comp in _JPEG_COMPRESSION and 273 in tags and 279 in tags and tags[273][1] == 1:
            chunk = (tags[273][2], tags[279][2])  # one strip holding a whole JPEG
        if chunk is None or chunk[0] + chunk[1] > len(data) or chunk[1] < 4:
            continue
        blob = data[chunk[0]:chunk[0] + chunk[1]]
        if blob[:2] != b"\xff\xd8":
            continue
        area = int(width) * int(height) if width and height else len(blob)
        if best is None or area > best[0]:
            best = (area, blob)
    if best is None:
        return None
    try:
        with Image.open(io.BytesIO(best[1])) as im:
            im.load()
            exif = im.getexif()
            if not exif.get(0x0112):
                orientation = ifds[0].get(0x0112, (0, 0, 1))[2]
                if orientation and orientation != 1:
                    exif[0x0112] = orientation
            img = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:  # noqa: BLE001 - not a JPEG Pillow decodes
        return None
    if max(img.size) < _MIN_PREVIEW_SIDE:
        return None
    return img


def probe_raw(path: Path, kind_mime_ext: tuple[str, str, str]) -> tuple[int, int, Image.Exif]:
    """(display width, display height, EXIF) of a RAW file. ValueError when neither LibRaw nor an embedded
    preview can read it."""
    try:
        import rawpy
    except ImportError as e:  # pragma: no cover - declared dependency
        raise ValueError("rawpy is not installed") from e
    try:
        with rawpy.imread(str(path)) as raw:
            sizes = raw.sizes
            width, height = int(sizes.width), int(sizes.height)
            if sizes.flip in (5, 6):
                width, height = height, width
            exif = _exif_from(Path(path), raw)
    except Exception as e:  # LibRawError and friends: LibRaw cannot read it, perhaps its preview can
        preview = embedded_preview(Path(path))
        if preview is None:
            raise ValueError(f"cannot read RAW file: {e}") from e
        log.info("RAW %s: LibRaw cannot read it (%s), using its embedded preview", path, e)
        width, height = preview.size
        exif = Image.Exif()
        try:
            with Image.open(path) as im:
                exif = im.getexif()
        except Exception:  # noqa: BLE001 - the preview alone then
            pass
    if width <= 0 or height <= 0:
        raise ValueError("RAW file has no pixels")
    return width, height, exif


def load_raw(path: Path, max_side: int | None = None) -> Image.Image:
    """Half-size demosaic to RGB (flip applied), or the embedded preview when that fails."""
    try:
        import rawpy
    except ImportError as e:  # pragma: no cover - declared dependency
        raise ValueError("rawpy is not installed") from e
    img: Image.Image | None = None
    try:
        with rawpy.imread(str(path)) as raw:
            try:
                rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                img = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
            except Exception as e:  # unsupported sensor layout: use the camera's own preview
                log.info("RAW demosaic failed for %s (%s), using the embedded preview", path, e)
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    with Image.open(io.BytesIO(thumb.data)) as im:
                        im.load()
                        img = ImageOps.exif_transpose(im).convert("RGB")
                else:
                    img = Image.fromarray(np.ascontiguousarray(thumb.data), "RGB")
    except ValueError:
        raise
    except Exception as e:
        img = embedded_preview(Path(path))  # LibRaw cannot open it: its embedded preview (a ProRAW JPEG XL DNG)
        if img is None:
            raise ValueError(f"cannot decode RAW file: {e}") from e
    if img is None:
        raise ValueError("RAW file has no decodable image")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS, reducing_gap=3.0)
    return img
