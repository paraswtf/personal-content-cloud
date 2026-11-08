"""Identify uploaded files and read their metadata.

CONTRACT. Every non-empty file is accepted and kept byte for byte; probing only decides how it is
processed. Kinds, judged by content (magic bytes plus a successful decode or probe), with the
original file name used only where content cannot tell (plain text, some camera RAW formats):
  image     JPEG, PNG, GIF, WebP, HEIC/HEIF, AVIF, TIFF, BMP, camera RAW (app.media.raw)
  video     MP4/M4V/3GP/MOV, WebM/MKV, AVI, MPEG-TS/M2TS
  pdf       PDF (app.media.documents)
  document  DOCX, XLSX, PPTX, plain text / Markdown / CSV (app.media.documents)
  other     anything else, and photos/videos/documents that do not decode: stored, no preview.
probe(path, filename=None) -> MediaInfo | None: None only for an empty or unreadable file.
Raises ToolFailure when ffprobe/ffmpeg could not run (no verdict on the file: retry later).
ValueError inside the probe_* helpers means a bad file (probe() then falls back to 'other').
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import struct
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from PIL import Image

from app.config import settings
# Importing images registers the HEIF opener and sets the decompression-bomb limit.
from app.media import images as _images
from app.media import raw as _raw

log = logging.getLogger(__name__)

# Value for the upload page's <input accept>. Any file is accepted; the list helps pickers show the
# usual ones. Browsers often report HEIC/MOV oddly, so list extensions too.
ACCEPT_ATTR = ("image/*,video/*,application/pdf,.heic,.heif,.avif,.mov,.m4v,.mkv,.3gp,.mts,.m2ts,"
               ".dng,.cr2,.cr3,.nef,.arw,.raf,.orf,.rw2,.pef,.srw,.pdf,.docx,.xlsx,.pptx,.txt,.md,.csv")
OTHER_MIME = "application/octet-stream"
_SAFE_EXT = re.compile(r"^\.[a-z0-9]{1,8}$")

FFPROBE_TIMEOUT_S = 60
DECODE_CHECK_TIMEOUT_S = 60
SNIFF_BYTES = 64 * 1024
# No camera records beyond these; such streams only exist to make decoding expensive.
MAX_VIDEO_PIXELS = 8192 * 4320
MAX_VIDEO_FPS = 300.0


class ToolFailure(RuntimeError):
    """ffmpeg/ffprobe could not do its job (timeout, spawn failure, killed by a signal, or a resource
    error such as no memory or too many open files). Says nothing about the file, so callers must
    retry later instead of rejecting the upload."""


# stderr of ffmpeg/ffprobe that exited non-zero for want of resources, not because of the file.
TOOL_TROUBLE = ("cannot allocate memory", "out of memory", "resource temporarily unavailable",
                "too many open files", "no space left on device", "input/output error", "error allocating")


# Bumped when the probe learns to read files it used to keep as 'other'; ingest.reprobe_plain then
# probes those again once (kv probe_version). 2: a placeholder index before the real one (input_flags).
PROBE_VERSION = 2


def _top_atoms(path: Path, limit: int = 64) -> list[tuple[bytes, int, int]]:
    """(type, offset, size) of an ISO media file's top-level atoms; [] when it is not one."""
    out: list[tuple[bytes, int, int]] = []
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            pos = 0
            while pos + 8 <= size and len(out) < limit:
                f.seek(pos)
                head = f.read(8)
                if len(head) < 8:
                    break
                n, typ = struct.unpack(">I4s", head)
                if n == 1:
                    big = f.read(8)
                    if len(big) < 8:
                        break
                    n = struct.unpack(">Q", big)[0]
                elif n == 0:
                    n = size - pos
                if n < 8:
                    break
                if not out and typ != b"ftyp":
                    return []
                out.append((typ, pos, n))
                pos += n
    except OSError:
        return []
    return out


@functools.lru_cache(maxsize=1024)
def _placeholder_index(path: str, size: int, mtime_ns: int) -> bool:
    atoms = _top_atoms(Path(path))
    names = [a[0] for a in atoms]
    if b"moov" not in names:
        return False
    first_moov = names.index(b"moov")
    try:
        with open(path, "rb") as f:
            for typ, pos, n in atoms[:first_moov]:
                if typ in (b"hoov", b"free") and n >= 16:
                    f.seek(pos + 8)
                    child = f.read(8)
                    if len(child) == 8 and child[4:8] in (b"mvhd", b"cmov"):
                        return True
    except OSError:
        return False
    return False


def input_flags(path: Path | str) -> list[str]:
    """Input options ffmpeg and ffprobe need for this file: ["-strict", "strict"] when a placeholder
    index (a 'hoov' or 'free' atom holding an 'mvhd', as some Apple encoders leave it) comes before the
    real 'moov'. ffmpeg's lenient default reads the placeholder, finds no packets and calls the video
    undecodable; strict mode skips it and reads the real index. [] for every other file."""
    try:
        st = os.stat(path)
    except OSError:
        return []
    return ["-strict", "strict"] if _placeholder_index(str(path), st.st_size, st.st_mtime_ns) else []


def tool_trouble(stderr: bytes | str | None) -> str | None:
    """The resource problem a tool's stderr names, or None when it describes the file."""
    if not stderr:
        return None
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
    low = text.lower()
    for mark in TOOL_TROUBLE:
        if mark in low:
            return mark
    return None


@dataclass
class MediaInfo:
    kind: Literal["image", "video", "pdf", "document", "other"]
    mime: str  # canonical MIME from sniffing, e.g. 'image/jpeg', 'video/mp4', 'application/pdf';
    #            'application/octet-stream' for 'other' files of unknown type
    ext: str  # canonical storage extension, lowercase with dot: .jpg .png .webp .gif .heic .avif .tif .bmp
    #           .dng .cr2 ... .mp4 .mov .m4v .webm .mkv .avi .3gp .mts .pdf .docx .xlsx .pptx .txt .md .csv;
    #           for 'other' the uploaded name's suffix when it is safe, else .bin
    width: int | None  # display orientation (EXIF orientation / video rotation applied); for documents
    height: int | None  # the page 1 render size; None for 'other'
    duration_s: float | None = None
    bitrate: int | None = None  # bits/s; video: format bit_rate or size*8/duration
    video_codec: str | None = None  # ffprobe codec_name: h264, hevc, vp9, av1, ...
    audio_codec: str | None = None
    fps: float | None = None
    has_audio: bool | None = None
    rotation: int = 0  # video display rotation, clockwise degrees (0/90/180/270), already reflected in width/height;
    # ffmpeg's `-display_rotation 90` (counter-clockwise) therefore probes as 270
    taken_at: str | None = None  # UTC, same format as util.now_iso ('YYYY-MM-DDTHH:MM:SS.000Z'): EXIF
    #           DateTimeOriginal [+OffsetTimeOriginal] or video creation time; naive times are read in
    #           settings.local_timezone
    exif_datetime: str | None = None  # raw 'YYYY:MM:DD HH:MM:SS' (images)
    exif_subsec: str | None = None  # raw SubSecTimeOriginal (images)
    camera: str | None = None  # 'Make Model' trimmed (just Model when it starts with the make), None if absent
    orientation: int = 1  # EXIF orientation tag (images)
    animated: bool = False  # animated GIF/WebP (stored as image; first frame used for analysis)
    gps_lat: float | None = None  # capture position, decimal degrees (EXIF GPS IFD / ISO 6709 video tags)
    gps_lon: float | None = None
    gps_alt: float | None = None  # metres above sea level (negative below)
    page_count: int | None = None  # pdf / document: pages (logical pages for office files)
    raw: bool = False  # camera RAW (decoded through app.media.raw)
    reason: str | None = None  # 'other' files that sniffed as a known type but did not decode: why
    content_id: str | None = None  # Apple ContentIdentifier shared by a Live Photo's still and MOV
    strict_input: bool = False  # video with a placeholder index (input_flags): never streamed as it is

    @property
    def pixels(self) -> int:
        return int(self.width or 0) * int(self.height or 0)

    @property
    def text_capable(self) -> bool:
        """The file itself carries text to extract (PDF text layer, office or plain text)."""
        return self.kind in ("pdf", "document")

    @property
    def visual(self) -> bool:
        """Has pixels to preview and analyse (images and videos)."""
        return self.kind in ("image", "video")


# ------------------------------------------------------------------ sniffing
_JPEG = ("image", "image/jpeg", ".jpg")
_PNG = ("image", "image/png", ".png")
_GIF = ("image", "image/gif", ".gif")
_WEBP = ("image", "image/webp", ".webp")
_HEIC = ("image", "image/heic", ".heic")
_HEIF = ("image", "image/heif", ".heif")
_AVIF = ("image", "image/avif", ".avif")
_TIFF = ("image", "image/tiff", ".tif")
_BMP = ("image", "image/bmp", ".bmp")
_MP4 = ("video", "video/mp4", ".mp4")
_M4V = ("video", "video/x-m4v", ".m4v")
_3GP = ("video", "video/3gpp", ".3gp")
_MOV = ("video", "video/quicktime", ".mov")
_WEBM = ("video", "video/webm", ".webm")
_MKV = ("video", "video/x-matroska", ".mkv")
_AVI = ("video", "video/x-msvideo", ".avi")
_MTS = ("video", "video/mp2t", ".mts")

_HEIC_BRANDS = {b"heic", b"heix", b"hevc", b"heim", b"heis", b"hevm", b"hevs", b"hevx"}
_HEIF_BRANDS = {b"mif1", b"msf1", b"mif2"}
_AVIF_BRANDS = {b"avif", b"avis"}
_AUDIO_OR_RAW_BRANDS = {b"M4A ", b"M4B ", b"M4P ", b"F4A ", b"F4B ", b"crx ", b"jp2 ", b"jpx ", b"jpm ", b"mjp2"}
_MP4_BRANDS = {
    b"isom", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"iso7", b"iso8", b"iso9", b"mp41", b"mp42",
    b"mp71", b"avc1", b"dash", b"mmp4", b"MSNV", b"NDAS", b"NDSC", b"NDSH", b"NDSM", b"NDSP", b"NDSS",
    b"NDXC", b"NDXH", b"NDXM", b"NDXP", b"NDXS", b"XAVC", b"MPUF", b"f4v ", b"av01", b"hvc1", b"cmfc",
}
_QT_ATOMS = {b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}
_BMP_DIB_SIZES = {12, 16, 40, 52, 56, 64, 108, 124}


def _sniff_ftyp(data: bytes) -> tuple[str, str, str] | None:
    size = struct.unpack(">I", data[0:4])[0]
    if size == 1 and len(data) >= 16:  # 64-bit size
        size = struct.unpack(">Q", data[8:16])[0]
        body = data[16:]
    else:
        body = data[8:]
    if size == 0 or size > len(data):
        size = len(data)
    major = body[0:4]
    end = max(8, min(size - (len(data) - len(body)), len(body)))
    compat = {body[i:i + 4] for i in range(8, end - 3, 4)}
    brands = {major} | compat

    if major in _AUDIO_OR_RAW_BRANDS:
        return None
    # Still images first: ffmpeg can demux HEIF/AVIF, so never let them fall through to video.
    if major in _AVIF_BRANDS:
        return _AVIF
    if major in _HEIC_BRANDS:
        return _HEIC
    if major in _HEIF_BRANDS:
        if compat & _AVIF_BRANDS:
            return _AVIF
        if compat & _HEIC_BRANDS:
            return _HEIC
        return _HEIF
    if major == b"qt  ":
        return _MOV
    if major in (b"M4V ", b"M4VH", b"M4VP"):
        return _M4V
    if major[:3] in (b"3gp", b"3g2", b"3ge", b"3gg", b"3gs"):
        return _3GP
    if major in _MP4_BRANDS:
        return _MP4
    # Unknown major brand: decide from the compatible brands.
    if compat & _AVIF_BRANDS and not compat & _MP4_BRANDS:
        return _AVIF
    if compat & _HEIC_BRANDS and not compat & _MP4_BRANDS:
        return _HEIC
    if compat & _HEIF_BRANDS and not compat & _MP4_BRANDS:
        return _HEIF
    if b"qt  " in brands:
        return _MOV
    if compat & _MP4_BRANDS:
        return _MP4
    return None


def _sniff_ebml(data: bytes) -> tuple[str, str, str] | None:
    i = data.find(b"\x42\x82", 4, 4096)
    if i < 0:
        return None
    first = data[i + 2] if i + 2 < len(data) else 0
    if first == 0:
        return None
    # EBML variable-length size: the number of leading zero bits gives the length.
    length = 8 - first.bit_length() + 1
    size = first & ((1 << (8 - length)) - 1)
    for b in data[i + 3:i + 2 + length]:
        size = (size << 8) | b
    start = i + 2 + length
    doctype = data[start:start + size].rstrip(b"\x00")
    if doctype == b"webm":
        return _WEBM
    if doctype == b"matroska":
        return _MKV
    return None


def _is_ts(data: bytes) -> bool:
    if len(data) >= 377 and data[0] == data[188] == data[376] == 0x47:
        return True
    return len(data) >= 389 and data[4] == data[196] == data[388] == 0x47


def sniff(path: Path, filename: str | None = None) -> tuple[str, str, str] | None:
    """Return (kind, mime, ext) from the file's leading bytes, or None if not a known type.

    Order: camera RAW (before TIFF and ftyp, which RAW formats reuse), images and videos, then
    documents (PDF, office zip, plain text by the file name's extension).

    Must recognise: JPEG, PNG, GIF, WebP, HEIC/HEIF (ftyp heic/heix/hevc/heim/heis/hevm/hevs/mif1/msf1),
    AVIF (ftyp avif/avis), TIFF, BMP; MP4/M4V/3GP/MOV (ftyp brands, or QuickTime 'moov'/'mdat'/'wide'
    atoms), WebM/MKV (EBML 1A45DFA3; DocType webm vs matroska), AVI (RIFF....AVI ), MPEG-TS/MTS
    (0x47 sync byte every 188 bytes, or 192-byte M2TS packets). Note 'mif1'/'msf1' can be HEIF or AVIF:
    check compatible brands for 'avif'.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(SNIFF_BYTES)
    except OSError:
        return None
    if not data:
        return None
    raw = _raw.sniff_raw(data, filename)
    if raw is not None:
        return raw
    media = _sniff_media(data) if len(data) >= 12 else None
    if media is not None:
        return media
    from app.media import documents

    return documents.sniff_document(data, Path(path), filename)


def _sniff_media(data: bytes) -> tuple[str, str, str] | None:
    if data[:3] == b"\xff\xd8\xff":
        return _JPEG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _PNG
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return _GIF
    if data[:4] == b"RIFF":
        if data[8:12] == b"WEBP":
            return _WEBP
        if data[8:12] == b"AVI ":
            return _AVI
        return None
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return _TIFF
    if data[:2] == b"BM" and len(data) >= 18 and struct.unpack("<I", data[14:18])[0] in _BMP_DIB_SIZES:
        return _BMP
    if data[4:8] == b"ftyp" and len(data) >= 16:
        return _sniff_ftyp(data)
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return _sniff_ebml(data)
    if data[4:8] in _QT_ATOMS:
        return _MOV
    if _is_ts(data):
        return _MTS
    return None


# ------------------------------------------------------------------ GPS
GPS_IFD = 0x8825
MAX_ABS_ALT_M = 100_000.0  # anything beyond this is a broken tag, not a capture position

Gps = tuple[float, float, float | None]


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def valid_gps(lat: float | None, lon: float | None, alt: float | None = None) -> Gps | None:
    """(lat, lon, alt) if lat/lon are finite, in range and not the (0, 0) 'no fix' value; an
    implausible altitude is dropped (None) without discarding the position."""
    lat, lon = _finite(lat), _finite(lon)
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    if lat == 0.0 and lon == 0.0:
        return None
    alt = _finite(alt)
    if alt is not None and abs(alt) > MAX_ABS_ALT_M:
        alt = None
    return round(lat, 7), round(lon, 7), None if alt is None else round(alt, 2)


def _dms(value: Any) -> float | None:
    """EXIF GPS coordinate (degrees, minutes, seconds rationals; some writers store fewer parts
    or one decimal value) -> decimal degrees."""
    if isinstance(value, (tuple, list)):
        parts = [_finite(v) for v in value[:3]]
        if not parts or any(p is None for p in parts):
            return None
        deg = parts[0]
        mins = parts[1] if len(parts) > 1 else 0.0
        secs = parts[2] if len(parts) > 2 else 0.0
        # 60 itself happens when a writer rounds 59.9999; beyond that the value is garbage
        if deg < 0 or mins < 0 or secs < 0 or mins > 60 or secs > 60:
            return None
        return deg + mins / 60.0 + secs / 3600.0
    return _finite(value)


def _ref(value: Any) -> str:
    return (_text(value) or "").upper()[:1]


def exif_gps(gps: Any) -> Gps | None:
    """Position from an EXIF GPS IFD mapping: 1/3 = N|S / E|W refs, 2/4 = lat/lon, 5 = altitude
    ref (1 = below sea level), 6 = altitude. None if absent or invalid."""
    if not gps:
        return None
    try:
        lat, lon = _dms(gps.get(2)), _dms(gps.get(4))
        if lat is None or lon is None:
            return None
        if _ref(gps.get(1)) == "S":
            lat = -abs(lat)
        if _ref(gps.get(3)) == "W":
            lon = -abs(lon)
        alt = _finite(gps.get(6))
        if alt is not None:
            alt_ref = gps.get(5)
            if isinstance(alt_ref, (bytes, bytearray)):
                alt_ref = alt_ref[0] if alt_ref else 0
            try:
                below = int(alt_ref or 0) == 1
            except (TypeError, ValueError):
                below = False
            alt = -abs(alt) if below else abs(alt)
        return valid_gps(lat, lon, alt)
    except Exception:  # malformed IFD contents
        return None


def _exif_gps_of(exif: Any) -> Gps | None:
    try:
        return exif_gps(exif.get_ifd(GPS_IFD))
    except Exception:  # malformed GPS IFD
        return None


_ISO6709 = re.compile(r"^\s*([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?")


def _iso6709_part(text: str, deg_digits: int) -> float | None:
    """One ISO 6709 coordinate: +DD.D, +DDMM.M or +DDMMSS.S (DDD for longitude), told apart by
    the number of integer digits."""
    sign = -1.0 if text[0] == "-" else 1.0
    body = text[1:]
    int_part, _, frac = body.partition(".")
    n = len(int_part)
    try:
        if n <= deg_digits:
            value = float(body)
        elif n == deg_digits + 2:
            value = float(int_part[:deg_digits]) + float(int_part[deg_digits:] + "." + (frac or "0")) / 60.0
        elif n == deg_digits + 4:
            secs = float(int_part[deg_digits + 2:] + "." + (frac or "0"))
            value = float(int_part[:deg_digits]) + float(int_part[deg_digits:deg_digits + 2]) / 60.0 + secs / 3600.0
        else:
            return None
    except ValueError:
        return None
    return sign * value


def parse_iso6709(value: Any) -> Gps | None:
    """'+18.5310+073.8305+560.000/' (also degree-minute forms and a trailing 'CRS...') -> (lat, lon, alt)."""
    text = _text(value)
    if not text:
        return None
    m = _ISO6709.match(text)
    if not m:
        return None
    lat = _iso6709_part(m.group(1), 2)
    lon = _iso6709_part(m.group(2), 3)
    alt = _finite(m.group(3)) if m.group(3) else None
    return valid_gps(lat, lon, alt)


_VIDEO_GPS_TAGS = ("com.apple.quicktime.location.iso6709", "location", "location-eng")


def video_gps(tags: dict) -> Gps | None:
    """Position from ffprobe format/stream tags (ISO 6709 strings)."""
    lowered = {str(k).lower(): v for k, v in (tags or {}).items()}
    for key in _VIDEO_GPS_TAGS:
        gps = parse_iso6709(lowered.get(key))
        if gps is not None:
            return gps
    return None


def read_gps(path: Path) -> Gps | None:
    """Just the capture position of a stored original (for the backfill of items stored before
    positions were read): image headers / EXIF only, or one ffprobe run without a decode check.
    None if the file has none or cannot be read; ToolFailure when ffprobe could not run."""
    kme = sniff(path)
    if kme is None or kme[0] not in ("image", "video") or kme[2] in _raw.RAW_EXTS:
        return None
    if kme[0] == "image":
        _images.register_heif()
        try:
            with Image.open(path) as im:
                return _exif_gps_of(im.getexif())
        except Exception as e:  # undecodable header: nothing to read
            log.info("could not read GPS from %s: %s", path, e)
            return None
    try:
        data = _ffprobe(path)
    except ValueError as e:
        log.info("could not read GPS from %s: %s", path, e)
        return None
    tags: dict = {}
    for stream in data.get("streams") or []:
        tags.update(stream.get("tags") or {})
    tags.update((data.get("format") or {}).get("tags") or {})
    return video_gps(tags)


# ------------------------------------------------------------------ images
_EXIF_DT = re.compile(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
_OFFSET = re.compile(r"^[+-]\d{2}:\d{2}$")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    s = str(value).replace("\x00", "").strip()
    return s or None


_TAKEN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.\d+)?\s*(Z|[+-]\d{2}:?\d{2})?$")


def _event_tz() -> Any:
    try:
        return ZoneInfo(settings.local_timezone or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("unknown LOCAL_TIMEZONE %r, reading naive capture times as UTC", settings.local_timezone)
        return timezone.utc


def parse_taken_at(value: str | None) -> datetime | None:
    """Aware datetime from a stored or probed capture time ('Z', '+05:30' or naive, which is read in
    settings.local_timezone). None if missing or malformed."""
    if not value:
        return None
    m = _TAKEN.match(value.strip())
    if not m:
        return None
    y, mo, d, h, mi, s, tz = m.groups()
    try:
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
    except ValueError:
        return None
    if not tz:
        return dt.replace(tzinfo=_event_tz())
    if tz == "Z":
        return dt.replace(tzinfo=timezone.utc)
    sign = 1 if tz[0] == "+" else -1
    digits = tz[1:].replace(":", "")
    try:
        offset = timedelta(hours=int(digits[:2]), minutes=int(digits[2:]))
        return dt.replace(tzinfo=timezone(sign * offset))
    except ValueError:
        return None


def normalize_taken_at(value: str | None) -> str | None:
    """Capture time as UTC in util.now_iso's format, so it sorts as text against created_at."""
    dt = parse_taken_at(value)
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_exif_datetime(raw: str | None) -> str | None:
    """'YYYY:MM:DD HH:MM:SS' -> 'YYYY-MM-DDTHH:MM:SS', None if malformed or zeroed."""
    if not raw:
        return None
    m = _EXIF_DT.match(raw)
    if not m:
        return None
    try:
        dt = datetime(*(int(x) for x in m.groups()))
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _exif_fields(exif: Any) -> dict[str, Any]:
    """camera, capture time and GPS from a Pillow Exif mapping."""
    make = _text(exif.get(0x010F))
    model = _text(exif.get(0x0110))
    if make and model and model.casefold().startswith(make.split()[0].casefold()):
        make = None  # "Canon" + "Canon EOS R6", "NIKON CORPORATION" + "NIKON D750": the model says it
    camera = " ".join(x for x in (make, model) if x) or None
    try:
        sub = exif.get_ifd(0x8769)
    except Exception:  # malformed EXIF sub-IFD
        sub = {}
    exif_datetime = _text(sub.get(0x9003))
    taken_at = _parse_exif_datetime(exif_datetime)
    if taken_at is None:
        exif_datetime = None
        taken_at = _parse_exif_datetime(_text(sub.get(0x9004)))  # DateTimeDigitized
    if taken_at is None:
        taken_at = _parse_exif_datetime(_text(exif.get(0x0132)))  # DateTime (RAW and edited files)
    subsec = _text(sub.get(0x9291))
    if subsec is not None and not subsec.isdigit():
        subsec = None
    offset = _text(sub.get(0x9011))
    if taken_at and offset and _OFFSET.match(offset):
        taken_at += offset
    gps = _exif_gps_of(exif)
    return dict(
        camera=camera, taken_at=normalize_taken_at(taken_at), exif_datetime=exif_datetime,
        exif_subsec=subsec if exif_datetime else None,
        gps_lat=gps[0] if gps else None, gps_lon=gps[1] if gps else None, gps_alt=gps[2] if gps else None,
    )


def probe_raw(path: Path, kind_mime_ext: tuple[str, str, str]) -> MediaInfo:
    """Camera RAW through LibRaw (dimensions with the camera's flip applied) plus EXIF when the
    container or its embedded preview carries it. ValueError if LibRaw cannot read the file."""
    kind, mime, ext = kind_mime_ext
    width, height, exif = _raw.probe_raw(path, kind_mime_ext)
    if width * height > _images.MAX_PIXELS:
        raise ValueError(f"image too large ({width}x{height})")
    return MediaInfo(kind="image", mime=mime, ext=ext, width=width, height=height, raw=True, **_exif_fields(exif))


def probe_image(path: Path, kind_mime_ext: tuple[str, str, str]) -> MediaInfo:
    """Decode headers with Pillow (+ pillow-heif) and read EXIF. Raises ValueError if undecodable.

    width/height are AFTER applying EXIF orientation (orientations 5-8 swap them).
    """
    _images.register_heif()
    kind, mime, ext = kind_mime_ext
    if ext in _raw.RAW_EXTS:
        return probe_raw(path, kind_mime_ext)
    try:
        with Image.open(path) as im:
            width, height = im.size
            if width <= 0 or height <= 0:
                raise ValueError("image has no pixels")
            if width * height > _images.MAX_PIXELS:
                raise ValueError(f"image too large ({width}x{height})")
            exif = im.getexif()
            animated = bool(getattr(im, "is_animated", False)) and int(getattr(im, "n_frames", 1)) > 1
            # Make sure pixel data actually decodes (cheaply for JPEG via draft mode).
            if im.format == "JPEG":
                im.draft("RGB", (max(1, width // 8), max(1, height // 8)))
            im.load()
    except ValueError:
        raise
    except Exception as e:  # OSError, SyntaxError, DecompressionBombError, decoder errors
        raise ValueError(f"cannot decode image: {e}") from e

    orientation = exif.get(0x0112, 1)
    try:
        orientation = int(orientation)
    except (TypeError, ValueError):
        orientation = 1
    if orientation not in range(1, 9):
        orientation = 1
    if orientation >= 5:
        width, height = height, width

    from app.media import live

    return MediaInfo(
        kind="image", mime=mime, ext=ext, width=width, height=height, orientation=orientation,
        animated=animated, content_id=live.content_identifier(exif), **_exif_fields(exif),
    )


# ------------------------------------------------------------------ videos
def _float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _rate(value: Any) -> float | None:
    if not value or not isinstance(value, str):
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            n, d = float(num), float(den)
            return n / d if d else None
        return float(value)
    except ValueError:
        return None


def _rotation(stream: dict) -> int:
    """Clockwise display rotation (0/90/180/270). ffprobe's displaymatrix rotation is
    counter-clockwise (-90 means 'turn 90 degrees clockwise'); the legacy rotate tag is clockwise."""
    deg: float | None = None
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            r = _float(sd.get("rotation"))
            if r is not None:
                deg = -r
                break
    if deg is None:
        r = _float((stream.get("tags") or {}).get("rotate"))
        if r is not None:
            deg = r
    if deg is None:
        return 0
    return int(round(deg / 90.0)) * 90 % 360


_ISO_TS = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.\d+)?\s*(Z|[+-]\d{2}:?\d{2})?$")


def _video_time(tags: dict) -> str | None:
    """Capture time (UTC, see normalize_taken_at) from QuickTime/MP4 creation tags (Apple's
    local-time tag preferred)."""
    lowered = {str(k).lower(): v for k, v in tags.items()}
    for key in ("com.apple.quicktime.creationdate", "creation_time", "date"):
        value = _text(lowered.get(key))
        if not value:
            continue
        m = _ISO_TS.match(value)
        if not m:
            continue
        y, mo, d, h, mi, s, tz = m.groups()
        try:
            dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
        except ValueError:
            continue
        if dt.year < 1980:  # zeroed container dates (1904/1970 epochs)
            continue
        out = dt.strftime("%Y-%m-%dT%H:%M:%S")
        if tz:
            out += "Z" if tz == "Z" else (tz if ":" in tz else f"{tz[:3]}:{tz[3:]}")
        return normalize_taken_at(out)
    return None


def _ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", *input_flags(path), "-print_format", "json", "-show_format", "-show_streams",
           str(path)]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=FFPROBE_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        raise ToolFailure("ffprobe timed out") from e
    except OSError as e:
        raise ToolFailure(f"could not run ffprobe: {e}") from e
    if res.returncode < 0:
        raise ToolFailure(f"ffprobe was killed by signal {-res.returncode}")
    if res.returncode != 0:
        trouble = tool_trouble(res.stderr)
        if trouble:
            raise ToolFailure(f"ffprobe could not run properly ({trouble})")
        raise ValueError(f"ffprobe failed: {res.stderr.decode('utf-8', 'replace').strip()[:300]}")
    try:
        return json.loads(res.stdout or b"{}")
    except json.JSONDecodeError as e:
        raise ValueError("ffprobe returned invalid JSON") from e


def _decodes_one_frame(path: Path) -> bool:
    """True if ffmpeg decodes at least one frame of the first real video stream.
    ToolFailure if ffmpeg itself could not run to completion."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", *input_flags(path), "-i", str(path), "-map", "0:V:0",
           "-frames:v", "1", "-f", "framemd5", "-"]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=DECODE_CHECK_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        raise ToolFailure("ffmpeg decode check timed out") from e
    except OSError as e:
        raise ToolFailure(f"could not run ffmpeg: {e}") from e
    if res.returncode < 0:
        raise ToolFailure(f"ffmpeg was killed by signal {-res.returncode}")
    if res.returncode != 0:
        trouble = tool_trouble(res.stderr)
        if trouble:
            raise ToolFailure(f"ffmpeg could not run properly ({trouble})")
        return False
    return any(line and not line.startswith(b"#") for line in res.stdout.splitlines())


def probe_video(path: Path, kind_mime_ext: tuple[str, str, str]) -> MediaInfo:
    """Read metadata with `ffprobe -v error -print_format json -show_format -show_streams`.

    Rotation from the video stream's side_data_list displaymatrix 'rotation' or tags.rotate;
    width/height swapped for +-90/270. Raises ValueError if there is no decodable video stream.
    """
    kind, mime, ext = kind_mime_ext
    data = _ffprobe(path)
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    video = next(
        (s for s in streams
         if s.get("codec_type") == "video"
         and not (s.get("disposition") or {}).get("attached_pic")
         and int(s.get("width") or 0) > 0 and int(s.get("height") or 0) > 0),
        None,
    )
    if video is None:
        raise ValueError("no video stream")
    if int(video["width"]) * int(video["height"]) > MAX_VIDEO_PIXELS:
        raise ValueError(f"video too large ({video['width']}x{video['height']})")
    # avg_frame_rate only: r_frame_rate of variable-rate phone clips is often a timebase like 1000/1.
    rate = _rate(video.get("avg_frame_rate"))
    if rate is not None and rate > MAX_VIDEO_FPS:
        raise ValueError(f"video frame rate too high ({rate:g} fps)")
    if not _decodes_one_frame(path):
        raise ValueError("video stream does not decode")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width, height = int(video["width"]), int(video["height"])
    rotation = _rotation(video)
    if rotation in (90, 270):
        width, height = height, width

    duration = _float(fmt.get("duration")) or _float(video.get("duration"))
    if duration is not None and duration <= 0:
        duration = None
    bitrate = _float(fmt.get("bit_rate"))
    if not bitrate and duration:
        size = _float(fmt.get("size"))
        if size is None:
            try:
                size = float(Path(path).stat().st_size)
            except OSError:
                size = None
        bitrate = size * 8 / duration if size else None
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    if fps is not None and not (0 < fps <= 1000):
        fps = None

    tags = {**(video.get("tags") or {}), **(fmt.get("tags") or {})}
    gps_tags: dict = {}
    for stream in streams:
        gps_tags.update(stream.get("tags") or {})
    gps_tags.update(fmt.get("tags") or {})
    gps = video_gps(gps_tags)
    return MediaInfo(
        kind="video", mime=mime, ext=ext, width=width, height=height,
        duration_s=round(duration, 3) if duration else None,
        bitrate=int(bitrate) if bitrate else None,
        video_codec=video.get("codec_name"),
        audio_codec=audio.get("codec_name") if audio else None,
        fps=round(fps, 3) if fps else None,
        has_audio=audio is not None,
        rotation=rotation,
        taken_at=_video_time(tags),
        gps_lat=gps[0] if gps else None, gps_lon=gps[1] if gps else None, gps_alt=gps[2] if gps else None,
        content_id=_live_content_id(tags),
        strict_input=bool(input_flags(path)),
    )


def _live_content_id(tags: dict) -> str | None:
    from app.media import live

    return live.video_content_identifier(tags)


def probe_document(path: Path, kind_mime_ext: tuple[str, str, str]) -> MediaInfo:
    """PDF / office / plain text: page count and page 1 render size. ValueError if unreadable."""
    from app.media import documents

    kind, mime, ext = kind_mime_ext
    pages, width, height = documents.probe_document(path, kind, ext)
    return MediaInfo(kind=kind, mime=mime, ext=ext, width=width, height=height, page_count=pages)  # type: ignore[arg-type]


def other_ext(filename: str | None) -> str:
    """Storage extension for an 'other' file: the uploaded name's suffix when it is short and plain
    (letters and digits), else '.bin'."""
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if "." in name:
        ext = "." + name.rsplit(".", 1)[-1].lower()
        if _SAFE_EXT.match(ext):
            return ext
    return ".bin"


def other_info(filename: str | None, kme: tuple[str, str, str] | None = None, reason: str | None = None) -> MediaInfo:
    """MediaInfo for a file kept without previews. A file that sniffed as a known type keeps that
    type's MIME and extension (it is what the bytes claim to be); unknown files are octet-stream."""
    if kme is not None:
        return MediaInfo(kind="other", mime=kme[1], ext=kme[2], width=None, height=None, reason=reason)
    return MediaInfo(kind="other", mime=OTHER_MIME, ext=other_ext(filename), width=None, height=None, reason=reason)


def probe(path: Path, filename: str | None = None) -> MediaInfo | None:
    """sniff + the matching probe. None only for an empty or unreadable file. A file of a known
    type that does not decode, and any file of an unknown type, comes back as kind 'other'.
    Raises ToolFailure when ffprobe/ffmpeg could not run: that is no verdict on the file."""
    try:
        if Path(path).stat().st_size <= 0:
            return None
    except OSError:
        return None
    kme = sniff(path, filename)
    if kme is None:
        return other_info(filename)
    try:
        if kme[0] == "image":
            return probe_image(path, kme)
        if kme[0] == "video":
            return probe_video(path, kme)
        return probe_document(path, kme)
    except ToolFailure:
        raise
    except Exception as e:  # a damaged file is still kept, just without previews
        log.info("probe: %s sniffed as %s but did not decode (%s); keeping it as 'other'", path, kme[1], e)
        return other_info(filename, kme, reason=str(e)[:200])
