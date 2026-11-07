"""File serving.

CONTRACT (implemented by the API agent). Router mounted at prefix /files.
Use starlette FileResponse (supports Range / 206, which Safari needs for video).

GET /files/thumb/{uid}?v=     WebP thumbnail of the current version
GET /files/preview/{uid}?v=   WebP large preview / video poster
GET /files/preview-small/{uid}?v=   the same at settings.preview_small_size; items stored before
                              small previews existed get theirs made from the large preview on
                              first request (per-uid lock, atomic write)
GET /files/original/{uid}?v=&download=1   original bytes with media.mime; download=1 adds
                              Content-Disposition attachment with the original file name (RFC 5987),
                              made safe by util.media_file_name (its suffix matches the stored file)
GET /files/video/{uid}?v=     the compressed H.264 copy if preview_status='done', else the original
GET /files/geotag/{uid}?v=&pos=tl|tc|tr|bl|bc|br&size=preview|full&download=1
                              JPEG of a photo with the geotag card (small map, coordinates, place,
                              capture time) drawn at pos. size=preview: on the small preview, for the
                              viewer's live preview; size=full (default): on the full-resolution
                              original, keeping its EXIF (Orientation 1). 404 unless a photo with a
                              position and settings.geotag_enabled. Rendered once per (version, pos,
                              size) and cached under DATA_DIR/geotag; at most
                              settings.geotag_max_concurrent renders at a time, 503 + Retry-After
                              when no slot frees up within GEOTAG_WAIT_S. download=1 names the
                              attachment "<original stem>-geotag.jpg". Photos only.
GET /files/frame/{uid}?t=&pos=none|tl|tc|tr|bl|bc|br&size=full|preview&v=&download=1
                              JPEG of one video frame: the first frame at or after t (seconds on the
                              file's timeline, as the browser's video currentTime / frame mediaTime,
                              finite, rounded DOWN to the millisecond, clamped to [0, duration minus a
                              frame and a half]). size=full (default): from the original at its display
                              resolution, quality settings.geotag_jpeg_quality, with EXIF (Orientation 1,
                              DateTimeOriginal = taken_at in settings.event_timezone, GPS when the video
                              has a position); size=preview: long side <= FRAME_PREVIEW_SIDE, from the
                              streaming copy when there is one (same timestamps), quality
                              FRAME_PREVIEW_QUALITY, no EXIF, the extracted frame kept in a small
                              in-memory LRU so moving the card does not run ffmpeg again. pos other than
                              none draws the geotag card (404 unless the video has a position and
                              settings.geotag_enabled). 404 for photos. At most FRAME_MAX_CONCURRENT
                              extractions at once, 503 + Retry-After when no slot frees up within
                              GEOTAG_WAIT_S. Nothing is written to disk. download=1 names the attachment
                              "<stem>-<m>m<ss.s>s.jpg" ("...-geotag.jpg" with the card).
GET /files/geotag-video/{uid}?pos=&v=&download=1
                              the finished MP4 of a geotag video render (app.media.geotag_video,
                              started by POST /api/media/{uid}/geotag-video), 404 when it was not made
                              or has expired; every GET counts as a download start for the expiry.
                              download=1 names the attachment "<stem>-geotag.mp4".
GET /files/face/{face_id}     face crop WebP
GET /files/superseded/{id}/thumb, /files/superseded/{id}/original   ADMIN ONLY (app.auth.is_admin)

Rules:
- Look up media by uid; 404 if missing. Hidden media: 404 unless admin. Public routes need
  app.auth.has_access (401 otherwise). Face crops of faces on hidden media: 404 unless admin.
- Path comes from app.storage using the CURRENT version and ext (never from user input).
- Cache-Control: if query v == current version -> "public, max-age=31536000, immutable";
  otherwise "no-cache". Face crops: "public, max-age=86400".
- Every response: X-Content-Type-Options: nosniff, and for originals
  Content-Security-Policy: "default-src 'none'; sandbox" so nothing served here can run script.
"""
from __future__ import annotations

import io
import logging
import math
import os
import sqlite3
import threading
import time
import unicodedata
import uuid
import zlib
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response
from PIL import Image

from app import storage
from app.api.common import geotag_available, place_label, video_geotag_available
from app.api.deps import get_db
from app.auth import has_access, is_admin
from app.config import settings
from app.media import geotag_video, images
from app.media import video as video_media
from app.media.probe import ToolFailure
from app.schemas import FramePosition, GeotagPosition, GeotagSize
from app.util import media_file_name, safe_filename

log = logging.getLogger(__name__)

router = APIRouter()

IMMUTABLE = "public, max-age=31536000, immutable"
NO_CACHE = "no-cache"
PRIVATE = "private, no-cache"
FACE_CACHE = "public, max-age=86400"
CSP = "default-src 'none'; sandbox"
METHODS = ["GET", "HEAD"]

# Striped locks for making missing small previews: two requests for one item never both decode
# the large preview, and the lock table does not grow with the number of items.
_SMALL_LOCKS = tuple(threading.Lock() for _ in range(64))

# Geotag renders: one render per (item, pos, size) at a time (striped locks), and at most
# settings.geotag_max_concurrent renders overall (a full-resolution render holds the decoded photo).
GEOTAG_WAIT_S = 20.0  # longest a request waits for a render slot before it gets a 503
GEOTAG_RETRY_AFTER_S = 10
_GEOTAG_LOCKS = tuple(threading.Lock() for _ in range(64))
_geotag_sem: tuple[int, threading.BoundedSemaphore] | None = None
_geotag_sem_guard = threading.Lock()

# Video frames: at most FRAME_MAX_CONCURRENT ffmpeg extractions at once (a full 4K frame is about
# 25 MB decoded). Preview-size frames are kept in a small LRU keyed (uid, version, source, ms).
FRAME_MAX_CONCURRENT = 2
FRAME_WAIT_S = 1.0  # a frame request waits this long for a slot, then gets 503: request threads are shared
# Containers whose input seek can land after the frame asked for (MPEG-TS from camcorders).
EXACT_SEEK_EXTS = frozenset({".mts", ".m2ts", ".ts"})
FRAME_PREVIEW_SIDE = 1280
FRAME_PREVIEW_QUALITY = 82
FRAME_CACHE_ENTRIES = 8
_frame_sem = threading.BoundedSemaphore(FRAME_MAX_CONCURRENT)
_frame_cache: OrderedDict[tuple[str, int, str, int], Image.Image] = OrderedDict()
_frame_cache_lock = threading.Lock()


def _check_access(request: Request) -> None:
    if not has_access(request):
        raise HTTPException(status_code=401, detail="access_key_required")


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="File not found")


def _media(conn: sqlite3.Connection, uid: str, request: Request) -> sqlite3.Row:
    _check_access(request)
    row = conn.execute(
        "SELECT id, uid, kind, version, ext, mime, original_filename, hidden, preview_status, "
        "gps_lat, gps_lon, gps_alt, taken_at, duration_s, fps, camera FROM media WHERE uid = ?",
        (uid,),
    ).fetchone()
    if row is None or (row["hidden"] and not is_admin(request)):
        raise _not_found()
    return row


def _cache(row: sqlite3.Row, v: int | None) -> str:
    if row["hidden"]:
        return PRIVATE  # only admins get here; keep it out of shared caches
    return IMMUTABLE if v is not None and v == row["version"] else NO_CACHE


def _send(path: Path, media_type: str, cache: str, extra: dict[str, str] | None = None) -> FileResponse:
    if not path.is_file():
        raise _not_found()
    headers = {"cache-control": cache, "x-content-type-options": "nosniff", "content-security-policy": CSP}
    if extra:
        headers.update(extra)
    return FileResponse(path, media_type=media_type, headers=headers)


def _ascii_fallback(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(c if 32 <= ord(c) < 127 and c not in '"\\;' else "_" for c in ascii_name).strip()
    stem = cleaned.rsplit(".", 1)[0].strip("_ ")
    return cleaned if stem else "file" + Path(name).suffix.encode("ascii", "ignore").decode("ascii")


def content_disposition(filename: str, ext: str = "") -> str:
    """attachment; filename="<ascii>"; filename*=UTF-8''<percent-encoded> (RFC 6266 / 5987)."""
    name = safe_filename(filename)
    if ext and not Path(name).suffix:
        name += ext
    return f"attachment; filename=\"{_ascii_fallback(name)}\"; filename*=UTF-8''{quote(name, safe='')}"


# --------------------------------------------------------------------- media
@router.api_route("/thumb/{uid}", methods=METHODS)
def thumb(uid: str, request: Request, v: int | None = None, conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _media(conn, uid, request)
    return _send(storage.thumb_path(row["uid"], row["version"]), "image/webp", _cache(row, v))


@router.api_route("/preview/{uid}", methods=METHODS)
def preview(uid: str, request: Request, v: int | None = None, conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _media(conn, uid, request)
    return _send(storage.preview_path(row["uid"], row["version"]), "image/webp", _cache(row, v))


def _ensure_small_preview(uid: str, version: int) -> bool:
    """Make the small preview from the large one if it is missing. True when the small file exists
    afterwards; False when it could not be made (no large preview, decode or disk error)."""
    small = storage.preview_small_path(uid, version)
    if small.is_file():
        return True
    with _SMALL_LOCKS[zlib.crc32(uid.encode()) % len(_SMALL_LOCKS)]:
        if small.is_file():
            return True
        large = storage.preview_path(uid, version)
        if not large.is_file():
            return False
        try:
            with Image.open(large) as im:
                im.load()
                img = im.convert("RGB")
            images.save_webp(img, small, settings.preview_small_size, settings.preview_quality)
        except Exception as e:  # corrupt preview, full disk: serve the large one instead
            log.warning("could not make the small preview of %s v%s: %s", uid, version, e)
            return False
    log.info("made the missing small preview of %s v%s", uid, version)
    return True


@router.api_route("/preview-small/{uid}", methods=METHODS)
def preview_small(uid: str, request: Request, v: int | None = None,
                  conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _media(conn, uid, request)
    if _ensure_small_preview(row["uid"], row["version"]):
        return _send(storage.preview_small_path(row["uid"], row["version"]), "image/webp", _cache(row, v))
    # Stopgap (no small file could be made): never cache the large file under the small URL.
    cache = PRIVATE if row["hidden"] else NO_CACHE
    return _send(storage.preview_path(row["uid"], row["version"]), "image/webp", cache)


@router.api_route("/original/{uid}", methods=METHODS)
def original(uid: str, request: Request, v: int | None = None, download: int = 0,
             conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _media(conn, uid, request)
    name = media_file_name(row["original_filename"], row["ext"])
    extra = {"content-disposition": content_disposition(name)} if download else None
    return _send(storage.original_path(row["uid"], row["version"], row["ext"]), row["mime"], _cache(row, v), extra)


@router.api_route("/video/{uid}", methods=METHODS)
def video(uid: str, request: Request, v: int | None = None, conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _media(conn, uid, request)
    if row["kind"] != "video":
        raise _not_found()
    if row["preview_status"] == "done":
        web = storage.video_preview_path(row["uid"], row["version"])
        if web.is_file():
            return _send(web, "video/mp4", _cache(row, v))
    return _send(storage.original_path(row["uid"], row["version"], row["ext"]), row["mime"], _cache(row, v))


# -------------------------------------------------------------------- geotag
def _geotag_semaphore() -> threading.BoundedSemaphore:
    """The render semaphore, rebuilt when settings.geotag_max_concurrent changes (callers release
    the object they acquired, so a rebuild never unbalances a semaphore)."""
    global _geotag_sem
    n = max(1, int(settings.geotag_max_concurrent))
    with _geotag_sem_guard:
        if _geotag_sem is None or _geotag_sem[0] != n:
            _geotag_sem = (n, threading.BoundedSemaphore(n))
        return _geotag_sem[1]


def _busy() -> HTTPException:
    return HTTPException(status_code=503, detail="Busy making other geotagged copies, please try again shortly",
                         headers={"Retry-After": str(GEOTAG_RETRY_AFTER_S)})


def _original_exif(original: Path) -> tuple[bytes | None, bytes | None]:
    """(EXIF with Orientation set to 1, ICC profile) of the original; (None, None) if unreadable."""
    try:
        with Image.open(original) as im:
            exif = im.getexif()
            icc = im.info.get("icc_profile")
            if not exif:
                return None, icc
            exif[0x0112] = 1  # the rendered pixels are already upright
            return exif.tobytes(), icc
    except Exception as e:  # odd EXIF must not stop the download: it is written without
        log.warning("could not read the EXIF of %s: %s", original.name, e)
        return None, None


def _geotag_base_image(row: sqlite3.Row, size: str) -> Image.Image:
    """Upright RGB photo to draw on: the full-resolution original, or the small preview (falling
    back to the large one)."""
    uid, version = row["uid"], row["version"]
    if size == "full":
        original = storage.original_path(uid, version, row["ext"])
        if not original.is_file():
            raise _not_found()
        try:
            return images.load_oriented(original)
        except ValueError as e:
            log.warning("geotag: cannot decode %s v%s: %s", uid, version, e)
            raise _not_found()
    src = storage.preview_small_path(uid, version) if _ensure_small_preview(uid, version) else None
    if src is None or not src.is_file():
        src = storage.preview_path(uid, version)
    if not src.is_file():
        raise _not_found()
    try:
        with Image.open(src) as im:
            im.load()
            return im.convert("RGB")
    except Exception as e:
        log.warning("geotag: cannot decode the preview of %s v%s: %s", uid, version, e)
        raise _not_found()


def _geotag_jpeg(row: sqlite3.Row, pos: str, size: str) -> bytes:
    from app.geo import overlay

    img = _geotag_base_image(row, size)
    lat, lon = float(row["gps_lat"]), float(row["gps_lon"])
    out = overlay.render_geotag(img, lat=lat, lon=lon, alt=row["gps_alt"], place=place_label(lat, lon),
                                captured_at=row["taken_at"], position=pos)
    del img
    if out.mode != "RGB":
        out = out.convert("RGB")
    exif, icc = _original_exif(storage.original_path(row["uid"], row["version"], row["ext"]))
    opts: dict = {"quality": int(settings.geotag_jpeg_quality)}
    if icc and size == "full":  # the previews were saved without a profile; the original's pixels need it
        opts["icc_profile"] = icc
    buf = io.BytesIO()
    try:
        out.save(buf, "JPEG", **({**opts, "exif": exif} if exif else opts))
    except (ValueError, OSError) as e:  # EXIF too large for one APP1 segment (big MakerNotes), bad ICC
        log.info("geotag: saving %s without EXIF/ICC (%s)", row["uid"], e)
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=int(settings.geotag_jpeg_quality))
    return buf.getvalue()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _ensure_geotag(row: sqlite3.Row, pos: str, size: str, dest: Path) -> bytes | None:
    """Render into the cache file dest unless it exists. Returns the JPEG bytes only when they
    could not be cached (full disk), so the caller serves them from memory."""
    deadline = time.monotonic() + GEOTAG_WAIT_S
    stripe = _GEOTAG_LOCKS[zlib.crc32(f"{row['uid']}.{pos}.{size}".encode()) % len(_GEOTAG_LOCKS)]
    if not stripe.acquire(timeout=GEOTAG_WAIT_S):
        raise _busy()
    try:
        if dest.is_file():
            return None  # made by the request we waited for
        sem = _geotag_semaphore()
        if not sem.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise _busy()
        try:
            started = time.monotonic()
            data = _geotag_jpeg(row, pos, size)
        finally:
            sem.release()
        log.info("geotag: rendered %s v%s %s %s in %.1fs (%d KB)", row["uid"], row["version"], pos, size,
                 time.monotonic() - started, len(data) // 1024)
        try:
            _write_atomic(dest, data)
        except OSError as e:
            log.warning("geotag: could not cache %s: %s", dest.name, e)
            return data
        return None
    finally:
        stripe.release()


@router.api_route("/geotag/{uid}", methods=METHODS)
def geotag(uid: str, request: Request, v: int | None = None, pos: GeotagPosition = "br",
           size: GeotagSize = "full", download: int = 0, conn: sqlite3.Connection = Depends(get_db)) -> Response:
    row = _media(conn, uid, request)
    # Same rule as MediaDetail.geotag_available, except that admins may also fetch hidden items.
    if row["kind"] != "image" or not geotag_available({**dict(row), "hidden": 0}):
        raise _not_found()
    cache = _cache(row, v)
    extra: dict[str, str] = {}
    if download:
        stem = Path(safe_filename(row["original_filename"], "photo")).stem or "photo"
        extra["content-disposition"] = content_disposition(f"{stem}-geotag.jpg")
    dest = storage.geotag_path(row["uid"], row["version"], pos, size)
    if not dest.is_file():
        data = _ensure_geotag(row, pos, size, dest)
        if data is not None:
            headers = {"cache-control": PRIVATE if row["hidden"] else NO_CACHE,
                       "x-content-type-options": "nosniff", "content-security-policy": CSP, **extra}
            return Response(content=data, media_type="image/jpeg", headers=headers)
    return _send(dest, "image/jpeg", cache, extra)


# -------------------------------------------------------------- video frames
def _frame_ms(row: sqlite3.Row, t: float) -> int:
    """t rounded DOWN to the millisecond (ffmpeg's input seek then returns the first frame at or
    after it, which is the frame shown at t when t is that frame's presentation time), within the
    clip. Past the last frame nothing decodes; _extract then steps back. The time is rounded to
    a microsecond first: 8.033 * 1000 is 8032.999..., which would otherwise lose a millisecond."""
    t = min(max(0.0, t), 1e7)
    if row["duration_s"]:
        t = min(t, float(row["duration_s"]))
    return max(0, math.floor(round(t * 1000, 3)))


def _back_off_ms(row: sqlite3.Row) -> int:
    """How far _extract steps back when nothing decodes at t (media.video.frames_at's margin)."""
    fps = row["fps"]
    return math.ceil(1000 * (max(0.1, 1.5 / fps) if fps else 0.1))


def _shown_ms(path: Path, ms: int) -> int:
    """ms moved back to the start of the frame on screen at that time (a player's time usually sits
    inside a frame, and ffmpeg's seek returns the first frame at or after the time it gets)."""
    t = video_media.shown_frame_time(path, ms / 1000.0)
    return ms if t is None else min(ms, max(0, math.floor(t * 1000)))


def _frame_source(row: sqlite3.Row, size: str) -> tuple[str, Path]:
    """("original" | "stream", file) to take the frame from: preview frames come from the streaming
    copy when it exists (same timestamps, much cheaper to decode), full frames from the original."""
    uid, version = row["uid"], row["version"]
    if size == "preview" and row["preview_status"] == "done":
        web = storage.video_preview_path(uid, version)
        if web.is_file():
            return "stream", web
    original = storage.original_path(uid, version, row["ext"])
    if not original.is_file():
        raise _not_found()
    return "original", original


def _extract(path: Path, ms: int, max_side: int | None, back_off_ms: int, exact_seek: bool) -> Image.Image:
    """The frame shown at ms (see _shown_ms), holding a frame slot for every ffprobe and ffmpeg run;
    no slot within FRAME_WAIT_S gives 503 so a burst cannot tie up the request threads."""
    if not _frame_sem.acquire(timeout=FRAME_WAIT_S):
        raise _busy()
    try:
        ms = _shown_ms(path, ms)

        def grab(at_ms: int) -> Image.Image | None:
            return video_media.extract_frame(path, at_ms / 1000.0, max_side, exact_seek=exact_seek, gentle=True)

        frame = grab(ms)
        # Nothing decodes at or after the last frame's time: step back, like media.video.frames_at.
        for back in (back_off_ms, 1000):
            if frame is not None or ms == 0:
                break
            frame = grab(max(0, ms - back))
    except ToolFailure as e:
        log.warning("frame: ffmpeg failed on %s at %d ms: %s", path.name, ms, e)
        raise _busy()
    finally:
        _frame_sem.release()
    if frame is None:
        raise _not_found()
    return frame


def _frame_image(row: sqlite3.Row, ms: int, size: str) -> Image.Image:
    """The upright RGB frame (shared with the LRU for previews: callers must not modify it)."""
    source, path = _frame_source(row, size)
    exact = source == "original" and (row["ext"] or "").lower() in EXACT_SEEK_EXTS
    if size != "preview":
        return _extract(path, ms, None, _back_off_ms(row), exact)
    key = (row["uid"], int(row["version"]), source, ms)
    with _frame_cache_lock:
        hit = _frame_cache.get(key)
        if hit is not None:
            _frame_cache.move_to_end(key)
            return hit
    frame = _extract(path, ms, FRAME_PREVIEW_SIDE, _back_off_ms(row), exact)
    with _frame_cache_lock:
        _frame_cache[key] = frame
        _frame_cache.move_to_end(key)
        while len(_frame_cache) > FRAME_CACHE_ENTRIES:
            _frame_cache.popitem(last=False)
    return frame


def _dms(value: float) -> tuple[float, float, float]:
    value = abs(float(value))
    d = int(value)
    m = int((value - d) * 60)
    return float(d), float(m), round((value - d - m / 60) * 3600, 4)


def _frame_exif(row: sqlite3.Row) -> bytes:
    """EXIF for a full frame: Orientation 1 (the pixels are upright), the capture time and, when
    known, camera and GPS position. The capture time is the item's taken_at: a container's creation
    time is the start of the recording on some phones and its end on others, so adding the frame's
    offset would be wrong as often as right."""
    exif = Image.Exif()
    exif[0x0112] = 1
    camera = (row["camera"] or "").strip()
    if camera:
        exif[0x010F] = camera.split(" ", 1)[0]  # Make
        exif[0x0110] = camera  # Model
    taken = row["taken_at"]
    if taken:
        try:
            dt = datetime.fromisoformat(str(taken).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            try:
                local = dt.astimezone(ZoneInfo(settings.event_timezone or "UTC"))
            except Exception:  # unknown zone name: write UTC rather than nothing
                local = dt.astimezone(timezone.utc)
            sub = exif.get_ifd(0x8769)
            sub[0x9003] = local.strftime("%Y:%m:%d %H:%M:%S")  # DateTimeOriginal
            off = local.utcoffset()
            if off is not None:
                mins = int(off.total_seconds() // 60)
                sign = "+" if mins >= 0 else "-"
                sub[0x9011] = f"{sign}{abs(mins) // 60:02d}:{abs(mins) % 60:02d}"  # OffsetTimeOriginal
        except ValueError:
            log.info("frame: unreadable taken_at %r", taken)
    if row["gps_lat"] is not None and row["gps_lon"] is not None:
        lat, lon = float(row["gps_lat"]), float(row["gps_lon"])
        gps = exif.get_ifd(0x8825)
        gps[0] = b"\x02\x03\x00\x00"  # GPSVersionID 2.3
        gps[1] = "S" if lat < 0 else "N"
        gps[2] = _dms(lat)
        gps[3] = "W" if lon < 0 else "E"
        gps[4] = _dms(lon)
        if row["gps_alt"] is not None:
            alt = float(row["gps_alt"])
            gps[5] = b"\x01" if alt < 0 else b"\x00"
            gps[6] = abs(alt)
    return exif.tobytes()


def frame_file_name(row: sqlite3.Row, ms: int, card: bool) -> str:
    """"IMG_6915-1m05.4s.jpg" / "IMG_6915-1m05.4s-geotag.jpg" (time in whole tenths, rounded down)."""
    stem = Path(media_file_name(row["original_filename"], row["ext"])).stem or "video"
    tenths = ms // 100
    stamp = f"{tenths // 600}m{(tenths % 600) // 10:02d}.{tenths % 10}s"
    return f"{stem}-{stamp}{'-geotag' if card else ''}.jpg"


@router.api_route("/frame/{uid}", methods=METHODS)
def frame(uid: str, request: Request, t: float, v: int | None = None, pos: FramePosition = "none",
          size: GeotagSize = "full", download: int = 0, conn: sqlite3.Connection = Depends(get_db)) -> Response:
    if not math.isfinite(t):
        raise HTTPException(status_code=422, detail="t must be a finite number of seconds")
    row = _media(conn, uid, request)
    if row["kind"] != "video":
        raise _not_found()
    card = pos != "none"
    # Same rule as MediaDetail.video_geotag_available, except that admins may also fetch hidden items.
    if card and not video_geotag_available({**dict(row), "hidden": 0}):
        raise _not_found()
    ms = _frame_ms(row, t)
    img = _frame_image(row, ms, size)
    if card:
        from app.geo import overlay

        lat, lon = float(row["gps_lat"]), float(row["gps_lon"])
        img = overlay.render_geotag(img, lat=lat, lon=lon, alt=row["gps_alt"], place=place_label(lat, lon),
                                    captured_at=row["taken_at"], position=pos)
    buf = io.BytesIO()
    if size == "full":
        img.save(buf, "JPEG", quality=int(settings.geotag_jpeg_quality), exif=_frame_exif(row))
    else:
        img.save(buf, "JPEG", quality=FRAME_PREVIEW_QUALITY)
    del img
    headers = {"cache-control": _cache(row, v), "x-content-type-options": "nosniff", "content-security-policy": CSP}
    if download:
        headers["content-disposition"] = content_disposition(frame_file_name(row, ms, card))
    return Response(content=buf.getvalue(), media_type="image/jpeg", headers=headers)


# ------------------------------------------------------------- geotag video
@router.api_route("/geotag-video/{uid}", methods=METHODS)
def geotag_video_file(uid: str, request: Request, v: int | None = None, pos: GeotagPosition = "br",
                      download: int = 0, conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _media(conn, uid, request)
    if not video_geotag_available({**dict(row), "hidden": 0}):
        raise _not_found()
    # HEAD (a size check) does not count as a download start.
    path = (geotag_video.finished_file(row["uid"], row["version"], pos) if request.method == "GET"
            else _peek_geotag_video(row, pos))
    if path is None:
        raise _not_found()
    extra: dict[str, str] = {}
    if download:
        stem = Path(media_file_name(row["original_filename"], row["ext"])).stem or "video"
        extra["content-disposition"] = content_disposition(f"{stem}-geotag.mp4")
    # A short-lived file (removed after geotag_video.KEEP_S): never kept in shared caches.
    return _send(path, "video/mp4", PRIVATE, extra)


def _peek_geotag_video(row: sqlite3.Row, pos: str) -> Path | None:
    st = geotag_video.status(row["uid"], row["version"], pos)
    if st is None or st.state != "done":
        return None
    path = geotag_video.output_path(row["uid"], row["version"], pos)
    return path if path.is_file() else None


@router.api_route("/face/{face_id}", methods=METHODS)
def face(face_id: int, request: Request, conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    _check_access(request)
    row = conn.execute(
        "SELECT f.id, m.hidden FROM faces f JOIN media m ON m.id = f.media_id WHERE f.id = ?", (face_id,)
    ).fetchone()
    if row is None or (row["hidden"] and not is_admin(request)):
        raise _not_found()
    return _send(storage.face_crop_path(row["id"]), "image/webp", PRIVATE if row["hidden"] else FACE_CACHE)


# ---------------------------------------------------------------- superseded
def _superseded(conn: sqlite3.Connection, superseded_id: int, request: Request) -> sqlite3.Row:
    if not is_admin(request):
        raise HTTPException(status_code=401, detail="Admin login required")
    row = conn.execute(
        "SELECT id, path, thumb_path, mime, ext, original_filename FROM superseded WHERE id = ?", (superseded_id,)
    ).fetchone()
    if row is None:
        raise _not_found()
    return row


def _stored_path(relpath: str | None) -> Path:
    if not relpath:
        raise _not_found()
    try:
        return storage.from_rel(relpath)
    except ValueError:
        log.warning("superseded path outside the data dir: %s", relpath)
        raise _not_found()


@router.api_route("/superseded/{superseded_id}/thumb", methods=METHODS)
def superseded_thumb(superseded_id: int, request: Request, conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _superseded(conn, superseded_id, request)
    return _send(_stored_path(row["thumb_path"]), "image/webp", PRIVATE)


@router.api_route("/superseded/{superseded_id}/original", methods=METHODS)
def superseded_original(superseded_id: int, request: Request, download: int = 0,
                        conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _superseded(conn, superseded_id, request)
    name = media_file_name(row["original_filename"], row["ext"])
    extra = {"content-disposition": content_disposition(name)} if download else None
    return _send(_stored_path(row["path"]), row["mime"], PRIVATE, extra)
