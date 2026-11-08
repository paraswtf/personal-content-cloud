"""File serving (docs/ARCHITECTURE.md sections 4 and 13).

CONTRACT. Router without a prefix, mounted at /files. GET and HEAD. Every route needs a logged-in
session (401 login_required). Uses starlette FileResponse (Range / 206, which Safari needs for video).

GET /files/thumb/{uid}?v=          WebP thumbnail (page 1 for documents, poster for videos)
GET /files/preview/{uid}?v=        large WebP preview
GET /files/preview-small/{uid}?v=  small WebP preview, made from the large one when missing
GET /files/original/{uid}?v=&download=1
                                   the original bytes with items.mime; download=1 adds Content-Disposition
                                   attachment named by util.media_file_name. Inline PDFs get
                                   CSP "default-src 'none'; sandbox allow-same-origin" and X-Frame-Options
                                   SAMEORIGIN so the browser's viewer works in the app; every other
                                   original gets "default-src 'none'; sandbox".
GET /files/video/{uid}?v=          the H.264 streaming copy when preview_status='done', else the original
GET /files/frame/{uid}?t=&size=full|preview&v=&download=1
                                   JPEG of the video frame shown at t (seconds); full frames carry EXIF
                                   (Orientation 1, camera, capture time in settings.local_timezone)
GET /files/page/{uid}/{n}?v=       document page render (rendered on first request when missing);
                                   for images, page 1 is the large preview
GET /files/face/{id}, /files/body/{id}   crops
GET /files/doc-photo/{uid}?v=      photo cropped out of an ID document
Rules:
- Items are looked up by uid; paths come from app.storage with the CURRENT version and ext.
  Trashed items are served (the trash page shows them).
- NSFW items (and their crops, pages and ID photos): 403 vault_locked unless the session's vault is
  unlocked; when served they are never cached ("private, no-store").
- Cache-Control: v == current version -> "private, max-age=31536000, immutable", otherwise
  "private, no-cache"; crops "private, max-age=86400". v is any string (vault items' URLs carry
  "<version>n", api.common.url_version, so flagging an item changes its URLs).
- Every response: X-Content-Type-Options nosniff and a sandboxing CSP.
"""
from __future__ import annotations

import io
import logging
import math
import sqlite3
import threading
import unicodedata
import zlib
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image

from app import storage
from app.api.deps import get_db, require_user
from app.auth.sessions import SessionInfo
from app.config import settings
from app.media import images
from app.media import video as video_media
from app.media.probe import ToolFailure
from app.util import media_file_name, safe_filename

log = logging.getLogger(__name__)

router = APIRouter()

IMMUTABLE = "private, max-age=31536000, immutable"
NO_CACHE = "private, no-cache"
NO_STORE = "private, no-store"
CROP_CACHE = "private, max-age=86400"
CSP = "default-src 'none'; sandbox"
PDF_CSP = "default-src 'none'; sandbox allow-same-origin"
METHODS = ["GET", "HEAD"]

# Striped locks for making missing small previews and page renders: two requests for one item never
# both do the work, and the lock table does not grow with the number of items.
_LOCKS = tuple(threading.Lock() for _ in range(64))

# Video frames: at most FRAME_MAX_CONCURRENT ffmpeg extractions at once (a full 4K frame is about
# 25 MB decoded). Preview-size frames are kept in a small LRU keyed (uid, version, source, ms).
FRAME_MAX_CONCURRENT = 2
FRAME_WAIT_S = 1.0  # a frame request waits this long for a slot, then gets 503: request threads are shared
FRAME_RETRY_AFTER_S = 2
EXACT_SEEK_EXTS = frozenset({".mts", ".m2ts", ".ts"})
FRAME_PREVIEW_SIDE = 1280
FRAME_PREVIEW_QUALITY = 82
FRAME_FULL_QUALITY = 92
FRAME_CACHE_ENTRIES = 8
_frame_sem = threading.BoundedSemaphore(FRAME_MAX_CONCURRENT)
_frame_cache: OrderedDict[tuple[str, int, str, int], Image.Image] = OrderedDict()
_frame_cache_lock = threading.Lock()

_ITEM_COLUMNS = ("id, uid, kind, version, ext, mime, original_filename, nsfw, preview_status, page_count, "
                 "taken_at, duration_s, fps, camera, deleted_at, companion_of")


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="File not found")


def _vault_locked() -> HTTPException:
    return HTTPException(status_code=403, detail="vault_locked")


def _lock_for(key: str) -> threading.Lock:
    return _LOCKS[zlib.crc32(key.encode()) % len(_LOCKS)]


def _item(conn: sqlite3.Connection, uid: str, session: SessionInfo) -> sqlite3.Row:
    row = conn.execute(f"SELECT {_ITEM_COLUMNS} FROM items WHERE uid = ?", (uid,)).fetchone()
    if row is None:
        raise _not_found()
    if row["nsfw"] and not session.vault_unlocked:
        raise _vault_locked()
    return row


def _cache(row: sqlite3.Row, v: str | None) -> str:
    """v is the URL's cache key (api.common.url_version: the version, 'n' added for vault items). Any
    value is accepted; only the current key of a non-vault item is cached for good."""
    if row["nsfw"]:
        return NO_STORE  # never left in the browser cache once the vault locks again
    return IMMUTABLE if v is not None and v.strip() == str(row["version"]) else NO_CACHE


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


def content_disposition(filename: str, ext: str = "", disposition: str = "attachment") -> str:
    """attachment; filename="<ascii>"; filename*=UTF-8''<percent-encoded> (RFC 6266 / 5987)."""
    name = safe_filename(filename)
    if ext and not Path(name).suffix:
        name += ext
    return f"{disposition}; filename=\"{_ascii_fallback(name)}\"; filename*=UTF-8''{quote(name, safe='')}"


# --------------------------------------------------------------------- items
def _picture_row(conn: sqlite3.Connection, row: sqlite3.Row, session: SessionInfo) -> sqlite3.Row:
    """The row whose thumb and previews stand for this item. A Live Photo clip (companion) keeps none of
    its own (section 20): its pictures are the still's, so an old link to the clip's thumb still shows
    the Live Photo."""
    if row["companion_of"]:
        still = conn.execute(f"SELECT {_ITEM_COLUMNS} FROM items WHERE id = ?", (int(row["companion_of"]),)).fetchone()
        if still is not None:
            if still["nsfw"] and not session.vault_unlocked:
                raise _vault_locked()
            return still
    return row


@router.api_route("/thumb/{uid}", methods=METHODS)
def thumb(uid: str, v: str | None = None, session: SessionInfo = Depends(require_user),
          conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _picture_row(conn, _item(conn, uid, session), session)
    return _send(storage.thumb_path(row["uid"], row["version"]), "image/webp", _cache(row, v))


@router.api_route("/preview/{uid}", methods=METHODS)
def preview(uid: str, v: str | None = None, session: SessionInfo = Depends(require_user),
            conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _picture_row(conn, _item(conn, uid, session), session)
    return _send(storage.preview_path(row["uid"], row["version"]), "image/webp", _cache(row, v))


def small_preview_file(uid: str, version: int) -> Path | None:
    """The file the small preview URL serves: the small preview; the large one when it already fits the
    small size (no small file is stored then, section 3); otherwise a small preview made now from the
    large one when it is missing. None when no large preview exists or the small one could not be made
    (decode or disk error)."""
    small = storage.preview_small_path(uid, version)
    if small.is_file():
        return small
    large = storage.preview_path(uid, version)
    if not large.is_file():
        return None
    try:
        with Image.open(large) as im:
            if images.small_is_large(im.size):
                return large
    except Exception as e:  # noqa: BLE001 - a corrupt preview is served as it is by the caller
        log.warning("could not read the preview of %s v%s: %s", uid, version, e)
        return None
    with _lock_for(f"small:{uid}"):
        if small.is_file():
            return small
        try:
            with Image.open(large) as im:
                im.load()
                img = im.convert("RGB")
            images.save_webp(img, small, settings.preview_small_size, settings.preview_quality)
        except Exception as e:  # corrupt preview, full disk: serve the large one instead
            log.warning("could not make the small preview of %s v%s: %s", uid, version, e)
            return None
    log.info("made the missing small preview of %s v%s", uid, version)
    return small


@router.api_route("/preview-small/{uid}", methods=METHODS)
def preview_small(uid: str, v: str | None = None, session: SessionInfo = Depends(require_user),
                  conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _picture_row(conn, _item(conn, uid, session), session)
    path = small_preview_file(row["uid"], row["version"])
    if path is not None:
        return _send(path, "image/webp", _cache(row, v))
    # Stopgap (no small file could be made): never cache the large file under the small URL.
    return _send(storage.preview_path(row["uid"], row["version"]), "image/webp",
                 NO_STORE if row["nsfw"] else NO_CACHE)


def _original_type(row: sqlite3.Row) -> str:
    mime = row["mime"] or "application/octet-stream"
    if mime.startswith("text/") and "charset" not in mime:
        mime += "; charset=utf-8"
    return mime


@router.api_route("/original/{uid}", methods=METHODS)
def original(uid: str, v: str | None = None, download: int = 0, session: SessionInfo = Depends(require_user),
             conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _item(conn, uid, session)
    name = media_file_name(row["original_filename"], row["ext"])
    extra: dict[str, str] = {}
    if download:
        extra["content-disposition"] = content_disposition(name)
    elif row["mime"] == "application/pdf":
        # The browser's PDF viewer needs a same-origin document it may frame.
        extra["content-security-policy"] = PDF_CSP
        extra["x-frame-options"] = "SAMEORIGIN"
        extra["content-disposition"] = content_disposition(name, disposition="inline")
    return _send(storage.original_path(row["uid"], row["version"], row["ext"]), _original_type(row),
                 _cache(row, v), extra)


@router.api_route("/video/{uid}", methods=METHODS)
def video(uid: str, v: str | None = None, session: SessionInfo = Depends(require_user),
          conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _item(conn, uid, session)
    if row["kind"] != "video":
        raise _not_found()
    if row["preview_status"] == "done":
        web = storage.video_preview_path(row["uid"], row["version"])
        if web.is_file():
            return _send(web, "video/mp4", _cache(row, v))
    return _send(storage.original_path(row["uid"], row["version"], row["ext"]), row["mime"], _cache(row, v))


# --------------------------------------------------------------------- pages
def _ensure_page(row: sqlite3.Row, n: int) -> Path | None:
    """The page render, made now when missing (documents only)."""
    uid, version = row["uid"], int(row["version"])
    dest = storage.page_path(uid, version, n)
    if dest.is_file():
        return dest
    if row["kind"] not in ("pdf", "document"):
        return None
    original = storage.original_path(uid, version, row["ext"])
    if not original.is_file():
        return None
    with _lock_for(f"page:{uid}:{n}"):
        if dest.is_file():
            return dest
        try:
            from app.media import documents

            made = documents.render_page(original, row["kind"], uid, version, n,
                                         max_side=settings.page_render_size, ext=row["ext"])
        except Exception as e:  # damaged file: no render
            log.warning("could not render page %s of %s: %s", n, uid, e)
            return None
    return dest if made is not None and dest.is_file() else None


@router.api_route("/page/{uid}/{n}", methods=METHODS)
def page(uid: str, n: int, v: str | None = None, session: SessionInfo = Depends(require_user),
         conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _item(conn, uid, session)
    if n < 1:
        raise _not_found()
    if row["kind"] in ("pdf", "document"):
        if row["page_count"] and n > int(row["page_count"]):
            raise _not_found()
        path = _ensure_page(row, n)
        if path is None:
            raise _not_found()
        return _send(path, "image/webp", _cache(row, v))
    rendered = storage.page_path(row["uid"], row["version"], n)
    if rendered.is_file():
        return _send(rendered, "image/webp", _cache(row, v))
    if row["kind"] == "image" and n == 1:
        return _send(storage.preview_path(row["uid"], row["version"]), "image/webp", _cache(row, v))
    raise _not_found()


@router.api_route("/doc-photo/{uid}", methods=METHODS)
def doc_photo(uid: str, v: str | None = None, session: SessionInfo = Depends(require_user),
              conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _item(conn, uid, session)
    return _send(storage.doc_photo_path(row["uid"], row["version"]), "image/webp", _cache(row, v))


# --------------------------------------------------------------------- crops
def _crop(conn: sqlite3.Connection, table: str, crop_id: int, session: SessionInfo) -> sqlite3.Row:
    row = conn.execute(
        f"SELECT c.id, i.nsfw FROM {table} c JOIN items i ON i.id = c.item_id WHERE c.id = ?", (crop_id,)
    ).fetchone()
    if row is None:
        raise _not_found()
    if row["nsfw"] and not session.vault_unlocked:
        raise _vault_locked()
    return row


@router.api_route("/face/{face_id}", methods=METHODS)
def face(face_id: int, session: SessionInfo = Depends(require_user),
         conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _crop(conn, "faces", face_id, session)
    return _send(storage.face_crop_path(int(row["id"])), "image/webp", NO_STORE if row["nsfw"] else CROP_CACHE)


@router.api_route("/animal/{animal_id}", methods=METHODS)
def animal(animal_id: int, session: SessionInfo = Depends(require_user),
           conn: sqlite3.Connection = Depends(get_db)) -> Response:
    row = _crop(conn, "animals", animal_id, session)
    return _send(storage.animal_crop_path(int(row["id"])), "image/webp", NO_STORE if row["nsfw"] else CROP_CACHE)


@router.api_route("/body/{body_id}", methods=METHODS)
def body(body_id: int, session: SessionInfo = Depends(require_user),
         conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    row = _crop(conn, "bodies", body_id, session)
    return _send(storage.body_crop_path(int(row["id"])), "image/webp", NO_STORE if row["nsfw"] else CROP_CACHE)


# -------------------------------------------------------------- video frames
def _busy() -> HTTPException:
    return HTTPException(status_code=503, detail="Busy extracting other frames, please try again shortly",
                         headers={"Retry-After": str(FRAME_RETRY_AFTER_S)})


def _frame_ms(row: sqlite3.Row, t: float) -> int:
    """t rounded DOWN to the millisecond, within the clip (rounded to a microsecond first: 8.033 *
    1000 is 8032.999..., which would otherwise lose a millisecond)."""
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
    """("original" | "stream", file): preview frames come from the streaming copy when it exists
    (same timestamps, much cheaper to decode), full frames from the original."""
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
    """The frame shown at ms, holding a frame slot for every ffprobe and ffmpeg run; no slot within
    FRAME_WAIT_S gives 503 so a burst cannot tie up the request threads."""
    if not _frame_sem.acquire(timeout=FRAME_WAIT_S):
        raise _busy()
    try:
        ms = _shown_ms(path, ms)

        def grab(at_ms: int) -> Image.Image | None:
            return video_media.extract_frame_image(path, at_ms / 1000.0, max_side, exact_seek=exact_seek,
                                                   gentle=True)

        frame = grab(ms)
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
    if not row["nsfw"]:
        with _frame_cache_lock:
            _frame_cache[key] = frame
            _frame_cache.move_to_end(key)
            while len(_frame_cache) > FRAME_CACHE_ENTRIES:
                _frame_cache.popitem(last=False)
    return frame


def _frame_exif(row: sqlite3.Row) -> bytes:
    """EXIF for a full frame: Orientation 1 (the pixels are upright), camera and the capture time
    (the item's taken_at in settings.local_timezone). No position is written."""
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
                local = dt.astimezone(ZoneInfo(settings.local_timezone or "UTC"))
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
    return exif.tobytes()


def frame_file_name(row: sqlite3.Row, ms: int) -> str:
    """"IMG_6915-1m05.4s.jpg" (time in whole tenths, rounded down)."""
    stem = Path(media_file_name(row["original_filename"], row["ext"])).stem or "video"
    tenths = ms // 100
    stamp = f"{tenths // 600}m{(tenths % 600) // 10:02d}.{tenths % 10}s"
    return f"{stem}-{stamp}.jpg"


@router.api_route("/frame/{uid}", methods=METHODS)
def frame(uid: str, t: float, v: str | None = None, size: str = "full", download: int = 0,
          session: SessionInfo = Depends(require_user), conn: sqlite3.Connection = Depends(get_db)) -> Response:
    if not math.isfinite(t):
        raise HTTPException(status_code=422, detail="t must be a finite number of seconds")
    if size not in ("full", "preview"):
        raise HTTPException(status_code=422, detail="size must be full or preview")
    row = _item(conn, uid, session)
    if row["kind"] != "video":
        raise _not_found()
    ms = _frame_ms(row, t)
    img = _frame_image(row, ms, size)
    buf = io.BytesIO()
    if size == "full":
        img.save(buf, "JPEG", quality=FRAME_FULL_QUALITY, exif=_frame_exif(row))
    else:
        img.save(buf, "JPEG", quality=FRAME_PREVIEW_QUALITY)
    del img
    headers = {"cache-control": _cache(row, v), "x-content-type-options": "nosniff", "content-security-policy": CSP}
    if download:
        headers["content-disposition"] = content_disposition(frame_file_name(row, ms))
    return Response(content=buf.getvalue(), media_type="image/jpeg", headers=headers)


@router.api_route("/live/{uid}", methods=METHODS)
def live_clip(uid: str, v: str | None = None, download: int = 0, session: SessionInfo = Depends(require_user),
              conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    """The clip embedded in a motion photo (section 20); a paired Live Photo plays its companion's
    /files/video instead."""
    row = _item(conn, uid, session)
    path = storage.live_clip_path(row["uid"], row["version"])
    extra = None
    if download:
        stem = Path(row["original_filename"] or uid).stem or uid
        extra = {"content-disposition": f'attachment; filename="{stem}-live.mp4"'}
    return _send(path, "video/mp4", _cache(row, v), extra)
