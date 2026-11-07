"""Live Photos and motion photos (docs/ARCHITECTURE.md section 20).

- content_identifier(exif) reads Apple's ContentIdentifier (a UUID) from a still's MakerNote, and
  video_content_identifier(tags) the same value from a MOV's QuickTime tags, so the two halves of an
  iPhone Live Photo can be paired.
- extract_motion_clip(path) returns the MP4 bytes embedded in an Android motion photo (Google's XMP
  MicroVideo / Container tags, or Samsung's MotionPhoto_Data trailer), or None.
- find_partner(conn, item) finds the other half of a pair already stored: the same content
  identifier, the same upload pair key, or the same file stem with a short clip uploaded close in time.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

UUID_RE = re.compile(rb"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}")
QT_CONTENT_ID = "com.apple.quicktime.content.identifier"
MAKERNOTE_TAG = 0x927C
EXIF_IFD = 0x8769
MAX_CLIP_S = 4.0           # a Live Photo clip is about 3 s; a longer MOV is a video of its own
PAIR_WINDOW_S = 3600       # stem pairing: both halves uploaded within an hour
STEM_RE = re.compile(r"^(.*?)(?:\.[A-Za-z0-9]{1,5})?$")
_MP4_BRANDS = (b"ftypisom", b"ftypmp42", b"ftypmp41", b"ftypqt  ", b"ftypM4V ", b"ftypavc1", b"ftypiso2")


# ------------------------------------------------------------------ identifiers
def content_identifier(exif: Any) -> str | None:
    """Apple's ContentIdentifier from a Pillow Exif mapping (MakerNote tag 0x11 holds a UUID)."""
    if exif is None:
        return None
    note = None
    try:
        sub = exif.get_ifd(EXIF_IFD)
        note = sub.get(MAKERNOTE_TAG)
    except Exception:  # noqa: BLE001 - malformed EXIF
        note = None
    if note is None:
        try:
            note = exif.get(MAKERNOTE_TAG)
        except Exception:  # noqa: BLE001
            note = None
    if isinstance(note, str):
        note = note.encode("latin-1", "ignore")
    if not isinstance(note, (bytes, bytearray)) or not note.startswith(b"Apple iOS"):
        return None
    m = UUID_RE.search(bytes(note))
    return m.group(0).decode("ascii").upper() if m else None


def video_content_identifier(tags: dict[str, Any] | None) -> str | None:
    """The same identifier from a MOV's ffprobe format/stream tags."""
    if not tags:
        return None
    for key, value in tags.items():
        if str(key).lower() == QT_CONTENT_ID and value:
            text = str(value).strip()
            if UUID_RE.fullmatch(text.encode("ascii", "ignore")):
                return text.upper()
    return None


def stem_of(filename: str | None) -> str:
    name = Path(filename or "").name
    m = STEM_RE.match(name)
    return (m.group(1) if m else name).casefold()


# ------------------------------------------------------------------ motion photos
_XMP_LENGTH = re.compile(rb'Item:Length="(\d+)"')
_XMP_SEMANTIC = re.compile(rb'Item:Semantic="(MotionPhoto|MicroVideo)"', re.I)
_XMP_MICRO_OFFSET = re.compile(rb'(?:GCamera:)?MicroVideoOffset="(\d+)"')
_SAMSUNG_MARK = b"MotionPhoto_Data"


def _is_mp4(data: bytes) -> bool:
    return len(data) > 12 and data[4:12] in _MP4_BRANDS


def extract_motion_clip(path: Path, max_bytes: int = 200 * 1024 * 1024) -> bytes | None:
    """The MP4 appended to a Google or Samsung motion photo, or None."""
    try:
        size = path.stat().st_size
        if size < 1024 or size > max_bytes:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if not data.startswith(b"\xff\xd8"):  # JPEG only
        return None
    head = data[: 2 * 1024 * 1024]
    # Google: the XMP says how long the trailing clip is (newest form), or where it starts (older form).
    sem = _XMP_SEMANTIC.search(head)
    if sem is not None:
        lengths = [int(m.group(1)) for m in _XMP_LENGTH.finditer(head)]
        for length in sorted(set(lengths), reverse=True):
            if 0 < length < size:
                clip = data[size - length:]
                if _is_mp4(clip):
                    return clip
    off = _XMP_MICRO_OFFSET.search(head)
    if off is not None:
        length = int(off.group(1))
        if 0 < length < size:
            clip = data[size - length:]
            if _is_mp4(clip):
                return clip
    # Samsung: a trailer with the marker right before the clip.
    pos = data.rfind(_SAMSUNG_MARK)
    if pos != -1:
        clip = data[pos + len(_SAMSUNG_MARK):]
        if _is_mp4(clip):
            return clip
        start = clip.find(b"ftyp")
        if 4 <= start <= 64:
            clip = clip[start - 4:]
            if _is_mp4(clip):
                return clip
    # Last resort: an MP4 header anywhere after the JPEG end marker.
    end = data.rfind(b"\xff\xd9")
    if end != -1:
        tail = data[end + 2:]
        start = tail.find(b"ftyp")
        if 4 <= start <= 1024:
            clip = tail[start - 4:]
            if _is_mp4(clip):
                return clip
    return None


# ------------------------------------------------------------------ pairing
def find_partner(conn: sqlite3.Connection, item: sqlite3.Row, pair_key: str | None) -> sqlite3.Row | None:
    """The stored other half of this item's Live Photo: for a still the MOV, for a MOV the still.
    Matches by content identifier, by the uploads' pair key, or by file stem with a clip under
    MAX_CLIP_S uploaded within PAIR_WINDOW_S. Already paired items and companions are skipped."""
    kind = item["kind"]
    if kind not in ("image", "video"):
        return None
    if kind == "video" and (item["duration_s"] is None or float(item["duration_s"]) > MAX_CLIP_S):
        return None
    want = "video" if kind == "image" else "image"
    free = ("i.deleted_at IS NULL AND i.companion_of IS NULL AND i.live_video_item_id IS NULL AND i.id != ? "
            "AND i.kind = ?")
    params: list[Any] = [int(item["id"]), want]
    if want == "video":
        free += f" AND i.duration_s IS NOT NULL AND i.duration_s <= {MAX_CLIP_S}"
    candidates: list[sqlite3.Row] = []
    if item["content_id"]:
        candidates += conn.execute(f"SELECT i.* FROM items i WHERE {free} AND i.content_id = ?",
                                   [*params, item["content_id"]]).fetchall()
    if not candidates and pair_key:
        candidates += conn.execute(
            f"SELECT i.* FROM items i JOIN uploads u ON u.item_id = i.id WHERE {free} AND u.pair_key = ?",
            [*params, pair_key]).fetchall()
    if not candidates:
        stem = stem_of(item["original_filename"])
        lo, hi = _window(item["created_at"])
        if stem and lo:
            rows = conn.execute(
                f"SELECT i.* FROM items i WHERE {free} AND i.created_at >= ? AND i.created_at <= ?",
                [*params, lo, hi]).fetchall()
            candidates += [r for r in rows if stem_of(r["original_filename"]) == stem]
    return candidates[0] if candidates else None


def _window(created_at: str | None) -> tuple[str | None, str | None]:
    """The upload-time window around created_at, in the same ISO form the rows use."""
    from datetime import timedelta

    from app.util import parse_iso

    dt = parse_iso(created_at)
    if dt is None:
        return None, None
    fmt = "%Y-%m-%dT%H:%M:%S.000Z"
    return (dt - timedelta(seconds=PAIR_WINDOW_S)).strftime(fmt), (dt + timedelta(seconds=PAIR_WINDOW_S + 1)).strftime(fmt)


def sync_companion(conn: sqlite3.Connection, still_id: int) -> None:
    """Copy the still's vault flag and trash state onto its companion clip item."""
    from app.util import now_iso

    row = conn.execute("SELECT live_video_item_id, nsfw, deleted_at FROM items WHERE id = ?", (still_id,)).fetchone()
    if row is None or not row["live_video_item_id"]:
        return
    conn.execute(
        "UPDATE items SET nsfw = ?, deleted_at = ?, updated_at = ? WHERE id = ? AND companion_of = ?",
        (int(row["nsfw"]), row["deleted_at"], now_iso(), int(row["live_video_item_id"]), int(still_id)),
    )
