"""Small helpers shared across the backend."""
from __future__ import annotations

import base64
import json
import os
import re
import secrets
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_UID_ALPHABET = "abcdefghijkmnopqrstuvwxyz23456789"
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SPACES = re.compile(r"\s+")


def now_iso(delta_s: float = 0.0) -> str:
    """Current UTC time as ISO 8601 with milliseconds and a Z suffix (sortable as text)."""
    t = datetime.now(timezone.utc) + timedelta(seconds=delta_s)
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO 8601 string produced by now_iso (or any ISO with offset) into an aware datetime."""
    if not value:
        return None
    try:
        v = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def local_date(value: str | None, tz_name: str) -> str | None:
    """Calendar date (YYYY-MM-DD) of an ISO UTC time in the given zone."""
    dt = parse_iso(value)
    if dt is None:
        return None
    try:
        zone = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        zone = timezone.utc
    return dt.astimezone(zone).date().isoformat()


def local_day_start_utc(value: str, tz_name: str, *, next_day: bool = False) -> str:
    """'YYYY-MM-DD' (a calendar date in tz_name) -> the UTC instant that day starts (or the next day
    starts), formatted like now_iso so it compares as text with stored times. Raises ValueError."""
    from datetime import date, time as dtime

    d = date.fromisoformat(value.strip()[:10])
    if next_day:
        d = d + timedelta(days=1)
    try:
        zone = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        zone = timezone.utc
    dt = datetime.combine(d, dtime(0, 0), tzinfo=zone).astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_uid(length: int = 14) -> str:
    """Random lowercase id, safe for URLs and file names (~70 bits for length 14)."""
    return "".join(secrets.choice(_UID_ALPHABET) for _ in range(length))


def new_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def clean_name(value: str, max_len: int = 80) -> str:
    """Normalise a human name: NFC, no control chars, single spaces, trimmed."""
    value = unicodedata.normalize("NFC", value or "")
    value = _CONTROL.sub("", value)
    value = _SPACES.sub(" ", value).strip()
    return value[:max_len].strip()


def name_key(value: str) -> str:
    return clean_name(value, 200).casefold()


def safe_filename(name: str, default: str = "file") -> str:
    """File name safe for Content-Disposition and logs: no paths, no control chars."""
    name = unicodedata.normalize("NFC", name or "")
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = _CONTROL.sub("", name).strip().strip(".")
    return name[:200] or default


# Suffixes an uploaded name may keep for each storage ext (items.ext). Any other suffix is replaced, so
# a photo uploaded as "invoice.hta" or "photo.jpg.bat" never lands on a downloader's disk as a program.
_EXT_ALIASES = {
    ".jpg": (".jpg", ".jpeg", ".jpe", ".jfif"),
    ".tif": (".tif", ".tiff"),
    ".dng": (".dng",), ".nef": (".nef",), ".cr2": (".cr2",), ".cr3": (".cr3",), ".arw": (".arw",),
    ".raf": (".raf",), ".orf": (".orf",), ".rw2": (".rw2",), ".pef": (".pef",), ".srw": (".srw",),
    ".heic": (".heic", ".heif"), ".heif": (".heif", ".heic"),
    ".mp4": (".mp4", ".m4v"), ".m4v": (".m4v", ".mp4"), ".3gp": (".3gp", ".3g2"), ".mov": (".mov", ".qt"),
    ".mts": (".mts", ".m2ts", ".ts"),
    ".txt": (".txt", ".text", ".log"), ".md": (".md", ".markdown"), ".csv": (".csv",),
    ".docx": (".docx",), ".xlsx": (".xlsx", ".xlsm"), ".pptx": (".pptx",), ".pdf": (".pdf",),
}
_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,8}")
_WINDOWS_INVALID = re.compile(r'[<>:"/\\|?*]')
_WINDOWS_RESERVED = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                               *(f"lpt{i}" for i in range(1, 10))})


def media_file_name(name: str | None, ext: str) -> str:
    """Name to save a stored file under (downloads): safe_filename without Unicode format
    characters (a bidi override can disguise "gpj.cmd" as "dmc.jpg"), without Windows-invalid
    characters and trailing dots or spaces, never a Windows device name, and with a suffix that
    matches the stored file: the uploaded suffix stays only when it is an alias of `ext`."""
    ext = (ext or "").lower()
    clean = "".join(c for c in safe_filename(name or "", "") if unicodedata.category(c) != "Cf")
    clean = _WINDOWS_INVALID.sub("_", clean).strip().rstrip(". ")
    stem, suffix = os.path.splitext(clean)
    if suffix.lower() not in _EXT_ALIASES.get(ext, (ext,)):
        stem = stem.rstrip(". ") if _SUFFIX.fullmatch(suffix) else clean
        suffix = ext
    if stem.split(".", 1)[0].strip().casefold() in _WINDOWS_RESERVED:
        stem = "_" + stem
    return (stem or "file") + suffix


def encode_cursor(values: list[Any]) -> str:
    raw = json.dumps(values, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str | None) -> list[Any] | None:
    """Inverse of encode_cursor. Returns None for a missing or malformed cursor."""
    if not cursor:
        return None
    try:
        pad = "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(cursor + pad))
        return value if isinstance(value, list) else None
    except (ValueError, json.JSONDecodeError):
        return None


def json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def json_loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def b64url_decode(value: str) -> bytes:
    pad = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + pad)
