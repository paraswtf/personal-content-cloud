"""Job handlers: turning completed uploads into gallery items, faces, video previews.

CONTRACT (implemented by the ingest agent). Full procedure in docs/SPEC.md section "Ingest".
Every handler is idempotent: it may run again after a crash at any point, so it must
(a) check current DB state first (terminal upload status -> no-op; media.version mismatch -> no-op),
(b) write files to deterministic paths (a new item reuses the upload uid as media uid; a replacement
    writes version+1 files) and look for the file at its destination if the staged file is gone,
(c) switch the DB over in one transaction, and only delete old files after that commit.

Implementation note on "move": the staged file is hard-linked to its destination (through a link
in data_dir/tmp + os.replace, so the switch is atomic) and the staged name is only unlinked after
the DB commit. Until then the upload's bytes always exist at the staged path, so a failed attempt
that is retried later (other ingest jobs run during the retry backoff) can never lose them, and a
crash at any point leaves a state the next run handles. The fallback lookup at the destination is
kept for files moved by other means.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

from PIL import Image

from app.changelog import trash
from app.changelog.capture import claim
from app import jobs, storage
from app.config import settings
from app.db import GPS_BACKFILL_CURSOR, GPS_BACKFILL_END, bump_faces_version, faces_version, kv_get, kv_set, tx
from app.export import people as people_meta
from app.faces import cluster
from app.faces.index import get_index
from app.faces.pipeline import CROP_SIZE, FaceRecord, analyze_image, analyze_video
from app.media import dedup, embed, images, video, xmp
from app.media.hashing import Fingerprint, image_fingerprint, sha256_file, video_fingerprint
from app.media.probe import MediaInfo, ToolFailure, normalize_taken_at, probe, read_gps
from app.util import json_dumps, new_uid, now_iso

log = logging.getLogger(__name__)

TERMINAL = frozenset({"stored", "duplicate", "replaced", "lower_quality", "rejected", "error", "cancelled"})
ACTIVE = frozenset({"uploading", "queued", "processing"})
ORPHAN_STAGING_AGE_S = 3600  # staged files without an upload row are removed after this long
CROP_QUALITY = 85
_UID_ALPHABET = "abcdefghijkmnopqrstuvwxyz23456789"  # same alphabet as app.util.new_uid
_ORIGINAL_NAME = re.compile(r"^(?P<uid>[a-z0-9]+)\.v(?P<version>\d+)(?P<ext>\.[a-z0-9]+)$")
# thumbs/previews/videos: <uid>.v<n>.webp, the small preview <uid>.v<n>.sm.webp, <uid>.v<n>.mp4
_DERIVED_NAME = re.compile(r"^(?P<uid>[a-z0-9]+)\.v(?P<version>\d+)\.(?:sm\.webp|webp|mp4)$")
_FACE_NAME = re.compile(r"^(?P<id>\d+)\.webp$")
# geotag cache: <uid>.v<n>.<pos>.<size>.jpg
_GEOTAG_NAME = re.compile(r"^(?P<uid>[a-z0-9]+)\.v(?P<version>\d+)\.(?:tl|tc|tr|bl|bc|br)\.(?:preview|full)\.jpg$")
_SWEPT_DIRS = ("originals", "thumbs", "previews", "videos", "superseded", "faces", "geotag")
GPS_BACKFILL_BATCH = 20


class _Stale(Exception):
    """The DB changed under a running job (item deleted or replaced): give up this attempt quietly."""


# ------------------------------------------------------------------ small helpers
def _same_path(a: Path | None, b: Path | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def _place(src: Path, dest: Path) -> bool:
    """Make dest a hard link to src (atomically replacing anything at dest). src stays in place.
    Returns False when src already is dest. Falls back to a copy where links are not possible."""
    if _same_path(src, dest):
        return False
    tmp = storage.tmp_path(".link")
    try:
        try:
            os.link(src, tmp)
        except OSError as e:
            log.warning("hard link %s -> %s failed (%s); copying instead", src, dest, e)
            shutil.copyfile(src, tmp)
        storage.move_into_place(tmp, dest)
    finally:
        storage.safe_unlink(tmp)
    return True


def _is_referenced(conn: sqlite3.Connection, path: Path) -> bool:
    """True if a committed row points at this file: a media original or derivative (thumb, preview,
    video preview) of a stored version, a superseded file or thumbnail, or a face crop."""
    try:
        relpath = storage.rel(path)
    except ValueError:
        return False
    if conn.execute("SELECT 1 FROM superseded WHERE path = ? OR thumb_path = ?", (relpath, relpath)).fetchone():
        return True
    name = Path(path).name
    top = relpath.split("/", 1)[0]
    if top == "originals":
        m = _ORIGINAL_NAME.match(name)
        if m and conn.execute(
            "SELECT 1 FROM media WHERE uid = ? AND version = ? AND ext = ?",
            (m["uid"], int(m["version"]), m["ext"]),
        ).fetchone():
            return True
    elif top in ("thumbs", "previews", "videos"):
        m = _DERIVED_NAME.match(name)
        if m and conn.execute(
            "SELECT 1 FROM media WHERE uid = ? AND version = ?", (m["uid"], int(m["version"]))
        ).fetchone():
            return True
    elif top == "faces":
        m = _FACE_NAME.match(name)
        if m and conn.execute("SELECT 1 FROM faces WHERE id = ?", (int(m["id"]),)).fetchone():
            return True
    return False


def _drop_source(conn: sqlite3.Connection, src: Path, dest: Path | None = None) -> None:
    """Delete the upload's source file once the DB no longer needs it (never a referenced file)."""
    if _same_path(src, dest):
        return
    if _is_referenced(conn, src):
        log.warning("not deleting %s: it is referenced by the database", src)
        return
    storage.safe_unlink(src)


def _fallback_candidates(upload_uid: str) -> list[Path]:
    """Where an interrupted earlier run may have put the upload's file."""
    out: list[Path] = []
    orig_dir = storage.original_path(upload_uid, 1, ".x").parent
    if orig_dir.is_dir():
        out.extend(sorted(p for p in orig_dir.glob(f"{upload_uid}.v1.*") if p.is_file()))
    sup_dir = storage.superseded_path(upload_uid, ".x").parent
    if sup_dir.is_dir():
        thumb = storage.superseded_thumb_path(upload_uid).name
        out.extend(sorted(p for p in sup_dir.glob(f"{upload_uid}.*") if p.is_file() and p.name != thumb))
    return out


def _locate_upload_file(conn: sqlite3.Connection, upload_uid: str) -> Path | None:
    staged = storage.staging_path(upload_uid)
    if staged.is_file():
        return staged
    for cand in _fallback_candidates(upload_uid):
        if not _is_referenced(conn, cand):
            log.info("upload %s: staged file gone, resuming from %s", upload_uid, cand)
            return cand
    return None


def short_error(exc: BaseException) -> str:
    """One line describing an exception, without data dir paths, for upload messages."""
    lines = str(exc).strip().splitlines()
    text = lines[0] if lines else type(exc).__name__
    return text.replace(str(settings.data_dir), "<data>")[:160]


def _preview_status(info: MediaInfo) -> str:
    if info.kind == "video" and settings.video_previews and video.needs_web_preview(info, info.ext):
        return "pending"
    return "none"


def _write_derivatives(info: MediaInfo, original: Path, img: Image.Image | None, thumb: Path, preview: Path,
                       preview_small: Path) -> None:
    if info.kind == "image":
        assert img is not None
        images.make_image_derivatives(img, thumb, preview, preview_small)
    else:
        video.make_video_derivatives(original, info, thumb, preview, preview_small)


def _fingerprint(info: MediaInfo, path: Path) -> tuple[Fingerprint, Image.Image | None]:
    """Fingerprint (and, for photos, the decoded full-resolution image). ValueError if undecodable."""
    if info.kind == "image":
        img = images.load_oriented(path)
        return image_fingerprint(img), img
    return video_fingerprint(path, info), None


MEDIA_ID_MARK = "media_id_mark"


def _next_media_id(conn: sqlite3.Connection) -> int:
    """Item ids are never handed out twice: an undone delete puts an item back under its old id
    (media.id is a plain INTEGER PRIMARY KEY, which would reuse the newest id after a delete)."""
    mark = int(kv_get(conn, MEDIA_ID_MARK, "0") or 0)
    top = int(conn.execute("SELECT COALESCE(MAX(id), 0) FROM media").fetchone()[0])
    new_id = max(mark, top) + 1
    kv_set(conn, MEDIA_ID_MARK, new_id)
    return new_id


def version_mark_key(media_id: int) -> str:
    return f"vmark:{int(media_id)}"


def next_version(conn: sqlite3.Connection, media_id: int, current: int) -> int:
    """The version a replacement gets. Never one the item had before: URLs (?v=), cached files
    and geotag renders are keyed by it, and an undone replacement rewinds media.version."""
    mark = int(kv_get(conn, version_mark_key(media_id), "0") or 0)
    return max(mark, int(current)) + 1


def _insert_media(conn: sqlite3.Connection, uid: str, info: MediaInfo, sha: str, fp: Fingerprint, size: int,
                  filename: str, uploader_id: int, preview_status: str, now: str) -> int:
    media_id = _next_media_id(conn)
    conn.execute(
        "INSERT INTO media(id, uid, kind, version, original_filename, ext, mime, sha256, size_bytes, width, height, "
        "duration_s, bitrate, video_codec, fps, has_audio, taken_at, exif_datetime, exif_subsec, camera, "
        "phash, dhash, sig, frames, uploader_id, hidden, faces_status, preview_status, face_count, "
        "gps_lat, gps_lon, gps_alt, created_at, updated_at) "
        "VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'pending', ?, 0, "
        "?, ?, ?, ?, ?)",
        (media_id, uid, info.kind, filename, info.ext, info.mime, sha, size, info.width, info.height,
         *_file_fields(info, fp), uploader_id, preview_status, info.gps_lat, info.gps_lon, info.gps_alt, now, now),
    )
    return media_id


def _file_fields(info: MediaInfo, fp: Fingerprint) -> tuple:
    """duration_s .. frames, in media column order."""
    return (
        info.duration_s, info.bitrate, info.video_codec, info.fps,
        None if info.has_audio is None else int(bool(info.has_audio)),
        info.taken_at, info.exif_datetime, info.exif_subsec, info.camera,
        fp.phash, fp.dhash, fp.sig, json_dumps(fp.frames) if fp.frames is not None else None,
    )


def _gps_from_old_original(info: MediaInfo, old_original: Path) -> None:
    """Neither copy has a stored position: the item may predate position reading and not be
    backfilled yet, so read the old original before it stops being the current file (the backfill
    only ever reads the current version). Fills info.gps_* when one is found."""
    try:
        gps = read_gps(old_original) if old_original.is_file() else None
    except Exception as e:  # ToolFailure or an unreadable file: the replacement goes ahead without
        log.info("replace: could not read the position of %s: %s", old_original.name, e)
        return
    if gps is not None:
        info.gps_lat, info.gps_lon, info.gps_alt = gps


def _gps_update_params(info: MediaInfo) -> tuple:
    """Parameters for the three gps CASE expressions of the replace UPDATE: each column keeps its
    old value when the new copy has no position (keyed on the new latitude, so an old altitude is
    never paired with new coordinates)."""
    lat = info.gps_lat if info.gps_lon is not None else None
    return (lat, lat, lat, info.gps_lon, lat, info.gps_alt)


def _add_contributor(conn: sqlite3.Connection, media_id: int, uploader_id: int | None, now: str) -> None:
    if uploader_id is None:
        return
    cur = conn.execute(
        "INSERT OR IGNORE INTO media_contributors(media_id, uploader_id, first_upload_at) VALUES (?, ?, ?)",
        (media_id, uploader_id, now),
    )
    if cur.rowcount == 0:
        claim(conn, "media_contributors", {"media_id": media_id, "uploader_id": uploader_id}, media_id)


def _set_known(conn: sqlite3.Connection, sha: str, media_id: int, relation: str, now: str) -> None:
    conn.execute(
        "INSERT INTO known_hashes(sha256, media_id, relation, created_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(sha256) DO UPDATE SET media_id = excluded.media_id, relation = excluded.relation",
        (sha, media_id, relation, now),
    )


def _set_upload(conn: sqlite3.Connection, upload_id: int, status: str, media_id: int | None,
                message: str | None, now: str, sha: str | None = None) -> None:
    conn.execute(
        "UPDATE uploads SET status = ?, media_id = ?, message = ?, updated_at = ?, sha256 = COALESCE(?, sha256) "
        "WHERE id = ?",
        (status, media_id, message, now, sha, upload_id),
    )


def _enqueue_followups(conn: sqlite3.Connection, media_id: int, version: int, preview_status: str,
                       people: bool = False) -> None:
    """Queue the item's faces job (and video preview). people=True only for a newly stored item:
    its faces job may then apply the people metadata the file carries (process_faces)."""
    payload = {"media_id": media_id, "version": version}
    if people:
        payload["people"] = 1
    jobs.enqueue(conn, "faces", payload, unique=True)
    if preview_status == "pending":
        jobs.enqueue(conn, "video_preview", {"media_id": media_id, "version": version}, unique=True)
    from app.search.indexer import notify

    notify()  # the search indexer picks the new item (or version) up within seconds


def _cleanup(paths: list[Path]) -> None:
    for p in paths:
        storage.safe_unlink(p)


def _cleanup_unreferenced(conn: sqlite3.Connection, paths: list[Path]) -> None:
    """Undo a failed attempt's files, except any a committed row points at: another process (or an
    earlier run) may have committed the same deterministic paths. When the check itself fails the
    file is kept; the orphan sweep in cleanup_stale_uploads removes it later if nothing uses it."""
    for p in paths:
        try:
            if _is_referenced(conn, p):
                log.warning("not deleting %s after a failed attempt: it is referenced by the database", p)
                continue
        except sqlite3.Error as e:
            log.warning("not deleting %s after a failed attempt: could not check the database (%s)", p, e)
            continue
        storage.safe_unlink(p)


def _rel_if_exists(path: Path) -> str | None:
    return storage.rel(path) if path.exists() else None


# ------------------------------------------------------------------ ingest
def process_upload(conn: sqlite3.Connection, upload_id: int) -> None:
    """'ingest' job: probe the staged file, re-check exact duplicates, find near-duplicates,
    then store as new / replace the existing item / keep as lower_quality, and record the outcome
    on the upload row (status, message, media_id) and contributors/known_hashes."""
    up = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if up is None:
        log.warning("ingest: upload %s does not exist", upload_id)
        return
    uid = up["uid"]
    if up["status"] in TERMINAL:
        # Done already (maybe a crash after the commit): only the staged name may be left over.
        storage.safe_unlink(storage.staging_path(uid))
        return
    if up["status"] == "uploading":
        log.warning("ingest: upload %s is still uploading, nothing to do", uid)
        return
    conn.execute(
        "UPDATE uploads SET status = 'processing', updated_at = ? WHERE id = ? AND status IN ('queued', 'processing')",
        (now_iso(), upload_id),
    )

    src = _locate_upload_file(conn, uid)
    if src is None:
        log.error("ingest: upload %s has no file (staged file and destinations missing)", uid)
        with tx(conn):
            _set_upload(conn, upload_id, "error", None, "The uploaded file is missing, please upload it again.",
                        now_iso())
        return

    info = probe(src)  # ToolFailure (ffmpeg could not run) propagates: the job is retried, nothing rejected
    if info is None:
        _reject(conn, up, src, None)
        return

    sha = up["sha256"] or sha256_file(src)
    known = conn.execute("SELECT media_id FROM known_hashes WHERE sha256 = ?", (sha,)).fetchone()
    if known is not None:
        now = now_iso()
        with tx(conn):
            _set_upload(conn, upload_id, "duplicate", int(known["media_id"]), None, now)
            conn.execute("UPDATE uploads SET sha256 = ? WHERE id = ?", (sha, upload_id))
            _add_contributor(conn, int(known["media_id"]), up["uploader_id"], now)
        _drop_source(conn, src)
        log.info("ingest: upload %s is an exact duplicate of media %s", uid, known["media_id"])
        return

    try:
        fp, img = _fingerprint(info, src)
    except ValueError as e:
        log.info("ingest: upload %s could not be decoded: %s", uid, e)
        _reject(conn, up, src, "The file could not be decoded. It may be damaged or incomplete.")
        return
    size = src.stat().st_size

    decision = dedup.find_duplicate(conn, info, fp, size)
    log.info("ingest: upload %s (%s %dx%d) -> %s %s", uid, info.kind, info.width, info.height,
             decision.action, decision.detail)
    if decision.action in ("replace", "lower_quality") and _exported_copy(conn, uid, src, info.ext, decision):
        # A file this portal exported (people metadata added) is the same item, not a new copy of it.
        target_id = int(decision.target_id or 0)
        now = now_iso()
        with tx(conn):
            _set_upload(conn, upload_id, "duplicate", target_id, None, now)
            conn.execute("UPDATE uploads SET sha256 = ? WHERE id = ?", (sha, upload_id))
            _add_contributor(conn, target_id, up["uploader_id"], now)
        _drop_source(conn, src)
        log.info("ingest: upload %s is an exported copy of media %s", uid, decision.target_uid)
        return
    if decision.action == "new":
        _store_new(conn, up, src, info, sha, fp, img, size)
    elif decision.action == "replace":
        _replace(conn, up, src, info, sha, fp, img, size, decision)
    else:
        _keep_lower_quality(conn, up, src, info, sha, img, size, decision)


def _exported_copy(conn: sqlite3.Connection, uid: str, src: Path, ext: str, decision: dedup.DedupDecision) -> bool:
    """The staged file carries a people block this portal signed for a file the matched item has
    had (a known_hashes row of it). Unreadable metadata only means: not an exported copy."""
    try:
        packet = embed.read_packet(src, ext)
        hub = xmp.read_hub(packet) if packet else None
        data = people_meta.verify(*hub) if hub else None
    except Exception as e:  # metadata must never fail an upload
        log.warning("ingest: upload %s: could not read its people metadata: %s", uid, short_error(e))
        return False
    if not data or not isinstance(data.get("sha256"), str):
        return False
    return conn.execute("SELECT 1 FROM known_hashes WHERE sha256 = ? AND media_id = ?",
                        (data["sha256"], decision.target_id)).fetchone() is not None


def _reject(conn: sqlite3.Connection, up: sqlite3.Row, src: Path, message: str | None) -> None:
    with tx(conn):
        _set_upload(conn, up["id"], "rejected", None, message, now_iso())
    _drop_source(conn, src)
    log.info("ingest: upload %s rejected (%s)", up["uid"], up["original_filename"])


def _store_new(conn: sqlite3.Connection, up: sqlite3.Row, src: Path, info: MediaInfo, sha: str,
               fp: Fingerprint, img: Image.Image | None, size: int) -> None:
    uid = up["uid"]
    dest = storage.original_path(uid, 1, info.ext)
    thumb, preview = storage.thumb_path(uid, 1), storage.preview_path(uid, 1)
    preview_small = storage.preview_small_path(uid, 1)
    written: list[Path] = []
    try:
        if _place(src, dest):
            written.append(dest)
        written += [thumb, preview, preview_small]
        _write_derivatives(info, dest, img, thumb, preview, preview_small)
        preview_status = _preview_status(info)
        now = now_iso()
        with tx(conn):
            if conn.execute("SELECT 1 FROM media WHERE uid = ?", (uid,)).fetchone():
                raise RuntimeError(f"media {uid} already exists")
            media_id = _insert_media(conn, uid, info, sha, fp, size, up["original_filename"], up["uploader_id"],
                                     preview_status, now)
            _add_contributor(conn, media_id, up["uploader_id"], now)
            _set_known(conn, sha, media_id, "current", now)
            _set_upload(conn, up["id"], "stored", media_id, None, now, sha)
            _enqueue_followups(conn, media_id, 1, preview_status, people=True)
    except BaseException:
        _cleanup_unreferenced(conn, written)
        raise
    _drop_source(conn, src, dest)
    log.info("ingest: upload %s stored as media %s", uid, media_id)


def _replace(conn: sqlite3.Connection, up: sqlite3.Row, src: Path, info: MediaInfo, sha: str, fp: Fingerprint,
             img: Image.Image | None, size: int, decision: dedup.DedupDecision) -> None:
    target = conn.execute("SELECT * FROM media WHERE id = ?", (decision.target_id,)).fetchone()
    if target is None:
        raise RuntimeError("the matched item disappeared, retrying")
    t_uid, old_v = target["uid"], int(target["version"])
    v = next_version(conn, int(target["id"]), old_v)
    if info.gps_lat is None and target["gps_lat"] is None:
        _gps_from_old_original(info, storage.original_path(t_uid, old_v, target["ext"]))
    dest = storage.original_path(t_uid, v, info.ext)
    thumb, preview = storage.thumb_path(t_uid, v), storage.preview_path(t_uid, v)
    preview_small = storage.preview_small_path(t_uid, v)
    written: list[Path] = []
    try:
        if _place(src, dest):
            written.append(dest)
        written += [thumb, preview, preview_small]
        _write_derivatives(info, dest, img, thumb, preview, preview_small)
        preview_status = _preview_status(info)
        now = now_iso()
        with tx(conn):
            cur = conn.execute("SELECT * FROM media WHERE id = ?", (target["id"],)).fetchone()
            if cur is None or int(cur["version"]) != old_v or cur["sha256"] != target["sha256"]:
                raise RuntimeError("the matched item changed during processing, retrying")
            old_original = storage.original_path(t_uid, old_v, cur["ext"])
            old_thumb = storage.thumb_path(t_uid, old_v)
            prev_upload = conn.execute(
                "SELECT id FROM uploads WHERE media_id = ? AND sha256 = ? AND status IN ('stored', 'replaced') "
                "AND removed = 0 ORDER BY id LIMIT 1",
                (cur["id"], cur["sha256"]),
            ).fetchone()
            conn.execute(
                "INSERT INTO superseded(media_id, reason, path, thumb_path, sha256, original_filename, ext, mime, "
                "size_bytes, width, height, duration_s, bitrate, uploader_id, upload_id, detail, created_at) "
                "VALUES (?, 'replaced', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cur["id"], _rel_if_exists(old_original), _rel_if_exists(old_thumb), cur["sha256"],
                 cur["original_filename"], cur["ext"], cur["mime"], cur["size_bytes"], cur["width"], cur["height"],
                 cur["duration_s"], cur["bitrate"], cur["uploader_id"], prev_upload["id"] if prev_upload else None,
                 decision.detail, now),
            )
            _set_known(conn, cur["sha256"], cur["id"], "replaced", now)
            _set_known(conn, sha, cur["id"], "current", now)
            # A replacement without EXIF (messaging apps strip it) keeps the item's capture time: the
            # dedup EXIF guard for later uploads depends on it. Its capture position likewise; the
            # position (lat, lon, alt) is taken as a whole from whichever copy has one.
            conn.execute(
                "UPDATE media SET version = ?, original_filename = ?, ext = ?, mime = ?, sha256 = ?, size_bytes = ?, "
                "width = ?, height = ?, duration_s = ?, bitrate = ?, video_codec = ?, fps = ?, has_audio = ?, "
                "taken_at = COALESCE(?, taken_at), exif_datetime = COALESCE(?, exif_datetime), "
                "exif_subsec = CASE WHEN ? IS NULL THEN exif_subsec ELSE ? END, camera = ?, "
                "gps_lat = CASE WHEN ? IS NULL THEN gps_lat ELSE ? END, "
                "gps_lon = CASE WHEN ? IS NULL THEN gps_lon ELSE ? END, "
                "gps_alt = CASE WHEN ? IS NULL THEN gps_alt ELSE ? END, "
                "phash = ?, dhash = ?, sig = ?, frames = ?, uploader_id = ?, faces_status = 'pending', "
                "preview_status = ?, updated_at = ? WHERE id = ?",
                (v, up["original_filename"], info.ext, info.mime, sha, size, info.width, info.height,
                 info.duration_s, info.bitrate, info.video_codec, info.fps,
                 None if info.has_audio is None else int(bool(info.has_audio)),
                 info.taken_at, info.exif_datetime, info.exif_datetime, info.exif_subsec, info.camera,
                 *_gps_update_params(info),
                 fp.phash, fp.dhash, fp.sig, json_dumps(fp.frames) if fp.frames is not None else None,
                 up["uploader_id"], preview_status, now, cur["id"]),
            )
            _add_contributor(conn, cur["id"], up["uploader_id"], now)
            _set_upload(conn, up["id"], "replaced", cur["id"], None, now, sha)
            kv_set(conn, version_mark_key(int(cur["id"])), v)
            _enqueue_followups(conn, int(cur["id"]), v, preview_status)
            # The old original and old thumbnail stay (superseded points at them); the old previews
            # wait in the trash (an undo of this upload brings the old copy back).
            for old in (storage.preview_path(t_uid, old_v), storage.preview_small_path(t_uid, old_v),
                        storage.video_preview_path(t_uid, old_v)):
                trash.discard(conn, old)
    except BaseException:
        _cleanup_unreferenced(conn, written)
        raise
    _cleanup(storage.geotag_files(t_uid, old_v))
    _drop_source(conn, src, dest)
    log.info("ingest: upload %s replaced media %s (now version %d)", up["uid"], t_uid, v)


def _superseded_thumb(info: MediaInfo, path: Path, img: Image.Image | None, thumb: Path) -> bool:
    if info.kind == "image":
        assert img is not None
        images.save_webp(img, thumb, settings.thumb_size, 80)
        return True
    t = video.poster_time(info)
    frame = video.extract_frame(path, t, settings.thumb_size)
    if frame is None and t > 0:
        frame = video.extract_frame(path, 0.0, settings.thumb_size)
    if frame is None:
        return False
    images.save_webp(frame, thumb, settings.thumb_size, 80)
    return True


def _keep_lower_quality(conn: sqlite3.Connection, up: sqlite3.Row, src: Path, info: MediaInfo, sha: str,
                        img: Image.Image | None, size: int, decision: dedup.DedupDecision) -> None:
    target_id = int(decision.target_id or 0)
    dest: Path | None = None
    thumb_rel: str | None = None
    written: list[Path] = []
    try:
        if settings.keep_superseded:
            dest = storage.superseded_path(up["uid"], info.ext)
            if _place(src, dest):
                written.append(dest)
            sthumb = storage.superseded_thumb_path(up["uid"])
            written.append(sthumb)
            if _superseded_thumb(info, dest, img, sthumb):
                thumb_rel = storage.rel(sthumb)
        now = now_iso()
        with tx(conn):
            if conn.execute("SELECT 1 FROM media WHERE id = ?", (target_id,)).fetchone() is None:
                raise RuntimeError("the matched item disappeared, retrying")
            conn.execute(
                "INSERT INTO superseded(media_id, reason, path, thumb_path, sha256, original_filename, ext, mime, "
                "size_bytes, width, height, duration_s, bitrate, uploader_id, upload_id, detail, created_at) "
                "VALUES (?, 'lower_quality', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (target_id, storage.rel(dest) if dest is not None else None, thumb_rel, sha, up["original_filename"],
                 info.ext, info.mime, size, info.width, info.height, info.duration_s, info.bitrate,
                 up["uploader_id"], up["id"], decision.detail, now),
            )
            if settings.keep_superseded:
                _set_known(conn, sha, target_id, "lower_quality", now)
            # else: the file is not kept, so its hash must not block a later upload of it either (a
            # false-positive match would otherwise make a distinct photo impossible to add again).
            _add_contributor(conn, target_id, up["uploader_id"], now)
            _set_upload(conn, up["id"], "lower_quality", target_id, None, now, sha)
    except BaseException:
        _cleanup_unreferenced(conn, written)
        raise
    _drop_source(conn, src, dest)
    log.info("ingest: upload %s kept as a lower-quality copy of media %s", up["uid"], decision.target_uid)


def mark_upload_failed(conn: sqlite3.Connection, upload_id: int, message: str) -> None:
    """Final ingest failure (called by the worker): status 'error' with a short message. The
    upload's bytes are moved to storage.failed_path (a short outage such as a full disk must not
    destroy a finished upload; requeue_failed_upload tries again), and whatever else an
    interrupted run left for this upload is deleted."""
    up = conn.execute("SELECT id, uid, status FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if up is None or up["status"] in TERMINAL:
        return
    with tx(conn):
        cur = conn.execute(
            "UPDATE uploads SET status = 'error', message = ?, updated_at = ? WHERE id = ? AND status IN "
            "('queued', 'processing')",
            (message[:300], now_iso(), upload_id),
        )
    if cur.rowcount != 1:
        return
    src = _locate_upload_file(conn, up["uid"])
    if src is not None:
        try:
            storage.move_into_place(src, storage.failed_path(up["uid"]))
        except OSError as e:
            log.error("upload %s: could not keep the failed file %s (%s); leaving it in place", up["uid"], src, e)
            return
    for cand in _fallback_candidates(up["uid"]):
        if not _is_referenced(conn, cand):
            storage.safe_unlink(cand)
    storage.safe_unlink(storage.superseded_thumb_path(up["uid"]))


def requeue_failed_upload(conn: sqlite3.Connection, upload_id: int) -> bool:
    """Admin retry of an upload whose ingest failed for good: move its kept file back to staging,
    set it to 'queued' and enqueue a new ingest job. False if it is not a failed upload with a kept file."""
    up = conn.execute("SELECT id, uid, status FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if up is None or up["status"] != "error":
        return False
    kept, staged = storage.failed_path(up["uid"]), storage.staging_path(up["uid"])
    if not kept.is_file():
        return False
    try:
        with tx(conn):
            cur = conn.execute(
                "UPDATE uploads SET status = 'queued', message = NULL, updated_at = ? WHERE id = ? AND status = 'error'",
                (now_iso(), upload_id),
            )
            if cur.rowcount != 1:
                return False
            conn.execute("DELETE FROM jobs WHERE type = 'ingest' AND status = 'error' AND payload = ?",
                         (json_dumps({"upload_id": int(upload_id)}),))
            jobs.enqueue(conn, "ingest", {"upload_id": int(upload_id)}, unique=True)
            storage.move_into_place(kept, staged)  # last, so a failure rolls the status back
    except BaseException:
        if staged.is_file() and not kept.exists():
            storage.move_into_place(staged, kept)
        raise
    return True


def discard_failed_upload(conn: sqlite3.Connection, upload_id: int) -> bool:
    """Admin discard of a failed upload's kept file (DATA_DIR/failed is never swept). The upload
    stays 'error' with a note. False if it is not a failed upload with a kept file."""
    up = conn.execute("SELECT id, uid, status FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if up is None or up["status"] != "error":
        return False
    kept = storage.failed_path(up["uid"])
    if not kept.is_file():
        return False
    with tx(conn):
        cur = conn.execute(
            "UPDATE uploads SET message = ?, updated_at = ? WHERE id = ? AND status = 'error'",
            ("Processing failed; the file was discarded by an organiser", now_iso(), upload_id),
        )
        if cur.rowcount != 1:
            return False
        trash.discard(conn, kept)  # kept for an undo, then deleted for good
    return True


# ------------------------------------------------------------------ faces
def _info_from_row(row: sqlite3.Row) -> MediaInfo:
    return MediaInfo(
        kind=row["kind"], mime=row["mime"], ext=row["ext"], width=int(row["width"]), height=int(row["height"]),
        duration_s=row["duration_s"], bitrate=row["bitrate"], video_codec=row["video_codec"], fps=row["fps"],
        has_audio=None if row["has_audio"] is None else bool(row["has_audio"]),
    )


def _encode_crop(img: Image.Image) -> bytes:
    if img.size != (CROP_SIZE, CROP_SIZE):
        img = img.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "WEBP", quality=CROP_QUALITY, method=4)
    return buf.getvalue()


def _drop_empty_people(conn: sqlite3.Connection, person_ids: set[int]) -> int:
    """Delete unnamed, visible people left with no faces, no tags and no linked profiles
    (nothing refers to them)."""
    n = 0
    for pid in sorted(person_ids):
        cur = conn.execute(
            "DELETE FROM persons WHERE id = ? AND name IS NULL AND hidden = 0 "
            "AND NOT EXISTS (SELECT 1 FROM faces WHERE person_id = ?) "
            "AND NOT EXISTS (SELECT 1 FROM media_tags WHERE person_id = ?) "
            "AND NOT EXISTS (SELECT 1 FROM person_links WHERE person_id = ?)",
            (pid, pid, pid, pid),
        )
        n += cur.rowcount
    return n


def process_faces(conn: sqlite3.Connection, media_id: int, version: int, people: bool = False) -> None:
    """'faces' job: detect faces on the current original, insert them with crops, carry over
    previous assignments (replacement/reprocess), auto-assign the rest, set faces_status/face_count.
    The first faces job of a newly stored item (people=True, set only by _store_new's job) on an
    item with no faces yet first takes the people metadata an exported file carries
    (app.export.people.apply_to_new_faces); failures there are logged and ignored. Every other run
    (reprocess, replacement, restore, the startup video re-run) never reads it, so a tag or person
    the admin removed since does not come back."""
    m = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
    if m is None or int(m["version"]) != int(version):
        log.info("faces: media %s version %s is gone or outdated, skipping", media_id, version)
        return
    if m["faces_status"] == "done":
        return
    original = storage.original_path(m["uid"], int(m["version"]), m["ext"])
    records: list[FaceRecord]
    if m["kind"] == "image":
        img = images.load_oriented(original)
        records = analyze_image(img)
        del img
    else:
        records = analyze_video(original, _info_from_row(m))
    crops = [_encode_crop(r.crop) for r in records]
    # People metadata (an exported file uploaded again) only counts for a newly stored item: faces
    # found before were sorted here already and are carried over instead.
    packet: bytes | None = None
    if people and conn.execute("SELECT 1 FROM faces WHERE media_id = ? LIMIT 1", (media_id,)).fetchone() is None:
        try:
            packet = embed.read_packet(original, m["ext"])
        except Exception as e:  # metadata must never fail the faces job
            log.warning("faces: media %s: could not read its people metadata: %s", m["uid"], short_error(e))

    # Bring the index up to date before taking the write lock, so the transaction below only has
    # to refresh this item's faces in it (sync_media) instead of re-reading every face.
    idx = get_index()
    idx.sync(conn)
    written: list[Path] = []
    old_ids: list[int] = []
    summary: str | None = None
    try:
        with tx(conn):
            cur = conn.execute("SELECT version FROM media WHERE id = ?", (media_id,)).fetchone()
            if cur is None or int(cur["version"]) != int(version):
                raise _Stale()
            base_version = faces_version(conn)
            old_faces = conn.execute("SELECT * FROM faces WHERE media_id = ? ORDER BY id", (media_id,)).fetchall()
            old_ids = [int(f["id"]) for f in old_faces]
            old_people = {int(f["person_id"]) for f in old_faces if f["person_id"] is not None}
            now = now_iso()
            new_ids: list[int] = []
            for rec, crop in zip(records, crops):
                emb = rec.embedding.astype("float32", copy=False).tobytes()
                fid = int(conn.execute(
                    "INSERT INTO faces(media_id, person_id, assigned_by, x, y, w, h, det_score, size_px, quality, "
                    "frame_time_s, spans, embedding, created_at) VALUES (?, NULL, 'auto', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (media_id, rec.x, rec.y, rec.w, rec.h, rec.det_score, int(rec.size_px), rec.quality,
                     rec.frame_time_s, json_dumps(rec.spans) if rec.spans is not None else None, emb, now),
                ).lastrowid)
                new_ids.append(fid)
                path = storage.face_crop_path(fid)
                written.append(path)
                storage.write_bytes_atomic(path, crop)
            bump_faces_version(conn)
            if old_faces and new_ids:
                cluster.carry_over(conn, old_faces, new_ids)
            if old_ids:
                _repoint_covers(conn, media_id, old_ids, new_ids, now)
                marks = ",".join("?" * len(old_ids))
                conn.execute(f"DELETE FROM faces WHERE id IN ({marks})", old_ids)
                for fid in old_ids:  # their crops wait in the trash: undoing this run puts them back
                    trash.discard(conn, storage.face_crop_path(fid))
            if packet and not old_faces:
                try:
                    with tx(conn):  # a savepoint: a failure undoes only what the metadata changed
                        summary = people_meta.apply_to_new_faces(conn, m, new_ids, packet)
                except Exception as e:  # metadata must never fail the faces job
                    log.warning("faces: media %s: could not apply its people metadata: %s", m["uid"], short_error(e))
            idx.sync_media(conn, media_id, base_version)
            if new_ids:
                marks = ",".join("?" * len(new_ids))
                remaining = [int(r[0]) for r in conn.execute(
                    f"SELECT id FROM faces WHERE id IN ({marks}) AND person_id IS NULL AND assigned_by = 'auto' "
                    "ORDER BY id", new_ids,
                )]
                if remaining:
                    cluster.assign_new_faces(conn, remaining, exclude_face_ids=set(old_ids))
            if old_people:
                _drop_empty_people(conn, old_people)
            conn.execute(
                "UPDATE media SET faces_status = 'done', face_count = ?, updated_at = ? WHERE id = ?",
                (len(new_ids), now_iso(), media_id),
            )
            bump_faces_version(conn)
    except _Stale:
        _cleanup(written)
        log.info("faces: media %s changed during analysis, skipping", media_id)
        return
    except BaseException:
        # The index may hold this transaction's faces and assignments, now rolled back (and
        # their ids free to be handed out again).
        idx.invalidate()
        _cleanup(written)
        raise
    log.info("faces: media %s v%s -> %d faces%s", m["uid"], version, len(records),
             f", {summary}" if summary else "")


def _repoint_covers(conn: sqlite3.Connection, media_id: int, old_ids: list[int], new_ids: list[int], now: str) -> None:
    """Admin-chosen covers that point at a re-detected face move to its carried-over successor."""
    marks = ",".join("?" * len(old_ids))
    people = conn.execute(f"SELECT id FROM persons WHERE cover_face_id IN ({marks})", old_ids).fetchall()
    for p in people:
        succ = None
        if new_ids:
            nmarks = ",".join("?" * len(new_ids))
            succ = conn.execute(
                f"SELECT id FROM faces WHERE media_id = ? AND person_id = ? AND id IN ({nmarks}) "
                "ORDER BY quality DESC, id LIMIT 1",
                (media_id, p["id"], *new_ids),
            ).fetchone()
        conn.execute("UPDATE persons SET cover_face_id = ?, updated_at = ? WHERE id = ?",
                     (succ["id"] if succ else None, now, p["id"]))


# ------------------------------------------------------------------ video previews
def process_video_preview(conn: sqlite3.Connection, media_id: int, version: int) -> None:
    """'video_preview' job (video queue): transcode the H.264 preview and set preview_status."""
    m = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
    if m is None or int(m["version"]) != int(version) or m["preview_status"] != "pending":
        return
    src = storage.original_path(m["uid"], int(version), m["ext"])
    dst = storage.video_preview_path(m["uid"], int(version))
    video.transcode_preview(src, dst, _info_from_row(m), settings.video_preview_max_height)
    with tx(conn):
        cur = conn.execute(
            "UPDATE media SET preview_status = 'done', updated_at = ? "
            "WHERE id = ? AND version = ? AND preview_status = 'pending'",
            (now_iso(), media_id, int(version)),
        )
        done = cur.rowcount == 1
    if not done:
        storage.safe_unlink(dst)
        log.info("video preview: media %s changed during the transcode, output discarded", media_id)
        return
    log.info("video preview: media %s v%s done", m["uid"], version)


def mark_preview_failed(conn: sqlite3.Connection, media_id: int, version: int) -> None:
    with tx(conn):
        conn.execute(
            "UPDATE media SET preview_status = 'error', updated_at = ? "
            "WHERE id = ? AND version = ? AND preview_status = 'pending'",
            (now_iso(), media_id, int(version)),
        )


def mark_faces_failed(conn: sqlite3.Connection, media_id: int, version: int) -> None:
    with tx(conn):
        conn.execute(
            "UPDATE media SET faces_status = 'error', updated_at = ? WHERE id = ? AND version = ? "
            "AND faces_status = 'pending'",
            (now_iso(), media_id, int(version)),
        )


# ------------------------------------------------------------------ restore / delete / purge
def _restore_uid(sha: str) -> str:
    """Deterministic media uid for restoring the file with this sha (a retried restore reuses it)."""
    n = int.from_bytes(hashlib.sha256(f"restore:{sha}".encode()).digest(), "big")
    chars = []
    for _ in range(14):
        n, r = divmod(n, len(_UID_ALPHABET))
        chars.append(_UID_ALPHABET[r])
    return "".join(chars)


def restore_superseded(conn: sqlite3.Connection, superseded_id: int) -> None:
    """'restore' job: turn a kept near-duplicate file into its own gallery item (no near-dup check),
    then delete the superseded row."""
    s = conn.execute("SELECT * FROM superseded WHERE id = ?", (superseded_id,)).fetchone()
    if s is None:
        return  # restored already
    if s["path"] is None:
        log.info("restore: superseded %s has no file any more (purged), nothing to restore", superseded_id)
        return
    uid = _restore_uid(s["sha256"])
    if conn.execute("SELECT 1 FROM media WHERE uid = ?", (uid,)).fetchone():
        uid = new_uid()
    src = storage.from_rel(s["path"])
    if not src.is_file():
        moved = [p for p in storage.original_path(uid, 1, ".x").parent.glob(f"{uid}.v1.*") if p.is_file()]
        if not moved:
            raise FileNotFoundError(f"kept file for superseded {superseded_id} is missing")
        src = moved[0]
    info = probe(src)
    if info is None:
        raise ValueError("the kept file can no longer be read as a photo or video")
    fp, img = _fingerprint(info, src)
    size = src.stat().st_size
    uploader_id = s["uploader_id"]
    if uploader_id is None and s["upload_id"] is not None:
        r = conn.execute("SELECT uploader_id FROM uploads WHERE id = ?", (s["upload_id"],)).fetchone()
        uploader_id = r["uploader_id"] if r else None
    if uploader_id is None and s["media_id"] is not None:
        r = conn.execute("SELECT uploader_id FROM media WHERE id = ?", (s["media_id"],)).fetchone()
        uploader_id = r["uploader_id"] if r else None
    if uploader_id is None:
        raise RuntimeError("no uploader known for this file")

    dest = storage.original_path(uid, 1, info.ext)
    thumb, preview = storage.thumb_path(uid, 1), storage.preview_path(uid, 1)
    preview_small = storage.preview_small_path(uid, 1)
    written: list[Path] = []
    try:
        if _place(src, dest):
            written.append(dest)
        written += [thumb, preview, preview_small]
        _write_derivatives(info, dest, img, thumb, preview, preview_small)
        preview_status = _preview_status(info)
        now = now_iso()
        with tx(conn):
            cur = conn.execute("SELECT path FROM superseded WHERE id = ?", (superseded_id,)).fetchone()
            if cur is None or cur["path"] != s["path"]:
                raise _Stale()
            media_id = _insert_media(conn, uid, info, s["sha256"], fp, size, s["original_filename"], uploader_id,
                                     preview_status, now)
            _add_contributor(conn, media_id, uploader_id, now)
            _move_contributors(conn, s, media_id, now)
            _set_known(conn, s["sha256"], media_id, "current", now)
            conn.execute("DELETE FROM superseded WHERE id = ?", (superseded_id,))
            _enqueue_followups(conn, media_id, 1, preview_status)
            # The kept file (now also linked as the new item's original) and its thumbnail wait in
            # the trash, so undoing the restore can bring the kept copy back.
            if not _same_path(src, dest) and not _is_referenced(conn, src):
                trash.discard(conn, src)
            if s["thumb_path"]:
                try:
                    trash.discard(conn, storage.from_rel(s["thumb_path"]))
                except ValueError:
                    pass
    except _Stale:
        _cleanup_unreferenced(conn, written)
        log.info("restore: superseded %s changed meanwhile, skipping", superseded_id)
        return
    except BaseException:
        _cleanup_unreferenced(conn, written)
        raise
    log.info("restore: superseded %s restored as media %s", superseded_id, uid)


_CREDITING = ("stored", "replaced", "duplicate", "lower_quality")


def _move_contributors(conn: sqlite3.Connection, s: sqlite3.Row, new_id: int, now: str) -> None:
    """Restore: everyone who uploaded the restored file (exact copies included) is credited on the
    new item, and loses the credit on the old item unless they also uploaded another file of it."""
    marks = ",".join("?" * len(_CREDITING))
    rows = conn.execute(
        f"SELECT uploader_id, MIN(created_at) AS first FROM uploads WHERE sha256 = ? AND status IN ({marks}) "
        "AND removed = 0 AND uploader_id IS NOT NULL GROUP BY uploader_id",
        (s["sha256"], *_CREDITING),
    ).fetchall()
    movers = {int(r["uploader_id"]): r["first"] for r in rows}
    if s["uploader_id"] is not None:
        movers.setdefault(int(s["uploader_id"]), now)
    if s["upload_id"] is not None:
        r = conn.execute("SELECT uploader_id, created_at FROM uploads WHERE id = ?", (s["upload_id"],)).fetchone()
        if r is not None and r["uploader_id"] is not None:
            movers.setdefault(int(r["uploader_id"]), r["created_at"])
    old_id = s["media_id"]
    for uploader_id, first in movers.items():
        cur = conn.execute(
            "INSERT OR IGNORE INTO media_contributors(media_id, uploader_id, first_upload_at) VALUES (?, ?, ?)",
            (new_id, uploader_id, first or now),
        )
        if cur.rowcount == 0:
            claim(conn, "media_contributors", {"media_id": new_id, "uploader_id": uploader_id}, new_id)
        if old_id is None:
            continue
        still = conn.execute(
            f"SELECT 1 FROM uploads WHERE media_id = ? AND uploader_id = ? AND status IN ({marks}) "
            "AND removed = 0 AND (sha256 IS NULL OR sha256 <> ?) LIMIT 1",
            (old_id, uploader_id, *_CREDITING, s["sha256"]),
        ).fetchone()
        owner = conn.execute("SELECT 1 FROM media WHERE id = ? AND uploader_id = ?", (old_id, uploader_id)).fetchone()
        if still is None and owner is None:
            conn.execute("DELETE FROM media_contributors WHERE media_id = ? AND uploader_id = ?", (old_id, uploader_id))


def _media_files(uid: str) -> list[Path]:
    out: list[Path] = []
    for path in (storage.original_path(uid, 1, ".x"), storage.thumb_path(uid, 1), storage.preview_path(uid, 1),
                 storage.video_preview_path(uid, 1)):
        d = path.parent
        if d.is_dir():
            out.extend(p for p in d.glob(f"{uid}.v*") if p.is_file())
    out.extend(storage.geotag_files(uid))
    return out


def delete_media(conn: sqlite3.Connection, media_id: int) -> None:
    """Admin delete: remove the item's rows (faces, tags, contributors, known_hashes, superseded)
    and move all its files (every version, previews, face crops, kept copies) to the trash of the
    change, in one transaction, so the change history can undo it (geotag renders are caches and
    are just deleted). Bumps faces_version. Uploads pointing at it keep their outcome, marked
    removed (media_id -> NULL)."""
    with tx(conn):
        m = conn.execute("SELECT id, uid FROM media WHERE id = ?", (media_id,)).fetchone()
        if m is None:
            return
        faces = conn.execute("SELECT id, person_id FROM faces WHERE media_id = ?", (media_id,)).fetchall()
        sup = conn.execute("SELECT path, thumb_path FROM superseded WHERE media_id = ?", (media_id,)).fetchall()
        people = {int(f["person_id"]) for f in faces if f["person_id"] is not None}
        conn.execute("DELETE FROM faces WHERE media_id = ?", (media_id,))
        conn.execute("DELETE FROM media_tags WHERE media_id = ?", (media_id,))
        conn.execute("DELETE FROM media_contributors WHERE media_id = ?", (media_id,))
        conn.execute("DELETE FROM known_hashes WHERE media_id = ?", (media_id,))
        conn.execute("DELETE FROM superseded WHERE media_id = ?", (media_id,))
        conn.execute("UPDATE uploads SET media_id = NULL, removed = 1 WHERE media_id = ?", (media_id,))
        conn.execute("DELETE FROM media WHERE id = ?", (media_id,))
        # The newest id must stay taken: an undo puts this item back under it.
        mark = int(kv_get(conn, MEDIA_ID_MARK, "0") or 0)
        if media_id > mark:
            kv_set(conn, MEDIA_ID_MARK, media_id)
        if people:
            _drop_empty_people(conn, people)
        bump_faces_version(conn)
        files = [p for p in _media_files(m["uid"]) if "geotag" not in p.relative_to(storage.root()).parts[:1]]
        files += [storage.face_crop_path(int(f["id"])) for f in faces]
        for r in sup:
            for relpath in (r["path"], r["thumb_path"]):
                if relpath:
                    try:
                        files.append(storage.from_rel(relpath))
                    except ValueError:
                        log.warning("delete: superseded path outside the data dir: %s", relpath)
        moved = 0
        for path in dict.fromkeys(files):  # a replaced original is listed twice (versions and superseded)
            moved += trash.discard(conn, path)
    _cleanup(storage.geotag_files(m["uid"]))
    log.info("deleted media %s (%d files to the trash)", m["uid"], moved)


def purge_superseded(conn: sqlite3.Connection, superseded_id: int) -> None:
    """Delete a kept near-duplicate file and its thumbnail; keep the row (path=NULL) as a record.
    Its known_hashes entry goes too: with the file gone, a later upload of that exact file is
    evaluated again instead of being answered 'duplicate' (if the match was a false positive, the
    photo could otherwise never be added)."""
    with tx(conn):
        s = conn.execute("SELECT path, thumb_path, sha256, media_id FROM superseded WHERE id = ?",
                         (superseded_id,)).fetchone()
        if s is None:
            return
        conn.execute("UPDATE superseded SET path = NULL, thumb_path = NULL WHERE id = ?", (superseded_id,))
        conn.execute(
            "DELETE FROM known_hashes WHERE sha256 = ? AND media_id = ? AND relation IN ('lower_quality', 'replaced') "
            "AND NOT EXISTS (SELECT 1 FROM superseded WHERE sha256 = ? AND path IS NOT NULL AND id <> ?)",
            (s["sha256"], s["media_id"], s["sha256"], superseded_id),
        )
        # The files wait in the change's trash (an undo puts them back), unless a current item
        # still uses them (should not happen).
        for relpath in dict.fromkeys(p for p in (s["path"], s["thumb_path"]) if p):
            try:
                path = storage.from_rel(relpath)
            except ValueError:
                log.warning("purge: superseded path outside the data dir: %s", relpath)
                continue
            if _is_referenced(conn, path):
                continue
            trash.discard(conn, path)


# ------------------------------------------------------------------ housekeeping
def cleanup_stale_uploads(conn: sqlite3.Connection) -> int:
    """Cancel 'uploading' rows idle longer than settings.staging_ttl_hours and delete their staged
    files; delete staged files that have no upload row. Returns the number cleaned."""
    cleaned = 0
    cutoff = now_iso(-settings.staging_ttl_hours * 3600)
    stale = conn.execute(
        "SELECT id, uid FROM uploads WHERE status = 'uploading' AND updated_at < ?", (cutoff,)
    ).fetchall()
    for r in stale:
        with tx(conn):
            cur = conn.execute(
                "UPDATE uploads SET status = 'cancelled', message = ?, updated_at = ? "
                "WHERE id = ? AND status = 'uploading' AND updated_at < ?",
                ("The upload was not finished in time and was discarded.", now_iso(), r["id"], cutoff),
            )
        if cur.rowcount == 1:
            storage.safe_unlink(storage.staging_path(r["uid"]))
            cleaned += 1
    staging = storage.root() / "staging"
    if staging.is_dir():
        now_s = time.time()
        for p in staging.iterdir():
            if not p.is_file() or not p.name.endswith(".part"):
                continue
            row = conn.execute("SELECT status FROM uploads WHERE uid = ?", (p.name[:-5],)).fetchone()
            if row is None:
                try:
                    old = now_s - p.stat().st_mtime > ORPHAN_STAGING_AGE_S
                except FileNotFoundError:
                    continue
                if not old:
                    continue  # the create endpoint may not have committed its row yet
            elif row["status"] in ACTIVE:
                continue
            storage.safe_unlink(p)
            cleaned += 1
    cleaned += sweep_orphan_files(conn)
    if cleaned:
        log.info("cleanup: removed %d stale uploads / staged files", cleaned)
    return cleaned


def sweep_orphan_files(conn: sqlite3.Connection, min_age_s: float = ORPHAN_STAGING_AGE_S) -> int:
    """Delete files under originals/, thumbs/, previews/, videos/, superseded/, faces/ and geotag/
    (renders of a version that is no longer current) that no row references and that are older than min_age_s: a crash between a commit and its file
    cleanup (delete, restore, purge, replacement, re-detected faces) leaves them behind for good.
    Files of uploads still being ingested are kept (an interrupted run may hold the only copy).
    Runs on the main worker thread between jobs, so no ingest is writing meanwhile."""
    root = storage.root()
    media = {(r[0], int(r[1])): r[2] for r in conn.execute("SELECT uid, version, ext FROM media")}
    sup_refs = {r[0] for r in conn.execute(
        "SELECT path FROM superseded WHERE path IS NOT NULL UNION SELECT thumb_path FROM superseded "
        "WHERE thumb_path IS NOT NULL")}
    face_ids = {int(r[0]) for r in conn.execute("SELECT id FROM faces")}
    marks = ",".join("?" * len(ACTIVE))
    active = {r[0] for r in conn.execute(f"SELECT uid FROM uploads WHERE status IN ({marks})", tuple(ACTIVE))}
    cutoff = time.time() - min_age_s
    removed = 0
    for top in _SWEPT_DIRS:
        base = root / top
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            try:
                if not p.is_file() or p.is_symlink() or p.stat().st_mtime > cutoff:
                    continue
            except FileNotFoundError:
                continue
            name = p.name
            relpath = str(p.relative_to(root))
            if relpath in sup_refs:
                continue
            if top == "faces":
                fm = _FACE_NAME.match(name)
                if fm and int(fm["id"]) in face_ids:
                    continue
            elif top == "superseded":
                if name.split(".", 1)[0] in active:
                    continue
            elif top == "geotag":
                gm = _GEOTAG_NAME.match(name)
                if gm and (gm["uid"], int(gm["version"])) in media:
                    continue
            else:
                m = (_ORIGINAL_NAME if top == "originals" else _DERIVED_NAME).match(name)
                if m:
                    if m["uid"] in active:
                        continue
                    ext = media.get((m["uid"], int(m["version"])))
                    if ext is not None and (top != "originals" or ext == m["ext"]):
                        continue
            log.info("cleanup: removing orphaned file %s", relpath)
            storage.safe_unlink(p)
            removed += 1
    return removed


def _normalize_stored_taken_at(conn: sqlite3.Connection) -> int:
    """One-time upgrade: capture times stored before they were normalised to UTC (naive local
    EXIF times, '+05:30' offsets) are converted, so sort=taken compares like with like."""
    rows = conn.execute(
        "SELECT id, taken_at FROM media WHERE taken_at IS NOT NULL AND taken_at NOT GLOB "
        "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9]Z'"
    ).fetchall()
    fixed = [(v, r["id"]) for r in rows if (v := normalize_taken_at(r["taken_at"])) is not None]
    if not fixed:
        return 0
    with tx(conn):
        conn.executemany("UPDATE media SET taken_at = ? WHERE id = ?", fixed)
    return len(fixed)


def _backfill_streaming_copies(conn: sqlite3.Connection) -> int:
    """Videos stored without a streaming copy (preview_status 'none') whose original is not
    stream_friendly under the current rules -> 'pending', so the loop in recover_on_startup queues
    their video_preview job. Items with 'error' keep it (the transcode already failed for good)."""
    if not settings.video_previews:
        return 0
    rows = conn.execute(
        "SELECT id, version, video_codec, ext, width, height, bitrate FROM media "
        "WHERE kind = 'video' AND preview_status = 'none' ORDER BY id"
    ).fetchall()
    heavy = [(r["id"], r["version"]) for r in rows
             if not video.stream_friendly(r["video_codec"], r["ext"], r["width"], r["height"], r["bitrate"])]
    if not heavy:
        return 0
    now = now_iso()
    with tx(conn):
        conn.executemany(
            "UPDATE media SET preview_status = 'pending', updated_at = ? "
            "WHERE id = ? AND version = ? AND preview_status = 'none'",
            [(now, mid, ver) for mid, ver in heavy],
        )
    return len(heavy)


# ------------------------------------------------------------------ capture positions (GPS)
def gps_backfill_pending(conn: sqlite3.Connection) -> bool:
    """True while items stored before positions were read still wait for backfill_gps_batch."""
    end = kv_get(conn, GPS_BACKFILL_END)
    if end is None:
        return False
    try:
        return int(kv_get(conn, GPS_BACKFILL_CURSOR, "0") or 0) < int(end)
    except ValueError:
        return False


def backfill_gps_batch(conn: sqlite3.Connection, limit: int = GPS_BACKFILL_BATCH) -> int | None:
    """Read the capture position of up to `limit` items stored before positions were read (ids up
    to kv gps_backfill_end, after the kv gps_backfill_cursor) and store the ones found. Files are
    read with no transaction open; the results and the new cursor are written in one short
    transaction, and only into rows still at the version that was read and still without a position.
    Returns the number of items looked at, 0 when ffprobe could not run (try again later; the cursor
    stays), or None when nothing is left."""
    end_raw = kv_get(conn, GPS_BACKFILL_END)
    if end_raw is None:
        return None
    try:
        end = int(end_raw)
        cursor = int(kv_get(conn, GPS_BACKFILL_CURSOR, "0") or 0)
    except ValueError:
        return None
    if cursor >= end:
        return None
    rows = conn.execute(
        "SELECT id, uid, version, ext FROM media WHERE id > ? AND id <= ? AND gps_lat IS NULL ORDER BY id LIMIT ?",
        (cursor, end, max(1, int(limit))),
    ).fetchall()
    if not rows:
        with tx(conn):
            kv_set(conn, GPS_BACKFILL_CURSOR, end)
        log.info("gps backfill: done")
        return None
    found: list[tuple[float, float, float | None, int, int]] = []
    last = cursor
    looked = 0
    for r in rows:
        path = storage.original_path(r["uid"], int(r["version"]), r["ext"])
        try:
            gps = read_gps(path) if path.is_file() else None
        except ToolFailure as e:
            log.warning("gps backfill: ffprobe could not run (%s); trying again later", e)
            break
        except Exception as e:  # one unreadable file must not stop the backfill
            log.info("gps backfill: could not read %s: %s", path.name, e)
            gps = None
        if gps is not None:
            found.append((gps[0], gps[1], gps[2], int(r["id"]), int(r["version"])))
        last = int(r["id"])
        looked += 1
    if looked == 0:
        return 0
    if len(rows) < max(1, int(limit)) and looked == len(rows):
        last = end  # nothing else up to the end mark is waiting
    with tx(conn):
        conn.executemany(
            "UPDATE media SET gps_lat = ?, gps_lon = ?, gps_alt = ? WHERE id = ? AND version = ? AND gps_lat IS NULL",
            found,
        )
        kv_set(conn, GPS_BACKFILL_CURSOR, last)
    if found:
        log.info("gps backfill: %d of %d items have a capture position", len(found), looked)
    return looked


GEOTAG_SIGNATURE = "geotag_cache_signature"


def _geotag_signature() -> str:
    """What a cached geotag render depends on besides the photo: when it changes (settings, or the
    renderer's own RENDER_VERSION), the cached renders are stale."""
    try:
        from app.geo import overlay

        render_version = str(getattr(overlay, "RENDER_VERSION", ""))
    except Exception:  # the geo package is optional at startup
        render_version = ""
    parts = [
        settings.geotag_map_km, settings.geotag_place_name, settings.geotag_place_lat, settings.geotag_place_lon,
        settings.geotag_place_radius_m, settings.geotag_jpeg_quality, settings.event_timezone, render_version,
    ]
    return json_dumps([str(x) for x in parts])


def _clear_stale_geotag_cache(conn: sqlite3.Connection) -> bool:
    """Remove every cached geotag render when _geotag_signature changed since the last start."""
    sig = _geotag_signature()
    old = kv_get(conn, GEOTAG_SIGNATURE)
    if old == sig:
        return False
    base = storage.root() / "geotag"
    if old is not None and base.is_dir() and not base.is_symlink():
        for d in base.iterdir():
            if d.is_dir() and not d.is_symlink():
                shutil.rmtree(d, ignore_errors=True)
            else:
                storage.safe_unlink(d)
    with tx(conn):
        kv_set(conn, GEOTAG_SIGNATURE, sig)
    return old is not None


VIDEO_FACE_TIMES_FLAG = "video_face_times_v2"
VIDEO_FACE_SPANS_FLAG = "video_face_spans_v3"
# One-time upgrades that find faces again in every analysed video: kv flag -> what it brings.
VIDEO_FACE_RERUNS = (
    # face times of videos analysed before frames were sampled at their exact presentation times
    # could point one frame early
    (VIDEO_FACE_TIMES_FLAG, "face times are frame exact"),
    # faces stored before tracks kept every frame they were detected in have no spans
    (VIDEO_FACE_SPANS_FLAG, "the viewer knows who is on screen when"),
)


def _rerun_video_faces_once(conn: sqlite3.Connection) -> tuple[int, list[str]]:
    """One-time upgrades (VIDEO_FACE_RERUNS): when a flag is missing, every video whose faces are
    done goes back to faces_status 'pending' (the loop in recover_on_startup then queues one faces
    job each; carry_over keeps names and admin decisions). The missing flags are set in the same
    transaction, so each upgrade runs once whatever happens to those jobs, and several missing
    flags share one run. Photos are untouched. Returns (videos sent back, what the run brings)."""
    if all(kv_get(conn, flag) is not None for flag, _ in VIDEO_FACE_RERUNS):
        return 0, []
    with tx(conn):
        missing = [(flag, why) for flag, why in VIDEO_FACE_RERUNS if kv_get(conn, flag) is None]
        if not missing:
            return 0, []
        cur = conn.execute(
            "UPDATE media SET faces_status = 'pending', updated_at = ? WHERE kind = 'video' AND faces_status = 'done'",
            (now_iso(),),
        )
        for flag, _ in missing:
            kv_set(conn, flag, now_iso())
    return int(cur.rowcount or 0), [why for _, why in missing]


def recover_on_startup(conn: sqlite3.Connection) -> None:
    """Re-queue running jobs (app.jobs.requeue_running); make sure every 'queued'/'processing'
    upload has an ingest job, every media with faces_status='pending' has a faces job and every
    preview_status='pending' video has a video_preview job; clear data_dir/tmp; convert capture
    times stored before UTC normalisation; drop cached geotag renders when the geotag settings
    changed; once per VIDEO_FACE_RERUNS flag send every video with faces done back to 'pending' so
    its faces are found again (exact frame times, on-screen spans). Capture positions of older items are read later by the worker (backfill_gps_batch). Videos stored as 'none' that are not stream_friendly
    under the current rules become 'pending' first (backfill of compressed streaming copies).
    Missing small previews are not regenerated here: GET /files/preview-small makes them lazily."""
    fixed = _normalize_stored_taken_at(conn)
    if fixed:
        log.info("startup: converted %d stored capture times to UTC", fixed)
    if _clear_stale_geotag_cache(conn):
        log.info("startup: geotag settings changed, cleared the cached geotag renders")
    n = jobs.requeue_running(conn)
    if n:
        log.info("startup: re-queued %d interrupted jobs", n)
    rerun, why = _rerun_video_faces_once(conn)
    if rerun:
        log.info("startup: finding faces again in %d videos so that %s", rerun, " and ".join(why))
    added = 0
    for r in conn.execute("SELECT id FROM uploads WHERE status IN ('queued', 'processing') ORDER BY id").fetchall():
        added += jobs.enqueue(conn, "ingest", {"upload_id": int(r["id"])}, unique=True) is not None
    for r in conn.execute("SELECT id, version FROM media WHERE faces_status = 'pending' ORDER BY id").fetchall():
        added += jobs.enqueue(conn, "faces", {"media_id": int(r["id"]), "version": int(r["version"])},
                              unique=True) is not None
    backfilled = _backfill_streaming_copies(conn)
    if backfilled:
        log.info("startup: %d videos stream too heavy as they are and get a compressed copy", backfilled)
    for r in conn.execute(
        "SELECT id, version FROM media WHERE kind = 'video' AND preview_status = 'pending' ORDER BY id"
    ).fetchall():
        added += jobs.enqueue(conn, "video_preview", {"media_id": int(r["id"]), "version": int(r["version"])},
                              unique=True) is not None
    if added:
        log.info("startup: queued %d missing jobs", added)
    tmp = storage.root() / "tmp"
    if tmp.is_dir():
        for p in tmp.iterdir():
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
            except FileNotFoundError:
                pass
