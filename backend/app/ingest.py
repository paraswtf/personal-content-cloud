"""Turning completed uploads into items, and the item lifecycle (trash, restore, purge).

CONTRACT (docs/ARCHITECTURE.md sections 3, 6 and 7).
- process_upload(conn, upload_id): the 'ingest' job. Probe (any non-empty file is kept; unknown or
  undecodable files become kind 'other'), sha256, exact duplicate (same sha256 as a live or trashed
  item: upload 'duplicate' pointing at it), fingerprint, near-duplicate link (media.dedup: the new
  item joins the match's group, nothing is replaced), place the original (hard link), derivatives
  (thumb + previews; page 1 render for documents; poster for videos), insert the item, enqueue the
  follow-up jobs, upload 'stored'. Statuses at insert: analysis_status 'pending' (images, videos,
  documents; 'done' for 'other', which has nothing to analyse), text_status 'pending' for pdf and
  document else 'none', preview_status 'pending' for videos that need a streaming copy else 'none'.
  Jobs: analyze {item_id, version}; text {item_id, version} (documents only; images get text from
  the analyze stage); video_preview {item_id, version}.
- process_video_preview(conn, item_id, version): the 'video_preview' job.
- soft_delete / restore (trash, 30 days), purge (rows and every file of the item), expire_trash.
- enqueue_reprocess(conn, item_id, stages), enqueue_text(conn, item_id, version).
- cleanup_stale_uploads, sweep_orphan_files, recover_on_startup, mark_*_failed (worker).
Every handler is idempotent: it checks DB state first (terminal upload -> no-op, version mismatch ->
no-op), writes files to deterministic paths (a new item reuses the upload uid) and switches the DB
over in one short transaction; heavy work (decode, ffmpeg) happens before it. The staged file is
hard-linked to its destination and only unlinked after the commit, so a retried attempt can never
lose the upload's bytes.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

from PIL import Image

from app import jobs, storage
from app.config import settings
from app.db import bump_faces_version, bump_search_version, kv_get, kv_set, next_item_id, tx
from app.media import dedup, images, video
from app.media.hashing import Fingerprint, image_fingerprint, sha256_file, video_fingerprint
from app.media.probe import MediaInfo, ToolFailure, other_info, probe
from app.util import json_dumps, now_iso

log = logging.getLogger(__name__)

TERMINAL = frozenset({"stored", "duplicate", "rejected", "error", "cancelled"})
ACTIVE = frozenset({"uploading", "queued", "processing"})
ORPHAN_AGE_S = 3600  # unreferenced files and staged files without an upload row go after this long
ANALYSED_KINDS = ("image", "video", "pdf", "document")
TEXT_KINDS = ("pdf", "document")
STAGES = ("analyze", "text", "preview")

# File names under the swept directories (storage.SWEPT_DIRS).
_ORIGINAL_NAME = re.compile(r"^(?P<uid>[a-z0-9]+)\.v(?P<version>\d+)(?P<ext>\.[a-z0-9]+)$")
# thumbs, previews (large and .sm), videos, docphoto: <uid>.v<n>.webp | .sm.webp | .mp4
_DERIVED_NAME = re.compile(r"^(?P<uid>[a-z0-9]+)\.v(?P<version>\d+)\.(?:sm\.webp|webp|mp4)$")
_PAGES_DIR_NAME = re.compile(r"^(?P<uid>[a-z0-9]+)\.v(?P<version>\d+)$")
_PAGE_NAME = re.compile(r"^(?P<page>\d+)\.webp$")
_CROP_NAME = re.compile(r"^(?P<id>\d+)\.webp$")  # faces/<id//1000>/<id>.webp, bodies/...


class _Stale(Exception):
    """The DB changed under a running job: give up this attempt quietly."""


# ------------------------------------------------------------------ payloads
def item_payload(item_id: int, version: int) -> dict[str, int]:
    """Payload of analyze, text and video_preview jobs (always built here: unique enqueue compares
    the JSON text, so the key order must never vary)."""
    return {"item_id": int(item_id), "version": int(version)}


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
    """True if a committed row points at this file: an item's original or a derivative of a stored
    version (thumb, previews, page render, streaming copy, ID photo), or a face or body crop."""
    try:
        relpath = storage.rel(path)
    except ValueError:
        return False
    parts = relpath.split("/")
    top, name = parts[0], parts[-1]
    if top == "originals":
        m = _ORIGINAL_NAME.match(name)
        return bool(m and conn.execute(
            "SELECT 1 FROM items WHERE uid = ? AND version = ? AND ext = ?",
            (m["uid"], int(m["version"]), m["ext"]),
        ).fetchone())
    if top in ("thumbs", "previews", "videos", "docphoto"):
        m = _DERIVED_NAME.match(name)
        return bool(m and conn.execute(
            "SELECT 1 FROM items WHERE uid = ? AND version = ?", (m["uid"], int(m["version"]))
        ).fetchone())
    if top == "pages" and len(parts) == 3:  # a version's page directory
        m = _PAGES_DIR_NAME.match(name)
        return bool(m and conn.execute(
            "SELECT 1 FROM items WHERE uid = ? AND version = ?", (m["uid"], int(m["version"]))
        ).fetchone())
    if top == "pages" and len(parts) >= 2:
        m = _PAGES_DIR_NAME.match(parts[-2])
        return bool(m and _PAGE_NAME.match(name) and conn.execute(
            "SELECT 1 FROM items WHERE uid = ? AND version = ?", (m["uid"], int(m["version"]))
        ).fetchone())
    if top in ("faces", "bodies"):
        m = _CROP_NAME.match(name)
        table = "faces" if top == "faces" else "bodies"
        return bool(m and conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (int(m["id"]),)).fetchone())
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
    orig_dir = storage.original_path(upload_uid, 1, ".x").parent
    if not orig_dir.is_dir():
        return []
    return sorted(p for p in orig_dir.glob(f"{upload_uid}.v1.*") if p.is_file())


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


def _cleanup_unreferenced(conn: sqlite3.Connection, paths: list[Path]) -> None:
    """Undo a failed attempt's files, except any a committed row points at. When the check itself
    fails the file is kept; the orphan sweep removes it later if nothing uses it."""
    for p in paths:
        try:
            if _is_referenced(conn, p):
                continue
        except sqlite3.Error as e:
            log.warning("not deleting %s after a failed attempt: could not check the database (%s)", p, e)
            continue
        if p.is_dir():
            storage.remove_tree(p)
        else:
            storage.safe_unlink(p)


def _set_upload(conn: sqlite3.Connection, upload_id: int, status: str, item_id: int | None,
                message: str | None, now: str, sha: str | None = None) -> None:
    conn.execute(
        "UPDATE uploads SET status = ?, item_id = ?, message = ?, updated_at = ?, sha256 = COALESCE(?, sha256) "
        "WHERE id = ?",
        (status, item_id, message, now, sha, upload_id),
    )


def _refresh_fts(conn: sqlite3.Connection, item_id: int) -> None:
    """Index the item's file name (and whatever else documents.extract knows) right away."""
    try:
        from app.documents.extract import refresh_fts
    except ImportError:
        return
    try:
        refresh_fts(conn, item_id)
    except Exception:  # search indexing must never fail an upload
        log.exception("could not index item %s for full-text search", item_id)


def _remove_fts(conn: sqlite3.Connection, item_id: int) -> None:
    try:
        from app.documents.extract import remove_fts
    except ImportError:
        conn.execute("DELETE FROM items_fts WHERE rowid = ?", (item_id,))
        return
    remove_fts(conn, item_id)


# ------------------------------------------------------------------ decoding and derivatives
def _decode(info: MediaInfo, path: Path, filename: str | None) -> tuple[MediaInfo, Fingerprint | None, Image.Image | None]:
    """Fingerprint (and, for photos, the decoded image used for the derivatives). A photo or video
    that probed but does not decode becomes 'other' (kept without previews)."""
    try:
        if info.kind == "image":
            img = images.load_oriented(path, raw=info.raw)
            return info, image_fingerprint(img), img
        if info.kind == "video":
            return info, video_fingerprint(path, info), None
    except ToolFailure:
        raise
    except ValueError as e:
        log.info("ingest: %s did not decode (%s); keeping it without previews", path.name, e)
        return other_info(filename, (info.kind, info.mime, info.ext), reason=short_error(e)), None, None
    return info, None, None


def _write_derivatives(info: MediaInfo, uid: str, version: int, original: Path,
                       img: Image.Image | None) -> tuple[list[Path], Fingerprint | None]:
    """Thumb and previews; for documents also the page renders (up to settings.page_render_max, so
    the text stage finds every page it scans on disk). Returns the files written and, for
    documents, the fingerprint of page 1. A derivative that cannot be made (a damaged file the
    probe still accepted) is logged and skipped: the item is kept without it. ffmpeg not running and
    a full disk are raised so the job is retried."""
    thumb, preview = storage.thumb_path(uid, version), storage.preview_path(uid, version)
    small = storage.preview_small_path(uid, version)
    written = [thumb, preview, small]
    fp: Fingerprint | None = None
    try:
        if info.kind == "image" and img is not None:
            images.make_image_derivatives(img, thumb, preview, small)
        elif info.kind == "video":
            video.make_video_derivatives(original, info, thumb, preview, small)
        elif info.kind in TEXT_KINDS:
            from app.media import documents

            page = documents.page_image(original, info.kind, 1, max_side=settings.preview_size, ext=info.ext)
            if page is not None:
                images.make_image_derivatives(page, thumb, preview, small)
                fp = image_fingerprint(page)
                del page
            written.append(storage.pages_dir(uid, version))
            documents.render_pages(original, info.kind, uid, version, max_side=settings.page_render_size,
                                   max_pages=settings.page_render_max, ext=info.ext)
        else:
            return [], None
    except ToolFailure:
        raise
    except Exception as e:
        from app.worker import _is_disk_full

        if _is_disk_full(e):
            raise
        log.warning("ingest: previews of %s are incomplete (%s)", uid, short_error(e))
    return written, fp


def _insert_item(conn: sqlite3.Connection, uid: str, info: MediaInfo, sha: str, fp: Fingerprint | None,
                 size: int, filename: str, now: str) -> tuple[int, dict[str, str]]:
    item_id = next_item_id(conn)
    statuses = {
        "analysis_status": "pending" if info.kind in ANALYSED_KINDS else "done",
        "text_status": "pending" if info.kind in TEXT_KINDS else "none",
        "preview_status": _preview_status(info),
    }
    cols = dict(
        id=item_id, uid=uid, kind=info.kind, version=1, original_filename=filename, ext=info.ext, mime=info.mime,
        sha256=sha, size_bytes=size, width=info.width, height=info.height, duration_s=info.duration_s,
        bitrate=info.bitrate, video_codec=info.video_codec, fps=info.fps,
        has_audio=None if info.has_audio is None else int(bool(info.has_audio)), page_count=info.page_count,
        animated=int(bool(info.animated)), taken_at=info.taken_at, exif_datetime=info.exif_datetime,
        exif_subsec=info.exif_subsec, camera=info.camera, gps_lat=info.gps_lat, gps_lon=info.gps_lon,
        gps_alt=info.gps_alt, phash=fp.phash if fp else None, dhash=fp.dhash if fp else None,
        sig=fp.sig if fp else None, frames=json_dumps(fp.frames) if fp and fp.frames is not None else None,
        category="documents" if info.kind in TEXT_KINDS else "misc", **statuses, created_at=now, updated_at=now,
    )
    conn.execute(f"INSERT INTO items({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", tuple(cols.values()))
    return item_id, statuses


def _enqueue_followups(conn: sqlite3.Connection, item_id: int, version: int, kind: str,
                       statuses: dict[str, str]) -> None:
    payload = item_payload(item_id, version)
    if statuses["analysis_status"] == "pending":
        jobs.enqueue(conn, "analyze", payload, unique=True)
    if statuses["text_status"] == "pending":
        jobs.enqueue(conn, "text", payload, unique=True)
    if statuses["preview_status"] == "pending":
        jobs.enqueue(conn, "video_preview", payload, unique=True)


# ------------------------------------------------------------------ ingest
def process_upload(conn: sqlite3.Connection, upload_id: int) -> None:
    """'ingest' job: probe the staged file, answer exact duplicates, link near-duplicates, store
    the item and record the outcome on the upload row."""
    up = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if up is None:
        log.warning("ingest: upload %s does not exist", upload_id)
        return
    uid = up["uid"]
    if up["status"] in TERMINAL:
        # Done already (maybe a crash after the commit): only the staged name may be left over.
        if not _is_referenced(conn, storage.staging_path(uid)):
            storage.safe_unlink(storage.staging_path(uid))
        return
    if up["status"] == "uploading":
        log.warning("ingest: upload %s is still uploading, nothing to do", uid)
        return
    with tx(conn):
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

    info = probe(src, up["original_filename"])  # ToolFailure propagates: the job is retried
    if info is None:
        with tx(conn):
            _set_upload(conn, upload_id, "rejected", None, "The file is empty.", now_iso())
        _drop_source(conn, src)
        return

    sha = up["sha256"] or sha256_file(src)
    known = conn.execute("SELECT id, deleted_at FROM items WHERE sha256 = ?", (sha,)).fetchone()
    if known is not None:
        message = ("Already in the library, in the trash (identical file)." if known["deleted_at"]
                   else None)
        with tx(conn):
            _set_upload(conn, upload_id, "duplicate", int(known["id"]), message, now_iso(), sha)
        _drop_source(conn, src)
        log.info("ingest: upload %s is an exact duplicate of item %s", uid, known["id"])
        return

    info, fp, img = _decode(info, src, up["original_filename"])
    size = src.stat().st_size
    decision = dedup.find_similar(conn, info, fp, size)
    log.info("ingest: upload %s (%s %sx%s) -> %s %s", uid, info.kind, info.width, info.height,
             decision.action, decision.detail)
    _store_new(conn, up, src, info, sha, fp, img, size, decision)


def _store_new(conn: sqlite3.Connection, up: sqlite3.Row, src: Path, info: MediaInfo, sha: str,
               fp: Fingerprint | None, img: Image.Image | None, size: int, decision: dedup.DedupDecision) -> None:
    uid = up["uid"]
    dest = storage.original_path(uid, 1, info.ext)
    written: list[Path] = []
    try:
        if _place(src, dest):
            written.append(dest)
        files, doc_fp = _write_derivatives(info, uid, 1, dest, img)
        written += files
        if doc_fp is not None:
            fp = doc_fp
        del img
        now = now_iso()
        with tx(conn):
            if conn.execute("SELECT 1 FROM items WHERE uid = ?", (uid,)).fetchone():
                raise RuntimeError(f"item {uid} already exists")
            if conn.execute("SELECT 1 FROM items WHERE sha256 = ?", (sha,)).fetchone():
                raise _Stale("an identical file was stored meanwhile")
            item_id, statuses = _insert_item(conn, uid, info, sha, fp, size, up["original_filename"], now)
            if decision.action == "link" and decision.target_id is not None:
                target = conn.execute("SELECT id FROM items WHERE id = ? AND deleted_at IS NULL",
                                      (decision.target_id,)).fetchone()
                if target is not None:
                    dedup.link_into_group(conn, item_id, int(target["id"]), decision.detail)
            _set_upload(conn, up["id"], "stored", item_id, None, now, sha)
            _enqueue_followups(conn, item_id, 1, info.kind, statuses)
            bump_search_version(conn)
    except _Stale:
        _cleanup_unreferenced(conn, written)
        known = conn.execute("SELECT id FROM items WHERE sha256 = ?", (sha,)).fetchone()
        with tx(conn):
            _set_upload(conn, up["id"], "duplicate", int(known["id"]) if known else None, None, now_iso(), sha)
        _drop_source(conn, src)
        return
    except BaseException:
        _cleanup_unreferenced(conn, written)
        raise
    _refresh_fts(conn, item_id)
    _drop_source(conn, src, dest)
    log.info("ingest: upload %s stored as item %s (%s)", uid, item_id, info.kind)


def mark_upload_failed(conn: sqlite3.Connection, upload_id: int, message: str) -> None:
    """Final ingest failure (called by the worker): status 'error' with a short message. The
    upload's bytes move to storage.failed_path (never swept), so requeue_failed_upload can try
    again; whatever else an interrupted run left for this upload is deleted."""
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


def requeue_failed_upload(conn: sqlite3.Connection, upload_id: int) -> bool:
    """Retry an upload whose ingest failed for good: move its kept file back to staging, set it to
    'queued' and enqueue a new ingest job. False if it is not a failed upload with a kept file."""
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


# ------------------------------------------------------------------ video previews
def _info_from_row(row: sqlite3.Row) -> MediaInfo:
    return MediaInfo(
        kind=row["kind"], mime=row["mime"], ext=row["ext"], width=row["width"], height=row["height"],
        duration_s=row["duration_s"], bitrate=row["bitrate"], video_codec=row["video_codec"], fps=row["fps"],
        has_audio=None if row["has_audio"] is None else bool(row["has_audio"]), page_count=row["page_count"],
    )


def process_video_preview(conn: sqlite3.Connection, item_id: int, version: int) -> None:
    """'video_preview' job (video queue): transcode the H.264 streaming copy, set preview_status."""
    m = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if m is None or int(m["version"]) != int(version) or m["preview_status"] != "pending" or m["kind"] != "video":
        return
    src = storage.original_path(m["uid"], int(version), m["ext"])
    dst = storage.video_preview_path(m["uid"], int(version))
    video.transcode_preview(src, dst, _info_from_row(m), settings.video_preview_max_height)
    with tx(conn):
        cur = conn.execute(
            "UPDATE items SET preview_status = 'done', updated_at = ? "
            "WHERE id = ? AND version = ? AND preview_status = 'pending'",
            (now_iso(), item_id, int(version)),
        )
        done = cur.rowcount == 1
    if not done:
        storage.safe_unlink(dst)
        log.info("video preview: item %s changed during the transcode, output discarded", item_id)
        return
    log.info("video preview: item %s v%s done", m["uid"], version)


def _mark_status(conn: sqlite3.Connection, column: str, item_id: int, version: int) -> None:
    with tx(conn):
        conn.execute(
            f"UPDATE items SET {column} = 'error', updated_at = ? WHERE id = ? AND version = ? AND {column} = 'pending'",
            (now_iso(), item_id, int(version)),
        )


def mark_preview_failed(conn: sqlite3.Connection, item_id: int, version: int) -> None:
    _mark_status(conn, "preview_status", item_id, version)


def mark_analysis_failed(conn: sqlite3.Connection, item_id: int, version: int) -> None:
    _mark_status(conn, "analysis_status", item_id, version)


def mark_text_failed(conn: sqlite3.Connection, item_id: int, version: int) -> None:
    _mark_status(conn, "text_status", item_id, version)


# ------------------------------------------------------------------ reprocessing
def enqueue_text(conn: sqlite3.Connection, item_id: int, version: int) -> None:
    """Set text_status 'pending' and queue the 'text' job (unique)."""
    with tx(conn):
        cur = conn.execute(
            "UPDATE items SET text_status = 'pending', updated_at = ? WHERE id = ? AND version = ?",
            (now_iso(), item_id, int(version)),
        )
        if cur.rowcount == 1:
            jobs.enqueue(conn, "text", item_payload(item_id, version), unique=True)


def _regenerate_derivatives(row: sqlite3.Row) -> None:
    """Make the thumb and previews again (and drop page renders, which are re-made on request)."""
    uid, version = row["uid"], int(row["version"])
    original = storage.original_path(uid, version, row["ext"])
    if not original.is_file():
        return
    info = _info_from_row(row)
    img = None
    if row["kind"] == "image":
        try:
            img = images.load_oriented(original)
        except ValueError as e:
            log.warning("reprocess: %s does not decode: %s", uid, e)
            return
    if row["kind"] in TEXT_KINDS:
        storage.remove_tree(storage.pages_dir(uid, version))
    _write_derivatives(info, uid, version, original, img)


def enqueue_reprocess(conn: sqlite3.Connection, item_id: int, stages: list[str]) -> None:
    """Run stages again for an item: 'analyze' (analysis_status back to pending), 'text' (any kind
    but video and other) and 'preview' (thumb and previews made again now; videos get a new
    streaming copy job). Unknown stages are ignored."""
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return
    version = int(row["version"])
    kind = row["kind"]
    wanted = [s for s in dict.fromkeys(stages) if s in STAGES]
    if "preview" in wanted and kind != "other":
        _regenerate_derivatives(row)
        if kind == "video" and settings.video_previews:
            with tx(conn):
                conn.execute("UPDATE items SET preview_status = 'pending', updated_at = ? WHERE id = ?",
                             (now_iso(), item_id))
                jobs.enqueue(conn, "video_preview", item_payload(item_id, version), unique=True)
    if "analyze" in wanted and kind in ANALYSED_KINDS:
        with tx(conn):
            conn.execute("UPDATE items SET analysis_status = 'pending', updated_at = ? WHERE id = ?",
                         (now_iso(), item_id))
            jobs.enqueue(conn, "analyze", item_payload(item_id, version), unique=True)
    if "text" in wanted and kind not in ("video", "other"):
        enqueue_text(conn, item_id, version)


# ------------------------------------------------------------------ trash, restore, purge
def soft_delete(conn: sqlite3.Connection, item_id: int) -> None:
    """Move an item to the trash (deleted_at). Its duplicate group re-chooses its best."""
    with tx(conn):
        cur = conn.execute(
            "UPDATE items SET deleted_at = ?, updated_at = ? WHERE id = ? AND deleted_at IS NULL",
            (now_iso(), now_iso(), item_id),
        )
        if cur.rowcount != 1:
            return
        members = dedup.group_member_ids(conn, item_id)
        if len(members) > 1:
            dedup.regroup(conn, members)
        bump_search_version(conn)


def restore(conn: sqlite3.Connection, item_id: int) -> None:
    """Take an item out of the trash."""
    with tx(conn):
        cur = conn.execute(
            "UPDATE items SET deleted_at = NULL, updated_at = ? WHERE id = ? AND deleted_at IS NOT NULL",
            (now_iso(), item_id),
        )
        if cur.rowcount != 1:
            return
        members = dedup.group_member_ids(conn, item_id)
        if len(members) > 1:
            dedup.regroup(conn, members)
        bump_search_version(conn)


def _item_files(uid: str) -> tuple[list[Path], list[Path]]:
    """(files, directories) of every version of an item."""
    files: list[Path] = []
    dirs: list[Path] = []
    for path in (storage.original_path(uid, 1, ".x"), storage.thumb_path(uid, 1), storage.preview_path(uid, 1),
                 storage.video_preview_path(uid, 1), storage.doc_photo_path(uid, 1)):
        d = path.parent
        if d.is_dir():
            files.extend(p for p in d.glob(f"{uid}.v*") if p.is_file())
    pages_parent = storage.pages_dir(uid, 1).parent
    if pages_parent.is_dir():
        dirs.extend(p for p in pages_parent.glob(f"{uid}.v*") if p.is_dir())
    return files, dirs


def purge(conn: sqlite3.Connection, item_id: int) -> None:
    """Delete an item for good: its rows (faces, bodies, embeddings, text, pages, document and tags
    cascade; uploads keep their outcome with item_id NULL), its full-text row, unnamed people left
    empty, and every file (original, thumbs, previews, page renders, streaming copy, face and body
    crops, ID photo). The duplicate group re-chooses its best first."""
    row = conn.execute("SELECT id, uid FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return
    with tx(conn):
        face_ids = [int(r[0]) for r in conn.execute("SELECT id FROM faces WHERE item_id = ?", (item_id,))]
        body_ids = [int(r[0]) for r in conn.execute("SELECT id FROM bodies WHERE item_id = ?", (item_id,))]
        people = {int(r[0]) for r in conn.execute(
            "SELECT person_id FROM faces WHERE item_id = ? AND person_id IS NOT NULL UNION "
            "SELECT person_id FROM bodies WHERE item_id = ? AND person_id IS NOT NULL UNION "
            "SELECT person_id FROM item_tags WHERE item_id = ?", (item_id, item_id, item_id))}
        members = [m for m in dedup.group_member_ids(conn, item_id) if m != int(item_id)]
        conn.execute("UPDATE items SET dup_group_id = NULL, dup_role = NULL WHERE id = ?", (item_id,))
        if members:
            dedup.regroup(conn, members)
        _remove_fts(conn, item_id)
        conn.execute("UPDATE uploads SET item_id = NULL WHERE item_id = ?", (item_id,))
        conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        mark = int(kv_get(conn, "item_id_mark", "0") or 0)
        if int(item_id) > mark:
            kv_set(conn, "item_id_mark", int(item_id))
        if people:
            from app.persons import drop_empty_unnamed

            drop_empty_unnamed(conn, people)
        bump_faces_version(conn)
        bump_search_version(conn)
    files, dirs = _item_files(row["uid"])
    files += [storage.face_crop_path(i) for i in face_ids] + [storage.body_crop_path(i) for i in body_ids]
    for p in files:
        storage.safe_unlink(p)
    for d in dirs:
        storage.remove_tree(d)
    log.info("purged item %s (%d files)", row["uid"], len(files))


def expire_trash(conn: sqlite3.Connection) -> int:
    """Purge items that have been in the trash longer than settings.trash_days."""
    cutoff = now_iso(-settings.trash_days * 86400)
    ids = [int(r[0]) for r in conn.execute(
        "SELECT id FROM items WHERE deleted_at IS NOT NULL AND deleted_at < ? ORDER BY deleted_at", (cutoff,))]
    for item_id in ids:
        try:
            purge(conn, item_id)
        except Exception:
            log.exception("trash: could not purge item %s", item_id)
    if ids:
        log.info("trash: purged %d items older than %d days", len(ids), settings.trash_days)
    return len(ids)


# ------------------------------------------------------------------ housekeeping
def cleanup_stale_uploads(conn: sqlite3.Connection) -> int:
    """Cancel 'uploading' rows idle longer than settings.staging_ttl_hours and delete their staged
    files; delete staged files that have no live upload; then sweep orphaned files. Returns the
    number of uploads and files cleaned."""
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
                    if now_s - p.stat().st_mtime <= ORPHAN_AGE_S:
                        continue  # the create endpoint may not have committed its row yet
                except FileNotFoundError:
                    continue
            elif row["status"] in ACTIVE:
                continue
            storage.safe_unlink(p)
            cleaned += 1
    cleaned += sweep_orphan_files(conn)
    if cleaned:
        log.info("cleanup: removed %d stale uploads and files", cleaned)
    return cleaned


def sweep_orphan_files(conn: sqlite3.Connection, min_age_s: float = ORPHAN_AGE_S) -> int:
    """Delete files under storage.SWEPT_DIRS that no row references and that are older than
    min_age_s (a crash between a commit and its file cleanup leaves them behind for good), plus
    empty page directories. Files of uploads still being ingested are kept. Runs on the main
    worker thread between jobs, so no ingest is writing meanwhile."""
    root = storage.root()
    items = {(r[0], int(r[1])): r[2] for r in conn.execute("SELECT uid, version, ext FROM items")}
    crops = {
        "faces": {int(r[0]) for r in conn.execute("SELECT id FROM faces")},
        "bodies": {int(r[0]) for r in conn.execute("SELECT id FROM bodies")},
    }
    marks = ",".join("?" * len(ACTIVE))
    active = {r[0] for r in conn.execute(f"SELECT uid FROM uploads WHERE status IN ({marks})", tuple(ACTIVE))}
    cutoff = time.time() - min_age_s
    removed = 0
    for top in storage.SWEPT_DIRS:
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
            keep = False
            if top in crops:
                m = _CROP_NAME.match(name)
                keep = bool(m and int(m["id"]) in crops[top])
            elif top == "pages":
                d = _PAGES_DIR_NAME.match(p.parent.name)
                keep = bool(d and _PAGE_NAME.match(name)
                            and (d["uid"] in active or (d["uid"], int(d["version"])) in items))
            else:
                m = (_ORIGINAL_NAME if top == "originals" else _DERIVED_NAME).match(name)
                if m:
                    ext = items.get((m["uid"], int(m["version"])))
                    keep = m["uid"] in active or (ext is not None and (top != "originals" or ext == m["ext"]))
            if keep or _is_referenced(conn, p):
                continue
            log.info("cleanup: removing orphaned file %s", p.relative_to(root))
            storage.safe_unlink(p)
            removed += 1
    pages = root / "pages"
    if pages.is_dir():
        for d in sorted(pages.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            try:
                if d.is_dir() and not any(d.iterdir()) and d.stat().st_mtime <= cutoff:
                    d.rmdir()
            except OSError:
                pass
    return removed


def recover_on_startup(conn: sqlite3.Connection) -> None:
    """Re-queue interrupted jobs; make sure every queued/processing upload has an ingest job and
    every item whose analysis, text or streaming copy is pending has its job; wipe data_dir/tmp."""
    n = jobs.requeue_running(conn)
    if n:
        log.info("startup: re-queued %d interrupted jobs", n)
    added = 0
    for r in conn.execute("SELECT id FROM uploads WHERE status IN ('queued', 'processing') ORDER BY id").fetchall():
        added += jobs.enqueue(conn, "ingest", {"upload_id": int(r["id"])}, unique=True) is not None
    kinds = ",".join(f"'{k}'" for k in ANALYSED_KINDS)
    for r in conn.execute(
        f"SELECT id, version FROM items WHERE analysis_status = 'pending' AND kind IN ({kinds}) ORDER BY id"
    ).fetchall():
        added += jobs.enqueue(conn, "analyze", item_payload(r["id"], r["version"]), unique=True) is not None
    for r in conn.execute("SELECT id, version FROM items WHERE text_status = 'pending' ORDER BY id").fetchall():
        added += jobs.enqueue(conn, "text", item_payload(r["id"], r["version"]), unique=True) is not None
    for r in conn.execute(
        "SELECT id, version FROM items WHERE kind = 'video' AND preview_status = 'pending' ORDER BY id"
    ).fetchall():
        added += jobs.enqueue(conn, "video_preview", item_payload(r["id"], r["version"]), unique=True) is not None
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
