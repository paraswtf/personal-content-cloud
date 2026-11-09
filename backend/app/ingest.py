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
- soft_delete / restore (trash, 30 days; restore queues again the stages still pending), purge
  (rows and every file of the item), expire_trash.
- enqueue_reprocess(conn, item_id, stages), enqueue_text(conn, item_id, version). A document's
  'preview' reprocess renders on a worker background thread into a scratch page directory and
  replaces the page files one by one (os.replace), so a running text job never misses a page.
- cleanup_stale_uploads, expire_failed, sweep_orphan_files, recover_on_startup, mark_*_failed
  (worker), requeue_failed_upload and retry_failed (POST /api/system/retry-failed).
Every handler is idempotent: it checks DB state first (terminal upload -> no-op, version mismatch ->
no-op), writes files to deterministic paths (a new item reuses the upload uid) and switches the DB
over in one short transaction; heavy work (decode, ffmpeg) happens before it. The staged file is
hard-linked to its destination and only unlinked after the commit, so a retried attempt can never
lose the upload's bytes.
"""
from __future__ import annotations

import json
from typing import Any
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
from app.util import json_dumps, new_uid, now_iso

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
_DERIVED_NAME = re.compile(r"^(?P<uid>[a-z0-9]+)\.v(?P<version>\d+)\.(?:sm\.webp|webp|mp4|live\.mp4)$")
_PAGES_DIR_NAME = re.compile(r"^(?P<uid>[a-z0-9]+)\.v(?P<version>\d+)$")
_PAGE_NAME = re.compile(r"^(?P<page>\d+)\.webp$")
_CROP_NAME = re.compile(r"^(?P<id>\d+)\.webp$")  # faces/<id//1000>/<id>.webp, bodies/..., animals/...


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
    tmp = storage.scratch_beside(dest, ".link")  # beside dest: originals may live on another disk than tmp/
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
    if top in ("faces", "bodies", "animals"):
        m = _CROP_NAME.match(name)
        return bool(m and conn.execute(f"SELECT 1 FROM {top} WHERE id = ?", (int(m["id"]),)).fetchone())
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
    kept = storage.failed_path(upload_uid)
    if kept.is_file():  # mark_upload_failed moved it but did not get to commit the status
        log.info("upload %s: staged file gone, resuming from %s", upload_uid, kept)
        return kept
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
                       img: Image.Image | None, pages_uid: str | None = None) -> tuple[list[Path], Fingerprint | None]:
    """Thumb and previews; for documents also the page renders (up to settings.page_render_max, so
    the text stage finds every page it scans on disk), into the page directory of pages_uid when
    given (a scratch name used by reprocess). Returns the files written and, for
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
            written.append(storage.pages_dir(pages_uid or uid, version))
            documents.render_pages(original, info.kind, pages_uid or uid, version, max_side=settings.page_render_size,
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


def _write_motion_clip(info: MediaInfo, uid: str, version: int, original: Path) -> Path | None:
    """The MP4 embedded in a JPEG motion photo, written next to the streaming copies (section 20)."""
    if info.kind != "image" or info.ext != ".jpg":
        return None
    from app.media import live

    try:
        clip = live.extract_motion_clip(original)
    except Exception:  # noqa: BLE001 - never block an upload on this
        log.exception("ingest: could not read the motion clip of %s", uid)
        return None
    if not clip:
        return None
    dest = storage.live_clip_path(uid, version)
    dest.parent.mkdir(parents=True, exist_ok=True)
    storage.write_bytes_atomic(dest, clip)
    return dest


def pair_live(conn: sqlite3.Connection, item_id: int, pair_key: str | None) -> bool:
    """Pair a just-stored still or short MOV with its other half (section 20). The MOV becomes a hidden
    companion of the still: no analysis, no search row, no duplicate group, the still's vault flag.
    Runs in the caller's transaction."""
    from app.media import live

    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None or row["companion_of"] or row["live_video_item_id"]:
        return False
    partner = live.find_partner(conn, row, pair_key)
    if partner is None:
        return False
    still, mov = (row, partner) if row["kind"] == "image" else (partner, row)
    now = now_iso()
    conn.execute("UPDATE items SET live = 1, live_video_item_id = ?, updated_at = ? WHERE id = ?",
                 (int(mov["id"]), now, int(still["id"])))
    members = dedup.group_member_ids(conn, int(mov["id"])) if mov["dup_group_id"] else []
    conn.execute(
        "UPDATE items SET companion_of = ?, analysis_status = 'done', dup_group_id = NULL, dup_role = NULL, "
        "nsfw = ?, deleted_at = ?, updated_at = ? WHERE id = ?",
        (int(still["id"]), int(still["nsfw"]), still["deleted_at"], now, int(mov["id"])),
    )
    conn.execute("DELETE FROM item_embeddings WHERE item_id = ?", (int(mov["id"]),))
    if len(members) > 1:
        dedup.regroup(conn, [m for m in members if m != int(mov["id"])])
    try:
        from app.documents.extract import remove_fts

        remove_fts(conn, int(mov["id"]))
    except ImportError:
        pass
    log.info("ingest: paired Live Photo still %s with clip %s", still["uid"], mov["uid"])
    return True


def _item_columns(info: MediaInfo, fp: Fingerprint | None, now: str,
                  forward: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """The item columns a probe result decides (everything but identity, name, bytes and timestamps),
    and the stage statuses, shared by a new item and a plain file probed again (reprobe_plain)."""
    statuses = {
        "analysis_status": "pending" if info.kind in ANALYSED_KINDS else "done",
        "text_status": "pending" if info.kind in TEXT_KINDS else "none",
        "preview_status": _preview_status(info),
    }
    # Section 29: what the file alone says; the analyze stage adds the model's signals and decides.
    # A file that is never analysed is decided here from the file signals.
    received_via = None
    forward_checked = 0
    if forward is not None and settings.forwards_enabled:
        from app.analysis import forwards as forwards_mod

        received_via = forwards_mod.decide(forward).received_via
        forward_checked = 0 if statuses["analysis_status"] == "pending" else 1
    elif forward is not None:
        forward_checked = 1
    cols = dict(
        kind=info.kind, ext=info.ext, mime=info.mime, width=info.width, height=info.height, duration_s=info.duration_s,
        bitrate=info.bitrate, video_codec=info.video_codec, fps=info.fps,
        has_audio=None if info.has_audio is None else int(bool(info.has_audio)), page_count=info.page_count,
        animated=int(bool(info.animated)), taken_at=info.taken_at, exif_datetime=info.exif_datetime,
        exif_subsec=info.exif_subsec, camera=info.camera, gps_lat=info.gps_lat, gps_lon=info.gps_lon,
        gps_alt=info.gps_alt, phash=fp.phash if fp else None, dhash=fp.dhash if fp else None,
        sig=fp.sig if fp else None, frames=json_dumps(fp.frames) if fp and fp.frames is not None else None,
        category="documents" if info.kind in TEXT_KINDS else "misc", content_id=info.content_id, **statuses,
        received_via=received_via, forward_signals=json_dumps(forward) if forward is not None else None,
        forward_checked=forward_checked, updated_at=now,
    )
    return cols, statuses


def _insert_item(conn: sqlite3.Connection, uid: str, info: MediaInfo, sha: str, fp: Fingerprint | None,
                 size: int, filename: str, now: str, forward: dict[str, Any] | None = None) -> tuple[int, dict[str, str]]:
    item_id = next_item_id(conn)
    cols, statuses = _item_columns(info, fp, now, forward)
    cols = dict(id=item_id, uid=uid, version=1, original_filename=filename, sha256=sha, size_bytes=size,
                created_at=now, **cols)
    conn.execute(f"INSERT INTO items({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", tuple(cols.values()))
    return item_id, statuses


# ------------------------------------------------------------------ derived files the app no longer keeps
TIDY_KEY = "derivatives_tidy"  # kv: the TIDY_VERSION whose pass has completed
TIDY_VERSION = 1


def _unlink_pictures(uid: str, version: int) -> int:
    """Delete an item version's thumb and both previews; returns how many existed."""
    n = 0
    for p in (storage.thumb_path(uid, version), storage.preview_path(uid, version),
              storage.preview_small_path(uid, version)):
        if p.is_file():
            storage.safe_unlink(p)
            n += 1
    return n


def _drop_companion_pictures(conn: sqlite3.Connection, item_id: int) -> int:
    """After a commit that may have paired a Live Photo (item_id is either half): delete the clip's own
    thumb and previews. The library never shows them and the file routes serve the still's (section 20)."""
    rows = conn.execute(
        "SELECT uid, version FROM items WHERE companion_of IS NOT NULL AND (id = ? OR companion_of = ?)",
        (int(item_id), int(item_id)),
    ).fetchall()
    return sum(_unlink_pictures(r["uid"], int(r["version"])) for r in rows)


def tidy_derivatives(conn: sqlite3.Connection, force: bool = False) -> int:
    """Remove derived files the app no longer keeps (docs section 3), once per TIDY_VERSION (kv
    derivatives_tidy; force runs it again): the thumb and previews of Live Photo clips (companions, whose
    pictures are the still's, section 20) and small previews that repeat their large preview (the large
    one already fits the small size; the small URL serves the large file). Runs from housekeeping. Only
    files whose replacement is in place are removed. Returns the number of files removed."""
    if not force and int(kv_get(conn, TIDY_KEY, "0") or 0) >= TIDY_VERSION:
        return 0
    removed = 0
    for r in conn.execute("SELECT uid, version FROM items WHERE companion_of IS NOT NULL").fetchall():
        removed += _unlink_pictures(r["uid"], int(r["version"]))
    for r in conn.execute("SELECT uid, version FROM items WHERE companion_of IS NULL").fetchall():
        uid, version = r["uid"], int(r["version"])
        small = storage.preview_small_path(uid, version)
        if not small.is_file():
            continue
        try:
            with Image.open(storage.preview_path(uid, version)) as im:
                same = images.small_is_large(im.size)
        except (OSError, ValueError):
            continue  # no readable large preview: the small one is all there is
        if same:
            storage.safe_unlink(small)
            removed += 1
    with tx(conn):
        kv_set(conn, TIDY_KEY, TIDY_VERSION)
    if removed:
        log.info("tidy: removed %d derived files the app no longer keeps", removed)
    return removed


# ------------------------------------------------------------------ plain files probed again
PROBE_VERSION_KEY = "probe_version"  # kv: the probe.PROBE_VERSION whose re-probe pass has completed
PLAIN_MEDIA_SQL = ("kind = 'other' AND deleted_at IS NULL AND companion_of IS NULL AND "
                   "(mime LIKE 'image/%' OR mime LIKE 'video/%' OR mime = 'application/pdf' OR ext = '.jxl')")


def plain_media_count(conn: sqlite3.Connection) -> int:
    """Items kept as plain files although their bytes claim to be a photo, a video or a PDF."""
    return int(conn.execute(f"SELECT COUNT(*) FROM items WHERE {PLAIN_MEDIA_SQL}").fetchone()[0])


def reprobe_plain(conn: sqlite3.Connection) -> dict[str, int]:
    """Probe again the items kept as 'other' whose bytes sniff as a photo, video or PDF (section 7): the
    probe has learned to read some of them since (probe.PROBE_VERSION). One that now probes as a real
    kind is rebuilt in place: derivatives written, the row's kind, media columns, fingerprint and
    statuses set from the probe, the analysis, text and streaming-copy jobs queued and a Live Photo
    pairing tried. A file whose storage extension would change is left alone (the row and the file name
    must change together). A ToolFailure ends the pass early. Returns {"checked", "fixed", "complete"}:
    complete is False when the pass stopped early, so the next start tries again."""
    rows = conn.execute(f"SELECT * FROM items WHERE {PLAIN_MEDIA_SQL} ORDER BY id").fetchall()
    out = {"checked": 0, "fixed": 0, "complete": 1}
    for row in rows:
        out["checked"] += 1
        try:
            if _reprobe_item(conn, row):
                out["fixed"] += 1
        except ToolFailure as e:
            log.warning("reprobe: %s; the rest waits for the next start", e)
            out["complete"] = 0
            break
        except Exception:  # noqa: BLE001 - one bad file never stops the pass
            log.exception("reprobe: item %s could not be rebuilt", row["uid"])
    if out["fixed"]:
        log.info("reprobe: %d of %d plain files were photos, videos or documents after all", out["fixed"], out["checked"])
    return out


def _reprobe_item(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    uid, version = row["uid"], int(row["version"])
    original = storage.original_path(uid, version, row["ext"])
    if not original.is_file():
        return False
    info = probe(original, row["original_filename"])
    if info is None or info.kind == "other" or info.ext != row["ext"]:
        return False
    info, fp, img = _decode(info, original, row["original_filename"])
    if info.kind == "other":
        return False
    written: list[Path] = []
    try:
        clip_path = _write_motion_clip(info, uid, version, original)
        if clip_path is not None:
            written.append(clip_path)
        files, doc_fp = _write_derivatives(info, uid, version, original, img)
        written += files
        if doc_fp is not None:
            fp = doc_fp
        del img
        now = now_iso()
        from app.analysis import forwards as forwards_mod

        forward = forwards_mod.file_signals(
            kind=info.kind, ext=info.ext, width=info.width, height=info.height, camera=info.camera,
            exif_datetime=info.exif_datetime, gps_lat=info.gps_lat, bitrate=info.bitrate,
            filename=row["original_filename"], path=original if info.kind == "image" else None)
        cols, statuses = _item_columns(info, fp, now, forward)
        if clip_path is not None:
            cols["live"] = 1
        with tx(conn):
            if conn.execute("SELECT 1 FROM items WHERE id = ? AND version = ? AND kind = 'other'",
                            (int(row["id"]), version)).fetchone() is None:
                raise _Stale("the item changed meanwhile")
            sets = ", ".join(f"{k} = ?" for k in cols)
            conn.execute(f"UPDATE items SET {sets} WHERE id = ?", (*cols.values(), int(row["id"])))
            _enqueue_followups(conn, int(row["id"]), version, info.kind, statuses)
            pair_live(conn, int(row["id"]), None)
            bump_search_version(conn)
    except _Stale:
        _cleanup_unreferenced(conn, written)
        return False
    except BaseException:
        _cleanup_unreferenced(conn, written)
        raise
    _refresh_fts(conn, int(row["id"]))
    _drop_companion_pictures(conn, int(row["id"]))
    log.info("reprobe: item %s is a %s after all (%s)", uid, info.kind, row["original_filename"])
    return True


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
        # Done already (maybe a crash after the commit): only the staged name may be left over. A
        # failed upload's staged file may be its only copy (housekeeping moves it to failed/).
        if up["status"] != "error" and not _is_referenced(conn, storage.staging_path(uid)):
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
        clip_path = _write_motion_clip(info, uid, 1, dest)
        if clip_path is not None:
            written.append(clip_path)
        files, doc_fp = _write_derivatives(info, uid, 1, dest, img)
        written += files
        if doc_fp is not None:
            fp = doc_fp
        del img
        now = now_iso()
        from app.analysis import forwards as forwards_mod

        forward = forwards_mod.file_signals(
            kind=info.kind, ext=info.ext, width=info.width, height=info.height, camera=info.camera,
            exif_datetime=info.exif_datetime, gps_lat=info.gps_lat, bitrate=info.bitrate,
            filename=up["original_filename"], path=dest if info.kind == "image" else None)
        with tx(conn):
            if conn.execute("SELECT 1 FROM items WHERE uid = ?", (uid,)).fetchone():
                raise RuntimeError(f"item {uid} already exists")
            if conn.execute("SELECT 1 FROM items WHERE sha256 = ?", (sha,)).fetchone():
                raise _Stale("an identical file was stored meanwhile")
            item_id, statuses = _insert_item(conn, uid, info, sha, fp, size, up["original_filename"], now, forward)
            if clip_path is not None:
                conn.execute("UPDATE items SET live = 1 WHERE id = ?", (item_id,))
            if decision.action == "link" and decision.target_id is not None:
                target = conn.execute("SELECT id FROM items WHERE id = ? AND deleted_at IS NULL",
                                      (decision.target_id,)).fetchone()
                if target is not None:
                    dedup.link_into_group(conn, item_id, int(target["id"]), decision.detail)
            _set_upload(conn, up["id"], "stored", item_id, None, now, sha)
            _enqueue_followups(conn, item_id, 1, info.kind, statuses)
            pair_live(conn, item_id, up["pair_key"] if "pair_key" in up.keys() else None)
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
    _drop_companion_pictures(conn, item_id)
    _drop_source(conn, src, dest)
    log.info("ingest: upload %s stored as item %s (%s)", uid, item_id, info.kind)


def _keep_failed(src: Path, upload_uid: str) -> Path:
    """Move a failed upload's bytes to storage.failed_path and date the file now, so the retention
    of expire_failed counts from the failure. Raises OSError."""
    dest = storage.failed_path(upload_uid)
    if not _same_path(src, dest):
        storage.move_into_place(src, dest)
    try:
        os.utime(dest)
    except OSError:
        pass
    return dest


def mark_upload_failed(conn: sqlite3.Connection, upload_id: int, message: str) -> None:
    """Final ingest failure (called by the worker): status 'error' with a short message. The
    upload's bytes move to storage.failed_path (never swept) before the status is committed, so
    requeue_failed_upload can try again; whatever else an interrupted run left for this upload is
    deleted. When the move fails the bytes stay in staging, where housekeeping keeps them and tries
    the move again."""
    up = conn.execute("SELECT id, uid, status FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if up is None or up["status"] in TERMINAL:
        return
    src = _locate_upload_file(conn, up["uid"])
    kept: Path | None = None
    if src is not None:
        try:
            kept = _keep_failed(src, up["uid"])
        except OSError as e:
            log.error("upload %s: could not keep the failed file %s (%s); leaving it in place", up["uid"], src, e)
    with tx(conn):
        cur = conn.execute(
            "UPDATE uploads SET status = 'error', message = ?, updated_at = ? WHERE id = ? AND status IN "
            "('queued', 'processing')",
            (message[:300], now_iso(), upload_id),
        )
    if cur.rowcount != 1:
        if kept is not None and not _same_path(src, kept):  # changed meanwhile: put the bytes back
            try:
                storage.move_into_place(kept, storage.staging_path(up["uid"]))
            except OSError as e:
                log.warning("upload %s: could not move %s back to staging (%s)", up["uid"], kept, e)
        return
    if src is not None and kept is None:
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
        if not staged.is_file():
            return False
        try:  # the move to failed/ did not happen: keep the bytes there first
            _keep_failed(staged, up["uid"])
        except OSError as e:
            log.warning("upload %s: could not move the staged file to failed/ (%s)", up["uid"], e)
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


_STATUS_OF_JOB = {"analyze": "analysis_status", "text": "text_status", "video_preview": "preview_status"}


def retry_failed(conn: sqlite3.Connection) -> int:
    """Retry everything that failed for good: uploads in 'error' whose bytes are kept (in failed/,
    or still in staging) go back through requeue_failed_upload; errored analyze, text and
    video_preview jobs are queued again with their item's status set back from 'error' to 'pending'
    (the handlers only run pending items); other errored jobs (guess, recluster) are queued again.
    Errored ingest jobs of uploads that cannot be retried are dropped. Returns how many uploads and
    jobs were queued again."""
    n = 0
    for r in conn.execute("SELECT id, uid FROM uploads WHERE status = 'error' ORDER BY id").fetchall():
        if not (storage.failed_path(r["uid"]).is_file() or storage.staging_path(r["uid"]).is_file()):
            continue
        try:
            n += requeue_failed_upload(conn, int(r["id"]))
        except Exception:
            log.exception("retry: could not requeue upload %s", r["uid"])
    with tx(conn):
        for job in conn.execute("SELECT id, type, payload FROM jobs WHERE status = 'error'").fetchall():
            try:
                payload = json.loads(job["payload"] or "{}")
            except ValueError:
                continue
            if job["type"] == "ingest":
                up = conn.execute("SELECT status FROM uploads WHERE id = ?", (payload.get("upload_id"),)).fetchone()
                if up is None or up["status"] in TERMINAL:
                    conn.execute("DELETE FROM jobs WHERE id = ?", (job["id"],))
                continue
            column = _STATUS_OF_JOB.get(job["type"])
            if column is not None and "item_id" in payload and "version" in payload:
                conn.execute(
                    f"UPDATE items SET {column} = 'pending', updated_at = ? WHERE id = ? AND version = ? "
                    f"AND {column} = 'error'",
                    (now_iso(), int(payload["item_id"]), int(payload["version"])),
                )
        n += jobs.retry_failed(conn)
    return n


def expire_failed(conn: sqlite3.Connection) -> int:
    """Delete kept failed uploads (failed/*.part) older than settings.trash_days."""
    base = storage.originals_root() / "failed"
    if not base.is_dir():
        return 0
    cutoff = time.time() - settings.trash_days * 86400
    removed = 0
    for p in base.iterdir():
        try:
            if not p.is_file() or p.is_symlink() or p.stat().st_mtime > cutoff:
                continue
        except FileNotFoundError:
            continue
        storage.safe_unlink(p)
        removed += 1
    if removed:
        log.info("failed uploads: deleted %d files older than %d days", removed, settings.trash_days)
    return removed


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


def _swap_pages(scratch: Path, final: Path) -> int:
    """Move the page renders of a scratch directory into the real one, one os.replace per page, so a
    reader (the text job) always finds every page, old or new. Pages past the new count stay (they
    are renders of the same file). Returns the number of pages moved."""
    if not scratch.is_dir():
        return 0
    names = sorted((p.name for p in scratch.iterdir() if _PAGE_NAME.match(p.name)), key=lambda n: int(n[:-5]))
    if names:
        final.mkdir(parents=True, exist_ok=True)
    for name in names:
        os.replace(scratch / name, final / name)
    return len(names)


def _regenerate_derivatives(row: sqlite3.Row, conn: sqlite3.Connection | None = None) -> None:
    """Make the thumb and previews again. A document's pages are rendered into a scratch directory
    and swapped in page by page (_swap_pages), and only while the item (with conn) still exists at
    this version."""
    uid, version = row["uid"], int(row["version"])
    if "companion_of" in row.keys() and row["companion_of"]:
        return  # a Live Photo clip keeps no pictures of its own (section 20)
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
    if row["kind"] not in TEXT_KINDS:
        _write_derivatives(info, uid, version, original, img)
        return
    scratch_uid = f"{uid}r{new_uid(8)}"  # same shard as uid, so the renames stay in one directory tree
    scratch = storage.pages_dir(scratch_uid, version)
    try:
        _write_derivatives(info, uid, version, original, None, pages_uid=scratch_uid)
        if conn is not None and conn.execute(
            "SELECT 1 FROM items WHERE id = ? AND version = ?", (int(row["id"]), version)
        ).fetchone() is None:
            log.info("reprocess: item %s changed during the page render, renders discarded", uid)
            return
        n = _swap_pages(scratch, storage.pages_dir(uid, version))
        log.info("reprocess: %s v%s has %d new page renders", uid, version, n)
    finally:
        storage.remove_tree(scratch)


def _render_document_again(item_id: int, version: int) -> None:
    """Background part of a document's 'preview' reprocess (runs on a worker-owned thread)."""
    from app.db import connect

    conn = connect()
    try:
        row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None or int(row["version"]) != int(version) or row["kind"] not in TEXT_KINDS:
            return
        _regenerate_derivatives(row, conn)
    finally:
        conn.close()


def enqueue_reprocess(conn: sqlite3.Connection, item_id: int, stages: list[str]) -> None:
    """Run stages again for an item: 'analyze' (analysis_status back to pending), 'text' (any kind
    but video and other) and 'preview' (thumb and previews made again now; for documents on a worker
    background thread, page renders included; videos get a new streaming copy job). Unknown stages
    are ignored."""
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return
    version = int(row["version"])
    kind = row["kind"]
    wanted = [s for s in dict.fromkeys(stages) if s in STAGES]
    if "preview" in wanted and kind != "other":
        if kind in TEXT_KINDS:
            from app import worker

            worker.run_in_background(f"pages-{row['uid']}.v{version}", _render_document_again, int(item_id), version)
        else:
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
        _sync_companion(conn, item_id)
        bump_search_version(conn)


def _sync_companion(conn: sqlite3.Connection, still_id: int) -> None:
    """A Live Photo's companion follows the still into and out of the trash and the vault."""
    from app.media import live

    live.sync_companion(conn, still_id)


def restore(conn: sqlite3.Connection, item_id: int) -> None:
    """Take an item out of the trash and queue (unique) the stages still pending: a job that ran
    while the item was in the trash finished without doing anything (the text stage skips trashed
    items)."""
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
        _sync_companion(conn, item_id)
        bump_search_version(conn)
        row = conn.execute("SELECT kind, version, analysis_status, text_status, preview_status FROM items "
                           "WHERE id = ?", (item_id,)).fetchone()
        statuses = {k: row[k] for k in ("analysis_status", "text_status", "preview_status")}
        if row["kind"] not in ANALYSED_KINDS:
            statuses["analysis_status"] = "none"
        if row["kind"] != "video":
            statuses["preview_status"] = "none"
        _enqueue_followups(conn, item_id, int(row["version"]), row["kind"], statuses)


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


def _purge_one(conn: sqlite3.Connection, item_id: int) -> None:
    """Delete an item for good: its rows (faces, bodies, embeddings, text, pages, document and tags
    cascade; uploads keep their outcome with item_id NULL), its full-text row, unnamed people left
    empty, and every file (original, thumbs, previews, page renders, streaming copy, face, body and
    pet crops, ID photo). The duplicate group re-chooses its best first."""
    row = conn.execute("SELECT id, uid FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        return
    with tx(conn):
        face_ids = [int(r[0]) for r in conn.execute("SELECT id FROM faces WHERE item_id = ?", (item_id,))]
        body_ids = [int(r[0]) for r in conn.execute("SELECT id FROM bodies WHERE item_id = ?", (item_id,))]
        animal_ids = [int(r[0]) for r in conn.execute("SELECT id FROM animals WHERE item_id = ?", (item_id,))]
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
    files += [storage.animal_crop_path(i) for i in animal_ids]
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
    files; delete staged files that have no live upload. The staged file of an upload in 'error' is
    its only copy when mark_upload_failed could not move it: it is moved to failed/ now, never
    deleted. Returns the number of uploads and files cleaned. The orphan file sweep is separate
    (sweep_orphan_files, on its own daily thread)."""
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
    staging = storage.originals_root() / "staging"
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
            elif row["status"] == "error":
                if not storage.failed_path(p.name[:-5]).exists():
                    try:
                        _keep_failed(p, p.name[:-5])
                        log.info("cleanup: moved the bytes of failed upload %s to failed/", p.name[:-5])
                    except OSError as e:
                        log.warning("cleanup: could not move %s to failed/ (%s); keeping it", p.name, e)
                continue
            storage.safe_unlink(p)
            cleaned += 1
    if cleaned:
        log.info("cleanup: removed %d stale uploads and files", cleaned)
    return cleaned


def sweep_orphan_files(conn: sqlite3.Connection, min_age_s: float = ORPHAN_AGE_S, stop=None) -> int:
    """Delete files under storage.SWEPT_DIRS that no row references and that are older than
    min_age_s (a crash between a commit and its file cleanup leaves them behind for good), plus
    empty page directories. Files of uploads still being ingested are kept. Runs on its own thread
    (worker._sweep_loop) while jobs run, so every delete re-checks the database: _is_referenced and
    the upload's status (an ingest may have started after the snapshot below). Stops early when the
    threading.Event stop is set."""
    items = {(r[0], int(r[1])): r[2] for r in conn.execute("SELECT uid, version, ext FROM items")}
    crops = {
        "faces": {int(r[0]) for r in conn.execute("SELECT id FROM faces")},
        "bodies": {int(r[0]) for r in conn.execute("SELECT id FROM bodies")},
        "animals": {int(r[0]) for r in conn.execute("SELECT id FROM animals")},
    }
    marks = ",".join("?" * len(ACTIVE))
    active = {r[0] for r in conn.execute(f"SELECT uid FROM uploads WHERE status IN ({marks})", tuple(ACTIVE))}
    cutoff = time.time() - min_age_s
    removed = 0
    def live_upload(uid: str) -> bool:
        return conn.execute(f"SELECT 1 FROM uploads WHERE uid = ? AND status IN ({marks})",
                            (uid, *ACTIVE)).fetchone() is not None

    for top, base in storage.swept_bases():
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if stop is not None and stop.is_set():
                return removed
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
            if top not in crops:
                m = _PAGES_DIR_NAME.match(p.parent.name) if top == "pages" else \
                    (_ORIGINAL_NAME if top == "originals" else _DERIVED_NAME).match(p.name)
                if m and live_upload(m["uid"]):
                    continue
            log.info("cleanup: removing orphaned file %s", storage.rel(p))
            storage.safe_unlink(p)
            removed += 1
    pages = storage.root() / "pages"
    if pages.is_dir():
        for d in sorted(pages.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            try:
                if d.is_dir() and not any(d.iterdir()) and d.stat().st_mtime <= cutoff:
                    d.rmdir()
            except OSError:
                pass
    return removed


FTS_REBUILD_CHUNK = 500  # items re-indexed per transaction by _rebuild_fts_if_flagged


def _rebuild_fts_if_flagged(conn: sqlite3.Connection) -> int:
    """When db.init_db recreated items_fts (kv fts_rebuild = 1, a tokenizer change), index every item
    again (trashed ones too: they keep their row until purged). The flag is cleared only after the
    whole pass, so an interrupted rebuild runs again on the next start. Returns the items indexed."""
    from app.db import FTS_REBUILD_KEY

    if kv_get(conn, FTS_REBUILD_KEY, "0") != "1":
        return 0
    from app.documents.extract import refresh_fts

    ids = [int(r[0]) for r in conn.execute("SELECT id FROM items ORDER BY id")]
    done = failed = 0
    for i in range(0, len(ids), FTS_REBUILD_CHUNK):
        with tx(conn):  # refresh_fts's own transactions become savepoints of this one
            for item_id in ids[i:i + FTS_REBUILD_CHUNK]:
                try:
                    refresh_fts(conn, item_id)
                    done += 1
                except Exception:
                    failed += 1
                    log.exception("startup: could not re-index item %s for full-text search", item_id)
    with tx(conn):
        conn.execute("DELETE FROM kv WHERE key = ?", (FTS_REBUILD_KEY,))
    log.info("startup: full-text index rebuilt for the new tokenizer (%d items, %d failed)", done, failed)
    return done


DOC_RULES_VERSION = 3  # sections 11 and 30: the document detection rule set; kv doc_rules_version records the one applied
DOC_RULES_NEW_TYPES = ("traffic_challan", "parking_receipt", "id_card")  # the types rule sets 2 and 3 added


def _documents_for_new_rules(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Text-job payloads for the automatic 'other' documents whose text scores for a type a newer rule set
    added (section 30), when kv doc_rules_version is below DOC_RULES_VERSION. The owner's own types
    (documents.source = 'user') are never touched. Reads only; the caller queues and records the version."""
    try:
        have = int(kv_get(conn, "doc_rules_version", "1") or 1)
    except ValueError:
        have = 1
    if have >= DOC_RULES_VERSION:
        return []
    from app.documents.detect import detect

    out: list[dict[str, Any]] = []
    for r in conn.execute(
        "SELECT d.item_id, i.version, t.text FROM documents d JOIN items i ON i.id = d.item_id "
        "JOIN item_text t ON t.item_id = d.item_id WHERE d.doc_type = 'other' AND d.source = 'auto' "
        "AND i.deleted_at IS NULL ORDER BY d.item_id"
    ).fetchall():
        try:
            if detect(r["text"] or "").doc_type in DOC_RULES_NEW_TYPES:
                out.append(item_payload(int(r["item_id"]), int(r["version"])))
        except Exception:  # noqa: BLE001 - one odd text never blocks the start
            log.debug("startup: detection failed for item %s", r["item_id"], exc_info=True)
    return out


def _reprobe_in_background() -> dict[str, int]:
    """reprobe_plain on its own connection (worker.run_in_background; drain() waits for it in tests);
    records probe.PROBE_VERSION in kv when the pass completed."""
    from app.db import connect
    from app.media.probe import PROBE_VERSION

    conn = connect()
    try:
        out = reprobe_plain(conn)
        if out["complete"]:
            with tx(conn):
                kv_set(conn, PROBE_VERSION_KEY, PROBE_VERSION)
        return out
    finally:
        conn.close()


def recover_on_startup(conn: sqlite3.Connection) -> None:
    """Rebuild the full-text index when init_db asked for it (kv fts_rebuild); re-queue interrupted
    jobs; make sure every queued/processing upload has an ingest job and every item whose analysis,
    text or streaming copy is pending has its job; queue a reparse of the automatic 'other' documents a
    newer detection rule set now types (kv doc_rules_version, section 30); apply a changed guessing window
    (guess.apply_window_change); when kv probe_version is below probe.PROBE_VERSION, probe again on a
    background thread the plain files whose bytes claim a media type (reprobe_plain); bring stored faces
    up to the current face analysis (app.analysis.face_upkeep.start: analysed videos re-tracked once,
    photo faces measured for clarity); wipe data_dir/tmp."""
    _rebuild_fts_if_flagged(conn)
    n, exhausted = jobs.requeue_running(conn)
    if n:
        log.info("startup: re-queued %d interrupted jobs", n)
    if exhausted:
        from app import worker

        worker.finish_exhausted(conn, exhausted)
    kinds = ",".join(f"'{k}'" for k in ANALYSED_KINDS)
    reparse = _documents_for_new_rules(conn)
    with tx(conn):  # one transaction for every missing job
        added = jobs.enqueue_many(conn, "ingest", (
            {"upload_id": int(r["id"])}
            for r in conn.execute("SELECT id FROM uploads WHERE status IN ('queued', 'processing') ORDER BY id")
            .fetchall()
        ), unique=True)
        added += jobs.enqueue_many(conn, "analyze", (
            item_payload(r["id"], r["version"]) for r in conn.execute(
                f"SELECT id, version FROM items WHERE analysis_status = 'pending' AND kind IN ({kinds}) ORDER BY id"
            ).fetchall()
        ), unique=True)
        added += jobs.enqueue_many(conn, "text", (
            item_payload(r["id"], r["version"])
            for r in conn.execute("SELECT id, version FROM items WHERE text_status = 'pending' ORDER BY id").fetchall()
        ), unique=True)
        added += jobs.enqueue_many(conn, "video_preview", (
            item_payload(r["id"], r["version"]) for r in conn.execute(
                "SELECT id, version FROM items WHERE kind = 'video' AND preview_status = 'pending' ORDER BY id"
            ).fetchall()
        ), unique=True)
        if reparse:
            added += jobs.enqueue_many(conn, "text", reparse, unique=True)
            log.info("startup: %d document(s) queued for a reparse under detection rule set %d", len(reparse),
                     DOC_RULES_VERSION)
        kv_set(conn, "doc_rules_version", str(DOC_RULES_VERSION))
    if added:
        log.info("startup: queued %d missing jobs", added)
    try:
        from app.analysis import guess  # local import: guess needs nothing from ingest

        withdrawn = guess.apply_window_change(conn)
        if withdrawn:
            log.info("startup: withdrew %d guess(es) outside the guessing window", withdrawn)
    except Exception:
        log.exception("startup: the guess window check failed")
    try:
        from app.media.probe import PROBE_VERSION

        if int(kv_get(conn, PROBE_VERSION_KEY, "1") or 1) < PROBE_VERSION:
            if plain_media_count(conn):
                from app import worker

                worker.run_in_background("reprobe-plain", _reprobe_in_background)
            else:
                with tx(conn):
                    kv_set(conn, PROBE_VERSION_KEY, PROBE_VERSION)
    except Exception:
        log.exception("startup: could not start the re-probe of plain files")
    from app.analysis import face_upkeep  # section 10: videos to re-track, photo faces to measure

    face_upkeep.start(conn)
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


def purge(conn: sqlite3.Connection, item_id: int) -> None:
    """Delete an item for good, and a Live Photo's companion clip item with it (section 20)."""
    row = conn.execute("SELECT live_video_item_id FROM items WHERE id = ?", (item_id,)).fetchone()
    companion = int(row["live_video_item_id"]) if row is not None and row["live_video_item_id"] else None
    _purge_one(conn, item_id)
    if companion is not None:
        _purge_one(conn, companion)
