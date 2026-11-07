"""Chunked, resumable upload API.

CONTRACT (implemented by the API agent). Router mounted at prefix /api/uploads, depends on
require_access. Protocol (docs/SPEC.md section "Upload protocol"):

POST   /api/uploads                     UploadCreate -> UploadOut (201)
    Validate size <= settings.max_upload_bytes (413) and free disk via
    storage.has_room_for(size + bytes still owed by active 'uploading' rows), checked inside the
    insert transaction (507 "The server is out of storage space"); clean the uploader name (app.util.clean_name,
    400 if empty), upsert uploaders by name_key. If sha256 is given and found in known_hashes,
    create the upload row directly as 'duplicate' (media_id = matched item, contributor added) and
    return it: the client then skips sending bytes. Otherwise create status 'uploading' with an
    empty staging file (storage.staging_path(uid)). uploads.admin = 1 when the admin is logged in
    (app.auth.is_admin): people metadata in such a file is then applied by name even when it was
    signed by another portal (app.export.people).
PUT   /api/uploads/{id}/chunk?offset=N raw body (application/octet-stream) -> ChunkOut
    Only while status 'uploading'. offset must equal received_bytes, else 409 with JSON
    {"detail": "offset_mismatch", "received_bytes": n}. Body length <= settings.chunk_size and
    offset+len <= declared size (413). Stream the request body (request.stream()) into the staging
    file opened r+b at that offset (no whole-chunk buffering beyond chunk_size), fsync, then
    update received_bytes. 507 when the disk is (nearly) full, before or during the write
    (received_bytes unchanged); 503 + Retry-After when too many chunks are in flight. A retried chunk with an offset below received_bytes for the same bytes
    is answered 409 as above, so the client resyncs.
POST   /api/uploads/{id}/complete -> UploadOut
    received_bytes must equal size (409 otherwise). Compute sha256 of the staged file; if it is in
    known_hashes -> status 'duplicate' (+ contributor), delete the staged file. Otherwise status
    'queued', store sha256, enqueue ingest {upload_id}. Idempotent: calling again returns the
    current state.
GET    /api/uploads/{id} -> UploadOut
GET    /api/uploads?ids=a,b,c -> UploadList   (max 200 ids; unknown ids are omitted)
GET    /api/uploads/mine?client_id=... -> UploadList  (latest 300 uploads of that browser)
DELETE /api/uploads/{id} -> UploadOut  cancel while 'uploading' (delete staged file); otherwise
    return the current state unchanged.
Upload ids are unguessable (14-char random uid) and act as the capability for that upload.
"""
from __future__ import annotations

import errno
import hashlib
import logging
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.changelog import context as changelog_context
from app.changelog import retention
from app.changelog.capture import claim
from app import jobs, storage
from app.api.common import UPLOAD_SELECT, upload_out
from app.api.deps import get_db, require_access, upload_change
from app.auth import client_ip, is_admin
from app.config import settings
from app.db import connect, tx
from app.schemas import ChunkOut, UploadCreate, UploadList, UploadOut
from app.util import clean_name, name_key, new_uid, now_iso, safe_filename

log = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_access), Depends(upload_change, scope="function")])

MAX_IDS = 200
MINE_LIMIT = 300
_HASH_BLOCK = 1024 * 1024
# An 'uploading' row reserves its remaining bytes while chunks keep arriving; an upload idle for
# longer stops blocking others (the per-chunk free-space check still guards it if it resumes).
RESERVE_IDLE_S = 3600
# At most this many chunk bodies are buffered at once (each up to settings.chunk_size); more get
# 503 + Retry-After, which the browser client retries with backoff.
MAX_CHUNKS_IN_FLIGHT = 64
_chunks_in_flight = 0  # only touched on the event loop thread, with no await in between
NO_SPACE = "The server is out of storage space"

# Per-upload locks so concurrent chunk / complete / cancel calls for one upload never interleave.
_locks_guard = threading.Lock()
_locks: dict[str, list] = {}  # uid -> [lock, users]


@contextmanager
def _upload_lock(uid: str) -> Iterator[None]:
    with _locks_guard:
        entry = _locks.setdefault(uid, [threading.Lock(), 0])
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _locks_guard:
            entry[1] -= 1
            if entry[1] == 0:
                _locks.pop(uid, None)


class _OffsetMismatch(Exception):
    def __init__(self, received_bytes: int) -> None:
        super().__init__("offset_mismatch")
        self.received_bytes = received_bytes


def _mismatch_response(received: int) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": "offset_mismatch", "received_bytes": received})


def _load(conn: sqlite3.Connection, uid: str) -> sqlite3.Row:
    row = conn.execute(f"{UPLOAD_SELECT} WHERE u.uid = ?", (uid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    return row


def _not_uploading(row: sqlite3.Row) -> HTTPException:
    return HTTPException(status_code=409, detail=f"This upload is no longer accepting data (status: {row['status']})")


def _too_large() -> HTTPException:
    return HTTPException(status_code=413, detail="Chunk is too large")


# -------------------------------------------------------------------- create
def _reserved_bytes(conn: sqlite3.Connection) -> int:
    """Bytes still owed by uploads that are actively sending (declared size minus received)."""
    return int(conn.execute(
        "SELECT COALESCE(SUM(size_bytes - received_bytes), 0) FROM uploads "
        "WHERE status = 'uploading' AND updated_at > ?",
        (now_iso(-RESERVE_IDLE_S),),
    ).fetchone()[0])


@router.post("", response_model=UploadOut, status_code=201)
def create_upload(body: UploadCreate, request: Request, conn: sqlite3.Connection = Depends(get_db)) -> UploadOut:
    if body.size > settings.max_upload_bytes:
        limit_gb = settings.max_upload_bytes / (1024 ** 3)
        raise HTTPException(status_code=413, detail=f"The file is larger than the {limit_gb:g} GB limit")
    name = clean_name(body.uploader_name)
    if not name:
        raise HTTPException(status_code=400, detail="Please enter your name")
    changelog_context.set_label(conn, name)
    filename = safe_filename(body.filename)
    client_id = (body.client_id or "").strip() or None
    mime = (body.mime or "").strip()[:128] or None
    now = now_iso()
    uid = new_uid()
    admin = 1 if is_admin(request) else 0  # before the write transaction: it reads the session epoch

    known = None
    if body.sha256:
        known = conn.execute("SELECT media_id FROM known_hashes WHERE sha256 = ?", (body.sha256,)).fetchone()
    staging = storage.staging_path(uid)
    if known is None and not storage.has_room_for(body.size):
        # Deleted files waiting in the trash give way to new uploads (never those under a day old).
        # Outside the write transaction: purging deletes files, which a rollback could not undo.
        retention.purge_for_space(conn, need_bytes=body.size)
    if known is None:
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.touch(exist_ok=False)
    try:
        with tx(conn):
            conn.execute(
                "INSERT INTO uploaders(name, name_key, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(name_key) DO NOTHING",
                (name, name_key(name), now),
            )
            uploader_id = conn.execute(
                "SELECT id FROM uploaders WHERE name_key = ?", (name_key(name),)
            ).fetchone()["id"]
            # Inside the write transaction (BEGIN IMMEDIATE), so concurrent creates see each
            # other's reservations and cannot all pass against the same free space.
            if known is None and not storage.has_room_for(body.size + _reserved_bytes(conn)):
                raise HTTPException(status_code=507, detail=NO_SPACE)
            if known is not None:
                conn.execute(
                    "INSERT INTO uploads(uid, uploader_id, client_id, original_filename, size_bytes, received_bytes, "
                    "mime, sha256, status, remote_addr, admin, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 0, ?, ?, 'duplicate', ?, ?, ?, ?)",
                    (uid, uploader_id, client_id, filename, body.size, mime, body.sha256,
                     client_ip(request), admin, now, now),
                )
                # Linked by an update, which the change history records (an undo of this upload then
                # marks it removed); the insert of an upload row itself is not recorded.
                conn.execute("UPDATE uploads SET media_id = ? WHERE uid = ?", (known["media_id"], uid))
                changelog_context.add_refs(conn, f"u:{uid}")
                cur = conn.execute(
                    "INSERT OR IGNORE INTO media_contributors(media_id, uploader_id, first_upload_at) VALUES (?, ?, ?)",
                    (known["media_id"], uploader_id, now),
                )
                if cur.rowcount == 0:
                    claim(conn, "media_contributors", {"media_id": known["media_id"], "uploader_id": uploader_id},
                          known["media_id"])
            else:
                conn.execute(
                    "INSERT INTO uploads(uid, uploader_id, client_id, original_filename, size_bytes, received_bytes, "
                    "mime, status, remote_addr, admin, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 0, ?, 'uploading', ?, ?, ?, ?)",
                    (uid, uploader_id, client_id, filename, body.size, mime, client_ip(request), admin, now, now),
                )
    except BaseException:
        if known is None:
            storage.safe_unlink(staging)
        raise
    return upload_out(_load(conn, uid))


# --------------------------------------------------------------------- chunk
def _precheck_chunk(uid: str, offset: int, declared_len: int | None) -> int:
    """Cheap checks before reading the body. Returns the byte limit for this chunk."""
    conn = connect()
    try:
        row = _load(conn, uid)
    finally:
        conn.close()
    if row["status"] != "uploading":
        raise _not_uploading(row)
    if offset != row["received_bytes"]:
        raise _OffsetMismatch(row["received_bytes"])
    limit = min(settings.chunk_size, row["size_bytes"] - offset)
    if declared_len is not None and declared_len > limit:
        raise _too_large()
    return limit


def _chunk_fits(nbytes: int) -> bool:
    """Room for one more chunk. Half the free-space margin: space for this upload was reserved
    at create, so only a disk filled by something else stops it here."""
    free, _ = storage.disk_usage()
    return free - nbytes >= settings.min_free_disk_bytes // 2


def _write_chunk(uid: str, offset: int, data: bytes | bytearray) -> int:
    """Positioned write + fsync + received_bytes update, serialised per upload."""
    with _upload_lock(uid):
        conn = connect()
        try:
            row = _load(conn, uid)
            if row["status"] != "uploading":
                raise _not_uploading(row)
            if offset != row["received_bytes"]:
                raise _OffsetMismatch(row["received_bytes"])
            if offset + len(data) > row["size_bytes"]:
                raise _too_large()
            if not _chunk_fits(len(data)):
                raise HTTPException(status_code=507, detail=NO_SPACE)
            path = storage.staging_path(uid)
            try:
                f = open(path, "r+b")
            except FileNotFoundError:
                if offset != 0:
                    raise HTTPException(status_code=409, detail="The uploaded data is missing, please start again")
                path.parent.mkdir(parents=True, exist_ok=True)
                f = open(path, "w+b")
            with f:
                try:
                    f.seek(offset)
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                except OSError as e:
                    if e.errno not in (errno.ENOSPC, errno.EDQUOT):
                        raise
                    # received_bytes is untouched, so the client can resend this chunk later.
                    try:
                        f.truncate(offset)
                    except OSError:
                        pass
                    log.warning("upload %s: disk full while writing a chunk", uid)
                    raise HTTPException(status_code=507, detail=NO_SPACE) from None
            received = offset + len(data)
            try:
                with tx(conn):
                    cur = conn.execute(
                        "UPDATE uploads SET received_bytes = ?, updated_at = ? "
                        "WHERE id = ? AND status = 'uploading' AND received_bytes = ?",
                        (received, now_iso(), row["id"], offset),
                    )
            except sqlite3.OperationalError as e:
                if "full" not in str(e).lower():
                    raise
                raise HTTPException(status_code=507, detail=NO_SPACE) from None
            if cur.rowcount != 1:
                raise _not_uploading(_load(conn, uid))
            return received
        finally:
            conn.close()


async def _drain(request: Request) -> None:
    """Read and discard the request body (capped) before answering an early error."""
    cap, n = settings.chunk_size + 64 * 1024, 0
    async for piece in request.stream():
        n += len(piece)
        if n > cap:
            break


@router.put("/{upload_id}/chunk", response_model=ChunkOut)
async def put_chunk(upload_id: str, request: Request, offset: int = Query(ge=0)):
    global _chunks_in_flight
    declared: int | None = None
    if request.headers.get("content-length"):
        try:
            declared = int(request.headers["content-length"])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")
    try:
        try:
            limit = await run_in_threadpool(_precheck_chunk, upload_id, offset, declared)
        except (_OffsetMismatch, HTTPException) as exc:
            if isinstance(exc, _OffsetMismatch) or exc.status_code in (404, 409):
                await _drain(request)  # answer after the body so the client sees the status, not a reset
            raise
        if _chunks_in_flight >= MAX_CHUNKS_IN_FLIGHT:
            await _drain(request)
            raise HTTPException(status_code=503, detail="The server is busy, retrying shortly",
                                headers={"Retry-After": "2"})
        _chunks_in_flight += 1
        try:
            buf = bytearray()
            async for piece in request.stream():
                if len(buf) + len(piece) > limit:
                    raise _too_large()
                buf.extend(piece)
            if not buf:
                raise HTTPException(status_code=400, detail="Empty chunk")
            received = await run_in_threadpool(_write_chunk, upload_id, offset, buf)  # no extra copy
        finally:
            _chunks_in_flight -= 1
    except _OffsetMismatch as exc:
        return _mismatch_response(exc.received_bytes)
    return ChunkOut(received_bytes=received)


# ------------------------------------------------------------------ complete
def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_HASH_BLOCK):
            h.update(block)
    return h.hexdigest()


@router.post("/{upload_id}/complete", response_model=UploadOut)
def complete_upload(upload_id: str, conn: sqlite3.Connection = Depends(get_db)) -> UploadOut:
    with _upload_lock(upload_id):
        row = _load(conn, upload_id)
        if row["status"] != "uploading":
            return upload_out(row)  # idempotent
        if row["received_bytes"] != row["size_bytes"]:
            raise HTTPException(status_code=409, detail="The upload is not complete yet")
        path = storage.staging_path(upload_id)
        try:
            on_disk = path.stat().st_size
        except FileNotFoundError:
            raise HTTPException(status_code=409, detail="The uploaded data is missing, please start again")
        if on_disk != row["size_bytes"]:
            raise HTTPException(status_code=409, detail="The uploaded data is incomplete, please start again")
        digest = _sha256_file(path)
        now = now_iso()
        duplicate = False
        who = conn.execute("SELECT name FROM uploaders WHERE id = ?", (row["uploader_id"],)).fetchone()
        changelog_context.set_label(conn, who["name"] if who else None)
        changelog_context.add_refs(conn, f"u:{upload_id}")
        with tx(conn):
            known = conn.execute("SELECT media_id FROM known_hashes WHERE sha256 = ?", (digest,)).fetchone()
            if known is not None:
                duplicate = True
                cur = conn.execute(
                    "UPDATE uploads SET status = 'duplicate', sha256 = ?, media_id = ?, message = NULL, updated_at = ? "
                    "WHERE id = ? AND status = 'uploading'",
                    (digest, known["media_id"], now, row["id"]),
                )
                if cur.rowcount == 1:
                    cur2 = conn.execute(
                        "INSERT OR IGNORE INTO media_contributors(media_id, uploader_id, first_upload_at) "
                        "VALUES (?, ?, ?)",
                        (known["media_id"], row["uploader_id"], now),
                    )
                    if cur2.rowcount == 0:
                        claim(conn, "media_contributors",
                              {"media_id": known["media_id"], "uploader_id": row["uploader_id"]}, known["media_id"])
            else:
                cur = conn.execute(
                    "UPDATE uploads SET status = 'queued', sha256 = ?, message = NULL, updated_at = ? "
                    "WHERE id = ? AND status = 'uploading'",
                    (digest, now, row["id"]),
                )
                if cur.rowcount == 1:
                    jobs.enqueue(conn, "ingest", {"upload_id": row["id"]})
        if duplicate:
            storage.safe_unlink(path)
        else:
            jobs.wake[jobs.QUEUE_OF["ingest"]].set()  # the job is committed now
        return upload_out(_load(conn, upload_id))


# ---------------------------------------------------------------------- read
@router.get("/mine", response_model=UploadList)
def my_uploads(client_id: str = Query(min_length=1, max_length=64),
               conn: sqlite3.Connection = Depends(get_db)) -> UploadList:
    rows = conn.execute(
        f"{UPLOAD_SELECT} WHERE u.client_id = ? ORDER BY u.id DESC LIMIT ?", (client_id.strip(), MINE_LIMIT)
    ).fetchall()
    return UploadList(uploads=[upload_out(r) for r in rows])


@router.get("", response_model=UploadList)
def many_uploads(ids: str = Query(default="", max_length=MAX_IDS * 20),
                 conn: sqlite3.Connection = Depends(get_db)) -> UploadList:
    wanted = list(dict.fromkeys(i.strip() for i in ids.split(",") if i.strip()))
    if len(wanted) > MAX_IDS:
        raise HTTPException(status_code=400, detail=f"At most {MAX_IDS} ids per request")
    if not wanted:
        return UploadList(uploads=[])
    ph = ",".join("?" * len(wanted))
    rows = {r["uid"]: r for r in conn.execute(f"{UPLOAD_SELECT} WHERE u.uid IN ({ph})", wanted)}
    return UploadList(uploads=[upload_out(rows[i]) for i in wanted if i in rows])


@router.get("/{upload_id}", response_model=UploadOut)
def get_upload(upload_id: str, conn: sqlite3.Connection = Depends(get_db)) -> UploadOut:
    return upload_out(_load(conn, upload_id))


# -------------------------------------------------------------------- cancel
@router.delete("/{upload_id}", response_model=UploadOut)
def cancel_upload(upload_id: str, conn: sqlite3.Connection = Depends(get_db)) -> UploadOut:
    with _upload_lock(upload_id):
        row = _load(conn, upload_id)
        if row["status"] != "uploading":
            return upload_out(row)
        with tx(conn):
            cur = conn.execute(
                "UPDATE uploads SET status = 'cancelled', message = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'uploading'",
                (now_iso(), row["id"]),
            )
        if cur.rowcount == 1:
            storage.safe_unlink(storage.staging_path(upload_id))
        return upload_out(_load(conn, upload_id))
