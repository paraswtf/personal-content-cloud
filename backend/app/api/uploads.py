"""Chunked, resumable upload API (docs/ARCHITECTURE.md section 6).

CONTRACT. Router without a prefix, mounted at /api/uploads; every route requires a logged-in
session (Depends(require_user), which also enforces the X-Requested-With CSRF header on writes).

POST   /api/uploads                     UploadCreate -> UploadOut (201)
    size <= settings.max_upload_bytes (413) and free disk via storage.has_room_for(size + bytes
    still owed by active 'uploading' rows), checked inside the insert transaction (507). If sha256
    is given and an item (live or in the trash) has it, the row is created directly as 'duplicate'
    pointing at that item and no bytes are sent; an item in the trash is restored first (message
    RESTORED_MESSAGE), on every path below too. Otherwise status 'uploading' with an empty staging
    file (storage.staging_path(uid)). `source` records the way in (web, share, camera, api).
PUT    /api/uploads/{id}/chunk?offset=N raw body -> ChunkOut
    Only while 'uploading'. offset must equal received_bytes, else 409
    {"detail": "offset_mismatch", "received_bytes": n}. Body length <= settings.chunk_size and
    offset+len <= declared size (413). Positioned write + fsync, then received_bytes. 507 when the
    disk is (nearly) full; 503 + Retry-After when too many chunks are in flight.
POST   /api/uploads/{id}/complete -> UploadOut
    received_bytes must equal size (409). The staged file is hashed in this worker thread WITHOUT
    the per-upload lock; the result is applied under the lock after re-checking the row. A known
    sha256 -> 'duplicate' (staged file deleted); otherwise 'queued' + an ingest job. Idempotent.
POST   /api/uploads/simple  multipart: file, source -> UploadOut (201)
    One request, up to settings.simple_upload_max_bytes (413), for scripts and tests.
GET    /api/uploads/{id} -> UploadOut;  GET /api/uploads?ids=a,b,c -> UploadList (max 200)
GET    /api/uploads/mine?client_id=... -> UploadList (latest 300 of that browser)
DELETE /api/uploads/{id} -> UploadOut (cancel while 'uploading'; otherwise unchanged)
Upload ids are unguessable 14-character uids.
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

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile

from app.schemas import SyncCheckIn, SyncCheckOut
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app import jobs, storage
from app.api.common import UPLOAD_SELECT, upload_out
from app.api.deps import get_db, require_user
from app.auth.sessions import SessionInfo, client_ip
from app.config import settings
from app.db import connect, tx
from app.schemas import ChunkOut, UploadCreate, UploadList, UploadOut
from app.util import new_uid, now_iso, safe_filename

log = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_user)])
SOURCES = ("web", "share", "camera", "api")

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
RESTORED_MESSAGE = "Restored from the trash (identical file)."

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


def _known_item(conn: sqlite3.Connection, sha: str | None) -> int | None:
    """The live item with this sha256 (a trashed one is restored by _restore_trashed first; one
    trashed again in between is left to ingest, so these bytes are never dropped here)."""
    if not sha:
        return None
    row = conn.execute("SELECT id FROM items WHERE sha256 = ? AND deleted_at IS NULL", (sha,)).fetchone()
    return int(row["id"]) if row else None


def _restore_trashed(conn: sqlite3.Connection, sha: str | None) -> bool:
    """Uploading a file again whose item is in the trash takes that item out of the trash (the
    upload then answers 'duplicate' with RESTORED_MESSAGE). Call outside a transaction, before the
    duplicate check. Returns True when an item was restored."""
    if not sha:
        return False
    row = conn.execute("SELECT id FROM items WHERE sha256 = ? AND deleted_at IS NOT NULL", (sha,)).fetchone()
    if row is None:
        return False
    from app import ingest

    ingest.restore(conn, int(row["id"]))
    return conn.execute("SELECT 1 FROM items WHERE id = ? AND deleted_at IS NULL", (row["id"],)).fetchone() is not None


@router.post("", response_model=UploadOut, status_code=201)
def create_upload(body: UploadCreate, request: Request, conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> UploadOut:
    if body.size > settings.max_upload_bytes:
        limit_gb = settings.max_upload_bytes / (1024 ** 3)
        raise HTTPException(status_code=413, detail=f"The file is larger than the {limit_gb:g} GB limit")
    filename = safe_filename(body.filename)
    client_id = (body.client_id or "").strip() or None
    mime = (body.mime or "").strip()[:128] or None
    now = now_iso()
    uid = new_uid()
    restored = _restore_trashed(conn, body.sha256)
    known = _known_item(conn, body.sha256)
    staging = storage.staging_path(uid)
    if known is None:
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.touch(exist_ok=False)
    try:
        with tx(conn):
            # Inside the write transaction (BEGIN IMMEDIATE), so concurrent creates see each
            # other's reservations and cannot all pass against the same free space.
            if known is None and not storage.has_room_for(body.size + _reserved_bytes(conn)):
                raise HTTPException(status_code=507, detail=NO_SPACE)
            conn.execute(
                "INSERT INTO uploads(uid, source, client_id, original_filename, size_bytes, received_bytes, mime, "
                "sha256, status, message, item_id, remote_addr, created_at, updated_at, pair_key) "
                "VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (uid, body.source, client_id, filename, body.size, mime, body.sha256 if known else None,
                 "duplicate" if known else "uploading", RESTORED_MESSAGE if known and restored else None, known,
                 client_ip(request), now, now, body.pair_key),
            )
    except BaseException:
        if known is None:
            storage.safe_unlink(staging)
        raise
    return upload_out(_load(conn, uid), session.vault_unlocked)


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


def _finish(conn: sqlite3.Connection, row: sqlite3.Row, digest: str) -> bool:
    """Apply a computed hash: 'duplicate' of a known item (restored first when it is in the trash), or
    'queued' with an ingest job. Returns True when the staged file should be deleted (duplicate)."""
    restored = _restore_trashed(conn, digest)
    now = now_iso()
    with tx(conn):
        known = _known_item(conn, digest)
        if known is not None:
            cur = conn.execute(
                "UPDATE uploads SET status = 'duplicate', sha256 = ?, item_id = ?, message = ?, updated_at = ? "
                "WHERE id = ? AND status = 'uploading'",
                (digest, known, RESTORED_MESSAGE if restored else None, now, row["id"]),
            )
            return cur.rowcount == 1
        cur = conn.execute(
            "UPDATE uploads SET status = 'queued', sha256 = ?, message = NULL, updated_at = ? "
            "WHERE id = ? AND status = 'uploading'",
            (digest, now, row["id"]),
        )
        if cur.rowcount == 1:
            jobs.enqueue(conn, "ingest", {"upload_id": int(row["id"])})
    return False


@router.post("/{upload_id}/complete", response_model=UploadOut)
def complete_upload(upload_id: str, conn: sqlite3.Connection = Depends(get_db),
                    session: SessionInfo = Depends(require_user)) -> UploadOut:
    vault_open = session.vault_unlocked
    with _upload_lock(upload_id):
        row = _load(conn, upload_id)
        if row["status"] != "uploading":
            return upload_out(row, vault_open)  # idempotent
        if row["received_bytes"] != row["size_bytes"]:
            raise HTTPException(status_code=409, detail="The upload is not complete yet")
        path = storage.staging_path(upload_id)
        try:
            on_disk = path.stat().st_size
        except FileNotFoundError:
            raise HTTPException(status_code=409, detail="The uploaded data is missing, please start again")
        if on_disk != row["size_bytes"]:
            raise HTTPException(status_code=409, detail="The uploaded data is incomplete, please start again")
    # Hash outside the per-upload lock: a status poll or cancel must not wait for a multi-GB read.
    # No chunk can change the file meanwhile (received_bytes == size refuses every offset).
    try:
        digest = _sha256_file(path)
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail="The uploaded data is missing, please start again")
    with _upload_lock(upload_id):
        row = _load(conn, upload_id)
        if row["status"] != "uploading":
            return upload_out(row, vault_open)  # cancelled or completed by a concurrent call
        duplicate = _finish(conn, row, digest)
        if duplicate:
            storage.safe_unlink(path)
        else:
            jobs.wake[jobs.QUEUE_OF["ingest"]].set()  # the job is committed now
        return upload_out(_load(conn, upload_id), vault_open)


@router.post("/simple", response_model=UploadOut, status_code=201)
def simple_upload(request: Request, file: UploadFile = File(...), source: str = Form("api"),
                  pair_key: str | None = Form(None),
                  conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> UploadOut:
    """Whole file in one multipart request (scripts and tests)."""
    if source not in SOURCES:
        raise HTTPException(status_code=422, detail="source must be one of web, share, camera, api")
    limit = settings.simple_upload_max_bytes
    uid = new_uid()
    staging = storage.staging_path(uid)
    staging.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    size = 0
    try:
        with open(staging, "xb") as out:
            while block := file.file.read(_HASH_BLOCK):
                size += len(block)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"The file is larger than the "
                                                                f"{limit // (1024 * 1024)} MB limit of this endpoint")
                h.update(block)
                try:
                    out.write(block)
                except OSError as e:
                    if e.errno in (errno.ENOSPC, errno.EDQUOT):
                        raise HTTPException(status_code=507, detail=NO_SPACE) from None
                    raise
            out.flush()
            os.fsync(out.fileno())
        if size == 0:
            raise HTTPException(status_code=400, detail="The file is empty")
        digest = h.hexdigest()
        now = now_iso()
        filename = safe_filename(file.filename or "upload")
        mime = (file.content_type or "").strip()[:128] or None
        restored = _restore_trashed(conn, digest)
        with tx(conn):
            known = _known_item(conn, digest)
            conn.execute(
                "INSERT INTO uploads(uid, source, client_id, original_filename, size_bytes, received_bytes, mime, "
                "sha256, status, message, item_id, remote_addr, created_at, updated_at, pair_key) "
                "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (uid, source, filename, size, size, mime, digest, "duplicate" if known else "queued",
                 RESTORED_MESSAGE if known and restored else None, known, client_ip(request), now, now,
                 (pair_key or "").strip()[:128] or None),
            )
            if known is None:
                upload_id = conn.execute("SELECT id FROM uploads WHERE uid = ?", (uid,)).fetchone()["id"]
                jobs.enqueue(conn, "ingest", {"upload_id": int(upload_id)})
    except BaseException:
        storage.safe_unlink(staging)
        raise
    if known is not None:
        storage.safe_unlink(staging)
    else:
        jobs.wake[jobs.QUEUE_OF["ingest"]].set()
    return upload_out(_load(conn, uid), session.vault_unlocked)


# ---------------------------------------------------------------------- read
@router.get("/mine", response_model=UploadList)
def my_uploads(client_id: str = Query(min_length=1, max_length=64),
               conn: sqlite3.Connection = Depends(get_db),
               session: SessionInfo = Depends(require_user)) -> UploadList:
    rows = conn.execute(
        f"{UPLOAD_SELECT} WHERE u.client_id = ? ORDER BY u.id DESC LIMIT ?", (client_id.strip(), MINE_LIMIT)
    ).fetchall()
    return UploadList(uploads=[upload_out(r, session.vault_unlocked) for r in rows])


@router.get("", response_model=UploadList)
def many_uploads(ids: str = Query(default="", max_length=MAX_IDS * 20),
                 conn: sqlite3.Connection = Depends(get_db),
                 session: SessionInfo = Depends(require_user)) -> UploadList:
    wanted = list(dict.fromkeys(i.strip() for i in ids.split(",") if i.strip()))
    if len(wanted) > MAX_IDS:
        raise HTTPException(status_code=400, detail=f"At most {MAX_IDS} ids per request")
    if not wanted:
        return UploadList(uploads=[])
    ph = ",".join("?" * len(wanted))
    rows = {r["uid"]: r for r in conn.execute(f"{UPLOAD_SELECT} WHERE u.uid IN ({ph})", wanted)}
    return UploadList(uploads=[upload_out(rows[i], session.vault_unlocked) for i in wanted if i in rows])


@router.get("/{upload_id}", response_model=UploadOut)
def get_upload(upload_id: str, conn: sqlite3.Connection = Depends(get_db),
               session: SessionInfo = Depends(require_user)) -> UploadOut:
    return upload_out(_load(conn, upload_id), session.vault_unlocked)


# -------------------------------------------------------------------- cancel
@router.delete("/{upload_id}", response_model=UploadOut)
def cancel_upload(upload_id: str, conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> UploadOut:
    with _upload_lock(upload_id):
        row = _load(conn, upload_id)
        if row["status"] != "uploading":
            return upload_out(row, session.vault_unlocked)
        with tx(conn):
            cur = conn.execute(
                "UPDATE uploads SET status = 'cancelled', message = NULL, updated_at = ? "
                "WHERE id = ? AND status = 'uploading'",
                (now_iso(), row["id"]),
            )
        if cur.rowcount == 1:
            storage.safe_unlink(storage.staging_path(upload_id))
        return upload_out(_load(conn, upload_id), session.vault_unlocked)


@router.post("/check", response_model=SyncCheckOut)
def sync_check(body: SyncCheckIn, session: SessionInfo = Depends(require_user),
               conn: sqlite3.Connection = Depends(get_db)) -> SyncCheckOut:
    """What the server already holds, for a device deciding what to send: item uids by file hash
    (live or in the trash) and by the upload pair key the device used (its asset identifier)."""
    known: dict[str, str] = {}
    shas = sorted({s.lower() for s in body.sha256 if len(s) == 64})
    for i in range(0, len(shas), 500):
        chunk = shas[i:i + 500]
        for r in conn.execute(f"SELECT sha256, uid FROM items WHERE sha256 IN ({','.join('?' * len(chunk))})", chunk):
            known[r["sha256"]] = r["uid"]
    assets: dict[str, str] = {}
    keys = sorted({a for a in body.asset_ids if a})
    for i in range(0, len(keys), 500):
        chunk = keys[i:i + 500]
        for r in conn.execute(
            f"SELECT u.pair_key, i.uid FROM uploads u JOIN items i ON i.id = u.item_id "
            f"WHERE u.pair_key IN ({','.join('?' * len(chunk))}) ORDER BY u.id", chunk
        ):
            assets.setdefault(r["pair_key"], r["uid"])
    return SyncCheckOut(known=known, assets=assets)
