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
POST   /api/uploads/simple  multipart: file, source, pair_key, resource -> UploadOut (201)
    One request, up to settings.simple_upload_max_bytes (413), for scripts and tests.
GET    /api/uploads/{id} -> UploadOut;  GET /api/uploads?ids=a,b,c -> UploadList (max 200)
GET    /api/uploads/mine?client_id=... -> UploadList (latest 300 of that browser)
DELETE /api/uploads/{id} -> UploadOut (cancel while 'uploading'; otherwise unchanged)
POST   /api/uploads/check  SyncCheckIn -> SyncCheckOut (section 21 and 28)
    known: sha256 -> item uid (live or in the trash); assets: pair key -> item uid; pending: the
    caller's own 'uploading' rows per asked pair key; trashed: asked pair keys with an item in the trash.
Upload ids are unguessable 14-character uids.

Device resume (section 28). Every create records uploads.session_id (the session that made it) and
uploads.resource ('still', 'video' or 'file'). A create from a bearer session with a pair_key is
idempotent: an 'uploading' row of the same session, pair_key, resource and size is answered as it
is (200, its received_bytes tells the client where to continue) instead of opening a second staging
file; 'uploading' rows of that key with another size, or with lost staged bytes, are cancelled
first and their staging files removed. Cookie sessions (the web, which has no pair_key) always get
a new row.
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

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app import jobs, storage
from app.api.common import UPLOAD_SELECT, upload_out
from app.api.deps import get_db, require_user
from app.auth.sessions import SessionInfo, client_ip
from app.config import settings
from app.db import connect, tx
from app.schemas import ChunkOut, SyncCheckIn, SyncCheckOut, SyncPending, UploadCreate, UploadList, UploadOut
from app.util import new_uid, now_iso, safe_filename

log = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_user)])
SOURCES = ("web", "share", "camera", "api")
RESOURCES = ("still", "video", "file")

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
# Uploads whose staged file complete_upload is hashing outside its lock: _drop_staging leaves those
# files alone, and complete_upload removes them itself when it finds the row cancelled afterwards.
_hashing: set[str] = set()


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


def _pair_key(value: str | None) -> str | None:
    return (value or "").strip()[:128] or None


def _resumable(conn: sqlite3.Connection, session_id: int, pair_key: str, resource: str) -> list[sqlite3.Row]:
    """The session's unfinished uploads of one half of one asset (index uploads_asset), the one with
    the most bytes first."""
    return conn.execute(
        "SELECT id, uid, size_bytes, received_bytes FROM uploads "
        "WHERE session_id = ? AND pair_key = ? AND resource = ? AND status = 'uploading' "
        "ORDER BY received_bytes DESC, id DESC",
        (session_id, pair_key, resource),
    ).fetchall()


def _staged_intact(row: sqlite3.Row) -> bool:
    """The staging file still holds at least the bytes the row says it received."""
    try:
        return storage.staging_path(row["uid"]).stat().st_size >= int(row["received_bytes"])
    except FileNotFoundError:
        return int(row["received_bytes"]) == 0


def _cancel_rows(conn: sqlite3.Connection, rows: list[sqlite3.Row], now: str) -> list[str]:
    """Mark 'uploading' rows cancelled (inside the caller's transaction). Returns the uids that changed;
    the caller removes their staging files after the commit, under each upload's lock."""
    cancelled: list[str] = []
    for r in rows:
        cur = conn.execute(
            "UPDATE uploads SET status = 'cancelled', message = NULL, updated_at = ? WHERE id = ? AND status = 'uploading'",
            (now, r["id"]),
        )
        if cur.rowcount == 1:
            cancelled.append(r["uid"])
    return cancelled


def _drop_staging(uids: list[str]) -> None:
    for uid in uids:
        # Under the upload's lock, so a chunk being written for it finishes first (it then finds
        # the row cancelled). Never called inside a transaction: _write_chunk takes this lock first.
        with _upload_lock(uid):
            with _locks_guard:
                hashing = uid in _hashing
            if hashing:
                continue  # complete_upload is reading it; it unlinks the file once it sees the cancellation
            storage.safe_unlink(storage.staging_path(uid))


@router.post("", response_model=UploadOut, status_code=201)
def create_upload(body: UploadCreate, request: Request, response: Response,
                  conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> UploadOut:
    """Open an upload (201), or hand back the device's unfinished one for the same asset half (200).

    A bearer session with a pair_key gets the idempotent form of section 28. A known sha256 answers
    duplicate at once and cancels every unfinished upload of that key and resource, including one
    the device may still be sending (the current clients send one request at a time, so that row is
    always the stale one)."""
    if body.size > settings.max_upload_bytes:
        limit_gb = settings.max_upload_bytes / (1024 ** 3)
        raise HTTPException(status_code=413, detail=f"The file is larger than the {limit_gb:g} GB limit")
    filename = safe_filename(body.filename)
    client_id = (body.client_id or "").strip() or None
    mime = (body.mime or "").strip()[:128] or None
    pair_key = _pair_key(body.pair_key)
    resource = body.resource
    # Idempotent create (section 28): a device that lost the reply, or was killed and relaunched,
    # asks again with the same key and continues the upload it already has.
    resume = bool(session.bearer and pair_key)
    now = now_iso()
    uid = new_uid()
    restored = _restore_trashed(conn, body.sha256)
    known = _known_item(conn, body.sha256)
    staging = storage.staging_path(uid)
    if known is None:
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.touch(exist_ok=False)
    reused: str | None = None
    cancelled: list[str] = []
    try:
        with tx(conn):
            # Lookup, cancellation and insert share one write transaction (BEGIN IMMEDIATE): two
            # creates for the same key (a retry racing the first call) end up with one row.
            if resume:
                rows = _resumable(conn, session.id, pair_key, resource)
                keep = None
                if known is None:
                    keep = next((r for r in rows if int(r["size_bytes"]) == body.size and _staged_intact(r)), None)
                # Everything else of this key is stale: another size (the asset changed), lost staged
                # bytes, a second row from an older race, or any row once the hash is known.
                cancelled = _cancel_rows(conn, [r for r in rows if keep is None or r["id"] != keep["id"]], now)
                if keep is not None:
                    conn.execute("UPDATE uploads SET updated_at = ? WHERE id = ?", (now, keep["id"]))
                    reused = keep["uid"]
            if reused is None:
                # Inside the write transaction, so concurrent creates see each other's reservations
                # and cannot all pass against the same free space (rows cancelled above no longer count).
                if known is None and not storage.has_room_for(body.size + _reserved_bytes(conn)):
                    raise HTTPException(status_code=507, detail=NO_SPACE)
                conn.execute(
                    "INSERT INTO uploads(uid, source, client_id, original_filename, size_bytes, received_bytes, mime, "
                    "sha256, status, message, item_id, remote_addr, created_at, updated_at, pair_key, session_id, "
                    "resource) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (uid, body.source, client_id, filename, body.size, mime, body.sha256 if known else None,
                     "duplicate" if known else "uploading", RESTORED_MESSAGE if known and restored else None, known,
                     client_ip(request), now, now, pair_key, session.id, resource),
                )
    except BaseException:
        if known is None:
            storage.safe_unlink(staging)
        raise
    _drop_staging(cancelled)
    if reused is not None:
        storage.safe_unlink(staging)  # the fresh staging file was not needed
        response.status_code = 200
        return upload_out(_load(conn, reused), session.vault_unlocked)
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
        with _locks_guard:
            _hashing.add(upload_id)
    # Hash outside the per-upload lock: a status poll or cancel must not wait for a multi-GB read.
    # No chunk can change the file meanwhile (received_bytes == size refuses every offset), and a
    # create that cancels this row meanwhile leaves the file to us (_hashing).
    try:
        digest = _sha256_file(path)
    except FileNotFoundError:
        raise HTTPException(status_code=409, detail="The uploaded data is missing, please start again")
    finally:
        with _locks_guard:
            _hashing.discard(upload_id)
    with _upload_lock(upload_id):
        row = _load(conn, upload_id)
        if row["status"] != "uploading":
            if row["status"] == "cancelled":
                storage.safe_unlink(path)  # cancelled while hashing: the cancel skipped the file for us
            return upload_out(row, vault_open)  # cancelled or completed by a concurrent call
        duplicate = _finish(conn, row, digest)
        if duplicate:
            storage.safe_unlink(path)
        else:
            jobs.wake[jobs.QUEUE_OF["ingest"]].set()  # the job is committed now
        return upload_out(_load(conn, upload_id), vault_open)


@router.post("/simple", response_model=UploadOut, status_code=201)
def simple_upload(request: Request, file: UploadFile = File(...), source: str = Form("api"),
                  pair_key: str | None = Form(None), resource: str = Form("file"),
                  conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> UploadOut:
    """Whole file in one multipart request (scripts and tests)."""
    if source not in SOURCES:
        raise HTTPException(status_code=422, detail="source must be one of web, share, camera, api")
    if resource not in RESOURCES:
        raise HTTPException(status_code=422, detail="resource must be one of still, video, file")
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
                "sha256, status, message, item_id, remote_addr, created_at, updated_at, pair_key, session_id, resource) "
                "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (uid, source, filename, size, size, mime, digest, "duplicate" if known else "queued",
                 RESTORED_MESSAGE if known and restored else None, known, client_ip(request), now, now,
                 _pair_key(pair_key), session.id, resource),
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
    (live or in the trash) and by the upload pair key the device used (its asset identifier); the
    caller's own unfinished uploads of those assets (to continue them from the server's offset);
    and the assets with an item in the trash (still known, so never re-sent, but shown apart so the
    owner decides before deleting them from the phone)."""
    known: dict[str, str] = {}
    shas = sorted({s.lower() for s in body.sha256 if len(s) == 64})
    for i in range(0, len(shas), 500):
        chunk = shas[i:i + 500]
        for r in conn.execute(f"SELECT sha256, uid FROM items WHERE sha256 IN ({','.join('?' * len(chunk))})", chunk):
            known[r["sha256"]] = r["uid"]
    assets: dict[str, str] = {}
    trashed: set[str] = set()
    pending: dict[str, list[SyncPending]] = {}
    keys = sorted({a for a in body.asset_ids if a})
    for i in range(0, len(keys), 500):
        chunk = keys[i:i + 500]
        ph = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT u.pair_key, i.uid, i.deleted_at FROM uploads u JOIN items i ON i.id = u.item_id "
            f"WHERE u.pair_key IN ({ph}) ORDER BY u.id", chunk
        ):
            assets.setdefault(r["pair_key"], r["uid"])
            if r["deleted_at"] is not None:
                trashed.add(r["pair_key"])
        for r in conn.execute(
            f"SELECT uid, pair_key, resource, received_bytes, size_bytes FROM uploads "
            f"WHERE session_id = ? AND pair_key IN ({ph}) AND status = 'uploading' ORDER BY id",
            [session.id, *chunk],
        ):
            pending.setdefault(r["pair_key"], []).append(SyncPending(
                upload_id=r["uid"], resource=r["resource"] or "file",
                received_bytes=int(r["received_bytes"]), size=int(r["size_bytes"])))
    return SyncCheckOut(known=known, assets=assets, pending=pending, trashed=sorted(trashed))
