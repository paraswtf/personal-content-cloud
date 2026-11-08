"""Share links, the public side (docs/ARCHITECTURE.md section 26). No session: every route checks the
share token itself and never answers 401. The router carries its full paths (/api/s and /files/s), so
main.py mounts it with an empty prefix; it sits before the SPA catch-all.

GET  /api/s/{token}                    -> SharePublic{title, kind, count, allow_download, expires_at, items[]}
                                          403 share_password_required, 404 share_not_found (unknown or
                                          revoked), 410 share_expired. Bumps view_count and last_viewed_at.
POST /api/s/{token}/unlock {password}  CSRF header required; 403 invalid_password, 429 too_many_attempts
                                          (ten failures per IP per ten minutes); sets pcc_share_<token>
                                          (7 days) and answers SharePublic.
GET|HEAD /files/s/{token}/thumb/{uid}, /preview/{uid}, /video/{uid} (the web copy when done, else the
     original), /original/{uid} (only with allow_download; ?download=1 adds the attachment disposition):
     the item must be a live, non-vault member; private, max-age=3600 and the sandboxing CSP.
ShareItem = {uid, kind, width, height, duration_s, taken_at, filename, urls{thumb, preview, video, download}}.
"""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import storage
from app.api.deps import check_csrf, get_db
from app.api.files import CSP, _send, content_disposition
from app.auth import passwords
from app.auth.sessions import client_ip
from app.sharing import links
from app.util import media_file_name

router = APIRouter()

SHARE_CACHE = "private, max-age=3600"
METHODS = ["GET", "HEAD"]


class ShareItemUrls(BaseModel):
    thumb: str
    preview: str
    video: str | None = None
    download: str | None = None


class ShareItem(BaseModel):
    uid: str
    kind: str
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    taken_at: str | None = None
    filename: str
    urls: ShareItemUrls


class SharePublic(BaseModel):
    title: str
    kind: str
    count: int
    allow_download: bool
    expires_at: str | None = None
    items: list[ShareItem]


class UnlockIn(BaseModel):
    password: str = Field(min_length=1, max_length=200)


def _item_out(token: str, allow_download: bool, row: sqlite3.Row) -> ShareItem:
    base = f"/files/s/{token}"
    v = int(row["version"])
    urls = ShareItemUrls(thumb=f"{base}/thumb/{row['uid']}?v={v}", preview=f"{base}/preview/{row['uid']}?v={v}")
    if row["kind"] == "video":
        urls.video = f"{base}/video/{row['uid']}?v={v}"
    if allow_download:
        urls.download = f"{base}/original/{row['uid']}?v={v}&download=1"
    return ShareItem(
        uid=row["uid"], kind=row["kind"], width=row["width"], height=row["height"], duration_s=row["duration_s"],
        taken_at=row["taken_at"], filename=media_file_name(row["original_filename"], row["ext"]), urls=urls,
    )


def _public(conn: sqlite3.Connection, row: sqlite3.Row) -> SharePublic:
    items = links.members(conn, row)
    allow = bool(row["allow_download"])
    return SharePublic(
        title=row["title"] or "", kind=row["kind"], count=len(items), allow_download=allow, expires_at=row["expires_at"],
        items=[_item_out(row["token"], allow, r) for r in items],
    )


@router.get("/api/s/{token}", response_model=SharePublic)
def public_share(token: str, request: Request, conn: sqlite3.Connection = Depends(get_db)) -> SharePublic:
    row = links.resolve(conn, token)
    links.require_unlocked(request, row)
    links.record_view(conn, int(row["id"]))
    return _public(conn, row)


@router.post("/api/s/{token}/unlock", response_model=SharePublic, dependencies=[Depends(check_csrf)])
def unlock_share(token: str, body: UnlockIn, request: Request, response: Response,
                 conn: sqlite3.Connection = Depends(get_db)) -> SharePublic:
    row = links.resolve(conn, token)
    if row["password_hash"]:
        ip = client_ip(request)
        if not passwords.claim_share_attempt(ip):
            raise HTTPException(status_code=429, detail="too_many_attempts")
        if not passwords.verify_password(row["password_hash"], body.password):
            raise HTTPException(status_code=403, detail="invalid_password")
        passwords.reset_share_failures(ip)
        links.set_unlock_cookie(response, request, row["token"])
    links.record_view(conn, int(row["id"]))
    return _public(conn, row)


def _member_file(conn: sqlite3.Connection, request: Request, token: str, uid: str) -> tuple[sqlite3.Row, sqlite3.Row]:
    share = links.resolve(conn, token)
    links.require_unlocked(request, share)
    item = links.member(conn, share, uid)
    if item is None:
        raise HTTPException(status_code=404, detail="File not found")
    return share, item


@router.api_route("/files/s/{token}/thumb/{uid}", methods=METHODS)
def share_thumb(token: str, uid: str, request: Request, conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    _, item = _member_file(conn, request, token, uid)
    return _send(storage.thumb_path(item["uid"], item["version"]), "image/webp", SHARE_CACHE)


@router.api_route("/files/s/{token}/preview/{uid}", methods=METHODS)
def share_preview(token: str, uid: str, request: Request, conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    _, item = _member_file(conn, request, token, uid)
    return _send(storage.preview_path(item["uid"], item["version"]), "image/webp", SHARE_CACHE)


@router.api_route("/files/s/{token}/video/{uid}", methods=METHODS)
def share_video(token: str, uid: str, request: Request, conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    _, item = _member_file(conn, request, token, uid)
    if item["kind"] != "video":
        raise HTTPException(status_code=404, detail="File not found")
    if item["preview_status"] == "done":
        web = storage.video_preview_path(item["uid"], item["version"])
        if web.is_file():
            return _send(web, "video/mp4", SHARE_CACHE)
    return _send(storage.original_path(item["uid"], item["version"], item["ext"]), item["mime"] or "video/mp4", SHARE_CACHE)


@router.api_route("/files/s/{token}/original/{uid}", methods=METHODS)
def share_original(token: str, uid: str, request: Request, download: int = 0,
                   conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    share, item = _member_file(conn, request, token, uid)
    if not share["allow_download"]:
        raise HTTPException(status_code=404, detail="File not found")
    mime = item["mime"] or "application/octet-stream"
    if mime.startswith("text/") and "charset" not in mime:
        mime += "; charset=utf-8"
    extra = {"content-security-policy": CSP}
    if download:
        extra["content-disposition"] = content_disposition(media_file_name(item["original_filename"], item["ext"]))
    return _send(storage.original_path(item["uid"], item["version"], item["ext"]), mime, SHARE_CACHE, extra)
