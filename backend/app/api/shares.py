"""Share links, the owner side (docs/ARCHITECTURE.md section 26). Router without a prefix, mounted at
/api/shares, every route behind require_user.

GET    ""        -> [ShareOut] newest first; revoked links stay listed for 30 days (revoked_at set)
POST   ""        {kind: album|items, album_id?, item_uids?, title?, password?, expires_days?, allow_download?}
                 -> ShareOut (201). 404 album_not_found / item_not_found, 422 vault_item (a vault item in
                 item_uids; an album share simply skips them), 422 no_items.
GET    /{id}     -> ShareOut
PATCH  /{id}     {title?, password?, expires_days?, allow_download?} -> ShareOut. An empty password
                 clears it; expires_days null (sent) means never; fields left out stay as they are.
DELETE /{id}     -> Ok: revokes (the public link answers 404 from then on).
ShareOut = {id, token, path, kind, album{id, name} | null, title, count, has_password, expires_at,
allow_download, created_at, revoked_at, view_count, last_viewed_at}; count is the live, non-vault
members. The browser shows location.origin + path.
"""
from __future__ import annotations

import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_db, require_user
from app.schemas import Ok
from app.sharing import links

router = APIRouter(dependencies=[Depends(require_user)])

MAX_TITLE = 120
MAX_PASSWORD = 200


class ShareCreate(BaseModel):
    kind: Literal["album", "items"]
    album_id: int | None = Field(default=None, ge=1)
    item_uids: list[str] = Field(default=[], max_length=2000)
    title: str = Field(default="", max_length=MAX_TITLE)
    password: str | None = Field(default=None, max_length=MAX_PASSWORD)
    expires_days: int | None = Field(default=None, ge=1, le=3650)
    allow_download: bool = True


class ShareUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=MAX_TITLE)
    password: str | None = Field(default=None, max_length=MAX_PASSWORD)
    expires_days: int | None = Field(default=None, ge=1, le=3650)
    allow_download: bool | None = None


class ShareAlbumRef(BaseModel):
    id: int
    name: str


class ShareOut(BaseModel):
    id: int
    token: str
    path: str
    kind: Literal["album", "items"]
    album: ShareAlbumRef | None = None
    title: str
    count: int
    has_password: bool
    expires_at: str | None = None
    allow_download: bool
    created_at: str
    revoked_at: str | None = None
    view_count: int
    last_viewed_at: str | None = None


def share_out(conn: sqlite3.Connection, row: sqlite3.Row) -> ShareOut:
    album = None
    if row["kind"] == "album" and row["album_id"] is not None and row["album_name"] is not None:
        album = ShareAlbumRef(id=int(row["album_id"]), name=row["album_name"])
    return ShareOut(
        id=int(row["id"]), token=row["token"], path=f"/s/{row['token']}", kind=row["kind"], album=album,
        title=row["title"] or "", count=links.count(conn, row), has_password=bool(row["password_hash"]),
        expires_at=row["expires_at"], allow_download=bool(row["allow_download"]), created_at=row["created_at"],
        revoked_at=row["revoked_at"], view_count=int(row["view_count"] or 0), last_viewed_at=row["last_viewed_at"],
    )


def _clean_title(title: str | None) -> str | None:
    if title is None:
        return None
    return " ".join(title.split())[:MAX_TITLE]


def _clean_password(password: str | None) -> str | None:
    if password is None:
        return None
    return password.strip() or None


@router.get("", response_model=list[ShareOut])
def list_shares(conn: sqlite3.Connection = Depends(get_db)) -> list[ShareOut]:
    return [share_out(conn, r) for r in links.listed(conn)]


@router.post("", response_model=ShareOut, status_code=201)
def create_share(body: ShareCreate, conn: sqlite3.Connection = Depends(get_db)) -> ShareOut:
    try:
        row = links.create(
            conn, kind=body.kind, album_id=body.album_id, item_uids=body.item_uids, title=_clean_title(body.title) or "",
            password=_clean_password(body.password), expires_days=body.expires_days, allow_download=body.allow_download,
        )
    except links.Refused as exc:
        raise HTTPException(status_code=exc.status, detail=exc.code) from None
    return share_out(conn, row)


@router.get("/{share_id}", response_model=ShareOut)
def get_share(share_id: int, conn: sqlite3.Connection = Depends(get_db)) -> ShareOut:
    row = links.get(conn, share_id)
    if row is None:
        raise HTTPException(status_code=404, detail="share_not_found")
    return share_out(conn, row)


@router.patch("/{share_id}", response_model=ShareOut)
def update_share(share_id: int, body: ShareUpdate, conn: sqlite3.Connection = Depends(get_db)) -> ShareOut:
    sent = body.model_fields_set
    row = links.update(
        conn, share_id, title=_clean_title(body.title), password=_clean_password(body.password),
        password_set="password" in sent, expires_days=body.expires_days, expires_set="expires_days" in sent,
        allow_download=body.allow_download,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="share_not_found")
    return share_out(conn, row)


@router.delete("/{share_id}", response_model=Ok)
def revoke_share(share_id: int, conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    row = links.get(conn, share_id)
    if row is None:
        raise HTTPException(status_code=404, detail="share_not_found")
    links.revoke(conn, share_id)
    return Ok()
