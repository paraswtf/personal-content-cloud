"""Albums (docs/ARCHITECTURE.md section 24): /api/albums.

GET    /api/albums?q&sort=updated|name|created&cursor&limit  AlbumPage{items, next_cursor, total}
POST   /api/albums {name, description?, item_uids?}            AlbumOut (201); 409 album_exists
GET    /api/albums/{id}                                        AlbumOut; 404
PATCH  /api/albums/{id} {name?, description?, cover_item_uid?} AlbumOut; null clears the chosen cover;
                                                               422 cover_not_member; 409 album_exists
DELETE /api/albums/{id}                                        Ok (the items stay)
POST   /api/albums/{id}/items {uids}                           {added, skipped}: unknown, trashed and
                                                               companion uids are skipped, so are members
DELETE /api/albums/{id}/items {uids}                           {removed}

Members are listed by GET /api/items?album=<id> (and the timeline buckets with album=), so an album is
the library with one more filter; ItemDetail.albums names the albums of an item and the album and
unalbum bulk actions live in api/library.py. Vault members are added or removed only while the
session's vault is open (as the bulk path does); the q search is a substring LIKE on name_key, which
albums_name cannot serve (fine at a few thousand albums). AlbumOut counts live members on the normal side (count)
and in the vault (vault_count, a number only); cover_url is the chosen cover's thumb when it is a live
normal-side member, else the newest such member's; newest and oldest are their sort keys.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.common import url_version
from app.api.deps import get_db, require_user
from app.api.library import _keyset
from app.auth.sessions import SessionInfo
from app.db import tx
from app.schemas import Ok
from app.urls import thumb_url
from app.util import encode_cursor, name_key, now_iso

router = APIRouter(dependencies=[Depends(require_user)])

NAME_MAX = 120
DESCRIPTION_MAX = 2000
# A member counts when its item is live, not a Live Photo companion, on the given vault side.
_MEMBER = "i.deleted_at IS NULL AND i.companion_of IS NULL AND i.nsfw = {nsfw}"
_SORT_KEY = "COALESCE(i.taken_at, i.created_at)"
SORTS: dict[str, tuple[str, str, str]] = {
    # name: (sort column, direction, keyset comparison)
    "updated": ("a.updated_at", "DESC", "<"),
    "name": ("a.name_key", "ASC", ">"),
    "created": ("a.created_at", "DESC", "<"),
}


# ------------------------------------------------------------------ models
class AlbumIn(BaseModel):
    name: str = Field(min_length=1, max_length=NAME_MAX)
    description: str = Field(default="", max_length=DESCRIPTION_MAX)
    item_uids: list[str] = Field(default=[], max_length=1000)


class AlbumUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=NAME_MAX)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)
    cover_item_uid: str | None = None  # null clears the chosen cover; absent leaves it


class AlbumItemsIn(BaseModel):
    uids: list[str] = Field(min_length=1, max_length=1000)


class AlbumOut(BaseModel):
    id: int
    name: str
    description: str
    count: int
    vault_count: int
    cover_url: str | None
    newest: str | None
    oldest: str | None
    created_at: str
    updated_at: str


class AlbumPage(BaseModel):
    items: list[AlbumOut]
    next_cursor: str | None = None
    total: int


class AlbumItemsOut(BaseModel):
    added: int
    skipped: int


class AlbumRemovedOut(BaseModel):
    removed: int


# ------------------------------------------------------------------ helpers
def _clean_name(raw: str) -> str:
    name = " ".join(raw.split())
    if not name or len(name) > NAME_MAX:
        raise HTTPException(status_code=422, detail=f"An album name is 1 to {NAME_MAX} characters")
    return name


def _ensure_free_name(conn: sqlite3.Connection, key: str, exclude_id: int | None = None) -> None:
    row = conn.execute(
        "SELECT id FROM albums WHERE name_key = ? AND (? IS NULL OR id != ?)", (key, exclude_id, exclude_id)
    ).fetchone()
    if row is not None:
        raise HTTPException(status_code=409, detail="album_exists")


# Every member subquery starts from album_items (CROSS JOIN fixes the join order): the planner would
# otherwise walk items_order for every album and the albums page took 40 s on 100 000 items.
_MEMBERS_FROM = "FROM album_items ai CROSS JOIN items i ON i.id = ai.item_id WHERE ai.album_id = a.id"
_OUT_SELECT = (
    "SELECT a.id, a.name, a.name_key, a.description, a.cover_item_id, a.created_at, a.updated_at, "
    f"(SELECT COUNT(*) {_MEMBERS_FROM} AND {_MEMBER.format(nsfw=0)}) AS count, "
    f"(SELECT COUNT(*) {_MEMBERS_FROM} AND {_MEMBER.format(nsfw=1)}) AS vault_count, "
    f"(SELECT MAX({_SORT_KEY}) {_MEMBERS_FROM} AND {_MEMBER.format(nsfw=0)}) AS newest, "
    f"(SELECT MIN({_SORT_KEY}) {_MEMBERS_FROM} AND {_MEMBER.format(nsfw=0)}) AS oldest, "
    f"(SELECT i.uid {_MEMBERS_FROM} AND ai.item_id = a.cover_item_id AND {_MEMBER.format(nsfw=0)}) AS cover_uid, "
    f"(SELECT i.version {_MEMBERS_FROM} AND ai.item_id = a.cover_item_id AND {_MEMBER.format(nsfw=0)}) AS cover_version, "
    f"(SELECT i.uid {_MEMBERS_FROM} AND {_MEMBER.format(nsfw=0)} ORDER BY {_SORT_KEY} DESC, i.id DESC LIMIT 1) AS newest_uid, "
    f"(SELECT i.version {_MEMBERS_FROM} AND {_MEMBER.format(nsfw=0)} ORDER BY {_SORT_KEY} DESC, i.id DESC LIMIT 1) AS newest_version "
    "FROM albums a"
)


def album_out(row: sqlite3.Row) -> AlbumOut:
    cover = None
    if row["cover_uid"]:
        cover = thumb_url(row["cover_uid"], url_version(row["cover_version"], False))
    elif row["newest_uid"]:
        cover = thumb_url(row["newest_uid"], url_version(row["newest_version"], False))
    return AlbumOut(
        id=int(row["id"]), name=row["name"], description=row["description"] or "", count=int(row["count"]),
        vault_count=int(row["vault_count"]), cover_url=cover, newest=row["newest"], oldest=row["oldest"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def _get(conn: sqlite3.Connection, album_id: int) -> sqlite3.Row:
    row = conn.execute(f"{_OUT_SELECT} WHERE a.id = ?", (album_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No such album")
    return row


def _touch(conn: sqlite3.Connection, album_id: int, now: str | None = None) -> None:
    conn.execute("UPDATE albums SET updated_at = ? WHERE id = ?", (now or now_iso(), album_id))


def _resolve_members(conn: sqlite3.Connection, uids: list[str], *, vault_open: bool) -> list[int]:
    """Item ids for the uids that can be members: known, live, not a Live Photo companion, and on the
    vault side only while the vault is open."""
    ids: list[int] = []
    for i in range(0, len(uids), 500):
        chunk = uids[i:i + 500]
        rows = conn.execute(
            f"SELECT id FROM items WHERE uid IN ({','.join('?' * len(chunk))}) "
            "AND deleted_at IS NULL AND companion_of IS NULL AND (nsfw = 0 OR ?) ORDER BY id",
            [*chunk, 1 if vault_open else 0],
        ).fetchall()
        ids.extend(int(r["id"]) for r in rows)
    return ids


def _add_members(conn: sqlite3.Connection, album_id: int, uids: list[str], now: str, *, vault_open: bool) -> tuple[int, int]:
    """Runs in the caller's transaction. Returns (added, skipped)."""
    uids = list(dict.fromkeys(uids))
    added = 0
    for item_id in _resolve_members(conn, uids, vault_open=vault_open):
        cur = conn.execute(
            "INSERT OR IGNORE INTO album_items(album_id, item_id, added_at) VALUES (?, ?, ?)", (album_id, item_id, now)
        )
        added += max(0, cur.rowcount)
    if added:
        _touch(conn, album_id, now)
    return added, len(uids) - added


# ------------------------------------------------------------------ routes
@router.get("", response_model=AlbumPage)
def list_albums(
    q: str | None = Query(default=None, max_length=NAME_MAX),
    sort: Literal["updated", "name", "created"] = "updated",
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=60, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_db),
) -> AlbumPage:
    column, direction, cmp = SORTS[sort]
    where: list[str] = []
    params: list[Any] = []
    key = name_key(q) if q else ""
    if key:
        escaped = key.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where.append("a.name_key LIKE ? ESCAPE '\\'")
        params.append(f"%{escaped}%")
    after = _keyset(cursor)
    total = 0
    if after is None:
        cond = " AND ".join(where) if where else "1"
        total = int(conn.execute(f"SELECT COUNT(*) FROM albums a WHERE {cond}", params).fetchone()[0])
    page_where = list(where)
    page_params = list(params)
    if after is not None:
        page_where.append(f"({column} {cmp} ? OR ({column} = ? AND a.id {cmp} ?))")
        page_params.extend([after[0], after[0], after[1]])
    cond = " AND ".join(page_where) if page_where else "1"
    rows = conn.execute(
        f"{_OUT_SELECT} WHERE {cond} ORDER BY {column} {direction}, a.id {direction} LIMIT ?", [*page_params, limit + 1]
    ).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = None
    if more and rows:
        last = rows[-1]
        next_cursor = encode_cursor([last[column.split(".", 1)[1]], int(last["id"])])
    return AlbumPage(items=[album_out(r) for r in rows], next_cursor=next_cursor, total=total)


@router.post("", response_model=AlbumOut, status_code=201)
def create_album(body: AlbumIn, conn: sqlite3.Connection = Depends(get_db),
                 session: SessionInfo = Depends(require_user)) -> AlbumOut:
    name = _clean_name(body.name)
    key = name_key(name)
    now = now_iso()
    with tx(conn):
        _ensure_free_name(conn, key)
        cur = conn.execute(
            "INSERT INTO albums(name, name_key, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (name, key, body.description.strip(), now, now),
        )
        album_id = int(cur.lastrowid)
        if body.item_uids:
            _add_members(conn, album_id, body.item_uids, now, vault_open=session.vault_unlocked)
    return album_out(_get(conn, album_id))


@router.get("/{album_id}", response_model=AlbumOut)
def get_album(album_id: int, conn: sqlite3.Connection = Depends(get_db)) -> AlbumOut:
    return album_out(_get(conn, album_id))


@router.patch("/{album_id}", response_model=AlbumOut)
def update_album(album_id: int, body: AlbumUpdate, conn: sqlite3.Connection = Depends(get_db)) -> AlbumOut:
    with tx(conn):
        _get(conn, album_id)
        sets: list[str] = []
        params: list[Any] = []
        if body.name is not None:
            name = _clean_name(body.name)
            key = name_key(name)
            _ensure_free_name(conn, key, exclude_id=album_id)
            sets.append("name = ?")
            params.append(name)
            sets.append("name_key = ?")
            params.append(key)
        if body.description is not None:
            sets.append("description = ?")
            params.append(body.description.strip())
        if "cover_item_uid" in body.model_fields_set:
            if body.cover_item_uid is None:
                sets.append("cover_item_id = NULL")
            else:
                row = conn.execute(
                    "SELECT i.id FROM items i JOIN album_items ai ON ai.item_id = i.id "
                    "WHERE i.uid = ? AND ai.album_id = ? AND i.deleted_at IS NULL",
                    (body.cover_item_uid, album_id),
                ).fetchone()
                if row is None:
                    raise HTTPException(status_code=422, detail="cover_not_member")
                sets.append("cover_item_id = ?")
                params.append(int(row["id"]))
        if sets:
            sets.append("updated_at = ?")
            params.append(now_iso())
            conn.execute(f"UPDATE albums SET {', '.join(sets)} WHERE id = ?", [*params, album_id])
    return album_out(_get(conn, album_id))


@router.delete("/{album_id}", response_model=Ok)
def delete_album(album_id: int, conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    with tx(conn):
        _get(conn, album_id)
        conn.execute("DELETE FROM albums WHERE id = ?", (album_id,))  # album_items and shares cascade
    return Ok()


@router.post("/{album_id}/items", response_model=AlbumItemsOut)
def add_items(album_id: int, body: AlbumItemsIn, conn: sqlite3.Connection = Depends(get_db),
              session: SessionInfo = Depends(require_user)) -> AlbumItemsOut:
    with tx(conn):
        _get(conn, album_id)
        added, skipped = _add_members(conn, album_id, body.uids, now_iso(), vault_open=session.vault_unlocked)
    return AlbumItemsOut(added=added, skipped=skipped)


@router.delete("/{album_id}/items", response_model=AlbumRemovedOut)
def remove_items(album_id: int, body: AlbumItemsIn, conn: sqlite3.Connection = Depends(get_db),
                 session: SessionInfo = Depends(require_user)) -> AlbumRemovedOut:
    uids = list(dict.fromkeys(body.uids))
    removed = 0
    with tx(conn):
        _get(conn, album_id)
        for i in range(0, len(uids), 500):
            chunk = uids[i:i + 500]
            cur = conn.execute(
                f"DELETE FROM album_items WHERE album_id = ? AND item_id IN (SELECT id FROM items WHERE uid IN ({','.join('?' * len(chunk))}) "
                "AND (nsfw = 0 OR ?))",
                [album_id, *chunk, 1 if session.vault_unlocked else 0],
            )
            removed += max(0, cur.rowcount)
        if removed:
            _touch(conn, album_id)
    return AlbumRemovedOut(removed=removed)
