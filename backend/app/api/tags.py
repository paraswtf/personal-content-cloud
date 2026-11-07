"""Tags the owner adds, and that the system learns (docs/ARCHITECTURE.md section 19).

Routes (full paths; mounted with no prefix, like the people router):
  GET    /api/tags?q=                       TagOut[] with counts over the items the session may see
  POST   /api/tags {name}                   get or create -> TagOut
  PATCH  /api/tags/{id} {name}              rename; onto an existing name merges into it
  DELETE /api/tags/{id}                     remove everywhere
  POST   /api/items/{uid}/user-tags {tag_id | name}   add (an auto row becomes a user row)
  DELETE /api/items/{uid}/user-tags/{tag_id}          remove (an auto row leaves a rejection)
Helpers for the bulk route and tests: ensure_tag, add_to_item, remove_from_item, item_tags_out.
"""
from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_db, require_user
from app.auth.sessions import SessionInfo
from app.db import tx
from app.schemas import ItemTagOut, Ok, TagNameIn, TagOut, UserTagIn
from app.util import name_key, now_iso

log = logging.getLogger(__name__)
router = APIRouter()
NAME_MAX = 40


def clean_tag_name(raw: str | None) -> str:
    name = " ".join((raw or "").split())
    if not name or len(name) > NAME_MAX:
        raise HTTPException(status_code=422, detail=f"A tag is 1 to {NAME_MAX} characters")
    return name


def _refresh_fts(conn: sqlite3.Connection, item_ids: list[int]) -> None:
    try:
        from app.documents.extract import refresh_fts
    except ImportError:
        return
    for item_id in dict.fromkeys(item_ids):
        try:
            refresh_fts(conn, int(item_id))
        except Exception:  # noqa: BLE001 - the index catches up on the next refresh
            log.exception("tags: could not refresh the search row of item %s", item_id)


def _record(conn: sqlite3.Connection, label: int, tag_row: sqlite3.Row, item_id: int) -> None:
    try:
        from app.learning import events

        events.record(conn, "tag", label, {"tag_id": int(tag_row["id"]), "name": tag_row["name"]}, item_id=item_id)
    except Exception:  # noqa: BLE001 - learning never blocks a tag change
        log.exception("tags: could not record a tag event")


def ensure_tag(conn: sqlite3.Connection, name: str) -> sqlite3.Row:
    """The tag with this name (by key), created when new. Runs in the caller's transaction."""
    key = name_key(name)
    row = conn.execute("SELECT * FROM tags WHERE name_key = ?", (key,)).fetchone()
    if row is not None:
        return row
    conn.execute("INSERT INTO tags(name, name_key, created_at) VALUES (?, ?, ?)", (name, key, now_iso()))
    return conn.execute("SELECT * FROM tags WHERE name_key = ?", (key,)).fetchone()


def add_to_item(conn: sqlite3.Connection, item_id: int, tag_row: sqlite3.Row) -> bool:
    """A user row for (item, tag); an auto row becomes a user row and a rejection is cleared.
    Returns True when something changed. Caller's transaction."""
    tag_id = int(tag_row["id"])
    existing = conn.execute("SELECT source FROM item_user_tags WHERE item_id = ? AND tag_id = ?", (item_id, tag_id)).fetchone()
    if existing is not None and existing["source"] == "user":
        return False
    conn.execute(
        "INSERT INTO item_user_tags(item_id, tag_id, source, score, created_at) VALUES (?, ?, 'user', NULL, ?) "
        "ON CONFLICT(item_id, tag_id) DO UPDATE SET source = 'user', score = NULL",
        (item_id, tag_id, now_iso()),
    )
    conn.execute("DELETE FROM tag_rejections WHERE item_id = ? AND tag_id = ?", (item_id, tag_id))
    _record(conn, 1, tag_row, item_id)
    return True


def remove_from_item(conn: sqlite3.Connection, item_id: int, tag_row: sqlite3.Row) -> bool:
    """Remove the row; a removed suggestion becomes a rejection (a counter-example). Caller's transaction."""
    tag_id = int(tag_row["id"])
    existing = conn.execute("SELECT source FROM item_user_tags WHERE item_id = ? AND tag_id = ?", (item_id, tag_id)).fetchone()
    if existing is None:
        return False
    if existing["source"] == "auto":
        conn.execute("INSERT OR IGNORE INTO tag_rejections(item_id, tag_id, created_at) VALUES (?, ?, ?)",
                     (item_id, tag_id, now_iso()))
        _record(conn, 0, tag_row, item_id)
    conn.execute("DELETE FROM item_user_tags WHERE item_id = ? AND tag_id = ?", (item_id, tag_id))
    return True


def item_tags_out(conn: sqlite3.Connection, item_id: int) -> list[ItemTagOut]:
    rows = conn.execute(
        "SELECT t.id, t.name, ut.source, ut.score FROM item_user_tags ut JOIN tags t ON t.id = ut.tag_id "
        "WHERE ut.item_id = ? ORDER BY CASE ut.source WHEN 'user' THEN 0 ELSE 1 END, ut.score DESC, t.name",
        (item_id,),
    ).fetchall()
    return [ItemTagOut(id=int(r["id"]), name=r["name"], source=r["source"], score=r["score"]) for r in rows]


def _counts(conn: sqlite3.Connection, vault_open: bool) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for r in conn.execute(
        "SELECT ut.tag_id, COUNT(*) AS n, SUM(CASE WHEN ut.source = 'auto' THEN 1 ELSE 0 END) AS a "
        "FROM item_user_tags ut JOIN items i ON i.id = ut.item_id WHERE i.deleted_at IS NULL AND (i.nsfw = 0 OR ?) "
        "GROUP BY ut.tag_id", (1 if vault_open else 0,)
    ):
        out[int(r["tag_id"])] = (int(r["n"]), int(r["a"] or 0))
    return out


def _tag_out(conn: sqlite3.Connection, row: sqlite3.Row, vault_open: bool) -> TagOut:
    n, a = _counts(conn, vault_open).get(int(row["id"]), (0, 0))
    return TagOut(id=int(row["id"]), name=row["name"], items=n, auto_items=a)


def _tag_or_404(conn: sqlite3.Connection, tag_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM tags WHERE id = ?", (tag_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No such tag")
    return row


def _item_or_404(conn: sqlite3.Connection, uid: str, session: SessionInfo) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM items WHERE uid = ?", (uid,)).fetchone()
    if row is None or (row["nsfw"] and not session.vault_unlocked):
        raise HTTPException(status_code=404, detail="Not found")
    return row


# ------------------------------------------------------------------ tags
@router.get("/api/tags", response_model=list[TagOut])
def list_tags(q: str | None = Query(default=None, max_length=40), session: SessionInfo = Depends(require_user),
              conn: sqlite3.Connection = Depends(get_db)) -> list[TagOut]:
    counts = _counts(conn, session.vault_unlocked)
    rows = conn.execute("SELECT * FROM tags ORDER BY name").fetchall()
    if q:
        key = name_key(q)
        rows = [r for r in rows if key and key in r["name_key"]]
    out = [TagOut(id=int(r["id"]), name=r["name"], items=counts.get(int(r["id"]), (0, 0))[0],
                  auto_items=counts.get(int(r["id"]), (0, 0))[1]) for r in rows]
    out.sort(key=lambda t: (-t.items, t.name.lower()))
    return out


@router.post("/api/tags", response_model=TagOut)
def create_tag(body: TagNameIn, session: SessionInfo = Depends(require_user),
               conn: sqlite3.Connection = Depends(get_db)) -> TagOut:
    name = clean_tag_name(body.name)
    with tx(conn):
        row = ensure_tag(conn, name)
    return _tag_out(conn, row, session.vault_unlocked)


@router.patch("/api/tags/{tag_id}", response_model=TagOut)
def rename_tag(tag_id: int, body: TagNameIn, session: SessionInfo = Depends(require_user),
               conn: sqlite3.Connection = Depends(get_db)) -> TagOut:
    name = clean_tag_name(body.name)
    key = name_key(name)
    affected: list[int] = []
    with tx(conn):
        row = _tag_or_404(conn, tag_id)
        target = conn.execute("SELECT * FROM tags WHERE name_key = ? AND id != ?", (key, tag_id)).fetchone()
        affected = [int(r["item_id"]) for r in conn.execute("SELECT item_id FROM item_user_tags WHERE tag_id = ?", (tag_id,))]
        if target is None:
            conn.execute("UPDATE tags SET name = ?, name_key = ? WHERE id = ?", (name, key, tag_id))
            result = conn.execute("SELECT * FROM tags WHERE id = ?", (tag_id,)).fetchone()
        else:
            # merge: user rows win over auto rows, rejections carry over, the old tag goes
            tid = int(target["id"])
            for r in conn.execute("SELECT item_id, source, score, created_at FROM item_user_tags WHERE tag_id = ?", (tag_id,)).fetchall():
                conn.execute(
                    "INSERT INTO item_user_tags(item_id, tag_id, source, score, created_at) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(item_id, tag_id) DO UPDATE SET source = CASE WHEN excluded.source = 'user' THEN 'user' "
                    "ELSE item_user_tags.source END, score = CASE WHEN excluded.source = 'user' THEN NULL ELSE item_user_tags.score END",
                    (r["item_id"], tid, r["source"], r["score"], r["created_at"]),
                )
            conn.execute("INSERT OR IGNORE INTO tag_rejections(item_id, tag_id, created_at) "
                         "SELECT item_id, ?, created_at FROM tag_rejections WHERE tag_id = ?", (tid, tag_id))
            conn.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
            result = conn.execute("SELECT * FROM tags WHERE id = ?", (tid,)).fetchone()
    _refresh_fts(conn, affected)
    return _tag_out(conn, result, session.vault_unlocked)


@router.delete("/api/tags/{tag_id}", response_model=Ok)
def delete_tag(tag_id: int, session: SessionInfo = Depends(require_user),
               conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    with tx(conn):
        _tag_or_404(conn, tag_id)
        affected = [int(r["item_id"]) for r in conn.execute("SELECT item_id FROM item_user_tags WHERE tag_id = ?", (tag_id,))]
        conn.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
    _refresh_fts(conn, affected)
    return Ok()


# ------------------------------------------------------------------ item tags
@router.post("/api/items/{uid}/user-tags", response_model=list[ItemTagOut])
def add_item_tag(uid: str, body: UserTagIn, session: SessionInfo = Depends(require_user),
                 conn: sqlite3.Connection = Depends(get_db)) -> list[ItemTagOut]:
    item = _item_or_404(conn, uid, session)
    item_id = int(item["id"])
    with tx(conn):
        if body.tag_id is not None:
            tag = _tag_or_404(conn, int(body.tag_id))
        elif body.name is not None:
            tag = ensure_tag(conn, clean_tag_name(body.name))
        else:
            raise HTTPException(status_code=422, detail="tag_id or name is required")
        changed = add_to_item(conn, item_id, tag)
    if changed:
        _refresh_fts(conn, [item_id])
    return item_tags_out(conn, item_id)


@router.delete("/api/items/{uid}/user-tags/{tag_id}", response_model=list[ItemTagOut])
def remove_item_tag(uid: str, tag_id: int, session: SessionInfo = Depends(require_user),
                    conn: sqlite3.Connection = Depends(get_db)) -> list[ItemTagOut]:
    item = _item_or_404(conn, uid, session)
    item_id = int(item["id"])
    with tx(conn):
        tag = _tag_or_404(conn, tag_id)
        changed = remove_from_item(conn, item_id, tag)
    if changed:
        _refresh_fts(conn, [item_id])
    return item_tags_out(conn, item_id)
