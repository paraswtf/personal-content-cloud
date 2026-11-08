"""Share links (docs/ARCHITECTURE.md section 26).

A share is a row of `shares` (token, kind 'album' or 'items', optional password hash, expiry, download
switch, revocation, view counter) plus `share_items` rows for a hand-picked set. Members are listed in
library order and only while they are live, non-vault, non-companion items of a kind with a preview
(never kind other, which has no thumb or preview); an album share reads the album's members the
same way and also leaves mis-clicks out, as the album page does. Vault items are refused at creation
(422 vault_item) and an album share skips them; kind other items are skipped at creation, and a set
with nothing left is 422 no_items.

Public access checks the token (404 share_not_found for unknown or revoked, 410 share_expired), then
the password: a correct POST /api/s/{token}/unlock sets the cookie pcc_share_<token>, an
itsdangerous.TimestampSigner signature over the token with the app secret, good for UNLOCK_DAYS.
"""
from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass

from fastapi import HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

from app.auth import passwords
from app.db import tx
from app.sharing.secret import secret_key
from app.util import now_iso, parse_iso

UNLOCK_DAYS = 7
COOKIE_PREFIX = "pcc_share_"
SALT = "pcc-share-unlock"
REVOKED_LISTED_DAYS = 30
MEMBER_WHERE = "i.deleted_at IS NULL AND i.nsfw = 0 AND i.companion_of IS NULL AND i.kind != 'other'"
ALBUM_MEMBER_WHERE = MEMBER_WHERE + " AND i.misclick = 0"  # what the album page shows by default
MEMBER_ORDER = "COALESCE(i.taken_at, i.created_at) DESC, i.id DESC"
SHARE_COLUMNS = ("s.id, s.token, s.kind, s.album_id, s.title, s.password_hash, s.expires_at, s.allow_download, "
                 "s.created_at, s.revoked_at, s.view_count, s.last_viewed_at, a.name AS album_name")
ITEM_COLUMNS = ("i.id, i.uid, i.kind, i.version, i.ext, i.mime, i.original_filename, i.width, i.height, "
                "i.duration_s, i.taken_at, i.created_at, i.preview_status")


@dataclass
class Refused(Exception):
    status: int
    code: str


def new_token() -> str:
    return secrets.token_urlsafe(16)


def cookie_name(token: str) -> str:
    return COOKIE_PREFIX + token


def _signer() -> TimestampSigner:
    return TimestampSigner(secret_key(), salt=SALT)


def unlock_value(token: str) -> str:
    return _signer().sign(token.encode()).decode()


def unlock_valid(token: str, value: str | None) -> bool:
    if not value or len(value) > 400:
        return False
    try:
        raw = _signer().unsign(value.encode(), max_age=UNLOCK_DAYS * 86400)
    except (BadSignature, SignatureExpired):
        return False
    return secrets.compare_digest(raw, token.encode())


def set_unlock_cookie(response: Response, request: Request, token: str) -> None:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    secure = proto.split(",")[0].strip().lower() == "https"
    response.set_cookie(cookie_name(token), unlock_value(token), max_age=UNLOCK_DAYS * 86400, path="/",
                        httponly=True, samesite="lax", secure=secure)


# ----------------------------------------------------------------- rows
def get(conn: sqlite3.Connection, share_id: int) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT {SHARE_COLUMNS} FROM shares s LEFT JOIN albums a ON a.id = s.album_id WHERE s.id = ?", (share_id,)
    ).fetchone()


def by_token(conn: sqlite3.Connection, token: str) -> sqlite3.Row | None:
    if not token or len(token) > 64:
        return None
    return conn.execute(
        f"SELECT {SHARE_COLUMNS} FROM shares s LEFT JOIN albums a ON a.id = s.album_id WHERE s.token = ?", (token,)
    ).fetchone()


def listed(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    cutoff = now_iso(-REVOKED_LISTED_DAYS * 86400)
    return conn.execute(
        f"SELECT {SHARE_COLUMNS} FROM shares s LEFT JOIN albums a ON a.id = s.album_id "
        "WHERE s.revoked_at IS NULL OR s.revoked_at > ? ORDER BY s.created_at DESC, s.id DESC", (cutoff,)
    ).fetchall()


def is_expired(row: sqlite3.Row, now: str | None = None) -> bool:
    exp = row["expires_at"]
    return bool(exp) and exp <= (now or now_iso())


def resolve(conn: sqlite3.Connection, token: str) -> sqlite3.Row:
    """The live share for a public token: 404 share_not_found (unknown or revoked), 410 share_expired."""
    row = by_token(conn, token)
    if row is None or row["revoked_at"] is not None:
        raise HTTPException(status_code=404, detail="share_not_found")
    if is_expired(row):
        raise HTTPException(status_code=410, detail="share_expired")
    return row


def unlocked(request: Request, row: sqlite3.Row) -> bool:
    if not row["password_hash"]:
        return True
    return unlock_valid(row["token"], request.cookies.get(cookie_name(row["token"])))


def require_unlocked(request: Request, row: sqlite3.Row) -> None:
    if not unlocked(request, row):
        raise HTTPException(status_code=403, detail="share_password_required")


# ----------------------------------------------------------------- members
def _member_sql(row: sqlite3.Row) -> tuple[str, str]:
    if row["kind"] == "album":
        return "album_items m JOIN items i ON i.id = m.item_id", "m.album_id = ?"
    return "share_items m JOIN items i ON i.id = m.item_id", "m.share_id = ?"


def _member_key(row: sqlite3.Row) -> int:
    return int(row["album_id"]) if row["kind"] == "album" else int(row["id"])


def _member_where(row: sqlite3.Row) -> str:
    return ALBUM_MEMBER_WHERE if row["kind"] == "album" else MEMBER_WHERE


def members(conn: sqlite3.Connection, row: sqlite3.Row) -> list[sqlite3.Row]:
    frm, cond = _member_sql(row)
    return conn.execute(
        f"SELECT {ITEM_COLUMNS} FROM {frm} WHERE {cond} AND {_member_where(row)} ORDER BY {MEMBER_ORDER}",
        (_member_key(row),),
    ).fetchall()


def member(conn: sqlite3.Connection, row: sqlite3.Row, uid: str) -> sqlite3.Row | None:
    frm, cond = _member_sql(row)
    return conn.execute(
        f"SELECT {ITEM_COLUMNS} FROM {frm} WHERE {cond} AND i.uid = ? AND {_member_where(row)}", (_member_key(row), uid)
    ).fetchone()


def count(conn: sqlite3.Connection, row: sqlite3.Row) -> int:
    frm, cond = _member_sql(row)
    return int(conn.execute(
        f"SELECT COUNT(*) FROM {frm} WHERE {cond} AND {_member_where(row)}", (_member_key(row),)
    ).fetchone()[0])


# ----------------------------------------------------------------- owner operations
def create(conn: sqlite3.Connection, *, kind: str, album_id: int | None, item_uids: list[str], title: str,
           password: str | None, expires_days: int | None, allow_download: bool) -> sqlite3.Row:
    """Insert a share. Raises Refused(404, 'album_not_found'), Refused(404, 'item_not_found') and
    Refused(422, 'vault_item'); kind other items and companions are skipped, and a hand-picked set with
    nothing left is Refused(422, 'no_items')."""
    now = now_iso()
    items: list[int] = []
    if kind == "album":
        if album_id is None or conn.execute("SELECT 1 FROM albums WHERE id = ?", (album_id,)).fetchone() is None:
            raise Refused(404, "album_not_found")
    else:
        album_id = None
        uids = list(dict.fromkeys(u for u in item_uids if u))
        if not uids:
            raise Refused(422, "no_items")
        rows: dict[str, sqlite3.Row] = {}
        for i in range(0, len(uids), 500):
            chunk = uids[i:i + 500]
            for r in conn.execute(
                f"SELECT id, uid, kind, nsfw, companion_of FROM items WHERE uid IN ({','.join('?' * len(chunk))})",
                chunk,
            ):
                rows[r["uid"]] = r
        missing = [u for u in uids if u not in rows]
        if missing:
            raise Refused(404, "item_not_found")
        if any(rows[u]["nsfw"] for u in uids):
            raise Refused(422, "vault_item")
        items = [int(rows[u]["id"]) for u in uids if rows[u]["companion_of"] is None and rows[u]["kind"] != "other"]
        if not items:
            raise Refused(422, "no_items")
    expires_at = now_iso(expires_days * 86400) if expires_days else None
    with tx(conn):
        token = new_token()
        cur = conn.execute(
            "INSERT INTO shares(token, kind, album_id, title, password_hash, expires_at, allow_download, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (token, kind, album_id, title, passwords.hash_password(password) if password else None, expires_at,
             1 if allow_download else 0, now),
        )
        share_id = int(cur.lastrowid)
        for pos, item_id in enumerate(items):
            conn.execute("INSERT OR IGNORE INTO share_items(share_id, item_id, position) VALUES (?, ?, ?)",
                         (share_id, item_id, pos))
    row = get(conn, share_id)
    assert row is not None
    return row


def update(conn: sqlite3.Connection, share_id: int, *, title: str | None, password: str | None, password_set: bool,
           expires_days: int | None, expires_set: bool, allow_download: bool | None) -> sqlite3.Row | None:
    row = get(conn, share_id)
    if row is None:
        return None
    sets: list[str] = []
    params: list[object] = []
    if title is not None:
        sets.append("title = ?")
        params.append(title)
    if password_set:
        sets.append("password_hash = ?")
        params.append(passwords.hash_password(password) if password else None)
    if expires_set:
        sets.append("expires_at = ?")
        params.append(now_iso(expires_days * 86400) if expires_days else None)
    if allow_download is not None:
        sets.append("allow_download = ?")
        params.append(1 if allow_download else 0)
    if sets:
        with tx(conn):
            conn.execute(f"UPDATE shares SET {', '.join(sets)} WHERE id = ?", (*params, share_id))
    return get(conn, share_id)


def revoke(conn: sqlite3.Connection, share_id: int) -> bool:
    with tx(conn):
        cur = conn.execute("UPDATE shares SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL", (now_iso(), share_id))
        return cur.rowcount == 1


def record_view(conn: sqlite3.Connection, share_id: int) -> None:
    with tx(conn):
        conn.execute("UPDATE shares SET view_count = view_count + 1, last_viewed_at = ? WHERE id = ?", (now_iso(), share_id))


def remaining_seconds(row: sqlite3.Row) -> float | None:
    dt = parse_iso(row["expires_at"])
    if dt is None:
        return None
    from datetime import datetime, timezone

    return (dt - datetime.now(timezone.utc)).total_seconds()
