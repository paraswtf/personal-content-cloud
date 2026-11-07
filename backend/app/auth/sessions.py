"""Browser sessions (docs/ARCHITECTURE.md section 4).

The cookie `pcc_session` carries a random token; the table stores sha256(token). Sessions slide:
`last_seen_at` is bumped at most once a minute and `expires_at` follows it by SESSION_DAYS.
When the row slides, `authenticate` leaves the raw token in `request.state.refresh_cookie` and
`app.main.SessionCookieMiddleware` sets the cookie again, so the browser's copy slides too.
The NSFW vault unlock is a per-session window (`vault_until`).
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Request, Response

from app.config import settings
from app.db import tx
from app.util import new_token, now_iso, parse_iso

COOKIE = "pcc_session"
TOUCH_INTERVAL_S = 60
MAX_SESSIONS = 50  # oldest live sessions are revoked beyond this


@dataclass
class SessionInfo:
    id: int
    created_at: str
    last_seen_at: str
    expires_at: str
    vault_until: str | None
    user_agent: str | None
    ip: str | None

    @property
    def vault_unlocked(self) -> bool:
        return vault_open(self.vault_until)


def vault_open(vault_until: str | None) -> bool:
    dt = parse_iso(vault_until)
    return dt is not None and dt > datetime.now(timezone.utc)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _expiry() -> str:
    return now_iso(settings.session_days * 86400)


def client_ip(request: Request) -> str:
    """The direct client address (uvicorn --proxy-headers resolves X-Forwarded-For)."""
    return (request.client.host if request.client else "")[:64]


def _secure(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return proto.split(",")[0].strip().lower() == "https"


def set_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        COOKIE, token, max_age=settings.session_days * 86400, path="/", httponly=True, samesite="lax",
        secure=_secure(request),
    )


def clear_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(COOKIE, path="/", httponly=True, samesite="lax", secure=_secure(request))


def create_session(conn: sqlite3.Connection, request: Request) -> tuple[str, SessionInfo]:
    """Insert a session and return (raw token, info). The caller sets the cookie."""
    token = new_token(32)
    now = now_iso()
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO sessions(token_hash, created_at, last_seen_at, expires_at, user_agent, ip) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (_hash(token), now, now, _expiry(), (request.headers.get("user-agent") or "")[:300], client_ip(request)),
        )
        session_id = int(cur.lastrowid)
        # Keep the table small: beyond MAX_SESSIONS live sessions, the oldest go.
        conn.execute(
            "UPDATE sessions SET revoked_at = ? WHERE revoked_at IS NULL AND id NOT IN ("
            "SELECT id FROM sessions WHERE revoked_at IS NULL ORDER BY last_seen_at DESC, id DESC LIMIT ?)",
            (now, MAX_SESSIONS),
        )
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return token, _info(row)


def _info(row: sqlite3.Row) -> SessionInfo:
    return SessionInfo(
        id=int(row["id"]), created_at=row["created_at"], last_seen_at=row["last_seen_at"],
        expires_at=row["expires_at"], vault_until=row["vault_until"], user_agent=row["user_agent"], ip=row["ip"],
    )


def authenticate(conn: sqlite3.Connection, request: Request) -> SessionInfo | None:
    """The live session of the request's cookie, or None. Slides the expiry (at most once a minute)."""
    token = request.cookies.get(COOKIE)
    if not token or len(token) > 200:
        return None
    row = conn.execute(
        "SELECT * FROM sessions WHERE token_hash = ? AND revoked_at IS NULL AND expires_at > ?",
        (_hash(token), now_iso()),
    ).fetchone()
    if row is None:
        return None
    info = _info(row)
    last = parse_iso(info.last_seen_at)
    if last is None or datetime.now(timezone.utc) - last > timedelta(seconds=TOUCH_INTERVAL_S):
        now = now_iso()
        with tx(conn):
            conn.execute("UPDATE sessions SET last_seen_at = ?, expires_at = ? WHERE id = ?", (now, _expiry(), info.id))
        info.last_seen_at, info.expires_at = now, _expiry()
        # The cookie's max-age was set at login; SessionCookieMiddleware re-issues it with this
        # token so the browser's copy slides along with the row.
        request.state.refresh_cookie = token
    return info


def revoke(conn: sqlite3.Connection, session_id: int) -> bool:
    with tx(conn):
        cur = conn.execute("UPDATE sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL", (now_iso(), session_id))
        return cur.rowcount == 1


def revoke_all(conn: sqlite3.Connection, except_id: int | None = None) -> int:
    with tx(conn):
        if except_id is None:
            cur = conn.execute("UPDATE sessions SET revoked_at = ? WHERE revoked_at IS NULL", (now_iso(),))
        else:
            cur = conn.execute("UPDATE sessions SET revoked_at = ? WHERE revoked_at IS NULL AND id != ?", (now_iso(), except_id))
        return cur.rowcount


def list_sessions(conn: sqlite3.Connection) -> list[SessionInfo]:
    rows = conn.execute(
        "SELECT * FROM sessions WHERE revoked_at IS NULL AND expires_at > ? ORDER BY last_seen_at DESC, id DESC",
        (now_iso(),),
    ).fetchall()
    return [_info(r) for r in rows]


def vault_unlock(conn: sqlite3.Connection, session_id: int) -> str:
    until = now_iso(settings.vault_unlock_min * 60)
    with tx(conn):
        conn.execute("UPDATE sessions SET vault_until = ? WHERE id = ?", (until, session_id))
    return until


def vault_lock(conn: sqlite3.Connection, session_id: int) -> None:
    with tx(conn):
        conn.execute("UPDATE sessions SET vault_until = NULL WHERE id = ?", (session_id,))


def purge_expired(conn: sqlite3.Connection) -> int:
    """Housekeeping: drop sessions expired or revoked more than a day ago, and stale challenges."""
    cutoff = now_iso(-86400)
    with tx(conn):
        cur = conn.execute("DELETE FROM sessions WHERE expires_at < ? OR revoked_at < ?", (cutoff, cutoff))
        conn.execute("DELETE FROM challenges WHERE expires_at < ?", (now_iso(),))
        return cur.rowcount
