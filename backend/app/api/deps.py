"""Shared FastAPI dependencies: the database connection, the logged-in session, the vault window
and the CSRF header (docs/ARCHITECTURE.md section 4)."""
from __future__ import annotations

import sqlite3
from typing import Iterator

from fastapi import Depends, HTTPException, Request

from app.auth.sessions import SessionInfo, authenticate
from app.db import connect

CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "pcc"
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


def get_db() -> Iterator[sqlite3.Connection]:
    """One SQLite connection per request (endpoints are sync `def`, run in the threadpool)."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def check_csrf(request: Request) -> None:
    """Every non-GET request must carry `X-Requested-With: pcc` (403 `csrf` otherwise)."""
    if request.method in SAFE_METHODS:
        return
    if request.headers.get(CSRF_HEADER, "").strip().lower() != CSRF_VALUE:
        raise HTTPException(status_code=403, detail="csrf")


def require_user(request: Request, conn: sqlite3.Connection = Depends(get_db)) -> SessionInfo:
    """401 `login_required` without a live session. Also enforces the CSRF header on writes."""
    check_csrf(request)
    session = authenticate(conn, request)
    if session is None:
        raise HTTPException(status_code=401, detail="login_required")
    request.state.session = session
    return session


def optional_user(request: Request, conn: sqlite3.Connection = Depends(get_db)) -> SessionInfo | None:
    """The session when there is one (bootstrap routes such as /api/auth/state)."""
    session = authenticate(conn, request)
    request.state.session = session
    return session


def require_vault(session: SessionInfo = Depends(require_user)) -> SessionInfo:
    """403 `vault_locked` unless the session unlocked the NSFW vault recently."""
    if not session.vault_unlocked:
        raise HTTPException(status_code=403, detail="vault_locked")
    return session


def vault_requested(request: Request) -> bool:
    return request.query_params.get("vault", "") in ("1", "true")


def want_vault(request: Request, session: SessionInfo = Depends(require_user)) -> bool:
    """True when the request asks for vault items (`vault=1`) and may see them; 403 when it asks
    and may not. Routes listing items pass this to the visibility filter."""
    if not vault_requested(request):
        return False
    if not session.vault_unlocked:
        raise HTTPException(status_code=403, detail="vault_locked")
    return True
