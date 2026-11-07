"""Admin session (single hardcoded account) and the optional visitor access key.

Admin: POST /api/admin/login sets a signed, HttpOnly cookie valid for 7 days.
CSRF: every mutating admin request must carry the header `X-Requested-With: hub`
(browsers cannot add custom headers cross-site without a CORS preflight, which we never allow).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from collections import deque

from fastapi import HTTPException, Request, Response
from itsdangerous import BadSignature, TimestampSigner

from app.config import ADMIN_PASSWORD, ADMIN_USERNAME, settings
from app.db import connect, kv_get, tx

ADMIN_COOKIE = "hub_admin"
ACCESS_COOKIE = "hub_access"
SESSION_MAX_AGE = 7 * 24 * 3600
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "hub"

_secret_lock = threading.Lock()
_secret: bytes | None = None
_secret_path: str | None = None


def secret_key() -> bytes:
    """Random key stored in DATA_DIR/secret.key (created with mode 0600 on first use)."""
    global _secret, _secret_path
    path = settings.data_dir / "secret.key"
    with _secret_lock:
        if _secret is None or _secret_path != str(path):
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as f:
                    f.write(secrets.token_bytes(32))
            _secret = path.read_bytes()
            _secret_path = str(path)
        return _secret


def _signer() -> TimestampSigner:
    return TimestampSigner(secret_key(), salt="hub-admin-session")


SESSION_EPOCH_KEY = "admin_session_epoch"  # kv: part of every signed session; bumped on logout


def _session_epoch() -> str:
    conn = connect()
    try:
        return kv_get(conn, SESSION_EPOCH_KEY, "0") or "0"
    finally:
        conn.close()


def _session_value() -> bytes:
    return f"admin:{ADMIN_USERNAME}:{_session_epoch()}".encode()


def end_all_sessions() -> None:
    """Invalidate every admin session token issued so far (logout signs out everywhere)."""
    conn = connect()
    try:
        with tx(conn):
            conn.execute(
                "INSERT INTO kv(key, value) VALUES (?, '1') "
                "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)",
                (SESSION_EPOCH_KEY,),
            )
    finally:
        conn.close()


def check_credentials(username: str, password: str) -> bool:
    user_ok = hmac.compare_digest(username.strip().casefold().encode(), ADMIN_USERNAME.casefold().encode())
    pass_ok = hmac.compare_digest(password.encode(), ADMIN_PASSWORD.encode())
    return user_ok and pass_ok


def _is_https(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return proto == "https" or request.url.scheme == "https"


def set_admin_cookie(response: Response, request: Request) -> None:
    token = _signer().sign(_session_value()).decode()
    response.set_cookie(
        ADMIN_COOKIE, token, max_age=SESSION_MAX_AGE, httponly=True, samesite="lax",
        secure=_is_https(request), path="/",
    )


def clear_admin_cookie(response: Response) -> None:
    response.delete_cookie(ADMIN_COOKIE, path="/")


def is_admin(request: Request) -> bool:
    token = request.cookies.get(ADMIN_COOKIE)
    if not token:
        return False
    try:
        value = _signer().unsign(token.encode(), max_age=SESSION_MAX_AGE)
    except BadSignature:
        return False
    return hmac.compare_digest(value, _session_value())


def require_admin(request: Request) -> None:
    """FastAPI dependency: 401 unless logged in as admin; 403 if a write lacks the CSRF header."""
    if not is_admin(request):
        raise HTTPException(status_code=401, detail="Admin login required")
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get(CSRF_HEADER) != CSRF_VALUE:
        raise HTTPException(status_code=403, detail="Missing X-Requested-With header")


# ----------------------------------------------------------------- login rate limit
_fail_lock = threading.Lock()
_failures: dict[str, deque] = {}
_MAX_FAILURES = 10
_WINDOW_S = 600


_MAX_TRACKED = 5000


def _prune(now: float) -> None:
    """Drop expired attempts; called with _fail_lock held. Keeps the dict bounded."""
    for key in list(_failures):
        q = _failures[key]
        while q and now - q[0] > _WINDOW_S:
            q.popleft()
        if not q:
            del _failures[key]


def login_allowed(ip: str) -> bool:
    now = time.monotonic()
    with _fail_lock:
        q = _failures.get(ip)
        if not q:
            return True
        while q and now - q[0] > _WINDOW_S:
            q.popleft()
        if not q:
            del _failures[ip]
            return True
        return len(q) < _MAX_FAILURES


def claim_login_attempt(ip: str) -> bool:
    """Atomically check the limit and count this attempt as a failure (a success resets it).

    Returns False when the client is over the limit; parallel requests cannot all slip past
    the check before any failure is recorded.
    """
    now = time.monotonic()
    with _fail_lock:
        if len(_failures) >= _MAX_TRACKED:
            _prune(now)
        q = _failures.get(ip)
        if q:
            while q and now - q[0] > _WINDOW_S:
                q.popleft()
            if len(q) >= _MAX_FAILURES:
                return False
        _failures.setdefault(ip, deque()).append(now)
        return True


def record_login_failure(ip: str) -> None:
    now = time.monotonic()
    with _fail_lock:
        if len(_failures) >= _MAX_TRACKED:
            _prune(now)
        _failures.setdefault(ip, deque()).append(now)


def reset_login_failures(ip: str) -> None:
    with _fail_lock:
        _failures.pop(ip, None)


# --------------------------------------------------------------- visitor access key
def access_token() -> str:
    key = settings.access_key or ""
    return hmac.new(secret_key(), key.encode(), hashlib.sha256).hexdigest()


def has_access(request: Request) -> bool:
    """True if no access key is configured, or the visitor unlocked it, or is admin."""
    if not settings.access_key:
        return True
    token = request.cookies.get(ACCESS_COOKIE)
    if token and hmac.compare_digest(token, access_token()):
        return True
    return is_admin(request)


def set_access_cookie(response: Response, request: Request) -> None:
    response.set_cookie(
        ACCESS_COOKIE, access_token(), max_age=365 * 24 * 3600, httponly=True, samesite="lax",
        secure=_is_https(request), path="/",
    )


def client_ip(request: Request) -> str:
    """The peer address. X-Forwarded-For is never read here: uvicorn runs with --proxy-headers
    --forwarded-allow-ips 127.0.0.1 (scripts/run.sh), so request.client is already the real client
    when a local reverse proxy is in front, and a client-supplied header cannot spoof it."""
    return (request.client.host if request.client else "unknown")[:64]
