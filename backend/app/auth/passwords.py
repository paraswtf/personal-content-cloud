"""Password hashing and the login rate limit (docs/ARCHITECTURE.md section 4).

Contract:
- hash_password(pw) -> str: argon2id hash (argon2-cffi PasswordHasher defaults).
- verify_password(hash, pw) -> bool: never raises; False on mismatch or a malformed hash.
- needs_rehash(hash) -> bool: True when the stored hash uses older parameters; the caller
  rewrites it after a successful verify (app.auth.account.check_password does).
- dummy_verify(pw): spends the same time as a real verify (unknown user names).
- claim_login_attempt(ip) -> bool: atomically checks the limit (10 failures per 600 s per IP,
  at most 5000 IPs tracked, in process memory) and counts this attempt as a failure up front;
  False means the caller answers 429. reset_login_failures(ip) runs after a success.
- claim_options_request(ip) -> bool: a separate, looser limit for the unauthenticated passkey
  login options route (30 requests per 60 s per IP), which stores a challenge row per call;
  False means 429. It never touches the login failure counter.
- reset_all(): clears every counter (tests).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

log = logging.getLogger(__name__)

MAX_FAILURES = 10
WINDOW_S = 600
MAX_TRACKED = 5000

_hasher = PasswordHasher()  # argon2id
_dummy_lock = threading.Lock()
_dummy_hash: str | None = None


def hash_password(pw: str) -> str:
    return _hasher.hash(pw)


def verify_password(hashed: str, pw: str) -> bool:
    if not hashed:
        return False
    try:
        return bool(_hasher.verify(hashed, pw))
    except (VerificationError, InvalidHashError):
        return False
    except Exception:  # an unexpected hash format must never turn into a 500
        log.warning("password verification failed on an unreadable hash", exc_info=True)
        return False


def needs_rehash(hashed: str) -> bool:
    try:
        return _hasher.check_needs_rehash(hashed)
    except Exception:
        return False


def dummy_verify(pw: str) -> None:
    """Verify against a throwaway hash so an unknown user name takes as long as a wrong password."""
    global _dummy_hash
    with _dummy_lock:
        if _dummy_hash is None:
            _dummy_hash = _hasher.hash("pcc-dummy-password-for-timing")
        hashed = _dummy_hash
    verify_password(hashed, pw)


# ----------------------------------------------------------------- login rate limit
_fail_lock = threading.Lock()
_failures: dict[str, deque[float]] = {}


def _prune(now: float) -> None:
    """Drop expired attempts and, if still at the cap, the IPs with the oldest last attempt.
    Called with _fail_lock held."""
    for key in list(_failures):
        q = _failures[key]
        while q and now - q[0] > WINDOW_S:
            q.popleft()
        if not q:
            del _failures[key]
    if len(_failures) >= MAX_TRACKED:
        excess = len(_failures) - MAX_TRACKED + 1
        for key, _ in sorted(_failures.items(), key=lambda kv: kv[1][-1])[:excess]:
            del _failures[key]


def claim_login_attempt(ip: str) -> bool:
    """Check the limit and count this attempt as a failure (a success resets it).

    Returns False when the client is over the limit. Counting before the check means parallel
    requests cannot all slip past it before any failure is recorded."""
    now = time.monotonic()
    with _fail_lock:
        q = _failures.get(ip)
        if q:
            while q and now - q[0] > WINDOW_S:
                q.popleft()
            if len(q) >= MAX_FAILURES:
                return False
        elif len(_failures) >= MAX_TRACKED:
            _prune(now)
        _failures.setdefault(ip, deque()).append(now)
        return True


def reset_login_failures(ip: str) -> None:
    with _fail_lock:
        _failures.pop(ip, None)


# ----------------------------------------------------------------- passkey login options rate limit
OPTIONS_MAX = 30
OPTIONS_WINDOW_S = 60

_options_lock = threading.Lock()
_options: dict[str, deque[float]] = {}


def claim_options_request(ip: str) -> bool:
    """Count one passkey login options request; False when the IP is over OPTIONS_MAX per window."""
    now = time.monotonic()
    with _options_lock:
        if ip not in _options and len(_options) >= MAX_TRACKED:
            for key in list(_options):
                q = _options[key]
                while q and now - q[0] > OPTIONS_WINDOW_S:
                    q.popleft()
                if not q:
                    del _options[key]
            if len(_options) >= MAX_TRACKED:  # still full: drop the IP with the oldest last request
                del _options[min(_options, key=lambda k: _options[k][-1])]
        q = _options.setdefault(ip, deque())
        while q and now - q[0] > OPTIONS_WINDOW_S:
            q.popleft()
        if len(q) >= OPTIONS_MAX:
            return False
        q.append(now)
        return True


# ----------------------------------------------------------------- share link password rate limit
SHARE_MAX_FAILURES = 10
SHARE_WINDOW_S = 600

_share_lock = threading.Lock()
_share_failures: dict[str, deque[float]] = {}


def claim_share_attempt(ip: str) -> bool:
    """Count one share password attempt as a failure up front (section 26); False when the IP is over
    SHARE_MAX_FAILURES per SHARE_WINDOW_S. Its own state: it never touches the login budget."""
    now = time.monotonic()
    with _share_lock:
        if ip not in _share_failures and len(_share_failures) >= MAX_TRACKED:
            for key in list(_share_failures):
                q = _share_failures[key]
                while q and now - q[0] > SHARE_WINDOW_S:
                    q.popleft()
                if not q:
                    del _share_failures[key]
            if len(_share_failures) >= MAX_TRACKED:
                del _share_failures[min(_share_failures, key=lambda k: _share_failures[k][-1])]
        q = _share_failures.setdefault(ip, deque())
        while q and now - q[0] > SHARE_WINDOW_S:
            q.popleft()
        if len(q) >= SHARE_MAX_FAILURES:
            return False
        q.append(now)
        return True


def reset_share_failures(ip: str) -> None:
    with _share_lock:
        _share_failures.pop(ip, None)


def reset_all() -> None:
    with _fail_lock:
        _failures.clear()
    with _options_lock:
        _options.clear()
    with _share_lock:
        _share_failures.clear()
