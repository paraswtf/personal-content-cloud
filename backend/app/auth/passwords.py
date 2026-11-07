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


def reset_all() -> None:
    with _fail_lock:
        _failures.clear()
