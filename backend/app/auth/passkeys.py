"""WebAuthn passkey ceremonies with py_webauthn 3.x (docs/ARCHITECTURE.md section 4).

Contract:
- registration_options(conn, request) -> (challenge_id, options): discoverable credential
  (resident_key=required), user_verification=preferred, existing passkeys excluded.
- verify_registration(conn, request, challenge_id, credential, name) -> sqlite3.Row (the passkeys row);
  raises PasskeyError("challenge_invalid" | "registration_failed" | "credential_exists").
- login_options(conn, request) -> (challenge_id, options): no allowCredentials (discoverable login).
- verify_login(conn, request, challenge_id, credential) -> bool.
- assert_options(conn, request) -> (challenge_id, options): re-authentication for the vault,
  listing the stored credentials.
- verify_assertion(conn, request, challenge_id, credential) -> bool.
- rp_id(request), expected_origins(request): from settings when set, else derived from the request
  (Host without port; scheme from X-Forwarded-Proto, then the URL scheme).

Challenges live in the `challenges` table under a random id with a 5-minute expiry. A challenge is
claimed (used=1) before the credential is even parsed, so every challenge serves one attempt only,
and its kind must match the ceremony. At most MAX_LOGIN_CHALLENGES login challenges are kept (oldest
dropped first). Options are JSON-ready dicts (options_to_json + json.loads).
"""
from __future__ import annotations

import json
import logging
import secrets
import sqlite3
from typing import Any

from fastapi import Request
from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import (
    base64url_to_bytes,
    options_to_json,
    parse_authentication_credential_json,
    parse_registration_credential_json,
)
from webauthn.helpers.structs import (
    AttestationConveyancePreference,
    AuthenticatorSelectionCriteria,
    AuthenticatorTransport,
    CredentialDeviceType,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

from app.auth.account import get_account
from app.config import settings
from app.db import tx
from app.util import clean_name, new_token, now_iso

log = logging.getLogger(__name__)

CHALLENGE_TTL_S = 300
TIMEOUT_MS = CHALLENGE_TTL_S * 1000
USER_HANDLE = b"pcc-account-1"  # stable WebAuthn user.id of the single account
KINDS = ("register", "login", "assert")
MAX_LOGIN_CHALLENGES = 500  # live unauthenticated login challenges kept at most


class PasskeyError(Exception):
    """A registration ceremony failed. `code` is the API detail string."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# ----------------------------------------------------------------- relying party
def _host_header(request: Request) -> str:
    host = (request.headers.get("host") or "").strip()
    if not host:
        host = request.url.netloc or (request.url.hostname or "localhost")
    return host


def _strip_port(host: str) -> str:
    if host.startswith("["):  # IPv6 literal: [::1]:8000
        end = host.find("]")
        return host[1:end] if end > 0 else host.strip("[]")
    if host.count(":") == 1:
        return host.split(":", 1)[0]
    return host


def _scheme(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return proto or request.url.scheme


def rp_id(request: Request) -> str:
    if settings.rp_id:
        return settings.rp_id.strip().lower()
    return _strip_port(_host_header(request)).lower()


def expected_origins(request: Request) -> list[str]:
    if settings.origin:
        return [o.strip().rstrip("/") for o in settings.origin.split(",") if o.strip()]
    return [f"{_scheme(request)}://{_host_header(request).lower()}"]


# ----------------------------------------------------------------- challenges
def _store_challenge(conn: sqlite3.Connection, kind: str, challenge: bytes) -> str:
    assert kind in KINDS
    challenge_id = new_token(18)
    now = now_iso()
    with tx(conn):
        conn.execute("DELETE FROM challenges WHERE expires_at < ? OR used = 1", (now,))
        conn.execute(
            "INSERT INTO challenges(id, kind, challenge, created_at, expires_at, used) VALUES (?, ?, ?, ?, ?, 0)",
            (challenge_id, kind, challenge, now, now_iso(CHALLENGE_TTL_S)),
        )
        if kind == "login":
            # Login options need no session, so a flood could fill the table: keep only the newest
            # MAX_LOGIN_CHALLENGES. Register and assert challenges (session only) are never evicted.
            conn.execute(
                "DELETE FROM challenges WHERE kind = 'login' AND id NOT IN ("
                "SELECT id FROM challenges WHERE kind = 'login' ORDER BY created_at DESC, rowid DESC LIMIT ?)",
                (MAX_LOGIN_CHALLENGES,),
            )
    return challenge_id


def _claim_challenge(conn: sqlite3.Connection, challenge_id: str, kind: str) -> bytes | None:
    """Mark the challenge used and return its bytes, or None when unknown, expired, used or of another kind."""
    if not challenge_id or len(challenge_id) > 64:
        return None
    with tx(conn):
        cur = conn.execute(
            "UPDATE challenges SET used = 1 WHERE id = ? AND kind = ? AND used = 0 AND expires_at > ?",
            (challenge_id, kind, now_iso()),
        )
        if cur.rowcount != 1:
            return None
        row = conn.execute("SELECT challenge FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
    return bytes(row["challenge"]) if row else None


def _options_dict(options: Any) -> dict:
    return json.loads(options_to_json(options))


def _transports(raw: Any) -> list[AuthenticatorTransport]:
    out: list[AuthenticatorTransport] = []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = []
    for value in raw or []:
        try:
            out.append(AuthenticatorTransport(value.value if isinstance(value, AuthenticatorTransport) else value))
        except (ValueError, AttributeError):
            continue
    return out


def _descriptors(conn: sqlite3.Connection) -> list[PublicKeyCredentialDescriptor]:
    rows = conn.execute("SELECT credential_id, transports FROM passkeys ORDER BY id").fetchall()
    return [
        PublicKeyCredentialDescriptor(id=bytes(r["credential_id"]), transports=_transports(r["transports"]) or None)
        for r in rows
    ]


# ----------------------------------------------------------------- registration
def registration_options(conn: sqlite3.Connection, request: Request) -> tuple[str, dict]:
    account = get_account(conn)
    username = account["username"] if account else "owner"
    challenge = secrets.token_bytes(32)
    options = generate_registration_options(
        rp_id=rp_id(request),
        rp_name=settings.rp_name or settings.site_title,
        user_name=username,
        user_id=USER_HANDLE,
        user_display_name=username,
        challenge=challenge,
        timeout=TIMEOUT_MS,
        attestation=AttestationConveyancePreference.NONE,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.REQUIRED,
            require_resident_key=True,
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
        exclude_credentials=_descriptors(conn),
    )
    return _store_challenge(conn, "register", challenge), _options_dict(options)


def verify_registration(conn: sqlite3.Connection, request: Request, challenge_id: str, credential: dict,
                        name: str) -> sqlite3.Row:
    challenge = _claim_challenge(conn, challenge_id, "register")
    if challenge is None:
        raise PasskeyError("challenge_invalid")
    try:
        parsed = parse_registration_credential_json(json.dumps(credential))
        verified = verify_registration_response(
            credential=parsed,
            expected_challenge=challenge,
            expected_rp_id=rp_id(request),
            expected_origin=expected_origins(request),
            require_user_verification=False,
        )
    except Exception as exc:  # py_webauthn errors, malformed dicts, bad base64
        log.info("passkey registration rejected: %s", exc)
        raise PasskeyError("registration_failed") from exc

    transports = [t.value for t in _transports(getattr(parsed.response, "transports", None) or [])]
    label = clean_name(name, 64) or "Passkey"
    now = now_iso()
    try:
        with tx(conn):
            cur = conn.execute(
                "INSERT INTO passkeys(credential_id, public_key, sign_count, transports, name, aaguid, "
                "backup_eligible, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    bytes(verified.credential_id), bytes(verified.credential_public_key), int(verified.sign_count),
                    json.dumps(transports), label, str(verified.aaguid or ""),
                    1 if verified.credential_device_type == CredentialDeviceType.MULTI_DEVICE else 0, now,
                ),
            )
            passkey_id = int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise PasskeyError("credential_exists") from exc
    log.info("passkey registered (id %s)", passkey_id)
    return conn.execute("SELECT * FROM passkeys WHERE id = ?", (passkey_id,)).fetchone()


# ----------------------------------------------------------------- authentication
def login_options(conn: sqlite3.Connection, request: Request) -> tuple[str, dict]:
    challenge = secrets.token_bytes(32)
    options = generate_authentication_options(
        rp_id=rp_id(request), challenge=challenge, timeout=TIMEOUT_MS,
        user_verification=UserVerificationRequirement.PREFERRED,
    )
    return _store_challenge(conn, "login", challenge), _options_dict(options)


def assert_options(conn: sqlite3.Connection, request: Request) -> tuple[str, dict]:
    challenge = secrets.token_bytes(32)
    options = generate_authentication_options(
        rp_id=rp_id(request), challenge=challenge, timeout=TIMEOUT_MS,
        allow_credentials=_descriptors(conn),
        user_verification=UserVerificationRequirement.PREFERRED,
    )
    return _store_challenge(conn, "assert", challenge), _options_dict(options)


def _credential_id(credential: dict) -> bytes | None:
    if not isinstance(credential, dict):
        return None
    raw = credential.get("rawId") or credential.get("id")
    if not isinstance(raw, str) or not raw or len(raw) > 1400:
        return None
    try:
        return base64url_to_bytes(raw)
    except Exception:
        return None


def _authenticate(conn: sqlite3.Connection, request: Request, challenge_id: str, credential: dict, kind: str) -> bool:
    challenge = _claim_challenge(conn, challenge_id, kind)
    if challenge is None:
        log.info("passkey %s rejected: challenge unknown, used or expired", kind)
        return False
    cred_id = _credential_id(credential)
    if cred_id is None:
        return False
    row = conn.execute("SELECT * FROM passkeys WHERE credential_id = ?", (cred_id,)).fetchone()
    if row is None:
        log.info("passkey %s rejected: unknown credential", kind)
        return False
    try:
        parsed = parse_authentication_credential_json(json.dumps(credential))
        user_handle = parsed.response.user_handle
        if user_handle is not None and bytes(user_handle) != USER_HANDLE:
            log.info("passkey %s rejected: user handle does not match", kind)
            return False
        verified = verify_authentication_response(
            credential=parsed,
            expected_challenge=challenge,
            expected_rp_id=rp_id(request),
            expected_origin=expected_origins(request),
            credential_public_key=bytes(row["public_key"]),
            credential_current_sign_count=int(row["sign_count"] or 0),
            require_user_verification=False,
        )
    except Exception as exc:
        log.info("passkey %s rejected: %s", kind, exc)
        return False
    with tx(conn):
        conn.execute(
            "UPDATE passkeys SET sign_count = ?, last_used_at = ?, backup_eligible = MAX(backup_eligible, ?) "
            "WHERE id = ?",
            (int(verified.new_sign_count), now_iso(),
             1 if verified.credential_device_type == CredentialDeviceType.MULTI_DEVICE else 0, row["id"]),
        )
    return True


def verify_login(conn: sqlite3.Connection, request: Request, challenge_id: str, credential: dict) -> bool:
    return _authenticate(conn, request, challenge_id, credential, "login")


def verify_assertion(conn: sqlite3.Connection, request: Request, challenge_id: str, credential: dict) -> bool:
    return _authenticate(conn, request, challenge_id, credential, "assert")


# ----------------------------------------------------------------- management
def list_passkeys(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM passkeys ORDER BY created_at, id").fetchall()


def count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM passkeys").fetchone()[0])


def rename(conn: sqlite3.Connection, passkey_id: int, name: str) -> sqlite3.Row | None:
    label = clean_name(name, 64)
    if not label:
        raise ValueError("empty name")
    with tx(conn):
        conn.execute("UPDATE passkeys SET name = ? WHERE id = ?", (label, passkey_id))
    return conn.execute("SELECT * FROM passkeys WHERE id = ?", (passkey_id,)).fetchone()


def delete(conn: sqlite3.Connection, passkey_id: int) -> bool:
    with tx(conn):
        return conn.execute("DELETE FROM passkeys WHERE id = ?", (passkey_id,)).rowcount == 1
