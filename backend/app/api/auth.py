"""/api/auth routes (docs/ARCHITECTURE.md sections 4 and 13).

Contract: `router` is a plain APIRouter without a prefix; main.py mounts it with
`app.include_router(auth.router, prefix="/api/auth")`.

- GET state needs no login (optional_user); setup_required until the account exists.
- POST setup, login, passkeys/login/options, passkeys/login/verify need no session but do need
  the CSRF header. Every other route needs require_user (which also checks CSRF on writes).
- Wrong credentials answer 401 `invalid_credentials`; too many failures 429 `too_many_attempts`;
  setup after the account exists 409 `already_set_up`; login before setup 409 `setup_required`.
- setup, login and passkey login create a session and set the `pcc_session` cookie.
- Adding (register options and verify) or removing a passkey needs an unlocked vault (403
  `vault_locked`): a passkey unlocks the vault and signs in, so a session that has not re-entered the
  password or a passkey recently must not be able to enroll one of its own.
- passkeys/login/options is rate limited per IP (429 `too_many_attempts`), separately from logins.
"""
from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.api.deps import check_csrf, get_db, optional_user, require_user, require_vault
from app.auth import account as accounts
from app.auth import passkeys, passwords, sessions
from app.auth.sessions import SessionInfo, client_ip, vault_open
from app.config import settings
from app.schemas import (
    AuthState,
    ChallengeOut,
    DeviceCreated,
    DeviceIn,
    LoginIn,
    Ok,
    PasskeyLoginVerifyIn,
    PasskeyOut,
    PasskeyRegisterVerifyIn,
    PasskeyRenameIn,
    PasswordChangeIn,
    SessionOut,
    SetupIn,
    VaultUnlockIn,
)

log = logging.getLogger(__name__)

router = APIRouter()


# ----------------------------------------------------------------- helpers
def _state(conn: sqlite3.Connection, session: SessionInfo | None, vault_until: str | None = None) -> AuthState:
    acc = accounts.get_account(conn)
    if session is not None and vault_until is None:
        row = conn.execute("SELECT vault_until FROM sessions WHERE id = ?", (session.id,)).fetchone()
        vault_until = row["vault_until"] if row else None
    return AuthState(
        setup_required=acc is None,
        authenticated=session is not None and acc is not None,
        username=acc["username"] if (acc is not None and session is not None) else None,
        passkeys=passkeys.count(conn) if session is not None else 0,
        vault_until=vault_until if (session is not None and vault_open(vault_until)) else None,
        session_id=session.id if session is not None else None,
        site_title=settings.site_title,
    )


def _start_session(conn: sqlite3.Connection, request: Request, response: Response) -> SessionInfo:
    token, info = sessions.create_session(conn, request)
    sessions.set_cookie(response, request, token)
    return info


def _claim(request: Request) -> str:
    ip = client_ip(request)
    if not passwords.claim_login_attempt(ip):
        raise HTTPException(status_code=429, detail="too_many_attempts")
    return ip


def _require_account(conn: sqlite3.Connection) -> None:
    if accounts.get_account(conn) is None:
        raise HTTPException(status_code=409, detail="setup_required")


def _passkey_out(row: sqlite3.Row) -> PasskeyOut:
    return PasskeyOut(
        id=int(row["id"]), name=row["name"], created_at=row["created_at"], last_used_at=row["last_used_at"],
        backup_eligible=bool(row["backup_eligible"]),
    )


# ----------------------------------------------------------------- state, setup, login
@router.get("/state", response_model=AuthState)
def state(session: SessionInfo | None = Depends(optional_user), conn: sqlite3.Connection = Depends(get_db)):
    return _state(conn, session)


@router.post("/setup", response_model=AuthState, dependencies=[Depends(check_csrf)])
def setup(body: SetupIn, request: Request, response: Response, conn: sqlite3.Connection = Depends(get_db)):
    try:
        accounts.create_account(conn, body.username, body.password)
    except accounts.AccountExists:
        raise HTTPException(status_code=409, detail="already_set_up")
    except ValueError:
        raise HTTPException(status_code=422, detail="username_required")
    info = _start_session(conn, request, response)
    return _state(conn, info)


@router.post("/login", response_model=AuthState, dependencies=[Depends(check_csrf)])
def login(body: LoginIn, request: Request, response: Response, conn: sqlite3.Connection = Depends(get_db)):
    _require_account(conn)
    ip = _claim(request)
    if not accounts.check_credentials(conn, body.username, body.password):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    passwords.reset_login_failures(ip)
    info = _start_session(conn, request, response)
    return _state(conn, info)


@router.post("/logout", response_model=AuthState)
def logout(request: Request, response: Response, session: SessionInfo = Depends(require_user),
           conn: sqlite3.Connection = Depends(get_db)):
    sessions.revoke(conn, session.id)
    sessions.clear_cookie(response, request)
    return _state(conn, None)


@router.post("/logout-all", response_model=AuthState)
def logout_all(request: Request, response: Response, session: SessionInfo = Depends(require_user),
               conn: sqlite3.Connection = Depends(get_db)):
    n = sessions.revoke_all(conn)
    log.info("signed out everywhere (%d sessions)", n)
    sessions.clear_cookie(response, request)
    return _state(conn, None)


# ----------------------------------------------------------------- sessions, password
@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(session: SessionInfo = Depends(require_user), conn: sqlite3.Connection = Depends(get_db)):
    return [
        SessionOut(id=s.id, created_at=s.created_at, last_seen_at=s.last_seen_at, user_agent=s.user_agent, ip=s.ip,
                   current=s.id == session.id)
        for s in sessions.list_sessions(conn) if not s.device_name
    ]


@router.delete("/sessions/{session_id}", response_model=Ok)
def revoke_session(session_id: int, request: Request, response: Response,
                   session: SessionInfo = Depends(require_user), conn: sqlite3.Connection = Depends(get_db)):
    if not sessions.revoke(conn, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    if session_id == session.id:
        sessions.clear_cookie(response, request)
    return Ok()


@router.post("/password", response_model=Ok)
def change_password(body: PasswordChangeIn, request: Request, session: SessionInfo = Depends(require_user),
                    conn: sqlite3.Connection = Depends(get_db)):
    ip = _claim(request)
    if not accounts.check_password(conn, body.current_password):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    passwords.reset_login_failures(ip)
    accounts.change_password(conn, body.new_password)
    sessions.revoke_all(conn, except_id=session.id)  # other devices sign in again with the new password
    return Ok()


# ----------------------------------------------------------------- passkeys
@router.get("/passkeys", response_model=list[PasskeyOut])
def list_passkeys(session: SessionInfo = Depends(require_user), conn: sqlite3.Connection = Depends(get_db)):
    return [_passkey_out(r) for r in passkeys.list_passkeys(conn)]


@router.post("/passkeys/register/options", response_model=ChallengeOut)
def passkey_register_options(request: Request, session: SessionInfo = Depends(require_vault),
                             conn: sqlite3.Connection = Depends(get_db)):
    challenge_id, options = passkeys.registration_options(conn, request)
    return ChallengeOut(challenge_id=challenge_id, options=options)


@router.post("/passkeys/register/verify", response_model=PasskeyOut)
def passkey_register_verify(body: PasskeyRegisterVerifyIn, request: Request,
                            session: SessionInfo = Depends(require_vault), conn: sqlite3.Connection = Depends(get_db)):
    try:
        row = passkeys.verify_registration(conn, request, body.challenge_id, body.credential, body.name)
    except passkeys.PasskeyError as exc:
        raise HTTPException(status_code=409 if exc.code == "credential_exists" else 400, detail=exc.code)
    return _passkey_out(row)


@router.patch("/passkeys/{passkey_id}", response_model=PasskeyOut)
def passkey_rename(passkey_id: int, body: PasskeyRenameIn, session: SessionInfo = Depends(require_user),
                   conn: sqlite3.Connection = Depends(get_db)):
    try:
        row = passkeys.rename(conn, passkey_id, body.name)
    except ValueError:
        raise HTTPException(status_code=422, detail="name_required")
    if row is None:
        raise HTTPException(status_code=404, detail="Passkey not found")
    return _passkey_out(row)


@router.delete("/passkeys/{passkey_id}", response_model=Ok)
def passkey_delete(passkey_id: int, session: SessionInfo = Depends(require_vault),
                   conn: sqlite3.Connection = Depends(get_db)):
    if not passkeys.delete(conn, passkey_id):
        raise HTTPException(status_code=404, detail="Passkey not found")
    return Ok()


@router.post("/passkeys/login/options", response_model=ChallengeOut, dependencies=[Depends(check_csrf)])
def passkey_login_options(request: Request, conn: sqlite3.Connection = Depends(get_db)):
    _require_account(conn)
    if not passwords.claim_options_request(client_ip(request)):
        raise HTTPException(status_code=429, detail="too_many_attempts")
    challenge_id, options = passkeys.login_options(conn, request)
    return ChallengeOut(challenge_id=challenge_id, options=options)


@router.post("/passkeys/login/verify", response_model=AuthState, dependencies=[Depends(check_csrf)])
def passkey_login_verify(body: PasskeyLoginVerifyIn, request: Request, response: Response,
                         conn: sqlite3.Connection = Depends(get_db)):
    _require_account(conn)
    ip = _claim(request)
    if not passkeys.verify_login(conn, request, body.challenge_id, body.credential):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    passwords.reset_login_failures(ip)
    info = _start_session(conn, request, response)
    return _state(conn, info)


@router.post("/passkeys/assert/options", response_model=ChallengeOut)
def passkey_assert_options(request: Request, session: SessionInfo = Depends(require_user),
                           conn: sqlite3.Connection = Depends(get_db)):
    challenge_id, options = passkeys.assert_options(conn, request)
    return ChallengeOut(challenge_id=challenge_id, options=options)


# ----------------------------------------------------------------- vault
@router.post("/vault/unlock", response_model=AuthState)
def vault_unlock(body: VaultUnlockIn, request: Request, session: SessionInfo = Depends(require_user),
                 conn: sqlite3.Connection = Depends(get_db)):
    if body.password is not None:
        ip = _claim(request)
        if not accounts.check_password(conn, body.password):
            raise HTTPException(status_code=401, detail="invalid_credentials")
        passwords.reset_login_failures(ip)
    elif body.challenge_id and body.credential is not None:
        ip = _claim(request)
        if not passkeys.verify_assertion(conn, request, body.challenge_id, body.credential):
            raise HTTPException(status_code=401, detail="invalid_credentials")
        passwords.reset_login_failures(ip)
    else:
        raise HTTPException(status_code=400, detail="password_or_passkey_required")
    until = sessions.vault_unlock(conn, session.id)
    return _state(conn, session, vault_until=until)


@router.post("/vault/lock", response_model=AuthState)
def vault_lock(session: SessionInfo = Depends(require_user), conn: sqlite3.Connection = Depends(get_db)):
    sessions.vault_lock(conn, session.id)
    return _state(conn, session)


# ----------------------------------------------------------------- device tokens (the iOS app)
@router.get("/devices", response_model=list[SessionOut])
def list_devices(session: SessionInfo = Depends(require_user), conn: sqlite3.Connection = Depends(get_db)):
    return [
        SessionOut(id=s.id, created_at=s.created_at, last_seen_at=s.last_seen_at, user_agent=s.user_agent, ip=s.ip,
                   current=s.id == session.id, device_name=s.device_name)
        for s in sessions.list_devices(conn)
    ]


@router.post("/devices", response_model=DeviceCreated)
def create_device(body: DeviceIn, request: Request, session: SessionInfo = Depends(require_vault),
                  conn: sqlite3.Connection = Depends(get_db)):
    """A long-lived token for a device (365 days, sliding), shown once. Needs the vault open, like
    enrolling a passkey: it is a credential."""
    name = " ".join(body.name.split())[:80] or "Device"
    token, info = sessions.create_session(conn, request, device_name=name)
    log.info("device token created for %r", name)
    return DeviceCreated(id=info.id, name=name, token=token)


@router.delete("/devices/{session_id}", response_model=Ok)
def revoke_device(session_id: int, session: SessionInfo = Depends(require_user),
                  conn: sqlite3.Connection = Depends(get_db)):
    row = conn.execute("SELECT device_name FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None or not row["device_name"] or not sessions.revoke(conn, session_id):
        raise HTTPException(status_code=404, detail="No such device")
    return Ok()
