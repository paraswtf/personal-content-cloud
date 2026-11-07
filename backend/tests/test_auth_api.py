"""/api/auth routes driven through a FastAPI app that mounts only the auth router (plus a test-only
route guarded by require_vault), so these tests do not depend on main.py."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from tests.conftest import ACCOUNT, CSRF


@pytest.fixture(autouse=True)
def _reset_rate_limit(monkeypatch):
    """Fresh limiter, and the relying party derived from the request whatever the shell exports."""
    from app.auth import passwords
    from app.config import settings

    passwords.reset_all()
    monkeypatch.delenv("RP_ID", raising=False)
    monkeypatch.delenv("ORIGIN", raising=False)
    monkeypatch.setattr(settings, "rp_id", None, raising=False)
    monkeypatch.setattr(settings, "origin", None, raising=False)
    yield
    passwords.reset_all()


@pytest.fixture()
def auth_app(data_dir):
    from app.api import auth
    from app.api.deps import require_vault

    app = FastAPI()
    app.include_router(auth.router, prefix="/api/auth")

    @app.get("/api/test/vault")
    def vault_only(session=Depends(require_vault)):
        return {"ok": True}

    return app


@pytest.fixture()
def anon(auth_app):
    with TestClient(auth_app) as c:
        yield c


@pytest.fixture()
def owner(auth_app):
    with TestClient(auth_app) as c:
        r = c.post("/api/auth/setup", json=ACCOUNT, headers=CSRF)
        assert r.status_code == 200, r.text
        c.headers.update(CSRF)
        yield c


def test_state_before_setup(anon):
    r = anon.get("/api/auth/state")
    assert r.status_code == 200
    body = r.json()
    assert body["setup_required"] is True and body["authenticated"] is False
    assert body["username"] is None and body["session_id"] is None and body["passkeys"] == 0
    assert body["site_title"]


def test_setup_once_then_409(anon):
    r = anon.post("/api/auth/setup", json=ACCOUNT, headers=CSRF)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["authenticated"] is True and body["setup_required"] is False
    assert body["username"] == ACCOUNT["username"] and body["session_id"]
    assert "pcc_session" in r.cookies or anon.cookies.get("pcc_session")
    # The cookie works right away.
    assert anon.get("/api/auth/sessions").status_code == 200
    r = anon.post("/api/auth/setup", json={"username": "other", "password": "x" * 12}, headers=CSRF)
    assert r.status_code == 409 and r.json()["detail"] == "already_set_up"


def test_setup_validation(anon):
    assert anon.post("/api/auth/setup", json={"username": "a", "password": "short"}, headers=CSRF).status_code == 422
    assert anon.post("/api/auth/setup", json={"username": "", "password": "x" * 12}, headers=CSRF).status_code == 422
    assert anon.post("/api/auth/setup", json={"username": "   ", "password": "x" * 12}, headers=CSRF).status_code == 422


def test_password_is_argon2id(owner, conn):
    row = conn.execute("SELECT * FROM account").fetchone()
    assert row["id"] == 1 and row["password_hash"].startswith("$argon2id$")
    assert row["username_key"] == "owner"
    assert ACCOUNT["password"] not in row["password_hash"]


def test_csrf_required_on_bootstrap_writes(anon):
    r = anon.post("/api/auth/setup", json=ACCOUNT)
    assert r.status_code == 403 and r.json()["detail"] == "csrf"
    anon.post("/api/auth/setup", json=ACCOUNT, headers=CSRF)
    assert anon.post("/api/auth/login", json=ACCOUNT).status_code == 403
    assert anon.post("/api/auth/passkeys/login/options").status_code == 403
    # Logged in, writes still need the header.
    assert anon.post("/api/auth/logout").status_code == 403
    assert anon.post("/api/auth/vault/lock").status_code == 403


def test_login_wrong_and_right(owner, auth_app):
    with TestClient(auth_app) as c:
        r = c.post("/api/auth/login", json={"username": "owner", "password": "wrong password"}, headers=CSRF)
        assert r.status_code == 401 and r.json()["detail"] == "invalid_credentials"
        r = c.post("/api/auth/login", json={"username": "nobody", "password": ACCOUNT["password"]}, headers=CSRF)
        assert r.status_code == 401
        # User name is case-insensitive.
        r = c.post("/api/auth/login", json={"username": "  OWNER ", "password": ACCOUNT["password"]}, headers=CSRF)
        assert r.status_code == 200, r.text
        assert r.json()["authenticated"] is True
        assert c.get("/api/auth/state").json()["authenticated"] is True


def test_login_before_setup_is_409(anon):
    r = anon.post("/api/auth/login", json=ACCOUNT, headers=CSRF)
    assert r.status_code == 409 and r.json()["detail"] == "setup_required"


def test_protected_route_without_cookie(anon):
    r = anon.get("/api/auth/sessions")
    assert r.status_code == 401 and r.json()["detail"] == "login_required"
    anon.post("/api/auth/setup", json=ACCOUNT, headers=CSRF)
    anon.cookies.clear()
    assert anon.get("/api/auth/sessions").status_code == 401
    assert anon.get("/api/auth/passkeys").status_code == 401
    assert anon.post("/api/auth/vault/unlock", json={"password": "x"}, headers=CSRF).status_code == 401


def test_state_when_logged_in(owner):
    body = owner.get("/api/auth/state").json()
    assert body["authenticated"] is True and body["username"] == "owner"
    assert body["vault_until"] is None and body["passkeys"] == 0


def test_logout_clears(owner):
    r = owner.post("/api/auth/logout")
    assert r.status_code == 200 and r.json()["authenticated"] is False
    assert owner.get("/api/auth/sessions").status_code == 401
    assert owner.get("/api/auth/state").json()["authenticated"] is False


def test_logout_revokes_the_token_server_side(owner):
    token = owner.cookies.get("pcc_session")
    owner.post("/api/auth/logout")
    owner.cookies.set("pcc_session", token)
    assert owner.get("/api/auth/sessions").status_code == 401


def test_sessions_list_and_logout_all(owner, auth_app):
    with TestClient(auth_app) as other:
        r = other.post("/api/auth/login", json=ACCOUNT, headers=CSRF)
        assert r.status_code == 200
        other_id = r.json()["session_id"]
        rows = owner.get("/api/auth/sessions").json()
        assert len(rows) == 2
        current = [s for s in rows if s["current"]]
        assert len(current) == 1 and current[0]["id"] != other_id
        # Revoke the other device.
        assert owner.delete(f"/api/auth/sessions/{other_id}").status_code == 200
        assert other.get("/api/auth/sessions").status_code == 401
        assert owner.delete(f"/api/auth/sessions/{other_id}").status_code == 404
        # Sign in again and sign out everywhere.
        other.post("/api/auth/login", json=ACCOUNT, headers=CSRF)
        r = owner.post("/api/auth/logout-all")
        assert r.status_code == 200 and r.json()["authenticated"] is False
        assert owner.get("/api/auth/sessions").status_code == 401
        assert other.get("/api/auth/sessions").status_code == 401


def test_revoke_own_session(owner):
    sid = owner.get("/api/auth/state").json()["session_id"]
    assert owner.delete(f"/api/auth/sessions/{sid}").status_code == 200
    assert owner.get("/api/auth/state").json()["authenticated"] is False


def test_change_password(owner, auth_app):
    with TestClient(auth_app) as other:
        other.post("/api/auth/login", json=ACCOUNT, headers=CSRF)
        r = owner.post("/api/auth/password", json={"current_password": "nope", "new_password": "y" * 12})
        assert r.status_code == 401
        r = owner.post("/api/auth/password",
                       json={"current_password": ACCOUNT["password"], "new_password": "a brand new password"})
        assert r.status_code == 200 and r.json() == {"ok": True}
        # This session stays; the other device is signed out.
        assert owner.get("/api/auth/sessions").status_code == 200
        assert other.get("/api/auth/sessions").status_code == 401
        assert other.post("/api/auth/login", json=ACCOUNT, headers=CSRF).status_code == 401
        r = other.post("/api/auth/login", json={"username": "owner", "password": "a brand new password"}, headers=CSRF)
        assert r.status_code == 200


def test_vault_unlock_with_password_and_window(owner, monkeypatch):
    from app.config import settings

    assert owner.get("/api/test/vault").status_code == 403
    assert owner.get("/api/test/vault").json()["detail"] == "vault_locked"
    r = owner.post("/api/auth/vault/unlock", json={"password": "wrong password"})
    assert r.status_code == 401 and r.json()["detail"] == "invalid_credentials"

    monkeypatch.setattr(settings, "vault_unlock_min", 15)
    before = datetime.now(timezone.utc)
    r = owner.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]})
    assert r.status_code == 200, r.text
    until = datetime.fromisoformat(r.json()["vault_until"].replace("Z", "+00:00"))
    assert timedelta(minutes=14, seconds=50) < until - before < timedelta(minutes=15, seconds=10)
    assert owner.get("/api/test/vault").status_code == 200
    assert owner.get("/api/auth/state").json()["vault_until"] == r.json()["vault_until"]

    r = owner.post("/api/auth/vault/lock")
    assert r.status_code == 200 and r.json()["vault_until"] is None
    assert owner.get("/api/test/vault").status_code == 403


def test_vault_window_expires(owner, monkeypatch, conn):
    from app.config import settings

    monkeypatch.setattr(settings, "vault_unlock_min", 1)
    r = owner.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert owner.get("/api/test/vault").status_code == 200
    # Move the window into the past.
    past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    conn.execute("UPDATE sessions SET vault_until = ? WHERE id = ?", (past, sid))
    assert owner.get("/api/test/vault").status_code == 403
    assert owner.get("/api/auth/state").json()["vault_until"] is None


def test_vault_unlock_needs_a_factor(owner):
    r = owner.post("/api/auth/vault/unlock", json={})
    assert r.status_code == 400
    r = owner.post("/api/auth/vault/unlock", json={"challenge_id": "x", "credential": {"id": "AAAA"}})
    assert r.status_code == 401


def test_login_rate_limit(owner, auth_app):
    with TestClient(auth_app) as c:
        for _ in range(10):
            r = c.post("/api/auth/login", json={"username": "owner", "password": "bad password"}, headers=CSRF)
            assert r.status_code == 401
        r = c.post("/api/auth/login", json=ACCOUNT, headers=CSRF)
        assert r.status_code == 429 and r.json()["detail"] == "too_many_attempts"
        # The vault password counts against the same budget.
        assert owner.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]}).status_code == 429


def test_success_resets_rate_limit(owner, auth_app):
    with TestClient(auth_app) as c:
        for _ in range(9):
            c.post("/api/auth/login", json={"username": "owner", "password": "bad password"}, headers=CSRF)
        assert c.post("/api/auth/login", json=ACCOUNT, headers=CSRF).status_code == 200
        for _ in range(9):
            assert c.post("/api/auth/login", json={"username": "owner", "password": "bad"}, headers=CSRF).status_code == 401
        assert c.post("/api/auth/login", json=ACCOUNT, headers=CSRF).status_code == 200


def test_passkey_management(owner, conn):
    from app.db import tx
    from app.util import now_iso

    with tx(conn):
        conn.execute(
            "INSERT INTO passkeys(credential_id, public_key, sign_count, transports, name, backup_eligible, created_at) "
            "VALUES (?, ?, 0, ?, 'Phone', 1, ?)",
            (b"\x01\x02\x03", b"\x04\x05", json.dumps(["internal"]), now_iso()),
        )
    rows = owner.get("/api/auth/passkeys").json()
    assert len(rows) == 1 and rows[0]["name"] == "Phone" and rows[0]["backup_eligible"] is True
    pid = rows[0]["id"]
    assert owner.get("/api/auth/state").json()["passkeys"] == 1
    r = owner.patch(f"/api/auth/passkeys/{pid}", json={"name": "  Work laptop "})
    assert r.status_code == 200 and r.json()["name"] == "Work laptop"
    assert owner.patch("/api/auth/passkeys/9999", json={"name": "x"}).status_code == 404
    _unlock(owner)
    assert owner.delete(f"/api/auth/passkeys/{pid}").status_code == 200
    assert owner.delete(f"/api/auth/passkeys/{pid}").status_code == 404
    assert owner.get("/api/auth/passkeys").json() == []


def _unlock(client):
    r = client.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]})
    assert r.status_code == 200, r.text


def test_passkey_option_routes(owner, anon):
    _unlock(owner)
    r = owner.post("/api/auth/passkeys/register/options")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["challenge_id"] and body["options"]["rp"]["id"] == "testserver"
    assert body["options"]["authenticatorSelection"]["residentKey"] == "required"
    r = owner.post("/api/auth/passkeys/assert/options")
    assert r.status_code == 200 and r.json()["options"]["rpId"] == "testserver"
    # Login options need no session (but the CSRF header).
    r = anon.post("/api/auth/passkeys/login/options", headers=CSRF)
    assert r.status_code == 200
    assert not r.json()["options"].get("allowCredentials")
    # Registration options need a session.
    assert anon.post("/api/auth/passkeys/register/options", headers=CSRF).status_code == 401


def test_passkey_login_verify_rejects_unknown(owner, anon):
    ch = anon.post("/api/auth/passkeys/login/options", headers=CSRF).json()
    cred = {"id": "AAECAw", "rawId": "AAECAw", "type": "public-key",
            "response": {"clientDataJSON": "e30", "authenticatorData": "AA", "signature": "AA"}}
    r = anon.post("/api/auth/passkeys/login/verify", json={"challenge_id": ch["challenge_id"], "credential": cred},
                  headers=CSRF)
    assert r.status_code == 401 and r.json()["detail"] == "invalid_credentials"
    assert anon.get("/api/auth/state").json()["authenticated"] is False


def test_passkey_register_verify_rejects_garbage(owner):
    _unlock(owner)
    ch = owner.post("/api/auth/passkeys/register/options").json()
    r = owner.post("/api/auth/passkeys/register/verify",
                   json={"challenge_id": ch["challenge_id"], "name": "Key", "credential": {"id": "x"}})
    assert r.status_code == 400 and r.json()["detail"] == "registration_failed"
    # The challenge was single use.
    r = owner.post("/api/auth/passkeys/register/verify",
                   json={"challenge_id": ch["challenge_id"], "name": "Key", "credential": {"id": "x"}})
    assert r.status_code == 400 and r.json()["detail"] == "challenge_invalid"


def test_passkey_enroll_and_delete_need_unlocked_vault(owner, conn):
    """Regression: a session with the vault locked must not enroll (or remove) a passkey, since a
    passkey both signs in and unlocks the vault."""
    from app.db import tx
    from app.util import now_iso

    with tx(conn):
        conn.execute(
            "INSERT INTO passkeys(credential_id, public_key, sign_count, transports, name, backup_eligible, created_at) "
            "VALUES (?, ?, 0, '[]', 'Phone', 0, ?)",
            (b"\x01\x02\x03", b"\x04\x05", now_iso()),
        )
    pid = owner.get("/api/auth/passkeys").json()[0]["id"]
    r = owner.post("/api/auth/passkeys/register/options")
    assert r.status_code == 403 and r.json()["detail"] == "vault_locked"
    r = owner.post("/api/auth/passkeys/register/verify", json={"challenge_id": "x", "name": "Key", "credential": {"id": "x"}})
    assert r.status_code == 403 and r.json()["detail"] == "vault_locked"
    r = owner.delete(f"/api/auth/passkeys/{pid}")
    assert r.status_code == 403 and r.json()["detail"] == "vault_locked"
    assert conn.execute("SELECT COUNT(*) FROM challenges WHERE kind = 'register'").fetchone()[0] == 0
    _unlock(owner)
    assert owner.post("/api/auth/passkeys/register/options").status_code == 200
    assert owner.delete(f"/api/auth/passkeys/{pid}").status_code == 200
    # Locking again closes the door again.
    owner.post("/api/auth/vault/lock")
    assert owner.post("/api/auth/passkeys/register/options").status_code == 403


def test_passkey_login_options_rate_limited(owner, anon):
    """Regression: anonymous login/options calls are limited per IP, without touching the login limit."""
    from app.auth import passwords

    for _ in range(passwords.OPTIONS_MAX):
        assert anon.post("/api/auth/passkeys/login/options", headers=CSRF).status_code == 200
    r = anon.post("/api/auth/passkeys/login/options", headers=CSRF)
    assert r.status_code == 429 and r.json()["detail"] == "too_many_attempts"
    # Password login keeps its own bucket.
    assert anon.post("/api/auth/login", json=ACCOUNT, headers=CSRF).status_code == 200
