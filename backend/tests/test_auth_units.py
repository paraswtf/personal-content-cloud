"""Unit tests for app.auth.passwords, app.auth.account and app.auth.passkeys (no authenticator:
options generation, relying party derivation and the rejection paths)."""
from __future__ import annotations

import json

import pytest
from starlette.requests import Request

from app.util import now_iso


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


def make_request(host: str = "photos.example.com:8443", proto: str | None = "https", scheme: str = "http") -> Request:
    headers = [(b"host", host.encode())]
    if proto:
        headers.append((b"x-forwarded-proto", proto.encode()))
    scope = {"type": "http", "method": "POST", "path": "/", "headers": headers, "scheme": scheme,
             "server": ("127.0.0.1", 41000), "client": ("127.0.0.1", 5000), "query_string": b""}
    return Request(scope)


# ----------------------------------------------------------------- passwords
def test_hash_and_verify():
    from app.auth import passwords

    h = passwords.hash_password("secret value 1")
    assert h.startswith("$argon2id$")
    assert passwords.verify_password(h, "secret value 1")
    assert not passwords.verify_password(h, "secret value 2")
    assert not passwords.verify_password("not a hash", "x")
    assert not passwords.verify_password("", "x")
    assert passwords.needs_rehash(h) is False


def test_rate_limit_counts_and_resets(monkeypatch):
    from app.auth import passwords

    for _ in range(passwords.MAX_FAILURES):
        assert passwords.claim_login_attempt("1.2.3.4")
    assert not passwords.claim_login_attempt("1.2.3.4")
    assert passwords.claim_login_attempt("5.6.7.8")
    passwords.reset_login_failures("1.2.3.4")
    assert passwords.claim_login_attempt("1.2.3.4")


def test_rate_limit_window_expires(monkeypatch):
    from app.auth import passwords

    t = [1000.0]
    monkeypatch.setattr(passwords.time, "monotonic", lambda: t[0])
    for _ in range(passwords.MAX_FAILURES):
        passwords.claim_login_attempt("9.9.9.9")
    assert not passwords.claim_login_attempt("9.9.9.9")
    t[0] += passwords.WINDOW_S + 1
    assert passwords.claim_login_attempt("9.9.9.9")


def test_rate_limit_tracking_is_bounded(monkeypatch):
    from app.auth import passwords

    monkeypatch.setattr(passwords, "MAX_TRACKED", 50)
    for i in range(200):
        passwords.claim_login_attempt(f"10.0.0.{i}")
    assert len(passwords._failures) <= 50


# ----------------------------------------------------------------- account
def test_account_create_once_and_check(conn):
    from app.auth import account

    assert account.get_account(conn) is None
    row = account.create_account(conn, "  The Owner ", "long enough password")
    assert row["id"] == 1 and row["username"] == "The Owner" and row["username_key"] == "the owner"
    with pytest.raises(account.AccountExists):
        account.create_account(conn, "someone", "long enough password")
    assert account.check_credentials(conn, "THE OWNER", "long enough password")
    assert not account.check_credentials(conn, "other", "long enough password")
    assert not account.check_credentials(conn, "the owner", "wrong")
    assert account.check_password(conn, "long enough password")
    account.change_password(conn, "another long password")
    assert not account.check_password(conn, "long enough password")
    assert account.check_password(conn, "another long password")


def test_account_rehash_on_login(conn):
    from argon2 import PasswordHasher

    from app.auth import account
    from app.db import tx

    account.create_account(conn, "owner", "long enough password")
    weak = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash("long enough password")
    with tx(conn):
        conn.execute("UPDATE account SET password_hash = ? WHERE id = 1", (weak,))
    assert account.check_credentials(conn, "owner", "long enough password")
    stored = account.get_account(conn)["password_hash"]
    assert stored != weak and "m=8," not in stored


def test_check_without_account(conn):
    from app.auth import account

    assert not account.check_password(conn, "anything")
    assert not account.check_credentials(conn, "owner", "anything")


# ----------------------------------------------------------------- relying party
def test_rp_derived_from_request(monkeypatch):
    from app.auth import passkeys
    from app.config import settings

    monkeypatch.setattr(settings, "rp_id", None)
    monkeypatch.setattr(settings, "origin", None)
    req = make_request("Photos.Example.com:8443", "https")
    assert passkeys.rp_id(req) == "photos.example.com"
    assert passkeys.expected_origins(req) == ["https://photos.example.com:8443"]
    req = make_request("localhost:41000", None, scheme="http")
    assert passkeys.rp_id(req) == "localhost"
    assert passkeys.expected_origins(req) == ["http://localhost:41000"]
    req = make_request("[::1]:41000", "http, https")
    assert passkeys.rp_id(req) == "::1"
    assert passkeys.expected_origins(req) == ["http://[::1]:41000"]


def test_rp_from_settings(monkeypatch):
    from app.auth import passkeys
    from app.config import settings

    monkeypatch.setattr(settings, "rp_id", "cloud.example.org")
    monkeypatch.setattr(settings, "origin", "https://cloud.example.org/")
    req = make_request("127.0.0.1:41000", None)
    assert passkeys.rp_id(req) == "cloud.example.org"
    assert passkeys.expected_origins(req) == ["https://cloud.example.org"]


# ----------------------------------------------------------------- options and challenges
def test_registration_options_store_challenge(conn, monkeypatch):
    from webauthn.helpers import base64url_to_bytes

    from app.auth import account, passkeys
    from app.config import settings
    from app.db import tx

    monkeypatch.setattr(settings, "rp_id", None)
    monkeypatch.setattr(settings, "origin", None)
    account.create_account(conn, "owner", "long enough password")
    with tx(conn):
        conn.execute(
            "INSERT INTO passkeys(credential_id, public_key, sign_count, transports, name, created_at) "
            "VALUES (?, ?, 0, ?, 'Old', ?)",
            (b"\xaa\xbb", b"\x01", json.dumps(["usb", "bogus"]), now_iso()),
        )
    cid, opts = passkeys.registration_options(conn, make_request())
    json.dumps(opts)  # JSON-ready
    assert opts["rp"]["id"] == "photos.example.com"
    assert opts["user"]["name"] == "owner"
    assert base64url_to_bytes(opts["user"]["id"]) == passkeys.USER_HANDLE
    assert opts["authenticatorSelection"]["residentKey"] == "required"
    assert opts["authenticatorSelection"]["userVerification"] == "preferred"
    assert opts["timeout"] == passkeys.TIMEOUT_MS
    assert [base64url_to_bytes(c["id"]) for c in opts["excludeCredentials"]] == [b"\xaa\xbb"]
    assert opts["excludeCredentials"][0]["transports"] == ["usb"]
    row = conn.execute("SELECT * FROM challenges WHERE id = ?", (cid,)).fetchone()
    assert row["kind"] == "register" and row["used"] == 0
    assert bytes(row["challenge"]) == base64url_to_bytes(opts["challenge"])
    assert row["expires_at"] > now_iso(290) and row["expires_at"] < now_iso(310)


def test_login_and_assert_options(conn):
    from webauthn.helpers import base64url_to_bytes

    from app.auth import passkeys
    from app.db import tx

    cid, opts = passkeys.login_options(conn, make_request())
    assert opts["rpId"] == "photos.example.com" and opts["userVerification"] == "preferred"
    assert not opts.get("allowCredentials")
    row = conn.execute("SELECT * FROM challenges WHERE id = ?", (cid,)).fetchone()
    assert row["kind"] == "login" and bytes(row["challenge"]) == base64url_to_bytes(opts["challenge"])
    with tx(conn):
        conn.execute("INSERT INTO passkeys(credential_id, public_key, name, created_at) VALUES (?, ?, 'K', ?)",
                     (b"\x10\x20", b"\x01", now_iso()))
    cid2, opts2 = passkeys.assert_options(conn, make_request())
    assert conn.execute("SELECT kind FROM challenges WHERE id = ?", (cid2,)).fetchone()["kind"] == "assert"
    assert [base64url_to_bytes(c["id"]) for c in opts2["allowCredentials"]] == [b"\x10\x20"]


def _cred(cred_id: bytes) -> dict:
    from webauthn.helpers import bytes_to_base64url

    b = bytes_to_base64url(cred_id)
    return {"id": b, "rawId": b, "type": "public-key",
            "response": {"clientDataJSON": "e30", "authenticatorData": "AAAA", "signature": "AAAA"}}


def test_verify_login_rejects_unknown_credential(conn):
    from app.auth import passkeys

    req = make_request()
    cid, _ = passkeys.login_options(conn, req)
    assert passkeys.verify_login(conn, req, cid, _cred(b"\x99\x98")) is False
    # The challenge was consumed by the attempt.
    assert conn.execute("SELECT used FROM challenges WHERE id = ?", (cid,)).fetchone()["used"] == 1
    assert passkeys.verify_login(conn, req, cid, _cred(b"\x99\x98")) is False


def test_verify_login_rejects_expired_and_wrong_kind(conn):
    from app.auth import passkeys
    from app.db import tx

    req = make_request()
    with tx(conn):
        conn.execute("INSERT INTO passkeys(credential_id, public_key, name, created_at) VALUES (?, ?, 'K', ?)",
                     (b"\x01\x01", b"\x01", now_iso()))
        conn.execute("INSERT INTO challenges(id, kind, challenge, created_at, expires_at) VALUES (?, 'login', ?, ?, ?)",
                     ("expired-one", b"c" * 32, now_iso(-600), now_iso(-300)))
    assert passkeys._claim_challenge(conn, "expired-one", "login") is None
    assert passkeys.verify_login(conn, req, "expired-one", _cred(b"\x01\x01")) is False
    # A registration challenge cannot serve a login, nor a login challenge a vault assertion.
    reg_id, _ = passkeys.registration_options(conn, req)
    assert passkeys.verify_login(conn, req, reg_id, _cred(b"\x01\x01")) is False
    login_id, _ = passkeys.login_options(conn, req)
    assert passkeys.verify_assertion(conn, req, login_id, _cred(b"\x01\x01")) is False
    assert passkeys.verify_login(conn, req, "missing", _cred(b"\x01\x01")) is False
    assert passkeys.verify_login(conn, req, "x" * 65, _cred(b"\x01\x01")) is False


def test_verify_login_known_credential_bad_signature(conn):
    from app.auth import passkeys
    from app.db import tx

    req = make_request()
    with tx(conn):
        conn.execute("INSERT INTO passkeys(credential_id, public_key, name, created_at) VALUES (?, ?, 'K', ?)",
                     (b"\x02\x02", b"\x01", now_iso()))
    cid, _ = passkeys.login_options(conn, req)
    assert passkeys.verify_login(conn, req, cid, _cred(b"\x02\x02")) is False
    cid, _ = passkeys.login_options(conn, req)
    assert passkeys.verify_login(conn, req, cid, {"nothing": 1}) is False
    row = conn.execute("SELECT last_used_at FROM passkeys WHERE credential_id = ?", (b"\x02\x02",)).fetchone()
    assert row["last_used_at"] is None


def test_verify_registration_errors(conn):
    from app.auth import passkeys

    req = make_request()
    with pytest.raises(passkeys.PasskeyError) as e:
        passkeys.verify_registration(conn, req, "nope", {}, "Key")
    assert e.value.code == "challenge_invalid"
    cid, _ = passkeys.registration_options(conn, req)
    with pytest.raises(passkeys.PasskeyError) as e:
        passkeys.verify_registration(conn, req, cid, {"id": "AA", "response": {}}, "Key")
    assert e.value.code == "registration_failed"
    assert conn.execute("SELECT COUNT(*) FROM passkeys").fetchone()[0] == 0


def test_stale_challenges_are_pruned(conn):
    from app.auth import passkeys
    from app.db import tx

    with tx(conn):
        conn.execute("INSERT INTO challenges(id, kind, challenge, created_at, expires_at) VALUES ('old', 'login', ?, ?, ?)",
                     (b"c", now_iso(-900), now_iso(-600)))
    passkeys.login_options(conn, make_request())
    assert conn.execute("SELECT COUNT(*) FROM challenges WHERE id = 'old'").fetchone()[0] == 0


def test_login_challenges_are_capped(conn, monkeypatch):
    """Regression: a flood of login options keeps at most MAX_LOGIN_CHALLENGES login rows and never
    evicts a register/assert challenge."""
    from app.auth import passkeys
    from app.db import tx

    monkeypatch.setattr(passkeys, "MAX_LOGIN_CHALLENGES", 5)
    with tx(conn):
        conn.execute("INSERT INTO challenges(id, kind, challenge, created_at, expires_at) VALUES ('keep', 'assert', ?, ?, ?)",
                     (b"c", now_iso(-10), now_iso(290)))
    ids = [passkeys.login_options(conn, make_request())[0] for _ in range(8)]
    left = {r[0] for r in conn.execute("SELECT id FROM challenges WHERE kind = 'login'")}
    assert left == set(ids[-5:])
    assert conn.execute("SELECT COUNT(*) FROM challenges WHERE id = 'keep'").fetchone()[0] == 1


def test_options_limiter_is_separate():
    from app.auth import passwords

    for _ in range(passwords.OPTIONS_MAX):
        assert passwords.claim_options_request("1.2.3.4")
    assert not passwords.claim_options_request("1.2.3.4")
    assert passwords.claim_options_request("5.6.7.8")
    assert passwords.claim_login_attempt("1.2.3.4")
    passwords.reset_all()
    assert passwords.claim_options_request("1.2.3.4")
