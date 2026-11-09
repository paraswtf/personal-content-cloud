"""Passkeys on custom domains (section 4): ORIGIN may list several origins (the dev server's {origins}: its own URL,
then its custom domains). A ceremony from any of them verifies, through WebAuthn related origins: the RP ID stays the
server's own host, so a passkey registered there signs in on the custom domain. An origin not listed fails. The
first origin is the server's own address (GET /api/auth/rp). The ceremonies run against a small software
authenticator (P-256, attestation 'none')."""
from __future__ import annotations

import hashlib
import json

import cbor2
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from webauthn.helpers import bytes_to_base64url

from tests.test_auth_units import make_request

HOME = "https://web.pcc.example.workspaces.test"
CUSTOM = "https://cloud.example.com"
RP = "web.pcc.example.workspaces.test"


class SoftKey:
    """A passkey: an EC P-256 key pair and a credential id, signing as an authenticator would."""

    def __init__(self) -> None:
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.cred_id = b"soft-passkey-0001"
        self.count = 0

    def _cose(self) -> bytes:
        nums = self.key.public_key().public_numbers()
        return cbor2.dumps({1: 2, 3: -7, -1: 1, -2: nums.x.to_bytes(32, "big"), -3: nums.y.to_bytes(32, "big")})

    @staticmethod
    def _client(kind: str, challenge: str, origin: str) -> bytes:
        return json.dumps({"type": kind, "challenge": challenge, "origin": origin, "crossOrigin": False}).encode()

    def register(self, options: dict, origin: str) -> dict:
        rp_hash = hashlib.sha256(options["rp"]["id"].encode()).digest()
        auth = (rp_hash + bytes([0x45]) + (0).to_bytes(4, "big") + b"\x00" * 16
                + len(self.cred_id).to_bytes(2, "big") + self.cred_id + self._cose())
        att = cbor2.dumps({"fmt": "none", "attStmt": {}, "authData": auth})
        cid = bytes_to_base64url(self.cred_id)
        return {"id": cid, "rawId": cid, "type": "public-key", "clientExtensionResults": {},
                "response": {"clientDataJSON": bytes_to_base64url(self._client("webauthn.create", options["challenge"], origin)),
                             "attestationObject": bytes_to_base64url(att)}}

    def sign(self, options: dict, origin: str, user_handle: bytes) -> dict:
        self.count += 1
        auth = hashlib.sha256(options["rpId"].encode()).digest() + bytes([0x05]) + self.count.to_bytes(4, "big")
        client = self._client("webauthn.get", options["challenge"], origin)
        sig = self.key.sign(auth + hashlib.sha256(client).digest(), ec.ECDSA(hashes.SHA256()))
        cid = bytes_to_base64url(self.cred_id)
        return {"id": cid, "rawId": cid, "type": "public-key", "clientExtensionResults": {},
                "response": {"clientDataJSON": bytes_to_base64url(client), "authenticatorData": bytes_to_base64url(auth),
                             "signature": bytes_to_base64url(sig), "userHandle": bytes_to_base64url(user_handle)}}


@pytest.fixture
def two_origins(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "rp_id", RP)
    monkeypatch.setattr(settings, "origin", f" {HOME}/, {CUSTOM} ,{HOME}, {CUSTOM.upper()}")


def test_expected_origins_lists_every_origin_the_first_is_home(two_origins):
    from app.auth import passkeys

    req = make_request("127.0.0.1:41006", None)
    assert passkeys.configured_origins() == [HOME, CUSTOM]          # trimmed, lowercase, no slash, no duplicate
    assert passkeys.expected_origins(req) == [HOME, CUSTOM]
    assert passkeys.home_origin(req) == HOME
    assert passkeys.rp_id(req) == RP                                 # the RP ID does not follow the origins


def test_one_origin_and_none_behave_as_before(monkeypatch):
    from app.auth import passkeys
    from app.config import settings

    monkeypatch.setattr(settings, "rp_id", None)
    monkeypatch.setattr(settings, "origin", f"{HOME}/")
    req = make_request("photos.example.com:8443", "https")
    assert passkeys.expected_origins(req) == [HOME] and passkeys.home_origin(req) == HOME
    monkeypatch.setattr(settings, "origin", None)
    assert passkeys.expected_origins(req) == ["https://photos.example.com:8443"]
    assert passkeys.home_origin(req) == "https://photos.example.com:8443" and passkeys.configured_origins() == []


def test_a_passkey_made_at_home_signs_in_on_the_custom_domain(conn, two_origins):
    from app.auth import account, passkeys

    account.create_account(conn, "owner", "long enough password")
    req = make_request("127.0.0.1:41006", None)
    soft = SoftKey()
    cid, opts = passkeys.registration_options(conn, req)
    assert opts["rp"]["id"] == RP
    row = passkeys.verify_registration(conn, req, cid, soft.register(opts, HOME), "Laptop")
    assert bytes(row["credential_id"]) == soft.cred_id
    # signing in from the custom domain (the browser checks /.well-known/webauthn on the RP ID; the server, the origin)
    cid, opts = passkeys.login_options(conn, req)
    assert passkeys.verify_login(conn, req, cid, soft.sign(opts, CUSTOM, passkeys.USER_HANDLE)) is True
    # and the vault's re-authentication too
    cid, opts = passkeys.assert_options(conn, req)
    assert passkeys.verify_assertion(conn, req, cid, soft.sign(opts, CUSTOM, passkeys.USER_HANDLE)) is True
    # an origin nobody listed does not
    cid, opts = passkeys.login_options(conn, req)
    assert passkeys.verify_login(conn, req, cid, soft.sign(opts, "https://evil.example.net", passkeys.USER_HANDLE)) is False


def test_registering_from_the_custom_domain_also_works(conn, two_origins):
    from app.auth import account, passkeys

    account.create_account(conn, "owner", "long enough password")
    req = make_request("127.0.0.1:41006", None)
    soft = SoftKey()
    cid, opts = passkeys.registration_options(conn, req)
    passkeys.verify_registration(conn, req, cid, soft.register(opts, CUSTOM), "Phone")
    cid, opts = passkeys.login_options(conn, req)
    assert passkeys.verify_login(conn, req, cid, soft.sign(opts, HOME, passkeys.USER_HANDLE)) is True
    # a registration from an unlisted origin is refused for its origin, not for anything else
    other = SoftKey()
    other.cred_id = b"soft-passkey-0002"
    cid, opts = passkeys.registration_options(conn, req)
    with pytest.raises(passkeys.PasskeyError) as err:
        passkeys.verify_registration(conn, req, cid, other.register(opts, "https://evil.example.net"), "X")
    assert err.value.code == "registration_failed" and passkeys.count(conn) == 1
    cid, opts = passkeys.registration_options(conn, req)
    passkeys.verify_registration(conn, req, cid, other.register(opts, HOME), "Tablet")
    assert passkeys.count(conn) == 2


def test_the_rp_route_names_the_home_address(anon_client, two_origins):
    r = anon_client.get("/api/auth/rp")      # before signing in: the sign-in page needs it
    assert r.status_code == 200 and r.json() == {"rp_id": RP, "home_url": HOME}


def test_the_rp_route_without_settings_answers_for_the_request(anon_client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "rp_id", None)
    monkeypatch.setattr(settings, "origin", None)
    r = anon_client.get("/api/auth/rp")
    assert r.status_code == 200 and r.json()["rp_id"] == "testserver"
    assert r.json()["home_url"] == "http://testserver"
