"""Settings, account side (ios/UITests/SettingsAccountScreens.swift): what its tests may change.

- a passkey, "Setacct test key", which the tests rename and remove (a made-up credential: it never signs in);
- a browser session, "Firefox on Windows", which the tests revoke;
- a device, "Setacct spare phone", which the tests end (they never end "iPhone simulator", the tests' token);
- the vault opened for the app's session, since adding a device or removing a passkey needs it. It closes again
  after VAULT_UNLOCK_MIN (15 minutes), so the tests also accept the locked case.
"""
from __future__ import annotations

from datetime import datetime, timezone

PASSKEY = "Setacct test key"
DEVICE = "Setacct spare phone"
BROWSER = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"


def library(kit) -> dict:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with kit.tx(kit.conn):
        kit.conn.execute("INSERT INTO passkeys(credential_id, public_key, name, created_at) VALUES (?, ?, ?, ?)",
                         (b"setacct-test-key", b"\x01", PASSKEY, now))
    return {"passkey": PASSKEY}


def api(kit) -> dict:
    # a browser that signed in with the password, to revoke
    browser = kit.Client(kit.server)
    browser.call("POST", "/api/auth/login", {"username": kit.username, "password": kit.password},
                 headers={"User-Agent": BROWSER})
    # a second device, to end (the owner's browser session has the vault open: the seed opened it)
    device = kit.client.call("POST", "/api/auth/devices", {"name": DEVICE})
    # the vault open for the app's own session
    kit.Client(kit.server).call("POST", "/api/auth/vault/unlock", {"password": kit.password},
                                headers={"Authorization": f"Bearer {kit.token}"})
    return {"device": device["name"], "browser": "Firefox on Windows"}
