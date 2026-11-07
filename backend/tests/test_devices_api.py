"""Device tokens for the iOS app (bearer sessions, created with the vault open, shown once, revocable)
and the sync check that tells a device what the server already holds."""
from __future__ import annotations

import hashlib
import io

from fastapi.testclient import TestClient

from app import db
from tests.conftest import ACCOUNT
from tests.helpers import jpeg_bytes, make_item, noise_image, run_jobs

H = {"X-Requested-With": "pcc"}


def test_device_token_lifecycle(client, app):
    assert client.post("/api/auth/devices", json={"name": "My iPhone"}, headers=H).status_code == 403  # vault locked
    assert client.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]}, headers=H).status_code == 200
    r = client.post("/api/auth/devices", json={"name": "  My   iPhone "}, headers=H)
    assert r.status_code == 200
    created = r.json()
    assert created["name"] == "My iPhone" and len(created["token"]) > 20
    devices = client.get("/api/auth/devices").json()
    assert [(d["device_name"], d["current"]) for d in devices] == [("My iPhone", False)]
    assert all(s["device_name"] is None for s in client.get("/api/auth/sessions").json() if s["current"])

    # the token works as a bearer credential, without the cookie and without the CSRF header
    with db.connect() as conn:
        it = make_item(conn, filename="a.jpg")
    phone = TestClient(app, headers={"Authorization": f"Bearer {created['token']}"})
    assert phone.get("/api/items").json()["total"] == 1
    assert phone.get("/api/auth/state").json()["authenticated"] is True
    r = phone.patch(f"/api/items/{it['uid']}", json={"favorite": True})
    assert r.status_code == 200 and r.json()["favorite"] is True
    assert phone.get("/api/items", params={"vault": 1}).status_code == 403  # a device starts with the vault locked
    assert TestClient(app, headers={"Authorization": "Bearer nonsense"}).get("/api/items").status_code == 401

    assert client.delete(f"/api/auth/devices/{created['id']}", headers=H).status_code == 200
    assert phone.get("/api/items").status_code == 401
    assert client.get("/api/auth/devices").json() == []
    assert client.delete("/api/auth/devices/99999", headers=H).status_code == 404


def test_sync_check_by_hash_and_asset_id(client):
    data = jpeg_bytes(noise_image(320, 240, seed=3))
    sha = hashlib.sha256(data).hexdigest()
    r = client.post("/api/uploads/simple", files={"file": ("IMG_0010.jpg", io.BytesIO(data))}, data={"pair_key": "asset-10"}, headers=H)
    assert r.status_code == 201
    run_jobs()
    uid = client.get("/api/items").json()["items"][0]["uid"]
    r = client.post("/api/uploads/check", json={"sha256": [sha, "0" * 64], "asset_ids": ["asset-10", "asset-11"]}, headers=H)
    assert r.status_code == 200
    assert r.json() == {"known": {sha: uid}, "assets": {"asset-10": uid}}
    # a trashed item still counts as known, so the device does not send it again
    client.delete(f"/api/items/{uid}", headers=H)
    assert client.post("/api/uploads/check", json={"sha256": [sha]}, headers=H).json()["known"] == {sha: uid}
