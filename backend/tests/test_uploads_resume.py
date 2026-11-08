"""Device uploads (docs/ARCHITECTURE.md section 28): idempotent create per (session, pair_key,
resource, size) for bearer sessions, cancellation of a stale row, the session and resource recorded
on every create, and the sync check's pending and trashed answers."""
from __future__ import annotations

import hashlib
import io

import pytest
from fastapi.testclient import TestClient

from app import db, storage
from tests.conftest import ACCOUNT
from tests.helpers import jpeg_bytes, noise_image, run_jobs

H = {"X-Requested-With": "pcc"}
OCTET = {"Content-Type": "application/octet-stream"}


def _device(client, app, name: str = "Phone") -> TestClient:
    """A bearer client for a new device token (no cookie, no CSRF header)."""
    assert client.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]}, headers=H).status_code == 200
    r = client.post("/api/auth/devices", json={"name": name}, headers=H)
    assert r.status_code == 200, r.text
    return TestClient(app, headers={"Authorization": f"Bearer {r.json()['token']}"})


@pytest.fixture()
def phone(client, app):
    return _device(client, app)


def _create(c, size: int, pair_key: str | None = "asset-1", resource: str | None = "still", **extra):
    body = {"filename": "IMG_0001.HEIC", "size": size, "source": "api", "client_id": "ios", **extra}
    if pair_key is not None:
        body["pair_key"] = pair_key
    if resource is not None:
        body["resource"] = resource
    return c.post("/api/uploads", json=body)


def _row(uid: str):
    with db.connect() as conn:
        return conn.execute("SELECT * FROM uploads WHERE uid = ?", (uid,)).fetchone()


def test_idempotent_create_continues_at_received_bytes(phone):
    data = b"a" * (256 * 1024 + 1000)
    r = _create(phone, len(data))
    assert r.status_code == 201, r.text
    up = r.json()
    assert up["status"] == "uploading" and up["received_bytes"] == 0
    r = phone.put(f"/api/uploads/{up['id']}/chunk", params={"offset": 0}, content=data[:256 * 1024], headers=OCTET)
    assert r.status_code == 200 and r.json()["received_bytes"] == 256 * 1024

    # the app was killed and asks again: the same row, where it stopped, and no second staging file
    staged = sorted(p.name for p in storage.staging_path("x").parent.iterdir())
    r = _create(phone, len(data))
    assert r.status_code == 200, r.text
    again = r.json()
    assert again["id"] == up["id"] and again["received_bytes"] == 256 * 1024 and again["status"] == "uploading"
    assert sorted(p.name for p in storage.staging_path("x").parent.iterdir()) == staged

    r = phone.put(f"/api/uploads/{up['id']}/chunk", params={"offset": 256 * 1024}, content=data[256 * 1024:], headers=OCTET)
    assert r.status_code == 200
    r = phone.post(f"/api/uploads/{up['id']}/complete")
    assert r.status_code == 200 and r.json()["status"] == "queued"
    row = _row(up["id"])
    assert row["resource"] == "still" and row["pair_key"] == "asset-1" and row["session_id"] is not None

    # the other half of the Live Photo is its own upload
    other = _create(phone, 500, resource="video").json()
    assert other["id"] != up["id"]


def test_other_size_cancels_the_old_row_and_its_staging_file(phone):
    first = _create(phone, 1000).json()
    phone.put(f"/api/uploads/{first['id']}/chunk", params={"offset": 0}, content=b"x" * 100, headers=OCTET)
    assert storage.staging_path(first["id"]).exists()
    r = _create(phone, 2000)
    assert r.status_code == 201
    second = r.json()
    assert second["id"] != first["id"] and second["received_bytes"] == 0
    assert _row(first["id"])["status"] == "cancelled"
    assert not storage.staging_path(first["id"]).exists()
    assert storage.staging_path(second["id"]).exists()


def test_lost_staging_file_starts_again(phone):
    first = _create(phone, 1000).json()
    phone.put(f"/api/uploads/{first['id']}/chunk", params={"offset": 0}, content=b"x" * 100, headers=OCTET)
    storage.staging_path(first["id"]).unlink()
    second = _create(phone, 1000).json()
    assert second["id"] != first["id"] and second["received_bytes"] == 0
    assert _row(first["id"])["status"] == "cancelled"


def test_known_hash_cancels_the_unfinished_row(phone, client):
    data = jpeg_bytes(noise_image(64, 48, seed=11))
    assert client.post("/api/uploads/simple", files={"file": ("a.jpg", io.BytesIO(data))}).status_code == 201
    run_jobs()
    first = _create(phone, len(data)).json()
    r = _create(phone, len(data), sha256=hashlib.sha256(data).hexdigest())
    assert r.status_code == 201 and r.json()["status"] == "duplicate"
    assert _row(first["id"])["status"] == "cancelled"
    assert not storage.staging_path(first["id"]).exists()


def test_cookie_session_never_dedups(client):
    a = _create(client, 1000).json()
    b = _create(client, 1000).json()
    assert a["id"] != b["id"]
    assert _row(a["id"])["status"] == "uploading" and _row(b["id"])["status"] == "uploading"
    assert _row(a["id"])["session_id"] == _row(b["id"])["session_id"] is not None


def test_bearer_without_pair_key_never_dedups(phone):
    a = _create(phone, 1000, pair_key=None).json()
    b = _create(phone, 1000, pair_key="  ").json()
    c = _create(phone, 1000, pair_key=None).json()
    assert len({a["id"], b["id"], c["id"]}) == 3
    assert _row(b["id"])["pair_key"] is None


def test_resource_defaults_to_file(client, phone):
    up = _create(client, 1000, pair_key=None, resource=None).json()
    assert _row(up["id"])["resource"] == "file"
    data = jpeg_bytes(noise_image(64, 48, seed=12))
    r = phone.post("/api/uploads/simple", files={"file": ("b.jpg", io.BytesIO(data))}, data={"pair_key": " k-9 "})
    assert r.status_code == 201
    row = _row(r.json()["id"])
    assert row["resource"] == "file" and row["pair_key"] == "k-9" and row["session_id"] is not None
    assert _create(client, 10, resource="clip").status_code == 422
    assert phone.post("/api/uploads/simple", files={"file": ("c.jpg", io.BytesIO(data))},
                      data={"resource": "clip"}).status_code == 422


def test_pending_lists_only_the_callers_rows(client, app, phone):
    up = _create(phone, 1000).json()
    phone.put(f"/api/uploads/{up['id']}/chunk", params={"offset": 0}, content=b"x" * 300, headers=OCTET)
    vid = _create(phone, 400, resource="video").json()
    _create(phone, 50, pair_key="asset-2").json()

    body = phone.post("/api/uploads/check", json={"asset_ids": ["asset-1", "asset-3"]}).json()
    assert body["assets"] == {} and body["trashed"] == []
    assert body["pending"] == {"asset-1": [
        {"upload_id": up["id"], "resource": "still", "received_bytes": 300, "size": 1000},
        {"upload_id": vid["id"], "resource": "video", "received_bytes": 0, "size": 400},
    ]}

    # another device (a second session) sees none of them, and does not reuse them either
    other = _device(client, app, "iPad")
    assert other.post("/api/uploads/check", json={"asset_ids": ["asset-1"]}).json()["pending"] == {}
    assert _create(other, 1000).json()["id"] != up["id"]
    assert client.post("/api/uploads/check", json={"asset_ids": ["asset-1"]}).json()["pending"] == {}

    # finished and cancelled rows are not pending
    phone.delete(f"/api/uploads/{vid['id']}")
    left = phone.post("/api/uploads/check", json={"asset_ids": ["asset-1"]}).json()["pending"]["asset-1"]
    assert [p["upload_id"] for p in left] == [up["id"]]


def test_trashed_assets_stay_known(client, phone):
    data = jpeg_bytes(noise_image(320, 240, seed=21))
    sha = hashlib.sha256(data).hexdigest()
    r = phone.post("/api/uploads/simple", files={"file": ("IMG_0021.jpg", io.BytesIO(data))},
                   data={"pair_key": "asset-21", "resource": "still"})
    assert r.status_code == 201
    run_jobs()
    uid = client.get("/api/items").json()["items"][0]["uid"]
    ids = {"sha256": [sha], "asset_ids": ["asset-21", "asset-22"]}
    body = phone.post("/api/uploads/check", json=ids).json()
    assert body["assets"] == {"asset-21": uid} and body["trashed"] == [] and body["known"] == {sha: uid}

    assert client.delete(f"/api/items/{uid}", headers=H).status_code == 200
    body = phone.post("/api/uploads/check", json=ids).json()
    assert body["trashed"] == ["asset-21"]
    assert body["assets"] == {"asset-21": uid} and body["known"] == {sha: uid}


def test_resume_lookup_uses_the_uploads_asset_index(data_dir):
    with db.connect() as conn:
        plan = " ".join(r[3] for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT id, uid, size_bytes, received_bytes FROM uploads "
            "WHERE session_id = ? AND pair_key = ? AND resource = ? AND status = 'uploading' "
            "ORDER BY received_bytes DESC, id DESC", (1, "k", "still")))
        assert "uploads_asset" in plan
        plan = " ".join(r[3] for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT uid FROM uploads WHERE session_id = ? AND pair_key IN (?, ?) "
            "AND status = 'uploading'", (1, "a", "b")))
        assert "uploads_asset" in plan


def test_drop_staging_leaves_a_file_that_complete_is_hashing(phone):
    from app.api import uploads as uploads_api

    up = _create(phone, 100).json()
    phone.put(f"/api/uploads/{up['id']}/chunk", params={"offset": 0}, content=b"x" * 100, headers=OCTET)
    path = storage.staging_path(up["id"])
    assert path.exists()
    with uploads_api._locks_guard:
        uploads_api._hashing.add(up["id"])
    try:
        uploads_api._drop_staging([up["id"]])
        assert path.exists(), "a file being hashed must not be unlinked under the reader"
    finally:
        with uploads_api._locks_guard:
            uploads_api._hashing.discard(up["id"])
    uploads_api._drop_staging([up["id"]])
    assert not path.exists()


def test_complete_racing_a_cancel_answers_the_cancelled_row(phone, monkeypatch):
    """The device re-creates the asset with another size while complete is hashing the old upload:
    the old row is cancelled, complete answers it instead of 409, and the staged file goes."""
    from app.api import uploads as uploads_api

    data = b"y" * 300
    first = _create(phone, len(data)).json()
    r = phone.put(f"/api/uploads/{first['id']}/chunk", params={"offset": 0}, content=data, headers=OCTET)
    assert r.status_code == 200 and r.json()["received_bytes"] == len(data)
    path = storage.staging_path(first["id"])
    real = uploads_api._sha256_file
    seen: dict[str, object] = {}

    def racing(p):
        if "second" not in seen:
            seen["second"] = _create(phone, 2000).json()  # cancels the first row during the hash
            assert path.exists(), "the cancel must leave the file to the hashing complete"
        return real(p)

    monkeypatch.setattr(uploads_api, "_sha256_file", racing)
    r = phone.post(f"/api/uploads/{first['id']}/complete")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "cancelled"
    assert not path.exists()
    second = seen["second"]
    assert second["id"] != first["id"] and _row(second["id"])["status"] == "uploading"
    assert storage.staging_path(second["id"]).exists()
