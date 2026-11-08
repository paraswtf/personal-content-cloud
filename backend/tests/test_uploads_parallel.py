"""Several uploads from one device at once (docs/ARCHITECTURE.md section 28, parallel transfers): the iOS
sync sends up to four assets together and the web queue three. Chunks of different uploads interleave from
several connections; every file must arrive whole, and the sync check must list every asset."""
from __future__ import annotations

import hashlib
import threading

from fastapi.testclient import TestClient

from tests.conftest import ACCOUNT
from tests.helpers import jpeg_bytes, noise_image, run_jobs

H = {"X-Requested-With": "pcc"}
OCTET = {"Content-Type": "application/octet-stream"}


def test_four_assets_at_once_from_one_device(client, app, conn):
    assert client.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]}, headers=H).status_code == 200
    token = client.post("/api/auth/devices", json={"name": "Phone"}, headers=H).json()["token"]
    auth = {"Authorization": f"Bearer {token}"}
    phone = TestClient(app, headers=auth)

    files = {f"asset-{k}": jpeg_bytes(noise_image(900, 700, seed=70 + k), quality=97) for k in range(4)}
    uploads = {}
    for asset, data in files.items():
        r = phone.post("/api/uploads", json={"filename": f"{asset}.JPG", "size": len(data), "source": "api",
                                              "client_id": "ios", "pair_key": asset, "resource": "still",
                                              "sha256": hashlib.sha256(data).hexdigest()})
        assert r.status_code == 201, r.text
        uploads[asset] = r.json()
    chunk = uploads["asset-0"]["chunk_size"]
    assert all(len(d) > 2 * chunk for d in files.values())  # several chunks each, so they interleave

    errors: list[str] = []
    start = threading.Barrier(len(files))

    def send(asset: str) -> None:
        c = TestClient(app, headers=auth)   # one connection per asset, like the background session
        data, up = files[asset], uploads[asset]
        start.wait()
        offset = 0
        while offset < len(data):
            r = c.put(f"/api/uploads/{up['id']}/chunk", params={"offset": offset}, content=data[offset:offset + chunk],
                      headers=OCTET)
            if r.status_code != 200:
                errors.append(f"{asset}: {r.status_code} {r.text}")
                return
            offset = r.json()["received_bytes"]
        r = c.post(f"/api/uploads/{up['id']}/complete")
        if r.status_code != 200 or r.json()["status"] not in ("queued", "stored"):
            errors.append(f"{asset}: complete {r.status_code} {r.text}")

    threads = [threading.Thread(target=send, args=(a,)) for a in files]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors, errors
    run_jobs()

    rows = {r["original_filename"]: r for r in conn.execute("SELECT original_filename, sha256 FROM items")}
    assert sorted(rows) == sorted(f"{a}.JPG" for a in files)
    for asset, data in files.items():
        assert rows[f"{asset}.JPG"]["sha256"] == hashlib.sha256(data).hexdigest()   # every byte in place
    check = phone.post("/api/uploads/check", json={"hashes": [], "asset_ids": list(files)})
    assert check.status_code == 200, check.text
    out = check.json()
    assert sorted(out["assets"]) == sorted(files) and not any(out["pending"].values())
