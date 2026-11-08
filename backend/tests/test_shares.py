"""Share links (docs/ARCHITECTURE.md section 26): owner routes, the public page, passwords, expiry,
revocation, downloads, and the rule that vault and trashed items never leave through a link."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.auth import passwords
from app.db import tx
from app.util import now_iso
from tests.helpers import make_item

CSRF = {"X-Requested-With": "pcc"}


@pytest.fixture()
def anon_client(app):
    """A visitor without a session. No `with`: the lifespan (and its instance lock) already runs for
    the owner's client; a second TestClient on the same app must not start it again."""
    return TestClient(app)


def _album(conn, name: str, item_ids: list[int]) -> int:
    now = now_iso()
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO albums(name, name_key, description, created_at, updated_at) VALUES (?, ?, '', ?, ?)",
            (name, name.casefold(), now, now),
        )
        album_id = int(cur.lastrowid)
        for item_id in item_ids:
            conn.execute("INSERT INTO album_items(album_id, item_id, added_at) VALUES (?, ?, ?)", (album_id, item_id, now))
    return album_id


@pytest.fixture()
def photos(conn):
    """Three live photos in time order (newest first in the library), one vault photo, one trashed photo."""
    a = make_item(conn, taken_at="2024-03-01T10:00:00.000Z", filename="a.jpg")
    b = make_item(conn, taken_at="2024-03-02T10:00:00.000Z", filename="b.jpg")
    c = make_item(conn, taken_at="2024-03-03T10:00:00.000Z", filename="c.jpg")
    private = make_item(conn, nsfw=1, taken_at="2024-03-04T10:00:00.000Z", filename="p.jpg")
    gone = make_item(conn, taken_at="2024-03-05T10:00:00.000Z", filename="g.jpg", deleted_at=now_iso())
    return {"a": a, "b": b, "c": c, "private": private, "gone": gone}


def _create(client, **body):
    return client.post("/api/shares", json={"kind": "items", **body})


def test_items_share_public_page_and_files(client, anon_client, photos):
    r = _create(client, item_uids=[photos["a"]["uid"], photos["c"]["uid"], photos["b"]["uid"]], title="Three")
    assert r.status_code == 201, r.text
    share = r.json()
    assert share["kind"] == "items" and share["count"] == 3 and share["path"] == f"/s/{share['token']}"
    assert share["has_password"] is False and share["allow_download"] is True and share["view_count"] == 0
    assert len(share["token"]) >= 20

    pub = anon_client.get(f"/api/s/{share['token']}")
    assert pub.status_code == 200, pub.text
    body = pub.json()
    assert body["title"] == "Three" and body["count"] == 3 and body["allow_download"] is True
    # library order: newest first, whatever order the uids came in
    assert [i["uid"] for i in body["items"]] == [photos["c"]["uid"], photos["b"]["uid"], photos["a"]["uid"]]
    item = body["items"][0]
    assert item["filename"] == "c.jpg" and item["kind"] == "image" and item["width"] == 640
    assert item["urls"]["thumb"] == f"/files/s/{share['token']}/thumb/{photos['c']['uid']}?v=1"
    assert item["urls"]["download"].endswith("&download=1") and item["urls"]["video"] is None

    for what, mime in (("thumb", "image/webp"), ("preview", "image/webp"), ("original", "image/jpeg")):
        f = anon_client.get(f"/files/s/{share['token']}/{what}/{photos['c']['uid']}")
        assert f.status_code == 200, (what, f.status_code)
        assert f.headers["content-type"].startswith(mime)
        assert f.headers["cache-control"] == "private, max-age=3600"
        assert "sandbox" in f.headers["content-security-policy"]
    head = anon_client.head(f"/files/s/{share['token']}/thumb/{photos['c']['uid']}")
    assert head.status_code == 200 and head.content == b""
    dl = anon_client.get(f"/files/s/{share['token']}/original/{photos['c']['uid']}?download=1")
    assert dl.status_code == 200 and dl.headers["content-disposition"].startswith('attachment; filename="c.jpg"')
    # the view counter moved
    assert client.get(f"/api/shares/{share['id']}").json()["view_count"] == 1
    assert client.get("/api/shares").json()[0]["id"] == share["id"]


def test_vault_items_are_refused_and_albums_skip_them(client, anon_client, conn, photos):
    r = _create(client, item_uids=[photos["a"]["uid"], photos["private"]["uid"]])
    assert r.status_code == 422 and r.json()["detail"] == "vault_item"
    album_id = _album(conn, "Trip", [int(photos["a"]["id"]), int(photos["private"]["id"]), int(photos["gone"]["id"])])
    r = client.post("/api/shares", json={"kind": "album", "album_id": album_id, "title": "Trip"})
    assert r.status_code == 201, r.text
    share = r.json()
    assert share["album"] == {"id": album_id, "name": "Trip"} and share["count"] == 1
    pub = anon_client.get(f"/api/s/{share['token']}").json()
    assert [i["uid"] for i in pub["items"]] == [photos["a"]["uid"]]
    # the vault photo is not reachable through the link even by uid
    assert anon_client.get(f"/files/s/{share['token']}/thumb/{photos['private']['uid']}").status_code == 404
    assert anon_client.get(f"/files/s/{share['token']}/thumb/{photos['gone']['uid']}").status_code == 404
    assert client.post("/api/shares", json={"kind": "album", "album_id": 999}).status_code == 404
    assert _create(client, item_uids=["nosuchuid"]).status_code == 404
    assert _create(client, item_uids=[]).status_code == 422


def test_trashed_member_drops_out_and_comes_back(client, anon_client, photos):
    share = _create(client, item_uids=[photos["a"]["uid"], photos["b"]["uid"]]).json()
    assert anon_client.get(f"/api/s/{share['token']}").json()["count"] == 2
    assert client.delete(f"/api/items/{photos['b']['uid']}").status_code == 200
    pub = anon_client.get(f"/api/s/{share['token']}").json()
    assert pub["count"] == 1 and [i["uid"] for i in pub["items"]] == [photos["a"]["uid"]]
    assert anon_client.get(f"/files/s/{share['token']}/thumb/{photos['b']['uid']}").status_code == 404
    assert client.post(f"/api/items/{photos['b']['uid']}/restore").status_code == 200
    assert anon_client.get(f"/api/s/{share['token']}").json()["count"] == 2
    assert client.get(f"/api/shares/{share['id']}").json()["count"] == 2


def test_password_flow_cookie_and_limit(app, client, anon_client, photos):
    share = _create(client, item_uids=[photos["a"]["uid"]], password="open sesame").json()
    assert share["has_password"] is True
    token = share["token"]
    r = anon_client.get(f"/api/s/{token}")
    assert r.status_code == 403 and r.json()["detail"] == "share_password_required"
    assert anon_client.get(f"/files/s/{token}/thumb/{photos['a']['uid']}").status_code == 403
    # the unlock is a browser POST: the CSRF header is required
    assert anon_client.post(f"/api/s/{token}/unlock", json={"password": "open sesame"}).status_code == 403
    r = anon_client.post(f"/api/s/{token}/unlock", json={"password": "wrong"}, headers=CSRF)
    assert r.status_code == 403 and r.json()["detail"] == "invalid_password"
    for _ in range(9):
        assert anon_client.post(f"/api/s/{token}/unlock", json={"password": "wrong"}, headers=CSRF).status_code == 403
    r = anon_client.post(f"/api/s/{token}/unlock", json={"password": "open sesame"}, headers=CSRF)
    assert r.status_code == 429 and r.json()["detail"] == "too_many_attempts"
    passwords.reset_all()
    r = anon_client.post(f"/api/s/{token}/unlock", json={"password": "open sesame"}, headers=CSRF)
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 1
    cookie = r.headers.get("set-cookie", "")
    assert cookie.startswith(f"pcc_share_{token}=") and "HttpOnly" in cookie and "Max-Age=604800" in cookie
    # the cookie opens the page and the files for this client, and nobody else
    assert anon_client.get(f"/api/s/{token}").status_code == 200
    assert anon_client.get(f"/files/s/{token}/thumb/{photos['a']['uid']}").status_code == 200
    stranger = TestClient(app)
    assert stranger.get(f"/api/s/{token}").status_code == 403
    # a forged cookie is refused
    forger = TestClient(app)
    forger.cookies.set(f"pcc_share_{token}", token + ".forged.signature")
    assert forger.get(f"/api/s/{token}").status_code == 403
    # the owner clears the password: the page opens without it
    assert client.patch(f"/api/shares/{share['id']}", json={"password": ""}).json()["has_password"] is False
    assert TestClient(app).get(f"/api/s/{token}").status_code == 200


def test_expiry_revocation_and_download_switch(client, anon_client, conn, photos):
    share = _create(client, item_uids=[photos["a"]["uid"]], expires_days=7, allow_download=False).json()
    assert share["expires_at"] is not None
    expected = datetime.now(timezone.utc) + timedelta(days=7)
    assert abs((datetime.fromisoformat(share["expires_at"].replace("Z", "+00:00")) - expected).total_seconds()) < 120
    token = share["token"]
    pub = anon_client.get(f"/api/s/{token}").json()
    assert pub["allow_download"] is False and pub["items"][0]["urls"]["download"] is None
    assert anon_client.get(f"/files/s/{token}/original/{photos['a']['uid']}").status_code == 404
    assert anon_client.get(f"/files/s/{token}/thumb/{photos['a']['uid']}").status_code == 200
    # expiry set to never, then to the past
    assert client.patch(f"/api/shares/{share['id']}", json={"expires_days": None}).json()["expires_at"] is None
    with tx(conn):
        conn.execute("UPDATE shares SET expires_at = ? WHERE id = ?", (now_iso(-60), share["id"]))
    r = anon_client.get(f"/api/s/{token}")
    assert r.status_code == 410 and r.json()["detail"] == "share_expired"
    assert anon_client.get(f"/files/s/{token}/thumb/{photos['a']['uid']}").status_code == 410
    # revoked: gone for the public, listed for the owner with revoked_at
    assert client.delete(f"/api/shares/{share['id']}").status_code == 200
    r = anon_client.get(f"/api/s/{token}")
    assert r.status_code == 404 and r.json()["detail"] == "share_not_found"
    assert anon_client.get(f"/files/s/{token}/thumb/{photos['a']['uid']}").status_code == 404
    assert anon_client.post(f"/api/s/{token}/unlock", json={"password": "x"}, headers=CSRF).status_code == 404
    listed = client.get("/api/shares").json()
    assert listed[0]["id"] == share["id"] and listed[0]["revoked_at"] is not None
    assert client.delete("/api/shares/999").status_code == 404
    assert client.patch("/api/shares/999", json={"title": "x"}).status_code == 404


def test_patch_title_and_download(client, anon_client, photos):
    share = _create(client, item_uids=[photos["a"]["uid"]], title="  Old   title ").json()
    assert share["title"] == "Old title"
    r = client.patch(f"/api/shares/{share['id']}", json={"title": "New", "allow_download": False})
    assert r.status_code == 200 and r.json()["title"] == "New" and r.json()["allow_download"] is False
    # fields left out stay as they are
    r = client.patch(f"/api/shares/{share['id']}", json={"expires_days": 1})
    assert r.json()["title"] == "New" and r.json()["allow_download"] is False and r.json()["expires_at"]
    assert anon_client.get(f"/api/s/{share['token']}").json()["title"] == "New"


def test_public_routes_never_answer_401_and_owner_routes_need_a_session(anon_client, photos):
    assert anon_client.get("/api/s/nosuchtoken").status_code == 404
    assert anon_client.get(f"/files/s/nosuchtoken/thumb/{photos['a']['uid']}").status_code == 404
    assert anon_client.post("/api/s/nosuchtoken/unlock", json={"password": "x"}, headers=CSRF).status_code == 404
    assert anon_client.get("/api/shares").status_code == 401
    assert anon_client.post("/api/shares", json={"kind": "items", "item_uids": [photos["a"]["uid"]]}, headers=CSRF).status_code == 401


def test_video_member_serves_the_web_copy_or_the_original(client, anon_client, conn):
    clip = make_item(conn, kind="video", ext=".mp4", mime="video/mp4", data=b"\x00" * 300, filename="clip.mp4",
                     taken_at="2024-01-01T00:00:00.000Z", duration_s=3.5)
    share = _create(client, item_uids=[clip["uid"]]).json()
    pub = anon_client.get(f"/api/s/{share['token']}").json()
    assert pub["items"][0]["urls"]["video"] == f"/files/s/{share['token']}/video/{clip['uid']}?v=1"
    assert pub["items"][0]["duration_s"] == 3.5
    f = anon_client.get(f"/files/s/{share['token']}/video/{clip['uid']}")
    assert f.status_code == 200 and f.headers["content-type"].startswith("video/mp4")
    # a photo has no video route
    photo = make_item(conn)
    share2 = _create(client, item_uids=[photo["uid"]]).json()
    assert anon_client.get(f"/files/s/{share2['token']}/video/{photo['uid']}").status_code == 404


def test_album_share_hides_misclicks_like_the_album_page(client, anon_client, conn, photos):
    blurry = make_item(conn, misclick=1, taken_at="2024-03-06T10:00:00.000Z", filename="blurry.jpg")
    album_id = _album(conn, "Trip", [int(photos["a"]["id"]), int(blurry["id"])])
    r = client.post("/api/shares", json={"kind": "album", "album_id": album_id})
    assert r.status_code == 201, r.text
    assert r.json()["count"] == 1
    pub = anon_client.get(f"/api/s/{r.json()['token']}").json()
    assert [i["filename"] for i in pub["items"]] == ["a.jpg"]
    assert anon_client.get(f"/files/s/{r.json()['token']}/thumb/{blurry['uid']}").status_code == 404
    # a hand-picked set keeps what the owner picked, mis-click or not
    r2 = _create(client, item_uids=[blurry["uid"]])
    assert r2.status_code == 201 and r2.json()["count"] == 1


def test_items_of_kind_other_are_left_out_of_shares(client, anon_client, conn, photos):
    blob = make_item(conn, kind="other", ext=".bin", mime="application/octet-stream", data=b"x" * 64, filename="blob.bin")
    r = _create(client, item_uids=[photos["a"]["uid"], blob["uid"]])
    assert r.status_code == 201, r.text
    assert r.json()["count"] == 1
    pub = anon_client.get(f"/api/s/{r.json()['token']}").json()
    assert [i["filename"] for i in pub["items"]] == ["a.jpg"]
    assert anon_client.get(f"/files/s/{r.json()['token']}/thumb/{blob['uid']}").status_code == 404
    only = _create(client, item_uids=[blob["uid"]])
    assert only.status_code == 422 and only.json()["detail"] == "no_items"
    album_id = _album(conn, "Files", [int(photos["b"]["id"]), int(blob["id"])])
    r = client.post("/api/shares", json={"kind": "album", "album_id": album_id})
    assert r.status_code == 201 and r.json()["count"] == 1
