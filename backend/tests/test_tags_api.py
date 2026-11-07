"""Tags the owner adds (section 19): the tag routes, item tags, bulk actions, the filters, search,
rename merges, deletion and the vault rule."""
from __future__ import annotations

from app import db
from app.db import tx
from app.documents import extract
from app.util import now_iso
from tests.helpers import make_item

H = {"X-Requested-With": "pcc"}


def _items(client, **params):
    return [it["filename"] for it in client.get("/api/items", params=params).json()["items"]]


def test_create_list_add_remove_and_filter(client):
    with db.connect() as conn:
        a = make_item(conn, filename="a.jpg")
        b = make_item(conn, filename="b.jpg")
        c = make_item(conn, filename="c.jpg")
    r = client.post("/api/tags", json={"name": "  Goa   Trip "}, headers=H)
    assert r.status_code == 200 and r.json()["name"] == "Goa Trip" and r.json()["items"] == 0
    goa = r.json()["id"]
    assert client.post("/api/tags", json={"name": "goa trip"}, headers=H).json()["id"] == goa  # same key
    assert client.post("/api/tags", json={"name": "   "}, headers=H).status_code == 422
    assert client.post("/api/tags", json={"name": "x" * 41}, headers=H).status_code == 422

    r = client.post(f"/api/items/{a['uid']}/user-tags", json={"name": "cake"}, headers=H)
    assert r.status_code == 200 and [(t["name"], t["source"]) for t in r.json()] == [("cake", "user")]
    cake = r.json()[0]["id"]
    client.post(f"/api/items/{b['uid']}/user-tags", json={"tag_id": cake}, headers=H)
    client.post(f"/api/items/{b['uid']}/user-tags", json={"tag_id": goa}, headers=H)
    assert client.post(f"/api/items/{a['uid']}/user-tags", json={}, headers=H).status_code == 422

    tags = client.get("/api/tags").json()
    assert [(t["name"], t["items"], t["auto_items"]) for t in tags] == [("cake", 2, 0), ("Goa Trip", 1, 0)]
    assert [t["name"] for t in client.get("/api/tags", params={"q": "goa"}).json()] == ["Goa Trip"]

    det = client.get(f"/api/items/{b['uid']}").json()
    assert sorted(t["name"] for t in det["user_tags"]) == ["Goa Trip", "cake"]
    assert sorted(_items(client, tag="cake")) == ["a.jpg", "b.jpg"]
    assert _items(client, tag=["cake", "goa trip"]) == ["b.jpg"]
    assert _items(client, tag="nothing") == []
    assert c["uid"]

    # search finds the tag name through the FTS labels column, and the tag filter applies to search
    r = client.get("/api/search", params={"q": "cake", "mode": "text"}).json()
    assert sorted(h["filename"] for h in r["items"]) == ["a.jpg", "b.jpg"]
    r = client.get("/api/search", params={"q": "cake", "mode": "text", "tag": "goa trip"}).json()
    assert [h["filename"] for h in r["items"]] == ["b.jpg"]

    r = client.delete(f"/api/items/{a['uid']}/user-tags/{cake}", headers=H)
    assert r.status_code == 200 and r.json() == []
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tag_rejections").fetchone()[0] == 0  # a user row leaves no rejection
        assert conn.execute("SELECT COUNT(*) FROM learning_events WHERE kind = 'tag' AND label = 1").fetchone()[0] == 3


def test_suggestions_accept_reject_and_counts(client):
    with db.connect() as conn:
        a = make_item(conn, filename="a.jpg")
        with tx(conn):
            conn.execute("INSERT INTO tags(name, name_key, created_at) VALUES ('beach', 'beach', ?)", (now_iso(),))
            tag_id = conn.execute("SELECT id FROM tags").fetchone()[0]
            conn.execute("INSERT INTO item_user_tags(item_id, tag_id, source, score, created_at) VALUES (?, ?, 'auto', 0.91, ?)",
                         (a["id"], tag_id, now_iso()))
    det = client.get(f"/api/items/{a['uid']}").json()
    assert det["user_tags"] == [{"id": tag_id, "name": "beach", "source": "auto", "score": 0.91}]
    assert client.get("/api/tags").json()[0]["auto_items"] == 1
    # removing a suggestion is a counter-example
    r = client.delete(f"/api/items/{a['uid']}/user-tags/{tag_id}", headers=H)
    assert r.status_code == 200 and r.json() == []
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tag_rejections WHERE item_id = ? AND tag_id = ?", (a["id"], tag_id)).fetchone()[0] == 1
        assert conn.execute("SELECT label FROM learning_events WHERE kind = 'tag'").fetchone()[0] == 0
    # accepting a suggestion (adding it by hand) clears the rejection and makes a user row
    with db.connect() as conn, tx(conn):
        conn.execute("INSERT INTO item_user_tags(item_id, tag_id, source, score, created_at) VALUES (?, ?, 'auto', 0.9, ?)",
                     (a["id"], tag_id, now_iso()))
    r = client.post(f"/api/items/{a['uid']}/user-tags", json={"tag_id": tag_id}, headers=H)
    assert r.json()[0]["source"] == "user" and r.json()[0]["score"] is None
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tag_rejections").fetchone()[0] == 0


def test_bulk_tag_untag_rename_merge_and_delete(client):
    with db.connect() as conn:
        a = make_item(conn, filename="a.jpg")
        b = make_item(conn, filename="b.jpg")
    r = client.post("/api/items/bulk", json={"uids": [a["uid"], b["uid"]], "action": "tag", "value": "Cakes"}, headers=H)
    assert r.status_code == 200 and r.json()["updated"] == 2
    assert sorted(_items(client, tag="cakes")) == ["a.jpg", "b.jpg"]
    r = client.post("/api/items/bulk", json={"uids": [a["uid"]], "action": "untag", "value": "cakes"}, headers=H)
    assert r.json()["updated"] == 1 and _items(client, tag="cakes") == ["b.jpg"]
    assert client.post("/api/items/bulk", json={"uids": [a["uid"]], "action": "tag", "value": ""}, headers=H).status_code == 422

    cakes = next(t for t in client.get("/api/tags").json() if t["name"] == "Cakes")
    client.post(f"/api/items/{a['uid']}/user-tags", json={"name": "cake"}, headers=H)
    cake = next(t for t in client.get("/api/tags").json() if t["name"] == "cake")
    # renaming Cakes onto cake merges them: both items carry cake, Cakes is gone
    r = client.patch(f"/api/tags/{cakes['id']}", json={"name": "cake"}, headers=H)
    assert r.status_code == 200 and r.json()["id"] == cake["id"] and r.json()["items"] == 2
    assert [t["name"] for t in client.get("/api/tags").json()] == ["cake"]
    assert sorted(_items(client, tag="cake")) == ["a.jpg", "b.jpg"]
    r = client.patch(f"/api/tags/{cake['id']}", json={"name": "Cake!"}, headers=H)
    assert r.json()["name"] == "Cake!" and sorted(_items(client, tag="cake!")) == ["a.jpg", "b.jpg"]
    assert client.delete(f"/api/tags/{cake['id']}", headers=H).status_code == 200
    assert client.get("/api/tags").json() == [] and client.get(f"/api/items/{a['uid']}").json()["user_tags"] == []
    assert client.delete("/api/tags/999", headers=H).status_code == 404


def test_vault_items_are_reachable_only_with_the_vault_open(client):
    from tests.conftest import ACCOUNT

    with db.connect() as conn:
        private = make_item(conn, filename="p.jpg", nsfw=1, category="nsfw")
    assert client.post("/api/auth/vault/lock", headers=H).status_code == 200
    assert client.post(f"/api/items/{private['uid']}/user-tags", json={"name": "secret"}, headers=H).status_code == 404
    assert client.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]}, headers=H).status_code == 200
    r = client.post(f"/api/items/{private['uid']}/user-tags", json={"name": "secret"}, headers=H)
    assert r.status_code == 200
    assert client.get("/api/tags").json()[0]["items"] == 1
    assert [it["filename"] for it in client.get("/api/items", params={"tag": "secret", "vault": 1}).json()["items"]] == ["p.jpg"]
    assert client.post("/api/auth/vault/lock", headers=H).status_code == 200
    assert client.get("/api/tags").json()[0]["items"] == 0          # counts hide the vault while locked
    assert _items(client, tag="secret") == []
