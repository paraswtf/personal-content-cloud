"""Albums (docs/ARCHITECTURE.md section 24): the /api/albums routes, member counts and covers, the album
filter of GET /api/items, ItemDetail.albums, the album and unalbum bulk actions, and the cascades."""
from __future__ import annotations

from app.db import connect
from app.util import now_iso
from tests.helpers import make_item


def _items(n: int = 3, **extra):
    conn = connect()
    try:
        return [make_item(conn, taken_at=f"2024-03-{10 + i:02d}T10:00:00.000Z", **extra) for i in range(n)]
    finally:
        conn.close()


def _uids(rows):
    return [r["uid"] for r in rows]


def test_create_list_get_update_delete(client):
    r = client.post("/api/albums", json={"name": "  Goa   trip ", "description": " Beach days "})
    assert r.status_code == 201, r.text
    album = r.json()
    assert album["name"] == "Goa trip" and album["description"] == "Beach days"
    assert album["count"] == 0 and album["vault_count"] == 0 and album["cover_url"] is None
    assert album["newest"] is None and album["oldest"] is None
    assert client.post("/api/albums", json={"name": "goa TRIP"}).status_code == 409
    assert client.post("/api/albums", json={"name": "   "}).status_code == 422
    other = client.post("/api/albums", json={"name": "Diwali"}).json()

    page = client.get("/api/albums").json()
    assert page["total"] == 2 and [a["name"] for a in page["items"]] == ["Diwali", "Goa trip"]  # updated desc
    assert client.get(f"/api/albums/{album['id']}").json()["id"] == album["id"]
    assert client.get("/api/albums/99999").status_code == 404

    r = client.patch(f"/api/albums/{album['id']}", json={"name": "Goa 2024", "description": "Sun"})
    assert r.status_code == 200 and r.json()["name"] == "Goa 2024" and r.json()["description"] == "Sun"
    assert client.patch(f"/api/albums/{album['id']}", json={"name": "diwali"}).status_code == 409
    assert client.patch(f"/api/albums/{other['id']}", json={"name": "Diwali"}).status_code == 200  # own name is fine
    assert client.patch("/api/albums/99999", json={"name": "x"}).status_code == 404

    assert client.delete(f"/api/albums/{album['id']}").status_code == 200
    assert client.get(f"/api/albums/{album['id']}").status_code == 404
    assert client.delete(f"/api/albums/{album['id']}").status_code == 404
    assert client.get("/api/albums").json()["total"] == 1


def test_members_counts_cover_and_dates(client, vault_client):
    live = _items(3)
    vault = _items(1, nsfw=1)
    trashed = _items(1, deleted_at=now_iso())
    uids = _uids(live) + _uids(vault) + _uids(trashed) + ["nosuchuid"]
    r = client.post("/api/albums", json={"name": "Mixed", "item_uids": uids})
    assert r.status_code == 201, r.text
    album = r.json()
    assert album["count"] == 3 and album["vault_count"] == 1
    assert album["newest"] == live[2]["taken_at"] and album["oldest"] == live[0]["taken_at"]
    assert album["cover_url"] == f"/files/thumb/{live[2]['uid']}?v=1"  # the newest live member

    # the chosen cover must be a member
    aid = album["id"]
    assert client.patch(f"/api/albums/{aid}", json={"cover_item_uid": "nosuchuid"}).status_code == 422
    stranger = _items(1)[0]
    assert client.patch(f"/api/albums/{aid}", json={"cover_item_uid": stranger["uid"]}).status_code == 422
    r = client.patch(f"/api/albums/{aid}", json={"cover_item_uid": live[0]["uid"]})
    assert r.status_code == 200 and r.json()["cover_url"] == f"/files/thumb/{live[0]['uid']}?v=1"
    # a vault member may be chosen, but the cover shown is never a vault item
    r = client.patch(f"/api/albums/{aid}", json={"cover_item_uid": vault[0]["uid"]})
    assert r.status_code == 200 and r.json()["cover_url"] == f"/files/thumb/{live[2]['uid']}?v=1"
    assert client.patch(f"/api/albums/{aid}", json={"cover_item_uid": live[0]["uid"]}).json()["cover_url"].startswith(f"/files/thumb/{live[0]['uid']}")
    # trashing the chosen cover falls back to the newest live member; restoring brings the count back
    assert client.delete(f"/api/items/{live[0]['uid']}").status_code == 200
    a = client.get(f"/api/albums/{aid}").json()
    assert a["count"] == 2 and a["cover_url"] == f"/files/thumb/{live[2]['uid']}?v=1" and a["oldest"] == live[1]["taken_at"]
    assert client.post(f"/api/items/{live[0]['uid']}/restore").status_code == 200
    a = client.get(f"/api/albums/{aid}").json()
    assert a["count"] == 3 and a["cover_url"] == f"/files/thumb/{live[0]['uid']}?v=1"
    # null clears the chosen cover
    assert client.patch(f"/api/albums/{aid}", json={"cover_item_uid": None}).json()["cover_url"] == f"/files/thumb/{live[2]['uid']}?v=1"

    # the members through the library: normal side, then the vault side
    page = client.get("/api/items", params={"album": aid}).json()
    assert page["total"] == 3 and [i["uid"] for i in page["items"]] == [live[2]["uid"], live[1]["uid"], live[0]["uid"]]
    # vault_client is the same session as client, unlocked: the vault side lists the vault member only
    assert vault_client.get("/api/items", params={"album": aid, "vault": 1}).json()["total"] == 1
    assert client.get("/api/items", params={"album": 99999}).status_code == 404
    detail = client.get(f"/api/items/{live[1]['uid']}").json()
    assert detail["albums"] == [{"id": aid, "name": "Mixed"}]
    assert client.get(f"/api/items/{stranger['uid']}").json()["albums"] == []


def test_add_and_remove_items(client):
    items = _items(4)
    trashed = _items(1, deleted_at=now_iso())[0]
    aid = client.post("/api/albums", json={"name": "Picks"}).json()["id"]
    before = client.get(f"/api/albums/{aid}").json()["updated_at"]
    uids = _uids(items[:2])
    r = client.post(f"/api/albums/{aid}/items", json={"uids": uids + uids + [trashed["uid"], "ghost"]})
    assert r.status_code == 200 and r.json() == {"added": 2, "skipped": 2}
    assert client.get(f"/api/albums/{aid}").json()["count"] == 2
    assert client.get(f"/api/albums/{aid}").json()["updated_at"] >= before
    r = client.post(f"/api/albums/{aid}/items", json={"uids": uids})
    assert r.json() == {"added": 0, "skipped": 2}
    r = client.request("DELETE", f"/api/albums/{aid}/items", json={"uids": [uids[0], "ghost"]})
    assert r.status_code == 200 and r.json() == {"removed": 1}
    assert client.get(f"/api/albums/{aid}").json()["count"] == 1
    assert client.post("/api/albums/99999/items", json={"uids": uids}).status_code == 404
    assert client.post(f"/api/albums/{aid}/items", json={"uids": []}).status_code == 422


def test_bulk_album_actions(client):
    items = _items(3)
    uids = _uids(items)
    aid = client.post("/api/albums", json={"name": "Bulk"}).json()["id"]
    r = client.post("/api/items/bulk", json={"uids": uids, "action": "album", "value": str(aid)})
    assert r.status_code == 200 and r.json()["updated"] == 3
    assert client.get(f"/api/albums/{aid}").json()["count"] == 3
    assert client.post("/api/items/bulk", json={"uids": uids, "action": "album", "value": str(aid)}).json()["updated"] == 0
    r = client.post("/api/items/bulk", json={"uids": uids[:2], "action": "unalbum", "value": str(aid)})
    assert r.json()["updated"] == 2 and client.get(f"/api/albums/{aid}").json()["count"] == 1
    assert client.post("/api/items/bulk", json={"uids": uids, "action": "album", "value": "99999"}).status_code == 404
    assert client.post("/api/items/bulk", json={"uids": uids, "action": "album", "value": "abc"}).status_code == 422


def test_cascades(client):
    items = _items(2)
    uids = _uids(items)
    aid = client.post("/api/albums", json={"name": "Gone", "item_uids": uids}).json()["id"]
    assert client.get(f"/api/albums/{aid}").json()["count"] == 2
    # purging an item removes its membership
    assert client.post("/api/items/bulk", json={"uids": uids[:1], "action": "delete"}).json()["updated"] == 1
    assert client.post("/api/items/bulk", json={"uids": uids[:1], "action": "purge"}).json()["updated"] == 1
    conn = connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM album_items WHERE album_id = ?", (aid,)).fetchone()[0] == 1
        # deleting the album removes its memberships, the items stay
        assert client.delete(f"/api/albums/{aid}").status_code == 200
        assert conn.execute("SELECT COUNT(*) FROM album_items WHERE album_id = ?", (aid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM items WHERE uid = ? AND deleted_at IS NULL", (uids[1],)).fetchone()[0] == 1
    finally:
        conn.close()


def test_list_sort_search_and_paging(client):
    for name in ["Delta", "alpha", "Charlie", "bravo", "Echo"]:
        assert client.post("/api/albums", json={"name": name}).status_code == 201
    page = client.get("/api/albums", params={"sort": "name", "limit": 2}).json()
    assert page["total"] == 5 and [a["name"] for a in page["items"]] == ["alpha", "bravo"]
    page2 = client.get("/api/albums", params={"sort": "name", "limit": 2, "cursor": page["next_cursor"]}).json()
    assert [a["name"] for a in page2["items"]] == ["Charlie", "Delta"] and page2["total"] == 0
    page3 = client.get("/api/albums", params={"sort": "name", "limit": 2, "cursor": page2["next_cursor"]}).json()
    assert [a["name"] for a in page3["items"]] == ["Echo"] and page3["next_cursor"] is None
    created = client.get("/api/albums", params={"sort": "created"}).json()
    assert [a["name"] for a in created["items"]] == ["Echo", "bravo", "Charlie", "alpha", "Delta"]
    # a change moves an album to the top of the default order
    bravo = next(a for a in created["items"] if a["name"] == "bravo")
    client.patch(f"/api/albums/{bravo['id']}", json={"description": "x"})
    assert client.get("/api/albums").json()["items"][0]["name"] == "bravo"
    found = client.get("/api/albums", params={"q": "ALP"}).json()
    assert found["total"] == 1 and found["items"][0]["name"] == "alpha"
    assert client.get("/api/albums", params={"q": "%"}).json()["total"] == 0
    assert client.get("/api/albums", params={"cursor": "nonsense"}).status_code == 400


def test_albums_need_a_session(anon_client):
    assert anon_client.get("/api/albums").status_code == 401


def test_member_subqueries_start_from_album_items(conn):
    """The AlbumOut subqueries must be planned from album_items (SEARCH ai), not from items: the other
    order walked every item per album (40 s for a page on 100 000 items, 1 000 albums)."""
    from app.api.albums import _OUT_SELECT

    plan = "\n".join(r[3] for r in conn.execute(f"EXPLAIN QUERY PLAN {_OUT_SELECT} WHERE a.id = 1").fetchall())
    scans = [line for line in plan.splitlines() if "items" in line.lower() and not line.startswith("SEARCH i USING INTEGER PRIMARY KEY")]
    assert "SEARCH ai USING" in plan, plan
    assert "SEARCH i USING INDEX items_order" not in plan and "SCAN i" not in plan, plan
    assert all("album_items" in s or "USING INTEGER PRIMARY KEY" in s for s in scans), plan


def test_vault_members_change_only_while_the_vault_is_open(client):
    # the client fixture's session is logged in but its vault is locked
    normal = _items(1)
    vault = _items(1, nsfw=1)
    r = client.post("/api/albums", json={"name": "Mixed", "item_uids": _uids(normal) + _uids(vault)})
    assert r.status_code == 201
    aid = r.json()["id"]
    assert r.json()["count"] == 1 and r.json()["vault_count"] == 0
    r = client.post(f"/api/albums/{aid}/items", json={"uids": _uids(vault)})
    assert r.json() == {"added": 0, "skipped": 1}
    # unlock: the same session now adds and removes the vault member
    from tests.conftest import ACCOUNT

    assert client.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]}).status_code == 200
    assert client.post(f"/api/albums/{aid}/items", json={"uids": _uids(vault)}).json() == {"added": 1, "skipped": 0}
    assert client.get(f"/api/albums/{aid}").json()["vault_count"] == 1
    assert client.post("/api/auth/vault/lock").status_code == 200
    assert client.request("DELETE", f"/api/albums/{aid}/items", json={"uids": _uids(vault)}).json() == {"removed": 0}
    assert client.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]}).status_code == 200
    assert client.request("DELETE", f"/api/albums/{aid}/items", json={"uids": _uids(vault)}).json() == {"removed": 1}
