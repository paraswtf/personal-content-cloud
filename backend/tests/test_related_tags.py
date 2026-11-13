"""Related items (section 10, 2026-11-08): a tag with relation 'related' says an item is about a person or a
pet without them in it. The tag route with both relations and the switch, the person's items filter and
count, the viewer's people list carrying the relation, and a search for the name finding the item."""
from __future__ import annotations

import pytest

from app.search import service
from tests.helpers import make_face, make_item, make_person
from tests.test_search_support import hit_ids, no_engine  # noqa: F401


def _relation(conn, item_id: int, person_id: int) -> str | None:
    row = conn.execute("SELECT relation FROM item_tags WHERE item_id = ? AND person_id = ?",
                       (item_id, person_id)).fetchone()
    return None if row is None else row["relation"]


def test_related_tag_switches_with_the_other_relation_and_shows_in_the_viewer(client, conn):
    item = make_item(conn)
    r = client.post(f"/api/items/{item['uid']}/tags", json={"new_person_name": "Meera", "relation": "related"})
    assert r.status_code == 200, r.text
    (ref,) = r.json()
    pid = ref["id"]
    assert (ref["kind"], ref["relation"]) == ("person", "related")
    assert _relation(conn, item["id"], pid) == "related"
    detail = client.get(f"/api/items/{item['uid']}").json()
    assert [(p["id"], p["relation"]) for p in detail["people"]] == [(pid, "related")]
    assert [(p["id"], p["relation"]) for p in detail["tags"]] == [(pid, "related")]
    # the other relation for the same person switches the tag
    r = client.post(f"/api/items/{item['uid']}/tags", json={"person_id": pid, "relation": "in"})
    assert r.status_code == 200 and r.json()[0]["relation"] == "in"
    assert _relation(conn, item["id"], pid) == "in"
    assert client.get(f"/api/items/{item['uid']}").json()["people"][0]["relation"] == "in"
    r = client.post(f"/api/items/{item['uid']}/tags", json={"person_id": pid, "relation": "related"})
    assert r.json()[0]["relation"] == "related" and _relation(conn, item["id"], pid) == "related"
    # the default relation is 'in'
    other = make_item(conn)
    r = client.post(f"/api/items/{other['uid']}/tags", json={"person_id": pid})
    assert r.json()[0]["relation"] == "in" and _relation(conn, other["id"], pid) == "in"
    assert conn.execute("SELECT COUNT(*) FROM item_tags").fetchone()[0] == 2
    assert client.post(f"/api/items/{other['uid']}/tags", json={"person_id": pid, "relation": "near"}).status_code == 422
    # removing a related tag works like removing any tag
    assert client.delete(f"/api/items/{item['uid']}/tags/{pid}").json() == []
    assert _relation(conn, item["id"], pid) is None


def test_a_box_keeps_the_person_in_the_item_whatever_the_tag_says(client, conn):
    pid = make_person(conn, "Meera")
    item = make_item(conn)
    make_face(conn, item["id"], pid)
    assert client.post(f"/api/items/{item['uid']}/tags", json={"person_id": pid, "relation": "related"}).status_code == 200
    detail = client.get(f"/api/items/{item['uid']}").json()
    assert [(p["id"], p["relation"]) for p in detail["people"]] == [(pid, "in")]
    assert [(p["id"], p["relation"]) for p in detail["tags"]] == [(pid, "related")]


def test_person_items_filter_and_counts_keep_in_as_the_default(client, conn):
    pid = make_person(conn, "Meera")
    a = make_item(conn, taken_at="2026-05-01T10:00:00.000Z")
    make_face(conn, a["id"], pid)
    b = make_item(conn, taken_at="2026-05-02T10:00:00.000Z")
    c = make_item(conn, taken_at="2026-05-03T10:00:00.000Z")
    assert client.post(f"/api/items/{b['uid']}/tags", json={"person_id": pid, "relation": "related"}).status_code == 200
    assert client.post(f"/api/items/{c['uid']}/tags", json={"person_id": pid}).status_code == 200

    def page(**params):
        r = client.get(f"/api/people/{pid}/items", params=params)
        assert r.status_code == 200, r.text
        return sorted(i["uid"] for i in r.json()["items"]), r.json()["total"]

    in_items = sorted([a["uid"], c["uid"]])
    assert page() == (in_items, 2)
    assert page(relation="in") == (in_items, 2)
    assert page(relation="related") == ([b["uid"]], 1)
    assert page(relation="all") == (sorted([a["uid"], b["uid"], c["uid"]]), 3)
    assert client.get(f"/api/people/{pid}/items", params={"relation": "nope"}).status_code == 422
    person = client.get(f"/api/people/{pid}").json()
    assert (person["item_count"], person["related_count"]) == (2, 1)
    listed = [p for p in client.get("/api/people").json()["items"] if p["id"] == pid]
    assert (listed[0]["item_count"], listed[0]["related_count"]) == (2, 1)
    # the library's person filter and the timeline keep meaning 'in'
    lib = client.get("/api/items", params={"person": pid}).json()
    assert sorted(i["uid"] for i in lib["items"]) == in_items
    assert client.get("/api/timeline/buckets", params={"person": pid}).json()["total"] == 2


@pytest.mark.usefixtures("no_engine")
def test_search_finds_the_related_item_by_the_name(client, conn):
    toy = make_item(conn, filename="toy.jpg")
    r = client.post(f"/api/items/{toy['uid']}/tags",
                    json={"new_person_name": "Chutney", "kind": "pet", "species": "cat", "relation": "related"})
    assert r.status_code == 200, r.text
    fts = conn.execute("SELECT people FROM items_fts WHERE rowid = ?", (toy["id"],)).fetchone()
    assert fts is not None and "chutney" in fts["people"].lower()
    assert hit_ids(service.search(conn, "Chutney", mode="text")) == [toy["id"]]
    # removing the tag takes the name out of the index again
    client.delete(f"/api/items/{toy['uid']}/tags/{r.json()[0]['id']}")
    assert hit_ids(service.search(conn, "Chutney", mode="text")) == []
