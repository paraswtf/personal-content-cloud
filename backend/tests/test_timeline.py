"""Timeline buckets (docs/ARCHITECTURE.md section 23): local calendar months, the library filters,
the vault, hidden rows, bucket paging and the key check."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.api.timeline import bucketize
from app.db import tx
from app.util import local_month_start_utc, now_iso
from tests.conftest import ACCOUNT, CSRF
from tests.helpers import make_face, make_item, make_person

# Asia/Kolkata (+05:30) is the default LOCAL_TIMEZONE: 18:30Z is local midnight.
MARCH_LAST = "2024-03-31T18:29:59.000Z"   # 23:59:59 on 31 March locally
APRIL_FIRST = "2024-03-31T18:30:00.000Z"  # 00:00:00 on 1 April locally
APRIL_MID = "2024-04-15T10:00:00.000Z"
JAN_FIRST = "2023-12-31T18:30:00.000Z"    # 00:00:00 on 1 January 2024 locally
DEC_LAST = "2023-12-31T18:29:59.000Z"


def test_month_starts_are_local_instants_in_the_stored_format():
    assert local_month_start_utc(2024, 4, "Asia/Kolkata") == "2024-03-31T18:30:00.000Z"
    assert local_month_start_utc(2024, 13, "Asia/Kolkata") == "2024-12-31T18:30:00.000Z"
    assert local_month_start_utc(2024, 0, "Asia/Kolkata") == "2023-11-30T18:30:00.000Z"
    assert local_month_start_utc(2024, 1, "UTC") == "2024-01-01T00:00:00.000Z"
    assert local_month_start_utc(2024, 1, "No/Such_Zone") == "2024-01-01T00:00:00.000Z"


def test_bucketize_walks_month_boundaries_and_gaps():
    keys = sorted([APRIL_MID, APRIL_FIRST, MARCH_LAST, JAN_FIRST, DEC_LAST, "2021-06-10T05:00:00.000Z"], reverse=True)
    out = bucketize(keys, "Asia/Kolkata")
    assert [(b.key, b.count) for b in out] == [("2024-04", 2), ("2024-03", 1), ("2024-01", 1), ("2023-12", 1), ("2021-06", 1)]
    assert out[0].newest == APRIL_MID and out[0].oldest == APRIL_FIRST
    assert bucketize([], "Asia/Kolkata") == []
    # keys without milliseconds and the exact boundary instant land in the new month
    assert [b.key for b in bucketize(["2024-03-31T18:30:00Z"], "Asia/Kolkata")] == ["2024-04"]
    assert [b.key for b in bucketize(["2024-03-31T18:29:59Z"], "Asia/Kolkata")] == ["2024-03"]


def _seed(conn):
    a = make_item(conn, taken_at=APRIL_MID, category="places", favorite=1)
    b = make_item(conn, taken_at=APRIL_FIRST, category="personal")
    c = make_item(conn, taken_at=MARCH_LAST, category="places")
    d = make_item(conn, taken_at=JAN_FIRST)
    e = make_item(conn, taken_at=DEC_LAST)
    return a, b, c, d, e


def test_buckets_group_by_local_month_newest_first(client, conn):
    _seed(conn)
    r = client.get("/api/timeline/buckets")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 5
    assert [(b["key"], b["count"]) for b in body["buckets"]] == [("2024-04", 2), ("2024-03", 1), ("2024-01", 1), ("2023-12", 1)]
    assert body["buckets"][0]["newest"] == APRIL_MID and body["buckets"][0]["oldest"] == APRIL_FIRST


def test_buckets_take_the_library_filters(client, conn):
    a, b, c, d, e = _seed(conn)
    assert [(x["key"], x["count"]) for x in client.get("/api/timeline/buckets", params={"category": "places"}).json()["buckets"]] == [("2024-04", 1), ("2024-03", 1)]
    assert client.get("/api/timeline/buckets", params={"favorite": 1}).json()["total"] == 1
    assert client.get("/api/timeline/buckets", params={"from": "2024-04-01"}).json()["total"] == 2
    assert client.get("/api/timeline/buckets", params={"to": "2024-03-31"}).json()["total"] == 3
    pid = make_person(conn, "Asha")
    make_face(conn, int(c["id"]), pid)
    assert [x["key"] for x in client.get("/api/timeline/buckets", params={"person": pid}).json()["buckets"]] == ["2024-03"]
    now = now_iso()
    with tx(conn):
        conn.execute("INSERT INTO albums(id, name, name_key, created_at, updated_at) VALUES (7, 'Trip', 'trip', ?, ?)", (now, now))
        conn.execute("INSERT INTO album_items(album_id, item_id, added_at) VALUES (7, ?, ?)", (int(d["id"]), now))
        conn.execute("INSERT INTO album_items(album_id, item_id, added_at) VALUES (7, ?, ?)", (int(e["id"]), now))
    assert [x["key"] for x in client.get("/api/timeline/buckets", params={"album": 7}).json()["buckets"]] == ["2024-01", "2023-12"]
    assert client.get("/api/timeline/buckets", params={"album": 99}).status_code == 404
    assert client.get("/api/timeline/buckets", params={"kind": "video"}).json() == {"buckets": [], "total": 0}


def test_buckets_hide_trashed_items_companions_and_the_other_vault_side(client, vault_client, conn):
    a, *_ = _seed(conn)
    make_item(conn, taken_at=APRIL_MID, deleted_at=now_iso())
    make_item(conn, taken_at=APRIL_MID, kind="video", ext=".mov", mime="video/quicktime", companion_of=int(a["id"]))
    make_item(conn, taken_at="2022-02-02T10:00:00.000Z", nsfw=1)
    body = client.get("/api/timeline/buckets").json()
    assert body["total"] == 5 and "2022-02" not in [b["key"] for b in body["buckets"]]
    vb = vault_client.get("/api/timeline/buckets", params={"vault": 1}).json()
    assert [(b["key"], b["count"]) for b in vb["buckets"]] == [("2022-02", 1)] and vb["total"] == 1
    assert vault_client.get("/api/timeline/bucket", params={"key": "2022-02", "vault": 1}).json()["total"] == 1
    assert client.get("/api/timeline/bucket", params={"key": "2022-02"}).json()["total"] == 0


def test_buckets_fall_back_to_created_at_and_sort_added(client, conn):
    make_item(conn, taken_at=None)
    make_item(conn, taken_at=DEC_LAST)
    this_month = datetime.now(timezone.utc).astimezone().strftime("%Y-%m")
    keys = [b["key"] for b in client.get("/api/timeline/buckets").json()["buckets"]]
    assert keys[0] >= "2023-12" and "2023-12" in keys and len(keys) == 2
    added = client.get("/api/timeline/buckets", params={"sort": "added"}).json()
    assert len(added["buckets"]) == 1 and added["buckets"][0]["count"] == 2
    assert added["buckets"][0]["key"][:4] == this_month[:4]


def test_bucket_lists_the_month_in_library_order_with_paging(client, conn):
    a, b, c, d, e = _seed(conn)
    r = client.get("/api/timeline/bucket", params={"key": "2024-04", "limit": 1})
    assert r.status_code == 200, r.text
    page = r.json()
    assert page["total"] == 2 and [i["uid"] for i in page["items"]] == [a["uid"]] and page["next_cursor"]
    page2 = client.get("/api/timeline/bucket", params={"key": "2024-04", "limit": 1, "cursor": page["next_cursor"]}).json()
    assert [i["uid"] for i in page2["items"]] == [b["uid"]] and page2["next_cursor"] is None and page2["total"] == 0
    march = client.get("/api/timeline/bucket", params={"key": "2024-03"}).json()
    assert [i["uid"] for i in march["items"]] == [c["uid"]]
    assert client.get("/api/timeline/bucket", params={"key": "2024-03", "category": "personal"}).json()["items"] == []
    empty = client.get("/api/timeline/bucket", params={"key": "2020-01"}).json()
    assert empty == {"items": [], "next_cursor": None, "total": 0}
    assert client.get("/api/timeline/bucket", params={"key": "2024-04", "cursor": "nope"}).status_code == 400


def test_bucket_rejects_bad_keys(client):
    for bad in ("2024-13", "2024-1", "24-01", "2024/01", "abcd-ef"):
        assert client.get("/api/timeline/bucket", params={"key": bad}).status_code == 422, bad
    assert client.get("/api/timeline/bucket").status_code == 422


def test_vault_buckets_need_an_unlocked_session(client, conn):
    make_item(conn, taken_at=APRIL_MID, nsfw=1)
    assert client.get("/api/timeline/buckets", params={"vault": 1}).status_code == 403
    assert client.get("/api/timeline/bucket", params={"key": "2024-04", "vault": 1}).status_code == 403
    assert client.get("/api/timeline/buckets").json()["total"] == 0


def test_timeline_needs_a_session(anon_client):
    assert anon_client.get("/api/timeline/buckets").status_code == 401
    assert anon_client.get("/api/timeline/bucket", params={"key": "2024-04"}).status_code == 401


def test_buckets_scale(client, conn):
    base = datetime(2015, 1, 1, tzinfo=timezone.utc)
    with tx(conn):
        pass
    for i in range(300):
        make_item(conn, taken_at=(base + timedelta(days=i * 11)).strftime("%Y-%m-%dT%H:%M:%S.000Z"), write_files=False)
    body = client.get("/api/timeline/buckets").json()
    assert body["total"] == 300 and sum(b["count"] for b in body["buckets"]) == 300
    keys = [b["key"] for b in body["buckets"]]
    assert keys == sorted(keys, reverse=True) and len(set(keys)) == len(keys)
