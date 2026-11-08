"""The map API (docs/ARCHITECTURE.md section 25): points with the library's rules and a bbox, items in
a box, places, status and refresh."""
from __future__ import annotations

import pytest

from app.config import settings
from app.db import tx
from app.geo import gazetteer, places
from tests.helpers import make_item
from tests.test_geo import write_geo


@pytest.fixture()
def geo_world(tmp_path, monkeypatch, conn):
    """A gazetteer plus items: two in Goa, one in Mumbai, one in Paris, one without GPS, one in the
    vault, one trashed and one Live Photo companion (both in Goa)."""
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "geo_enabled", True)
    write_geo(tmp_path / "models")
    gazetteer.reset()
    goa1 = make_item(conn, gps_lat=15.50, gps_lon=73.83, taken_at="2024-03-01T10:00:00.000Z", category="places")
    goa2 = make_item(conn, gps_lat=15.51, gps_lon=73.84, taken_at="2024-03-02T10:00:00.000Z", kind="video", ext=".mp4", mime="video/mp4", data=b"video-goa")
    mum = make_item(conn, gps_lat=19.07, gps_lon=72.88, taken_at="2024-02-01T10:00:00.000Z", favorite=1)
    par = make_item(conn, gps_lat=48.86, gps_lon=2.35, taken_at="2024-01-01T10:00:00.000Z")
    plain = make_item(conn, taken_at="2024-01-02T10:00:00.000Z")
    vault = make_item(conn, gps_lat=15.52, gps_lon=73.85, nsfw=1)
    trashed = make_item(conn, gps_lat=15.53, gps_lon=73.86, deleted_at="2026-01-01T00:00:00.000Z")
    comp = make_item(conn, gps_lat=15.54, gps_lon=73.87, kind="video", ext=".mov", mime="video/quicktime", companion_of=goa1["id"], data=b"video-companion")
    places.housekeeping(conn, budget_s=5.0)
    yield dict(goa1=goa1, goa2=goa2, mum=mum, par=par, plain=plain, vault=vault, trashed=trashed, comp=comp)
    gazetteer.reset()


def test_points_follow_the_library_rules_and_the_bbox(client, geo_world):
    w = geo_world
    r = client.get("/api/map/points")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 5 and body["with_gps"] == 4
    by_uid = {p[2]: p for p in body["points"]}
    assert set(by_uid) == {w["goa1"]["uid"], w["goa2"]["uid"], w["mum"]["uid"], w["par"]["uid"]}
    assert by_uid[w["goa1"]["uid"]][:2] == [15.5, 73.83] and by_uid[w["goa1"]["uid"]][3] == 0
    assert by_uid[w["goa2"]["uid"]][3] == 1
    # a box around Goa
    r = client.get("/api/map/points", params={"bbox": "73,15,74.5,16"})
    assert {p[2] for p in r.json()["points"]} == {w["goa1"]["uid"], w["goa2"]["uid"]}
    assert r.json()["with_gps"] == 4, "with_gps counts the filtered items, not the box"
    # a box crossing the antimeridian keeps the rule readable: west > east
    r = client.get("/api/map/points", params={"bbox": "170,-10,-170,10"})
    assert r.status_code == 200 and r.json()["points"] == []
    for bad in ("1,2,3", "a,b,c,d", "0,50,10,40", "200,0,210,1"):
        assert client.get("/api/map/points", params={"bbox": bad}).status_code == 422, bad
    # the library's filters apply
    assert [p[2] for p in client.get("/api/map/points", params={"favorite": 1}).json()["points"]] == [w["mum"]["uid"]]
    assert [p[2] for p in client.get("/api/map/points", params={"category": "places"}).json()["points"]] == [w["goa1"]["uid"]]
    assert [p[2] for p in client.get("/api/map/points", params={"kind": "video"}).json()["points"]] == [w["goa2"]["uid"]]
    assert client.get("/api/map/points", params={"from": "2024-03-01", "to": "2024-03-31"}).json()["total"] == 2


def test_vault_points_only_with_an_unlocked_session(client, geo_world):
    from tests.conftest import ACCOUNT

    w = geo_world
    assert client.get("/api/map/points", params={"vault": 1}).status_code == 403
    assert client.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]}).status_code == 200
    r = client.get("/api/map/points", params={"vault": 1})
    assert r.status_code == 200 and [p[2] for p in r.json()["points"]] == [w["vault"]["uid"]]
    assert w["vault"]["uid"] not in {p[2] for p in client.get("/api/map/points").json()["points"]}
    assert [p["place"] for p in client.get("/api/map/places", params={"vault": 1}).json()] == ["Panaji, Goa, India"]


def test_items_in_a_box_page_in_library_order(client, geo_world):
    w = geo_world
    r = client.get("/api/map/items", params={"bbox": "73,15,74.5,16", "limit": 1})
    assert r.status_code == 200, r.text
    page = r.json()
    assert page["total"] == 2 and [i["uid"] for i in page["items"]] == [w["goa2"]["uid"]] and page["next_cursor"]
    r = client.get("/api/map/items", params={"bbox": "73,15,74.5,16", "limit": 1, "cursor": page["next_cursor"]})
    assert [i["uid"] for i in r.json()["items"]] == [w["goa1"]["uid"]] and r.json()["next_cursor"] is None
    assert client.get("/api/map/items").json()["total"] == 4, "without a box: every item with GPS"
    assert client.get("/api/map/items", params={"bbox": "nope"}).status_code == 422


def test_places_group_by_name_most_photos_first(client, geo_world):
    r = client.get("/api/map/places")
    assert r.status_code == 200, r.text
    rows = r.json()
    assert [(p["place"], p["count"]) for p in rows] == [("Panaji, Goa, India", 2), ("Mumbai, Maharashtra, India", 1), ("Paris, Île-de-France, France", 1)]
    goa = rows[0]
    assert goa["place_city"] == "Panaji" and goa["place_admin"] == "Goa" and goa["place_country"] == "India"
    assert abs(goa["lat"] - 15.505) < 1e-6 and abs(goa["lon"] - 73.835) < 1e-6
    assert [p["place"] for p in client.get("/api/map/places", params={"favorite": 1}).json()] == ["Mumbai, Maharashtra, India"]


def test_status_and_refresh(client, geo_world, monkeypatch, conn):
    r = client.get("/api/map/status")
    assert r.status_code == 200, r.text
    # counts over the normal side; the fixture's housekeeping already checked the photo without GPS
    assert r.json() == {"gazetteer": True, "with_gps": 4, "unchecked": 0, "unplaced": 0, "running": False}
    ran: list[str] = []

    def fake_background(key, fn, *args):
        ran.append(key)
        fn(*args)
        return object()

    from app import worker

    monkeypatch.setattr(worker, "run_in_background", fake_background)

    def fake_refresh_all():
        ran.append("done")
        with places._refresh_lock:
            places._refresh_running = False

    monkeypatch.setattr(places, "_refresh_all", fake_refresh_all)
    r = client.post("/api/map/refresh")
    assert r.status_code == 200 and r.json() == {"started": True}
    assert ran == ["geo", "done"]
    assert client.get("/api/map/status").json()["running"] is False


def test_refresh_reports_a_running_one(client, monkeypatch):
    monkeypatch.setattr(places, "_refresh_running", True)
    assert client.post("/api/map/refresh").json() == {"started": False}
    assert client.get("/api/map/status").json()["running"] is True
    monkeypatch.setattr(places, "_refresh_running", False)


def test_map_routes_need_a_session(anon_client):
    for path in ("/api/map/points", "/api/map/items", "/api/map/places", "/api/map/status"):
        assert anon_client.get(path).status_code == 401, path
    assert anon_client.post("/api/map/refresh", headers={"X-Requested-With": "pcc"}).status_code == 401


def test_points_without_gazetteer_still_work(client, conn, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path / "none")  # after client, which reloads the settings
    gazetteer.reset()
    row = make_item(conn, gps_lat=10.0, gps_lon=20.0)
    assert places.housekeeping(conn, budget_s=1.0) >= 0
    assert client.get("/api/map/status").json()["gazetteer"] is False
    assert [p[2] for p in client.get("/api/map/points").json()["points"]] == [row["uid"]]
    assert client.get("/api/map/places").json() == []
    with tx(conn):
        conn.execute("UPDATE items SET place = 'Somewhere', place_source = 'geonames' WHERE id = ?", (row["id"],))
    assert client.get("/api/map/places").json()[0]["count"] == 1
    gazetteer.reset()
