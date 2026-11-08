"""Places (docs/ARCHITECTURE.md section 25): the gazetteer parses the GeoNames files and names the
nearest city, housekeeping tags items with GPS and back-fills GPS from originals, and place names
reach text search."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from app.config import settings
from app.db import tx
from app.geo import gazetteer, places
from app.media import probe
from tests.helpers import make_item

CITIES = [
    # geonameid, name, asciiname, alternatenames, lat, lon, fclass, fcode, cc, cc2, admin1, admin2, admin3, admin4, pop
    (1, "Panaji", "Panaji", "Panjim", "15.49574", "73.82624", "P", "PPLA", "IN", "", "33", "", "", "", "114405"),
    (2, "Mumbai", "Mumbai", "Bombay", "19.07283", "72.88261", "P", "PPLA", "IN", "", "16", "", "", "", "12691836"),
    (3, "Paris", "Paris", "", "48.85341", "2.3488", "P", "PPLC", "FR", "", "11", "", "", "", "2138551"),
    (4, "Paris 04 Hotel-de-Ville", "Paris 04", "", "48.8601", "2.3507", "P", "PPLX", "FR", "", "11", "", "", "", "28192"),
    (5, "Honolulu", "Honolulu", "", "21.30694", "-157.85833", "P", "PPLA", "US", "", "HI", "", "", "", "371657"),
]


def write_geo(root: Path) -> Path:
    d = root / "geo"
    d.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        rows = []
        for c in CITIES:
            row = [str(c[0]), c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14], "", "", "Asia/Kolkata", "2024-01-01"]
            rows.append("\t".join(row))
        zf.writestr(gazetteer.CITIES_TXT, "\n".join(rows) + "\n")
    (d / gazetteer.CITIES_ZIP).write_bytes(buf.getvalue())
    (d / gazetteer.ADMIN1).write_text("IN.33\tGoa\tGoa\t1271157\nIN.16\tMaharashtra\tMaharashtra\t1264418\nFR.11\tÎle-de-France\tIle-de-France\t3012874\nUS.HI\tHawaii\tHawaii\t5855797\n", encoding="utf-8")
    (d / gazetteer.COUNTRIES).write_text(
        "# GeoNames country information\n#ISO\tISO3\tISO-Numeric\tfips\tCountry\tCapital\n"
        "IN\tIND\t356\tIN\tIndia\tNew Delhi\nFR\tFRA\t250\tFR\tFrance\tParis\nUS\tUSA\t840\tUS\tUnited States\tWashington\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def tiny_geo(data_dir, tmp_path, monkeypatch):
    """A five-city gazetteer under a temporary MODELS_DIR (after data_dir, which reloads the settings)."""
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "geo_enabled", True)
    write_geo(tmp_path / "models")
    gazetteer.reset()
    yield tmp_path / "models" / "geo"
    gazetteer.reset()


def test_gazetteer_parses_and_names_the_nearest_city(tiny_geo):
    g = gazetteer.load()
    assert g is not None and len(g) == 5 and gazetteer.available()
    p = gazetteer.place_for(15.4989, 73.8278)
    assert p is not None and (p.city, p.admin, p.country) == ("Panaji", "Goa", "India")
    assert p.display == "Panaji, Goa, India"
    assert gazetteer.place_for(19.0760, 72.8777).city == "Mumbai"
    # a bigger place about as close names the spot: a district of Paris says Paris
    assert gazetteer.place_for(48.8605, 2.3510).city == "Paris"
    # across the antimeridian band and far from everything: nothing within 100 km
    assert gazetteer.place_for(0.0, 0.0) is None
    assert gazetteer.place_for(21.3, -157.86).display == "Honolulu, Hawaii, United States"
    assert gazetteer.place_for(None, 73.0) is None
    assert gazetteer.place_for(95.0, 73.0) is None


def test_gazetteer_off_without_files_or_when_disabled(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path / "nothing")
    gazetteer.reset()
    assert not gazetteer.available() and gazetteer.load() is None and gazetteer.place_for(15.5, 73.8) is None
    assert set(gazetteer.missing_files()) == set(gazetteer.FILES)
    write_geo(tmp_path / "nothing")
    monkeypatch.setattr(settings, "geo_enabled", False)
    gazetteer.reset()
    assert not gazetteer.available() and gazetteer.place_for(15.5, 73.8) is None
    assert gazetteer.download() == []
    gazetteer.reset()


def test_housekeeping_tags_places_and_indexes_them(tiny_geo, conn, client):
    goa = make_item(conn, gps_lat=15.4989, gps_lon=73.8278)
    sea = make_item(conn, gps_lat=0.0, gps_lon=0.0)
    plain = make_item(conn)
    trashed = make_item(conn, gps_lat=19.0760, gps_lon=72.8777, deleted_at="2026-01-01T00:00:00.000Z")
    assert places.housekeeping(conn, budget_s=5.0) >= 2
    rows = {r["id"]: r for r in conn.execute("SELECT id, place, place_city, place_admin, place_country, place_source FROM items")}
    assert rows[goa["id"]]["place"] == "Panaji, Goa, India" and rows[goa["id"]]["place_source"] == "geonames"
    assert rows[goa["id"]]["place_city"] == "Panaji" and rows[goa["id"]]["place_admin"] == "Goa" and rows[goa["id"]]["place_country"] == "India"
    assert rows[sea["id"]]["place"] is None and rows[sea["id"]]["place_source"] == "none"
    assert rows[plain["id"]]["place_source"] is None
    assert rows[trashed["id"]]["place_source"] is None, "trashed items wait until they are restored"
    # the place name reaches text search through the labels column
    r = client.get("/api/search", params={"q": "panaji", "mode": "text"})
    assert r.status_code == 200, r.text
    assert [h["uid"] for h in r.json()["items"]] == [goa["uid"]]
    assert client.get("/api/search", params={"q": "goa", "mode": "text"}).json()["total"] == 1
    # a second round finds nothing left to do
    assert places.housekeeping(conn, budget_s=5.0) == 0


def test_backfill_reads_gps_from_originals(tiny_geo, conn, monkeypatch):
    with_gps = make_item(conn)
    without = make_item(conn)
    doc = make_item(conn, kind="pdf", ext=".pdf", mime="application/pdf")
    answers = {with_gps["uid"]: (15.4989, 73.8278, 12.5)}
    seen: list[str] = []

    def fake_read_gps(path):
        uid = path.name.split(".v")[0]
        seen.append(uid)
        return answers.get(uid)

    monkeypatch.setattr(probe, "read_gps", fake_read_gps)
    touched = places.housekeeping(conn, budget_s=5.0)
    assert touched == 2 and sorted(seen) == sorted([with_gps["uid"], without["uid"]])
    rows = {r["id"]: r for r in conn.execute("SELECT id, gps_lat, gps_lon, gps_alt, gps_checked, place, place_source FROM items")}
    assert rows[with_gps["id"]]["gps_lat"] == 15.4989 and rows[with_gps["id"]]["gps_alt"] == 12.5
    assert rows[with_gps["id"]]["gps_checked"] == 1 and rows[with_gps["id"]]["place"] == "Panaji, Goa, India"
    assert rows[without["id"]]["gps_lat"] is None and rows[without["id"]]["gps_checked"] == 1 and rows[without["id"]]["place_source"] is None
    assert rows[doc["id"]]["gps_checked"] == 0, "documents are never re-read"
    assert places.housekeeping(conn, budget_s=5.0) == 0
    status = places.status(conn)
    assert status == {"gazetteer": True, "with_gps": 1, "unchecked": 0, "unplaced": 0, "running": False}


def test_backfill_pauses_on_a_tool_failure_and_honours_the_budget(tiny_geo, conn, monkeypatch):
    a = make_item(conn)
    b = make_item(conn)

    def broken(path):
        raise probe.ToolFailure("ffprobe missing")

    monkeypatch.setattr(probe, "read_gps", broken)
    assert places.housekeeping(conn, budget_s=5.0) == 0
    assert conn.execute("SELECT COUNT(*) FROM items WHERE gps_checked = 1").fetchone()[0] == 0
    monkeypatch.setattr(probe, "read_gps", lambda path: None)
    assert places.housekeeping(conn, budget_s=0.0) == 0, "no budget, nothing touched"
    assert places.housekeeping(conn, budget_s=5.0) == 2
    assert {r["gps_checked"] for r in conn.execute("SELECT gps_checked FROM items WHERE id IN (?, ?)", (a["id"], b["id"]))} == {1}


def test_tag_item_names_one_item(tiny_geo, conn):
    row = make_item(conn, gps_lat=19.0760, gps_lon=72.8777)
    assert places.tag_item(conn, row["id"]) is True
    assert conn.execute("SELECT place FROM items WHERE id = ?", (row["id"],)).fetchone()[0] == "Mumbai, Maharashtra, India"
    plain = make_item(conn)
    assert places.tag_item(conn, plain["id"]) is False
    with tx(conn):
        conn.execute("UPDATE items SET gps_lat = 0, gps_lon = 0, place_source = NULL WHERE id = ?", (row["id"],))
    assert places.tag_item(conn, row["id"]) is False
    after = conn.execute("SELECT place, place_source FROM items WHERE id = ?", (row["id"],)).fetchone()
    assert (after["place"], after["place_source"]) == (None, "none")


def test_housekeeping_yields_to_a_running_refresh(tiny_geo, conn, monkeypatch):
    make_item(conn, gps_lat=15.4989, gps_lon=73.8278)
    monkeypatch.setattr(places, "_refresh_running", True)
    assert places.housekeeping(conn, budget_s=5.0) == 0
    assert conn.execute("SELECT COUNT(*) FROM items WHERE place_source IS NULL").fetchone()[0] == 1
    monkeypatch.setattr(places, "_refresh_running", False)
    assert places.housekeeping(conn, budget_s=5.0) == 1


def test_status_counts_the_normal_side_in_one_pass(tiny_geo, conn):
    make_item(conn, gps_lat=15.4989, gps_lon=73.8278)              # with gps, unplaced until tagged
    make_item(conn)                                                 # unchecked image
    make_item(conn, kind="pdf", ext=".pdf", mime="application/pdf", write_files=False)  # documents are never checked
    make_item(conn, nsfw=1, gps_lat=19.0760, gps_lon=72.8777)       # vault: not counted
    make_item(conn, gps_lat=19.0760, gps_lon=72.8777, deleted_at="2026-01-01T00:00:00.000Z")  # trashed: not counted
    checked = make_item(conn)
    with tx(conn):
        conn.execute("UPDATE items SET gps_checked = 1 WHERE id = ?", (checked["id"],))
    s = places.status(conn)
    assert (s["with_gps"], s["unchecked"], s["unplaced"], s["gazetteer"]) == (1, 1, 1, True)
    places.housekeeping(conn, budget_s=5.0)
    s = places.status(conn)
    assert (s["with_gps"], s["unchecked"], s["unplaced"]) == (1, 0, 0)


def test_download_script_treats_places_data_as_optional(monkeypatch, capsys):
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "download_models.py"
    spec = importlib.util.spec_from_file_location("download_models_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from app.ml import models

    monkeypatch.setattr(models, "present", lambda key: True)

    def boom(progress=None):
        raise RuntimeError("no network")

    monkeypatch.setattr(gazetteer, "download", boom)
    assert module.main(["--quiet"]) == 0
    err = capsys.readouterr().err
    assert "places data not downloaded" in err and "model download failed" not in err
