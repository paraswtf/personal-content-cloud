"""Forwards (docs/ARCHITECTURE.md section 29): the signals, the decision, the analyze stage, the learner,
the look-alike vote, the API and the backfill. The models are off: meme_p stays None and the text signal
comes from the edge heuristic of section 9."""
from __future__ import annotations

import json

import pytest

from app.analysis import bodies, forwards, pipeline
from app.config import settings
from tests.helpers import jpeg_bytes, make_item, noise_image, text_image


@pytest.fixture()
def no_models(monkeypatch):
    monkeypatch.setattr(pipeline, "_model", lambda name: None)
    monkeypatch.setattr(bodies, "available", lambda: False)


def _row(conn, item_id: int):
    return conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()


def _text_page():
    return text_image([f"Good morning, have a blessed day {i}" for i in range(14)], w=1600, h=1200)


STRIPPED = dict(kind="image", ext=".jpg", width=1600, height=1200, camera=None, exif_datetime=None, gps_lat=None,
                bitrate=None, filename="IMG_0001.JPG")


# ------------------------------------------------------------------ signals and the rule
def test_name_patterns_name_the_app():
    assert forwards.name_app("IMG-20231105-WA0012.jpg") == "whatsapp"
    assert forwards.name_app("VID-20231105-WA0003.mp4") == "whatsapp"
    assert forwards.name_app("PTT-20231105-WA0007.opus") == "whatsapp"
    assert forwards.name_app("WhatsApp Image 2023-11-05 at 10.12.33.jpeg") == "whatsapp"
    assert forwards.name_app("WhatsApp Video 2023-11-05 at 10.12.33.mp4") == "whatsapp"
    assert forwards.name_app("photo_2023-11-05_10-12-33.jpg") == "telegram"
    assert forwards.name_app("video_2023-11-05_10-12-33.mp4") == "telegram"
    assert forwards.name_app("FB_IMG_1699171953123.jpg") == "facebook"
    assert forwards.name_app("Snapchat-1234567890.jpg") == "snapchat"
    assert forwards.name_app("received_1234567890.jpeg") == "messenger"
    for name in ("IMG_5468.MOV", "IMG_1234.JPG", "DSC_0001.JPG", "Screenshot 2024-01-01 at 10.00.png", "", None):
        assert forwards.name_app(name) is None, name


def test_jpeg_quality_is_read_back_from_the_quantisation_table(tmp_path):
    img = noise_image(640, 480, seed=1)
    for q, lo, hi in ((75, 68, 82), (95, 90, 100), (50, 44, 56)):
        p = tmp_path / f"q{q}.jpg"
        p.write_bytes(jpeg_bytes(img, quality=q))
        got = forwards.jpeg_quality(p)
        assert got is not None and lo <= got <= hi, (q, got)
    png = tmp_path / "x.png"
    img.save(png, "PNG")
    assert forwards.jpeg_quality(png) is None
    assert forwards.jpeg_quality(tmp_path / "missing.jpg") is None
    assert forwards.jpeg_quality(None) is None


def test_file_signals():
    s = forwards.file_signals(**STRIPPED)
    assert s["stripped"] and s["small"] and s["wa_size"] and s["side"] == 1600 and s["jpeg_quality"] is None
    assert s["name_app"] is None and s["video_small"] is False
    camera = forwards.file_signals(**{**STRIPPED, "width": 4032, "height": 3024, "camera": "Apple iPhone 15",
                                      "exif_datetime": "2024:01:01 10:00:00", "gps_lat": 18.5})
    assert not camera["stripped"] and not camera["small"] and not camera["wa_size"]
    png = forwards.file_signals(**{**STRIPPED, "ext": ".png", "width": 1170, "height": 2532})
    assert not png["stripped"]  # PNG is screenshots, not a stripped camera file
    video = forwards.file_signals(kind="video", ext=".mp4", width=848, height=480, camera=None, exif_datetime=None,
                                  gps_lat=None, bitrate=900_000, filename="VID_0003.mp4")
    assert video["video_small"]
    phone = forwards.file_signals(kind="video", ext=".mov", width=1920, height=1080, camera=None, exif_datetime=None,
                                  gps_lat=18.5, bitrate=12_000_000, filename="IMG_0004.MOV")
    assert not phone["video_small"]


def test_decision_table():
    base = forwards.file_signals(**STRIPPED)
    kept = forwards.decide({**base, "meme_p": 0.1, "text": False, "people": True})
    assert kept.received_via == "app" and kept.forward is False and kept.reasons == {}
    meme = forwards.decide({**base, "meme_p": 0.8, "text": False, "people": False})
    assert meme.received_via == "app" and meme.forward and meme.reasons["meme_p"] == 0.8 and meme.reasons["wa_size"]
    assert meme.reasons["stripped"] is True and meme.reasons["small"] == 1600
    text = forwards.decide({**base, "meme_p": None, "text": True, "people": False})
    assert text.forward and text.reasons["text"] is True
    text_people = forwards.decide({**base, "meme_p": None, "text": True, "people": True})
    assert text_people.received_via == "app" and not text_people.forward
    poster = forwards.decide({**base, "meme_p": 0.75, "text": False, "people": True})
    assert poster.forward  # a strong meme signal, people or not
    weak = forwards.decide({**base, "meme_p": 0.55, "text": False, "people": True})
    assert not weak.forward  # people in a mildly meme-like picture: kept
    named = forwards.decide({**base, "name_app": "whatsapp", "meme_p": 0.6, "text": False, "people": False})
    assert named.received_via == "whatsapp" and named.forward and named.reasons["name"] == "whatsapp"
    camera = forwards.file_signals(**{**STRIPPED, "width": 4032, "height": 3024, "camera": "Apple iPhone 15",
                                      "exif_datetime": "2024:01:01 10:00:00"})
    assert forwards.decide({**camera, "meme_p": 0.9, "text": True, "people": False}).received_via is None
    # an odd size: the JPEG quality decides whether the file came through an app
    odd = forwards.file_signals(**{**STRIPPED, "width": 1500, "height": 1100})
    assert forwards.decide({**odd, "jpeg_quality": 75, "meme_p": 0.9, "text": False, "people": False}).received_via == "app"
    assert forwards.decide({**odd, "jpeg_quality": 94, "meme_p": 0.9, "text": False, "people": False}).received_via is None
    video = forwards.file_signals(kind="video", ext=".mp4", width=848, height=480, camera=None, exif_datetime=None,
                                  gps_lat=None, bitrate=900_000, filename="VID_0003.mp4")
    assert forwards.decide({**video, "meme_p": None, "text": False, "people": False}).received_via == "app"
    # a labelled look-alike, then the learner, take over the second step
    assert forwards.decide({**base, "meme_p": 0.1, "text": False, "people": True}, learned=0.9).forward
    assert not forwards.decide({**base, "meme_p": 0.9, "text": False, "people": False}, learned=0.2).forward
    assert not forwards.decide({**base, "meme_p": 0.9, "text": False, "people": False}, learned=0.9, vote=False).forward
    assert forwards.decide({**base, "meme_p": 0.1, "text": False, "people": True}, vote=True).reasons["lookalike"]


# ------------------------------------------------------------------ the analyze stage
def test_analyze_sets_a_received_text_image_apart(conn, no_models):
    data = jpeg_bytes(_text_page(), quality=75)
    item = make_item(conn, data=data, width=1600, height=1200, analysis_status="pending", filename="IMG_0101.JPG")
    pipeline.analyze_item(conn, item["id"], 1)
    row = _row(conn, item["id"])
    assert row["received_via"] == "app" and row["forward"] == 1 and row["forward_checked"] == 1
    reasons, sig = json.loads(row["forward_reasons"]), json.loads(row["forward_signals"])
    assert reasons["text"] is True and reasons["stripped"] is True and 68 <= reasons["jpeg_quality"] <= 82
    assert sig["jpeg_quality"] == reasons["jpeg_quality"] and sig["meme_p"] is None and sig["people"] is False


def test_analyze_keeps_a_received_plain_photo(conn, no_models):
    data = jpeg_bytes(noise_image(1600, 1200, seed=5), quality=75)
    item = make_item(conn, data=data, width=1600, height=1200, analysis_status="pending", filename="IMG_0102.JPG")
    pipeline.analyze_item(conn, item["id"], 1)
    row = _row(conn, item["id"])
    assert row["received_via"] == "app" and row["forward"] == 0 and row["forward_reasons"] is None
    assert row["forward_checked"] == 1


def test_analyze_leaves_a_camera_photo_alone(conn, no_models):
    data = jpeg_bytes(noise_image(2400, 1800, seed=6), quality=92)
    item = make_item(conn, data=data, width=2400, height=1800, analysis_status="pending", filename="IMG_0103.JPG",
                     camera="Apple iPhone 15", exif_datetime="2024:01:01 10:00:00")
    pipeline.analyze_item(conn, item["id"], 1)
    row = _row(conn, item["id"])
    assert row["received_via"] is None and row["forward"] == 0 and row["forward_checked"] == 1
    assert json.loads(row["forward_signals"])["stripped"] is False


def test_the_owners_decision_survives_analysis(conn, no_models):
    data = jpeg_bytes(_text_page(), quality=75)
    item = make_item(conn, data=data, width=1600, height=1200, analysis_status="pending", filename="IMG_0104.JPG",
                     forward=0, forward_source="user")
    pipeline.analyze_item(conn, item["id"], 1)
    row = _row(conn, item["id"])
    assert row["forward"] == 0 and row["forward_source"] == "user" and row["received_via"] == "app"
    assert row["forward_checked"] == 1 and row["forward_reasons"] is None


def test_disabled_forwards_decide_nothing(conn, no_models, monkeypatch):
    monkeypatch.setattr(settings, "forwards_enabled", False)
    data = jpeg_bytes(_text_page(), quality=75)
    item = make_item(conn, data=data, width=1600, height=1200, analysis_status="pending", filename="IMG_0105.JPG")
    pipeline.analyze_item(conn, item["id"], 1)
    row = _row(conn, item["id"])
    assert row["received_via"] is None and row["forward"] == 0 and row["forward_checked"] == 1


# ------------------------------------------------------------------ learning
def test_learner_takes_over_after_ten_examples(conn):
    from app.learning import forwards as learned

    base = {"name_app": None, "stripped": True, "small": True, "side": 1600, "wa_size": True, "jpeg_quality": 75,
            "video_small": False, "text": False, "meme_p": 0.6}
    assert learned.probability({**base, "people": False}) is None
    for i in range(8):  # the owner kept photos of people received through an app
        make_item(conn, write_files=False, forward=0, forward_source="user",
                  forward_signals=json.dumps({**base, "people": True}))
    for i in range(8):  # and set the memes apart
        make_item(conn, write_files=False, forward=1, forward_source="user",
                  forward_signals=json.dumps({**base, "people": False}))
    learned.train(conn)
    st = learned.status()
    assert st["learned"] and st["examples"] == 16 and st["positives"] == 8
    assert learned.probability({**base, "people": False}) > 0.5
    assert learned.probability({**base, "people": True}) < 0.5


def test_learner_waits_for_both_labels(conn):
    from app.learning import forwards as learned

    for i in range(12):
        make_item(conn, write_files=False, forward=1, forward_source="user", forward_signals='{"stripped": true}')
    learned.train(conn)
    assert learned.status()["learned"] is False and learned.probability({"stripped": True}) is None


def test_lookalike_vote(conn):
    from app.learning import images
    from tests.test_learning_core import add_embedding, near, unit

    a = make_item(conn, write_files=False, forward=1, forward_source="user")
    add_embedding(conn, a["id"], unit(1))
    b = make_item(conn, write_files=False, forward=0, forward_source="user")
    add_embedding(conn, b["id"], unit(2))
    images.train(conn)
    assert images.status()["forward"]["examples"] == 2
    assert images.forward_vote(near(unit(1), 3, 0.2)) is True
    assert images.forward_vote(near(unit(2), 4, 0.2)) is False
    assert images.forward_vote(unit(9)) is None
    assert images.forward_vote(None) is None


# ------------------------------------------------------------------ the API
def test_api_filters_review_actions_and_events(client, conn):
    plain = make_item(conn, write_files=False)
    fwd = make_item(conn, write_files=False, received_via="whatsapp", forward=1, forward_checked=1,
                    forward_reasons='{"name": "whatsapp", "meme_p": 0.8}',
                    forward_signals='{"name_app": "whatsapp", "meme_p": 0.8, "people": false}')
    kept = make_item(conn, write_files=False, received_via="app", forward=0, forward_checked=1)
    page = client.get("/api/items").json()
    assert page["total"] == 2 and all(i["forward"] is False for i in page["items"])
    assert client.get("/api/items", params={"forwards": "include"}).json()["total"] == 3
    only = client.get("/api/items", params={"forwards": "only"}).json()
    assert only["total"] == 1 and only["items"][0]["uid"] == fwd["uid"]
    assert only["items"][0]["received_via"] == "whatsapp" and only["items"][0]["forward_reasons"]["name"] == "whatsapp"
    assert client.get("/api/items", params={"received": "any", "forwards": "include"}).json()["total"] == 2
    assert client.get("/api/items", params={"received": "whatsapp", "forwards": "include"}).json()["total"] == 1
    assert client.get("/api/items", params={"received": "telegram", "forwards": "include"}).json()["total"] == 0
    assert client.get("/api/review/forwards").json()["items"][0]["uid"] == fwd["uid"]
    assert client.get("/api/stats").json()["forwards"] == 1
    info = client.get("/api/system").json()["forwards"]
    assert info["forwards"] == 1 and info["received"] == 2 and info["enabled"] is True and info["unchecked"] == 1
    d = client.get(f"/api/items/{fwd['uid']}").json()
    assert d["forward"] is True and d["forward_source"] == "auto" and d["received_via"] == "whatsapp"
    # keep it: a learning event with label 0 and the signals of the time
    d = client.patch(f"/api/items/{fwd['uid']}", json={"forward": False}).json()
    assert d["forward"] is False and d["forward_reasons"] is None and d["forward_source"] == "user"
    ev = conn.execute("SELECT label, features FROM learning_events WHERE kind = 'forward'").fetchall()
    assert len(ev) == 1 and ev[0]["label"] == 0 and json.loads(ev[0]["features"])["name_app"] == "whatsapp"
    assert client.get("/api/items", params={"forwards": "only"}).json()["total"] == 0
    # bulk: set apart and back
    r = client.post("/api/items/bulk", json={"uids": [kept["uid"], plain["uid"]], "action": "forward"})
    assert r.status_code == 200 and r.json()["updated"] == 2
    assert client.get("/api/items", params={"forwards": "only"}).json()["total"] == 2
    r = client.post("/api/items/bulk", json={"uids": [plain["uid"]], "action": "unforward"})
    assert r.json()["updated"] == 1
    assert client.get("/api/items").json()["total"] == 2
    assert conn.execute("SELECT COUNT(*) FROM learning_events WHERE kind = 'forward'").fetchone()[0] == 4
    # the owner's decisions are what the learners read
    assert conn.execute("SELECT COUNT(*) FROM items WHERE forward_source = 'user'").fetchone()[0] == 3


# ------------------------------------------------------------------ the backfill
def test_backfill_decides_older_items_and_respects_the_owner(conn, no_models):
    old = make_item(conn, data=jpeg_bytes(_text_page(), quality=75), width=1600, height=1200,
                    filename="IMG_0201.JPG", doc_likeness=0.9)
    plain = make_item(conn, data=jpeg_bytes(noise_image(1600, 1200, seed=7), quality=75), width=1600, height=1200,
                      filename="IMG_0202.JPG")
    mine = make_item(conn, write_files=False, forward=0, forward_source="user", filename="IMG-20231105-WA0001.jpg")
    pending = make_item(conn, write_files=False, analysis_status="pending")
    assert forwards.status(conn)["unchecked"] == 3
    assert forwards.housekeeping(conn, budget_s=5.0) == 3
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM items")}
    assert rows[old["id"]]["forward"] == 1 and rows[old["id"]]["received_via"] == "app"
    assert json.loads(rows[old["id"]]["forward_reasons"])["text"] is True
    assert rows[plain["id"]]["forward"] == 0 and rows[plain["id"]]["received_via"] == "app"
    assert rows[mine["id"]]["forward"] == 0 and rows[mine["id"]]["received_via"] == "whatsapp"
    assert rows[mine["id"]]["forward_checked"] == 1 and rows[mine["id"]]["forward_source"] == "user"
    assert rows[pending["id"]]["forward_checked"] == 0
    assert forwards.housekeeping(conn, budget_s=5.0) == 0
    assert forwards.status(conn)["unchecked"] == 0 and forwards.status(conn)["running"] is False


def test_backfill_honours_the_budget(conn, no_models):
    for i in range(3):
        make_item(conn, write_files=False, filename=f"IMG-2023110{i}-WA000{i}.jpg")
    assert forwards.housekeeping(conn, budget_s=0.0) == 0
    assert forwards.status(conn)["unchecked"] == 3
