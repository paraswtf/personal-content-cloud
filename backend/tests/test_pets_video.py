"""Pets in videos (section 22, 2026-11-08, later): a pet tagged on an item the way a person is, a box the owner
draws on a video's frame, and the detector's second look over a video's sampled frames. The models are off
(PCC_NO_MODELS=1); the detector is faked where its boxes matter."""
from __future__ import annotations

import shutil

import pytest
from PIL import Image

from app import storage
from app.analysis import animals as animals_mod
from app.db import tx
from app.util import now_iso
from tests.helpers import make_item, make_video

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg makes the test clip")

BOX = {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6}


def _clip(conn, tmp_path, seconds: float = 2.0):
    """A 640x360 testsrc2 clip stored as a video item with its original on disk."""
    path = make_video(tmp_path / "clip.mp4", seconds=seconds, size="640x360", fps=25)
    return make_item(conn, kind="video", ext=".mp4", mime="video/mp4", data=path.read_bytes(), width=640, height=360,
                     duration_s=float(seconds), fps=25.0, filename="cat.mp4")


def _pet(conn, name: str, species: str) -> int:
    with tx(conn):
        return int(conn.execute(
            "INSERT INTO persons(name, name_key, kind, species, created_at, updated_at) VALUES (?, ?, 'pet', ?, ?, ?)",
            (name, name.casefold(), species, now_iso(), now_iso())).lastrowid)


def _animal(conn, animal_id: int):
    return conn.execute("SELECT * FROM animals WHERE id = ?", (animal_id,)).fetchone()


# ------------------------------------------------------------------ a pet tag
def test_pet_tag_with_a_new_name_and_species_puts_the_pet_in_the_item(client, conn):
    item = make_item(conn, kind="video", ext=".mp4", mime="video/mp4", write_files=False, data=None)
    r = client.post(f"/api/items/{item['uid']}/tags",
                    json={"new_person_name": "Chutney", "kind": "pet", "species": "cat"})
    assert r.status_code == 200, r.text
    (ref,) = r.json()
    assert (ref["name"], ref["kind"], ref["relation"]) == ("Chutney", "pet", "in")
    row = conn.execute("SELECT kind, species FROM persons WHERE id = ?", (ref["id"],)).fetchone()
    assert (row["kind"], row["species"]) == ("pet", "cat")
    detail = client.get(f"/api/items/{item['uid']}").json()
    assert [(p["name"], p["kind"], p["relation"]) for p in detail["people"]] == [("Chutney", "pet", "in")]
    assert [i["uid"] for i in client.get(f"/api/people/{ref['id']}/items").json()["items"]] == [item["uid"]]
    person = client.get(f"/api/people/{ref['id']}").json()
    assert (person["kind"], person["species"], person["item_count"], person["related_count"]) == ("pet", "cat", 1, 0)
    # the same name again is the same pet
    other = make_item(conn)
    r = client.post(f"/api/items/{other['uid']}/tags", json={"new_person_name": "chutney", "kind": "pet", "species": "cat"})
    assert r.status_code == 200 and r.json()[0]["id"] == ref["id"]


def test_pet_tag_needs_a_species_and_takes_an_existing_pet_by_id(client, conn):
    item = make_item(conn)
    r = client.post(f"/api/items/{item['uid']}/tags", json={"new_person_name": "Chutney", "kind": "pet"})
    assert r.status_code == 400 and r.json()["detail"] == "species_needed"
    assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 0
    bruno = _pet(conn, "Bruno", "dog")
    r = client.post(f"/api/items/{item['uid']}/tags", json={"person_id": bruno})
    assert r.status_code == 200 and [(p["id"], p["kind"]) for p in r.json()] == [(bruno, "pet")]
    assert client.post(f"/api/items/{item['uid']}/tags",
                       json={"new_person_name": "X", "kind": "pet", "species": "dragon"}).status_code == 422
    # a person tag still works as before, and a pet's tag is removed like a person's
    r = client.post(f"/api/items/{item['uid']}/tags", json={"new_person_name": "Dev"})
    assert r.status_code == 200 and sorted(p["kind"] for p in r.json()) == ["person", "pet"]
    assert [p["id"] for p in client.delete(f"/api/items/{item['uid']}/tags/{bruno}").json()] != [bruno]


# ------------------------------------------------------------------ a box on a video frame
@needs_ffmpeg
def test_video_box_is_cut_from_the_frame_at_t(client, conn, tmp_path):
    clip = _clip(conn, tmp_path)
    r = client.post(f"/api/items/{clip['uid']}/animals", json=dict(BOX, species="cat", t=1.0))
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["frame_time_s"] == pytest.approx(1.0) and out["source"] == "user" and out["species"] == "cat"
    row = _animal(conn, out["id"])
    assert row["frame_time_s"] == pytest.approx(1.0) and row["source"] == "user" and row["species_source"] == "user"
    with Image.open(storage.animal_crop_path(out["id"])) as crop:
        # the cut of a 640x360 frame at (0.1, 0.1, 0.6, 0.6) is 384x216 before any downscale
        assert crop.width > 0 and abs(crop.width / crop.height - 384 / 216) < 0.05
    assert conn.execute("SELECT animal_count FROM items WHERE id = ?", (clip["id"],)).fetchone()[0] == 1
    detail = client.get(f"/api/items/{clip['uid']}").json()
    assert [(a["id"], a["frame_time_s"]) for a in detail["animals"]] == [(out["id"], pytest.approx(1.0))]
    # past the end of the clip there is no frame
    assert client.post(f"/api/items/{clip['uid']}/animals", json=dict(BOX, species="cat", t=99.0)).status_code == 409


def test_video_box_needs_the_frame_time(client, conn):
    clip = make_item(conn, kind="video", ext=".mp4", mime="video/mp4", write_files=False, data=None)
    r = client.post(f"/api/items/{clip['uid']}/animals", json=dict(BOX, species="cat"))
    assert r.status_code == 400 and r.json()["detail"] == "frame_time_needed"
    assert conn.execute("SELECT COUNT(*) FROM animals").fetchone()[0] == 0
    with pytest.raises(animals_mod.FrameTimeNeeded):
        animals_mod.add_box(conn, clip, (0.1, 0.1, 0.6, 0.6), species="cat")
    doc = make_item(conn, kind="pdf", ext=".pdf", mime="application/pdf", write_files=False, data=None)
    assert client.post(f"/api/items/{doc['uid']}/animals", json=dict(BOX, species="cat", t=0.0)).status_code == 400
    assert client.post(f"/api/items/{doc['uid']}/animals/detect").status_code == 400


# ------------------------------------------------------------------ the second look on a video
def test_sample_times_spread_over_the_clip_with_the_poster_first():
    from app.media.probe import MediaInfo

    short = MediaInfo(kind="video", mime="video/mp4", ext=".mp4", width=640, height=360, duration_s=2.0, fps=25.0)
    assert animals_mod.sample_times(short) == [0.2, 0.75, 1.25, 1.75]
    long = MediaInfo(kind="video", mime="video/mp4", ext=".mp4", width=640, height=360, duration_s=60.0, fps=25.0)
    times = animals_mod.sample_times(long)
    assert len(times) == animals_mod.REDETECT_VIDEO_FRAMES and times[0] == 3.0 and times[-1] == 57.5
    assert all(b - a > animals_mod.SAMPLE_GAP_S for a, b in zip(times, times[1:]))
    unknown = MediaInfo(kind="video", mime="video/mp4", ext=".mp4", width=640, height=360, duration_s=None, fps=None)
    assert animals_mod.sample_times(unknown) == [0.0]


@needs_ffmpeg
def test_video_second_look_stores_boxes_with_their_times_and_skips_a_frames_stored_box(client, conn, tmp_path,
                                                                                         monkeypatch):
    """The faked detector sees the same cat in every frame; the owner already drew its box on the poster frame
    (0.2 s), so that frame adds nothing and the other sampled frames add a box each, timed."""
    from app.ml import persons

    clip = _clip(conn, tmp_path)
    drawn = animals_mod.add_box(conn, clip, (0.1, 0.1, 0.6, 0.6), species="cat", t=0.2)

    def fake_detect(bgr, thresholds):
        h, w = bgr.shape[:2]
        assert set(thresholds) == set(persons.ANIMAL_CLASSES) and all(v == pytest.approx(0.2) for v in thresholds.values())
        return {15: [(0.1 * w, 0.1 * h, 0.7 * w, 0.7 * h, 0.4)]}

    monkeypatch.setattr(animals_mod, "available", lambda: True)
    monkeypatch.setattr(persons, "detect_classes", fake_detect)
    monkeypatch.setattr(animals_mod, "look_engine", lambda: None)
    r = client.post(f"/api/items/{clip['uid']}/animals/detect", json={"threshold": 0.2})
    assert r.status_code == 200, r.text
    new = r.json()
    assert sorted(round(a["frame_time_s"], 2) for a in new) == [0.75, 1.25, 1.75]
    assert all(a["source"] == "detector" and a["species"] == "cat" and a["det_score"] == pytest.approx(0.4) for a in new)
    rows = [_animal(conn, a["id"]) for a in new]
    assert all(r["source"] == "detector" and storage.animal_crop_path(int(r["id"])).is_file() for r in rows)
    assert conn.execute("SELECT animal_count FROM items WHERE id = ?", (clip["id"],)).fetchone()[0] == 4
    assert _animal(conn, drawn)["source"] == "user"
    # once more: every frame has its box now, nothing is added
    assert client.post(f"/api/items/{clip['uid']}/animals/detect", json={"threshold": 0.2}).json() == []
    # the video's tracks share one group in the item detail (section 10)
    detail = client.get(f"/api/items/{clip['uid']}").json()
    assert len(detail["animals"]) == 4
