"""Face boxes the owner draws (section 10): the route, the face the detector must find inside, a stored face under
the box, videos, undo, and what a re-analysis keeps. The models are off (PCC_NO_MODELS=1); the face search is
faked where a face matters."""
from __future__ import annotations

import numpy as np
import pytest

from app import storage
from app.analysis import person_boxes, pipeline
from app.analysis.index import get_index
from app.db import tx
from tests.helpers import make_face, make_item, make_person, make_video
from tests.test_person_boxes import FakeFace, _record, _unit

BOX = {"x": 0.3, "y": 0.2, "w": 0.2, "h": 0.3}           # on an 800 x 600 photo: (240, 120) to (400, 300)
FOUND = (262.0, 140.0, 380.0, 280.0)                       # the face the detector finds in it, in pixels


def _photo(conn, **kw):
    return make_item(conn, width=800, height=600, **kw)


def _detector(monkeypatch, face: FakeFace | list[FakeFace] | None) -> None:
    monkeypatch.setattr(pipeline, "_model", lambda name: object() if name == "faces" else None)
    faces = face if isinstance(face, list) else [face] if face is not None else []
    monkeypatch.setattr(person_boxes, "_faces_in", lambda bgr, px: faces)


def _face(conn, face_id: int):
    return conn.execute("SELECT * FROM faces WHERE id = ?", (face_id,)).fetchone()


def test_a_drawn_face_box_holds_the_person_with_the_found_face(client, conn, monkeypatch):
    item = _photo(conn)
    _detector(monkeypatch, FakeFace(bbox=np.array(FOUND, dtype=np.float32), embedding=_unit(9)))
    r = client.post(f"/api/items/{item['uid']}/faces", json={**BOX, "new_person_name": "Meera"})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["person"]["name"] == "Meera" and out["assigned_by"] == "user" and out["source"] == "user"
    assert (out["x"], out["y"], out["w"], out["h"]) == pytest.approx((0.3, 0.2, 0.2, 0.3))   # the owner's box
    row = _face(conn, out["id"])
    assert np.allclose(np.frombuffer(row["embedding"], dtype=np.float32), _unit(9))
    assert row["det_score"] == pytest.approx(0.92) and row["clarity"] == pytest.approx(24.5)
    assert storage.face_crop_path(out["id"]).is_file()
    assert conn.execute("SELECT face_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 1
    idx = get_index()
    idx.sync(conn)
    assert np.allclose(idx.embeddings_of([out["id"]])[0], _unit(9))
    detail = client.get(f"/api/items/{item['uid']}").json()
    assert [f["id"] for f in detail["faces"]] == [out["id"]] and detail["people"][0]["name"] == "Meera"


def test_no_face_inside_no_detector_and_bad_boxes_are_refused(client, conn, monkeypatch):
    item = _photo(conn)
    asha = make_person(conn, "Asha")
    body = {**BOX, "person_id": asha}
    assert client.post(f"/api/items/{item['uid']}/faces", json=body).status_code == 409   # no detector here
    _detector(monkeypatch, None)
    r = client.post(f"/api/items/{item['uid']}/faces", json=body)
    assert r.status_code == 422 and r.json()["detail"] == "no_face"
    _detector(monkeypatch, FakeFace(bbox=np.array((10.0, 10.0, 60.0, 60.0), dtype=np.float32)))   # beside the box
    assert client.post(f"/api/items/{item['uid']}/faces", json=body).json()["detail"] == "no_face"
    r = client.post(f"/api/items/{item['uid']}/faces", json={"x": 0.3, "y": 0.2, "w": 0.01, "h": 0.01, "person_id": asha})
    assert r.status_code == 422 and "pixels" in r.json()["detail"]
    assert client.post(f"/api/items/{item['uid']}/faces", json=BOX).json()["detail"] == "person_needed"
    assert client.post("/api/items/nope/faces", json=body).status_code == 404
    assert conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 0


def test_a_stored_face_under_the_box_is_assigned_instead(client, conn, monkeypatch):
    item = _photo(conn)
    asha = make_person(conn, "Asha")
    stored = make_face(conn, item["id"], None, box=(262 / 800, 140 / 600, 118 / 800, 140 / 600), embedding=_unit(7))
    _detector(monkeypatch, FakeFace(bbox=np.array(FOUND, dtype=np.float32)))
    r = client.post(f"/api/items/{item['uid']}/faces", json={**BOX, "person_id": asha})
    assert r.status_code == 201, r.text
    assert r.json()["id"] == stored and r.json()["person"]["id"] == asha
    row = _face(conn, stored)
    assert row["person_id"] == asha and row["assigned_by"] == "user"
    assert conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 1


def test_a_video_face_box_needs_its_frame_and_gets_a_short_span(client, conn, monkeypatch, tmp_path):
    path = make_video(tmp_path / "clip.mp4", seconds=4, size="640x360", fps=25)
    clip = make_item(conn, kind="video", ext=".mp4", mime="video/mp4", data=path.read_bytes(), width=640, height=360,
                     duration_s=4.0, fps=25.0)
    asha = make_person(conn, "Asha")
    r = client.post(f"/api/items/{clip['uid']}/faces", json={**BOX, "person_id": asha})
    assert r.status_code == 400 and r.json()["detail"] == "frame_time_needed"
    _detector(monkeypatch, FakeFace(bbox=np.array((200.0, 80.0, 300.0, 170.0), dtype=np.float32)))   # in 640 x 360
    r = client.post(f"/api/items/{clip['uid']}/faces", json={**BOX, "t": 2.0, "person_id": asha})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["frame_time_s"] == pytest.approx(2.0) and out["spans"] == [[1.5, 2.5]]


def test_undo_takes_the_drawn_face_and_its_crop_away(client, conn, monkeypatch):
    item = _photo(conn)
    _detector(monkeypatch, FakeFace(bbox=np.array(FOUND, dtype=np.float32)))
    r = client.post(f"/api/items/{item['uid']}/faces", json={**BOX, "new_person_name": "Meera"})
    fid = r.json()["id"]
    u = client.post(f"/api/actions/{r.headers['x-action-id']}/undo")
    assert u.status_code == 200, u.text
    assert _face(conn, fid) is None and not storage.face_crop_path(fid).exists()
    assert conn.execute("SELECT face_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 0


def test_reanalysis_keeps_a_drawn_photo_face_and_drops_its_twin(client, conn, monkeypatch):
    item = _photo(conn)
    _detector(monkeypatch, FakeFace(bbox=np.array(FOUND, dtype=np.float32)))
    fid = client.post(f"/api/items/{item['uid']}/faces", json={**BOX, "new_person_name": "Meera"}).json()["id"]
    with tx(conn):
        conn.execute("UPDATE items SET analysis_status = 'pending' WHERE id = ?", (item["id"],))
    real_run = pipeline._run_models

    def run(item_row, v, siglip, nsfw, faces_engine, not_face=None):
        r = real_run(item_row, v, siglip, nsfw, faces_engine, not_face=not_face)
        r.faces = [_record((0.305, 0.205, 0.195, 0.295), 21), _record((0.7, 0.1, 0.1, 0.13), 22)]
        r.face_torsos = [None, None]
        r.faces_ran = True
        return r

    monkeypatch.setattr(pipeline, "_run_models", run)
    pipeline.analyze_item(conn, item["id"], 1)
    kept = _face(conn, fid)
    assert kept is not None and kept["source"] == "user" and kept["person_id"] is not None
    xs = sorted(round(r["x"], 2) for r in conn.execute("SELECT x FROM faces WHERE item_id = ?", (item["id"],)))
    assert xs == [0.3, 0.7]


def test_a_face_drawn_inside_a_detected_person_box_becomes_its_face(client, conn, monkeypatch):
    item = _photo(conn)
    group = make_person(conn, None)   # the unnamed group the box was left with after its face went
    with tx(conn):
        body = int(conn.execute(
            "INSERT INTO bodies(item_id, person_id, assigned_by, x, y, w, h, det_score, created_at) "
            "VALUES (?, ?, 'auto', 0.25, 0.15, 0.3, 0.8, 0.8, '2026-11-09T00:00:00Z')", (item["id"], group)).lastrowid)
    _detector(monkeypatch, FakeFace(bbox=np.array(FOUND, dtype=np.float32)))
    r = client.post(f"/api/items/{item['uid']}/faces", json={**BOX, "new_person_name": "Meera"})
    assert r.status_code == 201, r.text
    row = conn.execute("SELECT face_id, person_id FROM bodies WHERE id = ?", (body,)).fetchone()
    assert row["face_id"] == r.json()["id"] and row["person_id"] == r.json()["person"]["id"]
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE id = ?", (group,)).fetchone()[0] == 0
    detail = client.get(f"/api/items/{item['uid']}").json()
    assert [p["name"] for p in detail["people"]] == ["Meera"]


def test_the_only_face_of_an_unnamed_group_moves_to_someone(client, conn, monkeypatch):
    """Regression: the emptied group is pruned before the look-alikes are followed, so no memory is kept of it
    (a foreign key failure answered 500 before)."""
    item = _photo(conn)
    group = make_person(conn, None)
    fid = make_face(conn, item["id"], group, box=(262 / 800, 140 / 600, 118 / 800, 140 / 600))
    r = client.post("/api/faces/assign", json={"face_ids": [fid], "new_person_name": "Neil"})
    assert r.status_code == 200, r.text
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE id = ?", (group,)).fetchone()[0] == 0
    # and through a face box drawn over it
    other = make_person(conn, None)
    f2 = make_face(conn, item["id"], other, box=(0.6, 0.2, 0.15, 0.2), embedding=_unit(7))
    _detector(monkeypatch, FakeFace(bbox=np.array((480.0, 120.0, 600.0, 240.0), dtype=np.float32)))
    r = client.post(f"/api/items/{item['uid']}/faces", json={"x": 0.59, "y": 0.19, "w": 0.17, "h": 0.22, "new_person_name": "Buzz"})
    assert r.status_code == 201, r.text
    assert r.json()["id"] == f2 and r.json()["person"]["name"] == "Buzz"


# Overlapping faces (IMG_5192: two faces cheek to cheek, the second tilted): a box drawn over the second face never
# takes the first one's stored face, and a face the upright search misses is found in a turned copy.

PARAS = (240 / 800, 60 / 600, 160 / 800, 180 / 600)          # stored, named: (240, 60) to (400, 240)
OVER = {"x": 0.42, "y": 0.25, "w": 0.25, "h": 0.4}         # drawn over the second face: (336, 150) to (536, 390)


def test_a_box_over_a_neighbours_face_never_takes_it(client, conn, monkeypatch):
    item = _photo(conn)
    paras, anagha = make_person(conn, "Paras"), make_person(conn, "Anagha")
    stored = make_face(conn, item["id"], paras, box=PARAS, assigned_by="user", embedding=_unit(3))
    # the upright search only finds Paras's face, its centre just inside the drawn box
    _detector(monkeypatch, FakeFace(bbox=np.array((300.0, 70.0, 400.0, 230.0), dtype=np.float32), embedding=_unit(3)))
    monkeypatch.setattr(person_boxes, "_faces_turned", lambda bgr, px: [])
    r = client.post(f"/api/items/{item['uid']}/faces", json={**OVER, "person_id": anagha})
    assert r.status_code == 422 and r.json()["detail"] == "no_face"
    assert _face(conn, stored)["person_id"] == paras            # Paras keeps his face


def test_a_tilted_second_face_is_found_turned_and_stored_as_its_own(client, conn, monkeypatch):
    item = _photo(conn)
    paras, anagha = make_person(conn, "Paras"), make_person(conn, "Anagha")
    stored = make_face(conn, item["id"], paras, box=PARAS, assigned_by="user", embedding=_unit(3))
    _detector(monkeypatch, FakeFace(bbox=np.array((300.0, 70.0, 400.0, 230.0), dtype=np.float32), embedding=_unit(3)))
    turned = FakeFace(bbox=np.array((350.0, 170.0, 520.0, 370.0), dtype=np.float32), embedding=_unit(11))
    monkeypatch.setattr(person_boxes, "_faces_turned", lambda bgr, px: [turned])
    r = client.post(f"/api/items/{item['uid']}/faces", json={**OVER, "person_id": anagha})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["id"] != stored and out["person"]["id"] == anagha
    assert np.allclose(np.frombuffer(_face(conn, out["id"])["embedding"], dtype=np.float32), _unit(11))
    assert _face(conn, stored)["person_id"] == paras
    assert conn.execute("SELECT face_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 2


def test_two_heavily_overlapping_faces_are_told_apart_by_their_vectors(client, conn, monkeypatch):
    item = _photo(conn)
    paras, anagha = make_person(conn, "Paras"), make_person(conn, "Anagha")
    stored = make_face(conn, item["id"], paras, box=PARAS, assigned_by="user", embedding=_unit(3))
    # the found face lies over Paras's stored box (IoU >= 0.4) but is someone else (a different vector)
    other = FakeFace(bbox=np.array((270.0, 90.0, 420.0, 260.0), dtype=np.float32), embedding=_unit(11))
    _detector(monkeypatch, other)
    r = client.post(f"/api/items/{item['uid']}/faces",
                    json={"x": 0.33, "y": 0.14, "w": 0.2, "h": 0.3, "person_id": anagha})
    assert r.status_code == 201, r.text
    assert r.json()["id"] != stored and r.json()["person"]["id"] == anagha
    assert _face(conn, stored)["person_id"] == paras


def test_the_turned_search_maps_faces_back(monkeypatch):
    """_faces_turned turns the padded region, and a face the engine finds in the middle of the turned copy maps back
    to the middle of the region, whatever the angle; the same face found at several angles is kept once."""
    from types import SimpleNamespace

    class Engine:
        def analyze(self, img, sizes, threshold, min_size):
            h, w = img.shape[:2]
            return [FakeFace(bbox=np.array((w / 2 - 20, h / 2 - 25, w / 2 + 20, h / 2 + 25), dtype=np.float32),
                             kps=np.array([[w / 2, h / 2]] * 5, dtype=np.float32), det_score=0.7)]

    monkeypatch.setattr(pipeline, "_model", lambda name: SimpleNamespace(get_engine=Engine) if name == "faces" else None)
    bgr = np.zeros((600, 800, 3), dtype=np.uint8)
    px = (300.0, 200.0, 500.0, 400.0)                          # padded by 15 %: (270, 170) to (530, 430), centre (400, 300)
    found = person_boxes._faces_turned(bgr, px)
    assert len(found) == 1
    cx, cy = (found[0].bbox[0] + found[0].bbox[2]) / 2, (found[0].bbox[1] + found[0].bbox[3]) / 2
    assert abs(cx - 400) < 1.5 and abs(cy - 300) < 1.5
    assert np.allclose(found[0].kps.mean(axis=0), (400, 300), atol=1.5)
    # without the face engine there is no turned search
    monkeypatch.setattr(pipeline, "_model", lambda name: None)
    assert person_boxes._faces_turned(bgr, px) == []
