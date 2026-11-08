"""Person boxes the owner draws (section 10, like the pet boxes of section 22): the route, the face the
detector may find inside the box, and what a re-analysis keeps. The models are off here (PCC_NO_MODELS=1);
the face search is faked where a face matters."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from PIL import Image

from app import storage
from app.analysis import faces as faces_mod
from app.analysis import person_boxes, pipeline
from app.analysis.bodies import BodyRecord
from app.db import tx
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person

BOX = {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.8}   # on an 800 x 600 photo: (80, 60) to (400, 540)
FACE_PX = (180.0, 80.0, 300.0, 220.0)            # a face inside it, in the photo's pixels
FACE_BOX = (180 / 800, 80 / 600, 120 / 800, 140 / 600)


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@dataclass
class FakeFace:
    """What app.ml.faces would hand back for a face inside the drawn box."""

    bbox: np.ndarray
    det_score: float = 0.92
    embedding: np.ndarray = field(default_factory=lambda: _unit(7))
    norm: float = 24.5
    sharpness: float = 55.0
    kps: np.ndarray = field(default_factory=lambda: np.zeros((5, 2), dtype=np.float32))

    @property
    def size_px(self) -> int:
        return int(round(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])))


def _found(monkeypatch, face: FakeFace | None) -> None:
    monkeypatch.setattr(person_boxes, "_face_in", lambda bgr, px: face)


def _photo(conn, **kw):
    return make_item(conn, width=800, height=600, **kw)


def _body(conn, body_id: int):
    return conn.execute("SELECT * FROM bodies WHERE id = ?", (body_id,)).fetchone()


def _face(conn, face_id: int):
    return conn.execute("SELECT * FROM faces WHERE id = ?", (face_id,)).fetchone()


# ------------------------------------------------------------------ the route
def test_a_drawn_box_with_a_new_name_holds_the_person(client, conn):
    item = _photo(conn)
    r = client.post(f"/api/items/{item['uid']}/people-boxes", json=dict(BOX, new_person_name="Meera"))
    assert r.status_code == 201, r.text
    out = r.json()
    body = out["body"]
    assert out["face"] is None   # no model here: nothing found inside the box
    assert body["source"] == "user" and body["assigned_by"] == "user" and body["det_score"] == 1.0
    assert body["person"]["name"] == "Meera" and body["face_id"] is None
    assert (body["x"], body["y"], body["w"], body["h"]) == pytest.approx((0.1, 0.1, 0.4, 0.8))
    row = _body(conn, body["id"])
    assert row["source"] == "user" and row["person_id"] == body["person"]["id"] and row["frame_time_s"] is None
    assert storage.body_crop_path(body["id"]).is_file()
    assert conn.execute("SELECT body_count, face_count FROM items WHERE id = ?", (item["id"],)).fetchone()[:] == (1, 0)
    detail = client.get(f"/api/items/{item['uid']}").json()
    assert [(p["name"], p["relation"]) for p in detail["people"]] == [("Meera", "in")]
    assert [(b["id"], b["source"]) for b in detail["bodies"]] == [(body["id"], "user")]


def test_a_drawn_box_takes_an_existing_person(client, conn):
    item = _photo(conn)
    ravi = make_person(conn, "Ravi")
    r = client.post(f"/api/items/{item['uid']}/people-boxes", json=dict(BOX, person_id=ravi))
    assert r.status_code == 201, r.text
    assert r.json()["body"]["person"]["id"] == ravi
    assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 1


def test_a_drawn_box_needs_a_person(client, conn):
    item = _photo(conn)
    for extra in ({}, {"new_person_name": "   "}):
        r = client.post(f"/api/items/{item['uid']}/people-boxes", json=dict(BOX, **extra))
        assert r.status_code == 400 and r.json()["detail"] == "person_needed", (extra, r.text)
    assert conn.execute("SELECT COUNT(*) FROM bodies").fetchone()[0] == 0


def test_a_video_box_needs_the_frame_time(client, conn):
    clip = make_item(conn, kind="video", ext=".mp4", mime="video/mp4", write_files=False, data=None)
    r = client.post(f"/api/items/{clip['uid']}/people-boxes", json=dict(BOX, new_person_name="Meera"))
    assert r.status_code == 400 and r.json()["detail"] == "frame_time_needed"
    assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 0


def test_bad_boxes_other_kinds_and_missing_items_are_refused(client, conn):
    item = _photo(conn)
    for box in (dict(BOX, x=0.7, w=0.4), dict(BOX, w=0.05, h=0.05)):
        r = client.post(f"/api/items/{item['uid']}/people-boxes", json=dict(box, new_person_name="Meera"))
        assert r.status_code == 422, (box, r.text)
    doc = make_item(conn, kind="pdf", ext=".pdf", mime="application/pdf", write_files=False, data=None)
    assert client.post(f"/api/items/{doc['uid']}/people-boxes",
                       json=dict(BOX, new_person_name="Meera")).status_code == 400
    assert client.post("/api/items/nope12345678/people-boxes",
                       json=dict(BOX, new_person_name="Meera")).status_code == 404
    assert conn.execute("SELECT COUNT(*) FROM bodies").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 0


# ------------------------------------------------------------------ a face inside the box
def test_a_face_inside_the_box_becomes_the_persons_face(client, conn, monkeypatch):
    found = FakeFace(bbox=np.array(FACE_PX, dtype=np.float32))
    _found(monkeypatch, found)
    item = _photo(conn)
    r = client.post(f"/api/items/{item['uid']}/people-boxes", json=dict(BOX, new_person_name="Meera"))
    assert r.status_code == 201, r.text
    out = r.json()
    face = out["face"]
    assert face is not None and face["assigned_by"] == "user" and face["person"]["name"] == "Meera"
    assert out["body"]["face_id"] == face["id"]
    row = _face(conn, face["id"])
    assert row["person_id"] == out["body"]["person"]["id"] and row["assigned_by"] == "user"
    assert row["clarity"] == pytest.approx(24.5) and row["sharpness"] == pytest.approx(55.0)
    assert (row["x"], row["y"], row["w"], row["h"]) == pytest.approx(FACE_BOX, abs=1e-4)
    assert row["size_px"] == 120 and row["frame_time_s"] is None
    assert np.allclose(np.frombuffer(row["embedding"], dtype=np.float32), found.embedding)
    assert storage.face_crop_path(face["id"]).is_file()
    assert conn.execute("SELECT face_count, body_count FROM items WHERE id = ?", (item["id"],)).fetchone()[:] == (1, 1)


def test_a_stored_face_under_the_box_is_used_and_assigned(client, conn, monkeypatch):
    _found(monkeypatch, FakeFace(bbox=np.array(FACE_PX, dtype=np.float32)))
    item = _photo(conn)
    stored = make_face(conn, item["id"], None, box=FACE_BOX, embedding=_unit(9))
    ravi = make_person(conn, "Ravi")
    r = client.post(f"/api/items/{item['uid']}/people-boxes", json=dict(BOX, person_id=ravi))
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["face"]["id"] == stored and out["body"]["face_id"] == stored
    row = _face(conn, stored)
    assert row["person_id"] == ravi and row["assigned_by"] == "user"
    assert conn.execute("SELECT COUNT(*) FROM faces WHERE item_id = ?", (item["id"],)).fetchone()[0] == 1


def test_add_box_reports_the_stored_face(conn, monkeypatch):
    _found(monkeypatch, FakeFace(bbox=np.array(FACE_PX, dtype=np.float32)))
    item = _photo(conn)
    stored = make_face(conn, item["id"], None, box=FACE_BOX)
    pid = make_person(conn, "Ravi")
    drawn = person_boxes.add_box(conn, item, (0.1, 0.1, 0.4, 0.8), pid)
    assert drawn.face_id == stored and drawn.existing_face is True
    assert _body(conn, drawn.body_id)["face_id"] == stored


def test_a_drawn_box_is_deleted_like_a_person_box(client, conn):
    item = _photo(conn)
    body = client.post(f"/api/items/{item['uid']}/people-boxes", json=dict(BOX, new_person_name="Meera")).json()["body"]
    assert client.delete(f"/api/bodies/{body['id']}").status_code == 200
    assert _body(conn, body["id"]) is None
    assert not storage.body_crop_path(body["id"]).exists()
    assert conn.execute("SELECT body_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 0


# ------------------------------------------------------------------ re-analysis
def _record(box, seed: int) -> faces_mod.FaceRecord:
    return faces_mod.FaceRecord(
        x=box[0], y=box[1], w=box[2], h=box[3], det_score=0.9, size_px=100, quality=0.56, frame_time_s=None,
        embedding=_unit(seed), crop=Image.new("RGB", (192, 192), (90, 90, 90)), clarity=23.0, sharpness=45.0,
    )


def _body_record(box) -> BodyRecord:
    return BodyRecord(x=box[0], y=box[1], w=box[2], h=box[3], det_score=0.8, face_index=None, torso=None,
                      crop=Image.new("RGB", (64, 128), (60, 60, 60)))


def test_reanalysis_keeps_the_drawn_box_and_its_face_and_drops_their_twins(conn, monkeypatch):
    _found(monkeypatch, FakeFace(bbox=np.array(FACE_PX, dtype=np.float32)))
    item = _photo(conn)
    pid = make_person(conn, "Meera")
    drawn = person_boxes.add_box(conn, item, (0.1, 0.1, 0.4, 0.8), pid)
    old_face = make_face(conn, item["id"], None, box=(0.62, 0.1, 0.12, 0.16))
    with tx(conn):
        old_body = int(conn.execute(
            "INSERT INTO bodies(item_id, assigned_by, x, y, w, h, det_score, created_at) "
            "VALUES (?, 'auto', 0.6, 0.05, 0.3, 0.9, 0.7, ?)", (item["id"], now_iso())).lastrowid)
        conn.execute("UPDATE items SET analysis_status = 'pending' WHERE id = ?", (item["id"],))

    twin_face_box = (FACE_BOX[0] + 0.005, FACE_BOX[1] + 0.005, FACE_BOX[2], FACE_BOX[3])
    new_face_box = (0.7, 0.55, 0.1, 0.13)
    twin_body_box = (0.11, 0.1, 0.39, 0.8)
    new_body_box = (0.55, 0.3, 0.3, 0.65)
    real_run = pipeline._run_models

    def run(item_row, v, siglip, nsfw, faces_engine, not_face=None):
        r = real_run(item_row, v, siglip, nsfw, faces_engine, not_face=not_face)
        r.faces = [_record(twin_face_box, 21), _record(new_face_box, 22)]
        r.face_torsos = [None, None]
        r.faces_ran = True
        r.bodies = [_body_record(twin_body_box), _body_record(new_body_box)]
        r.body_torsos = [None, None]
        r.bodies_ran = True
        return r

    monkeypatch.setattr(pipeline, "_run_models", run)
    pipeline.analyze_item(conn, item["id"], 1)

    assert conn.execute("SELECT analysis_status FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == "done"
    kept_body = _body(conn, drawn.body_id)
    assert kept_body is not None and kept_body["source"] == "user" and kept_body["person_id"] == pid
    assert kept_body["face_id"] == drawn.face_id
    kept_face = _face(conn, drawn.face_id)
    assert kept_face is not None and kept_face["person_id"] == pid and kept_face["assigned_by"] == "user"
    assert _face(conn, old_face) is None and _body(conn, old_body) is None
    faces = conn.execute("SELECT id, x FROM faces WHERE item_id = ? ORDER BY id", (item["id"],)).fetchall()
    assert len(faces) == 2 and faces[0]["id"] == drawn.face_id and faces[1]["x"] == pytest.approx(new_face_box[0])
    bodies = conn.execute("SELECT id, x, source FROM bodies WHERE item_id = ? ORDER BY id", (item["id"],)).fetchall()
    assert [(b["source"], round(b["x"], 3)) for b in bodies] == [("user", 0.1), ("detector", new_body_box[0])]
    assert conn.execute("SELECT face_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 2


def test_drop_over_drawn_takes_only_the_twins(conn):
    item = _photo(conn)
    with tx(conn):
        ids = [int(conn.execute(
            "INSERT INTO bodies(item_id, assigned_by, x, y, w, h, det_score, created_at) VALUES (?, 'auto', ?, ?, ?, ?, 0.8, ?)",
            (item["id"], *box, now_iso())).lastrowid) for box in ((0.11, 0.1, 0.4, 0.8), (0.6, 0.1, 0.3, 0.8))]
        gone = person_boxes.drop_over_drawn(conn, "bodies", ids, [(0.1, 0.1, 0.4, 0.8)])
        assert person_boxes.drop_over_drawn(conn, "animals", ids, [(0.1, 0.1, 0.4, 0.8)]) == []
        assert person_boxes.drop_over_drawn(conn, "bodies", [], [(0.1, 0.1, 0.4, 0.8)]) == []
    assert gone == [ids[0]]
    assert [int(r[0]) for r in conn.execute("SELECT id FROM bodies WHERE item_id = ?", (item["id"],))] == [ids[1]]


def test_protected_faces_are_those_of_drawn_boxes(conn):
    item = _photo(conn)
    under_drawn = make_face(conn, item["id"], None, box=(0.15, 0.12, 0.1, 0.12))
    under_detected = make_face(conn, item["id"], None, box=(0.65, 0.12, 0.1, 0.12))
    with tx(conn):
        for source, face_id in (("user", under_drawn), ("user", None), ("detector", under_detected)):
            conn.execute(
                "INSERT INTO bodies(item_id, face_id, assigned_by, x, y, w, h, det_score, created_at, source) "
                "VALUES (?, ?, 'user', 0.1, 0.1, 0.3, 0.6, 1.0, ?, ?)", (item["id"], face_id, now_iso(), source))
    rows = conn.execute("SELECT * FROM bodies WHERE item_id = ? ORDER BY id", (item["id"],)).fetchall()
    drawn = person_boxes.drawn_body_rows(rows)
    assert [r["source"] for r in drawn] == ["user", "user"]
    assert person_boxes.protected_face_ids(drawn) == {under_drawn}
