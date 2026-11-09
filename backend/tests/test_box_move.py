"""Moving or resizing a face, person or pet box (section 10): the route, what a face box keeps or takes again,
undo and redo with the crop and the face index, the crop URL, and what a re-analysis keeps. The models are off
(PCC_NO_MODELS=1); the face search is faked where a face matters."""
from __future__ import annotations

import numpy as np
import pytest

from app import storage
from app.analysis import box_edit, person_boxes, pipeline
from app.analysis.index import get_index
from app.db import tx
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person
from tests.test_person_boxes import FakeFace, _body_record, _record, _unit

FACE = (0.3, 0.2, 0.2, 0.3)          # on an 800 x 600 photo: (240, 120) to (400, 300)
NUDGE = {"x": 0.32, "y": 0.22, "w": 0.18, "h": 0.28}
FAR = {"x": 0.65, "y": 0.6, "w": 0.2, "h": 0.3}


def _photo(conn, **kw):
    return make_item(conn, width=800, height=600, **kw)


def _row(conn, table: str, box_id: int):
    return conn.execute(f"SELECT * FROM {table} WHERE id = ?", (box_id,)).fetchone()


def _vec(row) -> np.ndarray:
    return np.frombuffer(row["embedding"], dtype=np.float32)


def _detector(monkeypatch, face: FakeFace | list[FakeFace] | None) -> None:
    """The face detector is loaded and finds `face` (or nothing) in the box."""
    monkeypatch.setattr(pipeline, "_model", lambda name: object() if name == "faces" else None)
    faces = face if isinstance(face, list) else [face] if face is not None else []
    monkeypatch.setattr(person_boxes, "_faces_in", lambda bgr, px: faces)


def _index_vec(conn, face_id: int) -> np.ndarray:
    idx = get_index()
    idx.sync(conn)
    return idx.embeddings_of([face_id])[0]


def test_a_face_box_moves_with_a_new_crop_and_keeps_its_name_and_vector_without_the_detector(client, conn):
    item = _photo(conn)
    asha = make_person(conn, "Asha")
    fid = make_face(conn, item["id"], asha, box=FACE, assigned_by="user", embedding=_unit(3))
    before = storage.face_crop_path(fid).read_bytes()
    url_before = client.get(f"/api/items/{item['uid']}").json()["faces"][0]["url"]

    r = client.put(f"/api/faces/{fid}/box", json=NUDGE)
    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["x"], out["y"], out["w"], out["h"]) == pytest.approx((0.32, 0.22, 0.18, 0.28))
    assert out["source"] == "user" and out["person"]["id"] == asha and out["assigned_by"] == "user"
    row = _row(conn, "faces", fid)
    assert row["source"] == "user" and np.allclose(_vec(row), _unit(3))   # no detector: the vector stays
    assert storage.face_crop_path(fid).read_bytes() != before
    assert out["url"].split("?")[0] == url_before.split("?")[0] and out["url"] != url_before   # a new crop, a new URL
    detail = client.get(f"/api/items/{item['uid']}").json()
    assert detail["faces"][0]["source"] == "user" and detail["faces"][0]["url"] == out["url"]
    faces = client.get(f"/api/people/{asha}/faces").json()["items"]
    assert faces[0]["source"] == "user"


def test_a_face_found_in_the_new_box_gives_the_new_vector(client, conn, monkeypatch):
    item = _photo(conn)
    fid = make_face(conn, item["id"], make_person(conn, "Asha"), box=FACE, embedding=_unit(3))
    _detector(monkeypatch, FakeFace(bbox=np.array((262.0, 140.0, 380.0, 280.0), dtype=np.float32), embedding=_unit(9)))
    r = client.put(f"/api/faces/{fid}/box", json=NUDGE)
    assert r.status_code == 200, r.text
    row = _row(conn, "faces", fid)
    assert np.allclose(_vec(row), _unit(9)) and row["det_score"] == pytest.approx(0.92)
    assert row["clarity"] == pytest.approx(24.5) and row["sharpness"] == pytest.approx(55.0)
    assert np.allclose(_index_vec(conn, fid), _unit(9))
    # the owner's box is stored, not the detector's
    assert (row["x"], row["w"]) == pytest.approx((0.32, 0.18))


def test_a_face_box_moved_where_no_face_is_is_refused_but_a_nudge_is_not(client, conn, monkeypatch):
    item = _photo(conn)
    fid = make_face(conn, item["id"], make_person(conn, "Asha"), box=FACE, embedding=_unit(3))
    _detector(monkeypatch, None)
    r = client.put(f"/api/faces/{fid}/box", json=FAR)
    assert r.status_code == 422 and r.json()["detail"] == "no_face"
    assert _row(conn, "faces", fid)["x"] == pytest.approx(0.3)
    # a face beside the box does not count either
    _detector(monkeypatch, FakeFace(bbox=np.array((10.0, 10.0, 60.0, 60.0), dtype=np.float32), embedding=_unit(9)))
    assert client.put(f"/api/faces/{fid}/box", json=FAR).status_code == 422
    r = client.put(f"/api/faces/{fid}/box", json=NUDGE)
    assert r.status_code == 200, r.text
    assert np.allclose(_vec(_row(conn, "faces", fid)), _unit(3))   # a nudge keeps the old vector


def test_undo_puts_the_box_its_crop_its_url_and_the_index_vector_back_and_redo_again(client, conn, monkeypatch):
    item = _photo(conn)
    fid = make_face(conn, item["id"], make_person(conn, "Asha"), box=FACE, embedding=_unit(3))
    assert np.allclose(_index_vec(conn, fid), _unit(3))
    crop_before = storage.face_crop_path(fid).read_bytes()
    url_before = client.get(f"/api/items/{item['uid']}").json()["faces"][0]["url"]
    _detector(monkeypatch, FakeFace(bbox=np.array((262.0, 140.0, 380.0, 280.0), dtype=np.float32), embedding=_unit(9)))
    r = client.put(f"/api/faces/{fid}/box", json=NUDGE)
    assert r.status_code == 200, r.text
    crop_moved, url_moved = storage.face_crop_path(fid).read_bytes(), r.json()["url"]
    assert np.allclose(_index_vec(conn, fid), _unit(9))

    u = client.post(f"/api/actions/{r.headers['x-action-id']}/undo")
    assert u.status_code == 200, u.text
    row = _row(conn, "faces", fid)
    assert (row["x"], row["y"], row["w"], row["h"]) == pytest.approx(FACE) and row["source"] == "detector"
    assert np.allclose(_vec(row), _unit(3)) and np.allclose(_index_vec(conn, fid), _unit(3))
    assert storage.face_crop_path(fid).read_bytes() == crop_before
    assert client.get(f"/api/items/{item['uid']}").json()["faces"][0]["url"] == url_before

    redo = client.post(f"/api/actions/{u.json()['undo_id']}/undo")
    assert redo.status_code == 200, redo.text
    row = _row(conn, "faces", fid)
    assert row["x"] == pytest.approx(0.32) and np.allclose(_vec(row), _unit(9))
    assert np.allclose(_index_vec(conn, fid), _unit(9))
    assert storage.face_crop_path(fid).read_bytes() == crop_moved
    assert client.get(f"/api/items/{item['uid']}").json()["faces"][0]["url"] == url_moved


def test_a_guess_box_keeps_its_guess_and_shows_no_owner(client, conn):
    item = _photo(conn)
    ravi = make_person(conn, "Ravi")
    fid = make_face(conn, item["id"], None, box=FACE)
    with tx(conn):
        conn.execute("UPDATE faces SET assigned_by = 'guess', guess_person_id = ?, guess_score = 0.8, "
                     "guess_evidence = '{}' WHERE id = ?", (ravi, fid))
    r = client.put(f"/api/faces/{fid}/box", json=NUDGE)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["person"] is None and out["guess"]["person"]["id"] == ravi and out["assigned_by"] == "guess"
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE type = 'guess'").fetchone()[0] == 0


def test_a_person_box_and_a_pet_box_move_and_become_the_owners(client, conn):
    item = _photo(conn)
    kabir = make_person(conn, "Kabir")
    with tx(conn):
        bid = int(conn.execute(
            "INSERT INTO bodies(item_id, person_id, assigned_by, x, y, w, h, det_score, created_at) "
            "VALUES (?, ?, 'auto', 0.1, 0.1, 0.3, 0.8, 0.8, ?)", (item["id"], kabir, now_iso())).lastrowid)
        aid = int(conn.execute(
            "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, created_at) "
            "VALUES (?, NULL, 'auto', 0.6, 0.5, 0.3, 0.4, 0.9, 'cat', ?)", (item["id"], now_iso())).lastrowid)
    for path in (storage.body_crop_path(bid), storage.animal_crop_path(aid)):
        storage.write_bytes_atomic(path, b"old crop")
    r = client.put(f"/api/bodies/{bid}/box", json={"x": 0.12, "y": 0.1, "w": 0.32, "h": 0.85})
    assert r.status_code == 200, r.text
    assert r.json()["source"] == "user" and r.json()["person"]["id"] == kabir and r.json()["assigned_by"] == "auto"
    assert storage.body_crop_path(bid).read_bytes() != b"old crop"
    r = client.put(f"/api/animals/{aid}/box", json={"x": 0.55, "y": 0.5, "w": 0.35, "h": 0.45})
    assert r.status_code == 200, r.text
    assert r.json()["source"] == "user" and r.json()["x"] == pytest.approx(0.55)
    assert storage.animal_crop_path(aid).read_bytes() != b"old crop"


def test_bad_boxes_missing_boxes_and_the_locked_vault(client, conn):
    item = _photo(conn)
    fid = make_face(conn, item["id"], None, box=FACE)
    assert client.put(f"/api/faces/{fid}/box", json={"x": 1.2, "y": 0.1, "w": 0.1, "h": 0.1}).status_code == 422
    r = client.put(f"/api/faces/{fid}/box", json={"x": 0.95, "y": 0.1, "w": 0.2, "h": 0.1})
    assert r.status_code == 422 and "inside" in r.json()["detail"]
    r = client.put(f"/api/faces/{fid}/box", json={"x": 0.3, "y": 0.2, "w": 0.01, "h": 0.01})   # 8 x 6 px
    assert r.status_code == 422 and "pixels" in r.json()["detail"]
    assert client.put("/api/faces/99999/box", json=NUDGE).status_code == 404
    assert client.put("/api/bodies/99999/box", json=NUDGE).status_code == 404
    # a tiny detector face may still be nudged when it does not get smaller
    tiny = make_face(conn, item["id"], None, box=(0.5, 0.5, 0.01, 0.01))
    assert client.put(f"/api/faces/{tiny}/box", json={"x": 0.51, "y": 0.5, "w": 0.01, "h": 0.01}).status_code == 200
    private = _photo(conn, nsfw=1)
    pf = make_face(conn, private["id"], None, box=FACE)
    r = client.put(f"/api/faces/{pf}/box", json=NUDGE)   # the vault is locked
    assert r.status_code == 403 and r.json()["detail"] == "vault_locked"


def test_check_box_clips_to_the_picture():
    assert box_edit.check_box((0.9, 0.9, 0.1004, 0.1), (0.9, 0.9, 0.1, 0.1), 800, 600) == pytest.approx((0.9, 0.9, 0.1, 0.1))
    with pytest.raises(ValueError):
        box_edit.check_box((0.5, 0.5, 0.0, 0.1), (0.5, 0.5, 0.1, 0.1), 800, 600)


def test_reanalysis_keeps_a_moved_photo_face_and_drops_its_twin(conn, client, monkeypatch):
    item = _photo(conn)
    asha = make_person(conn, "Asha")
    fid = make_face(conn, item["id"], asha, box=FACE, assigned_by="user")
    assert client.put(f"/api/faces/{fid}/box", json=NUDGE).status_code == 200
    with tx(conn):
        conn.execute("UPDATE items SET analysis_status = 'pending' WHERE id = ?", (item["id"],))
    twin = (0.31, 0.21, 0.19, 0.29)
    other = (0.7, 0.1, 0.1, 0.13)
    real_run = pipeline._run_models

    def run(item_row, v, siglip, nsfw, faces_engine, not_face=None):
        r = real_run(item_row, v, siglip, nsfw, faces_engine, not_face=not_face)
        r.faces = [_record(twin, 21), _record(other, 22)]
        r.face_torsos = [None, None]
        r.faces_ran = True
        r.bodies = [_body_record((0.6, 0.3, 0.3, 0.65))]
        r.body_torsos = [None]
        r.bodies_ran = True
        return r

    monkeypatch.setattr(pipeline, "_run_models", run)
    pipeline.analyze_item(conn, item["id"], 1)
    kept = _row(conn, "faces", fid)
    assert kept is not None and kept["x"] == pytest.approx(0.32) and kept["person_id"] == asha
    xs = sorted(round(r["x"], 2) for r in conn.execute("SELECT x FROM faces WHERE item_id = ?", (item["id"],)))
    assert xs == [0.32, 0.7]   # the twin over the moved face was dropped


def test_of_several_faces_the_box_means_the_one_it_overlaps_most(client, conn, monkeypatch):
    item = _photo(conn)
    fid = make_face(conn, item["id"], make_person(conn, "Asha"), box=FACE, embedding=_unit(3))
    own = FakeFace(bbox=np.array((262.0, 140.0, 380.0, 280.0), dtype=np.float32), det_score=0.7, embedding=_unit(9))
    # a neighbour's bigger, sharper face whose centre the enlarged box also takes in
    other = FakeFace(bbox=np.array((330.0, 150.0, 520.0, 330.0), dtype=np.float32), det_score=0.99, embedding=_unit(11))
    _detector(monkeypatch, [other, own])
    r = client.put(f"/api/faces/{fid}/box", json={"x": 0.3, "y": 0.2, "w": 0.3, "h": 0.32})
    assert r.status_code == 200, r.text
    assert np.allclose(_vec(_row(conn, "faces", fid)), _unit(9))


def test_an_enlarged_box_keeps_its_vector_when_the_face_is_not_found(client, conn, monkeypatch):
    item = _photo(conn)
    fid = make_face(conn, item["id"], make_person(conn, "Asha"), box=(0.4, 0.4, 0.1, 0.1), embedding=_unit(3))
    _detector(monkeypatch, None)
    r = client.put(f"/api/faces/{fid}/box", json={"x": 0.35, "y": 0.35, "w": 0.2, "h": 0.2})   # IoU 0.25
    assert r.status_code == 200, r.text
    assert np.allclose(_vec(_row(conn, "faces", fid)), _unit(3))
    assert box_edit.near_old((0.35, 0.35, 0.2, 0.2), (0.4, 0.4, 0.1, 0.1))
    assert not box_edit.near_old((0.7, 0.7, 0.2, 0.2), (0.4, 0.4, 0.1, 0.1))


def test_undo_swaps_a_person_box_and_a_pet_box_crop_back(client, conn):
    item = _photo(conn)
    with tx(conn):
        bid = int(conn.execute(
            "INSERT INTO bodies(item_id, assigned_by, x, y, w, h, det_score, created_at) "
            "VALUES (?, 'auto', 0.1, 0.1, 0.3, 0.8, 0.8, ?)", (item["id"], now_iso())).lastrowid)
        aid = int(conn.execute(
            "INSERT INTO animals(item_id, assigned_by, x, y, w, h, det_score, species, created_at) "
            "VALUES (?, 'auto', 0.6, 0.5, 0.3, 0.4, 0.9, 'cat', ?)", (item["id"], now_iso())).lastrowid)
    for path in (storage.body_crop_path(bid), storage.animal_crop_path(aid)):
        storage.write_bytes_atomic(path, b"old crop")
    for kind, box_id, path in (("bodies", bid, storage.body_crop_path(bid)), ("animals", aid, storage.animal_crop_path(aid))):
        r = client.put(f"/api/{kind}/{box_id}/box", json={"x": 0.12, "y": 0.1, "w": 0.3, "h": 0.4})
        assert r.status_code == 200, r.text
        moved = path.read_bytes()
        assert moved != b"old crop"
        u = client.post(f"/api/actions/{r.headers['x-action-id']}/undo")
        assert u.status_code == 200 and path.read_bytes() == b"old crop"
        assert _row(conn, kind, box_id)["source"] == "detector"
        assert client.post(f"/api/actions/{u.json()['undo_id']}/undo").status_code == 200
        assert path.read_bytes() == moved and _row(conn, kind, box_id)["source"] == "user"


def test_reanalysis_gives_a_body_around_the_twin_the_moved_face(conn, client, monkeypatch):
    item = _photo(conn)
    fid = make_face(conn, item["id"], make_person(conn, "Asha"), box=FACE, assigned_by="user")
    assert client.put(f"/api/faces/{fid}/box", json=NUDGE).status_code == 200
    with tx(conn):
        conn.execute("UPDATE items SET analysis_status = 'pending' WHERE id = ?", (item["id"],))
    real_run = pipeline._run_models

    def run(item_row, v, siglip, nsfw, faces_engine, not_face=None):
        r = real_run(item_row, v, siglip, nsfw, faces_engine, not_face=not_face)
        r.faces = [_record((0.31, 0.21, 0.19, 0.29), 21)]
        r.face_torsos = [None]
        r.faces_ran = True
        body = _body_record((0.25, 0.15, 0.35, 0.8))
        body.face_index = 0   # the detector's body around the twin face
        r.bodies = [body]
        r.body_torsos = [None]
        r.bodies_ran = True
        return r

    monkeypatch.setattr(pipeline, "_run_models", run)
    pipeline.analyze_item(conn, item["id"], 1)
    bodies = conn.execute("SELECT face_id FROM bodies WHERE item_id = ?", (item["id"],)).fetchall()
    assert [b["face_id"] for b in bodies] == [fid]   # around the kept face, not a faceless box
    assert [r["id"] for r in conn.execute("SELECT id FROM faces WHERE item_id = ?", (item["id"],))] == [fid]
