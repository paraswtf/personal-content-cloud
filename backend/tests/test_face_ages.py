"""Face ages (section 10): the FairFace age groups of every face, stored with it, read from the in-memory crop when a
face is found, drawn or moved, and from the stored crop for the faces that came before the model. The models are
off here (PCC_NO_MODELS=1); the age model is faked."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.analysis import ages, face_upkeep, person_boxes, pipeline
from app.db import faces_version, tx
from app.ml import models
from tests.helpers import make_face, make_item, make_person
from tests.test_face_boxes import BOX, FOUND, _detector
from tests.test_person_boxes import FakeFace, _record, _unit

CHILD = np.array([0.30, 0.55, 0.10, 0.03, 0.01, 0.01, 0.0, 0.0, 0.0], dtype=np.float32)
ADULT = np.array([0.0, 0.01, 0.04, 0.40, 0.40, 0.10, 0.05, 0.0, 0.0], dtype=np.float32)


def _fake_model(monkeypatch, groups: np.ndarray = CHILD) -> list[int]:
    """The age model, answering `groups` for every crop; returns the list of batch sizes it saw."""
    seen: list[int] = []

    def predict(crops):
        seen.append(len(crops))
        return np.tile(groups, (len(crops), 1))

    monkeypatch.setattr(ages, "_engine", lambda: SimpleNamespace(predict=predict))
    return seen


def _age(conn, face_id: int):
    return ages.decode(conn.execute("SELECT age_groups FROM faces WHERE id = ?", (face_id,)).fetchone()[0])


def test_what_the_groups_say():
    assert ages.decode(ages.encode(CHILD)) == pytest.approx(CHILD)
    assert ages.youth(CHILD) == pytest.approx(0.85) and ages.youth(ages.encode(ADULT)) == pytest.approx(0.01)
    assert ages.youth(None) is None and ages.decode(b"") is None and ages.decode(b"\x00" * 8) is None
    assert ages.expected_age(CHILD) == pytest.approx(0.3 + 3.3 + 1.5 + 0.75 + 0.35 + 0.45, abs=0.01)
    assert (ages.band(CHILD), ages.band(ADULT)) == ("child", "adult")
    assert ages.band(np.array([0.6, 0.3, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)) == "baby"


def test_without_the_model_nothing_is_measured(conn):
    assert ages.available() is False
    assert ages.measure([object(), object()]) == [None, None]
    item = make_item(conn, write_files=False)
    make_face(conn, item["id"])
    assert ages.measure_stored_faces(conn) == {"faces": 0, "complete": False}


def test_the_age_model_is_registered():
    spec = models.REGISTRY["ages"]
    assert spec.subdir == "ages" and spec.license == "Apache 2.0"
    assert {f.rel for f in spec.files} == {"onnx/model_int8.onnx", "config.json", "preprocessor_config.json"}
    assert all(f.url and "fd04cddf04982f6acc1ac76fb1a31b44c9285596" in f.url for f in spec.files)


def test_stored_faces_get_their_ages_from_their_crops(conn, monkeypatch):
    seen = _fake_model(monkeypatch)
    item = make_item(conn, write_files=False)
    a = make_face(conn, item["id"])
    b = make_face(conn, item["id"], write_crop=False)  # no crop file: skipped, tried again next time
    c = make_face(conn, item["id"])
    with tx(conn):
        conn.execute("UPDATE faces SET age_groups = ? WHERE id = ?", (ages.encode(ADULT), c))
    before = faces_version(conn)
    assert ages.measure_stored_faces(conn) == {"faces": 1, "complete": True}
    assert _age(conn, a) == pytest.approx(CHILD) and _age(conn, b) is None and _age(conn, c) == pytest.approx(ADULT)
    assert faces_version(conn) > before and seen == [1] and ages.ready(conn)
    assert ages.measure_stored_faces(conn) == {"faces": 0, "complete": True}


def test_a_measured_or_empty_library_is_marked_ready_at_startup(conn, monkeypatch):
    _fake_model(monkeypatch)
    started: list[str] = []
    from app import worker

    monkeypatch.setattr(worker, "run_in_background", lambda key, fn, *a: started.append(key))
    assert not ages.ready(conn)
    before = faces_version(conn)
    face_upkeep.start(conn)
    assert ages.ready(conn) and "face-ages" not in started and faces_version(conn) > before


def test_startup_reads_the_ages_on_a_background_thread(conn, monkeypatch):
    _fake_model(monkeypatch)
    started: list[str] = []
    from app import worker

    monkeypatch.setattr(worker, "run_in_background", lambda key, fn, *a: started.append(key))
    item = make_item(conn, write_files=False)
    make_face(conn, item["id"])
    face_upkeep.start(conn)
    assert "face-ages" in started


def test_a_model_failure_stops_the_pass_and_keeps_the_faces(conn, monkeypatch):
    def predict(crops):
        raise RuntimeError("broken")

    monkeypatch.setattr(ages, "_engine", lambda: SimpleNamespace(predict=predict))
    item = make_item(conn, write_files=False)
    fid = make_face(conn, item["id"])
    assert ages.measure_stored_faces(conn) == {"faces": 0, "complete": False}
    assert _age(conn, fid) is None


def test_new_faces_of_an_analysis_get_their_ages(conn, monkeypatch):
    _fake_model(monkeypatch)
    item = make_item(conn, width=800, height=600)
    with tx(conn):
        conn.execute("UPDATE items SET analysis_status = 'pending' WHERE id = ?", (item["id"],))
    real_run = pipeline._run_models

    def run(item_row, v, siglip, nsfw, faces_engine, not_face=None):
        r = real_run(item_row, v, siglip, nsfw, faces_engine, not_face=not_face)
        r.faces = [_record((0.3, 0.2, 0.2, 0.3), 41), _record((0.6, 0.2, 0.15, 0.2), 42)]
        r.face_torsos = [None, None]
        r.faces_ran = True
        return r

    monkeypatch.setattr(pipeline, "_run_models", run)
    pipeline.analyze_item(conn, item["id"], 1)
    rows = conn.execute("SELECT id FROM faces WHERE item_id = ? ORDER BY id", (item["id"],)).fetchall()
    assert len(rows) == 2 and all(_age(conn, int(r[0])) == pytest.approx(CHILD) for r in rows)


def test_drawn_and_moved_face_boxes_get_the_age_of_their_crop(client, conn, monkeypatch):
    _fake_model(monkeypatch, CHILD)
    item = make_item(conn, width=800, height=600)
    _detector(monkeypatch, FakeFace(bbox=np.array(FOUND, dtype=np.float32), embedding=_unit(9)))
    r = client.post(f"/api/items/{item['uid']}/faces", json={**BOX, "new_person_name": "Meera"})
    assert r.status_code == 201, r.text
    fid = r.json()["id"]
    assert _age(conn, fid) == pytest.approx(CHILD)
    # moved: the new crop's age; undo gives the old one back; without the model the old age does not stay
    _fake_model(monkeypatch, ADULT)
    m = client.put(f"/api/faces/{fid}/box", json={"x": 0.31, "y": 0.21, "w": 0.2, "h": 0.3})
    assert m.status_code == 200, m.text
    assert _age(conn, fid) == pytest.approx(ADULT)
    assert client.post(f"/api/actions/{m.headers['x-action-id']}/undo").status_code == 200
    assert _age(conn, fid) == pytest.approx(CHILD)
    monkeypatch.setattr(ages, "_engine", lambda: None)
    assert client.put(f"/api/faces/{fid}/box", json={"x": 0.31, "y": 0.21, "w": 0.2, "h": 0.3}).status_code == 200
    assert _age(conn, fid) is None


def test_a_person_box_with_a_face_inside_gives_the_face_its_age(client, conn, monkeypatch):
    _fake_model(monkeypatch, ADULT)
    item = make_item(conn, width=800, height=600)
    asha = make_person(conn, "Asha")
    monkeypatch.setattr(person_boxes, "_face_in", lambda bgr, px: FakeFace(bbox=np.array(FOUND, dtype=np.float32)))
    r = client.post(f"/api/items/{item['uid']}/people-boxes",
                    json={"x": 0.25, "y": 0.15, "w": 0.35, "h": 0.8, "person_id": asha})
    assert r.status_code in (200, 201), r.text
    fid = conn.execute("SELECT id FROM faces WHERE item_id = ?", (item["id"],)).fetchone()[0]
    assert _age(conn, int(fid)) == pytest.approx(ADULT)


def test_a_later_pass_refreshes_the_index_and_one_bad_crop_does_not_block_it(conn, monkeypatch):
    item = make_item(conn, write_files=False)
    good = make_face(conn, item["id"])
    bad = make_face(conn, item["id"])
    bad_path = str(__import__("app.storage", fromlist=["x"]).face_crop_path(bad))

    def predict(crops):
        if any(getattr(c, "filename", "") == bad_path for c in crops):
            raise RuntimeError("cannot read this one")
        return np.tile(CHILD, (len(crops), 1))

    monkeypatch.setattr(ages, "_engine", lambda: SimpleNamespace(predict=predict))
    monkeypatch.setattr(ages, "_open_crop", lambda path: SimpleNamespace(filename=str(path)))
    with tx(conn):
        ages.mark_ready(conn)  # the library was ready before; these faces came while the model was missing
    before = faces_version(conn)
    assert ages.measure_stored_faces(conn) == {"faces": 1, "complete": True}
    assert _age(conn, good) == pytest.approx(CHILD) and _age(conn, bad) is None
    assert faces_version(conn) > before

