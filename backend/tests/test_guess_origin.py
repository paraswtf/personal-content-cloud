"""A guess the owner confirms stays the system's find (section 10): the box is locked as the owner's (assigned_by
'user') but marked by_guess, so the apps show the bot; a tag from a confirmed faceless guess too; a later hand
assignment makes it the owner's; old confirmations are marked once from the guess learner's log; a re-analysis
keeps the mark."""
from __future__ import annotations

import pytest

from app import db as db_mod
from app.analysis import pipeline
from app.db import tx
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person
from tests.test_person_boxes import _record, _unit


def _guess(conn, table: str, box_id: int, person: int) -> None:
    with tx(conn):
        conn.execute(f"UPDATE {table} SET assigned_by = 'guess', guess_person_id = ?, guess_score = 0.8, "
                     "guess_evidence = '{}' WHERE id = ?", (person, box_id))


def _body(conn, item_id: int, face_id: int | None = None) -> int:
    with tx(conn):
        return int(conn.execute(
            "INSERT INTO bodies(item_id, face_id, assigned_by, x, y, w, h, det_score, created_at) "
            "VALUES (?, ?, 'auto', 0.1, 0.1, 0.3, 0.8, 0.8, ?)", (item_id, face_id, now_iso())).lastrowid)


def test_a_confirmed_face_guess_is_locked_but_stays_the_systems_find(client, conn):
    item = make_item(conn, write_files=False)
    ravi = make_person(conn, "Ravi")
    fid = make_face(conn, item["id"], None, write_crop=False)
    _guess(conn, "faces", fid, ravi)
    r = client.post("/api/guesses/confirm", json={"faces": [fid], "bodies": []})
    assert r.status_code == 200 and r.json()["updated"] == 1
    face = client.get(f"/api/items/{item['uid']}").json()["faces"][0]
    assert face["person"]["id"] == ravi and face["assigned_by"] == "user" and face["by_guess"] is True
    assert client.get(f"/api/people/{ravi}/faces").json()["items"][0]["by_guess"] is True
    people = client.post("/api/items/people", json={"uids": [item["uid"]]}).json()["people"]
    assert (people[0]["items"], people[0]["by_you"]) == (1, 0)
    # the owner's own say makes it theirs
    r = client.post("/api/faces/assign", json={"face_ids": [fid], "person_id": ravi})
    assert r.status_code == 200, r.text
    assert client.get(f"/api/items/{item['uid']}").json()["faces"][0]["by_guess"] is False
    people = client.post("/api/items/people", json={"uids": [item["uid"]]}).json()["people"]
    assert people[0]["by_you"] == 1


def test_a_confirmed_faceless_body_and_its_tag_stay_the_systems_find(client, conn):
    item = make_item(conn, write_files=False)
    asha = make_person(conn, "Asha")
    bid = _body(conn, item["id"])
    _guess(conn, "bodies", bid, asha)
    assert client.post("/api/guesses/confirm", json={"faces": [], "bodies": [bid]}).json()["updated"] == 1
    detail = client.get(f"/api/items/{item['uid']}").json()
    assert detail["bodies"][0]["by_guess"] is True
    assert [(t["id"], t["by_guess"]) for t in detail["tags"]] == [(asha, True)]
    people = client.post("/api/items/people", json={"uids": [item["uid"]]}).json()["people"]
    assert (people[0]["items"], people[0]["by_you"]) == (1, 0)
    # a tag the owner adds by hand is theirs
    other = make_item(conn, write_files=False)
    assert client.post(f"/api/items/{other['uid']}/tags", json={"person_id": asha}).status_code in (200, 201)
    assert client.get(f"/api/items/{other['uid']}").json()["tags"][0]["by_guess"] is None


def test_old_confirmations_are_marked_once_from_the_learners_log(conn):
    item = make_item(conn, write_files=False)
    ravi = make_person(conn, "Ravi")
    confirmed = make_face(conn, item["id"], ravi, assigned_by="user", write_crop=False)
    elsewhere = make_item(conn, write_files=False)
    by_hand = make_face(conn, elsewhere["id"], ravi, assigned_by="user", write_crop=False)
    with tx(conn):
        conn.execute("INSERT INTO learning_events(kind, label, features, item_id, person_id, created_at) "
                     "VALUES ('guess', 1, '{}', ?, ?, ?)", (item["id"], ravi, now_iso()))
        conn.execute("UPDATE kv SET value = '0' WHERE key = 'guess_marks_v1'")
        db_mod._mark_confirmed_guesses(conn)
    flags = dict(conn.execute("SELECT id, by_guess FROM faces").fetchall())
    assert flags == {confirmed: 1, by_hand: 0}
    assert db_mod.kv_get(conn, "guess_marks_v1") == "1"


def test_a_reanalysis_keeps_the_mark(conn, client, monkeypatch):
    item = make_item(conn, width=800, height=600)
    ravi = make_person(conn, "Ravi")
    fid = make_face(conn, item["id"], None, box=(0.3, 0.2, 0.2, 0.3), embedding=_unit(31))
    _guess(conn, "faces", fid, ravi)
    client.post("/api/guesses/confirm", json={"faces": [fid], "bodies": []})
    with tx(conn):
        conn.execute("UPDATE items SET analysis_status = 'pending' WHERE id = ?", (item["id"],))
    real_run = pipeline._run_models

    def run(item_row, v, siglip, nsfw, faces_engine, not_face=None):
        r = real_run(item_row, v, siglip, nsfw, faces_engine, not_face=not_face)
        r.faces = [_record((0.305, 0.205, 0.2, 0.3), 31)]
        r.face_torsos = [None]
        r.faces_ran = True
        return r

    monkeypatch.setattr(pipeline, "_run_models", run)
    pipeline.analyze_item(conn, item["id"], 1)
    rows = conn.execute("SELECT id, person_id, assigned_by, by_guess FROM faces WHERE item_id = ?", (item["id"],)).fetchall()
    assert [tuple(r)[1:] for r in rows] == [(ravi, "user", 1)] and rows[0]["id"] != fid   # carried to the new face
