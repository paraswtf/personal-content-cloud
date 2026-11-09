"""Correcting a guess in one step (sections 10 and 18): "Someone else" learns the guess as wrong (a 'no' for the guess
learner and a rejection, so it is not guessed again) and gives the box to the right person as the owner's choice, in
one action that one undo takes back. A pick that is the guessed person is a yes."""
from __future__ import annotations

import json

from tests.helpers import make_face, make_item, make_person
from tests.test_guess_origin import _body, _guess


def _row(conn, table: str, box_id: int):
    return conn.execute(f"SELECT * FROM {table} WHERE id = ?", (box_id,)).fetchone()


def _events(conn):
    return [(r["label"], r["person_id"]) for r in conn.execute(
        "SELECT label, person_id FROM learning_events WHERE kind = 'guess' ORDER BY id")]


def _rejected(conn, kind: str, box_id: int) -> set[int]:
    return {int(r[0]) for r in conn.execute(
        "SELECT person_id FROM guess_rejections WHERE kind = ? AND ref_id = ?", (kind, box_id))}


def test_a_face_guess_corrected_to_someone_else(client, conn):
    item = make_item(conn, write_files=False)
    ravi, asha = make_person(conn, "Ravi"), make_person(conn, "Asha")
    fid = make_face(conn, item["id"], None, write_crop=False)
    _guess(conn, "faces", fid, ravi)
    r = client.post("/api/guesses/correct", json={"faces": [fid], "person_id": asha})
    assert r.status_code == 200, r.text
    assert r.json()["updated"] == 1 and r.json()["person"]["id"] == asha
    row = _row(conn, "faces", fid)
    assert (row["person_id"], row["assigned_by"], row["by_guess"], row["guess_person_id"]) == (asha, "user", 0, None)
    assert _events(conn) == [(0, ravi)]              # the guess learner hears "no", never a "yes"
    assert _rejected(conn, "face", fid) == {ravi}     # not guessed as Ravi again
    face = client.get(f"/api/items/{item['uid']}").json()["faces"][0]
    assert face["person"]["id"] == asha and face["by_guess"] is False


def test_a_body_guess_corrected_to_a_new_name_and_undone(client, conn):
    item = make_item(conn, write_files=False)
    ravi = make_person(conn, "Ravi")
    bid = _body(conn, item["id"])
    _guess(conn, "bodies", bid, ravi)
    r = client.post("/api/guesses/correct", json={"bodies": [bid], "new_person_name": "Meera"})
    assert r.status_code == 200, r.text
    meera = r.json()["person"]["id"]
    assert r.json()["person"]["name"] == "Meera" and _row(conn, "bodies", bid)["person_id"] == meera
    assert _events(conn) == [(0, ravi)] and _rejected(conn, "body", bid) == {ravi}
    # one action: one undo puts the open guess back, without the "no"
    u = client.post(f"/api/actions/{r.headers['x-action-id']}/undo")
    assert u.status_code == 200, u.text
    row = _row(conn, "bodies", bid)
    assert (row["person_id"], row["assigned_by"], row["guess_person_id"]) == (None, "guess", ravi)
    assert _events(conn) == [] and _rejected(conn, "body", bid) == set()


def test_picking_the_guessed_person_is_a_yes(client, conn):
    item = make_item(conn, write_files=False)
    ravi = make_person(conn, "Ravi")
    fid = make_face(conn, item["id"], None, write_crop=False)
    _guess(conn, "faces", fid, ravi)
    r = client.post("/api/guesses/correct", json={"faces": [fid], "person_id": ravi})
    assert r.status_code == 200 and r.json()["updated"] == 1
    row = _row(conn, "faces", fid)
    assert (row["person_id"], row["assigned_by"], row["by_guess"]) == (ravi, "user", 1)
    assert _events(conn) == [(1, ravi)] and _rejected(conn, "face", fid) == set()


def test_mixed_boxes_and_bad_requests(client, conn):
    item = make_item(conn, write_files=False)
    ravi, asha = make_person(conn, "Ravi"), make_person(conn, "Asha")
    guessed = make_face(conn, item["id"], None, write_crop=False)
    _guess(conn, "faces", guessed, ravi)
    plain = make_face(conn, make_item(conn, write_files=False)["id"], None, write_crop=False)   # no guess at all
    r = client.post("/api/guesses/correct", json={"faces": [guessed, plain], "person_id": asha})
    assert r.status_code == 200 and r.json()["updated"] == 2
    assert {_row(conn, "faces", f)["person_id"] for f in (guessed, plain)} == {asha}
    assert _events(conn) == [(0, ravi)]
    assert client.post("/api/guesses/correct", json={"faces": [guessed]}).status_code == 400
    assert client.post("/api/guesses/correct", json={"person_id": asha}).status_code == 400
    assert client.post("/api/guesses/correct", json={"faces": [999999], "person_id": asha}).status_code == 404


def test_the_vault_stays_locked(client, conn):
    item = make_item(conn, write_files=False, nsfw=1)
    ravi, asha = make_person(conn, "Ravi"), make_person(conn, "Asha")
    fid = make_face(conn, item["id"], None, write_crop=False)
    _guess(conn, "faces", fid, ravi)
    assert client.post("/api/guesses/correct", json={"faces": [fid], "person_id": asha}).status_code == 403
    assert _row(conn, "faces", fid)["guess_person_id"] == ravi
    assert json.loads(json.dumps(_events(conn))) == []
