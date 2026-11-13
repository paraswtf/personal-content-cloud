"""Where the recognition trained on one person (section 10) meets the other sources of guesses: position
(section 35) and clothes. The model's own behaviour is in test_person_model.py."""
from __future__ import annotations

import json

from app.analysis import guess, person_scan, traits
from app.learning import person_model
from tests.helpers import make_face, make_item, make_person
from tests.test_person_model import World, guessed, row, unit
from tests.test_place_guess import HEAD, body_row
from tests.test_place_match import MID, body_at, link, photo


def _queued(conn) -> set[int]:
    return {int(r[0]) for r in conn.execute("SELECT item_id FROM trait_requeue")}


# ------------------------------------------------------------------ the position part of a guess the model beats
def test_a_model_guess_keeps_the_position_part_of_the_guess_it_agrees_with(conn):
    w = World(conn)
    w.train()
    fid = w.face(w.photo(), w.like_me())
    item_id = int(row(conn, fid)["item_id"])
    item = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    boxes = guess.open_boxes(conn, item_id)
    place = {"ref_item_uid": "x", "ref_face_id": 1, "seconds_apart": 8, "hops": 1}
    for basis in ("clothes+place", "place"):
        c = guess.Candidate(w.me, 0.3, {"basis": basis, "score": 0.3, "place_llr": 2.0, "place": place})
        ev = person_scan.merge_item(conn, item, boxes, {("face", fid): c})[("face", fid)].evidence
        # still the model's guess, and what the Review filter, the card and the position sweep read is there
        assert (ev["basis"], ev["also"], ev["place"], ev["place_llr"]) == ("face", basis, place, 2.0)


def test_a_model_guess_without_a_position_part_is_stored_as_before(conn):
    w = World(conn)
    w.train()
    fid = w.face(w.photo(), w.like_me())
    item_id = int(row(conn, fid)["item_id"])
    item = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    boxes = guess.open_boxes(conn, item_id)
    alone = person_scan.merge_item(conn, item, boxes, {})[("face", fid)].evidence
    c = guess.Candidate(w.me, 0.3, {"score": 0.3})
    with_clothes = person_scan.merge_item(conn, item, boxes, {("face", fid): c})[("face", fid)].evidence
    assert "place" not in alone and "place_llr" not in alone
    assert json.dumps(with_clothes) == json.dumps({**alone, "also": "clothes"})


# ------------------------------------------------------------------ taking the guesses back
def _agreeing_clothes_guess(conn, w: World) -> int:
    """A face whose clothes guess for the trained person scores higher than the model's: it carries `p`."""
    from tests.test_analysis_guess import _scene, _vec

    scene, torso = _vec(1), _vec(2)
    ref = make_item(conn, taken_at="2026-05-01T10:00:00.000Z", write_files=False, data=None)
    w.face(int(ref["id"]), w.like_me(), w.me, "user", torso=torso)
    _scene(conn, int(ref["id"]), scene)
    item = make_item(conn, taken_at="2026-05-01T10:12:00.000Z", write_files=False, data=None)
    _scene(conn, int(item["id"]), scene)
    f = w.face(int(item["id"]), w.like_me(), torso=torso)
    guess.guess_item(conn, int(item["id"]))
    pid, _, ev = guessed(conn, f)
    assert pid == w.me and "basis" not in ev and "p" in ev
    return f


def test_forgetting_the_model_takes_its_probability_off_the_guesses_it_agreed_with(conn):
    w = World(conn, lookalike=unit(5))
    w.train()
    f = _agreeing_clothes_guess(conn, w)
    person_model.forget(w.me)
    person_scan.take_back(conn, w.me)
    pid, state, ev = guessed(conn, f)
    assert (pid, state) == (w.me, "guess") and "p" not in ev  # the clothes guess stays, without the model's word


def test_taking_the_guesses_back_takes_the_probability_off_too(conn):
    w = World(conn, lookalike=unit(5))
    w.train()
    f = _agreeing_clothes_guess(conn, w)
    person_scan.withdraw(conn, w.me)  # pauses the model first
    pid, state, ev = guessed(conn, f)
    assert (pid, state) == (w.me, "guess") and "p" not in ev


# ------------------------------------------------------------------ a face that leaves or rejoins a group
def _followed_from_a_group_face(conn, w: World):
    """Photo a: a face of the trained person that sits in an unnamed group. Photo b, two seconds later and lined
    up with a: a faceless box at the same spot, guessed as that group by position."""
    u = make_person(conn, None)
    a, b = photo(conn, 0), photo(conn, 2)
    fa = make_face(conn, a["id"], u, box=MID, assigned_by="auto", embedding=w.like_me(), write_crop=False)
    make_face(conn, w.photo(), u, assigned_by="auto", embedding=unit(99999), write_crop=False)  # the group stays
    body = body_at(conn, b, *HEAD)
    link(conn, a, b)
    guess.guess_item(conn, b["id"])
    r = body_row(conn, body)
    assert r["guess_person_id"] == u and json.loads(r["guess_evidence"])["place"]["ref_face_id"] == fa
    return u, a, b, fa, body


def test_a_look_that_takes_a_face_out_of_a_group_guesses_the_photo_that_followed_it_again(conn):
    w = World(conn)
    w.train()
    u, a, b, fa, body = _followed_from_a_group_face(conn, w)
    person_scan.scan(conn, w.me)
    f = row(conn, fa)
    assert f["person_id"] is None and f["guess_person_id"] == w.me  # out of the group, a guess for the person
    assert b["id"] in _queued(conn)
    traits.drain_requeue(conn)
    r = body_row(conn, body)
    assert r["guess_person_id"] != u  # nobody of the group stands there any more


def test_a_face_that_goes_back_into_its_group_guesses_the_photo_lined_up_with_it_again(conn):
    w = World(conn)
    w.train()
    u, a, b, fa, body = _followed_from_a_group_face(conn, w)
    person_scan.scan(conn, w.me)
    traits.drain_requeue(conn)
    assert body_row(conn, body)["guess_person_id"] != u
    guess.reject(conn, [fa], [])  # "not them": the face goes back into the group it came out of
    assert row(conn, fa)["person_id"] == u
    assert b["id"] in _queued(conn)
    traits.drain_requeue(conn)
    r = body_row(conn, body)
    assert r["guess_person_id"] == u and json.loads(r["guess_evidence"])["place"]["ref_face_id"] == fa
