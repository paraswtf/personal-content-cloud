"""Position in guesses (section 35): place_match.fuse, and guess.guess_item with stored links."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from app.analysis import ages, guess, place
from app.analysis import place_match as pm
from app.analysis.guess import Candidate
from app.config import settings
from app.db import kv_get, kv_set, tx
from app.learning import people as learned_people
from app.util import now_iso
from tests.helpers import make_body, make_face, make_item, make_person
from tests.test_place_match import ADULT, CHILD, MID, body_at, face, link, near, photo, unit_vec

HEAD = (MID[0] + MID[2] / 2, MID[1] + MID[3] / 2)      # the head point of a face at MID


def scene(conn, item, v: np.ndarray) -> None:
    with tx(conn):
        conn.execute("INSERT INTO item_embeddings(item_id, version, model, frame_index, frame_time_s, embedding, created_at) "
                     "VALUES (?, 1, 'test', 0, NULL, ?, ?)", (item["id"], v.astype(np.float32).tobytes(), now_iso()))


def body_row(conn, body_id: int):
    return conn.execute("SELECT * FROM bodies WHERE id = ?", (body_id,)).fetchone()


def world(conn, *, gap: float = 2.0):
    """Photo a shows a named person's face; photo b, `gap` seconds later, a faceless box at the same spot. The two
    photos are lined up."""
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, gap)
    fa = face(conn, a, p, MID)
    body = body_at(conn, b, *HEAD)
    link(conn, a, b)
    return p, a, b, fa, body


# ------------------------------------------------------------------ fuse
def cand(pid: int, value: float, **ev) -> Candidate:
    return Candidate(pid, value, dict(ev))


def place_result(**llr: float) -> pm.PlaceResult:
    return pm.PlaceResult({int(k[1:]): v for k, v in llr.items()},
                          {int(k[1:]): {"ref_item_uid": "u", "ref_face_id": 1, "ref_body_id": None, "seconds_apart": 2.0,
                                        "ref_before": True, "hops": 1, "offset": 0.0, "kind": "still", "others": 0,
                                        "clothes": None, "llr": round(v, 2)} for k, v in llr.items()})


def test_fuse_agreement_keeps_the_score_and_names_the_basis():
    win = cand(1, 0.8, clothes=0.9, score=0.8)
    out = pm.fuse(win, [win], place_result(p1=3.9))
    assert out.person_id == 1 and out.score == 0.8
    assert out.evidence["basis"] == "clothes+place" and out.evidence["place_llr"] == 3.9
    assert out.evidence["place"]["hops"] == 1 and out.evidence["clothes"] == 0.9
    assert "basis" not in win.evidence                                   # the clothes candidate is not touched
    trait = cand(1, 0.9, basis="clothes+trait", traits=[{"kind": "tattoo"}], score=0.9)
    out = pm.fuse(trait, [trait], place_result(p1=3.9))
    assert out.evidence["basis"] == "clothes+trait" and out.evidence["place"]["kind"] == "still"
    assert pm.fuse(win, [win], place_result(p2=1.0)) is win              # position says nothing about the winner


def test_fuse_contradiction_at_admission_strength_gives_no_guess():
    win = cand(1, 0.8, score=0.8)
    assert pm.fuse(win, [win], place_result(p2=pm.ADMIT_LLR + 0.5)) is None
    assert pm.fuse(win, [win], place_result(p2=pm.ADMIT_LLR - 0.5)) is win       # too weak to contradict
    both = place_result(p1=3.0, p2=pm.ADMIT_LLR + 0.5)                           # no lead: not admitted, it agrees
    assert pm.fuse(win, [win], both).evidence["basis"] == "clothes+place"


def test_fuse_breaks_a_clothes_tie_for_exactly_one():
    a, b = cand(1, 0.80, clothes=0.9), cand(2, 0.78, clothes=0.89)
    out = pm.fuse(None, [a, b], place_result(p2=pm.TIE_LLR))
    assert out.person_id == 2 and out.score == 0.78 and out.evidence["basis"] == "clothes+place"
    assert out.evidence["score"] == 0.78
    assert pm.fuse(None, [a, b], place_result(p2=pm.TIE_LLR - 0.1)) is None      # too weak for a tie
    assert pm.fuse(None, [a, b], place_result(p1=1.5, p2=1.5)) is None           # it backs both
    far = cand(3, 0.60)
    assert pm.fuse(None, [a, b, far], place_result(p3=pm.TIE_LLR)) is None       # not among the tied


def test_fuse_position_alone_needs_strength_a_lead_and_the_switch(monkeypatch):
    out = pm.fuse(None, [], place_result(p1=pm.ADMIT_LLR))
    assert out.person_id == 1 and out.evidence["basis"] == "place"
    assert out.score == pytest.approx(round(1 / (1 + math.exp(-pm.ADMIT_LLR)), 4))
    assert out.evidence["score"] == out.score and out.evidence["ref_item_uid"] == "u"
    assert pm.fuse(None, [], place_result(p1=pm.ADMIT_LLR - 0.1)) is None
    assert pm.fuse(None, [], place_result(p1=3.9, p2=3.9 - pm.LEAD_LLR + 0.1)) is None      # no lead
    assert pm.fuse(None, [], place_result(p1=3.9, p2=3.9 - pm.LEAD_LLR)).person_id == 1
    assert pm.fuse(None, [], place_result(p1=9.0)).score == pm.MAX_SCORE
    other = cand(2, 0.5)
    assert pm.fuse(None, [other], place_result(p1=3.9)) is None                  # clothes rank someone else only
    monkeypatch.setattr(settings, "place_only_guesses", False)
    assert pm.fuse(None, [], place_result(p1=3.9)) is None
    win = cand(1, 0.8, score=0.8)
    assert pm.fuse(win, [win], place_result(p2=3.9)) is win                      # and nothing to contradict with


def test_the_last_tier_and_a_burst_never_name_alone():
    top = max(pm.TIERS[-1][1], pm.BURST_LLR) + pm.CLOTHES_BONUS + pm.KEPT_MAX
    assert top < pm.ADMIT_LLR
    assert pm.TIERS[0][1] - pm.FACELESS_COST >= pm.ADMIT_LLR                     # a faceless box seconds apart may


# ------------------------------------------------------------------ guess_item
def test_a_faceless_box_in_the_same_spot_is_guessed(conn):
    p, a, b, fa, body = world(conn)
    assert guess.guess_item(conn, b["id"]) == 1
    r = body_row(conn, body)
    llr = pm.tier_llr(2.0) - pm.FACELESS_COST
    assert (r["assigned_by"], r["guess_person_id"]) == ("guess", p)
    assert r["guess_score"] == pytest.approx(round(1 / (1 + math.exp(-llr)), 4))
    ev = json.loads(r["guess_evidence"])
    assert list(ev) == ["basis", "score", "place_llr", "ref_item_uid", "ref_face_id", "ref_body_id", "seconds_apart", "place"]
    assert (ev["basis"], ev["ref_item_uid"], ev["ref_face_id"], ev["seconds_apart"]) == ("place", a["uid"], fa, 2.0)
    assert ev["place"]["ref_before"] is True and ev["place"]["hops"] == 1 and ev["place_llr"] == round(llr, 2)
    assert guess.guess_item(conn, b["id"]) == 0                         # the same bytes: nothing is written again
    assert guess.guess_item(conn, a["id"]) == 0                         # nothing open there


def test_no_guess_without_the_switch_or_the_feature(conn, monkeypatch):
    p, a, b, fa, body = world(conn)
    guess.guess_item(conn, b["id"])
    monkeypatch.setattr(settings, "place_only_guesses", False)
    assert guess.guess_item(conn, b["id"]) == 1 and body_row(conn, body)["guess_person_id"] is None   # withdrawn
    monkeypatch.setattr(settings, "place_only_guesses", True)
    assert guess.guess_item(conn, b["id"]) == 1
    monkeypatch.setattr(settings, "place_enabled", False)
    assert guess.guess_item(conn, b["id"]) == 1 and body_row(conn, body)["guess_person_id"] is None


def test_too_far_apart_in_time_is_no_guess_alone(conn):
    p, a, b, fa, body = world(conn, gap=pm.TIERS[-1][0] - 10)
    assert guess.guess_item(conn, b["id"]) == 0 and body_row(conn, body)["guess_person_id"] is None


def test_rejected_present_and_vault(conn):
    p, a, b, fa, body = world(conn)
    with tx(conn):
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('body', ?, ?, ?)",
                     (body, p, now_iso()))
    assert guess.guess_item(conn, b["id"]) == 0
    with tx(conn):
        conn.execute("DELETE FROM guess_rejections")
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, relation, created_at) VALUES (?, ?, 'user', 'in', ?)",
                     (b["id"], p, now_iso()))
    assert guess.guess_item(conn, b["id"]) == 0                         # the person is in the photo already
    with tx(conn):
        conn.execute("DELETE FROM item_tags")
        conn.execute("UPDATE items SET nsfw = 1 WHERE id = ?", (b["id"],))
    assert guess.guess_item(conn, b["id"]) == 0                         # the vault gets no guesses


def test_one_person_for_one_box_of_a_photo(conn):
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, p, MID)
    near_box = body_at(conn, b, *HEAD)
    far_box = body_at(conn, b, 0.85, 0.7)
    link(conn, a, b)
    assert guess.guess_item(conn, b["id"]) == 1
    assert body_row(conn, near_box)["guess_person_id"] == p and body_row(conn, far_box)["guess_person_id"] is None


def test_position_agrees_with_a_clothes_guess(conn, monkeypatch):
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, 2)
    s, t = unit_vec(1), unit_vec(2)
    scene(conn, a, s)
    scene(conn, b, s)
    fa = face(conn, a, p, MID, torso=t)
    body = body_at(conn, b, *HEAD, torso=near(t, 0.93, 3))
    monkeypatch.setattr(settings, "place_enabled", False)
    guess.guess_item(conn, b["id"])
    plain = body_row(conn, body)
    plain_ev = json.loads(plain["guess_evidence"])
    assert plain["guess_person_id"] == p and "basis" not in plain_ev
    monkeypatch.setattr(settings, "place_enabled", True)
    assert guess.guess_item(conn, b["id"]) == 0                         # no link yet: byte for byte the clothes guess
    link(conn, a, b)
    assert guess.guess_item(conn, b["id"]) == 1
    r = body_row(conn, body)
    ev = json.loads(r["guess_evidence"])
    assert r["guess_score"] == plain["guess_score"]                     # the clothes score, unchanged
    assert ev["basis"] == "clothes+place" and ev["clothes"] == plain_ev["clothes"] and ev["ref_face_id"] == fa
    assert ev["place"]["clothes"] == pytest.approx(0.93, abs=1e-3) and ev["place"]["ref_face_id"] == fa
    assert {k: v for k, v in ev.items() if k not in ("basis", "place_llr", "place")} == plain_ev


def test_position_against_the_clothes_winner_gives_no_guess(conn):
    p, q = make_person(conn, "Asha"), make_person(conn, "Ravi")
    a, b, c = photo(conn, 0), photo(conn, 2), photo(conn, 600)
    s, t = unit_vec(1), unit_vec(2)
    for it in (a, b, c):
        scene(conn, it, s)
    face(conn, a, p, MID)                                               # in the spot, no clothes vector
    body = body_at(conn, b, *HEAD, torso=near(t, 0.95, 3))
    make_body(conn, c["id"], q, assigned_by="user", torso=t)            # the same clothes ten minutes later
    assert guess.guess_item(conn, b["id"]) == 1 and body_row(conn, body)["guess_person_id"] == q
    link(conn, a, b)
    assert guess.guess_item(conn, b["id"]) == 1 and body_row(conn, body)["guess_person_id"] is None


def test_position_breaks_a_clothes_tie(conn):
    p, q = make_person(conn, "Asha"), make_person(conn, "Ravi")
    a, b, c = photo(conn, 0), photo(conn, 30), photo(conn, 60)
    s, t = unit_vec(1), unit_vec(2)
    for it in (a, b, c):
        scene(conn, it, s)
    body = body_at(conn, b, *HEAD, torso=t)
    make_body(conn, a["id"], p, assigned_by="user", box=body_box(conn, body), torso=t)      # the same clothes twice
    make_body(conn, c["id"], q, assigned_by="user", box=(0.7, 0.5, 0.2, 0.4), torso=t)
    assert guess.guess_item(conn, b["id"]) == 0                         # a tie: no guess
    link(conn, a, b)
    assert guess.guess_item(conn, b["id"]) == 1
    r = body_row(conn, body)
    ev = json.loads(r["guess_evidence"])
    assert r["guess_person_id"] == p and ev["basis"] == "clothes+place" and ev["score"] == r["guess_score"]


def body_box(conn, body_id: int):
    r = body_row(conn, body_id)
    return (r["x"], r["y"], r["w"], r["h"])


def test_the_box_s_own_face_can_say_no(conn):
    p, q = make_person(conn, "Asha"), make_person(conn, "Ravi")
    a, b = photo(conn, 0), photo(conn, 2)
    e = np.zeros(512, np.float32)
    e[0] = 1.0
    face(conn, a, p, MID)
    for k in range(3):                                                  # the open face looks like Ravi
        other = photo(conn, 1000 + k)
        make_face(conn, other["id"], q, embedding=e, assigned_by="user", write_crop=False)
    fb = make_face(conn, b["id"], None, embedding=e, box=MID, write_crop=False)
    link(conn, a, b)
    guess.guess_item(conn, b["id"])
    assert conn.execute("SELECT guess_person_id FROM faces WHERE id = ?", (fb,)).fetchone()[0] is None


def test_a_child_is_not_followed_to_an_adult(conn):
    with tx(conn):
        kv_set(conn, ages.READY_KEY, ages.MODEL_REV)
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, 2)
    clear = dict(clarity=25.0, sharpness=40.0)
    face(conn, a, p, MID, age_groups=ages.encode(ADULT), **clear)
    fb = make_face(conn, b["id"], None, box=MID, write_crop=False, age_groups=ages.encode(CHILD), **clear)
    link(conn, a, b)
    guess.guess_item(conn, b["id"])
    assert conn.execute("SELECT guess_person_id FROM faces WHERE id = ?", (fb,)).fetchone()[0] is None


def test_a_box_is_followed_across_two_photos(conn):
    p = make_person(conn, "Asha")
    a, b, c = photo(conn, 0), photo(conn, 2), photo(conn, 4)
    face(conn, a, p, MID)
    mid = body_at(conn, b, *HEAD)
    last = body_at(conn, c, *HEAD)
    link(conn, a, b)
    link(conn, b, c)
    link(conn, a, c, kind="none")                                       # a and c did not line up directly
    guess.guess_item(conn, c["id"])
    ev = json.loads(body_row(conn, last)["guess_evidence"])
    assert body_row(conn, last)["guess_person_id"] == p and ev["place"]["hops"] == 2 and ev["seconds_apart"] == 4.0
    guess.guess_item(conn, b["id"])
    assert json.loads(body_row(conn, mid)["guess_evidence"])["place"]["hops"] == 1


# ------------------------------------------------------------------ answers, the learner, the window, the API
def test_outcomes_are_recorded_apart_from_clothes(conn):
    p, a, b, fa, body = world(conn)
    guess.guess_item(conn, b["id"])
    before = learned_people._clothes_events(conn)
    assert guess.confirm(conn, [], [body]) == 1
    row = conn.execute("SELECT label, features FROM learning_events WHERE kind = 'guess'").fetchone()
    f = json.loads(row["features"])
    assert row["label"] == 1 and f["basis"] == "place" and f["hops"] == 1 and f["seconds_apart"] == 2.0
    assert f["place_llr"] == round(pm.tier_llr(2.0) - pm.FACELESS_COST, 2) and "clothes" not in f
    assert learned_people._clothes_events(conn) == before               # the clothes fit never sees it
    assert "place" in learned_people.NOT_CLOTHES and "trait" in learned_people.NOT_CLOTHES
    r = body_row(conn, body)
    assert (r["person_id"], r["assigned_by"], r["by_guess"]) == (p, "user", 1)


def test_a_clothes_guess_with_position_keeps_its_clothes_features(conn):
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, 2)
    s, t = unit_vec(1), unit_vec(2)
    scene(conn, a, s)
    scene(conn, b, s)
    face(conn, a, p, MID, torso=t)
    body = body_at(conn, b, *HEAD, torso=near(t, 0.93, 3))
    link(conn, a, b)
    guess.guess_item(conn, b["id"])
    assert guess.reject(conn, [], [body]) == 1
    f = json.loads(conn.execute("SELECT features FROM learning_events WHERE kind = 'guess'").fetchone()[0])
    assert f["basis"] == "clothes+place" and f["clothes"] == pytest.approx(0.93, abs=1e-3) and f["place_llr"] > 0
    assert len(learned_people._clothes_events(conn)) == 1               # it is a clothes outcome


def test_a_window_change_keeps_position_guesses(conn, monkeypatch):
    p, a, b, fa, body = world(conn)
    guess.guess_item(conn, b["id"])
    with tx(conn):
        kv_set(conn, guess.WINDOW_KEY, "5")
    with tx(conn):                                                      # as if its reference were far away in time
        conn.execute("UPDATE items SET taken_at = '2026-05-03T10:00:00.000Z' WHERE id = ?", (a["id"],))
    assert guess.apply_window_change(conn) == 0
    assert body_row(conn, body)["guess_person_id"] == p and kv_get(conn, guess.WINDOW_KEY) == "1"


def test_changed_rules_queue_the_position_guesses_again(conn, monkeypatch):
    p, a, b, fa, body = world(conn)
    assert place.apply_settings_change(conn) == 0                       # the first start records the rules
    guess.guess_item(conn, b["id"])
    assert place.apply_settings_change(conn) == 0
    monkeypatch.setattr(settings, "place_only_guesses", False)
    assert place.apply_settings_change(conn) == 1
    assert [int(r[0]) for r in conn.execute("SELECT item_id FROM trait_requeue")] == [b["id"]]
    from app.analysis import traits

    assert traits.drain_requeue(conn) == 1 and body_row(conn, body)["guess_person_id"] is None
    assert place.apply_settings_change(conn) == 0


def test_review_lists_position_guesses_with_the_reference_crop(conn, client):
    p, a, b, fa, body = world(conn)
    guess.guess_item(conn, b["id"])
    got = client.get("/api/review/guesses", params={"basis": "place"}).json()
    assert got["total"] == 1 and got["items"][0]["kind"] == "body" and got["items"][0]["id"] == body
    ev = got["items"][0]["guess"]["evidence"]
    assert ev["basis"] == "place" and ev["place"]["ref_crop_url"].startswith(f"/files/face/{fa}")
    assert client.get("/api/review/guesses", params={"basis": "clothes"}).json()["total"] == 0
    assert client.get("/api/review/guesses", params={"basis": "trait"}).json()["total"] == 0
    assert client.get("/api/review/guesses").json()["total"] == 1
    assert client.get("/api/review/guesses", params={"basis": "nope"}).status_code == 422
    detail = client.get(f"/api/items/{b['uid']}").json()
    assert detail["bodies"][0]["guess"]["evidence"]["place"]["ref_crop_url"].startswith(f"/files/face/{fa}")


def test_the_learning_status_carries_the_progress(conn, client):
    world(conn)
    scan = client.get("/api/system/learning").json()["learners"]["place"]["scan"]
    assert scan["eligible"] == 2 and scan["scanned"] == 0 and scan["enabled"] is True and scan["last_error"] is None
    assert scan["ready"] is True          # the library was empty when the app started: nothing was waiting then
