"""Body traits in guessing (section 34): the clothes path untouched without traits, fusion rules, trait-only guesses,
outcomes, propagation, settings changes and hides."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.analysis import guess, trait_match, trait_sift
from app.config import settings
from app.db import faces_version, tx
from app.util import now_iso
from tests import test_analysis_guess as tag
from tests.helpers import make_body, make_face, make_item, make_person, make_scan, make_trait, set_traits_ready
from tests.test_trait_match import DESIGN, crop, hide, nails, photo, query, ref, vec


@pytest.fixture(autouse=True)
def _fresh_sift():
    trait_sift.reset()
    yield
    trait_sift.reset()


def body_row(conn, body_id):
    return conn.execute("SELECT * FROM bodies WHERE id = ?", (body_id,)).fetchone()


def ev_of(conn, body_id):
    return json.loads(body_row(conn, body_id)["guess_evidence"] or "null")


def clear(conn, body_id):
    with tx(conn):
        conn.execute("UPDATE bodies SET assigned_by = 'auto', guess_person_id = NULL, guess_score = NULL, "
                     "guess_evidence = NULL WHERE id = ?", (body_id,))


def requeued(conn) -> set[int]:
    return {int(r[0]) for r in conn.execute("SELECT item_id FROM trait_requeue")}


def p_body_in(conn, item_id, person, face_id=None):
    return make_body(conn, item_id, person, assigned_by="user", face_id=face_id)


# ------------------------------------------------------------------ the clothes path stays exactly as it was
WORLD_TESTS = [n for n in dir(tag) if n.startswith("test_") and n != "test_scoring_helpers"]


@pytest.mark.parametrize("name", WORLD_TESTS)
def test_every_clothes_world_holds_when_traits_are_ready(conn, name, monkeypatch):
    import inspect

    set_traits_ready(conn)
    fn = getattr(tag, name)
    fn(conn, monkeypatch) if "monkeypatch" in inspect.signature(fn).parameters else fn(conn)


def test_ready_with_scans_but_no_evidence_is_byte_identical(conn):
    p, a, b, face_a, body_b, _ = tag._world(conn)
    assert guess.guess_item(conn, b["id"]) == 1
    before = tuple(body_row(conn, body_b))
    clear(conn, body_b)
    set_traits_ready(conn)
    # a bare scan and an unmatched tattoo on the box: the fused path runs and finds nothing
    make_trait(conn, body_b, "tattoo", "calf_l", vec=vec(4), detail=crop(40),
               bare={"forearm_l": {"p": 0.9, "len": 200}, "forearm_r": {"p": 0.9, "len": 200}})
    assert guess.guess_item(conn, b["id"]) == 1
    assert tuple(body_row(conn, body_b)) == before
    v = faces_version(conn)
    assert guess.guess_item(conn, b["id"]) == 0 and faces_version(conn) == v


# ------------------------------------------------------------------ fusion with a clothes guess
def _clothes_world(conn):
    """tag._world plus P's body in A (with the face) so P can carry traits."""
    p, a, b, face_a, body_b, torso = tag._world(conn)
    body_a = p_body_in(conn, a["id"], p, face_id=face_a)
    return p, a, b, face_a, body_a, body_b, torso


def test_agreement_keeps_the_clothes_score(conn):
    p, a, b, _, body_a, body_b, _ = _clothes_world(conn)
    make_trait(conn, body_a, "tattoo", "forearm_l", vec=vec(1, DESIGN, 0.2), detail=crop(1, variant=0))
    make_trait(conn, body_b, "tattoo", "forearm_l", vec=vec(2, DESIGN, 0.2), detail=crop(1, variant=9))
    assert guess.guess_item(conn, b["id"]) == 1
    clothes_score = body_row(conn, body_b)["guess_score"]
    clothes_ev = ev_of(conn, body_b)
    clear(conn, body_b)
    set_traits_ready(conn)
    assert guess.guess_item(conn, b["id"]) == 1
    row = body_row(conn, body_b)
    ev = ev_of(conn, body_b)
    assert row["guess_person_id"] == p and row["guess_score"] == clothes_score
    assert ev["basis"] == "clothes+trait" and ev["trait_llr"] > 0 and ev["traits"][0]["kind"] == "tattoo"
    assert ev["score"] == clothes_ev["score"] and ev["clothes"] == clothes_ev["clothes"]
    assert ev["ref_item_uid"] == clothes_ev["ref_item_uid"]


def test_conflict_withdraws_the_clothes_guess(conn, monkeypatch):
    p, a, b, _, body_a, body_b, _ = _clothes_world(conn)
    q = make_person(conn, "Q")
    ref(conn, q, "2025-12-01T10:00:00.000Z", 5, 0)
    make_trait(conn, body_b, "tattoo", "forearm_l", vec=vec(3, DESIGN, 0.2), detail=crop(5, variant=9))
    set_traits_ready(conn)
    assert guess.guess_item(conn, b["id"]) == 0
    assert body_row(conn, body_b)["guess_person_id"] is None
    # trait-only guesses off: no admission, so no conflict either
    monkeypatch.setattr(settings, "trait_only_guesses", False)
    assert guess.guess_item(conn, b["id"]) == 1 and body_row(conn, body_b)["guess_person_id"] == p


def _track(conn, p, fam, days=("2025-11-01", "2026-01-15")):
    for k, day in enumerate(days):
        ref(conn, p, f"{day}T10:00:00.000Z", 60 + len(fam), k, part=f"{fam}_l", attrs={"ink": 0.3})


def test_absence_veto_needs_two_bare_families(conn):
    p, a, b, _, body_a, body_b, _ = _clothes_world(conn)
    _track(conn, p, "forearm")
    make_scan(conn, body_b, bare={"forearm_l": {"p": 0.9, "len": 200}, "forearm_r": {"p": 0.9, "len": 200}})
    set_traits_ready(conn)
    assert guess.guess_item(conn, b["id"]) == 1  # -0.7 only: the clothes guess stays as it was
    assert "basis" not in ev_of(conn, body_b)
    _track(conn, p, "upper_arm")
    make_scan(conn, body_b, bare={f"{f}_{s}": {"p": 0.9, "len": 200} for f in ("forearm", "upper_arm") for s in "lr"})
    trait_sift.reset()
    assert guess.guess_item(conn, b["id"]) == 1
    assert body_row(conn, body_b)["guess_person_id"] is None


def test_a_nails_negative_never_removes_a_clothes_guess(conn):
    p, a, b, _, body_a, body_b, _ = _clothes_world(conn)
    nails(conn, p, "2026-04-30T12:00:00.000Z", lab=[50.0, 60.0, 40.0])
    make_trait(conn, body_b, "nails", "hand_r", attrs={"lab": [40.0, 10.0, -45.0], "color": "blue", "back": True})
    with tx(conn):
        conn.execute("UPDATE items SET exif_datetime = '2026-05-01 11:00:00' WHERE id = ?", (b["id"],))
    set_traits_ready(conn)
    assert guess.guess_item(conn, b["id"]) == 1
    assert body_row(conn, body_b)["guess_person_id"] == p and "basis" not in ev_of(conn, body_b)


def _tie_world(conn):
    p, a, b, face_a, body_a, body_b, torso = _clothes_world(conn)
    q = make_person(conn, "Twin")
    c = make_item(conn, taken_at="2026-05-01T10:30:00.000Z", write_files=False, data=None)
    make_face(conn, c["id"], q, embedding=tag._vec(5, 512), assigned_by="user", torso=torso)
    tag._scene(conn, c["id"], tag._vec(1))
    assert guess.guess_item(conn, b["id"]) == 0  # tied on clothes
    return p, q, b, body_b


def test_a_strong_tattoo_pair_breaks_a_clothes_tie(conn):
    p, q, b, body_b = _tie_world(conn)
    ref(conn, p, "2025-10-01T10:00:00.000Z", 8, 0)
    make_trait(conn, body_b, "tattoo", "forearm_l", vec=vec(9, DESIGN, 0.2), detail=crop(8, variant=9))
    set_traits_ready(conn)
    assert guess.guess_item(conn, b["id"]) == 1
    row = body_row(conn, body_b)
    ev = ev_of(conn, body_b)
    assert row["guess_person_id"] == p and ev["basis"] == "clothes+trait"
    assert ev["score"] == round(row["guess_score"], 4)


def test_nails_never_break_a_tie(conn):
    p, q, b, body_b = _tie_world(conn)
    nails(conn, p, "2026-04-30T12:00:00.000Z")
    make_trait(conn, body_b, "nails", "hand_r", attrs={"lab": [40.0, 10.0, -45.0], "color": "blue", "back": True})
    with tx(conn):
        conn.execute("UPDATE items SET exif_datetime = '2026-05-01 11:00:00' WHERE id = ?", (b["id"],))
    set_traits_ready(conn)
    assert guess.guess_item(conn, b["id"]) == 0


# ------------------------------------------------------------------ trait-only
def test_an_open_face_gets_a_guess_through_its_body(conn):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    it = photo(conn, "2026-09-01T10:00:00.000Z")
    face = make_face(conn, it["id"], None)
    body = make_body(conn, it["id"], face_id=face)
    make_trait(conn, body, "tattoo", "forearm_l", vec=vec(11, DESIGN, 0.2), detail=crop(1, variant=9))
    set_traits_ready(conn)
    assert guess.guess_item(conn, it["id"]) == 1
    f = conn.execute("SELECT * FROM faces WHERE id = ?", (face,)).fetchone()
    assert f["guess_person_id"] == p and json.loads(f["guess_evidence"])["basis"] == "trait"


def test_a_face_of_someone_else_vetoes(conn):
    p, q = make_person(conn, "Kabir"), make_person(conn, "Q")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    q_face = tag._vec(70, 512)
    other = photo(conn, "2025-01-01T10:00:00.000Z")
    make_face(conn, other["id"], q, embedding=q_face, assigned_by="user")
    it = photo(conn, "2026-09-01T10:00:00.000Z")
    face = make_face(conn, it["id"], None, embedding=tag._near(q_face, 71, 1.2))
    body = make_body(conn, it["id"], face_id=face)
    make_trait(conn, body, "tattoo", "forearm_l", vec=vec(12, DESIGN, 0.2), detail=crop(1, variant=9))
    set_traits_ready(conn)
    assert guess.guess_item(conn, it["id"]) == 0


def test_one_person_per_photo_present_rejected_and_vault(conn):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    ref(conn, p, "2026-02-11T07:09:00.000Z", 2, 0, v=vec(2 * 31, DESIGN, 0.2))
    it = photo(conn, "2026-09-01T10:00:00.000Z")
    b1 = make_body(conn, it["id"], box=(0.05, 0.1, 0.3, 0.6))
    b2 = make_body(conn, it["id"], box=(0.6, 0.1, 0.3, 0.6))
    make_trait(conn, b1, "tattoo", "forearm_l", vec=vec(13, DESIGN, 0.2), detail=crop(1, variant=9))
    make_trait(conn, b2, "tattoo", "forearm_l", vec=vec(2 * 31, DESIGN, 0.2), detail=crop(2, variant=9))
    set_traits_ready(conn)
    assert guess.guess_item(conn, it["id"]) == 1
    assert sum(1 for b in (b1, b2) if body_row(conn, b)["guess_person_id"] == p) == 1
    guess.reject(conn, [], [b for b in (b1, b2) if body_row(conn, b)["guess_person_id"] == p])
    assert guess.guess_item(conn, it["id"]) == 1  # the other box now
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, created_at) VALUES (?, ?, 'user', ?)",
                     (it["id"], p, now_iso()))
    guess.guess_item(conn, it["id"])
    assert all(body_row(conn, b)["guess_person_id"] is None for b in (b1, b2))
    with tx(conn):
        conn.execute("DELETE FROM item_tags")
        conn.execute("DELETE FROM guess_rejections")
        conn.execute("UPDATE items SET nsfw = 1 WHERE id = ?", (it["id"],))
    assert guess.guess_item(conn, it["id"]) == 0
    assert all(body_row(conn, b)["guess_person_id"] is None for b in (b1, b2))


def _trait_guess(conn):
    p = make_person(conn, "Kabir")
    r_item, r_body, r_row = ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    q_item, q_body, q_row = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    assert guess.guess_item(conn, q_item) == 1
    return p, r_item, r_row, q_item, q_body


def test_window_change_keeps_trait_guesses(conn, monkeypatch):
    p, _, _, q_item, q_body = _trait_guess(conn)
    guess.apply_window_change(conn)
    monkeypatch.setattr(settings, "guess_window_h", 3.0)
    assert guess.apply_window_change(conn) == 0
    assert body_row(conn, q_body)["guess_person_id"] == p


def test_settings_change_requeues_and_withdraws_only_when_switched_off(conn, monkeypatch):
    p, _, _, q_item, q_body = _trait_guess(conn)
    assert trait_match.apply_settings_change(conn) == 0  # the first start after ready: recorded, nothing withdrawn
    assert body_row(conn, q_body)["guess_person_id"] == p and requeued(conn) == set()
    assert trait_match.apply_settings_change(conn) == 0
    monkeypatch.setattr(settings, "trait_margin_llr", 2.5)
    assert trait_match.apply_settings_change(conn) == 1
    assert body_row(conn, q_body)["guess_person_id"] == p and requeued(conn) == {q_item}
    monkeypatch.setattr(settings, "trait_only_guesses", False)
    assert trait_match.apply_settings_change(conn) == 1
    assert body_row(conn, q_body)["guess_person_id"] is None


def test_outcomes_of_trait_guesses_and_the_people_learner(conn):
    from app.learning import events, people

    assert people.RULES_REV == 3
    p, _, _, q_item, q_body = _trait_guess(conn)
    rng = np.random.default_rng(0)
    for i in range(40):
        events.record(conn, "guess", i % 2, {"clothes": float(rng.uniform(0.8, 1)), "scene_term": 0.9,
                                             "face_term": 0.5, "same_day": True, "has_gps": False, "score": 0.8})
    fit_before = people.guess_model(conn)
    assert guess.confirm(conn, [], [q_body]) == 1
    rows = events.rows(conn, "guess")
    f = json.loads(rows[-1]["features"])
    assert f["basis"] == "trait" and f["tattoo"] == 3.5 and f["kinds"] == ["tattoo"] and f["inliers"] >= 30
    assert f["trait_llr"] == 3.5 and "clothes" not in f and f["score"] is not None
    for i in range(10):
        events.record(conn, "guess", 0, {"basis": "trait", "trait_llr": 3.0, "score": 0.3})
    assert people.guess_model(conn) == fit_before
    people.train(conn)
    assert people.status()["events"] == 40


def test_clothes_and_trait_outcome_keeps_the_clothes_features(conn):
    from app.learning import events

    p, a, b, _, body_a, body_b, _ = _clothes_world(conn)
    make_trait(conn, body_a, "tattoo", "forearm_l", vec=vec(1, DESIGN, 0.2), detail=crop(1, variant=0))
    make_trait(conn, body_b, "tattoo", "forearm_l", vec=vec(2, DESIGN, 0.2), detail=crop(1, variant=9))
    set_traits_ready(conn)
    guess.guess_item(conn, b["id"])
    guess.confirm(conn, [], [body_b])
    f = json.loads(events.rows(conn, "guess")[-1]["features"])
    assert f["basis"] == "clothes+trait" and f["trait_llr"] > 0
    assert {"clothes", "scene_term", "face_term", "same_day", "has_gps", "score"} <= set(f)


def test_propagate_queues_trait_targets_without_verifying(conn):
    p = make_person(conn, "Kabir")
    r_item, _, _ = ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)  # eight months later
    far, _, _ = query(conn, "2026-09-02T10:00:00.000Z", 9, v=vec(31337))  # unrelated design
    set_traits_ready(conn)
    before = trait_sift.stats()["verified"]
    guess.propagate(conn, p, r_item)
    assert requeued(conn) == {q_item}
    assert trait_sift.stats()["verified"] == before
    assert trait_match.trait_targets(conn, p, None, 400) == [q_item]


def test_sweep_requeues_open_boxes_with_traits(conn):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    q_item, _, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    assert trait_match.sweep_open(conn) == 0  # not ready
    set_traits_ready(conn)
    assert trait_match.sweep_open(conn) == 1 and requeued(conn) == {q_item}


def test_a_hide_withdraws_the_guess_and_undoing_it_brings_it_back(conn):
    p, r_item, r_row, q_item, q_body = _trait_guess(conn)
    h = hide(conn, r_item, "tattoo", "not_trait", (0.2, 0.4, 0.05, 0.1), src=r_row)
    trait_match.after_hide_change(conn, [r_item])
    assert q_item in requeued(conn)
    assert guess.guess_item(conn, q_item) == 1  # what the runner does with the queue
    assert body_row(conn, q_body)["guess_person_id"] is None
    with tx(conn):
        conn.execute("DELETE FROM trait_hides WHERE id = ?", (h,))
    trait_match.after_hide_change(conn, [r_item])
    assert guess.guess_item(conn, q_item) == 1
    assert body_row(conn, q_body)["guess_person_id"] == p


def test_a_hide_on_the_query_row_reguesses_its_item_at_once(conn):
    p, r_item, r_row, q_item, q_body = _trait_guess(conn)
    hide(conn, q_item, "tattoo", "not_trait", (0.2, 0.4, 0.05, 0.1))
    assert trait_match.after_hide_change(conn, [q_item]) >= 1
    assert body_row(conn, q_body)["guess_person_id"] is None


def test_an_undated_reference_never_breaks_guessing(conn):
    """A person's tattoo row on a photo without a date, with a lasting track and a bare query: no exception, and the
    clothes guess stays."""
    p, a, b, _, body_a, body_b, _ = _clothes_world(conn)
    _track(conn, p, "forearm")
    undated = make_item(conn, taken_at=None, write_files=False, data=None)
    ref(conn, p, "", 60 + len("forearm"), 7, part="forearm_l", attrs={"ink": 0.3}, item=undated)
    make_scan(conn, body_b, bare={"forearm_l": {"p": 0.9, "len": 200}, "forearm_r": {"p": 0.9, "len": 200}})
    set_traits_ready(conn)
    assert guess.guess_item(conn, b["id"]) == 1
    assert body_row(conn, body_b)["guess_person_id"] == p
    assert trait_match.calibration_sample(conn, force=True)["nails"] == {"pos": [], "neg": []}


def test_a_failing_trait_path_falls_back_to_clothes(conn, monkeypatch):
    p, a, b, _, body_a, body_b, _ = _clothes_world(conn)
    make_trait(conn, body_b, "tattoo", "forearm_l", vec=vec(2, DESIGN, 0.2), detail=crop(1, variant=9))
    set_traits_ready(conn)

    def boom(*a, **k):
        raise RuntimeError("broken")

    monkeypatch.setattr(trait_match.TraitContext, "score_all", boom)
    assert guess.guess_item(conn, b["id"]) == 1
    assert body_row(conn, body_b)["guess_person_id"] == p and "basis" not in ev_of(conn, body_b)
