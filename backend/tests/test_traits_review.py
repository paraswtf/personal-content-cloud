"""The traits review round (section 34): mehndi off by default, the face veto for split groups, small pairs, the
reference bar for one event, admission without nails leading, tie-breaks following the learned floor, propagation
handing trait photos to the runner, and the runner surviving one odd photo. Models are off (PCC_NO_MODELS=1); SIFT
runs on the synthetic tattoo crops."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import guess, trait_match, trait_nails, trait_regions, trait_sift, traits
from app.analysis.trait_match import TraitResult
from app.config import settings
from app.learning import traits as ltraits
from tests.helpers import READY_REV, make_body, make_face, make_person, make_trait, set_traits_ready
from tests.test_trait_match import DESIGN, crop, photo, query, ref, vec
from tests.test_traits_extract import (BOX, FakeHands, FakePose, FakeSiglip, fake_engines, inked_skin, rows, W, H)
from tests.test_traits_extract import photo as px_photo


@pytest.fixture(autouse=True)
def _fresh_sift():
    trait_sift.reset()
    yield
    trait_sift.reset()


HENNA = {"tattoo": 0.05, "bare": 0.05, "cloth": 0.05, "henna": 0.8, "jewel": 0.05}


# ------------------------------------------------------------------ mehndi is off by default
def test_mehndi_is_off_by_default_and_its_rows_are_not_used(conn, monkeypatch):
    assert "TRAITS_MEHNDI" not in __import__("os").environ
    monkeypatch.setattr(settings, "traits_mehndi", False)
    assert traits.kinds_on() == ("nails", "tattoo") and traits.wanted_rev() == "2:nails,tattoo"
    item = photo(conn, "2026-02-01T10:00:00.000Z")
    body = make_body(conn, item["id"], box=(0.1, 0.1, 0.3, 0.6))
    make_trait(conn, body, "tattoo", "forearm_l", vec=vec(1, DESIGN), detail=crop(1))
    make_trait(conn, body, "mehndi", "hand_r", vec=vec(2, DESIGN), detail=crop(2), box=(0.3, 0.5, 0.05, 0.08))

    def kinds() -> set[str]:
        sql = ("SELECT t.kind FROM body_traits t JOIN trait_scans s ON s.body_id = t.body_id "
               "JOIN bodies b ON b.id = t.body_id JOIN items i ON i.id = t.item_id WHERE " + traits.usable_sql())
        return {r[0] for r in conn.execute(sql, {"ready_rev": READY_REV})}

    assert kinds() == {"tattoo"}
    monkeypatch.setattr(settings, "traits_mehndi", True)
    assert kinds() == {"tattoo", "mehndi"} and traits.wanted_rev() == "2:mehndi,nails,tattoo"


def test_a_hand_siglip_calls_henna_still_gets_its_nails_when_mehndi_is_off(conn, monkeypatch):
    """SigLIP read plain hands as mehndi on the owner's library: with mehndi off such a hand is no evidence, and
    its nails are still read; with mehndi on it is a mehndi row and no nails are read."""
    seen = []

    def nails(crop_, lm, conf, side, probs, *, src):
        seen.append(side)
        return {"lab": [40.0, 60.0, 40.0], "color": "red", "hex": "#c81e28", "finish": "plain", "finish_p": 0.9,
                "length": "medium", "r": 0.2, "fingers": 4, "p_tip": 0.9, "back": True, "sharp": 300.0, "src": src}

    monkeypatch.setattr(trait_nails, "describe_hand", nails)
    monkeypatch.setattr(settings, "traits_mehndi", False)
    off = px_photo(conn, inked_skin(W, H, ink=False))
    make_body(conn, off["id"], box=BOX)
    hands = FakeHands()
    fake_engines(monkeypatch, pose=FakePose(hands=0.9), hands=hands, siglip=FakeSiglip(HENNA))
    traits.run_pending(conn)
    got = rows(conn, off["id"])
    assert hands.calls == [2] and sorted(seen) == ["l", "r"]
    assert {r["kind"] for r in got} == {"nails"}            # no mehndi, no tattoo from henna-like hands
    monkeypatch.setattr(settings, "traits_mehndi", True)
    on = px_photo(conn, inked_skin(W, H, ink=False))
    make_body(conn, on["id"], box=BOX)
    hands2 = FakeHands()
    fake_engines(monkeypatch, pose=FakePose(hands=0.9), hands=hands2, siglip=FakeSiglip(HENNA))
    traits.run_pending(conn)
    assert hands2.calls == []                               # a mehndi hand: its nails are not read
    assert "mehndi" in {r["kind"] for r in rows(conn, on["id"])}


# ------------------------------------------------------------------ the face veto
def _face_query(conn, p):
    """P's owner-confirmed tattoo months earlier, and a photo whose open face sits in a body with the same tattoo."""
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    it = photo(conn, "2026-09-01T10:00:00.000Z")
    face = make_face(conn, it["id"], None)
    body = make_body(conn, it["id"], face_id=face)
    make_trait(conn, body, "tattoo", "forearm_l", vec=vec(11, DESIGN, 0.2), detail=crop(1, variant=9))
    set_traits_ready(conn)
    item = conn.execute("SELECT * FROM items WHERE id = ?", (it["id"],)).fetchone()
    ctx = trait_match.context(conn, item, guess.open_boxes(conn, it["id"]), {})
    assert ctx is not None
    return face, ctx


def test_an_unnamed_group_the_face_resembles_does_not_veto_the_tattoo(conn):
    """A tattooed person's photos split into an unnamed group: the face looks like that group (0.5) and like P
    (0.41). Only a named person, or a group when the face hardly looks like P, vetoes P."""
    p = make_person(conn, "Kabir")
    split = make_person(conn, None)
    face, ctx = _face_query(conn, p)
    res = ctx.score_all({face: {split: 0.5, p: 0.41}}, {})
    assert p in res[("face", face)].llr


def test_a_named_person_the_face_resembles_vetoes_the_tattoo(conn):
    p, q = make_person(conn, "Kabir"), make_person(conn, "Q")
    face, ctx = _face_query(conn, p)
    res = ctx.score_all({face: {q: 0.45, p: 0.41}}, {})
    assert ("face", face) not in res or p not in res[("face", face)].llr


def test_an_unnamed_group_vetoes_when_the_face_hardly_looks_like_p(conn):
    p = make_person(conn, "Kabir")
    split = make_person(conn, None)
    face, ctx = _face_query(conn, p)
    res = ctx.score_all({face: {split: 0.5, p: 0.15}}, {})
    assert ("face", face) not in res or p not in res[("face", face)].llr


# ------------------------------------------------------------------ small pairs and the reference bar
def test_small_references_give_no_support_and_no_bar(conn):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)                     # one confirmed, full-size reference
    for when, variant in (("2026-03-02T07:09:00.000Z", 1), ("2026-05-02T07:09:00.000Z", 2)):
        ref(conn, p, when, 1, variant, assigned_by="auto", attrs={"small": True, "kp": 40})
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    item = conn.execute("SELECT * FROM items WHERE id = ?", (q_item,)).fetchone()
    res = trait_match.context(conn, item, guess.open_boxes(conn, q_item), {}).score_all({}, {})
    t = res[("body", q_body)]
    assert t.facts[p]["tattoo_items"] == 1 and t.terms[p]["tattoo"] == 3.5   # no support from the small ones
    # and small rows alone make no claim, no bar and no guess
    p2 = make_person(conn, "Small")
    for when, variant in (("2025-01-02T07:09:00.000Z", 0), ("2025-02-02T07:09:00.000Z", 1),
                          ("2025-03-02T07:09:00.000Z", 2)):
        ref(conn, p2, when, 5, variant, attrs={"small": True, "kp": 40})
    q2_item, q2_body, _ = query(conn, "2026-09-02T10:00:00.000Z", 5)
    item2 = conn.execute("SELECT * FROM items WHERE id = ?", (q2_item,)).fetchone()
    res2 = trait_match.context(conn, item2, guess.open_boxes(conn, q2_item), {}).score_all({}, {})
    assert ("body", q2_body) not in res2 or p2 not in res2[("body", q2_body)].llr
    assert guess.guess_item(conn, q2_item) == 0


def _one_event(conn, person, seed: int, start_hour: int = 10) -> None:
    """Three auto (weak) photos of `person` with the tattoo `seed`, one day, minutes apart (not near-duplicates)."""
    for k in range(3):
        ref(conn, person, f"2026-01-11T{start_hour:02d}:{10 * k:02d}:00.000Z", seed, k, assigned_by="auto")


def test_three_separate_photos_of_a_named_person_pass_the_bar(conn):
    p = make_person(conn, "Kabir")
    _one_event(conn, p, 1)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    item = conn.execute("SELECT * FROM items WHERE id = ?", (q_item,)).fetchone()
    res = trait_match.context(conn, item, guess.open_boxes(conn, q_item), {}).score_all({}, {})
    f = res[("body", q_body)].facts[p]
    assert f["tattoo_bar"] is True and f["tattoo_items"] == 1   # one date, yet three separate photos
    assert guess.guess_item(conn, q_item) == 1
    assert conn.execute("SELECT guess_person_id FROM bodies WHERE id = ?", (q_body,)).fetchone()[0] == p


def test_three_photos_of_an_unnamed_group_do_not_pass_the_bar(conn):
    g = make_person(conn, None)
    _one_event(conn, g, 1)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    item = conn.execute("SELECT * FROM items WHERE id = ?", (q_item,)).fetchone()
    res = trait_match.context(conn, item, guess.open_boxes(conn, q_item), {}).score_all({}, {})
    assert res[("body", q_body)].facts[g]["tattoo_bar"] is False
    assert guess.guess_item(conn, q_item) == 0


# ------------------------------------------------------------------ admission and tie-breaks
def _result(people: dict[int, dict[str, float]], *, best: int = 25, items: int = 1) -> TraitResult:
    t = TraitResult()
    for pid, terms in people.items():
        full = {"tattoo": terms.get("tattoo", 0.0), "mehndi": 0.0, "nails": terms.get("nails", 0.0), "neg": 0.0,
                "absence": 0.0, "nails_neg": 0.0}
        t.terms[pid] = full
        t.llr[pid] = round(full["tattoo"] + full["nails"], 2)
        t.facts[pid] = {"tattoo_term": full["tattoo"], "tattoo_best": best if full["tattoo"] > 0 else 0,
                        "tattoo_items": items if full["tattoo"] > 0 else 0, "tattoo_bar": True,
                        "tattoo_exclusive": len(people) == 1, "mehndi_term": 0.0, "mehndi_best": 0,
                        "mehndi_items": 0, "mehndi_bar": False, "mehndi_exclusive": False}
    return t


def test_nails_never_lead_an_admission():
    params = ltraits.default_params()
    assert trait_match.admitted(_result({1: {"tattoo": 2.5, "nails": 1.0}}), params) == 1
    # a weak tattoo term (a low-keypoint crop) plus nails over the margin: not admitted
    assert trait_match.admitted(_result({1: {"tattoo": 1.5, "nails": 1.5}}), params) is None
    assert trait_match.admitted(_result({1: {"tattoo": 0.0, "nails": 1.5}}), params) is None


def test_the_margin_leaves_nails_out():
    params = ltraits.default_params()
    t = _result({1: {"tattoo": 2.5, "nails": 1.5}, 2: {"tattoo": 0.6}})
    for pid in (1, 2):
        t.facts[pid]["tattoo_exclusive"] = True   # exclusivity is not what this checks
    # L(1) - L(2) is 3.4, but without nails the lead is 1.9, under the 2.0 margin
    assert trait_match.admitted(t, params) is None
    t2 = _result({1: {"tattoo": 2.5, "nails": 1.5}, 2: {"tattoo": 0.4}})
    for pid in (1, 2):
        t2.facts[pid]["tattoo_exclusive"] = True
    assert trait_match.admitted(t2, params) == 1               # a lead of 2.1 without nails


def test_a_tie_break_follows_the_learned_single_pair_floor():
    t = _result({1: {"tattoo": 3.0}}, best=25, items=1)
    assert trait_match._tie_pair(t, 1, ltraits.default_params())
    strict = ltraits._apply(ltraits.default_params(), {"tattoo_single_min": 30})
    assert not trait_match._tie_pair(t, 1, strict)
    assert trait_match._tie_pair(_result({1: {"tattoo": 3.0}}, best=14, items=2), 1, strict)  # or a second photo


# ------------------------------------------------------------------ propagation hands trait photos to the runner
def test_propagate_leaves_photos_with_trait_rows_to_the_traits_runner(conn, monkeypatch):
    p = make_person(conn, "Kabir")
    a_item, _, _ = ref(conn, p, "2026-01-11T10:00:00.000Z", 1, 0)
    near, near_body, _ = query(conn, "2026-01-11T10:30:00.000Z", 1)      # in the clothes window, with a tattoo row
    plain = photo(conn, "2026-01-11T10:40:00.000Z")                    # in the window, an open box without rows
    make_body(conn, plain["id"], box=(0.1, 0.1, 0.3, 0.6))
    set_traits_ready(conn)
    assert trait_match.deferred(conn, near) and not trait_match.deferred(conn, plain["id"])
    called = []
    real = guess.guess_item
    monkeypatch.setattr(guess, "guess_item", lambda c, iid, cache=None: called.append(iid) or real(c, iid, cache))
    monkeypatch.setattr(traits, "kick", lambda conn=None: None)
    guess.propagate(conn, p, a_item)
    assert near not in called and plain["id"] in called
    assert near in {r[0] for r in conn.execute("SELECT item_id FROM trait_requeue")}
    # with matching off nothing is deferred
    monkeypatch.setattr(settings, "trait_matching", False)
    assert not trait_match.deferred(conn, near)


# ------------------------------------------------------------------ the runner survives one odd photo
def test_an_odd_photo_is_an_error_scan_and_the_run_goes_on(conn, monkeypatch):
    bad = px_photo(conn)
    b_bad = make_body(conn, bad["id"], box=BOX)
    good = px_photo(conn)
    make_body(conn, good["id"], box=BOX)
    fake_engines(monkeypatch)
    real = trait_regions.measure_item

    def measure(c, item_id, e):
        if item_id == bad["id"]:
            raise ValueError("an odd photo")
        return real(c, item_id, e)

    monkeypatch.setattr(trait_regions, "measure_item", measure)
    out = traits.run_pending(conn)
    assert out["complete"] and not out["aborted"]
    st = conn.execute("SELECT status FROM trait_scans WHERE body_id = ?", (b_bad,)).fetchone()
    assert st["status"] == "error"
    assert conn.execute("SELECT COUNT(*) FROM trait_scans WHERE item_id = ? AND status = 'ok'",
                        (good["id"],)).fetchone()[0] == 1


def test_an_engine_failing_on_the_same_photo_again_and_again_marks_it(conn, monkeypatch):
    bad = px_photo(conn)
    b_bad = make_body(conn, bad["id"], box=BOX)
    fake_engines(monkeypatch)

    def measure(c, item_id, e):
        raise trait_regions.EngineError("pose: CUDA error on this photo")

    monkeypatch.setattr(trait_regions, "measure_item", measure)
    for _ in range(traits.ABORTS_PER_ITEM - 1):
        out = traits.run_pending(conn)
        assert out["aborted"] and traits.progress(conn)["last_error"]
        assert conn.execute("SELECT COUNT(*) FROM trait_scans WHERE body_id = ?", (b_bad,)).fetchone()[0] == 0
    out = traits.run_pending(conn)
    assert not out["aborted"]
    assert conn.execute("SELECT status FROM trait_scans WHERE body_id = ?", (b_bad,)).fetchone()[0] == "error"


def test_one_engine_failure_on_another_photo_starts_the_count_again(conn, monkeypatch):
    first, second = px_photo(conn), px_photo(conn)
    make_body(conn, first["id"], box=BOX)
    make_body(conn, second["id"], box=BOX)
    fake_engines(monkeypatch)
    order = iter([first["id"], second["id"]] * 10)
    failing = {"id": None}

    def measure(c, item_id, e):
        failing["id"] = item_id
        raise trait_regions.EngineError("CUDA out of memory")

    monkeypatch.setattr(trait_regions, "measure_item", measure)
    monkeypatch.setattr(traits, "pending_items", lambda c, limit, skip=(): [next(order)])
    for _ in range(4):   # alternating photos: never three in a row on one, so the run always stops
        assert traits.run_pending(conn)["aborted"]
    assert conn.execute("SELECT COUNT(*) FROM trait_scans").fetchone()[0] == 0
