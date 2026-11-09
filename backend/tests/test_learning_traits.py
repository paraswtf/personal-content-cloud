"""The traits learner (app.learning.traits, sections 18 and 34): defaults, the hard-negative raise, the nail
tightening, the owner-outcome tightening, the calibration sample and the training signature."""
from __future__ import annotations

import json

import pytest

from app.analysis import guess, trait_match, trait_sift
from app.analysis import traits as atraits
from app.config import settings
from app.db import kv_set, tx
from app.learning import events, store, train
from app.learning import traits as ltraits
from app.util import now_iso
from tests.test_trait_match import query, ref


@pytest.fixture(autouse=True)
def _fresh_sift():
    trait_sift.reset()
    yield
    trait_sift.reset()


def calib(conn, sample: dict) -> None:
    with tx(conn):
        kv_set(conn, atraits.CALIB_KEY, json.dumps(sample))
        kv_set(conn, atraits.CALIB_AT_KEY, now_iso())


def hn(inliers: int, same_photo: bool = False, kind: str = "tattoo", persons=(1, 2), named: bool = True,
       weak: bool = False) -> dict:
    """A hard negative of the calibration sample: by default two named people, one of them confirmed by the owner."""
    return {"kind": kind, "a": 1, "b": 2, "inliers": inliers, "ratio": 0.6, "same_photo": same_photo,
            "persons": list(persons), "named": named, "weak": weak}


def test_defaults_on_little_data(conn):
    st = ltraits.train(conn)
    assert st["source"] == "default" and st["tattoo_min_inliers"] == 12 and st["nail_color_de"] == 10.0
    p = ltraits.params()
    assert p == ltraits.default_params() and p.tattoo_tiers == ltraits.TATTOO_TIERS
    assert p.single_pair_ok is True and p.tattoo_single_min == 20


def test_hard_negatives_raise_the_floor(conn):
    calib(conn, {"hard_negatives": [hn(14), hn(15, persons=(3, 4)), hn(40, same_photo=True)]})
    ltraits.train(conn)
    assert ltraits.params().tattoo_min_inliers == 12  # two across photos; the one inside a photo does not count
    calib(conn, {"hard_negatives": [hn(14), hn(15, persons=(3, 4)), hn(18), hn(11), hn(90, kind="mehndi")]})
    st = ltraits.train(conn)
    p = ltraits.params()
    # one above the third largest (14): a single high pair never sets the floor
    assert p.tattoo_min_inliers == 15 and p.source == "adjusted" and st["hard_negatives_counted"] == 3
    assert store.load(ltraits.MODEL)["report"]["hard_negative_pairs"] == 2
    calib(conn, {"hard_negatives": [hn(14), hn(15, persons=(3, 4)), hn(80)]})
    ltraits.train(conn)
    assert ltraits.params().tattoo_min_inliers == 15   # the outlier at 80 does not lift it to the cap
    calib(conn, {"hard_negatives": [hn(29), hn(40, persons=(3, 4)), hn(80), hn(90, persons=(5, 6))]})
    ltraits.train(conn)
    assert ltraits.params().tattoo_min_inliers == ltraits.TATTOO_MIN_INLIERS_CAP   # 40 + 1, capped at 30
    assert ltraits.params().tattoo_tiers == ltraits.TATTOO_TIERS  # tiers are never fitted


def test_only_named_confirmed_pairs_of_different_people_raise_the_floor(conn):
    """An unnamed group is usually the same person split in two, and two auto references prove nothing: neither
    raises the floor, and pairs of one couple of people alone do not either."""
    calib(conn, {"hard_negatives": [hn(14, named=False), hn(15, named=False, persons=(3, 4)), hn(18, named=False)]})
    st = ltraits.train(conn)
    assert ltraits.params().tattoo_min_inliers == 12 and st["hard_negatives_counted"] == 0
    calib(conn, {"hard_negatives": [hn(14, weak=True), hn(15, weak=True, persons=(3, 4)), hn(18, weak=True)]})
    ltraits.train(conn)
    assert ltraits.params().tattoo_min_inliers == 12
    calib(conn, {"hard_negatives": [hn(14), hn(15), hn(18)]})   # one pair of people only
    st = ltraits.train(conn)
    assert ltraits.params().tattoo_min_inliers == 12 and st["hard_negatives_counted"] == 3
    assert store.load(ltraits.MODEL)["report"]["hard_negative_pairs"] == 1


def test_the_raise_never_goes_below_the_setting(conn, monkeypatch):
    monkeypatch.setattr(settings, "tattoo_min_inliers", 25)
    calib(conn, {"hard_negatives": [hn(25), hn(26, persons=(3, 4)), hn(27)]})
    ltraits.train(conn)
    assert ltraits.params().tattoo_min_inliers == 26   # one above the third largest (25)
    monkeypatch.setattr(settings, "tattoo_min_inliers", 29)
    assert ltraits.params().tattoo_min_inliers == 29


def test_nail_colour_distance_only_tightens(conn):
    calib(conn, {"nails": {"pos": [12.0] * 25, "neg": [30.0] * 25}})
    ltraits.train(conn)
    assert ltraits.params().nail_color_de == 10.0  # min(q90 12, q10 30) is above 10: never loosens
    calib(conn, {"nails": {"pos": [7.0] * 25, "neg": [8.0] * 25}})
    ltraits.train(conn)
    assert ltraits.params().nail_color_de == 7.0
    calib(conn, {"nails": {"pos": [3.0] * 25, "neg": [2.0] * 25}})
    ltraits.train(conn)
    assert ltraits.params().nail_color_de == 6.0  # floor
    calib(conn, {"nails": {"pos": [7.0] * 10, "neg": [8.0] * 25}})
    ltraits.train(conn)
    assert ltraits.params().nail_color_de == 10.0  # too few positives


def _outcomes(conn, good: int, bad: int) -> None:
    for i in range(good + bad):
        events.record(conn, "guess", 1 if i < good else 0, {"basis": "trait", "trait_llr": 4.0, "score": 0.5})


@pytest.mark.parametrize("good,bad,single,pairs_ok", [(19, 0, 20, True), (19, 1, 20, True), (18, 2, 30, True),
                                                      (16, 4, 30, False)])
def test_owner_outcomes_tighten_admission(conn, good, bad, single, pairs_ok):
    _outcomes(conn, good, bad)
    ltraits.train(conn)
    p = ltraits.params()
    assert p.tattoo_single_min == single and p.single_pair_ok is pairs_ok
    assert p.admission_p >= settings.trait_min_p


def test_learning_off_gives_the_defaults(conn):
    calib(conn, {"hard_negatives": [hn(14), hn(15, persons=(3, 4)), hn(18)]})
    ltraits.train(conn)
    assert ltraits.params().tattoo_min_inliers == 15
    store.set_enabled(conn, False)
    try:
        assert ltraits.params() == ltraits.default_params()
    finally:
        store.set_enabled(conn, True)


def test_params_round_trip():
    p = ltraits._apply(ltraits.default_params(), {"tattoo_min_inliers": 19, "nail_color_de": 7.5, "single_pair_ok": False})
    assert ltraits.TraitParams.from_dict(json.loads(json.dumps(p.to_dict()))) == p


def test_train_verifies_nothing_and_requeues_trait_guesses_when_params_change(conn):
    p = conn.execute("INSERT INTO persons(name, name_key, created_at, updated_at) VALUES ('O', 'o', ?, ?)",
                     (now_iso(), now_iso())).lastrowid
    conn.commit()
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    from tests.helpers import set_traits_ready

    set_traits_ready(conn)
    assert guess.guess_item(conn, q_item) == 1
    before = trait_sift.stats()["verified"]
    ltraits.train(conn)
    assert conn.execute("SELECT COUNT(*) FROM trait_requeue").fetchone()[0] == 0  # unchanged params
    calib(conn, {"hard_negatives": [hn(14), hn(15, persons=(3, 4)), hn(18)]})
    ltraits.train(conn)
    assert trait_sift.stats()["verified"] == before
    assert [r[0] for r in conn.execute("SELECT item_id FROM trait_requeue")] == [q_item]


def test_calibration_sample(conn):
    from tests.helpers import make_person, set_traits_ready

    a, b = make_person(conn, "A"), make_person(conn, "B")
    ref(conn, a, "2026-01-11T07:09:00.000Z", 1, 0)
    ref(conn, a, "2026-02-11T07:09:00.000Z", 1, 1, assigned_by="auto")
    ref(conn, b, "2026-03-11T07:09:00.000Z", 1, 2, assigned_by="auto")  # the same design on someone else
    assert trait_match.calibration_sample(conn) == {}  # not ready
    set_traits_ready(conn)
    assert trait_match.calibration_due(conn)
    s = trait_match.calibration_sample(conn)
    assert s["verified"] <= trait_match.VERIFY_PER_CALIB
    pairs = [(h["kind"], h["persons"], h["same_photo"]) for h in s["hard_negatives"]]
    assert pairs == [("tattoo", sorted([a, b]), False)] * 2
    assert all(h["inliers"] >= 30 for h in s["hard_negatives"])
    # both people are named; one pair has A's owner-confirmed body on one side, the other only auto bodies
    assert sorted((h["named"], h["weak"]) for h in s["hard_negatives"]) == [(True, False), (True, True)]
    assert s["refs"]["tattoo"] == {str(a): {"strong": 1, "weak": 1}, str(b): {"strong": 0, "weak": 1}}
    assert json.loads(conn.execute("SELECT value FROM kv WHERE key = 'trait_calib'").fetchone()[0]) == s
    assert not trait_match.calibration_due(conn)  # fewer than 200 new rows since
    assert trait_match.calibration_sample(conn) == {}
    st = ltraits.train(conn)
    assert st["hard_negatives_counted"] == 1  # the auto-only pair does not count; under 3: no raise
    assert ltraits.params().tattoo_min_inliers == 12


def test_the_signature_follows_calibration_not_rows(conn):
    from tests.helpers import make_person, set_traits_ready

    sig0 = train.signature(conn)
    p = make_person(conn, "A")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0, assigned_by="auto")
    sig0 = train.signature(conn)
    ref(conn, p, "2026-02-11T07:09:00.000Z", 1, 1, assigned_by="auto")
    assert train.signature(conn) == sig0  # a new row alone retrains nothing
    set_traits_ready(conn)
    sig1 = train.signature(conn)
    assert sig1 != sig0
    trait_match.calibration_sample(conn, force=True)
    assert train.signature(conn) != sig1


def test_status_lists_traits_with_the_scan(conn):
    st = train.status(conn)
    t = st["learners"]["traits"]
    assert t["source"] == "default" and t["scan"]["ready"] is False and "eligible" in t["scan"]
