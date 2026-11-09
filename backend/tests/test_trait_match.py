"""Body trait matching (app.analysis.trait_match, section 34) with real SIFT on synthetic tattoos.

tattoo_crop(seed, variant=k) is another photo of the same tattoo (210 or more inliers); different seeds never verify
at 12. Vectors are hand-made so retrieval finds what a test means it to find."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.analysis import guess, trait_match, trait_sift
from app.config import settings
from app.db import kv_set, tx
from app.util import now_iso
from tests.helpers import (make_body, make_face, make_item, make_person, make_scan, make_trait, set_traits_ready,
                           tattoo_crop)

_CROPS: dict[tuple, np.ndarray] = {}


@pytest.fixture(autouse=True)
def _fresh_sift():
    trait_sift.reset()
    yield
    trait_sift.reset()


def crop(seed: int, **kw) -> np.ndarray:
    key = (seed, tuple(sorted(kw.items())))
    if key not in _CROPS:
        _CROPS[key] = tattoo_crop(seed, **kw)
    return _CROPS[key]


def vec(seed: int, base: np.ndarray | None = None, noise: float = 0.3) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
    v /= np.linalg.norm(v)
    if base is not None:
        v = base + noise * v
        v /= np.linalg.norm(v)
    return v.astype(np.float32)


DESIGN = vec(1000)


def photo(conn, when: str, **kw):
    kw.setdefault("exif_datetime", when.replace("T", " ")[:19])
    return make_item(conn, taken_at=when, write_files=False, data=None, **kw)


def ref(conn, person: int | None, when: str, seed: int, variant: int, *, assigned_by: str = "user", part="forearm_l",
        kind: str = "tattoo", v: np.ndarray | None = None, attrs: dict | None = None, item=None, box=(0.1, 0.1, 0.3, 0.6),
        **kw) -> tuple[int, int, int]:
    """A photo (or `item`) with a body of `person` carrying the tattoo `seed`. Returns (item id, body id, row id)."""
    it = item if item is not None else photo(conn, when)
    body = make_body(conn, it["id"], person, assigned_by=assigned_by if person else "auto", box=box)
    row = make_trait(conn, body, kind, part, vec=v if v is not None else vec(seed * 31 + variant, DESIGN, 0.2),
                     detail=crop(seed, variant=variant), attrs=attrs, **kw)
    return int(it["id"]), body, row


def query(conn, when: str, seed: int, variant: int = 9, **kw) -> tuple[int, int, int]:
    return ref(conn, None, when, seed, variant, **kw)


def body_row(conn, body_id: int):
    return conn.execute("SELECT * FROM bodies WHERE id = ?", (body_id,)).fetchone()


def evidence(conn, body_id: int) -> dict:
    return json.loads(body_row(conn, body_id)["guess_evidence"] or "null") or {}


def result_for(conn, item_id: int):
    """The TraitResult per open box of the item (no clothes), and the context."""
    item = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    boxes = guess.open_boxes(conn, item_id)
    ctx = trait_match.context(conn, item, boxes, {})
    assert ctx is not None
    return ctx.score_all({}, {}), ctx


# ------------------------------------------------------------------ tattoo
def test_a_verified_tattoo_names_the_person_months_later(conn):
    p = make_person(conn, "Kabir")
    _, ref_body, ref_row = ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    q_item, q_body, q_row = query(conn, "2026-09-01T10:00:00.000Z", 1)
    assert guess.guess_item(conn, q_item) == 0  # not ready yet: nothing
    set_traits_ready(conn)
    assert guess.guess_item(conn, q_item) == 1
    row = body_row(conn, q_body)
    assert row["guess_person_id"] == p and row["assigned_by"] == "guess"
    ev = evidence(conn, q_body)
    assert ev["basis"] == "trait" and ev["trait_llr"] == 3.5  # 30 or more inliers, one strong reference
    assert ev["score"] == round(trait_match.posterior(3.5), 4) == row["guess_score"]
    t = ev["traits"][0]
    assert t["kind"] == "tattoo" and t["trait_id"] == q_row and t["ref_trait_id"] == ref_row
    assert t["ref_body_id"] == ref_body and t["inliers"] >= 30 and t["strong"] is True and t["ref_before"] is True
    assert t["days_apart"] == 233.1 and ev["ref_body_id"] == ref_body
    for k in ("clothes", "scene", "scene_term", "same_day", "minutes_apart", "gps_m"):
        assert k not in ev
    assert guess.guess_item(conn, q_item) == 0  # deterministic: the same JSON again


def test_trait_only_guesses_switch_off(conn, monkeypatch):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    monkeypatch.setattr(settings, "trait_only_guesses", False)
    assert guess.guess_item(conn, q_item) == 0
    monkeypatch.setattr(settings, "trait_only_guesses", True)
    monkeypatch.setattr(settings, "trait_matching", False)
    assert guess.guess_item(conn, q_item) == 0


def test_one_weak_reference_is_halved_and_not_enough(conn):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0, assigned_by="auto")
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    t = res[("body", q_body)]
    assert t.terms[p]["tattoo"] == 1.75 and t.facts[p]["tattoo_bar"] is False
    assert guess.guess_item(conn, q_item) == 0


def test_two_weak_references_on_two_dates_pass_the_bar(conn):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0, assigned_by="auto")
    ref(conn, p, "2026-03-02T07:09:00.000Z", 1, 1, assigned_by="auto")
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    f = res[("body", q_body)].facts[p]
    assert f["tattoo_bar"] and f["tattoo_items"] == 2
    assert res[("body", q_body)].terms[p]["tattoo"] == 4.0  # 3.5 plus one further date
    assert guess.guess_item(conn, q_item) == 1
    assert evidence(conn, q_body)["traits"][0]["support"] == 1


def test_high_cosine_without_verification_is_nothing(conn):
    p = make_person(conn, "Kabir")
    v = vec(5)
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0, v=v)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 2, v=v)  # same vector, another design
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    assert res == {}
    assert guess.guess_item(conn, q_item) == 0


def test_a_mirror_photo_verifies(conn):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0, part="forearm_r")
    it = photo(conn, "2026-09-01T10:00:00.000Z")
    body = make_body(conn, it["id"])
    make_trait(conn, body, "tattoo", "forearm_l", vec=vec(77, DESIGN, 0.2), detail=crop(1, mirror=True))
    set_traits_ready(conn)
    assert guess.guess_item(conn, it["id"]) == 1
    t = evidence(conn, body)["traits"][0]
    assert t["mirrored"] is True and t["part"] == "forearm_l" and t["ref_part"] == "forearm_r"


def test_a_small_row_gives_nothing(conn):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1, attrs={"small": True, "kp": 40})
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    # a small mark verifies but is worth nothing: no claim at all (no term, support or bar for anyone)
    assert ("body", q_body) not in res or res[("body", q_body)].terms.get(p, {}).get("tattoo", 0.0) == 0.0
    assert guess.guess_item(conn, q_item) == 0


def test_few_keypoints_scale_the_term(conn):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1, attrs={"kp": 150})
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    assert res[("body", q_body)].terms[p]["tattoo"] == 1.75  # 3.5 x 150 / 300


# ------------------------------------------------------------------ shared design and the same photo
def test_a_design_two_people_share_counts_for_no_one(conn):
    p, q = make_person(conn, "A"), make_person(conn, "B")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    ref(conn, q, "2026-02-11T07:09:00.000Z", 1, 1)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    assert res == {} or all(v == 0 for v in res[("body", q_body)].llr.values())
    assert guess.guess_item(conn, q_item) == 0


def test_the_strong_owner_of_a_design_keeps_it_against_one_weak_carrier(conn):
    kabir, leela = make_person(conn, "Kabir"), make_person(conn, "Leela")
    for k, day in enumerate(("2026-01-05", "2026-01-08", "2026-01-11")):
        ref(conn, kabir, f"{day}T07:09:00.000Z", 1, k)
    ref(conn, leela, "2026-01-12T07:09:00.000Z", 1, 3, assigned_by="auto")  # his arm in her box
    q_item, q_body, _ = query(conn, "2026-01-20T10:00:00.000Z", 1)
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    t = res[("body", q_body)]
    assert leela not in t.llr or t.llr[leela] == 0
    assert t.facts[kabir]["tattoo_exclusive"] is True
    assert t.terms[kabir]["tattoo"] == round(3.5 * 0.75 + 1.0, 2)
    assert guess.guess_item(conn, q_item) == 1
    assert body_row(conn, q_body)["guess_person_id"] == kabir


def test_a_strong_reference_on_the_minor_side_makes_it_shared(conn):
    kabir, other = make_person(conn, "Kabir"), make_person(conn, "Other")
    for k, day in enumerate(("2026-01-05", "2026-01-08", "2026-01-11")):
        ref(conn, kabir, f"{day}T07:09:00.000Z", 1, k)
    ref(conn, other, "2026-01-12T07:09:00.000Z", 1, 3, assigned_by="user")
    q_item, _, _ = query(conn, "2026-01-20T10:00:00.000Z", 1)
    set_traits_ready(conn)
    assert guess.guess_item(conn, q_item) == 0


def test_another_body_in_the_query_photo_with_the_design_zeroes_the_row(conn):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    it = conn.execute("SELECT * FROM items WHERE id = ?", (q_item,)).fetchone()
    ref(conn, None, "", 1, 5, item=it, box=(0.5, 0.1, 0.3, 0.6))
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    assert not res or all(t.terms.get(p, {}).get("tattoo", 0) == 0 for t in res.values())
    assert guess.guess_item(conn, q_item) == 0


def test_a_reference_photo_where_two_bodies_verify_is_dropped(conn):
    p, q = make_person(conn, "Kabir"), make_person(conn, "Mohan")
    item_id, _, _ = ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    it = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    ref(conn, q, "", 1, 1, item=it, box=(0.5, 0.1, 0.3, 0.6))
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    assert res == {}


# ------------------------------------------------------------------ absence
def _track(conn, p, fam="forearm", ink=0.3, days=("2026-01-01", "2026-03-01")):
    for k, day in enumerate(days):
        ref(conn, p, f"{day}T10:00:00.000Z", 3, k, part=f"{fam}_l", attrs={"ink": ink})


def _bare_query(conn, when, fams=("forearm",), sides=("l", "r"), length=200.0, extra_rows=False):
    it = photo(conn, when)
    body = make_body(conn, it["id"])
    bare = {f"{f}_{s}": {"p": 0.9, "len": length} for f in fams for s in sides}
    make_scan(conn, body, bare=bare)
    if extra_rows:
        make_trait(conn, body, "tattoo", "calf_l", vec=vec(4242), detail=crop(30))
    return it["id"], body


def test_absence_on_bare_arms_of_someone_with_a_lasting_tattoo(conn):
    p = make_person(conn, "Kabir")
    _track(conn, p)
    q_item, q_body = _bare_query(conn, "2026-06-01T10:00:00.000Z")
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    t = res[("body", q_body)]
    assert t.terms[p]["absence"] == -0.7 and t.llr[p] == -0.7
    assert t.evidence[p]["against"] == [{"kind": "tattoo", "part": "forearm", "llr": -0.7, "why": "bare"}]


def test_absence_is_capped(conn):
    p = make_person(conn, "Kabir")
    _track(conn, p)
    _track(conn, p, fam="upper_arm")
    _track(conn, p, fam="calf")
    q_item, q_body = _bare_query(conn, "2026-06-01T10:00:00.000Z", fams=("forearm", "upper_arm", "calf"))
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    assert res[("body", q_body)].terms[p]["absence"] == -1.4


@pytest.mark.parametrize("case", ["one_event", "one_side", "before", "small_ink", "no_ink", "short"])
def test_no_absence(conn, case):
    p = make_person(conn, "Kabir")
    if case == "one_event":
        _track(conn, p, days=("2026-01-01", "2026-01-05"))
    elif case == "small_ink":
        _track(conn, p, ink=0.05)
    elif case == "no_ink":
        for k, day in enumerate(("2026-01-01", "2026-03-01")):
            ref(conn, p, f"{day}T10:00:00.000Z", 3, k)
    else:
        _track(conn, p)
    when = "2025-06-01T10:00:00.000Z" if case == "before" else "2026-06-01T10:00:00.000Z"
    q_item, q_body = _bare_query(conn, when, sides=("l",) if case == "one_side" else ("l", "r"),
                                 length=100.0 if case == "short" else 200.0)
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    assert ("body", q_body) not in res or res[("body", q_body)].terms.get(p, {}).get("absence", 0) == 0


def test_a_sleeve_gives_no_absence(conn):
    """A forearm the scan read as a tattoo is not bare: no absence for that family."""
    p = make_person(conn, "Kabir")
    _track(conn, p)
    it = photo(conn, "2026-06-01T10:00:00.000Z")
    body = make_body(conn, it["id"])
    make_trait(conn, body, "tattoo", "forearm_r", vec=vec(4243), detail=crop(31),
               bare={"forearm_l": {"p": 0.9, "len": 200}, "forearm_r": {"p": 0.9, "len": 200}})
    set_traits_ready(conn)
    res, _ = result_for(conn, it["id"])
    assert res.get(("body", body)) is None or res[("body", body)].terms.get(p, {}).get("absence", 0) == 0


# ------------------------------------------------------------------ nails and mehndi
def test_decays_and_rarity():
    assert trait_match.nails_decay(1) == 1.0
    assert round(trait_match.nails_decay(3), 2) == 0.84
    assert trait_match.nails_decay(6) == 0.5
    assert trait_match.nails_decay(20) == 0.0 and trait_match.nails_decay(30) == 0.0
    assert trait_match.mehndi_decay(3) == 1.0 and round(trait_match.mehndi_decay(10), 3) == round(0.5 ** 1.4, 3)
    assert trait_match.mehndi_decay(25) == 0.0
    assert trait_match.rarity(0.02) == 1.0 and trait_match.rarity(0.001) == 1.0
    assert round(trait_match.rarity(0.25), 2) == 0.35
    assert round(trait_match.rarity(0.6), 2) == 0.13
    assert round(trait_match.nail_de((50, 10, 10), (54, 13, 14)), 3) == round((4 + 9 + 16) ** 0.5, 3)


BLUE = [40.0, 10.0, -45.0]


def nails(conn, person, when, lab=BLUE, item=None, assigned_by="user", box=(0.1, 0.1, 0.3, 0.6), **attrs):
    it = item if item is not None else photo(conn, when, **({} if "forward" not in attrs else {"forward": attrs.pop("forward")}))
    body = make_body(conn, it["id"], person, assigned_by=assigned_by if person else "auto", box=box)
    a = {"lab": lab, "color": "blue", "hex": "#2040c0", "finish": "art", "finish_p": 0.9, "length": "long", "back": True}
    a.update(attrs)
    make_trait(conn, body, "nails", "hand_r", attrs=a)
    return it["id"], body


def test_nails_support_within_days(conn):
    p = make_person(conn, "Priya")
    nails(conn, p, "2026-05-01T10:00:00.000Z")
    q_item, q_body = nails(conn, None, "2026-05-02T10:00:00.000Z")
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    t = res[("body", q_body)]
    # few polished hands in the library: f = 0.25 (0.35), plus art 0.5 and long 0.2, 1 day apart
    assert t.terms[p]["nails"] == round(trait_match.rarity(0.25) + 0.5 + 0.2, 2)
    e = t.evidence[p]["traits"][0]
    assert e["kind"] == "nails" and e["de"] == 0.0 and e["decay"] == 1.0 and e["color"] == "blue"
    assert guess.guess_item(conn, q_item) == 0  # nails never name anyone alone


def test_nails_further_dates_and_cap(conn):
    p = make_person(conn, "Priya")
    nails(conn, p, "2026-05-01T10:00:00.000Z")
    nails(conn, p, "2026-05-14T10:00:00.000Z")
    q_item, q_body = nails(conn, None, "2026-05-01T12:00:00.000Z")
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    best = trait_match.rarity(0.25) + 0.7
    assert res[("body", q_body)].terms[p]["nails"] == round(min(1.5, best + 0.25 * trait_match.nails_decay(13 - 1 / 12)), 2)


def test_nails_other_colour_gives_nothing(conn):
    p = make_person(conn, "Priya")
    nails(conn, p, "2026-05-01T10:00:00.000Z", lab=[50.0, 60.0, 40.0])
    q_item, q_body = nails(conn, None, "2026-05-02T10:00:00.000Z")
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    t = res.get(("body", q_body))
    assert t is not None and t.terms[p]["nails"] == 0.0 and t.terms[p]["nails_neg"] == -0.5  # far colour, 1 day


@pytest.mark.parametrize("case", ["named", "same_photo", "forward", "no_exif"])
def test_nails_exclusivity_and_dates(conn, case):
    p = make_person(conn, "Priya")
    if case in ("forward", "no_exif"):
        it = photo(conn, "2026-05-01T10:00:00.000Z", **({"forward": 1} if case == "forward" else {"exif_datetime": None}))
        nails(conn, p, "", item=it)
    else:
        nails(conn, p, "2026-05-01T10:00:00.000Z")
    q_item, q_body = nails(conn, None, "2026-05-02T10:00:00.000Z")
    if case == "named":
        nails(conn, make_person(conn, "Other"), "2026-05-03T10:00:00.000Z")
    if case == "same_photo":
        it = conn.execute("SELECT * FROM items WHERE id = ?", (q_item,)).fetchone()
        nails(conn, None, "", item=it, box=(0.5, 0.1, 0.3, 0.6))
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    t = res.get(("body", q_body))
    assert t is None or t.terms.get(p, {}).get("nails", 0) == 0


def test_an_unnamed_body_elsewhere_does_not_block_nails(conn):
    p = make_person(conn, "Priya")
    nails(conn, p, "2026-05-01T10:00:00.000Z")
    nails(conn, None, "2026-05-03T10:00:00.000Z")
    q_item, q_body = nails(conn, None, "2026-05-02T10:00:00.000Z")
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    assert res[("body", q_body)].terms[p]["nails"] > 0


@pytest.mark.parametrize("days,expect", [(2, 1.0), (10, round(0.5 ** 1.4, 2)), (25, None)])
def test_mehndi_decays(conn, days, expect, monkeypatch):
    p = make_person(conn, "Bride")
    ref(conn, p, "2026-02-01T10:00:00.000Z", 7, 0, kind="mehndi", part="hand_r")
    q_item, q_body, _ = ref(conn, None, f"2026-02-{1 + days:02d}T10:00:00.000Z" if days < 28 else "", 7, 1,
                            kind="mehndi", part="hand_l")
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    t = res.get(("body", q_body))
    if expect is None:
        assert t is None or t.terms.get(p, {}).get("mehndi", 0) == 0
    else:
        # 60 or more inliers: 1.5 x decay, one strong reference
        assert t.terms[p]["mehndi"] == round(1.5 * expect, 2)
    assert guess.guess_item(conn, q_item) == 0  # mehndi alone guesses nothing by default


def test_mehndi_of_the_bride_and_a_bridesmaid_in_one_photo(conn):
    p = make_person(conn, "Bride")
    ref(conn, p, "2026-02-01T10:00:00.000Z", 7, 0, kind="mehndi", part="hand_r")
    q_item, q_body, _ = ref(conn, None, "2026-02-02T10:00:00.000Z", 7, 1, kind="mehndi", part="hand_l")
    it = conn.execute("SELECT * FROM items WHERE id = ?", (q_item,)).fetchone()
    ref(conn, None, "", 7, 2, kind="mehndi", part="hand_r", item=it, box=(0.5, 0.1, 0.3, 0.6))
    set_traits_ready(conn)
    res, _ = result_for(conn, q_item)
    t = res.get(("body", q_body))
    assert t is None or t.terms.get(p, {}).get("mehndi", 0) == 0


# ------------------------------------------------------------------ near-duplicates
@pytest.mark.parametrize("rule", ["dup", "live", "phash", "burst"])
def test_near_duplicates_give_no_evidence(conn, rule):
    p = make_person(conn, "Kabir")
    q_when = "2026-01-11T07:10:00.000Z"
    kw_ref: dict = {}
    kw_q: dict = {}
    if rule == "dup":
        kw_q["dup_group_id"] = None
    if rule == "live":
        kw_ref["content_id"] = kw_q["content_id"] = "ABC"
    if rule == "phash":
        kw_ref["phash"], kw_q["phash"] = "ffff0000ffff0000", "ffff0000ffff0007"
    ref_item = photo(conn, "2026-01-11T07:09:00.000Z", **kw_ref)
    ref(conn, p, "", 1, 0, item=ref_item)
    q = photo(conn, q_when, **kw_q)
    if rule == "dup":
        with tx(conn):
            conn.execute("UPDATE items SET dup_group_id = ? WHERE id IN (?, ?)", (ref_item["id"], ref_item["id"], q["id"]))
    if rule == "burst":
        from tests.test_analysis_guess import _scene, _vec

        _scene(conn, ref_item["id"], _vec(1))
        _scene(conn, q["id"], _vec(1))
    q_body = make_body(conn, q["id"])
    make_trait(conn, q_body, "tattoo", "forearm_l", vec=vec(99, DESIGN, 0.2), detail=crop(1, variant=9))
    set_traits_ready(conn)
    res, _ = result_for(conn, q["id"])
    assert res == {}


def test_hours_apart_is_not_a_near_duplicate(conn):
    """Bodies 14836 and 14913: the same sleeve hours apart (no scene burst, no dup group)."""
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-11T00:50:00.000Z", 1, 0)
    q_item, q_body, _ = query(conn, "2026-01-11T07:09:00.000Z", 1)
    set_traits_ready(conn)
    assert guess.guess_item(conn, q_item) == 1
    assert body_row(conn, q_body)["guess_person_id"] == p


# ------------------------------------------------------------------ hides
def hide(conn, item_id, kind, verdict, region, body_box=None, src=None):
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO trait_hides(item_id, kind, verdict, x, y, w, h, bx, by, bw, bh, src_trait_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (item_id, kind, verdict, *region, *(body_box or (None, None, None, None)), src, now_iso()))
        return int(cur.lastrowid)


def rows_of(conn, ids):
    marks = ",".join("?" * len(ids))
    return conn.execute(f"SELECT id, body_id, item_id, kind, part, x, y, w, h FROM body_traits WHERE id IN ({marks})",
                        ids).fetchall()


def test_hide_rules_a_b_c_without_verification(conn):
    p = make_person(conn, "Kabir")
    item_id, body, row = ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    region = (0.2, 0.4, 0.05, 0.1)
    before = trait_sift.stats()["verified"]
    assert trait_match.hidden_ids(conn, rows_of(conn, [row])) == set()
    h_a = hide(conn, item_id, "tattoo", "not_trait", region)
    assert trait_match.hidden_ids(conn, rows_of(conn, [row])) == {row}
    assert trait_match.hides_covering(conn, row) == [h_a]
    with tx(conn):
        conn.execute("DELETE FROM trait_hides")
    hide(conn, item_id, "tattoo", "not_theirs", region, body_box=(0.6, 0.1, 0.3, 0.6))  # another body's box
    assert trait_match.hidden_ids(conn, rows_of(conn, [row])) == set()
    h_b = hide(conn, item_id, "tattoo", "not_theirs", region, body_box=(0.1, 0.1, 0.3, 0.6))
    assert trait_match.hidden_ids(conn, rows_of(conn, [row])) == {row}
    assert trait_match.hides_covering(conn, row) == [h_b]
    with tx(conn):
        conn.execute("DELETE FROM trait_hides")
    # rule (c): a look-alike hit from a hide on another photo, kept visible by a 'shown' hide here
    other = photo(conn, "2026-02-01T10:00:00.000Z")
    h_c = hide(conn, other["id"], "tattoo", "not_trait", region)
    with tx(conn):
        conn.execute("INSERT INTO trait_hide_hits(trait_id, hide_id) VALUES (?, ?)", (row, h_c))
    assert trait_match.hidden_ids(conn, rows_of(conn, [row])) == {row}
    assert trait_match.hidden_by_lookalike(conn, row) is True
    assert trait_match.hides_covering(conn, row) == []
    hide(conn, item_id, "tattoo", "shown", region)
    assert trait_match.hidden_ids(conn, rows_of(conn, [row])) == set()
    assert trait_match.hidden_by_lookalike(conn, row) is False
    assert trait_sift.stats()["verified"] == before


def _desc(conn, hide_id, row_id):
    with tx(conn):
        conn.execute("INSERT INTO trait_hide_desc(hide_id, vec, detail, created_at) "
                     "SELECT ?, vec, detail, ? FROM body_traits WHERE id = ?", (hide_id, now_iso(), row_id))


def test_look_alikes_are_verified(conn):
    """A not_trait hide on printed fabric hides its verified repeats elsewhere, not a merely similar design."""
    item_id, body, row = ref(conn, None, "2026-01-11T07:09:00.000Z", 1, 0)
    _, _, same = ref(conn, None, "2026-02-11T07:09:00.000Z", 1, 1)
    _, _, other = ref(conn, None, "2026-03-11T07:09:00.000Z", 2, 0, v=vec(1 * 31 + 0, DESIGN, 0.2))
    h = hide(conn, item_id, "tattoo", "not_trait", (0.2, 0.4, 0.05, 0.1), src=row)
    _desc(conn, h, row)
    set_traits_ready(conn)
    assert trait_match.hide_hits_for_hide(conn, h) == 2  # its own row and the repeat
    hits = {r[0] for r in conn.execute("SELECT trait_id FROM trait_hide_hits WHERE hide_id = ?", (h,))}
    assert hits == {row, same} and other not in hits
    with tx(conn):
        conn.execute("DELETE FROM trait_hide_hits")
    assert trait_match.hide_hits_for_rows(conn, [same, other]) == 1


def test_a_hidden_reference_gives_no_evidence(conn):
    p = make_person(conn, "Kabir")
    item_id, _, row = ref(conn, p, "2026-01-11T07:09:00.000Z", 1, 0)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    hide(conn, item_id, "tattoo", "not_trait", (0.2, 0.4, 0.05, 0.1))
    assert guess.guess_item(conn, q_item) == 0


# ------------------------------------------------------------------ budget, child band, determinism
def test_at_most_60_verifications_per_guess_item(conn):
    for k in range(25):
        p = make_person(conn, f"P{k}")
        for d in range(3):
            ref(conn, p, f"2026-0{1 + d}-{10 + k:02d}T10:00:00.000Z", 100 + k * 3 + d, 0,
                v=vec(7000 + k * 3 + d, DESIGN, 0.1))
    q_item, _, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    before = trait_sift.stats()["verified"]
    guess.guess_item(conn, q_item)
    assert trait_sift.stats()["verified"] - before <= trait_match.VERIFY_PER_ITEM


def _child_face(conn, item_id, person):
    from app.analysis import ages

    groups = np.array([0.6, 0.3, 0.05, 0.05, 0, 0, 0, 0, 0], np.float32)
    return make_face(conn, item_id, person, assigned_by="user", det_score=0.95, size_px=200,
                     age_groups=ages.encode(groups))


def test_a_child_body_is_never_a_tattoo_reference(conn, monkeypatch):
    from app.analysis import ages

    p = make_person(conn, "Kid")
    it = photo(conn, "2026-01-11T07:09:00.000Z")
    face = _child_face(conn, it["id"], p)
    body = make_body(conn, it["id"], p, assigned_by="user", face_id=face)
    make_trait(conn, body, "tattoo", "forearm_l", vec=vec(55, DESIGN, 0.2), detail=crop(1, variant=0))
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    with tx(conn):
        kv_set(conn, ages.READY_KEY, ages.MODEL_REV)
    monkeypatch.setattr(settings, "face_age_matching", True)
    assert guess.guess_item(conn, q_item) == 0
    # with face age matching off there is no child band (the documented gap)
    monkeypatch.setattr(settings, "face_age_matching", False)
    assert guess.guess_item(conn, q_item) == 1


def test_scores_are_deterministic(conn):
    p = make_person(conn, "Kabir")
    for k, day in enumerate(("2026-01-05", "2026-02-08")):
        ref(conn, p, f"{day}T07:09:00.000Z", 1, k)
    q_item, q_body, _ = query(conn, "2026-09-01T10:00:00.000Z", 1)
    set_traits_ready(conn)
    guess.guess_item(conn, q_item)
    first = body_row(conn, q_body)["guess_evidence"]
    trait_sift.reset()
    with tx(conn):
        conn.execute("UPDATE bodies SET assigned_by = 'auto', guess_person_id = NULL, guess_score = NULL, "
                     "guess_evidence = NULL WHERE id = ?", (q_body,))
    guess.guess_item(conn, q_item)
    assert body_row(conn, q_body)["guess_evidence"] == first


def test_person_groups_and_count(conn):
    p = make_person(conn, "Kabir")
    ref(conn, p, "2026-01-05T07:09:00.000Z", 1, 0)
    ref(conn, p, "2026-02-05T07:09:00.000Z", 1, 1)
    ref(conn, p, "2026-02-05T07:09:00.000Z", 2, 0, part="calf_r")
    nails(conn, p, now_iso(-86400.0))
    set_traits_ready(conn)
    g = trait_match.person_groups(conn, p, vault_open=False)
    assert [x["part"] for x in g["tattoos"]] == ["forearm_l", "calf_r"]
    assert g["tattoos"][0]["photos"] == 2 and g["tattoos"][0]["first_seen"].startswith("2026-01-05")
    assert g["tattoos"][0]["best"]["crop_url"].startswith("/files/trait/")
    assert len(g["nails"]) == 1 and g["nails"][0]["hex"] == "#2040c0" and g["mehndi"] == []
    assert trait_match.trait_count(conn, p) == 4
