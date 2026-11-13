"""Following people from one photo to the next (section 35): app.analysis.place_match on hand-made links and boxes."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.analysis import ages, place, place_align
from app.analysis import place_match as pm
from app.analysis.guess import Box, open_boxes
from app.config import settings
from app.db import kv_set, tx
from app.util import now_iso
from tests.helpers import make_body, make_face, make_item, make_person

IDENT = place_align.IDENTITY
ADULT = np.array([0.0, 0.01, 0.04, 0.40, 0.40, 0.10, 0.05, 0.0, 0.0], dtype=np.float32)
CHILD = np.array([0.30, 0.60, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
LEFT, MID, RIGHT = (0.10, 0.20, 0.08, 0.12), (0.45, 0.20, 0.08, 0.12), (0.80, 0.20, 0.08, 0.12)


def photo(conn, second: float):
    whole, frac = divmod(float(second), 1.0)
    m, s = divmod(int(whole), 60)
    extra = {"exif_subsec": str(int(round(frac * 100))).zfill(2)} if frac else {}
    return make_item(conn, taken_at="2026-05-01T10:%02d:%02d.000Z" % (m, s), write_files=False, data=None, **extra)


def link(conn, a, b, *, m=IDENT, kind="still", dt=None, inliers=120) -> place.Link:
    dt = abs(place.capture_time(b) - place.capture_time(a)) if dt is None else dt
    with tx(conn):
        conn.execute("INSERT OR REPLACE INTO place_links(a_item, b_item, rev, kind, m, inliers, spread, dt_s, computed_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, 0.5, ?, ?)",
                     (a["id"], b["id"], place.PLACE_REV, kind, json.dumps(list(m)) if kind != "none" else "", inliers, dt,
                      now_iso()))
    return place.Link(a["id"], b["id"], kind, tuple(m), inliers, dt)


def face(conn, item, person=None, box=MID, *, user=True, **extra) -> int:
    return make_face(conn, item["id"], person, box=box, assigned_by="user" if (person and user) else "auto",
                     write_crop=False, **extra)


def body_at(conn, item, hx: float, hy: float, *, person=None, width: float = 0.256, **extra) -> int:
    """A faceless person box whose head point is (hx, hy) and whose unit width is `width`, in a 640 x 480 photo."""
    bw = width / pm.BOX_WIDTH
    return make_body(conn, item["id"], person, assigned_by="user" if person else "auto",
                     box=(hx - bw / 2, hy - pm.HEAD_DOWN * bw * 640 / 480, bw, 0.6), **extra)


def pair(conn, ln: place.Link) -> tuple[pm.PairMatch, list[pm.Unit], list[pm.Unit]]:
    info = pm.load_info(conn, [ln.a_item, ln.b_item])
    units = pm.load_units(conn, [ln.a_item, ln.b_item], {i: v.width / v.height for i, v in info.items()})
    return pm.match_pair(ln, units[ln.a_item], units[ln.b_item], info[ln.a_item], info[ln.b_item]), units[ln.a_item], units[ln.b_item]


def unit_vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(768).astype(np.float32)
    return v / np.linalg.norm(v)


def near(base: np.ndarray, cos: float, seed: int) -> np.ndarray:
    """A unit vector at the given cosine to base."""
    r = unit_vec(seed)
    r = r - (r @ base) * base
    r /= np.linalg.norm(r)
    return (cos * base + np.sqrt(1 - cos * cos) * r).astype(np.float32)


# ------------------------------------------------------------------ units
def test_units_are_faces_with_their_box_and_faceless_boxes(conn):
    p = make_person(conn, "Asha")
    a = photo(conn, 0)
    f1 = face(conn, a, p, LEFT)
    f2 = face(conn, a, None, RIGHT)
    b1 = make_body(conn, a["id"], p, face_id=f1, box=(0.05, 0.15, 0.2, 0.7))
    b2 = make_body(conn, a["id"], None, box=(0.4, 0.3, 0.2, 0.6))
    with tx(conn):  # a guessed box has no person as a reference
        conn.execute("UPDATE faces SET assigned_by = 'guess', guess_person_id = ? WHERE id = ?", (p, f2))
    units = pm.load_units(conn, [a["id"]])[a["id"]]
    assert [(u.kind, u.id) for u in units] == [("face", f1), ("face", f2), ("body", b2)]
    u1, u2, u3 = units
    assert (u1.person_id, u1.body_id, u1.face_id) == (p, b1, f1) and u2.person_id is None
    assert (u1.hx, u1.hy) == pytest.approx((0.14, 0.26)) and u1.width == pytest.approx(pm.FACE_WIDTHS * 0.08)
    assert (u3.hx, u3.hy, u3.width) == pytest.approx((0.5, 0.3 + pm.HEAD_DOWN * 0.2, pm.BOX_WIDTH * 0.2))
    assert u3.body_id == b2
    wide = pm.load_units(conn, [a["id"]], {a["id"]: 640 / 480})[a["id"]][2]     # box widths in a 4:3 picture
    assert wide.hy == pytest.approx(0.3 + pm.HEAD_DOWN * 0.2 * 640 / 480)
    assert pm.load_info(conn, [a["id"]])[a["id"]].t == place.capture_time(a)


def test_only_live_dated_photos_have_info(conn):
    v = make_item(conn, taken_at="2026-05-01T10:00:00.000Z", write_files=False, data=None, kind="video")
    x = make_item(conn, taken_at="2026-05-01T10:00:01.000Z", write_files=False, data=None, nsfw=1)
    t = make_item(conn, taken_at="2026-05-01T10:00:02.000Z", write_files=False, data=None, deleted_at=now_iso())
    u = make_item(conn, write_files=False, data=None)
    assert pm.load_info(conn, [v["id"], x["id"], t["id"], u["id"]]) == {}


# ------------------------------------------------------------------ one pair
def test_the_same_spot_is_a_match_and_another_spot_is_not(conn):
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, p, LEFT)
    face(conn, a, None, RIGHT)
    face(conn, b, None, (0.105, 0.2, 0.08, 0.12))       # where the person stood
    face(conn, b, None, (0.50, 0.6, 0.08, 0.12))        # somewhere new
    m, ua, ub = pair(conn, link(conn, a, b))
    assert m.pairs == {0: 0} and m.residual[0] < 0.05 and not m.swapped and m.kept == 0


def test_the_link_moves_the_points(conn):
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, None, LEFT)
    face(conn, b, None, (0.30, 0.25, 0.08, 0.12))
    assert pair(conn, link(conn, a, b))[0].pairs == {}
    shift = (1.0, 0.0, 0.20, 0.0, 1.0, 0.05)             # the camera panned
    m, _, _ = pair(conn, link(conn, a, b, m=shift, kind="moved"))
    assert m.pairs == {0: 0} and m.residual[0] == pytest.approx(0.0, abs=1e-6)
    zoom = (2.0, 0.0, 0.0, 0.0, 2.0, 0.0)                # twice as big: the box must be twice as wide too
    assert pair(conn, link(conn, a, b, m=zoom, kind="moved"))[0].pairs == {}


def test_two_candidates_equally_near_give_no_match(conn):
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, None, MID)
    face(conn, b, None, (0.43, 0.2, 0.08, 0.12))
    face(conn, b, None, (0.47, 0.2, 0.08, 0.12))
    assert pair(conn, link(conn, a, b))[0].pairs == {}


def test_a_box_of_another_size_is_someone_else(conn):
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, None, (0.40, 0.20, 0.10, 0.15))
    face(conn, b, None, (0.425, 0.24, 0.05, 0.07))       # the same centre, half the size
    assert pair(conn, link(conn, a, b))[0].pairs == {}


def test_a_faceless_box_matches_a_face_at_its_head(conn):
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, None, (0.46, 0.20, 0.08, 0.12))        # head point (0.50, 0.26), width 0.256
    body_at(conn, b, 0.50, 0.26)
    m, ua, ub = pair(conn, link(conn, a, b))
    assert m.pairs == {0: 0} and (ua[0].kind, ub[0].kind) == ("face", "body") and m.residual[0] < 0.01


def test_a_faceless_box_may_sit_a_little_further_off_than_a_face(conn):
    off = (pm.RESIDUAL_MAX + pm.RESIDUAL_MAX_BOX) / 2      # between the two bounds, in widths of the person
    width = pm.FACE_WIDTHS * 0.08
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, None, (0.20, 0.20, 0.08, 0.12))
    face(conn, b, None, (0.20 + off * width, 0.20, 0.08, 0.12))
    assert pair(conn, link(conn, a, b))[0].pairs == {}
    c, d = photo(conn, 10), photo(conn, 12)
    face(conn, c, None, (0.20, 0.20, 0.08, 0.12))        # head point (0.24, 0.26)
    body_at(conn, d, 0.24 + off * width, 0.26)
    assert pair(conn, link(conn, c, d))[0].pairs == {0: 0}


def test_known_people_who_swapped_places_spoil_the_pair(conn):
    p, q = make_person(conn, "Asha"), make_person(conn, "Ravi")
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, p, LEFT)
    face(conn, a, q, RIGHT)
    face(conn, a, None, MID)
    face(conn, b, q, LEFT)                               # Ravi now stands where Asha stood
    face(conn, b, None, MID)
    m, _, _ = pair(conn, link(conn, a, b))
    assert m.swapped
    ln = place.links_for(conn, [a["id"]])[0]
    units = pm.load_units(conn, [a["id"], b["id"]])
    assert pm.link_llr(ln, m, 2, units[a["id"]][2], units[b["id"]][1]) is None


def test_known_people_in_their_place_are_support(conn):
    p, q = make_person(conn, "Asha"), make_person(conn, "Ravi")
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, p, LEFT)
    face(conn, a, q, RIGHT)
    face(conn, a, None, MID)
    face(conn, b, p, LEFT)
    face(conn, b, q, RIGHT)
    face(conn, b, None, MID)
    m, ua, ub = pair(conn, link(conn, a, b))
    assert not m.swapped and m.kept == 2 and m.kept_people == {p, q} and m.pairs == {0: 0, 1: 1, 2: 2}
    ln = place.links_for(conn, [a["id"]])[0]
    alone = pm.tier_llr(2.0)
    assert pm.link_llr(ln, m, 2, ua[2], ub[2]) == pytest.approx(alone + pm.KEPT_MAX)
    assert pm.link_llr(ln, m, 0, ua[0], ub[0], exclude=p) == pytest.approx(alone + pm.KEPT_BONUS)   # not their own place


def test_clothes_say_no_or_back_the_link(conn):
    a, b = photo(conn, 0), photo(conn, 2)
    t = unit_vec(1)
    face(conn, a, None, LEFT, torso=t)
    face(conn, a, None, MID, torso=t)
    face(conn, a, None, RIGHT, torso=t)
    face(conn, b, None, LEFT, torso=near(t, pm.CLOTHES_NO - 0.05, 2))     # another outfit in that spot
    face(conn, b, None, MID, torso=near(t, pm.CLOTHES_YES + 0.05, 3))
    face(conn, b, None, RIGHT)                                            # no clothes vector: position alone
    m, ua, ub = pair(conn, link(conn, a, b))
    assert set(m.pairs) == {1, 2} and m.clothes[2] is None and m.clothes[1] == pytest.approx(pm.CLOTHES_YES + 0.05, abs=1e-4)
    ln = place.links_for(conn, [a["id"]])[0]
    assert pm.link_llr(ln, m, 1, ua[1], ub[1]) == pytest.approx(pm.tier_llr(2.0) + pm.CLOTHES_BONUS)
    assert pm.link_llr(ln, m, 2, ua[2], ub[2]) == pytest.approx(pm.tier_llr(2.0))


def test_a_face_and_a_faceless_box_compare_clothes_more_loosely(conn):
    a, b = photo(conn, 0), photo(conn, 2)
    t = unit_vec(1)
    cos = (pm.CLOTHES_NO + pm.CLOTHES_NO_MIXED) / 2
    face(conn, a, None, (0.46, 0.20, 0.08, 0.12), torso=t)
    body_at(conn, b, 0.50, 0.26, torso=near(t, cos, 2))
    assert pair(conn, link(conn, a, b))[0].pairs == {0: 0}
    with tx(conn):
        conn.execute("UPDATE bodies SET torso_embedding = ? WHERE item_id = ?",
                     (near(t, pm.CLOTHES_NO_MIXED - 0.05, 3).tobytes(), b["id"]))
    assert pair(conn, link(conn, a, b))[0].pairs == {}


def test_a_child_and_an_adult_are_never_one_person(conn):
    with tx(conn):
        kv_set(conn, ages.READY_KEY, ages.MODEL_REV)
    a, b = photo(conn, 0), photo(conn, 2)
    clear = dict(clarity=25.0, sharpness=40.0)
    face(conn, a, None, MID, age_groups=ages.encode(ADULT), **clear)
    face(conn, b, None, MID, age_groups=ages.encode(CHILD), **clear)
    m, ua, ub = pair(conn, link(conn, a, b))
    assert (ua[0].child, ub[0].child) == (False, True) and m.pairs == {}


def test_tiers_by_the_gap():
    assert pm.tier_llr(0.5) == pm.TIERS[0][1]
    assert pm.tier_llr(pm.TIERS[-1][0]) == pm.TIERS[-1][1]
    assert pm.tier_llr(pm.TIERS[-1][0] + 1) is None
    gaps = [g for g, _ in pm.TIERS]
    strengths = [s for _, s in pm.TIERS]
    assert gaps == sorted(gaps) and strengths == sorted(strengths, reverse=True)


# ------------------------------------------------------------------ chains
def results(conn, item, cache=None):
    row = conn.execute("SELECT id, uid, kind, taken_at, gps_lat, gps_lon, nsfw, deleted_at FROM items WHERE id = ?",
                       (item["id"],)).fetchone()
    boxes: list[Box] = open_boxes(conn, item["id"])
    ctx = pm.context(conn, row, boxes, cache)
    return {} if ctx is None else ctx.score_all({}, {})


def test_a_box_is_followed_to_a_known_person_one_photo_away(conn):
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, 2)
    fa = face(conn, a, p, MID)
    fb = face(conn, b, None, MID)
    link(conn, a, b)
    res = results(conn, b)[("face", fb)]
    assert res.llr == {p: pytest.approx(pm.tier_llr(2.0))}
    frag = res.evidence[p]
    assert frag == {"ref_item_uid": a["uid"], "ref_face_id": fa, "ref_body_id": None, "seconds_apart": 2.0,
                    "ref_before": True, "hops": 1, "offset": 0.0, "kind": "still", "others": 0, "clothes": None,
                    "llr": round(pm.tier_llr(2.0), 2)}
    assert list(frag) == ["ref_item_uid", "ref_face_id", "ref_body_id", "seconds_apart", "ref_before", "hops", "offset",
                          "kind", "others", "clothes", "llr"]      # a fixed key order: the stored JSON is compared
    # and the other way round: the known person is in the later photo
    c = photo(conn, 4)
    fc = face(conn, c, None, MID)
    link(conn, b, c)
    with tx(conn):
        conn.execute("UPDATE faces SET person_id = NULL, assigned_by = 'auto' WHERE id = ?", (fa,))
        conn.execute("UPDATE faces SET person_id = ?, assigned_by = 'user' WHERE id = ?", (p, fc))
    back = results(conn, b)[("face", fb)]
    assert back.evidence[p]["ref_before"] is False and back.evidence[p]["ref_face_id"] == fc


def test_a_faceless_box_costs_a_little_and_names_its_reference(conn):
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, 2)
    ref = make_body(conn, a["id"], p, assigned_by="user", box=(0.37, 0.1, 0.26, 0.7))
    tgt = make_body(conn, b["id"], None, box=(0.37, 0.1, 0.26, 0.7))
    link(conn, a, b)
    res = results(conn, b)[("body", tgt)]
    assert res.llr[p] == pytest.approx(pm.tier_llr(2.0) - pm.FACELESS_COST)
    assert (res.evidence[p]["ref_body_id"], res.evidence[p]["ref_face_id"]) == (ref, None)


def test_beyond_a_few_seconds_position_never_names_alone(conn):
    p, q = make_person(conn, "Asha"), make_person(conn, "Ravi")
    for gap, far in ((2, False), (8, True), (20, True)):
        a, b = photo(conn, 100 * gap), photo(conn, 100 * gap + gap)
        face(conn, a, p, MID)
        make_body(conn, a["id"], p, assigned_by="user", box=(0.67, 0.1, 0.26, 0.7))
        tf = face(conn, b, None, MID)
        tb = make_body(conn, b["id"], None, box=(0.67, 0.1, 0.26, 0.7))
        face(conn, a, q, LEFT)
        face(conn, b, q, LEFT)                           # someone else stayed in place: support
        link(conn, a, b)
        got = results(conn, b)
        for key, cost in ((("face", tf), 0.0), (("body", tb), pm.FACELESS_COST)):
            full = pm.tier_llr(gap) - cost + pm.KEPT_BONUS
            assert got[key].llr[p] == pytest.approx(min(full, pm.FAR_LLR) if far else full)
            assert (pm.admitted(got[key]) is None) == far
    assert pm.FAR_LLR < pm.ADMIT_LLR <= pm.tier_llr(pm.ALONE_S) - pm.FACELESS_COST


def test_a_chain_through_open_boxes_is_as_strong_as_its_weakest_link(conn):
    p = make_person(conn, "Asha")
    a, b, c, d, e = (photo(conn, s) for s in (0, 2, 4, 30, 32))
    face(conn, a, p, MID)
    for it in (b, c, d):
        face(conn, it, None, MID)
    fe = face(conn, e, None, MID)
    link(conn, a, b)
    link(conn, b, c)
    link(conn, c, d)                                     # 26 s: a weaker tier, and beyond ALONE_S
    link(conn, d, e)
    fd = conn.execute("SELECT id FROM faces WHERE item_id = ?", (d["id"],)).fetchone()[0]
    fc = conn.execute("SELECT id FROM faces WHERE item_id = ?", (c["id"],)).fetchone()[0]
    two = results(conn, c)[("face", fc)]
    assert two.llr[p] == pytest.approx(pm.tier_llr(2.0) - pm.HOP_COST) and two.evidence[p]["hops"] == 2
    assert two.evidence[p]["seconds_apart"] == 4.0 and two.evidence[p]["ref_item_uid"] == a["uid"]
    three = results(conn, d)[("face", fd)]
    weakest = min(pm.tier_llr(26.0), pm.FAR_LLR)
    assert three.llr[p] == pytest.approx(weakest - 2 * pm.HOP_COST) and three.evidence[p]["hops"] == 3
    assert results(conn, e) == {}                        # four photos away: too far


def test_a_chain_stops_at_the_first_known_person(conn):
    p, q = make_person(conn, "Asha"), make_person(conn, "Ravi")
    a, b, c = photo(conn, 0), photo(conn, 2), photo(conn, 4)
    face(conn, a, q, MID)                                # further back the spot was Ravi's
    face(conn, b, p, MID)
    fc = face(conn, c, None, MID)
    link(conn, a, b)                                     # a swap: this pair gives nothing anyway
    link(conn, b, c)
    assert set(results(conn, c)[("face", fc)].llr) == {p}


def test_people_already_in_the_photo_are_not_proposed(conn):
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, p, MID)
    face(conn, b, None, MID)
    make_body(conn, b["id"], p, assigned_by="user", box=(0.7, 0.3, 0.2, 0.6))    # tagged elsewhere in the photo
    link(conn, a, b)
    assert results(conn, b) == {}


def test_no_link_no_context_and_the_cache_is_shared(conn, monkeypatch):
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, p, MID)
    fb = face(conn, b, None, MID)
    assert results(conn, b) == {}
    link(conn, a, b, kind="none")
    assert results(conn, b) == {}
    link(conn, a, b)
    cache: dict = {}
    assert results(conn, b, cache)[("face", fb)].llr
    assert set(cache["place"]["links"]) >= {b["id"]} and (a["id"], b["id"]) in cache["place"]["pairs"]
    monkeypatch.setattr(settings, "place_enabled", False)
    assert results(conn, b) == {}


def test_a_burst_without_a_fit_is_weaker(conn):
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, 1)
    face(conn, a, p, MID)
    fb = face(conn, b, None, MID)
    link(conn, a, b, kind="burst", inliers=0)
    res = results(conn, b)[("face", fb)]
    assert res.llr[p] == pytest.approx(pm.BURST_LLR) and res.evidence[p]["kind"] == "burst"
    assert pm.BURST_LLR < pm.ADMIT_LLR                   # it supports and breaks ties, it never names alone


# ------------------------------------------------------------------ what a chain may not do
def test_a_chain_of_three_photos_never_names_alone(conn):
    p = make_person(conn, "Asha")
    a, b, c, d = (photo(conn, s) for s in (0, 2, 4, 6))
    face(conn, a, p, MID)
    face(conn, b, None, MID)
    fc = face(conn, c, None, MID)
    fd = face(conn, d, None, MID)
    for x, y in ((a, b), (b, c), (c, d)):                              # only neighbours lined up
        link(conn, x, y)
    two = results(conn, c)[("face", fc)]
    assert two.llr[p] == pytest.approx(pm.tier_llr(2.0) - pm.HOP_COST) and pm.admitted(two) == p
    three = results(conn, d)[("face", fd)]
    assert three.evidence[p]["hops"] == 3 and three.llr[p] == pytest.approx(pm.LONG_CHAIN_LLR)
    assert pm.admitted(three) is None and pm.LONG_CHAIN_LLR < pm.ADMIT_LLR


def test_a_chain_is_void_when_a_photo_on_the_way_says_no(conn):
    p = make_person(conn, "Asha")
    z, y, x = photo(conn, 0), photo(conn, 2), photo(conn, 4)
    face(conn, z, p, MID)
    mid = face(conn, y, None, MID)
    fx = face(conn, x, None, MID)
    link(conn, z, y)
    link(conn, y, x)
    assert results(conn, x)[("face", fx)].llr[p] == pytest.approx(pm.tier_llr(2.0) - pm.HOP_COST)
    with tx(conn):                                                     # the owner said no for the box in the middle
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (mid, p, now_iso()))
    assert results(conn, x) == {}
    with tx(conn):
        conn.execute("DELETE FROM guess_rejections")
    assert p in results(conn, x)[("face", fx)].llr
    face(conn, y, p, RIGHT)                                            # the middle photo shows the person elsewhere
    assert results(conn, x) == {}


def test_a_child_is_not_followed_to_an_adult_through_a_faceless_box(conn):
    with tx(conn):
        kv_set(conn, ages.READY_KEY, ages.MODEL_REV)
    p = make_person(conn, "Asha")
    z, y, x = photo(conn, 0), photo(conn, 2), photo(conn, 4)
    clear = dict(clarity=25.0, sharpness=40.0)
    face(conn, z, p, MID, age_groups=ages.encode(ADULT), **clear)
    body_at(conn, y, MID[0] + MID[2] / 2, MID[1] + MID[3] / 2)         # no face: no age band of its own
    fx = face(conn, x, None, MID, age_groups=ages.encode(CHILD), **clear)
    link(conn, z, y)
    link(conn, y, x)
    assert results(conn, x) == {}
    with tx(conn):
        conn.execute("UPDATE faces SET age_groups = ? WHERE id = ?", (ages.encode(ADULT), fx))
    assert p in results(conn, x)[("face", fx)].llr


def test_a_weaker_direct_link_does_not_hide_a_stronger_chain(conn):
    p = make_person(conn, "Asha")
    x, w, y, z = (photo(conn, s) for s in (0, 2, 4, 6))
    fx = face(conn, x, None, MID)
    face(conn, w, None, MID)
    face(conn, y, None, MID)
    face(conn, z, p, MID)
    for m, n in ((x, w), (w, y), (y, z)):
        link(conn, m, n)
    chain = results(conn, x)[("face", fx)].llr[p]
    assert chain == pytest.approx(pm.LONG_CHAIN_LLR)
    link(conn, x, y, dt=200.0)                                         # a weak link straight to the third photo
    assert pm.tier_llr(200.0) - pm.HOP_COST < chain
    assert results(conn, x)[("face", fx)].llr[p] == pytest.approx(chain)


def test_a_photo_that_became_a_document_no_longer_takes_part(conn):
    p = make_person(conn, "Asha")
    a, b = photo(conn, 0), photo(conn, 2)
    face(conn, a, p, MID)
    fb = face(conn, b, None, MID)
    link(conn, a, b)
    assert p in results(conn, b)[("face", fb)].llr
    for category in ("documents", "screenshots"):
        with tx(conn):
            conn.execute("UPDATE items SET category = ? WHERE id = ?", (category, b["id"]))
        assert results(conn, b) == {}
    with tx(conn):
        conn.execute("UPDATE items SET category = 'misc' WHERE id = ?", (b["id"],))
        conn.execute("UPDATE items SET category = 'documents' WHERE id = ?", (a["id"],))
    assert results(conn, b) == {}                                      # nor is anyone followed from one


def test_links_written_while_a_cache_is_shared_are_seen(conn):
    p = make_person(conn, "Asha")
    n, b, c = photo(conn, 0), photo(conn, 2), photo(conn, 4)
    face(conn, n, p, MID)
    fb = face(conn, b, None, MID)
    face(conn, c, None, MID)
    link(conn, b, c)
    cache: dict = {}
    assert results(conn, c, cache) == {}                               # b's links are in the cache now
    link(conn, n, b)                                                   # the runner writes a link meanwhile
    assert p in results(conn, b, cache)[("face", fb)].llr
