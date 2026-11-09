"""Age in matching (section 10): a face in the child band (a baby or young child, app.analysis.ages) meets the
childhood photos of each person only, with its own cut, a lead over the next person and no learned loosening;
merges and moves learn band by band (a cross-age link teaches nothing), merges the matching sees as unlike teach
nothing either; the learner leaves childhood photos out; childhood suggestions and "who is this as a child". Every
other face is matched exactly as before. Vectors are built with known similarities; the models are off."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.analysis import ages, cluster
from app.analysis.index import _topk_means, get_index
from app.db import kv_set, tx
from app.learning import people as learned_people
from tests.helpers import make_face, make_item, make_person

CHILD = np.array([0.30, 0.55, 0.10, 0.03, 0.01, 0.01, 0.0, 0.0, 0.0], dtype=np.float32)
ADULT = np.array([0.0, 0.01, 0.04, 0.40, 0.40, 0.10, 0.05, 0.0, 0.0], dtype=np.float32)
RNG = np.random.default_rng(11)
BASIS = np.linalg.qr(RNG.standard_normal((512, 64)))[0].T.astype(np.float32)  # 64 orthonormal directions


def vec(*parts: tuple[int, float]) -> np.ndarray:
    """A unit vector from (direction, weight) parts; parts on different directions are orthogonal, so
    cos(vec((0, a), (1, b)), vec((0, 1))) = a / |(a, b)|."""
    v = sum(w * BASIS[d] for d, w in parts)
    return (v / np.linalg.norm(v)).astype(np.float32)


def at(sim: float, d: int, rest: int) -> np.ndarray:
    """A unit vector with cosine `sim` to direction d (the rest along direction `rest`)."""
    return vec((d, sim), (rest, float(np.sqrt(max(0.0, 1.0 - sim * sim)))))


def face(conn, person, emb, groups=None, user=False):
    item = make_item(conn, write_files=False)
    extra = {"age_groups": ages.encode(groups)} if groups is not None else {}
    return make_face(conn, item["id"], person, embedding=emb, assigned_by="user" if user else "auto",
                     write_crop=False, **extra)


def ready(conn, on=True):
    with tx(conn):
        kv_set(conn, ages.READY_KEY, ages.MODEL_REV if on else "")
        from app.db import bump_faces_version

        bump_faces_version(conn)


def options(conn, fid):
    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        rows = cluster._face_rows(conn, [fid])
        return cluster._score(idx, rows, frozenset({fid}), conn)[0]


@pytest.fixture(autouse=True)
def _fresh_learning(monkeypatch):
    monkeypatch.setattr(cluster, "_learned", lambda pid: None)


# ------------------------------------------------------------------ the child band
def test_the_band_and_its_switch(conn):
    assert ages.in_child_band(CHILD, 1.0) and not ages.in_child_band(ADULT, 1.0)
    assert not ages.in_child_band(CHILD, 0.2)  # a blurred face is not judged
    teen = np.array([0.05, 0.36, 0.45, 0.10, 0.04, 0, 0, 0, 0], dtype=np.float32)  # about 11: stays out
    assert not ages.in_child_band(teen, 1.0)
    p = make_person(conn, "Asha")
    kid = face(conn, p, vec((0, 1.0)), CHILD)
    idx = get_index()
    idx.sync(conn)
    assert idx.child_flags([kid]).tolist() == [False]  # not before the library's ages are all read
    ready(conn)
    idx.sync(conn)
    assert idx.child_flags([kid]).tolist() == [True]


def test_adult_faces_score_exactly_as_before(conn):
    a, b = make_person(conn, "Asha"), make_person(conn, "Ravi")
    face(conn, a, at(0.8, 0, 10), ADULT)                     # one adult face: her childhood photo is in the top 3
    for k in range(4):
        face(conn, b, at(0.6, 0, 20 + k), ADULT)
    face(conn, a, at(0.7, 0, 30), CHILD, user=True)
    ready(conn)
    idx = get_index()
    idx.sync(conn)
    q = np.stack([at(0.75, 0, 40), at(0.5, 0, 41)])
    new = idx.person_scores_many(q, 3, child=[False, False])
    base = (idx._person >= 0) & (idx._weight > 0)
    sims = idx._emb @ q.T
    old = [_topk_means(idx._person, base, sims[:, i], 3, idx._user, idx._weight) for i in range(2)]
    assert new == old == idx.person_scores_many(q, 3)


def test_one_childhood_photo_draws_the_next(conn):
    asha = make_person(conn, "Asha")
    for k in range(5):
        face(conn, asha, at(0.3, 0, 10 + k), ADULT)        # her adult faces: 0.3 to the new photo
    face(conn, asha, at(0.62, 0, 20), CHILD, user=True)    # one childhood photo the owner tagged
    new = face(conn, None, vec((0, 1.0)), CHILD)
    assert options(conn, new) == []                         # without ages: a diluted top-3 mean (about 0.42)
    ready(conn)
    opts = options(conn, new)
    assert [p for p, _ in opts] == [asha] and opts[0][1] == pytest.approx(0.62, abs=0.01)


def test_a_child_never_joins_someone_seen_only_as_an_adult(conn):
    dad = make_person(conn, "Ravi")
    for k in range(5):
        face(conn, dad, at(0.56, 0, 10 + k), ADULT)        # the family look
    kid = face(conn, None, vec((0, 1.0)), CHILD)
    assert [p for p, _ in options(conn, kid)] == [dad]     # today it would join him
    ready(conn)
    assert options(conn, kid) == []


def test_the_child_cut_and_the_lead(conn):
    a, b = make_person(conn, "Asha"), make_person(conn, "Meera")
    face(conn, a, at(0.50, 0, 10), CHILD, user=True)
    face(conn, b, at(0.45, 0, 11), CHILD, user=True)
    kid = face(conn, None, vec((0, 1.0)), CHILD)
    ready(conn)
    assert options(conn, kid) == []                       # 0.50 passes the cut (0.48) but leads by 0.05 only
    with tx(conn):
        conn.execute("UPDATE faces SET embedding = ? WHERE person_id = ?", (at(0.40, 0, 11).tobytes(), b))
        from app.db import bump_faces_version

        bump_faces_version(conn)
    get_index().invalidate()
    assert [p for p, _ in options(conn, kid)] == [a]      # a lead of 0.10
    with tx(conn):
        conn.execute("UPDATE faces SET embedding = ? WHERE person_id = ?", (at(0.46, 0, 10).tobytes(), a))
        conn.execute("UPDATE faces SET embedding = ? WHERE person_id = ?", (at(0.30, 0, 11).tobytes(), b))
        bump_faces_version(conn)
    get_index().invalidate()
    assert options(conn, kid) == []                       # under the cut


def test_a_learned_threshold_only_tightens_a_childs_cut(conn, monkeypatch):
    a = make_person(conn, "Asha")
    face(conn, a, at(0.52, 0, 10), CHILD, user=True)
    kid = face(conn, None, vec((0, 1.0)), CHILD)
    adult = face(conn, None, at(0.40, 0, 12), ADULT)
    face(conn, a, at(0.40 * 0.40 + 0.6, 12, 13), ADULT, user=True)
    ready(conn)
    weak = face(conn, None, at(0.40 / 0.52 if 0.40 < 0.52 else 1.0, 0, 14), CHILD)  # 0.40 to her childhood photo
    monkeypatch.setattr(cluster, "_learned", lambda pid: 0.30 if pid == a else None)  # loosened by adults
    assert [p for p, _ in options(conn, kid)] == [a]       # 0.52 >= the child cut
    assert options(conn, weak) == []                        # 0.40: over her loosened 0.30, under the child cut
    monkeypatch.setattr(cluster, "_learned", lambda pid: 0.60 if pid == a else None)  # tightened by rejections
    assert options(conn, kid) == []
    assert adult  # the adult face is not part of this check


# ------------------------------------------------------------------ learning
def _events(conn, person):
    return [json.loads(r[0]) for r in conn.execute(
        "SELECT features FROM learning_events WHERE kind = 'merge' AND person_id = ? ORDER BY id", (person,))]


def test_a_childhood_group_merged_into_an_adult_teaches_nothing(conn):
    asha = make_person(conn, "Asha")
    for k in range(4):
        face(conn, asha, at(0.9, 0, 10 + k), ADULT, user=True)
    group = make_person(conn, None)
    face(conn, group, at(0.5, 0, 20), CHILD)                 # 0.45 to her faces: would loosen her to 0.43
    face(conn, group, at(0.5, 0, 21), CHILD)
    lookalike = make_person(conn, None)                      # another childhood group, like the merged one
    face(conn, lookalike, vec((0, 0.5), (20, 0.6), (21, 0.6)), CHILD)
    adults = make_person(conn, None)                         # an adult group like her: not pulled by this merge
    face(conn, adults, at(0.9, 0, 22), ADULT)
    ready(conn)
    before = learned_people.person_thresholds(conn).get(asha)
    accepted, followers = cluster.merge_and_follow(conn, asha, [group])
    assert accepted is None and followers == [lookalike]    # the childhood follow, nothing for adults
    ev = _events(conn, asha)[-1]
    assert ev["cross_age"] is True and ev["accepted"] is None
    assert learned_people.person_thresholds(conn).get(asha) == before


def test_a_merge_the_matching_sees_as_unlike_teaches_nothing(conn):
    asha = make_person(conn, "Asha")
    for k in range(4):
        face(conn, asha, at(0.9, 0, 10 + k), ADULT, user=True)
    other = make_person(conn, None)
    face(conn, other, at(0.3, 0, 20), ADULT)              # 0.27 to her faces: above MIN_SIM, under the floor
    accepted, followers = cluster.merge_and_follow(conn, asha, [other])
    assert accepted is None and followers == []
    ev = _events(conn, asha)[-1]
    assert ev["accepted"] is None and ev["below_floor"] == pytest.approx(0.27, abs=0.01)
    assert asha not in learned_people.person_thresholds(conn)


def test_old_merges_under_the_floor_no_longer_loosen(conn):
    asha = make_person(conn, "Asha")
    for k in range(2):
        face(conn, asha, at(0.9, 0, 10 + k), ADULT)
    from app.learning import events

    with tx(conn):
        events.record(conn, "merge", 1, {"accepted": 0.2692, "sources": 1}, person_id=asha)
    assert asha not in learned_people.person_thresholds(conn)
    with tx(conn):
        events.record(conn, "merge", 1, {"accepted": 0.40, "sources": 1}, person_id=asha)
    assert learned_people.person_thresholds(conn)[asha] == pytest.approx(0.38)


def test_the_learner_leaves_childhood_photos_out(conn):
    asha = make_person(conn, "Asha")
    for k in range(4):
        face(conn, asha, at(0.85, 0, 10 + k), ADULT, user=True)
    without = learned_people.person_thresholds(conn).get(asha)
    face(conn, asha, at(0.2, 0, 20), CHILD, user=True)
    face(conn, asha, at(0.2, 0, 21), CHILD, user=True)
    loosened = learned_people.person_thresholds(conn).get(asha)
    assert loosened is not None and (without is None or loosened < without)  # before the ages are read
    ready(conn)
    assert learned_people.person_thresholds(conn).get(asha) == without


def test_a_child_to_child_merge_pulls_childhood_groups_only(conn):
    asha = make_person(conn, "Asha")
    for k in range(3):
        face(conn, asha, at(0.3, 0, 10 + k), ADULT, user=True)
    face(conn, asha, at(0.9, 0, 20), CHILD, user=True)
    tagged = make_person(conn, None)
    face(conn, tagged, at(0.9, 0, 21), CHILD)
    lookalike = make_person(conn, None)
    face(conn, lookalike, at(0.97, 0, 22), CHILD)        # 0.87 to both: as alike as the merged pair (0.81)
    adult_group = make_person(conn, None)
    face(conn, adult_group, at(0.32, 0, 23), ADULT)
    ready(conn)
    accepted, followers = cluster.merge_and_follow(conn, asha, [tagged])
    assert accepted is None and followers == [lookalike]
    ev = _events(conn, asha)[-1]
    assert ev["child_accepted"] == pytest.approx(0.81, abs=0.05) and not ev.get("cross_age")
    assert conn.execute("SELECT person_id FROM faces WHERE person_id = ?", (adult_group,)).fetchone() is not None


# ------------------------------------------------------------------ suggestions
def test_childhood_suggestions_and_who_is_this_as_a_child(client, conn):
    asha = make_person(conn, "Asha")
    for k in range(3):
        face(conn, asha, at(0.30, 0, 10 + k), ADULT, user=True)
    face(conn, asha, at(0.9, 1, 20), CHILD, user=True)        # her childhood photo
    ravi = make_person(conn, "Ravi")
    for k in range(3):
        face(conn, ravi, at(0.10, 0, 30 + k), ADULT, user=True)
    group = make_person(conn, None)                          # childhood photos, unnamed
    face(conn, group, vec((0, 0.5), (1, 0.8)), CHILD)
    face(conn, group, vec((0, 0.5), (1, 0.85)), CHILD)
    ready(conn)
    pairs = cluster.childhood_suggestions(conn)
    assert [(a, b) for a, b, _ in pairs] == [tuple(sorted((asha, group)))]
    r = client.get("/api/suggestions?limit=10").json()["items"]
    young = [s for s in r if s["kind"] == "childhood"]
    assert len(young) == 1 and len(young[0]["a_samples"]) >= 1 and len(young[0]["b_samples"]) >= 1
    ranked = cluster.childhood_candidates(conn, group)
    assert [p for p, _ in ranked] == [asha, ravi]
    out = client.get(f"/api/people/{group}/childhood").json()["candidates"]
    assert [c["person"]["id"] for c in out] == [asha, ravi]
    assert client.get(f"/api/people/{asha}/childhood").json() == {"candidates": []}  # named: not a childhood group


def test_nothing_changes_while_age_matching_is_off(conn, monkeypatch):
    from app.config import settings

    dad = make_person(conn, "Ravi")
    for k in range(5):
        face(conn, dad, at(0.56, 0, 10 + k), ADULT)
    kid = face(conn, None, vec((0, 1.0)), CHILD)
    ready(conn)
    monkeypatch.setattr(settings, "face_age_matching", False)
    get_index().invalidate()
    assert [p for p, _ in options(conn, kid)] == [dad]


def test_a_bulk_move_of_childhood_photos_is_learned_as_cross_age(conn):
    asha = make_person(conn, "Asha")
    for k in range(4):
        face(conn, asha, at(0.9, 0, 10 + k), ADULT, user=True)
    moved = [face(conn, None, at(0.5, 0, 20 + k), CHILD) for k in range(2)]
    adults = make_person(conn, None)
    face(conn, adults, at(0.9, 0, 30), ADULT)
    ready(conn)
    with tx(conn):
        conn.execute(f"UPDATE faces SET person_id = ?, assigned_by = 'user' WHERE id IN ({moved[0]}, {moved[1]})", (asha,))
        from app.db import bump_faces_version

        bump_faces_version(conn)
        accepted, pulled = cluster.accept_faces(conn, asha, moved)
    assert accepted is None and pulled == []
    ev = _events(conn, asha)[-1]
    assert ev["cross_age"] is True and ev["faces"] == 2


def test_who_is_this_ranks_matching_childhood_photos_first(conn):
    asha = make_person(conn, "Asha")
    for k in range(3):
        face(conn, asha, at(0.10, 0, 10 + k), ADULT, user=True)   # her adult photos: unlike the group
    face(conn, asha, at(0.9, 1, 20), CHILD, user=True)           # her childhood photo: like the group
    ravi = make_person(conn, "Ravi")
    for k in range(3):
        face(conn, ravi, at(0.6, 0, 30 + k), ADULT, user=True)    # his adult photos: closer to the group than hers
    group = make_person(conn, None)
    face(conn, group, vec((0, 0.5), (1, 0.8)), CHILD)
    face(conn, group, vec((0, 0.5), (1, 0.85)), CHILD)
    ready(conn)
    assert [p for p, _ in cluster.childhood_candidates(conn, group)] == [asha, ravi]


def test_the_suggestions_route_adds_childhood_pairs_beyond_the_limit(client, conn):
    a, b = make_person(conn, "Asha"), make_person(conn, "Meera")      # look alike as adults: a usual pair
    face(conn, a, at(0.9, 0, 10), ADULT, user=True)
    face(conn, b, at(0.9, 0, 11), ADULT, user=True)
    c, d = make_person(conn, None), make_person(conn, None)           # their childhood photos look alike
    face(conn, c, at(0.9, 5, 12), CHILD)
    face(conn, d, at(0.9, 5, 13), CHILD)
    ready(conn)
    items = client.get("/api/suggestions?limit=1").json()["items"]
    kinds = sorted((s["kind"], tuple(sorted((s["a"]["id"], s["b"]["id"])))) for s in items)
    assert kinds == [("childhood", tuple(sorted((c, d)))), ("same", tuple(sorted((a, b))))]
    # a pair in both lists shows as a childhood one
    face(conn, a, at(0.9, 6, 14), CHILD, user=True)        # unlike c and d's childhood photos
    face(conn, b, at(0.9, 6, 15), CHILD, user=True)
    items = client.get("/api/suggestions?limit=5").json()["items"]
    pair = [s for s in items if {s["a"]["id"], s["b"]["id"]} == {a, b}]
    assert len(pair) == 1 and pair[0]["kind"] == "childhood"


def test_no_childhood_card_on_vault_photos_alone(client, conn):
    c, d = make_person(conn, None), make_person(conn, None)
    face(conn, c, at(0.9, 5, 12), CHILD)
    hidden = face(conn, d, at(0.9, 5, 13), CHILD)
    face(conn, d, at(0.9, 7, 16), ADULT)                     # d has a visible item, its childhood photo is in the vault
    with tx(conn):
        conn.execute("UPDATE items SET nsfw = 1 WHERE id = (SELECT item_id FROM faces WHERE id = ?)", (hidden,))
    ready(conn)
    assert [(a, b) for a, b, _ in cluster.childhood_suggestions(conn)] == [tuple(sorted((c, d)))]
    items = client.get("/api/suggestions?limit=5").json()["items"]
    assert not [s for s in items if s["kind"] == "childhood"]

