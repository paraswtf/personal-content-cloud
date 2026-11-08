"""Face weights (section 10, 2026-11-08): every face counts by how clear and visible it is, from 0 (turned
away, a helmet or a mask, blurred, not a face) to 1. The weight decides how much a face is a person's
example in matching and centroids, how strict its own cut is, whether it may start a person, and how much
it moves a learned threshold. Re-detection carries the owner's decisions over when a video's old tracks
become fewer, longer ones."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import cluster
from app.analysis.faces import face_weight, face_weights, is_clear_face
from app.analysis.index import get_index
from app.config import settings
from app.db import tx
from app.learning import people as people_learning
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person

DIM = 512
CLEAR = {"clarity": 24.0, "sharpness": 45.0}
HELMET = {"clarity": 9.0, "sharpness": 6.0}


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _rand(seed: int) -> np.ndarray:
    return _unit(np.random.default_rng(seed).standard_normal(DIM))


def _near(base: np.ndarray, seed: int, sigma: float = 0.25) -> np.ndarray:
    n = np.random.default_rng(seed).standard_normal(DIM)
    n -= (n @ base) * base
    return _unit(base + sigma * _unit(n))


def test_weight_ramps_and_old_faces():
    assert face_weight(0.9, 120, 24.0, 45.0) == pytest.approx(1.0)
    assert face_weight(0.9, 120, 9.0, 45.0) == 0.0  # turned away or covered
    assert face_weight(0.9, 120, 24.0, 4.0) == 0.0  # blurred
    assert face_weight(0.9, 120) == pytest.approx(1.0)  # measured before clarity existed: detector score and size
    assert face_weight(0.65, 120) == pytest.approx(0.5)
    assert face_weight(0.9, settings.face_min_size_px) == 0.0
    w = face_weights(np.array([0.9, 0.9]), np.array([120, 120]), np.array([np.nan, 17.5]), np.array([np.nan, 17.5]))
    assert w[0] == pytest.approx(1.0) and 0.0 < w[1] < 1.0
    assert is_clear_face(0.9, 120, **CLEAR) and not is_clear_face(0.9, 120, **HELMET)
    assert not is_clear_face(0.6, 120, **CLEAR)  # the good-face rule still holds


def test_an_unclear_face_teaches_nothing_about_who_someone_is(conn):
    item = make_item(conn)
    pid = make_person(conn, "Asha")
    face, helmet = _rand(1), _rand(2)
    make_face(conn, item["id"], pid, embedding=face, assigned_by="user", **CLEAR)
    make_face(conn, item["id"], pid, embedding=helmet, assigned_by="user", box=(0.6, 0.2, 0.2, 0.3), **HELMET)
    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        assert [p for p, _ in idx.person_scores(helmet, 3)] == []  # another helmet does not look like Asha
        ((p, score),) = idx.person_scores(face, 3)
        pids, cents = idx.centroids()
    assert p == pid and score == pytest.approx(1.0, abs=1e-5)
    assert list(pids) == [pid] and float(cents[0] @ face) == pytest.approx(1.0, abs=1e-5)


def test_a_half_clear_example_counts_half(conn):
    item = make_item(conn)
    pid = make_person(conn, "Ravi")
    a, b = _rand(3), _rand(4)
    make_face(conn, item["id"], pid, embedding=a, **CLEAR)
    make_face(conn, item["id"], pid, embedding=b, box=(0.6, 0.2, 0.2, 0.3), clarity=17.5, sharpness=45.0)
    query = _unit(a + b)
    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        ((_, score),) = idx.person_scores(query, 3)
    s_a, s_b = float(query @ a), float(query @ b)
    w_b = face_weight(0.9, 120, 17.5, 45.0)
    assert score == pytest.approx((s_a + w_b * s_b) / (1 + w_b), abs=1e-4)


def test_the_cut_rises_as_the_weight_falls_and_only_clear_faces_start_people(conn):
    def row(**kw):
        base = {"det_score": 0.9, "size_px": 120, "clarity": None, "sharpness": None}
        base.update(kw)
        return base

    good, low = settings.cluster_match_threshold, settings.cluster_match_threshold_low
    assert cluster._threshold(row(**CLEAR)) == pytest.approx(good)
    assert cluster._threshold(row(**HELMET)) == pytest.approx(low)
    half = face_weight(0.9, 120, 17.5, 45.0)
    assert cluster._threshold(row(clarity=17.5, sharpness=45.0)) == pytest.approx(low - (low - good) * half)

    item = make_item(conn)
    clear = make_face(conn, item["id"], None, embedding=_rand(5), **CLEAR)
    helmet = make_face(conn, item["id"], None, embedding=_rand(6), box=(0.6, 0.2, 0.2, 0.3), **HELMET)
    out = cluster.assign_new_faces(conn, [clear, helmet])
    assert out[clear] is not None and out[helmet] is None


def test_unclear_tags_do_not_move_a_learned_threshold(conn):
    item = make_item(conn)
    pid = make_person(conn, "Meera")
    who = _rand(7)
    for k in range(4):
        make_face(conn, item["id"], pid, embedding=_near(who, 10 + k), assigned_by="user",
                  box=(0.05 + 0.2 * k, 0.1, 0.1, 0.1), **CLEAR)
    only_clear = people_learning.person_thresholds(conn).get(pid)
    for k in range(4):  # a helmet in every one of these: unlike each other and unlike her face
        make_face(conn, item["id"], pid, embedding=_rand(20 + k), assigned_by="user",
                  box=(0.05 + 0.2 * k, 0.6, 0.1, 0.1), **HELMET)
    assert people_learning.person_thresholds(conn).get(pid) == only_clear


def _rows(conn, ids):
    marks = ",".join("?" * len(ids))
    return conn.execute(f"SELECT * FROM faces WHERE id IN ({marks}) ORDER BY id", ids).fetchall()


def _reject(conn, face_id, person_id):
    with tx(conn):
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (face_id, person_id, now_iso()))


def test_the_owners_choice_wins_when_old_tracks_become_one(conn):
    """The old tracker split one person into three tracks: the clustering put Priya on one, the owner put
    Priya on another and said "not Arjun" on the third. The new track gets the owner's choice, locked, and
    keeps the rejection."""
    item = make_item(conn, kind="video", ext=".mp4", mime="video/mp4")
    priya, arjun = make_person(conn, "Priya"), make_person(conn, "Arjun")
    who = _rand(30)
    auto = make_face(conn, item["id"], priya, embedding=_near(who, 31, 0.1), box=(0.1, 0.1, 0.1, 0.1))
    mine = make_face(conn, item["id"], priya, embedding=_near(who, 32, 0.5), assigned_by="user", box=(0.5, 0.1, 0.1, 0.1))
    other = make_face(conn, item["id"], None, embedding=_near(who, 33, 0.3), box=(0.7, 0.5, 0.1, 0.1))
    _reject(conn, other, arjun)
    old = _rows(conn, [auto, mine, other])
    new = make_face(conn, item["id"], None, embedding=who, box=(0.3, 0.6, 0.1, 0.1))
    assert cluster.carry_over(conn, old, [new]) == 1
    row = conn.execute("SELECT person_id, assigned_by FROM faces WHERE id = ?", (new,)).fetchone()
    assert (row["person_id"], row["assigned_by"]) == (priya, "user")
    assert cluster.rejected_ids(conn, "face", [new]) == {new}
    assert conn.execute("SELECT person_id FROM guess_rejections WHERE ref_id = ?", (new,)).fetchone()[0] == arjun


def test_a_carried_automatic_person_the_owner_rejected_is_dropped(conn):
    item = make_item(conn, kind="video", ext=".mp4", mime="video/mp4")
    dev = make_person(conn, "Dev")
    who = _rand(40)
    auto = make_face(conn, item["id"], dev, embedding=_near(who, 41, 0.1), box=(0.1, 0.1, 0.1, 0.1))
    said_no = make_face(conn, item["id"], None, embedding=_near(who, 42, 0.3), box=(0.6, 0.1, 0.1, 0.1))
    _reject(conn, said_no, dev)
    old = _rows(conn, [auto, said_no])
    new = make_face(conn, item["id"], None, embedding=who, box=(0.3, 0.6, 0.1, 0.1))
    assert cluster.carry_over(conn, old, [new]) == 0
    row = conn.execute("SELECT person_id, assigned_by FROM faces WHERE id = ?", (new,)).fetchone()
    assert (row["person_id"], row["assigned_by"]) == (None, "auto")  # assigned afresh, never to Dev
    assert conn.execute("SELECT person_id FROM guess_rejections WHERE ref_id = ?", (new,)).fetchone()[0] == dev


def test_plan_carry_pairs_one_to_one_and_lists_followers():
    who = _rand(50)
    old = [_near(who, 51, 0.1), _near(who, 52, 0.2), _rand(53)]
    new = [who]
    boxes = [(0.1, 0.1, 0.1, 0.1)] * 3
    pairs, follows = cluster.plan_carry(boxes, np.stack(old), [False, True, False], [(0.5, 0.5, 0.1, 0.1)], np.stack(new))
    assert pairs == [(1, 0)]  # the user-locked one first
    assert follows == [(0, 0)]  # the unrelated third face follows nobody
