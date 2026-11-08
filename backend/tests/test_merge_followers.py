"""A merge pulls look-alike unnamed people along and teaches the matching (sections 10 and 18).

Embeddings are built analytically: axis(0) is the person, mix(c, k) has cosine exactly c to it, and
two mixes along different axes have cosine c1 * c2 to each other, so every score below is known."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.analysis import cluster
from app.analysis.index import get_index
from app.db import tx
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person

DIM = 512


def axis(k: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[k] = 1.0
    return v


def mix(c: float, k: int) -> np.ndarray:
    return (c * axis(0) + np.sqrt(1 - c * c) * axis(k)).astype(np.float32)


def _face(conn, person, emb, by="auto", item=None):
    item = item or make_item(conn, write_files=False)
    return make_face(conn, item["id"], person, embedding=emb, assigned_by=by, write_crop=False), item


@pytest.fixture()
def world(conn):
    """Asha (two faces on the axis, one confirmed) and the unnamed people around her."""
    get_index().invalidate()
    asha = make_person(conn, "Asha")
    _, a_item = _face(conn, asha, axis(0), by="user")
    _face(conn, asha, axis(0))
    source = make_person(conn)      # the owner merges this one in: alike at 0.43
    _face(conn, source, mix(0.43, 1))
    near = make_person(conn)        # alike at 0.60: pulled along
    _face(conn, near, mix(0.60, 2))
    far = make_person(conn)         # 0.38: less alike than what the owner accepted
    _face(conn, far, mix(0.38, 3))
    named = make_person(conn, "Mira")   # named: never pulled, however alike
    _face(conn, named, mix(0.70, 4))
    dismissed = make_person(conn)   # the owner said these two are not the same person
    _face(conn, dismissed, mix(0.60, 5))
    with tx(conn):
        conn.execute("INSERT INTO merge_dismissals(person_a, person_b, created_at) VALUES (?, ?, ?)",
                     (min(asha, dismissed), max(asha, dismissed), now_iso()))
    together = make_person(conn)    # in the same photo as Asha: someone else
    _face(conn, together, mix(0.60, 6), item=a_item)
    touched = make_person(conn)     # a face the owner assigned by hand
    _face(conn, touched, mix(0.60, 7), by="user")
    rejected = make_person(conn)    # a face the owner took away from Asha
    rf, _ = _face(conn, rejected, mix(0.60, 8))
    with tx(conn):
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (rf, asha, now_iso()))
    return dict(asha=asha, source=source, near=near, far=far, named=named, dismissed=dismissed,
                together=together, touched=touched, rejected=rejected)


def _persons(conn):
    return {int(r[0]) for r in conn.execute("SELECT id FROM persons")}


def test_merge_pulls_lookalikes_along(conn, world):
    w = world
    accepted, followers = cluster.merge_and_follow(conn, w["asha"], [w["source"]])
    assert accepted == pytest.approx(0.43, abs=0.01)
    assert followers == [w["near"]]
    persons = _persons(conn)
    assert w["source"] not in persons and w["near"] not in persons
    for key in ("far", "named", "dismissed", "together", "touched", "rejected"):
        assert w[key] in persons, key
    assert conn.execute("SELECT COUNT(*) FROM faces WHERE person_id = ?", (w["asha"],)).fetchone()[0] == 4
    ev = conn.execute("SELECT label, features, person_id FROM learning_events WHERE kind = 'merge'").fetchone()
    assert ev["person_id"] == w["asha"] and ev["label"] == 1
    f = json.loads(ev["features"])
    assert f["sources"] == 1 and f["followed"] == 1 and f["accepted"] == pytest.approx(0.43, abs=0.01)


def test_merge_without_faces_pulls_nothing(conn):
    a, b = make_person(conn, "Asha"), make_person(conn)
    accepted, followers = cluster.merge_and_follow(conn, a, [b])
    assert accepted is None and followers == []
    assert b not in _persons(conn)


def test_merge_loosens_the_threshold_and_a_rejection_tightens_it(conn, world):
    from app.learning import people as learned

    w = world
    with tx(conn):
        conn.execute("DELETE FROM guess_rejections")
    rf = conn.execute("SELECT id FROM faces WHERE person_id = ?", (w["rejected"],)).fetchone()[0]
    # nothing protects the 0.60 faces now: `near` and `rejected` both come along
    _, followers = cluster.merge_and_follow(conn, w["asha"], [w["source"]])
    assert set(followers) == {w["near"], w["rejected"]}
    assert learned.person_thresholds(conn)[w["asha"]] == pytest.approx(0.41, abs=0.01)  # 0.43 accepted, minus 0.02
    # the owner takes the 0.60 face away again (POST faces/assign records this): the threshold
    # rises above it, whatever the merge loosened
    with tx(conn):
        conn.execute("UPDATE faces SET person_id = NULL WHERE id = ?", (rf,))
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (rf, w["asha"], now_iso()))
    assert learned.person_thresholds(conn)[w["asha"]] == pytest.approx(0.62, abs=0.01)


def test_a_rejection_makes_the_next_pull_more_careful(conn, world):
    from app.learning import people as learned

    w = world
    learned.train(conn)   # the rejected 0.60 face raises Asha's threshold to 0.62
    assert learned.person_threshold(w["asha"]) == pytest.approx(0.62, abs=0.01)
    accepted, followers = cluster.merge_and_follow(conn, w["asha"], [w["source"]])
    assert accepted == pytest.approx(0.43, abs=0.01)
    assert followers == []            # near (0.60) stays apart now
    assert w["near"] in _persons(conn)


def test_clustering_honours_a_loosened_threshold(conn, world):
    from app.learning import people as learned

    w = world
    with tx(conn):
        conn.execute("DELETE FROM guess_rejections")
    cluster.merge_and_follow(conn, w["asha"], [w["source"]])
    learned.train(conn)
    assert learned.person_threshold(w["asha"]) == pytest.approx(0.41, abs=0.01)
    get_index().invalidate()
    # Asha's reference is now axis(0) twice (one confirmed, weight 1.5), the 0.43 and the 0.60 face.
    # A face at 0.48 to the axis scores (0.72 + 0.48 + 0.288) / 3.5 = 0.425 as the weighted top-3
    # mean: below the 0.45 default, above the learned 0.41. One at 0.42 scores 0.372: below both.
    f, _ = _face(conn, None, mix(0.48, 9))
    assert cluster.assign_new_faces(conn, [f])[f] == w["asha"]
    g, _ = _face(conn, None, mix(0.42, 10))
    assert cluster.assign_new_faces(conn, [g])[g] != w["asha"]


def test_merge_route_reports_the_followers(client, conn, world):
    w = world
    r = client.post("/api/people/merge", json={"target_id": w["asha"], "source_ids": [w["source"]]})
    assert r.status_code == 200, r.text
    assert r.json()["id"] == w["asha"] and r.json()["followed"] == 1
    assert w["near"] not in _persons(conn)
