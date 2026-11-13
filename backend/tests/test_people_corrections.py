"""Corrections that reach beyond one face (section 10): taking faces away from a person pulls the
look-alikes along and keeps them away for good; the person reference is every face, the owner's
assignments weighing more."""
from __future__ import annotations

import numpy as np

from app.analysis import cluster, index as face_index
from app.analysis.index import get_index
from app.db import tx
from tests.helpers import make_face, make_item, make_person
from tests.test_learning_core import near, unit


def _face(conn, item, person, emb, by="auto"):
    return make_face(conn, item["id"], person, embedding=emb, assigned_by=by, write_crop=False)


def _person_of(conn, fid):
    r = conn.execute("SELECT person_id, assigned_by FROM faces WHERE id = ?", (fid,)).fetchone()
    return (r["person_id"], r["assigned_by"])


def _rejected(conn, pid):
    return {int(r[0]) for r in conn.execute("SELECT ref_id FROM guess_rejections WHERE kind = 'face' AND person_id = ?", (pid,))}


def _mixed_person(conn):
    """Asha (A faces, one confirmed) with three wrongly grouped B faces; returns (pid, a_faces, b_faces)."""
    get_index().invalidate()
    A, B = unit(1, 512), unit(2, 512)
    asha = make_person(conn, "Asha")
    items = [make_item(conn, filename=f"f{k}.jpg", write_files=False) for k in range(8)]
    a_faces = [_face(conn, items[0], asha, near(A, 11, 0.3), by="user"), _face(conn, items[1], asha, near(A, 12, 0.3)),
               _face(conn, items[2], asha, near(A, 13, 0.3))]
    b_faces = [_face(conn, items[3], asha, near(B, 21, 0.3)), _face(conn, items[4], asha, near(B, 22, 0.3)),
               _face(conn, items[5], asha, near(B, 23, 0.3))]
    return asha, a_faces, b_faces, items, A, B


def test_followers_are_the_lookalikes_of_the_removed_face_only(conn):
    asha, a_faces, b_faces, items, A, B = _mixed_person(conn)
    assert sorted(cluster.followers(conn, [b_faces[0]], asha)) == sorted(b_faces[1:])
    assert cluster.followers(conn, [a_faces[1]], asha) == [a_faces[2]]   # the confirmed A face never follows
    # a B face the owner assigned herself stays put
    with tx(conn):
        conn.execute("UPDATE faces SET assigned_by = 'user' WHERE id = ?", (b_faces[2],))
    assert cluster.followers(conn, [b_faces[0]], asha) == [b_faces[1]]


def test_not_this_person_moves_the_lookalikes_and_keeps_them_away(client, conn):
    asha, a_faces, b_faces, items, A, B = _mixed_person(conn)
    r = client.post("/api/faces/assign", json={"face_ids": [b_faces[0]], "person_id": None})
    assert r.status_code == 200, r.text
    assert r.json()["updated"] == 1 and r.json()["followed"] == 2
    assert _person_of(conn, b_faces[0]) == (None, "user")
    # the two look-alikes left Asha together and were regrouped into a new unnamed person
    others = {_person_of(conn, f)[0] for f in b_faces[1:]}
    assert None not in others and asha not in others and len(others) == 1
    assert all(_person_of(conn, f)[0] == asha for f in a_faces)
    assert _rejected(conn, asha) == set(b_faces)
    # a new B face does not go back to Asha: it joins the look-alikes' new group
    new_b = _face(conn, items[6], None, near(B, 24, 0.3))
    cluster.assign_new_faces(conn, [new_b])
    assert _person_of(conn, new_b)[0] == others.pop()
    # a new A face still joins Asha
    new_a = _face(conn, items[7], None, near(A, 14, 0.3))
    cluster.assign_new_faces(conn, [new_a])
    assert _person_of(conn, new_a)[0] == asha
    assert client.get(f"/api/people/{asha}").json()["confirmed_faces"] == 1


def test_moving_a_face_to_someone_else_takes_the_lookalikes_along(client, conn):
    asha, a_faces, b_faces, items, A, B = _mixed_person(conn)
    bina = make_person(conn, "Bina")
    r = client.post("/api/faces/assign", json={"face_ids": [b_faces[0]], "person_id": bina})
    assert r.json()["followed"] == 2 and r.json()["person"]["face_count"] == 3
    assert _person_of(conn, b_faces[0]) == (bina, "user")
    assert all(_person_of(conn, f) == (bina, "auto") for f in b_faces[1:])
    assert _rejected(conn, asha) == set(b_faces)
    # moving within the same person is no correction
    r = client.post("/api/faces/assign", json={"face_ids": [a_faces[1]], "person_id": asha})
    assert r.json()["followed"] == 0 and _rejected(conn, asha) == set(b_faces)


def test_reference_weighs_confirmed_faces_more():
    person = np.array([7, 7, 7, 9], dtype=np.int64)
    base = np.ones(4, dtype=bool)
    sims = np.array([0.9, 0.5, 0.5, 0.3], dtype=np.float32)
    plain = dict(face_index._topk_means(person, base, sims, 3))
    weighted = dict(face_index._topk_means(person, base, sims, 3, np.array([True, False, False, False])))
    assert abs(plain[7] - (0.9 + 0.5 + 0.5) / 3) < 1e-6
    assert abs(weighted[7] - (1.5 * 0.9 + 0.5 + 0.5) / 3.5) < 1e-6
    assert weighted[7] > plain[7] and plain[9] == weighted[9]
