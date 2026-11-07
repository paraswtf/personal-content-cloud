"""Face clustering, merge, suggestions and pruning (app.analysis.cluster) with synthetic embeddings."""
from __future__ import annotations

import numpy as np

from app.analysis import cluster
from app.db import tx
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(base: np.ndarray, seed: int, noise: float = 0.15) -> np.ndarray:
    v = base + noise * _vec(seed, base.shape[0])
    return (v / np.linalg.norm(v)).astype(np.float32)


def _person_of(conn, face_id: int):
    return conn.execute("SELECT person_id, assigned_by FROM faces WHERE id = ?", (face_id,)).fetchone()


def test_new_faces_cluster_into_people(conn):
    alice, bob = _vec(1), _vec(2)
    i1, i2, i3 = (make_item(conn, write_files=False, data=None) for _ in range(3))
    f1 = make_face(conn, i1["id"], embedding=alice)
    res = cluster.assign_new_faces(conn, [f1])
    p_alice = res[f1]
    assert p_alice is not None
    f2 = make_face(conn, i2["id"], embedding=_near(alice, 10))
    f3 = make_face(conn, i3["id"], embedding=bob)
    res = cluster.assign_new_faces(conn, [f2, f3])
    assert res[f2] == p_alice
    assert res[f3] not in (None, p_alice)
    assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 2


def test_one_person_per_photo_and_small_faces(conn):
    alice = _vec(3)
    i1, i2 = make_item(conn, write_files=False, data=None), make_item(conn, write_files=False, data=None)
    f1 = make_face(conn, i1["id"], embedding=alice)
    p = cluster.assign_new_faces(conn, [f1])[f1]
    a = make_face(conn, i2["id"], embedding=_near(alice, 11, 0.1))
    b = make_face(conn, i2["id"], embedding=_near(alice, 12, 0.3))
    small = make_face(conn, i2["id"], embedding=_vec(99), size_px=30, det_score=0.6)
    res = cluster.assign_new_faces(conn, [a, b, small])
    assert res[a] == p
    assert res[b] not in (None, p)  # the second face cannot be the same person in one photo
    assert res[small] is None  # not good enough to start a person


def test_user_locked_faces_are_untouched(conn):
    alice = _vec(4)
    i1, i2 = make_item(conn, write_files=False, data=None), make_item(conn, write_files=False, data=None)
    f1 = make_face(conn, i1["id"], embedding=alice)
    cluster.assign_new_faces(conn, [f1])
    locked = make_face(conn, i2["id"], embedding=_near(alice, 13), assigned_by="user")
    assert cluster.assign_new_faces(conn, [locked])[locked] is None
    assert _person_of(conn, locked)["assigned_by"] == "user"
    assert cluster.recluster_unassigned(conn) == 0


def test_recluster_never_creates_people(conn):
    alice = _vec(5)
    i1, i2, i3 = (make_item(conn, write_files=False, data=None) for _ in range(3))
    f1 = make_face(conn, i1["id"], embedding=alice)
    p = cluster.assign_new_faces(conn, [f1])[f1]
    weak = make_face(conn, i2["id"], embedding=_near(alice, 14, 0.2), size_px=30, det_score=0.6)
    stranger = make_face(conn, i3["id"], embedding=_vec(77))
    assert cluster.recluster_unassigned(conn) == 1
    assert _person_of(conn, weak)["person_id"] == p
    assert _person_of(conn, stranger)["person_id"] is None
    assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 1


def test_merge_moves_everything(conn):
    target = make_person(conn, None)
    source = make_person(conn, "Asha")
    other = make_person(conn, "Ravi")
    i1, i2, i3 = (make_item(conn, write_files=False, data=None) for _ in range(3))
    fs = make_face(conn, i1["id"], source, embedding=_vec(6))
    ft = make_face(conn, i2["id"], target, embedding=_vec(7))
    open_face = make_face(conn, i3["id"], None, embedding=_vec(8))
    now = now_iso()
    with tx(conn):
        conn.execute("UPDATE persons SET cover_face_id = ? WHERE id = ?", (fs, source))
        body = conn.execute(
            "INSERT INTO bodies(item_id, person_id, assigned_by, x, y, w, h, det_score, created_at) "
            "VALUES (?, ?, 'user', 0.1, 0.1, 0.3, 0.6, 0.9, ?)", (i3["id"], source, now)).lastrowid
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, created_at) VALUES (?, ?, 'user', ?)",
                     (i2["id"], source, now))
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (open_face, source, now))
        conn.execute("INSERT INTO merge_dismissals(person_a, person_b, created_at) VALUES (?, ?, ?)",
                     (min(source, other), max(source, other), now))
        conn.execute("UPDATE faces SET assigned_by = 'guess', guess_person_id = ?, guess_score = 0.8 WHERE id = ?",
                     (source, open_face))
    cluster.merge_persons(conn, target, [source])
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE id = ?", (source,)).fetchone()[0] == 0
    t = conn.execute("SELECT name, cover_face_id FROM persons WHERE id = ?", (target,)).fetchone()
    assert t["name"] == "Asha" and t["cover_face_id"] == fs
    assert _person_of(conn, fs)["person_id"] == target and _person_of(conn, ft)["person_id"] == target
    assert conn.execute("SELECT person_id FROM bodies WHERE id = ?", (body,)).fetchone()[0] == target
    assert conn.execute("SELECT COUNT(*) FROM item_tags WHERE person_id = ?", (target,)).fetchone()[0] == 1
    assert conn.execute("SELECT person_id FROM guess_rejections").fetchone()[0] == target
    assert conn.execute("SELECT person_a, person_b FROM merge_dismissals").fetchone()[:] == (
        min(target, other), max(target, other))
    g = conn.execute("SELECT guess_person_id, assigned_by FROM faces WHERE id = ?", (open_face,)).fetchone()
    assert g["guess_person_id"] == target and g["assigned_by"] == "guess"


def test_merge_rejects_bad_ids(conn):
    p = make_person(conn, "A")
    for target, sources in ((p, [p]), (p, [9999]), (p, [])):
        try:
            cluster.merge_persons(conn, target, sources)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_merge_suggestions_skip_people_in_one_photo(conn):
    base = _vec(20)
    a, b = make_person(conn), make_person(conn)
    i1, i2 = make_item(conn, write_files=False, data=None), make_item(conn, write_files=False, data=None)
    make_face(conn, i1["id"], a, embedding=_near(base, 21, 0.5))
    make_face(conn, i2["id"], b, embedding=_near(base, 22, 0.5))
    pairs = cluster.merge_suggestions(conn, 10)
    assert [(x, y) for x, y, _ in pairs] == [(min(a, b), max(a, b))]
    assert pairs[0][2] >= 0.35
    # Together in one photo: different people.
    i3 = make_item(conn, write_files=False, data=None)
    make_face(conn, i3["id"], a, embedding=_near(base, 23, 0.5))
    make_face(conn, i3["id"], b, embedding=_near(base, 24, 0.5), box=(0.6, 0.2, 0.2, 0.3))
    assert cluster.merge_suggestions(conn, 10) == []


def test_merge_suggestions_respect_dismissals(conn):
    base = _vec(30)
    a, b = make_person(conn), make_person(conn)
    i1, i2 = make_item(conn, write_files=False, data=None), make_item(conn, write_files=False, data=None)
    make_face(conn, i1["id"], a, embedding=_near(base, 31, 0.5))
    make_face(conn, i2["id"], b, embedding=_near(base, 32, 0.5))
    with tx(conn):
        conn.execute("INSERT INTO merge_dismissals(person_a, person_b, created_at) VALUES (?, ?, ?)",
                     (min(a, b), max(a, b), now_iso()))
    assert cluster.merge_suggestions(conn, 10) == []


def test_prune_and_person_helpers(conn):
    empty = make_person(conn, None)
    named = make_person(conn, "Meera")
    with tx(conn):
        assert cluster.prune_empty_persons(conn, {empty, named}) == 1
    assert conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 1
    with tx(conn):
        assert cluster.find_person_by_name(conn, "  meera ") == named
        assert cluster.find_person_by_name(conn, "meera", exclude_id=named) is None
        assert cluster.find_or_create_person(conn, "Meera") == named
        new = cluster.find_or_create_person(conn, "Kiran")
        assert new != named
        assert cluster.find_or_create_person(conn, "   ") is None


def test_carry_over_keeps_names(conn):
    p = make_person(conn, "Asha")
    item = make_item(conn, write_files=False, data=None)
    emb = _vec(40)
    old = make_face(conn, item["id"], p, embedding=emb, assigned_by="user")
    old_rows = conn.execute("SELECT * FROM faces WHERE id = ?", (old,)).fetchall()
    new = make_face(conn, item["id"], None, embedding=_near(emb, 41, 0.1))
    assert cluster.carry_over(conn, old_rows, [new]) == 1
    row = _person_of(conn, new)
    assert row["person_id"] == p and row["assigned_by"] == "user"
