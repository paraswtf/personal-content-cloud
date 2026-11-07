"""Person guessing (app.analysis.guess) with hand-made scene, torso and face embeddings."""
from __future__ import annotations

import json

import numpy as np

from app.analysis import guess
from app.db import tx
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person


def _vec(seed: int, dim: int = 768) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(base: np.ndarray, seed: int, noise: float = 0.1) -> np.ndarray:
    v = base + noise * _vec(seed, base.shape[0])
    return (v / np.linalg.norm(v)).astype(np.float32)


def _scene(conn, item_id: int, vec: np.ndarray) -> None:
    with tx(conn):
        conn.execute(
            "INSERT INTO item_embeddings(item_id, version, model, frame_index, frame_time_s, embedding, created_at) "
            "VALUES (?, 1, 'test', 0, NULL, ?, ?)", (item_id, vec.astype(np.float32).tobytes(), now_iso()))


def _body(conn, item_id: int, torso: np.ndarray | None, person_id: int | None = None, assigned_by: str = "auto",
          face_id: int | None = None) -> int:
    with tx(conn):
        return int(conn.execute(
            "INSERT INTO bodies(item_id, face_id, person_id, assigned_by, x, y, w, h, det_score, torso_embedding, "
            "created_at) VALUES (?, ?, ?, ?, 0.1, 0.1, 0.3, 0.7, 0.9, ?, ?)",
            (item_id, face_id, person_id, assigned_by, None if torso is None else torso.tobytes(), now_iso()),
        ).lastrowid)


def _world(conn, *, t_b="2026-05-01T11:00:00.000Z", scene_b=None, torso_b=None, gps=None):
    """Item A (10:00 UTC) shows person P (user-confirmed face with torso T); item B holds a faceless body."""
    scene, torso = _vec(1), _vec(2)
    p = make_person(conn, "Asha")
    a = make_item(conn, taken_at="2026-05-01T10:00:00.000Z", write_files=False, data=None,
                  **({"gps_lat": gps[0][0], "gps_lon": gps[0][1]} if gps else {}))
    b = make_item(conn, taken_at=t_b, write_files=False, data=None,
                  **({"gps_lat": gps[1][0], "gps_lon": gps[1][1]} if gps else {}))
    face_a = make_face(conn, a["id"], p, embedding=_vec(3, 512), assigned_by="user", torso=torso)
    _scene(conn, a["id"], scene)
    _scene(conn, b["id"], scene if scene_b is None else scene_b)
    body_b = _body(conn, b["id"], _near(torso, 4) if torso_b is None else torso_b)
    return p, a, b, face_a, body_b, torso


def _body_row(conn, body_id: int):
    return conn.execute("SELECT * FROM bodies WHERE id = ?", (body_id,)).fetchone()


def test_same_day_same_scene_same_clothes(conn):
    p, a, b, face_a, body_b, _ = _world(conn)
    assert guess.guess_item(conn, b["id"]) == 1
    row = _body_row(conn, body_b)
    assert row["assigned_by"] == "guess" and row["guess_person_id"] == p and row["person_id"] is None
    ev = json.loads(row["guess_evidence"])
    assert ev["ref_face_id"] == face_a and ev["ref_item_uid"] == a["uid"] and ev["same_day"] is True
    assert ev["clothes"] >= 0.85 and ev["scene"] >= 0.99 and ev["face"] is None
    expected = 0.5 * ev["clothes"] + 0.3 * ev["scene"] + 0.2 * 0.5
    assert abs(row["guess_score"] - expected) < 1e-3
    # Idempotent.
    assert guess.guess_item(conn, b["id"]) == 0


def test_different_clothes_no_guess(conn):
    _, _, b, _, body_b, _ = _world(conn, torso_b=_vec(50))
    assert guess.guess_item(conn, b["id"]) == 0
    assert _body_row(conn, body_b)["guess_person_id"] is None


def test_different_scene_needs_gps(conn):
    _, _, b, _, body_b, _ = _world(conn, scene_b=_vec(60))
    assert guess.guess_item(conn, b["id"]) == 0


def test_gps_counts_as_same_place(conn):
    p, _, b, _, body_b, _ = _world(conn, scene_b=_vec(61), gps=((18.5204, 73.8567), (18.5210, 73.8570)))
    assert guess.guess_item(conn, b["id"]) == 1
    row = _body_row(conn, body_b)
    assert row["guess_person_id"] == p
    assert json.loads(row["guess_evidence"])["gps_m"] < 250


def test_outside_window(conn):
    _, _, b, _, body_b, _ = _world(conn, t_b="2026-05-03T10:00:00.000Z")
    assert guess.guess_item(conn, b["id"]) == 0


def test_same_local_day_beyond_window(conn, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "guess_window_h", 1.0)
    # 10:00 and 15:00 UTC are both 2026-05-01 in Asia/Kolkata.
    p, _, b, _, body_b, _ = _world(conn, t_b="2026-05-01T15:00:00.000Z")
    assert guess.guess_item(conn, b["id"]) == 1
    assert _body_row(conn, body_b)["guess_person_id"] == p


def test_rejected_and_present_people_are_skipped(conn):
    p, _, b, _, body_b, _ = _world(conn)
    assert guess.guess_item(conn, b["id"]) == 1
    assert guess.reject(conn, [], [body_b]) == 1
    row = _body_row(conn, body_b)
    assert row["guess_person_id"] is None and row["assigned_by"] == "auto"
    assert conn.execute("SELECT person_id FROM guess_rejections WHERE kind = 'body' AND ref_id = ?",
                        (body_b,)).fetchone()[0] == p
    assert guess.guess_item(conn, b["id"]) == 0
    # A person already in the item is not proposed for another box there.
    with tx(conn):
        conn.execute("DELETE FROM guess_rejections")
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, created_at) VALUES (?, ?, 'user', ?)",
                     (b["id"], p, now_iso()))
    assert guess.guess_item(conn, b["id"]) == 0


def test_tie_needs_margin(conn):
    p, a, b, _, body_b, torso = _world(conn)
    q = make_person(conn, "Twin")
    c = make_item(conn, taken_at="2026-05-01T10:30:00.000Z", write_files=False, data=None)
    make_face(conn, c["id"], q, embedding=_vec(5, 512), assigned_by="user", torso=torso)
    _scene(conn, c["id"], _vec(1))
    assert guess.guess_item(conn, b["id"]) == 0
    assert _body_row(conn, body_b)["guess_person_id"] is None


def test_face_contradiction_and_weak_face_match(conn):
    p, a, b, face_a, _, torso = _world(conn)
    q = make_person(conn, "Other")
    c = make_item(conn, taken_at="2026-05-10T10:00:00.000Z", write_files=False, data=None)
    q_face = _vec(70, 512)
    make_face(conn, c["id"], q, embedding=q_face, assigned_by="user")
    # A face in B that looks like Q (cos ~0.6): never guessed as P.
    looks_like_q = _near(q_face, 71, 1.2)
    assert float(np.dot(looks_like_q, q_face)) >= 0.4
    f_bad = make_face(conn, b["id"], None, embedding=looks_like_q, torso=_near(torso, 72), box=(0.6, 0.1, 0.1, 0.15))
    # A face in B that weakly matches P's face: guessed as P with a raised face term.
    p_face = np.frombuffer(conn.execute("SELECT embedding FROM faces WHERE id = ?", (face_a,)).fetchone()[0],
                           dtype=np.float32)
    weak = _near(p_face, 73, 2.5)
    sim = float(np.dot(weak, p_face))
    assert 0.2 <= sim < 0.4, sim
    f_good = make_face(conn, b["id"], None, embedding=weak, torso=_near(torso, 74), box=(0.2, 0.1, 0.1, 0.15))
    guess.guess_item(conn, b["id"])
    bad = conn.execute("SELECT * FROM faces WHERE id = ?", (f_bad,)).fetchone()
    good = conn.execute("SELECT * FROM faces WHERE id = ?", (f_good,)).fetchone()
    assert bad["guess_person_id"] is None
    assert good["guess_person_id"] == p and good["assigned_by"] == "guess"
    ev = json.loads(good["guess_evidence"])
    assert ev["face"] is not None and ev["face"] >= 0.2
    assert good["guess_score"] > 0.5 * ev["clothes"] + 0.3 * ev["scene"] + 0.2 * 0.5


def test_confirm_sets_person_tag_and_job(conn):
    p, _, b, _, body_b, _ = _world(conn)
    guess.guess_item(conn, b["id"])
    assert guess.confirm(conn, [], [body_b]) == 1
    row = _body_row(conn, body_b)
    assert row["person_id"] == p and row["assigned_by"] == "user" and row["guess_person_id"] is None
    assert conn.execute("SELECT source FROM item_tags WHERE item_id = ? AND person_id = ?",
                        (b["id"], p)).fetchone()[0] == "user"
    job = conn.execute("SELECT type, payload FROM jobs WHERE type = 'guess'").fetchone()
    assert json.loads(job["payload"]) == {"person_id": p, "item_id": b["id"]}
    assert job["payload"].index("person_id") < job["payload"].index("item_id")
    # Confirming again changes nothing; the job is not queued twice.
    assert guess.confirm(conn, [], [body_b]) == 0
    guess.enqueue_guess(conn, p, b["id"])
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE type = 'guess'").fetchone()[0] == 1


def test_propagate_from_a_tagged_item(conn):
    p, a, b, _, body_b, _ = _world(conn)
    assert guess.propagate(conn, p, a["id"]) == 1
    assert _body_row(conn, body_b)["guess_person_id"] == p
    assert guess.propagate(conn, p, None) == 0  # already up to date
    assert guess.propagate(conn, 99999, None) == 0


def test_stale_guess_is_cleared(conn):
    p, a, b, face_a, body_b, _ = _world(conn)
    guess.guess_item(conn, b["id"])
    with tx(conn):
        conn.execute("UPDATE faces SET torso_embedding = NULL WHERE id = ?", (face_a,))
    assert guess.guess_item(conn, b["id"]) == 1
    row = _body_row(conn, body_b)
    assert row["guess_person_id"] is None and row["assigned_by"] == "auto"


def test_scoring_helpers():
    assert guess.face_term(None) == 0.5 and guess.face_term(0.1) == 0.5
    assert guess.face_term(0.45) == 1.0 and 0.5 < guess.face_term(0.3) < 1.0
    assert abs(guess.guess_score(0.9, 0.85, None) - (0.45 + 0.255 + 0.1)) < 1e-9
    assert 60 < guess.haversine_m(18.5204, 73.8567, 18.5210, 73.8570) < 90
