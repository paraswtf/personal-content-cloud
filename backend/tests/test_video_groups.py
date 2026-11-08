"""Whole-video editing (section 10): the tracks of a video are grouped on read, by person when assigned and
by look otherwise, so the viewer can edit a person's every track at once."""
from __future__ import annotations

import numpy as np

from app.api.common import VIDEO_GROUP_SIM, video_groups
from app.db import tx
from tests.helpers import make_face, make_item, make_person


def unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def near(v: np.ndarray, seed: int, noise: float = 0.6) -> np.ndarray:
    w = v + noise * unit(seed, v.shape[0])
    return (w / np.linalg.norm(w)).astype(np.float32)


def _face(conn, item_id, person, emb, t):
    return make_face(conn, item_id, person, embedding=emb, frame_time_s=t, write_crop=False)


def test_tracks_group_by_person_then_by_look(conn):
    video = make_item(conn, kind="video", ext=".mp4", write_files=False, data=None)
    asha = make_person(conn, "Asha")
    a = unit(1)
    f1 = _face(conn, video["id"], asha, a, 1.0)
    f2 = _face(conn, video["id"], asha, near(a, 2), 9.0)           # the same person, a second track
    stranger = unit(3)
    f3 = _face(conn, video["id"], None, stranger, 3.0)             # unknown, first look
    f4 = _face(conn, video["id"], None, near(stranger, 4), 12.0)   # the same unknown later (cosine about 0.86)
    f5 = _face(conn, video["id"], None, unit(5), 15.0)             # another unknown
    rows = conn.execute("SELECT * FROM faces WHERE item_id = ? ORDER BY frame_time_s", (video["id"],)).fetchall()
    g = video_groups(rows, [], [])
    assert g[("face", f1)] == g[("face", f2)]
    assert g[("face", f3)] == g[("face", f4)]
    assert len({g[("face", f1)], g[("face", f3)], g[("face", f5)]}) == 3
    # a body of Asha shares her group across kinds
    with tx(conn):
        bid = conn.execute(
            "INSERT INTO bodies(item_id, person_id, assigned_by, x, y, w, h, det_score, frame_time_s, torso_embedding, created_at) "
            "VALUES (?, ?, 'auto', 0.1, 0.1, 0.3, 0.7, 0.9, 20.0, ?, '2026-01-01T00:00:00.000Z')",
            (video["id"], asha, unit(9, 768).tobytes())).lastrowid
    bodies = conn.execute("SELECT * FROM bodies WHERE item_id = ?", (video["id"],)).fetchall()
    g2 = video_groups(rows, bodies, [])
    assert g2[("body", bid)] == g2[("face", f1)]


def test_look_alikes_need_the_threshold_and_kinds_never_mix(conn):
    video = make_item(conn, kind="video", ext=".mp4", write_files=False, data=None)
    a = unit(11)
    f1 = _face(conn, video["id"], None, a, 1.0)
    f2 = _face(conn, video["id"], None, near(a, 12, noise=2.5), 5.0)   # cosine about 0.37: apart
    rows = conn.execute("SELECT * FROM faces WHERE item_id = ?", (video["id"],)).fetchall()
    g = video_groups(rows, [], [])
    assert g[("face", f1)] != g[("face", f2)]
    assert VIDEO_GROUP_SIM == 0.5


def test_item_detail_carries_groups_for_videos_only(client, conn):
    from app.api.library import item_detail

    video = make_item(conn, kind="video", ext=".mp4", write_files=False, data=None)
    photo = make_item(conn, write_files=False, data=None)
    a = unit(21)
    _face(conn, video["id"], None, a, 1.0)
    _face(conn, video["id"], None, near(a, 22), 4.0)
    make_face(conn, photo["id"], None, embedding=unit(23), write_crop=False)
    vrow = conn.execute("SELECT * FROM items WHERE id = ?", (video["id"],)).fetchone()
    prow = conn.execute("SELECT * FROM items WHERE id = ?", (photo["id"],)).fetchone()
    v = item_detail(conn, vrow)
    assert len({f.group for f in v.faces}) == 1 and v.faces[0].group is not None
    p = item_detail(conn, prow)
    assert p.faces[0].group is None
    r = client.get(f"/api/items/{video['uid']}")
    assert r.status_code == 200 and all(f["group"] == r.json()["faces"][0]["group"] for f in r.json()["faces"])
