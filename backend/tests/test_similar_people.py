"""The name picker's order: POST /api/people/similar (app.analysis.similar, section 10 "Naming by likeness")."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import cluster
from app.config import settings
from app.db import tx
from app.util import name_key, now_iso
from tests.helpers import make_face, make_item, make_person


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(base: np.ndarray, seed: int, noise: float = 0.8) -> np.ndarray:
    v = base + noise * _vec(seed, base.shape[0])
    return (v / np.linalg.norm(v)).astype(np.float32)


def _item(conn, taken_at: str | None = "2026-05-01T10:00:00.000Z", **kw):
    return make_item(conn, taken_at=taken_at, write_files=False, data=None, **kw)


def _person_with_faces(conn, name: str | None, base: np.ndarray, n: int, seed: int, **item_kw) -> int:
    pid = make_person(conn, name)
    for k in range(n):
        it = _item(conn, taken_at=f"2025-0{1 + k % 9}-0{1 + k % 9}T08:00:00.000Z", **item_kw)
        make_face(conn, it["id"], pid, embedding=_near(base, seed * 100 + k), assigned_by="user", write_crop=False)
    return pid


def _ask(client, **body):
    r = client.post("/api/people/similar", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _ids(out) -> list[int]:
    return [o["person"]["id"] for o in out["items"]]


@pytest.fixture()
def people(conn):
    """Three looks: two named people and an unnamed group, five faces each."""
    a, b, c = _vec(1), _vec(2), _vec(3)
    pa = _person_with_faces(conn, "Alpha", a, 5, 10)
    pb = _person_with_faces(conn, "Bravo", b, 5, 20)
    pc = _person_with_faces(conn, None, c, 5, 30)
    return {"a": (pa, a), "b": (pb, b), "c": (pc, c)}


def test_a_face_lists_the_nearest_named_person_first(client, conn, people):
    pa, a = people["a"]
    pb, _ = people["b"]
    it = _item(conn)
    f = make_face(conn, it["id"], None, embedding=_near(a, 999), write_crop=False)
    out = _ask(client, face_ids=[f])
    assert _ids(out)[0] == pa
    best = out["items"][0]
    assert best["likely"] is True and best["basis"] == "face" and best["reason"]
    assert 0.4 < best["score"] <= 1.0
    # the other named person is not likely, and scores below
    others = {o["person"]["id"]: o for o in out["items"]}
    if pb in others:
        assert others[pb]["likely"] is False and others[pb]["score"] < best["score"]
    assert str(pa) in out["scores"]


def test_unnamed_people_are_never_listed(client, conn, people):
    pc, c = people["c"]
    it = _item(conn)
    f = make_face(conn, it["id"], None, embedding=_near(c, 998), write_crop=False)
    out = _ask(client, face_ids=[f])
    assert pc not in _ids(out)
    assert str(pc) not in out["scores"]


def test_the_open_guess_comes_first_and_answered_people_are_left_out(client, conn, people):
    pa, a = people["a"]
    pb, _ = people["b"]
    it = _item(conn)
    f = make_face(conn, it["id"], None, embedding=_near(a, 997), assigned_by="guess", write_crop=False,
                  guess_person_id=pb, guess_score=0.7)
    out = _ask(client, face_ids=[f])
    assert _ids(out)[:2] == [pb, pa]
    assert out["items"][0]["first"] is True and out["items"][0]["basis"] == "guess"
    # "Not Alpha" for this face: Alpha is left out for it
    with tx(conn):
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (f, pa, now_iso()))
    assert pa not in _ids(_ask(client, face_ids=[f]))


def test_people_already_in_the_photo_and_the_face_own_person_are_left_out(client, conn, people):
    pa, a = people["a"]
    pb, b = people["b"]
    it = _item(conn)
    make_face(conn, it["id"], pa, embedding=_near(a, 996), assigned_by="user", write_crop=False)
    f = make_face(conn, it["id"], None, embedding=_near(a, 995), write_crop=False)
    assert pa not in _ids(_ask(client, face_ids=[f]))  # one person per photo
    g = make_face(conn, _item(conn)["id"], pb, embedding=_near(b, 994), write_crop=False)
    assert pb not in _ids(_ask(client, face_ids=[g]))  # what it already is changes nothing


def test_a_person_box_ranks_by_the_clothes_of_the_hour(client, conn):
    pa = make_person(conn, "Alpha")
    torso = _vec(5, 768)
    gps = {"gps_lat": 12.9716, "gps_lon": 77.5946}
    ref = _item(conn, taken_at="2026-05-01T10:00:00.000Z", **gps)
    make_face(conn, ref["id"], pa, embedding=_vec(6), assigned_by="user", torso=torso, write_crop=False)
    it = _item(conn, taken_at="2026-05-01T10:20:00.000Z", **gps)
    with tx(conn):
        body = int(conn.execute(
            "INSERT INTO bodies(item_id, assigned_by, x, y, w, h, det_score, torso_embedding, created_at) "
            "VALUES (?, 'auto', 0.1, 0.1, 0.3, 0.7, 0.9, ?, ?)",
            (it["id"], _near(torso, 7, 0.1).tobytes(), now_iso())).lastrowid)
    out = _ask(client, body_ids=[body])
    assert _ids(out)[0] == pa
    assert out["items"][0]["basis"] == "clothes" and out["items"][0]["likely"] is True


def test_a_box_with_no_vector_still_lists_who_was_there_that_hour(client, conn):
    pa = make_person(conn, "Alpha")
    near = _item(conn, taken_at="2026-05-01T10:30:00.000Z")
    make_face(conn, near["id"], pa, embedding=_vec(8), assigned_by="user", write_crop=False)
    far = _item(conn, taken_at="2026-05-03T10:30:00.000Z")
    pz = make_person(conn, "Zulu")
    make_face(conn, far["id"], pz, embedding=_vec(9), assigned_by="user", write_crop=False)
    it = _item(conn, taken_at="2026-05-01T10:00:00.000Z")
    with tx(conn):
        body = int(conn.execute(
            "INSERT INTO bodies(item_id, assigned_by, x, y, w, h, det_score, created_at) "
            "VALUES (?, 'auto', 0.1, 0.1, 0.3, 0.7, 0.9, ?)", (it["id"], now_iso())).lastrowid)
    out = _ask(client, body_ids=[body])
    assert _ids(out) == [pa]
    assert out["items"][0]["basis"] == "hour" and out["items"][0]["likely"] is False
    assert out["items"][0]["score"] <= 0.12
    # a box id that does not exist answers an empty list, not an error
    assert _ask(client, face_ids=[987654], body_ids=[876543]) == {"items": [], "scores": {}}


def test_several_faces_and_a_person_on_items(client, conn, people):
    pa, a = people["a"]
    pb, b = people["b"]
    # someone wrongly put on Bravo in three photos, who looks like Alpha
    uids = []
    faces = []
    for k in range(3):
        it = _item(conn)
        uids.append(it["uid"])
        faces.append(make_face(conn, it["id"], pb, embedding=_near(a, 900 + k), write_crop=False))
    out = _ask(client, uids=uids, person_id=pb)
    assert _ids(out)[0] == pa and pb not in _ids(out)
    assert out["items"][0]["likely"] is True
    out2 = _ask(client, face_ids=faces)
    assert _ids(out2)[0] == pa and pb not in _ids(out2)
    r = client.post("/api/people/similar", json={"uids": uids})
    assert r.status_code == 400


def test_boxes_that_name_nobody_do_not_water_down_the_others(client, conn, people):
    from app.analysis import similar

    pa, a = people["a"]
    pb, _ = people["b"]
    uids = []
    for k in range(similar.MAX_FACES * 2 + 6):  # more faces than are scored
        it = _item(conn, taken_at=f"2024-0{1 + k % 9}-1{k % 9}T08:00:00.000Z")
        uids.append(it["uid"])
        make_face(conn, it["id"], pb, embedding=_near(a, 700 + k), write_crop=False)
    for k in range(5):  # person boxes of theirs with nothing to go on
        it = _item(conn, taken_at=None)
        uids.append(it["uid"])
        with tx(conn):
            conn.execute("INSERT INTO bodies(item_id, person_id, assigned_by, x, y, w, h, det_score, created_at) "
                         "VALUES (?, ?, 'user', 0.1, 0.1, 0.3, 0.7, 0.9, ?)", (it["id"], pb, now_iso()))
    out = _ask(client, uids=uids, person_id=pb)
    best = out["items"][0]
    assert best["person"]["id"] == pa and best["likely"] is True and best["score"] > 0.4


def test_the_whole_person_for_merge_into(client, conn, people):
    pa, a = people["a"]
    pb, b = people["b"]
    pd = _person_with_faces(conn, "Delta", a, 4, 40)  # the same look as Alpha under another name
    out = _ask(client, person_id=pd)
    assert _ids(out)[0] == pa and pd not in _ids(out)
    assert out["items"][0]["basis"] == "person" and out["items"][0]["likely"] is True
    # seen together in one photo: different people, never offered
    it = _item(conn)
    make_face(conn, it["id"], pd, embedding=_near(a, 801), assigned_by="user", write_crop=False)
    make_face(conn, it["id"], pa, embedding=_near(a, 802), assigned_by="user", box=(0.6, 0.2, 0.2, 0.3),
              write_crop=False)
    assert pa not in _ids(_ask(client, person_id=pd))


def test_vault_boxes_need_the_vault_and_vault_only_people_stay_hidden(client, conn, people):
    pa, a = people["a"]
    secret = _vec(4)
    pv = _person_with_faces(conn, "Victor", secret, 4, 50, nsfw=1)
    hidden = _item(conn, nsfw=1)
    f = make_face(conn, hidden["id"], None, embedding=_near(a, 990), write_crop=False)
    r = client.post("/api/people/similar", json={"face_ids": [f]})
    assert r.status_code == 403 and r.json()["detail"] == "vault_locked"
    r = client.post("/api/people/similar", json={"uids": [hidden["uid"]], "person_id": pa})
    assert r.status_code == 403
    # a face outside the vault that looks like someone seen only in the vault does not name them
    g = make_face(conn, _item(conn)["id"], None, embedding=_near(secret, 989), write_crop=False)
    out = _ask(client, face_ids=[g])
    assert pv not in _ids(out) and str(pv) not in out["scores"]
    assert client.post("/api/people/similar", json={"person_id": pv}).status_code == 404


def test_vault_unlocked_may_ask_about_a_vault_box(vault_client, conn, people):
    pa, a = people["a"]
    hidden = _item(conn, nsfw=1)
    f = make_face(conn, hidden["id"], None, embedding=_near(a, 988), write_crop=False)
    out = _ask(vault_client, face_ids=[f])
    assert _ids(out)[0] == pa


def _pet(conn, name: str | None, species: str) -> int:
    now = now_iso()
    with tx(conn):
        return int(conn.execute(
            "INSERT INTO persons(name, name_key, kind, species, created_at, updated_at) VALUES (?, ?, 'pet', ?, ?, ?)",
            (name, name_key(name) if name else None, species, now, now)).lastrowid)


def _crop(conn, item_id: int, pet: int | None, species: str, vec: np.ndarray, assigned_by: str = "user") -> int:
    with tx(conn):
        return int(conn.execute(
            "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, embedding, created_at) "
            "VALUES (?, ?, ?, 0.2, 0.2, 0.4, 0.4, 0.9, ?, ?, ?)",
            (item_id, pet, assigned_by, species, vec.astype(np.float32).tobytes(), now_iso())).lastrowid)


def test_a_pet_box_lists_the_pet_it_looks_like(client, conn):
    d1, d2 = _vec(60, 768), _vec(61, 768)
    rex, max_ = _pet(conn, "Rex", "dog"), _pet(conn, "Max", "dog")
    for k in range(3):
        _crop(conn, _item(conn, taken_at=f"2025-02-0{k + 1}T08:00:00.000Z")["id"], rex, "dog", _near(d1, 600 + k, 0.2))
        _crop(conn, _item(conn, taken_at=f"2025-03-0{k + 1}T08:00:00.000Z")["id"], max_, "dog", _near(d2, 610 + k, 0.2))
    box = _crop(conn, _item(conn)["id"], None, "dog", _near(d1, 620, 0.2), assigned_by="auto")
    out = _ask(client, animal_ids=[box])
    assert _ids(out)[0] == rex
    assert out["items"][0]["basis"] == "look" and out["items"][0]["likely"] is True
    assert all(o["person"]["kind"] == "pet" for o in out["items"])


def test_around_an_item_lists_who_was_there_and_not_who_is_in_it(client, conn):
    pa, pb = make_person(conn, "Alpha"), make_person(conn, "Bravo")
    for k, p in enumerate((pa, pb)):
        near = _item(conn, taken_at=f"2026-05-01T10:1{k}:00.000Z")
        make_face(conn, near["id"], p, embedding=_vec(70 + k), assigned_by="user", write_crop=False)
    it = _item(conn, taken_at="2026-05-01T10:00:00.000Z")
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, relation, created_at) VALUES (?, ?, 'user', 'in', ?)",
                     (it["id"], pa, now_iso()))
    out = _ask(client, around_uid=it["uid"])
    assert _ids(out) == [pb] and out["items"][0]["basis"] == "hour"


@pytest.mark.parametrize("thr,young,learned,threshold,expected", [
    (0.45, False, None, None, 0.45),
    (0.50, False, None, None, 0.50),
    (0.50, False, 0.40, None, 0.45),           # learned + the weak face's margin
    (0.45, False, None, 0.60, 0.60),           # a caller's fixed cut
    (0.45, True, None, None, 0.48),            # the child cut
    (0.50, True, 0.60, None, 0.65),            # a learned threshold only raises a child's cut
    (0.45, True, None, 0.30, 0.48),            # a fixed cut never lowers it
])
def test_option_cut_is_the_matching_cut(thr, young, learned, threshold, expected):
    assert settings.cluster_match_threshold == 0.45 and settings.cluster_child_threshold == 0.48
    assert cluster.option_cut(thr, young, learned, threshold) == pytest.approx(expected)
