"""Unnamed groups stay in the background (docs/ARCHITECTURE.md section 10): the matching keeps grouping faces nobody
is named for, but no answer of the API shows such a group, and no guess for one; naming one of its faces turns the
group's other faces into guesses for that person (app.analysis.hidden_groups)."""
from __future__ import annotations

import json

import numpy as np

from app.analysis import guess, hidden_groups
from app.db import kv_set, tx
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(base: np.ndarray, seed: int, spread: float = 0.35) -> np.ndarray:
    v = base + spread * _vec(seed)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _item(conn, **kw):
    return make_item(conn, write_files=False, data=None, **kw)


def _group(conn, n: int, base_seed: int = 100, **item_kw) -> tuple[int, list[int], list[dict]]:
    """An unnamed group of n faces, one per photo: (group id, face ids, items)."""
    gid = make_person(conn)
    base = _vec(base_seed)
    items, faces = [], []
    for k in range(n):
        it = _item(conn, **item_kw)
        items.append(it)
        faces.append(make_face(conn, it["id"], gid, embedding=_near(base, base_seed + 1 + k)))
    return gid, faces, items


def _person_of(conn, face_id: int):
    return conn.execute("SELECT person_id, assigned_by, guess_person_id, guess_evidence FROM faces WHERE id = ?",
                        (face_id,)).fetchone()


# ------------------------------------------------------------------ what the API shows
def test_the_people_list_has_named_people_and_pets_only(client, conn):
    asha = make_person(conn, "Asha")
    make_face(conn, _item(conn)["id"], asha, embedding=_vec(1))
    group, _, _ = _group(conn, 2)
    with tx(conn):
        pet = int(conn.execute("INSERT INTO persons(name, created_at, updated_at, kind, species) "
                               "VALUES (NULL, ?, ?, 'pet', 'dog')", (now_iso(), now_iso())).lastrowid)
    people = client.get("/api/people").json()
    assert {p["id"] for p in people["items"]} == {asha, pet} and people["total"] == 2
    assert client.get("/api/people?kind=person&filter=unnamed").json() == {"items": [], "next_cursor": None, "total": 0}
    assert {p["id"] for p in client.get("/api/people?filter=unnamed").json()["items"]} == {pet}  # pets keep theirs
    assert client.get(f"/api/people?q=unnamed%20%23{group}").json()["items"] == []
    stats = client.get("/api/stats").json()
    assert stats["people"] == 2 and stats["named_people"] == 1


def test_an_unnamed_group_is_not_found_by_id(client, conn):
    asha = make_person(conn, "Asha")
    make_face(conn, _item(conn)["id"], asha, embedding=_vec(1))
    group, faces, _ = _group(conn, 2)
    for path in ("", "/faces", "/items", "/traits", "/recognition", "/related-suggestions"):
        assert client.get(f"/api/people/{group}{path}").status_code == 404, path
    assert client.get(f"/api/people/{group}/childhood").status_code in (404, 405)  # the route went with the groups
    assert client.patch(f"/api/people/{group}", json={"name": "Ravi"}).status_code == 404
    assert client.delete(f"/api/people/{group}").status_code == 404
    assert client.post("/api/people/merge", json={"target_id": asha, "source_ids": [group]}).status_code == 404
    assert client.post("/api/faces/assign", json={"face_ids": [faces[0]], "person_id": group}).status_code == 404
    # nothing changed in the background
    assert conn.execute("SELECT COUNT(*) FROM faces WHERE person_id = ?", (group,)).fetchone()[0] == 2


def test_a_face_of_an_unnamed_group_reads_as_a_face_nobody_is_named_for(client, conn):
    group = make_person(conn)
    other = make_person(conn)  # an unnamed group a guess points at
    item = _item(conn)
    face = make_face(conn, item["id"], group, embedding=_vec(3))
    loose = make_face(conn, item["id"], None, embedding=_vec(4), box=(0.6, 0.2, 0.2, 0.3), assigned_by="guess",
                      guess_person_id=other, guess_score=0.8, guess_evidence=json.dumps({"clothes": 0.9}))
    with tx(conn):
        conn.execute("UPDATE faces SET assigned_by = 'user' WHERE id = ?", (face,))  # even one the owner placed there
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, created_at) VALUES (?, ?, 'user', ?)",
                     (item["id"], other, now_iso()))
    detail = client.get(f"/api/items/{item['uid']}").json()
    by_id = {f["id"]: f for f in detail["faces"]}
    assert by_id[face]["person"] is None and by_id[face]["assigned_by"] == "auto" and by_id[face]["by_guess"] is False
    assert by_id[loose]["guess"] is None  # a guess for an unnamed group is not shown
    assert detail["people"] == [] and detail["tags"] == []
    # the background keeps the group
    assert _person_of(conn, face)["person_id"] == group


def test_guesses_for_unnamed_groups_are_hidden_until_the_group_has_a_name(client, conn):
    other = make_person(conn)
    make_face(conn, _item(conn)["id"], other, embedding=_vec(5))
    item = _item(conn)
    ev = json.dumps({"clothes": 0.9, "scene": 0.85})
    face = make_face(conn, item["id"], None, embedding=_vec(6), assigned_by="guess", guess_person_id=other,
                     guess_score=0.8, guess_evidence=ev)
    assert client.get("/api/review/guesses").json() == {"items": [], "next_cursor": None, "total": 0}
    assert client.get("/api/stats").json()["guesses"] == 0
    with tx(conn):  # named some other way (a merge into a named person keeps the guess fields where they are)
        conn.execute("UPDATE persons SET name = 'Ravi', name_key = 'ravi' WHERE id = ?", (other,))
    page = client.get("/api/review/guesses").json()
    assert page["total"] == 1 and page["items"][0]["id"] == face and page["items"][0]["guess"]["person"]["id"] == other
    assert client.get("/api/stats").json()["guesses"] == 1


def test_suggestions_pair_named_people_only(client, conn):
    base = _vec(50)
    a, b, c = make_person(conn, "Asha"), make_person(conn, "Meera"), make_person(conn)
    for pid, seed in ((a, 51), (b, 52), (c, 53)):
        make_face(conn, _item(conn)["id"], pid, embedding=_near(base, seed, 0.5))
    pairs = [{s["a"]["id"], s["b"]["id"]} for s in client.get("/api/suggestions").json()["items"]]
    assert pairs == [{a, b}]


def test_faces_to_name_lists_group_faces_side_by_side(client, conn):
    small, small_faces, _ = _group(conn, 2, base_seed=200)
    big, big_faces, _ = _group(conn, 3, base_seed=300)
    loose = make_face(conn, _item(conn)["id"], None, embedding=_vec(7), det_score=0.95)
    named = make_person(conn, "Asha")
    make_face(conn, _item(conn)["id"], named, embedding=_vec(8))
    make_face(conn, _item(conn, nsfw=1)["id"], big, embedding=_vec(9))  # a vault face: not listed, not counted
    with tx(conn):
        conn.execute("UPDATE faces SET quality = 0.9 WHERE id = ?", (big_faces[2],))
    page = client.get("/api/faces/unassigned").json()
    ids = [f["id"] for f in page["items"]]
    assert page["total"] == 6
    assert ids[:3] == [big_faces[2], big_faces[1], big_faces[0]]  # the larger group first, its best face first
    assert set(ids[3:5]) == set(small_faces) and ids[5] == loose
    assert all(f["person_id"] is None for f in page["items"])
    seen, cursor = [], None
    while True:
        part = client.get("/api/faces/unassigned", params={"limit": 2, **({"cursor": cursor} if cursor else {})}).json()
        seen += [f["id"] for f in part["items"]]
        cursor = part["next_cursor"]
        if not cursor:
            break
    assert seen == ids
    assert client.get("/api/faces/unassigned", params={"cursor": "bad"}).status_code == 400


def test_the_selection_sheet_counts_group_faces_as_not_named(client, conn):
    asha = make_person(conn, "Asha")
    group = make_person(conn)
    item = _item(conn)
    make_face(conn, item["id"], asha, embedding=_vec(10))
    make_face(conn, item["id"], group, embedding=_vec(11), box=(0.6, 0.2, 0.2, 0.3))
    make_face(conn, item["id"], None, embedding=_vec(12), box=(0.1, 0.6, 0.2, 0.3))
    out = client.post("/api/items/people", json={"uids": [item["uid"]]}).json()
    assert [p["id"] for p in out["people"]] == [asha] and out["unnamed_faces"] == 2


# ------------------------------------------------------------------ naming a face of a group
def test_naming_one_face_turns_the_rest_of_its_group_into_guesses(client, conn):
    group, faces, items = _group(conn, 3)
    r = client.post("/api/faces/assign", json={"face_ids": [faces[0]], "new_person_name": "Asha"})
    assert r.status_code == 200, r.text
    out = r.json()
    asha = out["person"]["id"]
    assert out["updated"] == 1 and out["guessed"] == 2 and out["followed"] == 0
    assert out["person"]["face_count"] == 1  # only that face took the name
    for f in faces[1:]:
        row = _person_of(conn, f)
        ev = json.loads(row["guess_evidence"])
        assert (row["person_id"], row["assigned_by"], row["guess_person_id"]) == (None, "guess", asha)
        assert ev["basis"] == "group" and ev["was_person_id"] == group and ev["ref_face_id"] == faces[0]
        assert 0.0 <= ev["score"] <= 1.0
    # the group is gone (left empty); its faces wait as guesses, in Review and on the photos
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE id = ?", (group,)).fetchone()[0] == 0
    page = client.get("/api/review/guesses").json()
    assert page["total"] == 2 and {g["id"] for g in page["items"]} == set(faces[1:])
    assert client.get("/api/review/guesses?basis=group").json()["total"] == 2
    assert client.get("/api/review/guesses?basis=clothes").json()["total"] == 0
    detail = client.get(f"/api/items/{items[1]['uid']}").json()
    assert detail["faces"][0]["guess"]["person"]["id"] == asha

    # Yes names the face; No puts it back into an unnamed group (one comes back for it)
    assert client.post("/api/guesses/confirm", json={"faces": [faces[1]]}).json() == {"updated": 1}
    assert _person_of(conn, faces[1])["person_id"] == asha
    assert client.post("/api/guesses/reject", json={"faces": [faces[2]]}).json() == {"updated": 1}
    back = _person_of(conn, faces[2])
    assert back["person_id"] is not None and back["person_id"] != asha and back["guess_person_id"] is None
    assert conn.execute("SELECT name FROM persons WHERE id = ?", (back["person_id"],)).fetchone()[0] is None


def test_faces_that_cannot_be_the_person_stay_in_the_group(client, conn):
    asha = make_person(conn, "Asha")
    make_face(conn, _item(conn)["id"], asha, embedding=_vec(20))
    group, faces, items = _group(conn, 2, base_seed=400)
    with_asha = _item(conn)  # Asha is in this photo already
    make_face(conn, with_asha["id"], asha, embedding=_vec(21))
    in_her_photo = make_face(conn, with_asha["id"], group, embedding=_near(_vec(400), 450), box=(0.6, 0.2, 0.2, 0.3))
    vault = make_face(conn, _item(conn, nsfw=1)["id"], group, embedding=_near(_vec(400), 451))
    turned_down = make_face(conn, _item(conn)["id"], group, embedding=_near(_vec(400), 452))
    two = _item(conn)  # two faces of the group in one photo: only the more alike becomes a guess
    alike = make_face(conn, two["id"], group, embedding=_near(_vec(400), 453, 0.1))
    less = make_face(conn, two["id"], group, embedding=_near(_vec(400), 454, 1.5), box=(0.6, 0.2, 0.2, 0.3))
    with tx(conn):
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (turned_down, asha, now_iso()))
    r = client.post("/api/faces/assign", json={"face_ids": [faces[0]], "person_id": asha}).json()
    assert r["guessed"] == 2
    assert _person_of(conn, faces[1])["guess_person_id"] == asha and _person_of(conn, alike)["guess_person_id"] == asha
    for f in (in_her_photo, vault, turned_down, less):
        row = _person_of(conn, f)
        assert (row["person_id"], row["guess_person_id"]) == (group, None), f
    # the named face is remembered as not the group, which is still there
    assert conn.execute("SELECT 1 FROM guess_rejections WHERE kind = 'face' AND ref_id = ? AND person_id = ?",
                        (faces[0], group)).fetchone() is not None


def test_a_group_the_owner_said_is_not_the_person_gives_no_guesses(client, conn):
    asha = make_person(conn, "Asha")
    make_face(conn, _item(conn)["id"], asha, embedding=_vec(30))
    group, faces, _ = _group(conn, 3, base_seed=500)
    a, b = sorted((asha, group))
    with tx(conn):
        conn.execute("INSERT INTO merge_dismissals(person_a, person_b, created_at) VALUES (?, ?, ?)", (a, b, now_iso()))
    r = client.post("/api/faces/assign", json={"face_ids": [faces[0]], "person_id": asha}).json()
    assert r["updated"] == 1 and r["guessed"] == 0
    assert all(_person_of(conn, f)["person_id"] == group for f in faces[1:])


def test_nobody_for_a_group_face_changes_only_that_face(client, conn):
    group, faces, _ = _group(conn, 3, base_seed=600)
    r = client.post("/api/faces/assign", json={"face_ids": [faces[0]]}).json()
    assert r == {"updated": 1, "person": None, "followed": 0, "guessed": 0}
    assert (_person_of(conn, faces[0])["person_id"], _person_of(conn, faces[0])["assigned_by"]) == (None, "user")
    assert all(_person_of(conn, f)["person_id"] == group for f in faces[1:])


def test_naming_or_clearing_the_only_face_of_a_group(client, conn):
    """The move leaves the group empty and it goes: nothing to remember it by, nothing to guess (no 500)."""
    group, faces, _ = _group(conn, 1, base_seed=1100)
    r = client.post("/api/faces/assign", json={"face_ids": [faces[0]], "new_person_name": "Asha"})
    assert r.status_code == 200, r.text
    assert r.json()["guessed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM persons WHERE id = ?", (group,)).fetchone()[0] == 0
    group, faces, _ = _group(conn, 1, base_seed=1200)
    r = client.post("/api/faces/assign", json={"face_ids": [faces[0]]})
    assert r.status_code == 200, r.text


def test_a_new_person_from_group_faces_follows_the_same_rule(client, conn):
    group, faces, _ = _group(conn, 3, base_seed=700)
    r = client.post("/api/people", json={"name": "Meera", "face_ids": [faces[0]]})
    assert r.status_code == 200, r.text
    meera = r.json()["id"]
    assert r.json()["face_count"] == 1
    assert {_person_of(conn, f)["guess_person_id"] for f in faces[1:]} == {meera}


def test_someone_else_on_a_group_face_follows_the_same_rule(client, conn):
    group, faces, _ = _group(conn, 3, base_seed=800)
    r = client.post("/api/guesses/correct", json={"faces": [faces[0]], "new_person_name": "Kiran"}).json()
    assert r["updated"] == 1 and r["guessed"] == 2
    kiran = r["person"]["id"]
    assert {_person_of(conn, f)["guess_person_id"] for f in faces[1:]} == {kiran}


# ------------------------------------------------------------------ the background keeps them
def test_guessing_again_keeps_a_group_guess_until_it_no_longer_stands(client, conn):
    group, faces, items = _group(conn, 2, base_seed=900)
    asha = client.post("/api/faces/assign", json={"face_ids": [faces[0]], "new_person_name": "Asha"}).json()["person"]["id"]
    before = _person_of(conn, faces[1])
    guess.guess_item(conn, int(items[1]["id"]))  # the clothes path finds nothing here
    after = _person_of(conn, faces[1])
    assert (after["guess_person_id"], after["guess_evidence"]) == (asha, before["guess_evidence"])
    # once Asha is tagged in that photo, the guess no longer stands: it goes, and the face goes back into a group
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, created_at) VALUES (?, ?, 'user', ?)",
                     (items[1]["id"], asha, now_iso()))
    guess.guess_item(conn, int(items[1]["id"]))
    row = _person_of(conn, faces[1])
    assert row["guess_person_id"] is None and row["person_id"] is not None and row["person_id"] != asha


def test_a_changed_guess_window_keeps_group_guesses(client, conn):
    group, faces, _ = _group(conn, 2, base_seed=1000)
    asha = client.post("/api/faces/assign", json={"face_ids": [faces[0]], "new_person_name": "Asha"}).json()["person"]["id"]
    with tx(conn):
        kv_set(conn, guess.WINDOW_KEY, "999")
    assert guess.apply_window_change(conn) == 0
    assert _person_of(conn, faces[1])["guess_person_id"] == asha


def test_hold_drops_another_box_guess_for_the_same_person_in_the_photo(conn):
    asha = make_person(conn, "Asha")
    item = _item(conn)
    ev = json.dumps({"basis": hidden_groups.BASIS, "score": 0.6, "face": 0.6, "was_person_id": 1})
    held = make_face(conn, item["id"], None, embedding=_vec(40), assigned_by="guess", guess_person_id=asha,
                     guess_score=0.6, guess_evidence=ev)
    other = make_face(conn, item["id"], None, embedding=_vec(41), box=(0.6, 0.2, 0.2, 0.3))
    boxes = guess.open_boxes(conn, int(item["id"]))
    row = conn.execute("SELECT id, kind, nsfw FROM items WHERE id = ?", (item["id"],)).fetchone()
    computed = {("face", held): None, ("face", other): guess.Candidate(asha, 0.9, {"clothes": 0.95})}
    out = hidden_groups.hold(conn, row, boxes, computed)
    assert out[("face", held)].person_id == asha and out[("face", held)].evidence["basis"] == "group"
    assert out[("face", other)] is None
    # nothing held: the same object comes back
    plain = {("face", other): None}
    assert hidden_groups.hold(conn, row, [b for b in boxes if b.id == other], plain) is plain


def test_a_face_named_in_a_vault_photo_is_not_the_reference_of_its_guesses(vault_client, conn):
    """The guesses sit in photos outside the vault, which a locked session reviews too: their evidence must not
    point at the vault photo the owner named the face in (no other guess has a vault photo as its reference)."""
    group, faces, _ = _group(conn, 3, base_seed=1300)
    vault_item = _item(conn, nsfw=1)
    hidden = make_face(conn, vault_item["id"], group, embedding=_near(_vec(1300), 1350))
    r = vault_client.post("/api/faces/assign", json={"face_ids": [hidden], "new_person_name": "Asha"})
    assert r.status_code == 200, r.text
    assert r.json()["guessed"] == 3
    for f in faces:
        ev = json.loads(_person_of(conn, f)["guess_evidence"])
        assert ev["basis"] == "group" and ev["was_person_id"] == group
        assert "ref_face_id" not in ev and "ref_item_uid" not in ev, ev
    assert vault_client.post("/api/auth/vault/lock").status_code == 200
    page = vault_client.get("/api/review/guesses").json()
    assert page["total"] == 3 and vault_item["uid"] not in json.dumps(page)


def test_the_reference_of_group_guesses_is_a_named_face_outside_the_vault(vault_client, conn):
    """Faces named together from one group, one of them in the vault: the reference is the one in the library."""
    group, faces, items = _group(conn, 4, base_seed=1400)
    # the vault face is the better one, the reference a plain choice would take
    hidden = make_face(conn, _item(conn, nsfw=1)["id"], group, embedding=_near(_vec(1400), 1450, 0.01),
                       det_score=0.99, size_px=200)
    r = vault_client.post("/api/faces/assign", json={"face_ids": [hidden, faces[0]], "new_person_name": "Asha"})
    assert r.status_code == 200, r.text
    for f in faces[1:]:
        ev = json.loads(_person_of(conn, f)["guess_evidence"])
        assert (ev["ref_face_id"], ev["ref_item_uid"]) == (faces[0], items[0]["uid"]), ev
