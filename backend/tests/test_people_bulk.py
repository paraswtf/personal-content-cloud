"""People on a selection (docs/ARCHITECTURE.md section 10): who the selected items show, and moving, removing
or tagging a person or a pet on all of them at once."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.analysis import cluster
from app.analysis.index import get_index
from app.db import tx
from app.util import now_iso
from tests.conftest import ACCOUNT
from tests.helpers import make_face, make_item, make_person
from tests.test_learning_core import near, unit


def _items(conn, n: int, **kw):
    return [make_item(conn, filename=f"s{k}.jpg", write_files=False, **kw) for k in range(n)]


def _pet(conn, name: str | None, species: str) -> int:
    with tx(conn):
        return cluster.new_person(conn, name, kind="pet", species=species)


def _animal(conn, item_id: int, person_id: int | None, species: str = "cat") -> int:
    with tx(conn):
        return int(conn.execute(
            "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, created_at) "
            "VALUES (?, ?, 'auto', 0.1, 0.1, 0.4, 0.4, 0.9, ?, ?)", (item_id, person_id, species, now_iso())).lastrowid)


def _body(conn, item_id: int, person_id: int | None, face_id: int | None = None) -> int:
    with tx(conn):
        return int(conn.execute(
            "INSERT INTO bodies(item_id, face_id, person_id, assigned_by, x, y, w, h, det_score, created_at) "
            "VALUES (?, ?, ?, 'auto', 0.2, 0.2, 0.5, 0.7, 0.9, ?)", (item_id, face_id, person_id, now_iso())).lastrowid)


def _tag(conn, item_id: int, person_id: int, relation: str = "in") -> None:
    with tx(conn):
        conn.execute("INSERT INTO item_tags(item_id, person_id, source, relation, created_at) VALUES (?, ?, 'user', ?, ?)",
                     (item_id, person_id, relation, now_iso()))


def _tags(conn, item_id: int) -> dict[int, str]:
    return {int(r[0]): r[1] for r in conn.execute("SELECT person_id, relation FROM item_tags WHERE item_id = ?", (item_id,))}


def _face_owner(conn, fid: int):
    r = conn.execute("SELECT person_id, assigned_by FROM faces WHERE id = ?", (fid,)).fetchone()
    return (r["person_id"], r["assigned_by"])


def _rejected(conn, pid: int) -> set[int]:
    return {int(r[0]) for r in conn.execute("SELECT ref_id FROM guess_rejections WHERE kind = 'face' AND person_id = ?", (pid,))}


def _mixed_person(conn):
    """Asha: three A faces (one confirmed) and three automatically grouped B faces, one per item."""
    get_index().invalidate()
    A, B = unit(1, 512), unit(2, 512)
    asha = make_person(conn, "Asha")
    items = _items(conn, 8)
    a_faces = [make_face(conn, items[0]["id"], asha, embedding=near(A, 11, 0.3), assigned_by="user", write_crop=False),
               make_face(conn, items[1]["id"], asha, embedding=near(A, 12, 0.3), write_crop=False),
               make_face(conn, items[2]["id"], asha, embedding=near(A, 13, 0.3), write_crop=False)]
    b_faces = [make_face(conn, items[3]["id"], asha, embedding=near(B, 21, 0.3), write_crop=False),
               make_face(conn, items[4]["id"], asha, embedding=near(B, 22, 0.3), write_crop=False),
               make_face(conn, items[5]["id"], asha, embedding=near(B, 23, 0.3), write_crop=False)]
    return asha, a_faces, b_faces, items


def test_the_list_counts_faces_bodies_pet_boxes_and_both_relations(client, conn):
    a, b, c, d = _items(conn, 4)
    gone = make_item(conn, filename="gone.jpg", write_files=False, deleted_at=now_iso())
    asha, ben, nobody = make_person(conn, "Asha"), make_person(conn, "Ben"), make_person(conn, None)
    chikki = _pet(conn, "Chikki", "cat")
    make_face(conn, a["id"], asha, write_crop=False)
    make_face(conn, b["id"], asha, write_crop=False)
    _body(conn, c["id"], asha)
    _tag(conn, a["id"], ben, "in")
    _tag(conn, d["id"], ben, "related")
    _animal(conn, c["id"], chikki)
    make_face(conn, b["id"], nobody, write_crop=False)
    make_face(conn, d["id"], None, write_crop=False)
    make_face(conn, gone["id"], asha, write_crop=False)   # in the trash: skipped

    r = client.post("/api/items/people", json={"uids": [a["uid"], b["uid"], c["uid"], d["uid"], gone["uid"], "nope", a["uid"]]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["items"] == 4 and out["unnamed_faces"] == 1
    people = out["people"]
    assert [p["id"] for p in people] == [asha, ben, chikki, nobody]   # most items first, named before unnamed
    by = {p["id"]: p for p in people}
    assert (by[asha]["items"], by[asha]["faces"], by[asha]["bodies"], by[asha]["tags"]) == (3, 2, 1, 0)
    assert (by[ben]["items"], by[ben]["tags"], by[ben]["related"], by[ben]["faces"]) == (2, 1, 1, 0)
    assert (by[chikki]["kind"], by[chikki]["species"], by[chikki]["animals"], by[chikki]["items"]) == ("pet", "cat", 1, 1)
    assert by[nobody]["name"] is None and by[nobody]["faces"] == 1
    assert by[asha]["cover_url"] and by[asha]["name"] == "Asha" and by[asha]["kind"] == "person"
    assert client.get(f"/api/items/{a['uid']}").status_code == 200   # the item route is not shadowed


def test_the_list_counts_the_items_where_the_owner_put_someone(client, conn):
    """`by_you`: items with a box the owner tagged or drew, or a tag; recognition alone does not count."""
    a, b, c, d, e = _items(conn, 5)
    asha, ben, carl = make_person(conn, "Asha"), make_person(conn, "Ben"), make_person(conn, "Carl")
    chikki = _pet(conn, "Chikki", "cat")
    make_face(conn, a["id"], asha, write_crop=False)                       # recognised
    make_face(conn, b["id"], asha, assigned_by="user", write_crop=False)   # the owner tagged it
    make_face(conn, b["id"], asha, box=(0.6, 0.2, 0.2, 0.3), write_crop=False)   # recognised beside it
    second = _body(conn, b["id"], asha)                                    # a second box of the owner's on b
    drawn = _body(conn, c["id"], asha)
    _body(conn, d["id"], asha)                                             # found and grouped by the system
    _tag(conn, e["id"], ben, "related")
    make_face(conn, a["id"], carl, box=(0.6, 0.2, 0.2, 0.3), write_crop=False)
    _animal(conn, a["id"], chikki)
    pet_box = _animal(conn, b["id"], chikki)
    with tx(conn):
        conn.execute("UPDATE bodies SET source = 'user' WHERE id = ?", (drawn,))     # drawn, even if not locked
        conn.execute("UPDATE bodies SET assigned_by = 'user' WHERE id = ?", (second,))   # b still counts once
        conn.execute("UPDATE animals SET source = 'user' WHERE id = ?", (pet_box,))

    r = client.post("/api/items/people", json={"uids": [x["uid"] for x in (a, b, c, d, e)]})
    assert r.status_code == 200, r.text
    by = {p["id"]: p for p in r.json()["people"]}
    assert (by[asha]["items"], by[asha]["by_you"]) == (4, 2)
    assert (by[ben]["items"], by[ben]["by_you"]) == (1, 1)
    assert (by[chikki]["items"], by[chikki]["by_you"]) == (2, 1)
    assert (by[carl]["items"], by[carl]["by_you"]) == (1, 0)


def test_a_move_to_someone_else_takes_lookalikes_and_tags_along(client, conn):
    asha, a_faces, b_faces, items = _mixed_person(conn)
    bela = make_person(conn, "Bela")
    _tag(conn, items[6]["id"], asha, "related")
    _tag(conn, items[7]["id"], asha, "in")
    _tag(conn, items[7]["id"], bela, "related")
    uids = [items[3]["uid"], items[4]["uid"], items[6]["uid"], items[7]["uid"]]
    r = client.post("/api/items/people/move", json={"uids": uids, "from_person_id": asha, "person_id": bela})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["updated"] == 4 and out["followed"] == 1 and out["person"]["id"] == bela
    assert _face_owner(conn, b_faces[0]) == (bela, "user") and _face_owner(conn, b_faces[1]) == (bela, "user")
    assert _face_owner(conn, b_faces[2]) == (bela, "auto")   # the look-alike outside the selection followed
    assert all(_face_owner(conn, f)[0] == asha for f in a_faces)
    assert _rejected(conn, asha) == set(b_faces)
    assert _tags(conn, items[6]["id"]) == {bela: "related"}    # the relation goes along
    assert _tags(conn, items[7]["id"]) == {bela: "in"}         # 'in' wins over Bela's 'related'


def test_a_lookalike_the_owner_refused_for_the_target_stays(client, conn):
    asha, a_faces, b_faces, items = _mixed_person(conn)
    bela = make_person(conn, "Bela")
    with tx(conn):   # the owner once took this face off Bela
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (b_faces[2], bela, now_iso()))
    r = client.post("/api/items/people/move", json={"uids": [items[3]["uid"], items[4]["uid"]], "from_person_id": asha,
                                                    "person_id": bela})
    assert r.status_code == 200 and r.json()["followed"] == 0, r.text
    assert _face_owner(conn, b_faces[2]) == (asha, "auto") and _rejected(conn, bela) == {b_faces[2]}


def test_a_rejection_merged_onto_a_face_of_that_person_is_no_rejection(conn):
    """The owner said face X is not Sam; X is Asha's; Sam is merged into Asha: the copied "not Asha" names
    one of her own faces, and neither matching nor the learner may use it against her."""
    from app.learning import people as learned

    get_index().invalidate()
    A = unit(5, 512)
    asha, sam = make_person(conn, "Asha"), make_person(conn)
    items = _items(conn, 6)
    for k in range(3):
        make_face(conn, items[k]["id"], asha, embedding=near(A, 50 + k, 0.3), assigned_by="user", write_crop=False)
    x = make_face(conn, items[3]["id"], asha, embedding=near(A, 60, 0.3), write_crop=False)
    make_face(conn, items[4]["id"], sam, embedding=near(A, 61, 0.3), write_crop=False)
    before = learned.person_thresholds(conn).get(asha)
    with tx(conn):
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (x, sam, now_iso()))
    cluster.merge_persons(conn, asha, [sam])
    assert x in _rejected(conn, asha) and _face_owner(conn, x)[0] == asha
    assert learned.person_thresholds(conn).get(asha) == before
    assert asha not in cluster.rejected_examples(conn, [asha])


def test_a_move_to_a_new_name_and_the_refusals(client, conn):
    a, b = _items(conn, 2)
    asha = make_person(conn, "Asha")
    face = make_face(conn, a["id"], asha, write_crop=False)
    _body(conn, b["id"], asha)    # a faceless body moves too
    r = client.post("/api/items/people/move", json={"uids": [a["uid"], b["uid"]], "from_person_id": asha,
                                                    "new_person_name": "  Priya "})
    assert r.status_code == 200, r.text
    priya = r.json()["person"]
    assert priya["name"] == "Priya" and priya["kind"] == "person" and r.json()["updated"] == 2
    assert _face_owner(conn, face) == (priya["id"], "user")
    assert conn.execute("SELECT person_id FROM bodies WHERE item_id = ?", (b["id"],)).fetchone()[0] == priya["id"]
    move = lambda body: client.post("/api/items/people/move", json={"uids": [a["uid"]], **body})  # noqa: E731
    assert move({"from_person_id": priya["id"], "person_id": priya["id"]}).status_code == 400   # the same person
    assert move({"from_person_id": priya["id"], "new_person_name": "priya"}).status_code == 400  # the same, by name
    assert move({"from_person_id": priya["id"]}).status_code == 400                             # no target
    assert move({"from_person_id": priya["id"], "person_id": asha, "new_person_name": "X"}).status_code == 400
    assert move({"from_person_id": 999999, "person_id": asha}).status_code == 404


def test_a_pet_moves_only_to_a_pet_of_its_species(client, conn):
    x, y = _items(conn, 2)
    chikki, mau, bruno = _pet(conn, "Chikki", "cat"), _pet(conn, "Mau", "cat"), _pet(conn, "Bruno", "dog")
    asha = make_person(conn, "Asha")
    boxes = [_animal(conn, x["id"], chikki), _animal(conn, y["id"], chikki)]
    face = make_face(conn, x["id"], asha, write_crop=False)
    uids = [x["uid"], y["uid"]]
    r = client.post("/api/items/people/move", json={"uids": uids, "from_person_id": chikki, "person_id": asha})
    assert r.status_code == 400 and "person" in r.json()["detail"]
    r = client.post("/api/items/people/move", json={"uids": uids, "from_person_id": chikki, "person_id": bruno})
    assert r.status_code == 400 and "dog" in r.json()["detail"]
    r = client.post("/api/items/people/move", json={"uids": uids, "from_person_id": chikki, "person_id": mau})
    assert r.status_code == 200 and r.json()["updated"] == 2, r.text
    assert {tuple(r) for r in conn.execute(f"SELECT person_id, assigned_by FROM animals WHERE id IN ({boxes[0]}, {boxes[1]})")} == {(mau, "user")}
    assert conn.execute("SELECT 1 FROM persons WHERE id = ?", (chikki,)).fetchone() is not None   # named: kept
    r = client.post("/api/items/people/move", json={"uids": uids, "from_person_id": mau, "new_person_name": "Kitty"})
    assert r.status_code == 200, r.text
    kitty = r.json()["person"]
    assert (kitty["name"], kitty["kind"], kitty["species"]) == ("Kitty", "pet", "cat")
    # a person never moves to a pet
    r = client.post("/api/items/people/move", json={"uids": [x["uid"]], "from_person_id": asha, "person_id": mau})
    assert r.status_code == 400 and "pet" in r.json()["detail"]
    assert _face_owner(conn, face)[0] == asha


def test_remove_unassigns_their_boxes_and_deletes_their_tags_there(client, conn):
    a, b, c = _items(conn, 3)
    asha = make_person(conn, "Asha")
    fa = make_face(conn, a["id"], asha, write_crop=False)
    fb = make_face(conn, b["id"], asha, write_crop=False)
    _tag(conn, c["id"], asha, "related")
    r = client.post("/api/items/people/remove", json={"uids": [a["uid"], c["uid"]], "person_id": asha})
    assert r.status_code == 200, r.text
    assert r.json() == {"updated": 2, "person": None, "followed": 0, "pulled": 0}
    assert _face_owner(conn, fa) == (None, "user") and _face_owner(conn, fb)[0] == asha
    assert _tags(conn, c["id"]) == {} and fa in _rejected(conn, asha)


def test_remove_lets_the_lookalikes_go_and_regroup(client, conn):
    asha, a_faces, b_faces, items = _mixed_person(conn)
    r = client.post("/api/items/people/remove", json={"uids": [items[3]["uid"]], "person_id": asha})
    assert r.status_code == 200 and r.json()["followed"] == 2, r.text
    assert _face_owner(conn, b_faces[0]) == (None, "user")
    others = {_face_owner(conn, f)[0] for f in b_faces[1:]}
    assert None not in others and asha not in others and len(others) == 1
    assert all(_face_owner(conn, f)[0] == asha for f in a_faces)


def test_a_bulk_tag_of_both_kinds_and_relations(client, conn):
    a, b, c = _items(conn, 3)
    asha = make_person(conn, "Asha")
    make_face(conn, a["id"], asha, write_crop=False)
    uids = [a["uid"], b["uid"], c["uid"]]
    tag = lambda body: client.post("/api/items/people/tag", json={"uids": uids, **body})  # noqa: E731
    r = tag({"person_id": asha})
    assert r.status_code == 200 and r.json()["updated"] == 2, r.text   # a face already says she is in a
    assert _tags(conn, a["id"]) == {} and _tags(conn, b["id"]) == {asha: "in"} == _tags(conn, c["id"])
    assert tag({"person_id": asha}).json()["updated"] == 0              # nothing new
    r = tag({"person_id": asha, "relation": "related"})
    assert r.json()["updated"] == 3 and _tags(conn, a["id"]) == {asha: "related"}   # 'related' is always recorded
    r = tag({"new_person_name": "Bruno", "kind": "pet"})
    assert r.status_code == 400 and r.json()["detail"] == "species_needed"
    r = tag({"new_person_name": "Bruno", "kind": "pet", "species": "dog"})
    assert r.status_code == 200, r.text
    bruno = r.json()["person"]
    assert (bruno["kind"], bruno["species"], r.json()["updated"]) == ("pet", "dog", 3)
    r = tag({"new_person_name": "Zoya", "relation": "related"})
    zoya = r.json()["person"]["id"]
    assert all(_tags(conn, i["id"])[zoya] == "related" for i in (a, b, c))
    assert tag({}).status_code == 400 and tag({"person_id": asha, "new_person_name": "X"}).status_code == 400
    assert tag({"person_id": asha, "kind": "pet", "species": "unicorn"}).status_code == 422


def test_vault_items_need_the_vault_open(client, conn):
    v = make_item(conn, filename="v.jpg", write_files=False, nsfw=1)
    asha, bela = make_person(conn, "Asha"), make_person(conn, "Bela")
    face = make_face(conn, v["id"], asha, write_crop=False)
    calls = [("/api/items/people", {"uids": [v["uid"]]}),
             ("/api/items/people/move", {"uids": [v["uid"]], "from_person_id": asha, "person_id": bela}),
             ("/api/items/people/remove", {"uids": [v["uid"]], "person_id": asha}),
             ("/api/items/people/tag", {"uids": [v["uid"]], "person_id": bela})]
    for path, body in calls:
        r = client.post(path, json=body)
        assert r.status_code == 403 and r.json()["detail"] == "vault_locked", (path, r.text)
    assert _face_owner(conn, face)[0] == asha
    assert client.post("/api/auth/vault/unlock", json={"password": ACCOUNT["password"]}).status_code == 200
    assert client.post(*calls[0][:1], json=calls[0][1]).json()["people"][0]["id"] == asha
    r = client.post(calls[1][0], json=calls[1][1])
    assert r.status_code == 200 and _face_owner(conn, face)[0] == bela, r.text


def _body_rejected(conn, pid: int) -> set[int]:
    return {int(r[0]) for r in conn.execute("SELECT ref_id FROM guess_rejections WHERE kind = 'body' AND person_id = ?", (pid,))}


def test_bodies_taken_away_are_remembered_and_forgotten_when_put_back(client, conn):
    a, b, c = _items(conn, 3)
    asha, bela = make_person(conn, "Asha"), make_person(conn, "Bela")
    face = make_face(conn, a["id"], asha, write_crop=False)
    around = _body(conn, a["id"], asha, face_id=face)   # goes with its face
    faceless = _body(conn, b["id"], asha)
    other = _body(conn, c["id"], asha)
    r = client.post("/api/items/people/move", json={"uids": [a["uid"], b["uid"]], "from_person_id": asha, "person_id": bela})
    assert r.status_code == 200, r.text
    assert _body_rejected(conn, asha) == {around, faceless} and _rejected(conn, asha) == {face}
    assert client.post("/api/items/people/remove", json={"uids": [c["uid"]], "person_id": asha}).status_code == 200
    assert _body_rejected(conn, asha) == {around, faceless, other}
    # the owner puts the face back on Asha: its "not Asha", and its body's, are forgotten
    r = client.post("/api/faces/assign", json={"face_ids": [face], "person_id": asha})
    assert r.status_code == 200, r.text
    assert _rejected(conn, asha) == set() and _body_rejected(conn, asha) == {faceless, other}
    assert _face_owner(conn, face) == (asha, "user") and face in _rejected(conn, bela)
    r = client.post("/api/bodies/assign", json={"body_ids": [faceless], "person_id": asha})
    assert r.status_code == 200, r.text
    assert _body_rejected(conn, asha) == {other}


def _axis(k: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[k] = 1.0
    return v


def _mix(c: float, k: int) -> np.ndarray:
    """Cosine exactly c to _axis(0); two mixes along different axes have cosine c1 * c2."""
    return (c * _axis(0) + np.sqrt(1 - c * c) * _axis(k)).astype(np.float32)


def _merge_events(conn, pid: int) -> list[dict]:
    return [json.loads(r[0]) for r in conn.execute(
        "SELECT features FROM learning_events WHERE kind = 'merge' AND person_id = ? ORDER BY id", (pid,))]


def test_a_move_to_someone_is_learned_like_a_merge_and_pulls_lookalike_groups(client, conn):
    get_index().invalidate()
    one = lambda: make_item(conn, write_files=False)  # noqa: E731
    bela = make_person(conn, "Bela")
    for k in range(3):   # Bela: three faces on the axis, one the owner confirmed
        make_face(conn, one()["id"], bela, embedding=_axis(0), assigned_by="user" if k == 0 else "auto", write_crop=False)
    asha = make_person(conn, "Asha")
    x, y, z = one(), one(), one()
    moved = [make_face(conn, x["id"], asha, embedding=_mix(0.5, 1), write_crop=False),
             make_face(conn, y["id"], asha, embedding=_mix(0.5, 2), write_crop=False)]
    stays = make_face(conn, z["id"], asha, embedding=_axis(9), write_crop=False)
    # a look-alike of the first moved face (0.9) follows it to Bela: left out of what the owner accepted
    along = make_face(conn, one()["id"], asha, embedding=(0.9 * _mix(0.5, 1) + np.sqrt(1 - 0.81) * _axis(6)).astype(np.float32),
                      write_crop=False)
    near, far = make_person(conn), make_person(conn)   # unnamed groups
    near_face = make_face(conn, one()["id"], near, embedding=_mix(0.6, 3), write_crop=False)   # 0.6 >= 0.5: pulled
    make_face(conn, one()["id"], far, embedding=_mix(0.45, 4), write_crop=False)               # 0.45 < 0.5: stays

    r = client.post("/api/items/people/move", json={"uids": [x["uid"], y["uid"]], "from_person_id": asha, "person_id": bela})
    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["updated"], out["followed"], out["pulled"]) == (2, 1, 1)
    assert all(_face_owner(conn, f) == (bela, "user") for f in moved) and _face_owner(conn, along) == (bela, "auto")
    assert _face_owner(conn, near_face)[0] == bela and _face_owner(conn, stays)[0] == asha
    persons = {int(r[0]) for r in conn.execute("SELECT id FROM persons")}
    assert near not in persons and far in persons
    (ev,) = _merge_events(conn, bela)
    assert ev["accepted"] == pytest.approx(0.5, abs=0.01) and ev["faces"] == 2 and ev["followed"] == 1

    # a new name has no other faces to compare with: nothing is pulled, nothing loosens
    r = client.post("/api/items/people/move", json={"uids": [z["uid"]], "from_person_id": asha, "new_person_name": "Chandni"})
    assert r.status_code == 200, r.text
    chandni = r.json()["person"]["id"]
    assert r.json()["pulled"] == 0 and far in {int(r[0]) for r in conn.execute("SELECT id FROM persons")}
    (ev,) = _merge_events(conn, chandni)
    assert ev["accepted"] is None and ev["faces"] == 1 and ev["followed"] == 0


def test_a_move_the_matching_finds_unlike_the_target_is_not_learned_as_a_merge(client, conn):
    get_index().invalidate()
    one = lambda: make_item(conn, write_files=False)  # noqa: E731
    bela = make_person(conn, "Bela")
    for k in range(3):
        make_face(conn, one()["id"], bela, embedding=_axis(0), assigned_by="user" if k == 0 else "auto", write_crop=False)
    asha = make_person(conn, "Asha")
    x = one()
    moved = make_face(conn, x["id"], asha, embedding=_mix(0.1, 1), write_crop=False)   # 0.1 to Bela: below the floor
    loose = make_person(conn)
    loose_face = make_face(conn, one()["id"], loose, embedding=_mix(0.35, 2), write_crop=False)

    r = client.post("/api/items/people/move", json={"uids": [x["uid"]], "from_person_id": asha, "person_id": bela})
    assert r.status_code == 200, r.text
    assert r.json()["pulled"] == 0
    assert _face_owner(conn, moved) == (bela, "user")            # still the owner's example of Bela
    assert _face_owner(conn, loose_face)[0] == loose              # nothing pulled at the floor
    (ev,) = _merge_events(conn, bela)
    # the index reports a similarity under its minimum as 0: what matters is that it is under the floor
    assert ev["accepted"] is None and ev["below_floor"] is not None and ev["below_floor"] < cluster.PULL_FLOOR
