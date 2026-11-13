"""Not this pet (docs/ARCHITECTURE.md section 22): crops the owner takes away from a pet are remembered
(guess_rejections kind 'animal'), the pet's automatic look-alikes go with them, new crops like them are never
matched to that pet again, and the pet's learned threshold rises above them.

Embeddings are built from orthogonal axes, so every similarity is known: Chutney's crops have cosine 0.98 to
each other, so do the crops of the other cat, and a crop of one has 0.83 to a crop of the other. Chutney's
centroid is 0.84 from the other cat's crops: above the 0.82 a pet match needs, so without the owner's word
the other cat would keep landing on Chutney."""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from app.analysis import pets
from app.db import tx
from app.util import now_iso
from tests.helpers import make_item
from tests.test_db_upgrade import T0, old_db  # noqa: F401 - the fixture

DIM = 768
A = 0.99   # a crop's cosine to its cat
B = float(np.sqrt(1 - A * A))
CAT_SIM = 0.85   # the two cats


def _axis(k: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[k] = 1.0
    return v


CHUTNEY = _axis(0)
OTHER = (CAT_SIM * _axis(0) + np.sqrt(1 - CAT_SIM ** 2) * _axis(1)).astype(np.float32)


def chutney_crop(k: int) -> np.ndarray:
    return (A * CHUTNEY + B * _axis(10 + k)).astype(np.float32)


def other_crop(k: int) -> np.ndarray:
    return (A * OTHER + B * _axis(40 + k)).astype(np.float32)


def _animal(conn, item_id: int, emb: np.ndarray, person_id: int | None = None, assigned_by: str = "auto") -> int:
    with tx(conn):
        return int(conn.execute(
            "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, embedding, created_at) "
            "VALUES (?, ?, ?, 0.1, 0.1, 0.4, 0.4, 0.9, 'cat', ?, ?)",
            (item_id, person_id, assigned_by, emb.astype(np.float32).tobytes(), now_iso())).lastrowid)


def _owner(conn, animal_id: int):
    r = conn.execute("SELECT person_id, assigned_by FROM animals WHERE id = ?", (animal_id,)).fetchone()
    return (r["person_id"], r["assigned_by"])


def _rejected(conn, pet: int) -> set[int]:
    return {int(r[0]) for r in conn.execute(
        "SELECT ref_id FROM guess_rejections WHERE kind = 'animal' AND person_id = ?", (pet,))}


def _new_item(conn, name: str):
    return make_item(conn, filename=name, write_files=False)


@pytest.fixture()
def world(conn):
    """Chutney: three crops of her own and three of another cat, all matched automatically."""
    items = [_new_item(conn, f"n{k}.jpg") for k in range(6)]
    with tx(conn):
        chutney = pets.new_pet(conn, "cat", "Chutney")
    own = [_animal(conn, items[k]["id"], chutney_crop(k), chutney) for k in range(3)]
    other = [_animal(conn, items[3 + k]["id"], other_crop(k), chutney) for k in range(3)]
    return dict(items=items, chutney=chutney, own=own, other=other)


def test_a_bulk_move_off_a_pet_is_remembered_its_lookalikes_follow_and_its_threshold_rises(client, conn, world):
    w = world
    uids = [w["items"][3]["uid"], w["items"][4]["uid"]]
    r = client.post("/api/items/people/move", json={"uids": uids, "from_person_id": w["chutney"], "new_person_name": "Mau"})
    assert r.status_code == 200, r.text
    out = r.json()
    mau = out["person"]["id"]
    assert (out["person"]["kind"], out["person"]["species"]) == ("pet", "cat")
    assert (out["updated"], out["followed"], out["pulled"]) == (2, 1, 0)
    assert [_owner(conn, a) for a in w["other"]] == [(mau, "user"), (mau, "user"), (mau, "auto")]  # the third followed
    assert all(_owner(conn, a) == (w["chutney"], "auto") for a in w["own"])
    assert _rejected(conn, w["chutney"]) == set(w["other"])
    assert conn.execute("SELECT COUNT(*) FROM learning_events WHERE kind = 'merge'").fetchone()[0] == 0  # pets: no merge

    from app.learning import people as learned
    centroid = sum(chutney_crop(k) for k in range(3))
    centroid = centroid / np.linalg.norm(centroid)
    closest = max(float(other_crop(k) @ centroid) for k in range(3))
    assert closest == pytest.approx(0.839, abs=1e-3) and closest > 0.82
    assert learned.person_thresholds(conn)[w["chutney"]] == pytest.approx(closest + learned.TIGHTEN, abs=1e-3)
    learned.train(conn)   # what matching reads
    assert pets.match_threshold(w["chutney"]) == pytest.approx(closest + learned.TIGHTEN, abs=1e-3)

    item = _new_item(conn, "later.jpg")
    later = _animal(conn, item["id"], other_crop(3))
    with tx(conn):
        pets.assign_new_animals(conn, item["id"])
    assert _owner(conn, later) == (mau, "auto")


def test_a_removal_lets_the_lookalikes_go_and_keeps_new_ones_off_the_pet(client, conn, world):
    w = world
    uids = [w["items"][3]["uid"], w["items"][4]["uid"]]
    r = client.post("/api/items/people/remove", json={"uids": uids, "person_id": w["chutney"]})
    assert r.status_code == 200, r.text
    assert (r.json()["updated"], r.json()["followed"]) == (2, 1)
    # the third crop of the other cat followed and is loose: nothing to pair with yet (the owner's two are locked)
    assert [_owner(conn, a) for a in w["other"]] == [(None, "user"), (None, "user"), (None, "auto")]
    assert _rejected(conn, w["chutney"]) == set(w["other"])

    a, b = _new_item(conn, "a.jpg"), _new_item(conn, "b.jpg")
    other_new = _animal(conn, a["id"], other_crop(3))   # 0.84 to Chutney: a match, but it looks like a crop taken away
    own_new = _animal(conn, b["id"], chutney_crop(3))    # 0.83 to those crops: under 0.86, still Chutney
    with tx(conn):
        pets.assign_new_animals(conn, a["id"])
        pets.assign_new_animals(conn, b["id"])
    pet = _owner(conn, other_new)[0]
    assert pet is not None and pet != w["chutney"]
    assert _owner(conn, w["other"][2]) == (pet, "auto")    # it paired with the loose look-alike: a new pet
    assert _owner(conn, own_new) == (w["chutney"], "auto")


def test_animals_assign_remembers_not_this_pet_and_forgets_it_when_put_back(client, conn, world):
    w = world
    first, *rest = w["other"]
    r = client.post("/api/animals/assign", json={"animal_ids": [first]})
    assert r.status_code == 200, r.text
    assert r.json()["followed"] == 2
    assert _owner(conn, first) == (None, "user")
    (pet, by), = {_owner(conn, a) for a in rest}           # the look-alikes left together: a new unnamed pet
    assert pet not in (None, w["chutney"]) and by == "auto"
    assert _rejected(conn, w["chutney"]) == set(w["other"])
    r = client.post("/api/animals/assign", json={"animal_ids": [first], "person_id": w["chutney"]})
    assert r.status_code == 200, r.text
    assert _owner(conn, first) == (w["chutney"], "user") and _rejected(conn, w["chutney"]) == set(rest)


def test_a_lookalike_the_owner_refused_for_the_target_stays(client, conn, world):
    w = world
    with tx(conn):
        mau = pets.new_pet(conn, "cat", "Mau")
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('animal', ?, ?, ?)",
                     (w["other"][2], mau, now_iso()))
    r = client.post("/api/animals/assign", json={"animal_ids": w["other"][:2], "person_id": mau})
    assert r.status_code == 200 and r.json()["followed"] == 0, r.text
    assert _owner(conn, w["other"][2]) == (w["chutney"], "auto")


def test_reanalysis_carries_a_rejection_to_the_new_box_and_deleting_a_box_forgets_it(client, conn, world):
    w = world
    assert client.post("/api/animals/assign", json={"animal_ids": [w["other"][0]]}).status_code == 200
    pet = _owner(conn, w["other"][1])[0]
    item_id = w["items"][3]["id"]
    old = conn.execute("SELECT * FROM animals WHERE item_id = ? ORDER BY id", (item_id,)).fetchall()
    new = _animal(conn, item_id, other_crop(0))            # the same box, detected again
    with tx(conn):
        pets.carry_over(conn, old, [new])
        conn.execute("DELETE FROM animals WHERE id = ?", (w["other"][0],))   # as the pipeline does
        pets.assign_new_animals(conn, item_id)
    assert _rejected(conn, w["chutney"]) == {new, *w["other"][1:]}
    assert _owner(conn, new) == (pet, "auto")               # free to match, never Chutney: it joined its look-alikes
    assert client.delete(f"/api/animals/{w['other'][1]}").status_code == 200
    assert _rejected(conn, w["chutney"]) == {new, w["other"][2]}


def test_an_old_database_learns_the_animal_kind(old_db):  # noqa: F811
    from app import db

    conn = sqlite3.connect(old_db)
    conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', 11, 2, ?)", (T0,))
    conn.commit()
    conn.close()
    db.init_db()
    conn = db.connect()
    try:
        assert [tuple(r) for r in conn.execute("SELECT kind, ref_id, person_id FROM guess_rejections")] == [("face", 11, 2)]
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('animal', 12, 2, ?)", (T0,))
        assert conn.execute("SELECT COUNT(*) FROM guess_rejections").fetchone()[0] == 2
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name LIKE '%_old'").fetchone()
    finally:
        conn.close()
