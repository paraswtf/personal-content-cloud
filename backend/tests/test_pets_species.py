"""Species corrections (section 22): the zero-shot swap, one box per animal, the owner's species on boxes
and on a pet, the assign popover's fix_species, carry-over of a user species, the faces inside animal
boxes, and the backfill that never touches a user row."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from app import storage
from app.analysis import animals as animals_mod
from app.analysis import cluster, pets, pipeline
from app.db import tx
from app.util import now_iso
from tests.helpers import make_face, make_item
from tests.test_learning_core import near, unit


def _animal(conn, item_id, species, emb, *, person_id=None, assigned_by="auto", box=(0.1, 0.1, 0.4, 0.4),
            species_source="auto", scores=None, frame_time_s=None, det_score=0.9):
    with tx(conn):
        return int(conn.execute(
            "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, species_source, "
            "species_scores, frame_time_s, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (item_id, person_id, assigned_by, *box, det_score, species, species_source,
             None if scores is None else json.dumps(scores), frame_time_s,
             None if emb is None else np.asarray(emb, dtype=np.float32).tobytes(), now_iso()),
        ).lastrowid)


def _row(conn, animal_id):
    return conn.execute("SELECT * FROM animals WHERE id = ?", (animal_id,)).fetchone()


def _events(conn):
    return [json.loads(r["features"]) | {"person_id": r["person_id"], "item_id": r["item_id"]}
            for r in conn.execute("SELECT features, person_id, item_id FROM learning_events WHERE kind = 'species' ORDER BY id")]


# ------------------------------------------------------------------ the zero-shot check
def test_species_check_swaps_only_when_sure():
    table = {s: 0.0 for s in animals_mod.SPECIES}
    table.update(cat=0.8, dog=0.1)
    assert animals_mod.species_check(None, "dog", scores=table) == ("cat", table, True)
    # the detector's class stands when the zero-shot agrees, is unsure, or the detector still scores
    assert animals_mod.species_check(None, "cat", scores=table)[0] == "cat"
    unsure = dict(table, cat=0.5, dog=0.2)
    assert animals_mod.species_check(None, "dog", scores=unsure) == ("dog", unsure, False)
    both = dict(table, cat=0.65, dog=0.3)
    assert animals_mod.species_check(None, "dog", scores=both)[2] is False
    # without a model nothing changes
    assert animals_mod.species_check(np.ones(768, dtype=np.float32), "dog") == ("dog", None, False)


def test_settle_species_lets_the_owners_lookalikes_vote(monkeypatch):
    from app.learning import images as learned

    monkeypatch.setattr(learned, "species_vote", lambda emb: "cat")
    table = {s: 0.0 for s in animals_mod.SPECIES}
    table.update(dog=0.9)
    assert animals_mod.settle_species(np.ones(768, dtype=np.float32), "dog", scores=table)[0] == "cat"
    monkeypatch.setattr(learned, "species_vote", lambda emb: None)
    assert animals_mod.settle_species(np.ones(768, dtype=np.float32), "dog", scores=table)[0] == "dog"


# ------------------------------------------------------------------ one box per animal
def _rec(x, y, w, h, score, species, t=None):
    return animals_mod.AnimalRecord(x=x, y=y, w=w, h=h, det_score=score, species=species, look=None,
                                    crop=Image.new("RGB", (8, 8)), frame_time_s=t)


def test_one_box_per_animal_keeps_the_better_scored_species():
    cat = _rec(0.40, 0.55, 0.45, 0.40, 0.81, "cat")
    dog = _rec(0.41, 0.54, 0.46, 0.42, 0.62, "dog")
    other_dog = _rec(0.05, 0.05, 0.20, 0.20, 0.70, "dog")
    kept = animals_mod.one_per_animal(sorted([cat, dog, other_dog], key=lambda a: -a.det_score))
    assert [(k.species, k.det_score) for k in kept] == [("cat", 0.81), ("dog", 0.70)]
    # two frames of a video are two animals
    far = _rec(0.41, 0.54, 0.46, 0.42, 0.62, "dog", t=3.0)
    assert len(animals_mod.one_per_animal([cat, far])) == 2


def test_stored_duplicates_collapse_and_keep_the_assignment(conn):
    item = make_item(conn, filename="two.jpg", write_files=False)
    with tx(conn):
        chutney = pets.new_pet(conn, "cat", "Chutney")
    a_cat = _animal(conn, item["id"], "cat", unit(1), box=(0.4, 0.55, 0.45, 0.40), det_score=0.81)
    a_dog = _animal(conn, item["id"], "dog", unit(2), box=(0.41, 0.54, 0.46, 0.42), det_score=0.62)
    # the dropped box's pet moves to the kept one when their species agree
    with tx(conn):
        conn.execute("UPDATE animals SET person_id = ?, assigned_by = 'user' WHERE id = ?", (chutney, a_dog))
        conn.execute("UPDATE persons SET species = 'cat' WHERE id = ?", (chutney,))
        dropped = animals_mod.dedup_item(conn, item["id"])
    assert dropped == [a_dog]
    kept = _row(conn, a_cat)
    assert kept["person_id"] == chutney and kept["assigned_by"] == "user"
    assert conn.execute("SELECT animal_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 1


# ------------------------------------------------------------------ faces inside animal boxes
def test_animal_faces_are_the_ones_inside_a_box_without_a_person():
    faces = [(0.42, 0.57, 0.15, 0.15), (0.10, 0.10, 0.10, 0.10), (0.50, 0.60, 0.10, 0.10)]
    animals = [(0.40, 0.55, 0.45, 0.40)]
    bodies = [(0.50, 0.62, 0.25, 0.30)]   # a person box over the third face only
    assert animals_mod.animal_faces(faces, animals, []) == [0, 2]
    assert animals_mod.animal_faces(faces, animals, bodies) == [0]
    assert animals_mod.animal_faces(faces, [], []) == []


def test_analyze_result_drops_animal_faces_and_repoints_bodies():
    from app.analysis.faces import FaceRecord

    def face(x, y):
        return FaceRecord(x=x, y=y, w=0.12, h=0.12, det_score=0.9, size_px=100, quality=0.5, embedding=np.zeros(512, np.float32),
                          crop=Image.new("RGB", (8, 8)), frame_time_s=None, spans=None, torso=None)

    r = pipeline.Result()
    r.faces = [face(0.42, 0.57), face(0.10, 0.10)]
    r.face_torsos = [None, np.ones(768, np.float32)]
    r.animals = [_rec(0.40, 0.55, 0.45, 0.40, 0.8, "cat")]
    from app.analysis.bodies import BodyRecord

    r.bodies = [BodyRecord(x=0.05, y=0.05, w=0.3, h=0.6, det_score=0.9, face_index=1, torso=None, crop=Image.new("RGB", (8, 8)))]
    assert pipeline._drop_animal_faces(r) == 1
    assert [(f.x, f.y) for f in r.faces] == [(0.10, 0.10)] and len(r.face_torsos) == 1
    assert r.bodies[0].face_index == 0


def test_stored_animal_faces_are_removed_by_the_sweep(conn):
    item = make_item(conn, filename="cat.jpg", write_files=False)
    with tx(conn):
        shreya = cluster.new_person(conn, "Shreya")
    on_cat = make_face(conn, item["id"], shreya, box=(0.42, 0.57, 0.15, 0.15), assigned_by="auto")
    real = make_face(conn, item["id"], shreya, box=(0.05, 0.05, 0.12, 0.12), assigned_by="user")
    _animal(conn, item["id"], "cat", unit(1), box=(0.40, 0.55, 0.45, 0.40))
    with tx(conn):
        removed = animals_mod.drop_animal_faces(conn, item["id"])
    assert removed == [on_cat]
    assert [int(r[0]) for r in conn.execute("SELECT id FROM faces WHERE item_id = ?", (item["id"],))] == [real]
    assert conn.execute("SELECT face_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 1
    assert conn.execute("SELECT 1 FROM persons WHERE id = ?", (shreya,)).fetchone()   # still has a real face


# ------------------------------------------------------------------ the owner's corrections
@pytest.fixture()
def cats(conn):
    """Chutney, a 'dog' pet whose crop X is really a cat; Y looks like X (another photo), Z does not."""
    i1 = make_item(conn, filename="x.jpg", write_files=False)
    i2 = make_item(conn, filename="y.jpg", write_files=False)
    i3 = make_item(conn, filename="z.jpg", write_files=False)
    with tx(conn):
        chutney = pets.new_pet(conn, "dog", "Chutney")
    base = unit(7, 768)
    x = _animal(conn, i1["id"], "dog", base, person_id=chutney, assigned_by="user")
    y = _animal(conn, i2["id"], "dog", near(base, 8, 0.3))        # cosine about 0.96
    z = _animal(conn, i3["id"], "dog", unit(9, 768))              # a stranger
    buf = io.BytesIO(); Image.new("RGB", (32, 32), (90, 90, 90)).save(buf, "WEBP")
    for aid in (x, y, z):
        storage.write_bytes_atomic(storage.animal_crop_path(aid), buf.getvalue())
    return dict(i1=i1, i2=i2, i3=i3, chutney=chutney, x=x, y=y, z=z)


def test_species_on_a_box_moves_it_out_and_lookalikes_follow(client, conn, cats):
    c = cats
    r = client.post("/api/animals/species", json={"animal_ids": [c["x"]], "species": "cat"})
    assert r.status_code == 200, r.text
    assert r.json()["updated"] == 1 and r.json()["followed"] == 1
    x, y, z = _row(conn, c["x"]), _row(conn, c["y"]), _row(conn, c["z"])
    assert (x["species"], x["species_source"]) == ("cat", "user")
    assert (y["species"], y["species_source"]) == ("cat", "auto")
    assert (z["species"], z["species_source"]) == ("dog", "auto")
    # X left the dog Chutney (a named pet stays, empty, for the owner to delete) and met Y in a new unnamed cat
    assert conn.execute("SELECT COUNT(*) FROM animals WHERE person_id = ?", (c["chutney"],)).fetchone()[0] == 0
    assert x["person_id"] is not None and x["person_id"] == y["person_id"]
    pet = conn.execute("SELECT kind, species, name FROM persons WHERE id = ?", (x["person_id"],)).fetchone()
    assert (pet["kind"], pet["species"], pet["name"]) == ("pet", "cat", None)
    ev = _events(conn)
    assert [(e["from"], e["to"], e["followed"]) for e in ev] == [("dog", "cat", False), ("dog", "cat", True)]
    assert ev[0]["person_id"] == c["chutney"] and ev[0]["item_id"] == c["i1"]["id"]
    det = client.get(f"/api/items/{c['i1']['uid']}").json()
    assert det["animals"][0]["species"] == "cat" and det["animals"][0]["species_source"] == "user"


def test_species_on_a_pet_keeps_its_name_and_merges_into_a_namesake(client, conn, cats):
    c = cats
    r = client.post(f"/api/people/{c['chutney']}/species", json={"species": "cat"})
    assert r.status_code == 200, r.text
    assert r.json()["person"]["id"] == c["chutney"] and r.json()["person"]["species"] == "cat"
    assert r.json()["person"]["name"] == "Chutney" and r.json()["followed"] == 1
    x = _row(conn, c["x"])
    assert (x["species"], x["species_source"], x["person_id"]) == ("cat", "user", c["chutney"])
    assert _row(conn, c["y"])["species"] == "cat"
    # a cat called Chutney already there takes the crops when the pet is corrected again
    with tx(conn):
        other = pets.new_pet(conn, "cat", "Chutney")
    r = client.post(f"/api/people/{c['chutney']}/species", json={"species": "cat"})
    assert r.status_code == 200 and r.json()["person"]["id"] == other
    assert _row(conn, c["x"])["person_id"] == other
    assert client.get(f"/api/people/{c['chutney']}").status_code == 404


def test_assign_with_fix_species_moves_a_cat_to_a_dog_pet_after_fixing_it(client, conn, cats):
    c = cats
    i4 = make_item(conn, filename="w.jpg", write_files=False)
    w = _animal(conn, i4["id"], "cat", near(unit(7, 768), 10, 0.3))
    r = client.post("/api/animals/assign", json={"animal_ids": [w], "person_id": c["chutney"]})
    assert r.status_code == 400 and "dog" in r.json()["detail"]
    r = client.post("/api/animals/assign", json={"animal_ids": [w], "person_id": c["chutney"], "fix_species": True})
    assert r.status_code == 200, r.text
    assert r.json()["person"]["species"] == "cat" and r.json()["person"]["name"] == "Chutney"
    assert _row(conn, w)["person_id"] == c["chutney"] and _row(conn, c["x"])["species"] == "cat"


def test_carry_over_keeps_a_user_species(conn):
    item = make_item(conn, filename="c.jpg", write_files=False)
    old = _animal(conn, item["id"], "cat", unit(2), box=(0.1, 0.1, 0.3, 0.3), species_source="user")
    old_rows = conn.execute("SELECT * FROM animals WHERE item_id = ? ORDER BY id", (item["id"],)).fetchall()
    new = _animal(conn, item["id"], "dog", None, box=(0.12, 0.1, 0.3, 0.3))
    with tx(conn):
        assert pets.carry_over(conn, old_rows, [new]) == 1
    r = _row(conn, new)
    assert (r["species"], r["species_source"]) == ("cat", "user")
    assert old


# ------------------------------------------------------------------ the backfill
class _Engine:
    """A SigLIP stand-in: every crop is a cat."""
    def zero_shot(self, embs, groups):
        out = np.zeros((embs.shape[0], len(groups)), dtype=np.float32)
        out[:, list(groups).index("cat")] = 0.9
        return out


def test_backfill_rechecks_automatic_crops_and_never_a_user_row(conn, monkeypatch):
    monkeypatch.setattr(animals_mod, "_engine", lambda: _Engine())
    item = make_item(conn, filename="b.jpg", write_files=False)
    with tx(conn):
        bruno = pets.new_pet(conn, "dog", "Bruno")
    auto = _animal(conn, item["id"], "dog", unit(3), person_id=bruno)
    item2 = make_item(conn, filename="u.jpg", write_files=False)
    user = _animal(conn, item2["id"], "dog", unit(4), species_source="user")
    assert animals_mod.housekeeping(conn, budget_s=5.0) == 1
    a, u = _row(conn, auto), _row(conn, user)
    assert a["species"] == "cat" and a["species_source"] == "auto" and json.loads(a["species_scores"])["cat"] == 0.9
    assert a["person_id"] is None                      # left the dog Bruno
    assert u["species"] == "dog" and u["species_scores"] is None
    assert conn.execute("SELECT COUNT(*) FROM animals WHERE person_id = ?", (bruno,)).fetchone()[0] == 0
    st = animals_mod.status(conn)
    assert st["unchecked"] == 0 and st["corrected"] == 1 and st["running"] is False
    assert animals_mod.housekeeping(conn, budget_s=5.0) == 0


# ------------------------------------------------------------------ more kinds (2026-11-09)
class _RabbitEngine:
    """A SigLIP stand-in: every crop is a rabbit."""
    def zero_shot(self, embs, groups):
        out = np.full((embs.shape[0], len(groups)), 0.005, dtype=np.float32)
        out[:, list(groups).index("rabbit")] = 0.86
        return out


def test_the_check_knows_kinds_the_detector_has_no_class_for():
    for kind in ("rabbit", "alpaca", "guinea_pig", "turtle", "monkey", "duck"):
        assert kind in animals_mod.SPECIES and kind in animals_mod.SPECIES_PROMPTS
    from app import schemas

    assert set(schemas.SPECIES_NAMES) == set(animals_mod.SPECIES_PROMPTS)
    scores = {sp: 0.005 for sp in animals_mod.SPECIES}
    scores.update(rabbit=0.82, cat=0.1)
    species, _, changed = animals_mod.species_check(None, "cat", scores=scores)   # the detector said cat
    assert (species, changed) == ("rabbit", True)


def test_new_kinds_check_the_systems_labels_again_but_never_a_named_pet_or_the_owners(conn, monkeypatch):
    item = make_item(conn, filename="r.jpg", write_files=False)
    with tx(conn):
        bunny = pets.new_pet(conn, "cat", "Bunny")       # named as a cat when there was no rabbit
        unnamed = pets.new_pet(conn, "cat", None)
    loose = _animal(conn, item["id"], "cat", unit(5))
    grouped = _animal(conn, item["id"], "cat", unit(6), person_id=unnamed, box=(0.5, 0.1, 0.3, 0.3))
    named = _animal(conn, item["id"], "cat", unit(7), person_id=bunny, box=(0.1, 0.5, 0.3, 0.3))
    owners = _animal(conn, item["id"], "cat", unit(8), species_source="user", box=(0.5, 0.5, 0.3, 0.3))
    with tx(conn):   # all were checked once against the ten old kinds
        conn.execute("UPDATE animals SET species_scores = '{\"cat\": 0.9}'")
        conn.execute("UPDATE kv SET value = '1' WHERE key = ?", (animals_mod.SPECIES_LABELS_KEY,))
    monkeypatch.setattr(animals_mod, "_engine", lambda: _RabbitEngine())
    assert animals_mod.housekeeping(conn, budget_s=5.0) == 2   # the loose crop and the unnamed group's
    assert _row(conn, loose)["species"] == "rabbit" and _row(conn, grouped)["species"] == "rabbit"
    assert (_row(conn, named)["species"], _row(conn, named)["person_id"]) == ("cat", bunny)   # Bunny keeps hers
    assert _row(conn, owners)["species"] == "cat"
    assert animals_mod.housekeeping(conn, budget_s=5.0) == 0   # once


def test_the_owner_can_pick_the_new_kinds(client, conn):
    item = make_item(conn, width=800, height=600)
    r = client.post(f"/api/items/{item['uid']}/animals", json={"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6, "species": "alpaca"})
    assert r.status_code == 201, r.text
    assert r.json()["species"] == "alpaca"
    r = client.post(f"/api/items/{item['uid']}/tags", json={"new_person_name": "Thumper", "kind": "pet", "species": "rabbit"})
    assert r.status_code in (200, 201), r.text
    assert client.post(f"/api/items/{item['uid']}/animals", json={"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6,
                                                                   "species": "unicorn"}).status_code == 422
