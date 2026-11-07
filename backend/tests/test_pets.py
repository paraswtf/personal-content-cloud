"""Pets (section 22): animal classes from one detector run, animal records, pet identities from the
look embeddings, carry-over on re-analysis, the people routes for pets, the animal routes, the item
detail, the search index, the vault, and the learned thresholds."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from app import storage
from app.analysis import animals as animals_mod
from app.analysis import bodies, cluster, pets, pipeline
from app.db import tx
from app.ml import persons
from app.util import now_iso
from tests.helpers import jpeg_bytes, make_face, make_item, noise_image
from tests.test_learning_core import near, unit


# ------------------------------------------------------------------ detector
def _raw(boxes: list[tuple[int, int, int, int, int, float]]) -> np.ndarray:
    """A fake YOLOX output with one confident box per entry (stride, gx, gy, size_px, class, prob)."""
    raw = np.zeros((8400, 85), dtype=np.float32)
    offsets = {8: 0, 16: 6400, 32: 8000}
    for stride, gx, gy, size, cls, prob in boxes:
        n = 640 // stride
        i = offsets[stride] + gy * n + gx
        raw[i, 0:2] = 0.5
        raw[i, 2:4] = np.log(size / stride)
        raw[i, 4] = 0.95
        raw[i, 5 + cls] = prob
    return raw


def test_decode_classes_answers_for_several_classes_from_one_output():
    raw = _raw([(8, 10, 10, 64, persons.PERSON_CLASS, 0.9), (16, 20, 20, 128, 16, 0.8), (32, 5, 5, 200, 15, 0.3)])
    out = persons.decode_classes(raw, 1.0, 640, 640, {persons.PERSON_CLASS: 0.5, 16: 0.5, 15: 0.5})
    (px0, py0, px1, py1, ps), = out[persons.PERSON_CLASS]
    assert (round(px0), round(py0), round(px1), round(py1)) == (52, 52, 116, 116) and ps == pytest.approx(0.855, abs=1e-3)
    (dx0, dy0, dx1, dy1, ds), = out[16]
    assert (round(dx0), round(dy0), round(dx1), round(dy1)) == (264, 264, 392, 392)
    assert out[15] == []                                     # the cat is under its threshold
    assert persons.decode(raw, 1.0, 640, 640, 0.5) == out[persons.PERSON_CLASS]
    assert persons.ANIMAL_CLASSES[16] == "dog" and persons.ANIMAL_CLASSES[15] == "cat"


def test_animal_records_from_boxes_keep_species_and_crops():
    img = noise_image(640, 480, seed=3)
    bgr = np.asarray(img)[:, :, ::-1].copy()
    found = {16: [(52.0, 52.0, 116.0, 116.0, 0.9), (300.0, 300.0, 310.0, 310.0, 0.8)], 15: [(200.0, 100.0, 400.0, 400.0, 0.7)], 2: [(0.0, 0.0, 100.0, 100.0, 0.9)]}
    recs = animals_mod.from_boxes(img, bgr, found, frame_time_s=None)
    assert [(r.species, round(r.det_score, 2)) for r in recs] == [("dog", 0.9), ("cat", 0.7)]   # tiny box and the car dropped
    dog = recs[0]
    assert (dog.x, dog.y) == pytest.approx((52 / 640, 52 / 480)) and dog.w == pytest.approx(64 / 640)
    assert dog.look is not None and dog.look.shape[0] > 64 and dog.look.shape[1] > 64   # padded by 10 percent
    assert max(dog.crop.size) <= bodies.CROP_SIDE and dog.crop.mode == "RGB"
    assert len(animals_mod.encode_crop(dog.crop)) > 100


# ------------------------------------------------------------------ identities
def _animal(conn, item_id, species, emb, *, person_id=None, assigned_by="auto", box=(0.1, 0.1, 0.4, 0.4)):
    with tx(conn):
        return int(conn.execute(
            "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0.9, ?, ?, ?)",
            (item_id, person_id, assigned_by, *box, species, None if emb is None else np.asarray(emb, dtype=np.float32).tobytes(), now_iso()),
        ).lastrowid)


def _pid(conn, animal_id):
    return conn.execute("SELECT person_id, assigned_by FROM animals WHERE id = ?", (animal_id,)).fetchone()


def test_new_boxes_join_a_known_pet_or_pair_into_a_new_one(conn):
    items = [make_item(conn, filename=f"p{k}.jpg", write_files=False) for k in range(6)]
    with tx(conn):
        bruno = pets.new_pet(conn, "dog", "Bruno")
    _animal(conn, items[0]["id"], "dog", near(unit(1), 101, 0.3), person_id=bruno, assigned_by="user")
    _animal(conn, items[1]["id"], "dog", near(unit(1), 102, 0.3), person_id=bruno)
    # item 2: a box that looks like Bruno, a dog that looks like nothing yet, and a cat
    like_bruno = _animal(conn, items[2]["id"], "dog", near(unit(1), 103, 0.3))
    stranger = _animal(conn, items[2]["id"], "dog", unit(50))
    cat = _animal(conn, items[2]["id"], "cat", unit(50))
    assert pets.assign_new_animals(conn, items[2]["id"]) == 1
    assert tuple(_pid(conn, like_bruno)) == (bruno, "auto")
    assert _pid(conn, stranger)["person_id"] is None and _pid(conn, cat)["person_id"] is None
    # item 3: the same strange dog again (another item): the two become a new unnamed dog
    again = _animal(conn, items[3]["id"], "dog", near(unit(50), 104, 0.2))
    assert pets.assign_new_animals(conn, items[3]["id"]) == 2
    new_pet = _pid(conn, again)["person_id"]
    assert new_pet is not None and new_pet != bruno and _pid(conn, stranger)["person_id"] == new_pet
    row = conn.execute("SELECT kind, species, name FROM persons WHERE id = ?", (new_pet,)).fetchone()
    assert tuple(row) == ("pet", "dog", None)
    assert _pid(conn, cat)["person_id"] is None                     # a cat never joins a dog
    # two look-alike dogs in one photo are two dogs: they never pair with each other
    a = _animal(conn, items[4]["id"], "dog", unit(70))
    b = _animal(conn, items[4]["id"], "dog", near(unit(70), 105, 0.1))
    assert pets.assign_new_animals(conn, items[4]["id"]) == 0
    assert _pid(conn, a)["person_id"] is None and _pid(conn, b)["person_id"] is None
    # the pet's centroid is every assigned crop, so a third crop joins through the pair
    third = _animal(conn, items[5]["id"], "dog", near(unit(50), 106, 0.25))
    assert pets.assign_new_animals(conn, items[5]["id"]) == 1 and _pid(conn, third)["person_id"] == new_pet
    assert len(pets.pet_centroids(conn, "dog")) == 2


def test_carry_over_keeps_user_assignments_by_overlap_and_species(conn):
    item = make_item(conn, filename="c.jpg", write_files=False)
    with tx(conn):
        pet = pets.new_pet(conn, "cat", "Tom")
    old_user = _animal(conn, item["id"], "cat", unit(2), person_id=pet, assigned_by="user", box=(0.1, 0.1, 0.3, 0.3))
    old_auto = _animal(conn, item["id"], "cat", unit(2), person_id=pet, box=(0.6, 0.6, 0.3, 0.3))
    old_rows = conn.execute("SELECT * FROM animals WHERE item_id = ? ORDER BY id", (item["id"],)).fetchall()
    new_same = _animal(conn, item["id"], "cat", None, box=(0.12, 0.1, 0.3, 0.3))
    new_dog = _animal(conn, item["id"], "dog", None, box=(0.6, 0.6, 0.3, 0.3))
    with tx(conn):
        assert pets.carry_over(conn, old_rows, [new_same, new_dog]) == 1
    assert tuple(_pid(conn, new_same)) == (pet, "user")
    assert _pid(conn, new_dog)["person_id"] is None
    assert old_auto  # the automatic one is not carried: the new analysis decides it again


def test_pipeline_stores_animal_boxes_and_makes_a_pet_photo(conn, monkeypatch):
    monkeypatch.setattr(pipeline, "_model", lambda name: None)
    monkeypatch.setattr(bodies, "available", lambda: False)
    monkeypatch.setattr(animals_mod, "available", lambda: True)
    monkeypatch.setattr(persons, "detect_classes", lambda bgr, asks: {16: [(100.0, 100.0, 500.0, 420.0, 0.88)]})
    img = noise_image(800, 600, seed=5)
    item = make_item(conn, data=jpeg_bytes(img), width=800, height=600, analysis_status="pending")
    pipeline.analyze_item(conn, item["id"], 1)
    row = conn.execute("SELECT category, animal_count, quality_scores FROM items WHERE id = ?", (item["id"],)).fetchone()
    assert row["animal_count"] == 1 and row["category"] == "pets"
    assert "animals" not in json.loads(row["quality_scores"])["skipped"]
    a = conn.execute("SELECT * FROM animals WHERE item_id = ?", (item["id"],)).fetchone()
    assert a["species"] == "dog" and a["embedding"] is None and a["person_id"] is None
    assert (round(a["x"], 3), round(a["w"], 3)) == (0.125, 0.5)
    assert storage.animal_crop_path(int(a["id"])).is_file()
    # a second run replaces the boxes and drops the old crop
    with tx(conn):
        conn.execute("UPDATE items SET analysis_status = 'pending' WHERE id = ?", (item["id"],))
    monkeypatch.setattr(persons, "detect_classes", lambda bgr, asks: {15: [(10.0, 10.0, 60.0, 60.0, 0.7)]})
    pipeline.analyze_item(conn, item["id"], 1)
    rows = conn.execute("SELECT species FROM animals WHERE item_id = ?", (item["id"],)).fetchall()
    assert [r["species"] for r in rows] == ["cat"] and not storage.animal_crop_path(int(a["id"])).is_file()
    assert conn.execute("SELECT category FROM items WHERE id = ?", (item["id"],)).fetchone()[0] != "pets"  # a tiny cat


# ------------------------------------------------------------------ routes
@pytest.fixture()
def zoo(conn):
    """Two dog boxes (one Bruno's) and a cat box on photos, a person Meera with a face, and crops on disk."""
    i1 = make_item(conn, filename="dogs.jpg", write_files=False)
    i2 = make_item(conn, filename="cat.jpg", write_files=False)
    i3 = make_item(conn, filename="meera.jpg", write_files=False)
    with tx(conn):
        bruno = pets.new_pet(conn, "dog", "Bruno")
        meera = cluster.new_person(conn, "Meera")
    d1 = _animal(conn, i1["id"], "dog", unit(1), person_id=bruno, assigned_by="user")
    d2 = _animal(conn, i1["id"], "dog", unit(3), box=(0.5, 0.5, 0.3, 0.3))
    c1 = _animal(conn, i2["id"], "cat", unit(5))
    make_face(conn, i3["id"], meera, assigned_by="user")
    buf = io.BytesIO(); Image.new("RGB", (64, 64), (120, 90, 60)).save(buf, "WEBP")
    for aid in (d1, d2, c1):
        storage.write_bytes_atomic(storage.animal_crop_path(aid), buf.getvalue())
    with tx(conn):
        for it in (i1, i2):
            conn.execute("UPDATE items SET animal_count = (SELECT COUNT(*) FROM animals WHERE item_id = ?) WHERE id = ?", (it["id"], it["id"]))
    return dict(i1=i1, i2=i2, i3=i3, bruno=bruno, meera=meera, d1=d1, d2=d2, c1=c1)


def test_people_routes_know_pets(client, zoo):
    pets_page = client.get("/api/people", params={"kind": "pet"}).json()
    assert [(p["name"], p["kind"], p["species"], p["animal_count"], p["face_count"], p["item_count"]) for p in pets_page["items"]] == \
        [("Bruno", "pet", "dog", 1, 0, 1)]
    bruno = pets_page["items"][0]
    assert bruno["cover_url"] == f"/files/animal/{zoo['d1']}" and bruno["cover_face_id"] is None
    assert [p["name"] for p in client.get("/api/people", params={"kind": "person"}).json()["items"]] == ["Meera"]
    assert {p["name"] for p in client.get("/api/people").json()["items"]} == {"Bruno", "Meera"}
    assert client.get(f"/api/people/{zoo['bruno']}").json()["kind"] == "pet"
    crops = client.get(f"/api/people/{zoo['bruno']}/animals").json()
    assert crops["total"] == 1 and crops["items"][0]["species"] == "dog" and crops["items"][0]["url"] == f"/files/animal/{zoo['d1']}"
    assert client.get(f"/api/people/{zoo['meera']}/animals").json()["total"] == 0
    assert [i["filename"] for i in client.get(f"/api/people/{zoo['bruno']}/items").json()["items"]] == ["dogs.jpg"]
    assert [i["filename"] for i in client.get("/api/items", params={"person": zoo["bruno"]}).json()["items"]] == ["dogs.jpg"]
    assert client.get(f"/files/animal/{zoo['d1']}").status_code == 200


def test_item_detail_lists_animals_and_pets(client, zoo):
    det = client.get(f"/api/items/{zoo['i1']['uid']}").json()
    assert [(a["species"], a["assigned_by"], a["person"] and a["person"]["name"]) for a in det["animals"]] == \
        [("dog", "user", "Bruno"), ("dog", "auto", None)]
    assert det["animals"][0]["person"]["kind"] == "pet" and det["animals"][0]["url"].startswith("/files/animal/")
    assert [(p["name"], p["kind"]) for p in det["people"]] == [("Bruno", "pet")]


def test_assign_animals_creates_pets_of_the_boxes_species(client, conn, zoo):
    r = client.post("/api/animals/assign", json={"animal_ids": [zoo["c1"]], "new_person_name": "Tom"})
    assert r.status_code == 200, r.text
    tom = r.json()["person"]
    assert (tom["kind"], tom["species"], tom["animal_count"]) == ("pet", "cat", 1)
    assert tuple(_pid(conn, zoo["c1"])) == (tom["id"], "user")
    # a dog box cannot go to a cat, nor to a person; a person's face cannot go to a pet
    assert client.post("/api/animals/assign", json={"animal_ids": [zoo["d2"]], "person_id": tom["id"]}).status_code == 400
    assert client.post("/api/animals/assign", json={"animal_ids": [zoo["d2"]], "person_id": zoo["meera"]}).status_code == 400
    assert client.post("/api/animals/assign", json={"animal_ids": [zoo["d2"], zoo["c1"]], "person_id": zoo["bruno"]}).status_code == 400
    face = conn.execute("SELECT id FROM faces WHERE item_id = ?", (zoo["i3"]["id"],)).fetchone()[0]
    assert client.post("/api/faces/assign", json={"face_ids": [face], "person_id": zoo["bruno"]}).status_code == 400
    # the same name again (any case) reuses the pet; a person named Tom is a different Tom
    i4 = make_item(conn, filename="cat2.jpg", write_files=False)
    c2 = _animal(conn, i4["id"], "cat", unit(6))
    assert client.post("/api/animals/assign", json={"animal_ids": [c2], "new_person_name": "tom"}).json()["person"]["id"] == tom["id"]
    person_tom = client.post("/api/people", json={"name": "Tom"}).json()
    assert person_tom["id"] != tom["id"] and person_tom["kind"] == "person"
    # the search index knows the pet's name
    hits = client.get("/api/search", params={"q": "Tom", "mode": "text"}).json()["items"]
    assert {h["filename"] for h in hits} >= {"cat.jpg", "cat2.jpg"}
    # unassign locks the box as the owner's choice
    assert client.post("/api/animals/assign", json={"animal_ids": [c2], "person_id": None}).json()["person"] is None
    assert tuple(_pid(conn, c2)) == (None, "user")
    # the second dog joins Bruno; deleting a box updates the count and the crop goes
    assert client.post("/api/animals/assign", json={"animal_ids": [zoo["d2"]], "person_id": zoo["bruno"]}).json()["updated"] == 1
    assert client.get(f"/api/people/{zoo['bruno']}").json()["animal_count"] == 2
    assert client.delete(f"/api/animals/{zoo['d2']}").json() == {"ok": True}
    assert conn.execute("SELECT animal_count FROM items WHERE id = ?", (zoo["i1"]["id"],)).fetchone()[0] == 1
    assert not storage.animal_crop_path(zoo["d2"]).exists()
    assert client.delete(f"/api/animals/{zoo['d2']}").status_code == 404


def test_pets_rename_merge_and_delete(client, conn, zoo):
    i4 = make_item(conn, filename="dog2.jpg", write_files=False)
    d3 = _animal(conn, i4["id"], "dog", unit(9))
    rex = client.post("/api/animals/assign", json={"animal_ids": [d3], "new_person_name": "Rex"}).json()["person"]
    # cover picking is ignored for pets, renaming works, and a rename into an existing pet's name merges
    assert client.patch(f"/api/people/{rex['id']}", json={"cover_face_id": 1}).status_code == 200
    assert client.patch(f"/api/people/{rex['id']}", json={"name": "Rexy"}).json()["name"] == "Rexy"
    merged = client.patch(f"/api/people/{rex['id']}", json={"name": "bruno"}).json()
    assert merged["id"] == zoo["bruno"] and merged["animal_count"] == 2
    assert client.get(f"/api/people/{rex['id']}").status_code == 404
    # a pet and a person never merge, nor two species
    i5 = make_item(conn, filename="cat3.jpg", write_files=False)
    c3 = _animal(conn, i5["id"], "cat", unit(11))
    tom = client.post("/api/animals/assign", json={"animal_ids": [c3], "new_person_name": "Tom"}).json()["person"]
    assert client.post("/api/people/merge", json={"target_id": zoo["meera"], "source_ids": [zoo["bruno"]]}).status_code == 400
    assert client.post("/api/people/merge", json={"target_id": zoo["bruno"], "source_ids": [tom["id"]]}).status_code == 400
    # a person named like a pet is not merged into it by a rename
    meera = client.patch(f"/api/people/{zoo['meera']}", json={"name": "Tom"}).json()
    assert meera["id"] == zoo["meera"] and meera["kind"] == "person"
    # deleting a pet frees its boxes and keeps the owner's say
    assert client.delete(f"/api/people/{zoo['bruno']}").json() == {"ok": True}
    assert tuple(_pid(conn, zoo["d1"])) == (None, "user")
    assert client.get("/api/people", params={"kind": "pet"}).json()["total"] == 1


def test_vault_hides_pets_crops_and_blocks_changes(vault_client, conn, zoo):
    with tx(conn):
        conn.execute("UPDATE items SET nsfw = 1 WHERE id = ?", (zoo["i1"]["id"],))
    assert vault_client.post("/api/auth/vault/lock").status_code in (200, 204)
    assert vault_client.get(f"/files/animal/{zoo['d1']}").status_code == 403
    r = vault_client.post("/api/animals/assign", json={"animal_ids": [zoo["d2"]], "person_id": zoo["bruno"]})
    assert r.status_code == 403 and r.json()["detail"] == "vault_locked"
    assert vault_client.delete(f"/api/animals/{zoo['d2']}").status_code == 403
    assert vault_client.get(f"/api/people/{zoo['bruno']}").json()["animal_count"] == 0   # nothing visible
    assert vault_client.get(f"/api/people/{zoo['bruno']}/animals").json()["total"] == 0
    assert vault_client.post("/api/auth/vault/unlock", json={"password": "correct horse battery staple"}).status_code == 200
    assert vault_client.get(f"/files/animal/{zoo['d1']}").status_code == 200


def test_pets_learn_their_own_threshold(conn):
    from app.learning import people as learned
    with tx(conn):
        pet = pets.new_pet(conn, "dog", "Bruno")
    items = [make_item(conn, filename=f"t{k}.jpg", write_files=False) for k in range(4)]
    for k, it in enumerate(items):
        _animal(conn, it["id"], "dog", near(unit(1), 200 + k, 0.9 + 0.1 * k), person_id=pet, assigned_by="user")
    t = learned.person_thresholds(conn)
    assert pet in t and t[pet] < 0.82
