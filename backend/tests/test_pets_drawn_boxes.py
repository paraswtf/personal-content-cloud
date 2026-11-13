"""Boxes the owner adds (section 22): the drawn-box route, the second look by the detector, and what a
re-analysis keeps. The models are off here (PCC_NO_MODELS=1); a fake SigLIP engine stands in where an
embedding matters."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app import storage
from app.analysis import animals as animals_mod
from app.analysis import pets
from app.db import tx
from app.util import now_iso
from tests.helpers import make_item

CAT_BOX = {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6}


class FakeEngine:
    """Answers every crop with one vector and calls it a cat."""

    def __init__(self, vec: np.ndarray):
        self.vec = vec.astype(np.float32)

    def embed_images(self, images):
        return np.stack([self.vec for _ in images])

    def zero_shot(self, embs, prompts):
        names = list(prompts)
        out = np.full((len(embs), len(names)), 0.01)
        out[:, names.index("cat")] = 0.9
        return out


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(768).astype(np.float32)
    return v / np.linalg.norm(v)


def _pet(conn, species: str, name: str, item_id: int, emb: np.ndarray) -> int:
    with tx(conn):
        pid = int(conn.execute(
            "INSERT INTO persons(name, name_key, kind, species, created_at, updated_at) VALUES (?, ?, 'pet', ?, ?, ?)",
            (name, name.casefold(), species, now_iso(), now_iso())).lastrowid)
        conn.execute(
            "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, embedding, created_at) "
            "VALUES (?, ?, 'user', 0.2, 0.2, 0.5, 0.5, 0.9, ?, ?, ?)", (item_id, pid, species, emb.tobytes(), now_iso()))
    return pid


def _animal(conn, animal_id: int):
    return conn.execute("SELECT * FROM animals WHERE id = ?", (animal_id,)).fetchone()


def test_species_names_match_the_analysis_module():
    from app import schemas

    assert schemas.SPECIES_NAMES == animals_mod.SPECIES and schemas.MIN_BOX_SHARE == animals_mod.MIN_BOX_SHARE


def test_drawn_box_is_stored_with_its_crop_and_the_owners_species(client, conn):
    item = make_item(conn, width=800, height=600)
    r = client.post(f"/api/items/{item['uid']}/animals", json=dict(CAT_BOX, species="cat"))
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["species"] == "cat" and out["species_source"] == "user" and out["source"] == "user"
    assert out["assigned_by"] == "auto" and out["det_score"] == 1.0 and out["person"] is None
    assert (out["x"], out["y"], out["w"], out["h"]) == (0.1, 0.1, 0.6, 0.6)
    row = _animal(conn, out["id"])
    assert row["source"] == "user" and row["embedding"] is None   # no model here: no look embedding
    assert storage.animal_crop_path(out["id"]).is_file()
    assert conn.execute("SELECT animal_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 1
    detail = client.get(f"/api/items/{item['uid']}").json()
    assert [a["id"] for a in detail["animals"]] == [out["id"]]


def test_drawn_box_needs_a_species_without_the_model(client, conn):
    item = make_item(conn)
    r = client.post(f"/api/items/{item['uid']}/animals", json=CAT_BOX)
    assert r.status_code == 400 and r.json()["detail"] == "species_needed"
    assert conn.execute("SELECT COUNT(*) FROM animals").fetchone()[0] == 0


def test_drawn_box_takes_the_zero_shot_species_and_joins_a_lookalike_pet(client, conn, monkeypatch):
    vec = _unit(5)
    monkeypatch.setattr(animals_mod, "look_engine", lambda: FakeEngine(vec))
    other = make_item(conn)
    chutney = _pet(conn, "cat", "Chutney", other["id"], vec)
    item = make_item(conn)
    r = client.post(f"/api/items/{item['uid']}/animals", json=CAT_BOX)
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["species"] == "cat" and out["species_source"] == "auto"
    assert out["person"] is not None and out["person"]["id"] == chutney
    row = _animal(conn, out["id"])
    assert row["person_id"] == chutney and row["assigned_by"] == "auto"
    assert json.loads(row["species_scores"])["cat"] == pytest.approx(0.9)
    assert np.allclose(np.frombuffer(row["embedding"], dtype=np.float32), vec)
    # the pet's page tells the drawn box from the found one, even though the pet was matched for it
    crops = {c["id"]: c["source"] for c in client.get(f"/api/people/{chutney}/animals").json()["items"]}
    assert crops[out["id"]] == "user" and sorted(crops.values()) == ["detector", "user"]


def test_drawn_box_outside_the_photo_or_too_small_is_refused(client, conn):
    item = make_item(conn)
    for box in (dict(CAT_BOX, x=0.6, w=0.6), dict(CAT_BOX, w=0.05, h=0.05), dict(CAT_BOX, species="dragon")):
        r = client.post(f"/api/items/{item['uid']}/animals", json=box)
        assert r.status_code == 422, (box, r.text)
    assert conn.execute("SELECT COUNT(*) FROM animals").fetchone()[0] == 0


def test_videos_and_missing_items_are_refused(client, conn):
    clip = make_item(conn, kind="video", ext=".mp4", mime="video/mp4", write_files=False, data=None)
    r = client.post(f"/api/items/{clip['uid']}/animals", json=dict(CAT_BOX, species="cat"))
    assert r.status_code == 400
    assert client.post("/api/items/nope12345678/animals", json=dict(CAT_BOX, species="cat")).status_code == 404


def test_second_look_needs_the_detector(client, conn):
    item = make_item(conn)
    r = client.post(f"/api/items/{item['uid']}/animals/detect", json={"threshold": 0.2})
    assert r.status_code == 409 and r.json()["detail"] == "models_unavailable"
    assert client.post(f"/api/items/{item['uid']}/animals/detect").status_code == 409


def test_second_look_stores_only_boxes_over_nothing_stored(conn, monkeypatch):
    """The detector is faked: it answers two cat boxes, one over the box the owner drew."""
    from app.ml import persons

    item = make_item(conn, width=800, height=600)
    drawn = animals_mod.add_box(conn, item, (0.1, 0.1, 0.6, 0.6), species="cat")
    monkeypatch.setattr(animals_mod, "available", lambda: True)
    monkeypatch.setattr(persons, "detect_classes",
                        lambda bgr, thresholds: {15: [(90.0, 70.0, 540.0, 410.0, 0.3), (600.0, 400.0, 790.0, 590.0, 0.25)]})
    monkeypatch.setattr(animals_mod, "look_engine", lambda: None)
    new = animals_mod.redetect(conn, item, 0.2)
    assert len(new) == 1
    row = _animal(conn, new[0])
    assert row["source"] == "detector" and row["species"] == "cat" and row["det_score"] == pytest.approx(0.25)
    assert row["x"] == pytest.approx(0.75) and storage.animal_crop_path(new[0]).is_file()
    assert conn.execute("SELECT animal_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 2
    assert _animal(conn, drawn)["source"] == "user"


def test_reanalysis_keeps_the_drawn_box_and_drops_a_detector_twin(conn):
    item = make_item(conn, width=800, height=600)
    drawn = animals_mod.add_box(conn, item, (0.1, 0.1, 0.6, 0.6), species="cat")
    rows = conn.execute("SELECT * FROM animals WHERE item_id = ?", (item["id"],)).fetchall()
    assert animals_mod.drawn_ids(rows) == [drawn]
    # carry_over leaves a drawn box alone: it is not in the rows a re-analysis replaces
    assert pets.carry_over(conn, [], [drawn]) == 0
    with tx(conn):
        twin = int(conn.execute(
            "INSERT INTO animals(item_id, assigned_by, x, y, w, h, det_score, species, created_at) "
            "VALUES (?, 'auto', 0.12, 0.12, 0.58, 0.58, 0.8, 'cat', ?)", (item["id"], now_iso())).lastrowid)
        apart = int(conn.execute(
            "INSERT INTO animals(item_id, assigned_by, x, y, w, h, det_score, species, created_at) "
            "VALUES (?, 'auto', 0.8, 0.8, 0.15, 0.15, 0.8, 'dog', ?)", (item["id"], now_iso())).lastrowid)
        dropped = animals_mod.drop_over_drawn(conn, int(item["id"]), [twin, apart])
    assert dropped == [twin]
    left = {int(r[0]) for r in conn.execute("SELECT id FROM animals WHERE item_id = ?", (item["id"],))}
    assert left == {drawn, apart}
    assert conn.execute("SELECT animal_count FROM items WHERE id = ?", (item["id"],)).fetchone()[0] == 2


def test_duplicate_collapse_keeps_the_drawn_box(conn):
    item = make_item(conn, width=800, height=600)
    drawn = animals_mod.add_box(conn, item, (0.1, 0.1, 0.6, 0.6), species="cat")
    with tx(conn):
        twin = int(conn.execute(
            "INSERT INTO animals(item_id, assigned_by, x, y, w, h, det_score, species, created_at) "
            "VALUES (?, 'auto', 0.11, 0.11, 0.6, 0.6, 0.99, 'dog', ?)", (item["id"], now_iso())).lastrowid)
        dropped = animals_mod.dedup_item(conn, int(item["id"]))
    assert dropped == [twin] and _animal(conn, drawn) is not None
