"""The pet naming learner (sections 18 and 22): a model trained on the owner's named pets names the automatic crops.

The fixtures make two pets whose look embeddings (SigLIP) all sit close together, as two cats of one home do, while
their identity vectors (DINOv2) differ; a look-alike the owner took away from the first pet is another animal."""
from __future__ import annotations

import numpy as np
import pytest

from app.analysis import cluster, pet_ident, pets
from app.db import kv_get, tx
from app.learning import pets as learned
from app.learning import store, train as learning_train
from app.util import now_iso
from tests.helpers import make_item
from tests.test_learning_core import near, unit

LOOK = unit(1)                      # every cat looks alike to the look embedding
ONE, TWO, LOOKALIKE = unit(10, 384), unit(20, 384), unit(40, 384)


def _crop(conn, species, look, ident, *, person_id=None, assigned_by="auto", item=None):
    item_id = item if item is not None else make_item(conn, write_files=False)["id"]
    with tx(conn):
        return int(conn.execute(
            "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, embedding, ident, "
            "created_at) VALUES (?, ?, ?, 0.1, 0.1, 0.4, 0.4, 0.9, ?, ?, ?, ?)",
            (item_id, person_id, assigned_by, species, np.asarray(look, np.float32).tobytes(),
             None if ident is None else np.asarray(ident, np.float32).tobytes(), now_iso()),
        ).lastrowid)


def _on(conn, animal_id):
    return conn.execute("SELECT person_id, assigned_by FROM animals WHERE id = ?", (animal_id,)).fetchone()


@pytest.fixture()
def home(conn):
    """Two named cats with ten crops each that the owner assigned, a look-alike the owner took away from the
    first one (six crops), and a few automatic crops far from both."""
    with tx(conn):
        one = pets.new_pet(conn, "cat", "Pet one")
        two = pets.new_pet(conn, "cat", "Pet two")
    seed = iter(range(1000, 9000))
    for k in range(10):
        _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(ONE, next(seed), 0.35), person_id=one, assigned_by="user")
        _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(TWO, next(seed), 0.35), person_id=two, assigned_by="user")
    # the owner took these away (the app keeps them as the owner's, on no pet)
    taken = [_crop(conn, "cat", near(LOOK, next(seed), 0.35), near(LOOKALIKE, next(seed), 0.35), assigned_by="user")
             for _ in range(6)]
    with tx(conn):
        cluster.record_box_rejections(conn, "animal", taken, one)
    for k in range(4):
        _crop(conn, "cat", near(LOOK, next(seed), 0.35), unit(300 + k, 384))
    return {"one": one, "two": two, "taken": taken, "seed": seed}


def test_the_model_learns_the_named_pets_and_checks_itself(conn, home):
    out = learned.train(conn)
    cat = out["species"]["cat"]
    assert cat["used"] and cat["pets"] == 2
    assert cat["check"]["examples"] == 20 and cat["check"]["right"] >= 0.9 and cat["check"]["wrong"] == 0
    assert cat["check"]["trusted_negatives"] == 6 and cat["check"]["negatives_named"] == 0
    assert set(learned.pets_of("cat")) == {home["one"], home["two"]}
    assert learned.pets_of("dog") == []
    st = learned.status()
    assert st["species"]["cat"]["pets"] == 2 and st["trained_at"]


def test_a_species_without_enough_confirmed_crops_has_no_model(conn):
    with tx(conn):
        pet = pets.new_pet(conn, "dog", "Pet three")
    for k in range(learned.MIN_CONFIRMED - 1):
        _crop(conn, "dog", near(LOOK, 50 + k, 0.3), near(ONE, 60 + k, 0.3), person_id=pet, assigned_by="user")
    assert learned.train(conn)["species"] == {}
    assert learned.current() is None


def test_rename_moves_lookalike_crops_to_the_named_pets_and_leaves_the_rest(conn, home):
    seed = home["seed"]
    one, two = home["one"], home["two"]
    with tx(conn):
        unnamed = pets.new_pet(conn, "cat")
    in_group = [_crop(conn, "cat", near(LOOK, next(seed), 0.35), near(ONE, next(seed), 0.35), person_id=unnamed)
                for _ in range(3)]
    loose_two = _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(TWO, next(seed), 0.35))
    stranger = _crop(conn, "cat", near(LOOK, next(seed), 0.35), unit(700, 384))
    no_vector = _crop(conn, "cat", near(LOOK, next(seed), 0.35), None)
    with tx(conn):
        kept_apart = pets.new_pet(conn, "cat")
    owner_said = _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(ONE, next(seed), 0.35),
                       person_id=kept_apart, assigned_by="user")
    learned.train(conn)
    with tx(conn):
        moved, items = pets.rename_crops(conn)
    assert moved == 4 and len(items) == 4
    assert all(tuple(_on(conn, a)) == (one, "auto") for a in in_group)
    assert tuple(_on(conn, loose_two)) == (two, "auto")
    assert _on(conn, stranger)["person_id"] is None and _on(conn, no_vector)["person_id"] is None
    assert tuple(_on(conn, owner_said)) == (kept_apart, "user")   # the owner's crops never move
    assert all(_on(conn, a)["person_id"] is None for a in home["taken"])
    # the unnamed pet the crops left is gone
    assert conn.execute("SELECT 1 FROM persons WHERE id = ?", (unnamed,)).fetchone() is None
    with tx(conn):
        assert pets.rename_crops(conn) == (0, set())               # nothing more to do


def test_a_crop_on_another_named_pet_moves_only_when_the_model_is_sure(conn, home, monkeypatch):
    seed = home["seed"]
    one, two = home["one"], home["two"]
    a = _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(ONE, next(seed), 0.35), person_id=two)
    b = _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(ONE, next(seed), 0.35), person_id=two)
    learned.train(conn)
    sure = {a: [0.97, 0.02, 0.01], b: [0.9, 0.05, 0.05]}

    def fake(species, x):
        return [one, two], np.array([sure[a], sure[b]])

    monkeypatch.setattr(learned, "probabilities", fake)
    rows = conn.execute("SELECT id, embedding, ident FROM animals WHERE id IN (?, ?) ORDER BY id", (a, b)).fetchall()
    names = pets._model_names(conn, "cat", rows, {a: two, b: two})
    assert names == {a: one}                     # 0.9 is enough for a loose crop, not to take one from another pet
    assert pets._model_names(conn, "cat", rows, {a: None, b: None}) == {a: one, b: one}


def test_a_crop_like_one_the_owner_took_away_never_gets_that_name(conn, home, monkeypatch):
    seed = home["seed"]
    one, two = home["one"], home["two"]
    taken_ident = np.frombuffer(conn.execute("SELECT ident FROM animals WHERE id = ?", (home["taken"][0],))
                                .fetchone()[0], dtype=np.float32)
    again = _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(taken_ident, next(seed), 0.05))
    learned.train(conn)
    monkeypatch.setattr(learned, "probabilities", lambda species, x: ([one, two], np.array([[0.99, 0.005, 0.005]])))
    rows = conn.execute("SELECT id, embedding, ident FROM animals WHERE id = ?", (again,)).fetchall()
    assert pets._model_names(conn, "cat", rows) == {}


def test_new_crops_are_named_by_the_model_and_lookalikes_do_not_reach_the_pet(conn, home):
    seed = home["seed"]
    one = home["one"]
    learned.train(conn)
    item = make_item(conn, write_files=False)["id"]
    mine = _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(ONE, next(seed), 0.35), item=item)
    # looks exactly like the pet's centroid to the look embedding, but is the animal the owner took away
    lookalike = _crop(conn, "cat", pets.pet_centroids(conn, "cat")[one], near(LOOKALIKE, next(seed), 0.2), item=item)
    pets.assign_new_animals(conn, item)
    assert tuple(_on(conn, mine)) == (one, "auto")
    assert _on(conn, lookalike)["person_id"] != one


def test_without_a_model_or_with_learning_off_the_old_matching_runs(conn, home):
    seed = home["seed"]
    one = home["one"]
    centroid = pets.pet_centroids(conn, "cat")[one]
    item = make_item(conn, write_files=False)["id"]
    close = _crop(conn, "cat", centroid, unit(900, 384), item=item)
    assert learned.current() is None
    pets.assign_new_animals(conn, item)
    assert _on(conn, close)["person_id"] in (one, home["two"])     # the look embedding decides, as before
    learned.train(conn)
    store.set_enabled(conn, False)
    try:
        assert learned.current() is None and learned.pets_of("cat") == []
        with tx(conn):
            assert pets.rename_crops(conn) == (0, set())
    finally:
        store.set_enabled(conn, True)


def test_housekeeping_renames_once_per_training(conn, home):
    seed = home["seed"]
    loose = _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(ONE, next(seed), 0.35))
    assert pet_ident.housekeeping(conn) == {"pets_renamed": 0}      # no model yet
    learned.train(conn)
    assert pet_ident.housekeeping(conn) == {"pets_renamed": 1}
    assert _on(conn, loose)["person_id"] == home["one"]
    assert kv_get(conn, pet_ident.KV_NAMED) == learned.trained_at()
    assert pet_ident.housekeeping(conn) == {"pets_renamed": 0}


def test_vectors_need_the_model_and_the_signature_follows_the_owner(conn, home):
    assert not pet_ident.available()                                # PCC_NO_MODELS in tests
    assert pet_ident.ensure(conn, home["taken"]) == 0
    before = learning_train.signature(conn)
    with tx(conn):
        conn.execute("UPDATE animals SET person_id = ?, assigned_by = 'user' WHERE id = ?",
                     (home["two"], home["taken"][0]))
    assert learning_train.signature(conn) != before
    assert "pets" in dict(learning_train.LEARNERS)


def test_followers_of_a_correction_are_not_the_owners_corrections(conn, home, monkeypatch):
    """leave_pets also remembers the automatic look-alikes that followed the owner's crops as not the pet; they may be
    the pet itself (the look embedding takes one white cat for another), so they neither veto it nor train as
    another animal."""
    seed = home["seed"]
    one, two = home["one"], home["two"]
    follower = _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(ONE, next(seed), 0.35), person_id=two)
    with tx(conn):
        cluster.record_box_rejections(conn, "animal", [follower], one, followed=True)
    assert pets.rejected_idents(conn, [one])[one].shape[0] == 6          # the owner's six, not the follower
    out = learned.train(conn)
    assert out["species"]["cat"]["taken_away_or_other_pets"] == 6
    monkeypatch.setattr(learned, "probabilities", lambda species, x: ([one, two], np.array([[0.99, 0.005, 0.005]])))
    like = _crop(conn, "cat", near(LOOK, next(seed), 0.35),
                 near(np.frombuffer(conn.execute("SELECT ident FROM animals WHERE id = ?", (follower,)).fetchone()[0],
                                    dtype=np.float32), next(seed), 0.05))
    rows = conn.execute("SELECT id, embedding, ident FROM animals WHERE id = ?", (like,)).fetchall()
    assert pets._model_names(conn, "cat", rows) == {like: one}


def test_a_crop_taken_from_one_pet_that_looks_like_the_other_is_not_another_animal(conn, home):
    seed = home["seed"]
    one = home["one"]
    mixed_up = [_crop(conn, "cat", near(LOOK, next(seed), 0.35), near(TWO, next(seed), 0.35), assigned_by="user")
                for _ in range(3)]
    with tx(conn):
        cluster.record_box_rejections(conn, "animal", mixed_up, one)
    ex = learned._species_examples(conn, "cat")
    assert not set(mixed_up) & set(ex["ids"])                  # left out of the training: they may be the other pet
    assert ex["negatives"]["taken_away_or_other_pets"] == 6


def test_a_deleted_pet_or_one_that_failed_its_own_check_gets_no_names(conn, home, monkeypatch):
    seed = home["seed"]
    one, two = home["one"], home["two"]
    learned.train(conn)
    crop = _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(ONE, next(seed), 0.35))
    rows = conn.execute("SELECT id, embedding, ident FROM animals WHERE id = ?", (crop,)).fetchall()
    monkeypatch.setattr(learned, "probabilities", lambda species, x: ([one, two], np.array([[0.99, 0.005, 0.005]])))
    assert pets._model_names(conn, "cat", rows) == {crop: one}
    monkeypatch.setattr(learned, "pets_of", lambda species: [two])          # one failed its own check
    assert pets._model_names(conn, "cat", rows) == {}
    monkeypatch.setattr(learned, "pets_of", lambda species: [one, two])
    with tx(conn):
        conn.execute("UPDATE animals SET person_id = NULL WHERE person_id = ?", (one,))
        conn.execute("DELETE FROM persons WHERE id = ?", (one,))
    assert pets._model_names(conn, "cat", rows) == {}                       # deleted since the training
    with tx(conn):
        assert pets.rename_crops(conn) == (0, set())


def test_the_check_reports_each_pet(conn, home):
    out = learned.train(conn)
    per = out["species"]["cat"]["check"]["per_pet"]
    assert set(per) == {str(home["one"]), str(home["two"])} and min(per.values()) >= 0.6
    assert learned.status()["species"]["cat"]["named_by_model"] == 2


def test_a_reanalysed_box_keeps_the_owners_not_this_pet(conn, home):
    one = home["one"]
    old = home["taken"][0]
    item = conn.execute("SELECT item_id FROM animals WHERE id = ?", (old,)).fetchone()[0]
    old_row = conn.execute("SELECT * FROM animals WHERE id = ?", (old,)).fetchone()
    seed = home["seed"]
    new = _crop(conn, "cat", near(LOOK, next(seed), 0.35), near(LOOKALIKE, next(seed), 0.35), item=item)
    with tx(conn):
        pets.carry_over(conn, [old_row], [new])
        conn.execute("DELETE FROM animals WHERE id = ?", (old,))
    assert _on(conn, new)["assigned_by"] == "auto"              # free to match again (as before)
    assert new in {int(r[0]) for r in conn.execute(
        "SELECT ref_id FROM guess_rejections WHERE kind = 'animal' AND person_id = ? AND followed = 0", (one,))}
    assert pets.rejected_idents(conn, [one])[one].shape[0] == 6   # still the owner's "not this pet"
    learned.train(conn)
    pets.assign_new_animals(conn, item)
    assert _on(conn, new)["person_id"] != one


def test_crops_the_owner_put_on_another_pet_stay_hard_negatives(conn, home):
    seed = home["seed"]
    one = home["one"]
    with tx(conn):
        other_pet = pets.new_pet(conn, "cat", "Pet four")
    moved = [_crop(conn, "cat", near(LOOK, next(seed), 0.35), near(TWO, next(seed), 0.35), person_id=other_pet,
                   assigned_by="user") for _ in range(2)]
    with tx(conn):
        cluster.record_box_rejections(conn, "animal", moved, one)
    ex = learned._species_examples(conn, "cat")
    idx = {a: k for k, a in enumerate(ex["ids"])}
    assert all(a in idx for a in moved)
    assert all(ex["w"][idx[a]] == learned.HARD_WEIGHT and ex["y"][idx[a]] == 2 for a in moved)
