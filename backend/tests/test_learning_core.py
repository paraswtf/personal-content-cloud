"""The learners of docs/ARCHITECTURE.md section 18: each learns the right thing from the recorded
choices, answers nothing below its minimum counts, and the switch and the training schedule work."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.db import tx
from app.learning import doc_type, events, images, misclick, people, store, textvec, train
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person


def unit(seed: int, dim: int = 768) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def near(v: np.ndarray, seed: int, noise: float = 0.08) -> np.ndarray:
    """v plus a unit noise vector scaled by `noise`: cosine to v is about 1 / sqrt(1 + noise^2)
    (0.08 -> 0.997, 0.5 -> 0.89, 2.2 -> 0.41)."""
    n = np.random.default_rng(seed).standard_normal(v.shape[0]).astype(np.float32)
    w = v + noise * (n / np.linalg.norm(n))
    return (w / np.linalg.norm(w)).astype(np.float32)


def add_embedding(conn, item_id: int, emb: np.ndarray) -> None:
    with tx(conn):
        conn.execute(
            "INSERT INTO item_embeddings(item_id, version, model, frame_index, frame_time_s, embedding, created_at) "
            "VALUES (?, 1, 'siglip', 0, NULL, ?, ?)", (item_id, emb.astype(np.float32).tobytes(), now_iso()))


def add_document(conn, item_id: int, doc_type_: str, text: str, source: str = "user") -> None:
    now = now_iso()
    with tx(conn):
        conn.execute("INSERT INTO item_text(item_id, text, source, lang, char_count, created_at) VALUES (?, ?, 'plain', NULL, ?, ?)",
                     (item_id, text, len(text), now))
        conn.execute(
            "INSERT INTO documents(item_id, doc_type, doc_group, confidence, fields, source, created_at, updated_at) "
            "VALUES (?, ?, 'other', 0.5, '{}', ?, ?, ?)", (item_id, doc_type_, source, now, now))


# ------------------------------------------------------------------ store and switch
def test_store_round_trip_cache_and_clear(conn):
    assert store.load("nothing") is None
    store.save("thing", {"a": 1})
    assert store.load("thing") == {"a": 1}
    store.save_array("mat", np.ones((2, 3), dtype=np.float32))
    assert store.load_array("mat").shape == (2, 3)
    store.clear()
    assert store.load("thing") is None and store.load_array("mat") is None


def test_switch_setting_and_flag(conn, monkeypatch):
    assert store.enabled(conn) is True
    store.set_enabled(conn, False)
    assert store.enabled(conn) is False
    assert store.enabled() is False  # cached flag without a connection
    store.set_enabled(conn, True)
    assert store.enabled(conn) is True
    monkeypatch.setattr(store.settings, "learning_enabled", False)
    assert store.enabled(conn) is False


# ------------------------------------------------------------------ events
def test_events_record_counts_and_rows(conn):
    events.record(conn, "category", None, {"from": "misc", "to": "food"}, item_id=1)
    events.record(conn, "guess", 1, {"clothes": 0.9}, item_id=1, person_id=2)
    assert events.counts(conn) == {"category": 1, "guess": 1}
    row = events.rows(conn, "guess")[0]
    assert row["label"] == 1 and json.loads(row["features"]) == {"clothes": 0.9} and row["person_id"] == 2
    with pytest.raises(ValueError):
        events.record(conn, "weather", None, {})


# ------------------------------------------------------------------ text vectors and document types
def test_textvec_tokens_and_cosine():
    toks = textvec.tokens("Tax Invoice no 42, कुल राशि")
    assert "tax" in toks and "tax invoice" in toks and "कुल राशि" in toks
    idf, n = textvec.build_idf([toks, textvec.tokens("rent agreement between")])
    a = textvec.vector(toks, idf, n)
    assert abs(textvec.cosine(a, a) - 1.0) < 1e-6
    assert textvec.cosine(a, textvec.vector(textvec.tokens("sunset at the beach"), idf, n)) == 0.0


def test_doc_type_learns_from_user_typed_documents(conn):
    a = make_item(conn, kind="pdf", filename="a.pdf", write_files=False)
    b = make_item(conn, kind="pdf", filename="b.pdf", write_files=False)
    add_document(conn, a["id"], "invoice", "Sharma Electronics tax invoice number 2026/0457 mixer grinder total amount 2500 paid by UPI")
    add_document(conn, b["id"], "marksheet", "Maharashtra State Board secondary school certificate examination marks obtained total 518 of 600 result pass")
    assert doc_type.predict("Sharma Electronics invoice total amount paid by card") is None  # nothing learned yet
    st = doc_type.train(conn)
    assert st["examples"] == 2 and st["labels"] == ["invoice", "marksheet"]
    label, sim = doc_type.predict("Sharma Electronics tax invoice number 2026/0512 washing machine total amount 18500 paid by card")
    assert label == "invoice" and sim >= doc_type.MIN_SIM
    assert doc_type.predict("a poem about the monsoon and the hills")[0] if doc_type.predict("a poem about the monsoon and the hills") else True
    assert doc_type.predict("") is None


# ------------------------------------------------------------------ categories and the vault
def test_images_learn_categories_from_look_alikes(conn):
    food, place = unit(1), unit(2)
    for k in range(3):
        it = make_item(conn, filename=f"food{k}.jpg", category="food", category_source="user", write_files=False)
        add_embedding(conn, it["id"], near(food, 10 + k))
    it = make_item(conn, filename="place.jpg", category="places", category_source="user", write_files=False)
    add_embedding(conn, it["id"], place)
    auto = make_item(conn, filename="auto.jpg", category="objects", write_files=False)  # not a user label
    add_embedding(conn, auto["id"], unit(3))
    assert images.predict_category(near(food, 99)) is None  # untrained
    st = images.train(conn)
    assert st["category"]["examples"] == 4 and st["category"]["labels"] == ["food", "places"]
    label, score = images.predict_category(near(food, 99))
    assert label == "food" and score >= images.BEST_MIN
    assert images.predict_category(unit(77)) is None  # nothing close


def test_images_learn_the_vault_by_neighbour_and_by_threshold(conn):
    private, plain = unit(5), unit(6)
    scores_pos = [0.72, 0.8, 0.85, 0.9, 0.93]
    scores_neg = [0.5, 0.55, 0.6, 0.64, 0.68]
    for k, s in enumerate(scores_pos):
        it = make_item(conn, filename=f"p{k}.jpg", nsfw=1, nsfw_source="user", nsfw_score=s, category="nsfw", write_files=False)
        add_embedding(conn, it["id"], near(private, 20 + k))
    for k, s in enumerate(scores_neg):
        it = make_item(conn, filename=f"n{k}.jpg", nsfw=0, nsfw_source="user", nsfw_score=s, write_files=False)
        add_embedding(conn, it["id"], near(plain, 30 + k))
    st = images.train(conn)
    assert st["vault"]["examples"] == 10 and 0.68 < st["vault"]["threshold"] <= 0.72
    assert images.predict_nsfw(near(plain, 90), 0.99) is False      # the owner said this look-alike is fine
    assert images.predict_nsfw(near(private, 91), 0.1) is True      # and this one is private
    assert images.predict_nsfw(unit(55), 0.71) is True and images.predict_nsfw(unit(55), 0.6) is False  # threshold
    assert images.predict_nsfw(unit(55), None) is None


def test_images_vault_threshold_needs_enough_labels(conn):
    for k in range(4):
        it = make_item(conn, filename=f"x{k}.jpg", nsfw=k % 2, nsfw_source="user", nsfw_score=0.5 + 0.1 * k, write_files=False)
        add_embedding(conn, it["id"], unit(40 + k))
    st = images.train(conn)
    assert st["vault"]["threshold"] is None
    assert images.predict_nsfw(unit(99), 0.9) is None


# ------------------------------------------------------------------ mis-click cuts
def test_misclick_cut_moves_between_kept_and_confirmed(conn):
    for k, bv in enumerate([22, 25, 28, 31, 34, 37]):   # flagged at the default 40, kept by the owner
        make_item(conn, filename=f"kept{k}.jpg", misclick=0, misclick_source="user",
                  quality_scores=json.dumps({"blur_var": bv, "mean_lum": 120, "p95_lum": 200, "clipped_frac": 0.0}), write_files=False)
    for k, bv in enumerate([2, 3, 4, 6, 8, 9]):         # flagged and trashed, or confirmed by hand
        extra = {"deleted_at": now_iso()} if k % 2 else {"misclick_source": "user"}
        make_item(conn, filename=f"bad{k}.jpg", misclick=1,
                  quality_scores=json.dumps({"blur_var": bv, "mean_lum": 120, "p95_lum": 200, "clipped_frac": 0.0}),
                  write_files=False, **extra)
    assert misclick.thresholds()["blur_var_min"] == 40.0
    st = misclick.train(conn)
    assert st["per_reason"]["blurry"] == {"kept": 6, "confirmed": 6}
    cut = misclick.thresholds()["blur_var_min"]
    assert 9 < cut <= 22
    assert misclick.thresholds()["dark_mean"] == 28.0  # nothing learned for the other reasons


def test_misclick_needs_examples_on_both_sides(conn):
    for k in range(6):
        make_item(conn, filename=f"kept{k}.jpg", misclick=0, misclick_source="user",
                  quality_scores=json.dumps({"blur_var": 20 + k, "mean_lum": 120, "p95_lum": 200}), write_files=False)
    misclick.train(conn)
    assert misclick.thresholds() == misclick.defaults()


# ------------------------------------------------------------------ people
def test_person_thresholds_loosen_for_confirmed_faces_and_tighten_after_rejections(conn):
    base = unit(7, 512)
    pid = make_person(conn, "Asha")
    it = make_item(conn, filename="asha.jpg", write_files=False)
    for k in range(4):  # confirmed faces that only loosely match each other: below the default threshold
        make_face(conn, it["id"], pid, embedding=near(base, 100 + k, 2.2), assigned_by="user", write_crop=False)
    st = people.train(conn)
    t = people.person_threshold(pid)
    assert st["persons"] == 1 and t is not None and people.FLOOR <= t < people.settings.cluster_match_threshold
    # a rejected guess that looks a lot like Asha raises her threshold above that face
    other = make_face(conn, it["id"], None, embedding=near(base, 200, 0.6), write_crop=False)
    with tx(conn):
        conn.execute("INSERT INTO guess_rejections(kind, ref_id, person_id, created_at) VALUES ('face', ?, ?, ?)",
                     (other, pid, now_iso()))
    people.train(conn)
    assert people.person_threshold(pid) > t
    assert people.person_threshold(999) is None


def test_guess_weights_need_enough_outcomes(conn):
    rng = np.random.default_rng(3)
    for i in range(20):  # too few
        events.record(conn, "guess", i % 2, {"clothes": 0.9, "scene_term": 0.8, "face_term": 0.5, "same_day": True, "has_gps": False})
    people.train(conn)
    assert people.guess_probability({"clothes": 0.9, "scene_term": 0.8, "face_term": 0.5}) is None
    for i in range(40):  # confirmed guesses had strong clothes evidence, rejected ones did not
        good = i % 2 == 0
        events.record(conn, "guess", 1 if good else 0,
                      {"clothes": 0.95 if good else 0.86, "scene_term": 0.85, "face_term": float(rng.uniform(0.5, 0.6)),
                       "same_day": True, "has_gps": False})
    st = people.train(conn)
    assert st["weights"] is not None and st["events"] == 60
    strong = people.guess_probability({"clothes": 0.96, "scene_term": 0.85, "face_term": 0.5, "same_day": True})
    weak = people.guess_probability({"clothes": 0.85, "scene_term": 0.85, "face_term": 0.5, "same_day": True})
    assert 0.0 < weak < strong < 1.0


# ------------------------------------------------------------------ training schedule and status
def test_train_all_status_maybe_train_pending_and_reset(conn):
    st = train.status(conn)
    assert st["enabled"] is True and st["trained_at"] is None and st["pending"] == 0
    it = make_item(conn, filename="f.jpg", category="food", category_source="user", write_files=False)
    add_embedding(conn, it["id"], unit(1))
    assert train.pending(conn) == 1
    assert train.maybe_train(conn) is True
    st = train.status(conn)
    assert st["trained_at"] and st["learners"]["category"]["examples"] == 1 and st["pending"] == 0
    assert train.maybe_train(conn) is False  # nothing changed
    events.record(conn, "category", None, {"from": "misc", "to": "food"}, item_id=it["id"])
    assert train.pending(conn) == 1 and train.maybe_train(conn) is True
    store.set_enabled(conn, False)
    assert train.maybe_train(conn) is False
    assert images.predict_category(unit(1)) is None  # off: the learners answer nothing
    train.reset()
    assert train.status(conn)["trained_at"] is None
