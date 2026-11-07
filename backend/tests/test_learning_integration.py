"""The pipeline follows the learners when they answer and the rules when they do not, and the API
and the guess code record the owner's choices as learning events (section 18)."""
from __future__ import annotations

import json

import numpy as np

from app.analysis import categories, cluster, guess, quality
from app.db import tx
from app.documents import extract
from app.documents.fields import PageText
from app.learning import events
from app.util import now_iso
from tests.helpers import make_face, make_item, make_person

H = {"X-Requested-With": "pcc"}


def test_decide_follows_learned_category_and_vault(monkeypatch):
    d = categories.decide("image", learned=("food", 0.9), scores={"places": 0.95})
    assert d.category == "food" and d.reason == "learned" and d.details == {"score": 0.9}
    assert categories.decide("image", scores={"places": 0.95}).category == "places"      # nothing learned
    assert categories.decide("image", learned=("nsfw", 0.9), scores={"places": 0.95}).category == "places"
    assert categories.decide("image", nsfw_score=0.2, nsfw_learned=True).category == "nsfw"
    assert categories.decide("image", nsfw_score=0.9, nsfw_learned=False, scores={}).category != "nsfw"
    assert categories.decide("image", nsfw_score=0.9, nsfw_learned=False, nsfw_override=True).nsfw is True
    assert categories.decide("pdf", learned=("food", 0.9)).category == "documents"


def test_quality_uses_learned_cuts(monkeypatch):
    from app.learning import misclick

    scores = {"blur_var": 20.0, "mean_lum": 120.0, "p95_lum": 200.0, "clipped_frac": 0.0}
    assert quality.decide("image", "personal", scores, good_face=False)[0] is True
    monkeypatch.setattr(misclick, "thresholds", lambda: {**misclick.defaults(), "blur_var_min": 10.0})
    assert quality.decide("image", "personal", scores, good_face=False)[0] is False


def test_guess_score_uses_learned_probability(monkeypatch):
    from app.learning import people

    fixed = guess.guess_score(0.9, 0.8, None)
    assert abs(fixed - (0.5 * 0.9 + 0.3 * 0.8 + 0.2 * 0.5)) < 1e-9
    seen = {}
    monkeypatch.setattr(people, "guess_probability", lambda f: seen.setdefault("f", f) and 0.77)
    assert guess.guess_score(0.9, 0.8, None, same_day=True, has_gps=True) == 0.77
    assert seen["f"]["same_day"] is True and seen["f"]["has_gps"] is True and seen["f"]["face_term"] == 0.5


def test_cluster_threshold_takes_the_learned_value(monkeypatch):
    from app.learning import people

    assert cluster.learned_threshold(1, 0.45) == 0.45
    monkeypatch.setattr(people, "person_threshold", lambda pid: 0.6 if pid == 1 else 0.3)
    assert cluster.learned_threshold(1, 0.45) == 0.6 and cluster.learned_threshold(2, 0.45) == 0.3


def test_analyze_text_follows_the_document_type_learner(monkeypatch):
    from app.learning import doc_type

    text = "some words that match no rule at all " * 5
    assert extract.analyze_text(text, [PageText(1, text, "layer")]) is not None  # 'other' with text
    assert extract.analyze_text(text, [PageText(1, text, "layer")])[0].doc_type == "other"
    monkeypatch.setattr(doc_type, "predict", lambda t: ("invoice", 0.4))
    parsed, score = extract.analyze_text(text, [PageText(1, text, "layer")])
    assert parsed.doc_type == "invoice" and score >= 0.4
    # a weak hint does not beat a rule-based type, a strong one does
    receipt = "RECEIPT received with thanks from Rahul the sum of Rs 500 towards rent receipt no 12 date 01/02/2026"
    monkeypatch.setattr(doc_type, "predict", lambda t: ("invoice", 0.4))
    assert extract.analyze_text(receipt, [PageText(1, receipt, "layer")])[0].doc_type == "receipt"
    monkeypatch.setattr(doc_type, "predict", lambda t: ("invoice", 0.6))
    assert extract.analyze_text(receipt, [PageText(1, receipt, "layer")])[0].doc_type == "invoice"


def test_item_edits_record_learning_events(client):
    from app import db

    with db.connect() as conn:
        it = make_item(conn, filename="a.jpg", category="objects", category_scores=json.dumps({"objects": 0.6}),
                       quality_scores=json.dumps({"blur_var": 12.0}), misclick=1,
                       misclick_reasons=json.dumps({"blurry": {"blur_var": 12.0}}))
    uid = it["uid"]
    assert client.patch(f"/api/items/{uid}", json={"category": "food"}, headers=H).status_code == 200
    assert client.patch(f"/api/items/{uid}", json={"misclick": False}, headers=H).status_code == 200
    assert client.patch(f"/api/items/{uid}", json={"nsfw": True}, headers=H).status_code == 200
    with db.connect() as conn:
        rows = {r["kind"]: json.loads(r["features"]) for r in conn.execute("SELECT kind, features FROM learning_events")}
    assert rows["category"] == {"from": "objects", "to": "food", "scores": {"objects": 0.6}}
    assert rows["misclick"]["from"] == 1 and rows["misclick"]["to"] == 0 and rows["misclick"]["scores"] == {"blur_var": 12.0}
    assert rows["vault"]["from"] == 0 and rows["vault"]["to"] == 1


def test_document_type_change_records_an_event(client):
    from app import db

    with db.connect() as conn:
        it = make_item(conn, kind="pdf", filename="d.pdf", text_status="done", write_files=False)
        now = now_iso()
        with db.tx(conn):
            conn.execute("INSERT INTO item_text(item_id, text, source, lang, char_count, created_at) VALUES (?, ?, 'pdf', NULL, 40, ?)",
                         (it["id"], "RENTAL AGREEMENT made on 12 March 2026 for a flat", now))
            conn.execute("INSERT INTO documents(item_id, doc_type, doc_group, confidence, fields, source, created_at, updated_at) "
                         "VALUES (?, 'other', 'other', 0.5, '{}', 'auto', ?, ?)", (it["id"], now, now))
    r = client.patch(f"/api/documents/{it['uid']}", json={"doc_type": "invoice"}, headers=H)
    assert r.status_code == 200 and r.json()["doc_type"] == "invoice"
    with db.connect() as conn:
        rows = [json.loads(r["features"]) for r in conn.execute("SELECT features FROM learning_events WHERE kind = 'doc_type'")]
    assert rows == [{"from": "other", "to": "invoice"}]


def test_guess_confirm_and_reject_record_outcomes(conn):
    pid = make_person(conn, "Rahul")
    a = make_item(conn, filename="a.jpg", write_files=False)
    b = make_item(conn, filename="b.jpg", write_files=False)
    ev = json.dumps({"clothes": 0.91, "scene_term": 0.83, "scene": 0.83, "face": None, "same_day": True, "gps_m": None, "score": 0.8})
    fa = make_face(conn, a["id"], None, assigned_by="guess", guess_person_id=pid, guess_score=0.8, guess_evidence=ev, write_crop=False)
    fb = make_face(conn, b["id"], None, assigned_by="guess", guess_person_id=pid, guess_score=0.8, guess_evidence=ev, write_crop=False)
    assert guess.confirm(conn, [fa], []) == 1
    assert guess.reject(conn, [fb], []) == 1
    rows = events.rows(conn, "guess")
    assert [r["label"] for r in rows] == [1, 0] and all(r["person_id"] == pid for r in rows)
    f = json.loads(rows[0]["features"])
    assert f["clothes"] == 0.91 and f["scene_term"] == 0.83 and f["face_term"] == 0.5 and f["same_day"] is True and f["has_gps"] is False
