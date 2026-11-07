"""Document type learner (section 18): TF-IDF exemplars of the documents whose type the owner set or
confirmed (documents.source = 'user'). predict(text) answers with the nearest exemplar's type when the
cosine is at least MIN_SIM; extract.analyze_text lets it beat the rules at STRONG_SIM, or at MIN_SIM
when the rules found nothing."""
from __future__ import annotations

import sqlite3
from typing import Any

from app.learning import store, textvec
from app.util import now_iso

MODEL = "doc_type"
MIN_SIM = 0.35      # nearest exemplar similarity for a learned answer
STRONG_SIM = 0.5    # beats a rule-based type at or above this
MAX_TEXT = 60000    # characters of a document used for its vector


def examples(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """(doc_type, text) of live documents typed or confirmed by the owner."""
    rows = conn.execute(
        "SELECT d.doc_type, t.text FROM documents d JOIN items i ON i.id = d.item_id "
        "JOIN item_text t ON t.item_id = d.item_id WHERE d.source = 'user' AND i.deleted_at IS NULL "
        "AND t.text IS NOT NULL AND t.text != '' ORDER BY d.item_id"
    ).fetchall()
    return [(str(r["doc_type"]), str(r["text"])[:MAX_TEXT]) for r in rows]


def train(conn: sqlite3.Connection) -> dict[str, Any]:
    ex = examples(conn)
    toks = [textvec.tokens(text) for _label, text in ex]
    idf, n = textvec.build_idf(toks)
    exemplars = [{"label": label, "vec": textvec.vector(t, idf, n)} for (label, _text), t in zip(ex, toks)]
    exemplars = [e for e in exemplars if e["vec"]]
    labels = sorted({e["label"] for e in exemplars})
    model = {"idf": idf, "n_docs": n, "exemplars": exemplars, "examples": len(exemplars), "labels": labels,
             "trained_at": now_iso()}
    store.save(MODEL, model)
    return {"examples": len(exemplars), "labels": labels}


def predict(text: str) -> tuple[str, float] | None:
    """(doc_type, similarity) of the nearest exemplar, or None when nothing is learned or the text is
    too different."""
    if not text or not store.enabled():
        return None
    model = store.load(MODEL)
    if not model or not model.get("exemplars"):
        return None
    vec = textvec.vector(textvec.tokens(text[:MAX_TEXT]), model.get("idf", {}), int(model.get("n_docs", 0)))
    if not vec:
        return None
    best_label, best_sim = None, 0.0
    for ex in model["exemplars"]:
        sim = textvec.cosine(vec, ex["vec"])
        if sim > best_sim:
            best_label, best_sim = ex["label"], sim
    if best_label is None or best_sim < MIN_SIM:
        return None
    return best_label, round(best_sim, 4)


def status() -> dict[str, Any]:
    model = store.load(MODEL) or {}
    return {"examples": int(model.get("examples", 0)), "labels": list(model.get("labels", []))}
