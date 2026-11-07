"""The vault decision (section 9): the classifier needs agreement unless it is very sure, explicit
content counts alone, suggestive content counts alone when a person is in the photo, ordinary skin
blocks the weak paths, the sensitivity shifts every cut, and the learner fits weights over the signals."""
from __future__ import annotations

import json

from app.analysis import categories as C
from app.learning import images, store
from tests.helpers import make_item
from tests.test_learning_core import add_embedding, unit


def test_classifier_alone_needs_to_be_very_sure():
    assert C.vault_decision(0.95, 0.05, 0.05, 0.05)[0] is True
    assert C.vault_decision(0.75, 0.05, 0.05, 0.05)[0] is False      # the old rule flagged this
    assert C.vault_decision(0.65, 0.35, 0.05, 0.05)[1] == {"classifier_agreed": 0.65}
    assert C.vault_decision(0.65, 0.05, 0.45, 0.05)[0] is True


def test_explicit_and_suggestive_signals_on_their_own():
    assert C.vault_decision(0.1, 0.65, 0.1, 0.1)[1] == {"explicit": 0.65}
    assert C.vault_decision(0.1, 0.1, 0.75, 0.1, has_person=True)[1] == {"suggestive": 0.75}
    assert C.vault_decision(0.1, 0.1, 0.75, 0.1, has_person=False)[0] is False  # no person: a pattern, not a selfie
    assert C.vault_decision(0.1, 0.1, 0.55, 0.1, has_person=True)[0] is False
    # a skin-coloured pattern: SigLIP says explicit, nobody is in it and the classifier saw nothing
    assert C.vault_decision(0.01, 0.86, 0.79, 0.02, has_person=False)[0] is False
    assert C.vault_decision(0.01, 0.86, 0.79, 0.02, has_person=True)[0] is True
    assert C.vault_decision(0.15, 0.86, 0.79, 0.02, has_person=False)[1] == {"explicit": 0.86}


def test_ordinary_skin_blocks_the_weak_paths_but_not_explicit_content():
    assert C.vault_decision(0.95, 0.05, 0.05, 0.8)[0] is False       # beach photo the classifier dislikes
    assert C.vault_decision(0.2, 0.1, 0.8, 0.8, has_person=True)[0] is False
    assert C.vault_decision(0.2, 0.7, 0.1, 0.8)[0] is True           # explicit still wins
    assert C.vault_decision(0.7, 0.4, 0.1, 0.8)[0] is True           # agreement still wins


def test_sensitivity_shifts_the_cuts():
    assert C.vault_decision(0.2, 0.1, 0.65, 0.1, has_person=True, sensitivity="normal")[0] is False
    assert C.vault_decision(0.2, 0.1, 0.65, 0.1, has_person=True, sensitivity="high")[0] is True
    assert C.vault_decision(0.95, 0.1, 0.1, 0.1, sensitivity="low")[0] is False   # needs 0.99 on low
    assert C.vault_decision(0.1, 0.65, 0.1, 0.1, sensitivity="low")[0] is False
    assert C.vault_rules("normal") == C.VAULT_RULES


def test_decide_uses_the_new_rule_when_signals_exist_and_the_old_one_otherwise():
    d = C.decide("image", nsfw_score=0.75, explicit_p=0.1, suggestive_p=0.1, safe_p=0.1, has_person=True, scores={})
    assert d.category != "nsfw"
    d = C.decide("image", nsfw_score=0.2, explicit_p=0.1, suggestive_p=0.8, safe_p=0.1, has_person=True, scores={})
    assert d.category == "nsfw" and d.details == {"suggestive": 0.8}
    assert C.decide("image", nsfw_score=0.75, explicit_p=0.1, scores={}).category == "nsfw"  # no new signals: old rule
    assert C.decide("image", nsfw_score=0.2, explicit_p=0.1, suggestive_p=0.8, safe_p=0.1, has_person=True,
                    nsfw_override=False, scores={}).category == "personal"          # the owner's choice wins
    assert C.decide("image", nsfw_score=0.2, explicit_p=0.1, suggestive_p=0.8, safe_p=0.1, has_person=True,
                    nsfw_learned=False, scores={}).category == "personal"           # so does the learner


def test_vault_learner_fits_weights_over_the_signals(conn):
    # the owner put suggestive photos in the vault and took beach photos out: the classifier alone was wrong both ways
    rows = [(1, 0.5, 0.2, 0.9, 0.1), (1, 0.4, 0.3, 0.8, 0.1), (1, 0.6, 0.2, 0.85, 0.2), (1, 0.3, 0.1, 0.9, 0.1), (1, 0.5, 0.4, 0.7, 0.1),
            (0, 0.9, 0.1, 0.1, 0.8), (0, 0.85, 0.1, 0.2, 0.7), (0, 0.8, 0.2, 0.1, 0.9), (0, 0.9, 0.1, 0.1, 0.6), (0, 0.7, 0.1, 0.2, 0.8)]
    for k, (y, cls, ex, sug, safe) in enumerate(rows):
        it = make_item(conn, filename=f"v{k}.jpg", nsfw=y, nsfw_source="user", nsfw_score=cls, category="nsfw" if y else "personal",
                       category_scores=json.dumps({"explicit_p": ex, "suggestive_p": sug, "safe_p": safe}), write_files=False)
        add_embedding(conn, it["id"], unit(300 + k))
    st = images.train(conn)
    assert st["vault"]["weights"] is not None
    assert images.predict_nsfw(unit(999), {"cls": 0.4, "explicit": 0.2, "suggestive": 0.9, "safe": 0.1}) is True
    assert images.predict_nsfw(unit(998), {"cls": 0.9, "explicit": 0.1, "suggestive": 0.1, "safe": 0.8}) is False
    store.set_enabled(conn, False)
    try:
        assert images.predict_nsfw(unit(999), {"cls": 0.4, "explicit": 0.2, "suggestive": 0.9, "safe": 0.1}) is None
    finally:
        store.set_enabled(conn, True)


def test_sensitivity_shifts_the_learned_cuts_but_not_a_neighbours_label(conn):
    rows = [(1, 0.5, 0.2, 0.9, 0.1), (1, 0.4, 0.3, 0.8, 0.1), (1, 0.6, 0.2, 0.85, 0.2), (1, 0.3, 0.1, 0.9, 0.1), (1, 0.5, 0.4, 0.7, 0.1),
            (0, 0.9, 0.1, 0.1, 0.8), (0, 0.85, 0.1, 0.2, 0.7), (0, 0.8, 0.2, 0.1, 0.9), (0, 0.9, 0.1, 0.1, 0.6), (0, 0.7, 0.1, 0.2, 0.8)]
    for k, (y, cls, ex, sug, safe) in enumerate(rows):
        it = make_item(conn, filename=f"s{k}.jpg", nsfw=y, nsfw_source="user", nsfw_score=cls, category="nsfw" if y else "personal",
                       category_scores=json.dumps({"explicit_p": ex, "suggestive_p": sug, "safe_p": safe}), write_files=False)
        add_embedding(conn, it["id"], unit(400 + k))
    images.train(conn)
    model = store.load(images.MODEL)
    w, b = model["vault"]["weights"]["w"], model["vault"]["weights"]["b"]
    # a point near the learned boundary: find signals whose probability sits between the high and low cuts
    import numpy as np
    def p(sig):
        z = sum(w[k] * sig[k] for k in images.SIGNALS) + b
        return 1.0 / (1.0 + np.exp(-z))
    sig = {"cls": 0.5, "explicit": 0.2, "suggestive": 0.5, "safe": 0.4}
    for _ in range(200):
        pr = p(sig)
        if 0.36 < pr < 0.64:
            break
        sig["suggestive"] += 0.02 if pr <= 0.36 else -0.02
    assert images.predict_nsfw(unit(997), sig, sensitivity="high") is True
    assert images.predict_nsfw(unit(997), sig, sensitivity="low") is False
    # the owner's own label on a look-alike is not shifted
    assert images.predict_nsfw(unit(400), {"cls": 0.9, "explicit": 0.1, "suggestive": 0.1, "safe": 0.8}, sensitivity="low") is True
    assert images.predict_nsfw(unit(405), {"cls": 0.1, "explicit": 0.1, "suggestive": 0.9, "safe": 0.1}, sensitivity="high") is False


def test_resort_moves_auto_items_with_signals_and_counts_stale_ones(conn):
    from app.analysis import vault

    def item(name, **kw):
        base = dict(filename=name, kind="image", analysis_status="done", write_files=False)
        return make_item(conn, **{**base, **kw})

    sig = lambda e, s, f: json.dumps({"explicit_p": e, "suggestive_p": s, "safe_p": f, "places": 0.9})
    suggestive = item("selfie.jpg", nsfw=0, nsfw_source="auto", nsfw_score=0.2, category="personal", category_scores=sig(0.1, 0.65, 0.1), face_count=1)
    beach = item("beach.jpg", nsfw=1, nsfw_source="auto", nsfw_score=0.92, category="nsfw", category_scores=sig(0.1, 0.1, 0.8), face_count=0)
    chosen = item("chosen.jpg", nsfw=1, nsfw_source="user", nsfw_score=0.2, category="nsfw", category_scores=sig(0.1, 0.1, 0.1), face_count=1)
    old = item("old.jpg", nsfw=1, nsfw_source="auto", nsfw_score=0.75, category="nsfw", category_scores=json.dumps({"explicit_p": 0.1}), face_count=0)
    pattern = item("pattern.jpg", nsfw=0, nsfw_source="auto", nsfw_score=0.2, category="places", category_scores=sig(0.1, 0.8, 0.1), face_count=0)

    moved = vault.resort(conn, "normal")
    assert moved == {"checked": 3, "moved_in": 0, "moved_out": 1, "stale": 1}, vault.describe(moved)
    row = lambda it: conn.execute("SELECT nsfw, category FROM items WHERE id = ?", (it["id"],)).fetchone()
    assert tuple(row(beach)) == (0, "places")          # ordinary skin left the vault and got its scene back
    assert tuple(row(suggestive)) == (0, "personal")
    assert tuple(row(chosen)) == (1, "nsfw")           # the owner's choice is never touched
    assert tuple(row(old)) == (1, "nsfw")              # stale: no signals stored, needs a fresh analysis

    moved = vault.resort(conn, "high")
    assert (moved["moved_in"], moved["moved_out"]) == (1, 0)
    assert tuple(row(suggestive)) == (1, "nsfw")       # suggestive selfie with a person: in on high
    assert tuple(row(pattern)) == (0, "places")        # no person: stays out even on high

    moved = vault.resort(conn, "low")
    assert (moved["moved_in"], moved["moved_out"]) == (0, 1)
    assert tuple(row(suggestive)) == (0, "personal")


def test_patch_sensitivity_resorts_and_reports(client, conn):
    make_item(conn, filename="s.jpg", kind="image", analysis_status="done", write_files=False, nsfw=0, nsfw_source="auto",
              nsfw_score=0.2, category="personal", face_count=1,
              category_scores=json.dumps({"explicit_p": 0.1, "suggestive_p": 0.65, "safe_p": 0.1}))
    r = client.patch("/api/system/learning", json={"vault_sensitivity": "high"})
    assert r.status_code == 200, r.text
    assert r.json()["vault_sensitivity"] == "high" and r.json()["vault_moved"] == {"checked": 1, "moved_in": 1, "moved_out": 0, "stale": 0}
    assert client.get("/api/system/learning").json()["vault_moved"] is None
    assert client.patch("/api/system/learning", json={"vault_sensitivity": "medium"}).status_code == 422
    r = client.patch("/api/system/learning", json={"vault_sensitivity": "normal"})
    assert r.json()["vault_moved"]["moved_out"] == 1
    assert client.patch("/api/system/learning", json={"enabled": True}).json()["vault_moved"] is None
