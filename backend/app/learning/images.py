"""Category and vault learner (section 18): nearest neighbours over the SigLIP whole-image embeddings
(frame 0) of the items whose category or vault flag the owner set. The category of a new item is the
weighted vote of its close neighbours; the vault answer comes from a very close neighbour, else from a
threshold on the classifier score calibrated on the owner's labels."""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from app.config import settings
from app.learning import store
from app.util import now_iso

MODEL = "images"
CAT_ARRAY = "images_category"
NSFW_ARRAY = "images_vault"
FORWARD_ARRAY = "images_forward"  # section 29: the items the owner kept or set apart as forwards
NEIGHBOUR_MIN = 0.80   # a neighbour counts from this cosine
BEST_MIN = 0.85        # the best neighbour must be at least this close
VOTE_MIN = 0.6         # the winner's share of the neighbours' weight
TOP_K = 5
NSFW_MIN_LABELS = 8
NSFW_MIN_SIDE = 3
SIGNALS = ("cls", "explicit", "suggestive", "safe")
WEIGHT_CUT = {"low": 0.65, "normal": 0.5, "high": 0.35}   # probability from the learned weights
THRESHOLD_SHIFT = {"low": 0.10, "normal": 0.0, "high": -0.10}   # on the calibrated classifier cut


def _embeddings(conn: sqlite3.Connection, item_ids: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for i in range(0, len(item_ids), 500):
        chunk = item_ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT item_id, embedding FROM item_embeddings WHERE frame_index = 0 AND item_id IN ({ph})", chunk
        ):
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 0:
                out[int(r["item_id"])] = (v / n).astype(np.float32)
    return out


def _matrix(embs: dict[int, np.ndarray], ids: list[int]) -> tuple[np.ndarray, list[int]]:
    kept = [i for i in ids if i in embs]
    dims = {embs[i].shape[0] for i in kept}
    if len(dims) > 1:  # one model's rows only (the most common size)
        dim = max(dims, key=lambda d: sum(1 for i in kept if embs[i].shape[0] == d))
        kept = [i for i in kept if embs[i].shape[0] == dim]
    if not kept:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.stack([embs[i] for i in kept]).astype(np.float32), kept


def calibrate_nsfw(pairs: list[tuple[float, int]]) -> float | None:
    """The cut on the classifier score that best separates the owner's vault labels (balanced
    accuracy), searched in [0.5, 0.95]; None without enough labels on both sides."""
    if len(pairs) < NSFW_MIN_LABELS:
        return None
    pos = np.array([s for s, y in pairs if y], dtype=np.float64)
    neg = np.array([s for s, y in pairs if not y], dtype=np.float64)
    if len(pos) < NSFW_MIN_SIDE or len(neg) < NSFW_MIN_SIDE:
        return None
    best_t, best_acc = None, -1.0
    default = float(settings.nsfw_threshold)
    for t in np.arange(0.50, 0.951, 0.01):
        acc = 0.5 * (float((pos >= t).mean()) + float((neg < t).mean()))
        if acc > best_acc + 1e-9 or (abs(acc - best_acc) <= 1e-9 and best_t is not None and abs(t - default) < abs(best_t - default)):
            best_t, best_acc = float(t), acc
    return None if best_t is None else round(best_t, 2)


def train(conn: sqlite3.Connection) -> dict[str, Any]:
    cat_rows = conn.execute(
        "SELECT id, category FROM items WHERE category_source = 'user' AND deleted_at IS NULL "
        "AND kind IN ('image', 'video') ORDER BY id"
    ).fetchall()
    nsfw_rows = conn.execute(
        "SELECT id, nsfw, nsfw_score, category_scores FROM items WHERE nsfw_source = 'user' AND deleted_at IS NULL "
        "AND kind IN ('image', 'video') ORDER BY id"
    ).fetchall()
    fwd_rows = conn.execute(
        "SELECT id, forward FROM items WHERE forward_source = 'user' AND deleted_at IS NULL "
        "AND kind IN ('image', 'video') ORDER BY id"
    ).fetchall()
    embs = _embeddings(conn, sorted({int(r["id"]) for r in [*cat_rows, *nsfw_rows, *fwd_rows]}))
    fwd_x, fwd_ids = _matrix(embs, [int(r["id"]) for r in fwd_rows])
    fwd_label = {int(r["id"]): int(r["forward"]) for r in fwd_rows}
    cat_x, cat_ids = _matrix(embs, [int(r["id"]) for r in cat_rows])
    cat_label = {int(r["id"]): str(r["category"]) for r in cat_rows}
    nsfw_x, nsfw_ids = _matrix(embs, [int(r["id"]) for r in nsfw_rows])
    nsfw_label = {int(r["id"]): int(r["nsfw"]) for r in nsfw_rows}
    pairs = [(float(r["nsfw_score"]), int(r["nsfw"])) for r in nsfw_rows if r["nsfw_score"] is not None]
    threshold = calibrate_nsfw(pairs)
    weights = fit_vault_weights(nsfw_rows)
    if cat_ids:
        store.save_array(CAT_ARRAY, cat_x)
    else:
        store.remove(CAT_ARRAY)
    if nsfw_ids:
        store.save_array(NSFW_ARRAY, nsfw_x)
    else:
        store.remove(NSFW_ARRAY)
    if fwd_ids:
        store.save_array(FORWARD_ARRAY, fwd_x)
    else:
        store.remove(FORWARD_ARRAY)
    labels = [cat_label[i] for i in cat_ids]
    model = {
        "category": {"ids": cat_ids, "labels": labels, "examples": len(cat_ids), "names": sorted(set(labels))},
        "vault": {"ids": nsfw_ids, "labels": [nsfw_label[i] for i in nsfw_ids], "examples": len(nsfw_rows),
                  "threshold": threshold, "scored": len(pairs), "weights": weights},
        "forward": {"ids": fwd_ids, "labels": [fwd_label[i] for i in fwd_ids], "examples": len(fwd_rows)},
        "trained_at": now_iso(),
    }
    store.save(MODEL, model)
    return status()


def signal_vector(signals: dict[str, Any] | float | None) -> list[float] | None:
    """[cls, explicit, suggestive, safe] from the pipeline's signals (a bare float is the classifier)."""
    if signals is None:
        return None
    if isinstance(signals, (int, float)):
        return [float(signals), 0.0, 0.0, 0.0]
    try:
        return [float(signals.get(k) or 0.0) for k in SIGNALS]
    except (TypeError, ValueError, AttributeError):
        return None


def fit_vault_weights(rows: list[sqlite3.Row]) -> dict[str, Any] | None:
    """A logistic model over the vault signals from the owner's labels (needs NSFW_MIN_LABELS with
    both sides), or None."""
    from app.learning.people import fit_logistic
    from app.util import json_loads

    xs, ys = [], []
    for r in rows:
        if r["nsfw_score"] is None:
            continue
        scores = json_loads(r["category_scores"], {}) or {}
        xs.append([float(r["nsfw_score"]), float(scores.get("explicit_p") or 0.0),
                   float(scores.get("suggestive_p") or 0.0), float(scores.get("safe_p") or 0.0)])
        ys.append(int(r["nsfw"]))
    if len(ys) < NSFW_MIN_LABELS or sum(ys) < NSFW_MIN_SIDE or len(ys) - sum(ys) < NSFW_MIN_SIDE:
        return None
    w, b = fit_logistic(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), steps=800, lr=0.8, l2=0.005)
    return {"w": dict(zip(SIGNALS, w)), "b": b, "examples": len(ys)}


def _neighbours(arr_name: str, emb: np.ndarray) -> np.ndarray | None:
    x = store.load_array(arr_name)
    if x is None or x.size == 0 or x.shape[1] != emb.shape[0]:
        return None
    v = np.asarray(emb, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 0:
        return None
    return x @ (v / n)


def predict_category(emb: np.ndarray | None) -> tuple[str, float] | None:
    """(category, score) voted by the owner's labelled look-alikes, or None."""
    if emb is None or not store.enabled():
        return None
    model = store.load(MODEL)
    if not model or not model["category"]["ids"]:
        return None
    sims = _neighbours(CAT_ARRAY, emb)
    if sims is None:
        return None
    labels = model["category"]["labels"]
    order = np.argsort(-sims)[:TOP_K]
    close = [(labels[i], float(sims[i])) for i in order if sims[i] >= NEIGHBOUR_MIN]
    if not close or close[0][1] < BEST_MIN:
        return None
    weight: dict[str, float] = {}
    for label, s in close:
        weight[label] = weight.get(label, 0.0) + s
    winner = max(weight, key=lambda k: weight[k])
    if weight[winner] / sum(weight.values()) < VOTE_MIN:
        return None
    return winner, round(close[0][1], 4)


def forward_vote(emb: np.ndarray | None) -> bool | None:
    """Section 29: True or False when the owner's kept or set-apart look-alikes decide (a neighbour at
    cosine >= BEST_MIN, the winner holding VOTE_MIN of the close neighbours' weight), else None."""
    if emb is None or not store.enabled():
        return None
    model = store.load(MODEL)
    fwd = (model or {}).get("forward") or {}
    if not fwd.get("ids"):
        return None
    sims = _neighbours(FORWARD_ARRAY, emb)
    if sims is None:
        return None
    labels = fwd["labels"]
    order = np.argsort(-sims)[:TOP_K]
    close = [(int(labels[i]), float(sims[i])) for i in order if sims[i] >= NEIGHBOUR_MIN]
    if not close or close[0][1] < BEST_MIN:
        return None
    weight = {0: 0.0, 1: 0.0}
    for label, s in close:
        weight[label] += s
    winner = 1 if weight[1] >= weight[0] else 0
    if weight[winner] / (weight[0] + weight[1]) < VOTE_MIN:
        return None
    return bool(winner)


def predict_nsfw(emb: np.ndarray | None, signals: dict[str, Any] | float | None, *,
                 sensitivity: str = "normal") -> bool | None:
    """True/False from a very close labelled neighbour, else from the weights learned over the vault
    signals (classifier, explicit, suggestive, safe), else from the calibrated classifier threshold,
    else None. `signals` is the pipeline's dict or the bare classifier score. The sensitivity shifts
    the learned cuts (low: fewer in the vault, high: more), never a neighbour's own label."""
    if not store.enabled():
        return None
    model = store.load(MODEL)
    if not model:
        return None
    vault = model["vault"]
    if emb is not None and vault["ids"]:
        sims = _neighbours(NSFW_ARRAY, emb)
        if sims is not None and sims.size:
            i = int(np.argmax(sims))
            if sims[i] >= BEST_MIN:
                return bool(vault["labels"][i])
    vec = signal_vector(signals)
    weights = vault.get("weights")
    if weights and vec is not None and (isinstance(signals, dict)):
        w = np.asarray([float(weights["w"].get(k, 0.0)) for k in SIGNALS], dtype=np.float64)
        z = float(np.asarray(vec, dtype=np.float64) @ w + float(weights.get("b", 0.0)))
        return bool(1.0 / (1.0 + np.exp(-z)) >= WEIGHT_CUT.get(sensitivity, 0.5))
    t = vault.get("threshold")
    if t is not None and vec is not None and (isinstance(signals, (int, float)) or signals.get("cls") is not None):
        return vec[0] >= float(t) + THRESHOLD_SHIFT.get(sensitivity, 0.0)
    return None


def status() -> dict[str, Any]:
    model = store.load(MODEL) or {}
    cat = model.get("category", {})
    vault = model.get("vault", {})
    fwd = model.get("forward", {})
    return {
        "category": {"examples": int(cat.get("examples", 0)), "labels": list(cat.get("names", []))},
        "vault": {"examples": int(vault.get("examples", 0)), "threshold": vault.get("threshold"),
                  "weights": ((vault.get("weights") or {}).get("w") if vault.get("weights") else None)},
        "forward": {"examples": int(fwd.get("examples", 0))},
    }
