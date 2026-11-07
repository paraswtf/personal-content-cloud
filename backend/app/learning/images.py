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
NEIGHBOUR_MIN = 0.80   # a neighbour counts from this cosine
BEST_MIN = 0.85        # the best neighbour must be at least this close
VOTE_MIN = 0.6         # the winner's share of the neighbours' weight
TOP_K = 5
NSFW_MIN_LABELS = 8
NSFW_MIN_SIDE = 3


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
        "SELECT id, nsfw, nsfw_score FROM items WHERE nsfw_source = 'user' AND deleted_at IS NULL "
        "AND kind IN ('image', 'video') ORDER BY id"
    ).fetchall()
    embs = _embeddings(conn, sorted({int(r["id"]) for r in [*cat_rows, *nsfw_rows]}))
    cat_x, cat_ids = _matrix(embs, [int(r["id"]) for r in cat_rows])
    cat_label = {int(r["id"]): str(r["category"]) for r in cat_rows}
    nsfw_x, nsfw_ids = _matrix(embs, [int(r["id"]) for r in nsfw_rows])
    nsfw_label = {int(r["id"]): int(r["nsfw"]) for r in nsfw_rows}
    pairs = [(float(r["nsfw_score"]), int(r["nsfw"])) for r in nsfw_rows if r["nsfw_score"] is not None]
    threshold = calibrate_nsfw(pairs)
    if cat_ids:
        store.save_array(CAT_ARRAY, cat_x)
    else:
        store.remove(CAT_ARRAY)
    if nsfw_ids:
        store.save_array(NSFW_ARRAY, nsfw_x)
    else:
        store.remove(NSFW_ARRAY)
    labels = [cat_label[i] for i in cat_ids]
    model = {
        "category": {"ids": cat_ids, "labels": labels, "examples": len(cat_ids), "names": sorted(set(labels))},
        "vault": {"ids": nsfw_ids, "labels": [nsfw_label[i] for i in nsfw_ids], "examples": len(nsfw_rows),
                  "threshold": threshold, "scored": len(pairs)},
        "trained_at": now_iso(),
    }
    store.save(MODEL, model)
    return status()


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


def predict_nsfw(emb: np.ndarray | None, nsfw_score: float | None) -> bool | None:
    """True/False from a very close labelled neighbour, else from the calibrated threshold, else None."""
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
    t = vault.get("threshold")
    if t is not None and nsfw_score is not None:
        return float(nsfw_score) >= float(t)
    return None


def status() -> dict[str, Any]:
    model = store.load(MODEL) or {}
    cat = model.get("category", {})
    vault = model.get("vault", {})
    return {
        "category": {"examples": int(cat.get("examples", 0)), "labels": list(cat.get("names", []))},
        "vault": {"examples": int(vault.get("examples", 0)), "threshold": vault.get("threshold")},
    }
