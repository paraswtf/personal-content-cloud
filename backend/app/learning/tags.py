"""Tag learner (docs/ARCHITECTURE.md section 19).

Per tag, the owner's tagged items are positives (SigLIP whole-image embeddings for photos and videos,
TF-IDF text vectors for pdf and office documents) and removed suggestions are negatives. A tag is
learned once it has MIN_POS positives of a kind; it is suggested on a new item when the best positive
cosine is at least IMG_MIN (images) or TEXT_MIN (text) and beats the best negative by MARGIN.
"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from app.db import tx
from app.learning import store, textvec
from app.util import now_iso

MODEL = "tags"
POS_ARRAY = "tags_pos"
NEG_ARRAY = "tags_neg"
MIN_POS = 3
IMG_MIN = 0.80
TEXT_MIN = 0.45
MARGIN = 0.02
MAX_TEXT = 60000
IMAGE_KINDS = ("image", "video")
TEXT_KINDS = ("pdf", "document")


def _embeddings(conn: sqlite3.Connection, item_ids: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for i in range(0, len(item_ids), 500):
        chunk = item_ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        for r in conn.execute(f"SELECT item_id, embedding FROM item_embeddings WHERE frame_index = 0 AND item_id IN ({ph})", chunk):
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 0:
                out[int(r["item_id"])] = (v / n).astype(np.float32)
    return out


def _texts(conn: sqlite3.Connection, item_ids: list[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for i in range(0, len(item_ids), 500):
        chunk = item_ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        for r in conn.execute(f"SELECT item_id, text FROM item_text WHERE item_id IN ({ph})", chunk):
            if r["text"]:
                out[int(r["item_id"])] = str(r["text"])[:MAX_TEXT]
    return out


def _examples(conn: sqlite3.Connection) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]:
    """(positives, negatives) as (item_id, tag_id, kind) of live items."""
    pos = [(int(r["item_id"]), int(r["tag_id"]), str(r["kind"])) for r in conn.execute(
        "SELECT ut.item_id, ut.tag_id, i.kind FROM item_user_tags ut JOIN items i ON i.id = ut.item_id "
        "WHERE ut.source = 'user' AND i.deleted_at IS NULL ORDER BY ut.tag_id, ut.item_id")]
    neg = [(int(r["item_id"]), int(r["tag_id"]), str(r["kind"])) for r in conn.execute(
        "SELECT g.item_id, g.tag_id, i.kind FROM tag_rejections g JOIN items i ON i.id = g.item_id "
        "WHERE i.deleted_at IS NULL ORDER BY g.tag_id, g.item_id")]
    return pos, neg


def train(conn: sqlite3.Connection) -> dict[str, Any]:
    pos, neg = _examples(conn)
    ids = sorted({i for i, _t, _k in pos} | {i for i, _t, _k in neg})
    embs = _embeddings(conn, [i for i in ids])
    texts = _texts(conn, ids)
    # images
    pos_img = [(i, t) for i, t, k in pos if k in IMAGE_KINDS and i in embs]
    neg_img = [(i, t) for i, t, k in neg if k in IMAGE_KINDS and i in embs]
    # text
    pos_txt = [(i, t) for i, t, k in pos if k in TEXT_KINDS and i in texts]
    neg_txt = [(i, t) for i, t, k in neg if k in TEXT_KINDS and i in texts]
    toks = {i: textvec.tokens(texts[i]) for i in {i for i, _t in pos_txt + neg_txt}}
    idf, n_docs = textvec.build_idf(toks.values())
    vecs = {i: textvec.vector(t, idf, n_docs) for i, t in toks.items()}

    def dims_ok(rows: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(i, t) for i, t in rows if embs[i].shape[0] == embs[pos_img[0][0]].shape[0]] if pos_img else []

    pos_img = dims_ok(pos_img)
    neg_img = dims_ok(neg_img)
    if pos_img:
        store.save_array(POS_ARRAY, np.stack([embs[i] for i, _t in pos_img]))
    else:
        store.remove(POS_ARRAY)
    if neg_img:
        store.save_array(NEG_ARRAY, np.stack([embs[i] for i, _t in neg_img]))
    else:
        store.remove(NEG_ARRAY)
    counts: dict[str, dict[str, int]] = {}
    for _i, t in pos_img:
        counts.setdefault(str(t), {"img": 0, "txt": 0})["img"] += 1
    for _i, t in pos_txt:
        counts.setdefault(str(t), {"img": 0, "txt": 0})["txt"] += 1
    learned = sorted({t for t, c in counts.items() if c["img"] >= MIN_POS or c["txt"] >= MIN_POS}, key=int)
    names = {int(r["id"]): r["name"] for r in conn.execute("SELECT id, name FROM tags")}
    model = {
        "pos_tag": [t for _i, t in pos_img], "neg_tag": [t for _i, t in neg_img],
        "text_pos": [{"tag": t, "vec": vecs[i]} for i, t in pos_txt if vecs.get(i)],
        "text_neg": [{"tag": t, "vec": vecs[i]} for i, t in neg_txt if vecs.get(i)],
        "idf": idf, "n_docs": n_docs, "counts": counts, "learned": [int(t) for t in learned],
        "names": {str(t): names.get(int(t)) for t in learned},
        "examples": len(pos), "rejections": len(neg), "trained_at": now_iso(),
    }
    store.save(MODEL, model)
    return status()


def predict(emb: np.ndarray | None, text: str | None, kind: str) -> list[tuple[int, float]]:
    """[(tag_id, score)] of the learned tags this item looks like, best first."""
    if not store.enabled():
        return []
    model = store.load(MODEL)
    if not model or not model.get("learned"):
        return []
    learned = {int(t) for t in model["learned"]}
    counts = model.get("counts", {})
    out: list[tuple[int, float]] = []
    if kind in IMAGE_KINDS and emb is not None:
        pos = store.load_array(POS_ARRAY)
        v = np.asarray(emb, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if pos is not None and pos.size and n > 0 and pos.shape[1] == v.shape[0]:
            v = v / n
            sims = pos @ v
            neg = store.load_array(NEG_ARRAY)
            neg_sims = neg @ v if neg is not None and neg.size and neg.shape[1] == v.shape[0] else None
            pos_tags = np.asarray(model["pos_tag"], dtype=np.int64)
            neg_tags = np.asarray(model.get("neg_tag", []), dtype=np.int64)
            for t in sorted(learned):
                if counts.get(str(t), {}).get("img", 0) < MIN_POS:
                    continue
                mask = pos_tags == t
                if not mask.any():
                    continue
                best = float(sims[mask].max())
                worst = float(neg_sims[neg_tags == t].max()) if neg_sims is not None and (neg_tags == t).any() else -1.0
                if best >= IMG_MIN and best > worst + MARGIN:
                    out.append((t, round(best, 4)))
    elif kind in TEXT_KINDS and text:
        vec = textvec.vector(textvec.tokens(text[:MAX_TEXT]), model.get("idf", {}), int(model.get("n_docs", 0)))
        if vec:
            best_by: dict[int, float] = {}
            for ex in model.get("text_pos", []):
                t = int(ex["tag"])
                if t in learned and counts.get(str(t), {}).get("txt", 0) >= MIN_POS:
                    best_by[t] = max(best_by.get(t, 0.0), textvec.cosine(vec, ex["vec"]))
            worst_by: dict[int, float] = {}
            for ex in model.get("text_neg", []):
                t = int(ex["tag"])
                worst_by[t] = max(worst_by.get(t, -1.0), textvec.cosine(vec, ex["vec"]))
            for t, best in sorted(best_by.items()):
                if best >= TEXT_MIN and best > worst_by.get(t, -1.0) + MARGIN:
                    out.append((t, round(best, 4)))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def apply_item(conn: sqlite3.Connection, item_id: int, *, emb: np.ndarray | None, text: str | None, kind: str) -> int:
    """Insert auto rows for the learned tags this item looks like, skipping tags already on the item
    and rejected pairs. Runs inside the caller's transaction. Returns how many rows were added."""
    hits = predict(emb, text, kind)
    if not hits:
        return 0
    present = {int(r["tag_id"]) for r in conn.execute("SELECT tag_id FROM item_user_tags WHERE item_id = ?", (item_id,))}
    rejected = {int(r["tag_id"]) for r in conn.execute("SELECT tag_id FROM tag_rejections WHERE item_id = ?", (item_id,))}
    known = {int(r["id"]) for r in conn.execute("SELECT id FROM tags")}
    now = now_iso()
    added = 0
    for tag_id, score in hits:
        if tag_id in present or tag_id in rejected or tag_id not in known:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO item_user_tags(item_id, tag_id, source, score, created_at) VALUES (?, ?, 'auto', ?, ?)",
            (item_id, tag_id, float(score), now),
        )
        added += 1
    return added


def apply_all(conn: sqlite3.Connection) -> int:
    """Suggest learned tags on every live item from its stored embedding or text (no model runs)."""
    if not store.enabled(conn):
        return 0
    model = store.load(MODEL)
    if not model or not model.get("learned"):
        return 0
    from app.documents.extract import refresh_fts

    rows = conn.execute("SELECT id, kind FROM items WHERE deleted_at IS NULL AND kind IN ('image', 'video', 'pdf', 'document')").fetchall()
    ids = [int(r["id"]) for r in rows]
    kinds = {int(r["id"]): str(r["kind"]) for r in rows}
    embs = _embeddings(conn, [i for i in ids if kinds[i] in IMAGE_KINDS])
    texts = _texts(conn, [i for i in ids if kinds[i] in TEXT_KINDS])
    total = 0
    changed: list[int] = []
    with tx(conn):
        for item_id in ids:
            n = apply_item(conn, item_id, emb=embs.get(item_id), text=texts.get(item_id), kind=kinds[item_id])
            if n:
                total += n
                changed.append(item_id)
    for item_id in changed:
        refresh_fts(conn, item_id)
    return total


def status() -> dict[str, Any]:
    model = store.load(MODEL) or {}
    return {
        "tags": len(model.get("learned", [])),
        "examples": int(model.get("examples", 0)),
        "rejections": int(model.get("rejections", 0)),
        "names": [n for n in (model.get("names") or {}).values() if n],
    }
