"""Re-sorting the vault without a fresh analysis (section 9): when the sensitivity changes, every
automatically decided photo and video is decided again from the signals the last analysis stored
(classifier score, explicit, suggestive and safe probabilities, faces and bodies) and from the
learner's answer. Items analysed before the signals existed are counted as stale: they need the
"apply to existing items" re-analysis."""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np

from app.analysis import categories
from app.db import bump_faces_version, bump_search_version, tx
from app.util import json_loads, now_iso

log = logging.getLogger(__name__)


def _embedding(conn: sqlite3.Connection, item_id: int) -> np.ndarray | None:
    row = conn.execute(
        "SELECT embedding FROM item_embeddings WHERE item_id = ? AND frame_index = 0", (item_id,)
    ).fetchone()
    return None if row is None else np.frombuffer(row["embedding"], dtype=np.float32)


def resort(conn: sqlite3.Connection, sensitivity: str) -> dict[str, int]:
    """Decide the vault again for the automatically decided items. Returns the counts of items
    checked, moved into the vault, moved out, and stale (no signals stored)."""
    from app.learning import images as learned_images
    from app.media import live as live_media

    rows = conn.execute(
        "SELECT id, kind, nsfw, nsfw_score, category, category_source, category_scores, doc_likeness, "
        "face_count, body_count, live_video_item_id FROM items WHERE deleted_at IS NULL AND companion_of IS NULL "
        "AND kind IN ('image', 'video') AND nsfw_source = 'auto' AND analysis_status = 'done' ORDER BY id"
    ).fetchall()
    out = {"checked": 0, "moved_in": 0, "moved_out": 0, "stale": 0}
    now = now_iso()
    with tx(conn):
        for r in rows:
            scores = json_loads(r["category_scores"], {}) or {}
            if r["nsfw_score"] is None or scores.get("suggestive_p") is None or scores.get("safe_p") is None:
                out["stale"] += 1
                continue
            out["checked"] += 1
            item_id = int(r["id"])
            signals = {"cls": float(r["nsfw_score"]), "explicit": scores.get("explicit_p"),
                       "suggestive": scores.get("suggestive_p"), "safe": scores.get("safe_p")}
            emb0 = _embedding(conn, item_id)
            learned_cat = learned_nsfw = None
            try:
                learned_cat = learned_images.predict_category(emb0)
                learned_nsfw = learned_images.predict_nsfw(emb0, signals, sensitivity=sensitivity)
            except Exception:  # noqa: BLE001 - learning never blocks a decision
                log.exception("vault resort: image learner failed")
            has_person = int(r["face_count"] or 0) > 0 or int(r["body_count"] or 0) > 0
            dec = categories.decide(
                str(r["kind"]), nsfw_score=float(r["nsfw_score"]), explicit_p=scores.get("explicit_p"),
                screenshot_p=scores.get("screenshot_p"), doc_like=r["doc_likeness"], has_person=has_person,
                scores=scores, learned=learned_cat, nsfw_learned=learned_nsfw,
                suggestive_p=scores.get("suggestive_p"), safe_p=scores.get("safe_p"), sensitivity=sensitivity,
            )
            new_nsfw = int(dec.nsfw)
            if new_nsfw == int(r["nsfw"]):
                continue
            out["moved_in" if new_nsfw else "moved_out"] += 1
            if r["category_source"] == "user":
                category = r["category"]
            else:
                category = dec.category
            conn.execute("UPDATE items SET nsfw = ?, category = ?, updated_at = ? WHERE id = ?",
                         (new_nsfw, category, now, item_id))
            if r["live_video_item_id"]:
                live_media.sync_companion(conn, item_id)
        if out["moved_in"] or out["moved_out"]:
            bump_faces_version(conn)
            bump_search_version(conn)
    return out


def describe(moved: dict[str, Any]) -> str:
    """One line for logs and tests."""
    return (f"{moved['checked']} checked, {moved['moved_in']} into the vault, {moved['moved_out']} out, "
            f"{moved['stale']} need a fresh analysis")
