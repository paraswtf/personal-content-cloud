"""Train every learner, keep learned/status.json, decide when training is due (section 18).

A training pass is cheap (seconds) and runs under one lock, from worker housekeeping every ten
minutes when a choice is newer than the last pass, and on demand from the API. The signature of the
choices (newest timestamps and counts) is stored with the status, so an unchanged library trains
nothing again."""
from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any

from app.learning import doc_type, events, forwards, images, misclick, not_faces, people, store, tags
from app.util import now_iso

log = logging.getLogger(__name__)

STATUS = "status"
_lock = threading.Lock()
LEARNERS = (("doc_type", doc_type), ("images", images), ("misclick", misclick), ("people", people), ("tags", tags),
            ("forwards", forwards), ("not_faces", not_faces))


def signature(conn: sqlite3.Connection) -> list[Any]:
    """What the learners read, summarised: newest change times and counts."""
    def one(sql: str, *params: Any) -> Any:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None

    return [
        one("SELECT MAX(updated_at) FROM items WHERE category_source = 'user' OR nsfw_source = 'user' "
            "OR misclick_source = 'user' OR forward_source = 'user'"),
        one("SELECT COUNT(*) FROM items WHERE category_source = 'user' OR nsfw_source = 'user' OR misclick_source = 'user' "
            "OR forward_source = 'user'"),
        one("SELECT COUNT(*) FROM items WHERE deleted_at IS NOT NULL AND misclick = 1"),
        one("SELECT MAX(updated_at) FROM documents WHERE source = 'user'"),
        one("SELECT COUNT(*) FROM documents WHERE source = 'user'"),
        one("SELECT COUNT(*) FROM faces WHERE assigned_by = 'user'"),
        one("SELECT COUNT(*) FROM bodies WHERE assigned_by = 'user'"),
        one("SELECT COUNT(*) FROM animals WHERE species_source = 'user'"),
        one("SELECT COUNT(*) FROM guess_rejections"),
        one("SELECT MAX(id) FROM learning_events"),
        one("SELECT COUNT(*) FROM item_user_tags WHERE source = 'user'"),
        one("SELECT COUNT(*) FROM tag_rejections"),
        one("SELECT COUNT(*) FROM not_faces"),
    ]


def train_all(conn: sqlite3.Connection) -> dict[str, Any]:
    """Run every learner (a failing one is logged and reported, the others still train)."""
    with _lock:
        sig = signature(conn)
        learned: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for name, mod in LEARNERS:
            try:
                learned[name] = mod.train(conn)
            except Exception as e:  # noqa: BLE001 - one learner must not stop the others
                log.exception("learning: %s failed to train", name)
                errors[name] = str(e)[:300]
        store.save(STATUS, {"trained_at": now_iso(), "signature": sig, "learned": learned, "errors": errors})
    return status(conn)


def maybe_train(conn: sqlite3.Connection) -> bool:
    """Train when a choice is newer than the last pass. Returns True when it trained."""
    if not store.enabled(conn):
        return False
    st = store.load(STATUS)
    if st is not None and st.get("signature") == signature(conn):
        return False
    train_all(conn)
    return True


def pending(conn: sqlite3.Connection) -> int:
    """Choices made after the last training (all of them before the first)."""
    st = store.load(STATUS) or {}
    since = st.get("trained_at") or ""
    n = conn.execute(
        "SELECT COUNT(*) FROM items WHERE (category_source = 'user' OR nsfw_source = 'user' OR misclick_source = 'user' "
        "OR forward_source = 'user') AND updated_at > ?", (since,)
    ).fetchone()[0]
    n += conn.execute("SELECT COUNT(*) FROM documents WHERE source = 'user' AND updated_at > ?", (since,)).fetchone()[0]
    n += conn.execute("SELECT COUNT(*) FROM learning_events WHERE created_at >= ?", (since,)).fetchone()[0]
    n += conn.execute("SELECT COUNT(*) FROM not_faces WHERE created_at >= ?", (since,)).fetchone()[0]
    if not since:
        n += conn.execute("SELECT COUNT(*) FROM faces WHERE assigned_by = 'user'").fetchone()[0]
    return int(n)


def reset() -> None:
    store.clear()


def status(conn: sqlite3.Connection) -> dict[str, Any]:
    st = store.load(STATUS) or {}
    img = images.status()
    return {
        "enabled": store.enabled(conn),
        "trained_at": st.get("trained_at"),
        "pending": pending(conn),
        "learners": {
            "doc_type": doc_type.status(),
            "category": img["category"],
            "vault": img["vault"],
            "misclick": misclick.status(),
            "people": people.status(),
            "tags": tags.status(),
            "forwards": forwards.status(),
            "not_faces": not_faces.status(),
        },
        "events": _event_counts(conn),
        "errors": st.get("errors") or {},
        "vault_sensitivity": vault_sensitivity(conn),
    }


def vault_sensitivity(conn: sqlite3.Connection) -> str:
    from app.config import settings
    from app.db import kv_get

    value = (kv_get(conn, "vault_sensitivity", settings.vault_sensitivity) or "normal").lower()
    return value if value in ("low", "normal", "high") else "normal"


def set_vault_sensitivity(conn: sqlite3.Connection, value: str) -> None:
    from app.db import kv_set, tx

    with tx(conn):
        kv_set(conn, "vault_sensitivity", value)


def _event_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Per kind; for tags only the removed suggestions (label 0) count as corrections."""
    out = events.counts(conn)
    if "tag" in out:
        out["tag"] = int(conn.execute("SELECT COUNT(*) FROM learning_events WHERE kind = 'tag' AND label = 0").fetchone()[0])
    return out
