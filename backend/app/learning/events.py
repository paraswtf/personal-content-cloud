"""learning_events: the corrections and outcomes that the main tables do not keep (section 18).

Kinds: guess (label 1 confirm / 0 reject, features = the guess evidence terms), category, vault,
misclick and doc_type (label None, features {"from", "to", ...}), tag (label 1 added by hand / 0 a
suggestion removed). Rows are written inside the caller's transaction (tx() nests as a savepoint).
"""
from __future__ import annotations

import sqlite3
from typing import Any

from app.db import tx
from app.util import json_dumps, now_iso

KINDS = ("guess", "category", "vault", "misclick", "doc_type", "tag", "merge", "forward", "species")


def record(conn: sqlite3.Connection, kind: str, label: int | None, features: dict[str, Any] | None,
           item_id: int | None = None, person_id: int | None = None) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown learning event kind {kind!r}")
    with tx(conn):
        conn.execute(
            "INSERT INTO learning_events(kind, label, features, item_id, person_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (kind, None if label is None else int(label), json_dumps(features or {}), item_id, person_id, now_iso()),
        )


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {r["kind"]: int(r["n"]) for r in conn.execute("SELECT kind, COUNT(*) AS n FROM learning_events GROUP BY kind")}


def rows(conn: sqlite3.Connection, kind: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, label, features, item_id, person_id, created_at FROM learning_events WHERE kind = ? ORDER BY id",
        (kind,),
    ).fetchall()
