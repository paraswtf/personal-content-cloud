"""Person helpers shared by the API routers and the worker (no API imports here).

CONTRACT.
- clean_person_name(value): app.util.clean_name with max_len NAME_MAX (120); empty or None -> None.
- create_person(conn, name): inserts a person (name_key = util.name_key(name), NULL when unnamed)
  and returns its id. Call inside the caller's transaction (and bump faces_version there).
- find_or_create_named(conn, name): cleans the name; None when nothing is left. Otherwise the
  person whose name_key matches, preferring the one with most faces, then the lowest id; creates a
  named person when none matches. Never commits.
- drop_empty_unnamed(conn, person_ids): deletes unnamed persons left without faces, bodies or
  tags (after a purge). Named persons are kept. Returns the number deleted. Inside a tx.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Iterable

from app.util import clean_name, name_key, now_iso

log = logging.getLogger(__name__)

NAME_MAX = 120


def clean_person_name(value: str | None) -> str | None:
    """clean_name with the person limit; empty -> None."""
    if value is None:
        return None
    name = clean_name(value, max_len=NAME_MAX)
    return name or None


def create_person(conn: sqlite3.Connection, name: str | None) -> int:
    now = now_iso()
    cur = conn.execute(
        "INSERT INTO persons(name, name_key, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (name, name_key(name) if name else None, now, now),
    )
    return int(cur.lastrowid)


def find_or_create_named(conn: sqlite3.Connection, name: str | None) -> int | None:
    clean = clean_person_name(name) if isinstance(name, str) else None
    if not clean:
        return None
    row = conn.execute(
        "SELECT p.id FROM persons p WHERE p.name_key = ? "
        "ORDER BY (SELECT COUNT(*) FROM faces f WHERE f.person_id = p.id) DESC, p.id LIMIT 1",
        (name_key(clean),),
    ).fetchone()
    if row is not None:
        return int(row[0])
    return create_person(conn, clean)


def drop_empty_unnamed(conn: sqlite3.Connection, person_ids: Iterable[int]) -> int:
    dropped = 0
    for pid in sorted({int(p) for p in person_ids}):
        cur = conn.execute(
            "DELETE FROM persons WHERE id = ? AND name IS NULL "
            "AND NOT EXISTS (SELECT 1 FROM faces WHERE person_id = ?) "
            "AND NOT EXISTS (SELECT 1 FROM bodies WHERE person_id = ?) "
            "AND NOT EXISTS (SELECT 1 FROM item_tags WHERE person_id = ?)",
            (pid, pid, pid, pid),
        )
        dropped += cur.rowcount
    return dropped
