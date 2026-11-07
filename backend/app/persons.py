"""Person helpers shared by the admin API and the worker (no API imports here).

CONTRACT.
- clean_person_name(value): app.util.clean_name with max_len NAME_MAX (120); empty or None -> None.
- create_person(conn, name): inserts a visible person (name_key = name.casefold(), NULL when unnamed)
  and returns its id. Call inside the caller's transaction.
- find_or_create_named(conn, name): cleans the name; None when nothing is left. Otherwise the
  person whose name_key equals the cleaned name casefolded, preferring visible people, then the one
  with most faces, then the lowest id; creates a named person when none matches. Never commits.
"""
from __future__ import annotations

import sqlite3

from app.util import clean_name, now_iso

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
        "INSERT INTO persons(name, name_key, hidden, created_at, updated_at) VALUES (?, ?, 0, ?, ?)",
        (name, name.casefold() if name else None, now, now),
    )
    return int(cur.lastrowid)


def find_or_create_named(conn: sqlite3.Connection, name: str | None) -> int | None:
    clean = clean_person_name(name) if isinstance(name, str) else None
    if not clean:
        return None
    row = conn.execute(
        "SELECT p.id FROM persons p WHERE p.name_key = ? "
        "ORDER BY p.hidden, (SELECT COUNT(*) FROM faces f WHERE f.person_id = p.id) DESC, p.id LIMIT 1",
        (clean.casefold(),),
    ).fetchone()
    if row is not None:
        return int(row[0])
    return create_person(conn, clean)
