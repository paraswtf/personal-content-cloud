"""SQLite access.

- One connection per thread or request (`connect()`); never share a connection across threads.
- Connections are in autocommit mode. Group writes with `with tx(conn): ...`, which starts
  BEGIN IMMEDIATE (or a SAVEPOINT when already inside a transaction).
- Rows are sqlite3.Row (index by column name).
"""
from __future__ import annotations

import itertools
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.config import settings

SCHEMA_VERSION = 1
# Columns added after the first release: (table, column, SQL type). init_db adds the missing ones,
# so an existing database upgrades in place on the next start without losing data.
MIGRATED_COLUMNS: tuple[tuple[str, str, str], ...] = ()
_SCHEMA_FILE = Path(__file__).with_name("schema.sql")
_savepoints = itertools.count(1)


def connect() -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def init_db() -> None:
    """Create the database and tables if needed. Safe to call repeatedly."""
    conn = connect()
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA_FILE.read_text())
        with tx(conn):
            _add_missing_columns(conn)
            conn.execute("INSERT OR IGNORE INTO kv(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            conn.execute(
                "UPDATE kv SET value = ? WHERE key = 'schema_version' AND CAST(value AS INTEGER) < ?",
                (str(SCHEMA_VERSION), SCHEMA_VERSION),
            )
            conn.execute("INSERT OR IGNORE INTO kv(key, value) VALUES ('faces_version', '1')")
            conn.execute("INSERT OR IGNORE INTO kv(key, value) VALUES ('search_version', '1')")
            # Item ids are never reused, so a restored or re-added item never takes an old id.
            conn.execute(
                "INSERT OR IGNORE INTO kv(key, value) SELECT 'item_id_mark', CAST(COALESCE(MAX(id), 0) AS TEXT) FROM items"
            )
    finally:
        conn.close()


def _add_missing_columns(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    added: list[tuple[str, str]] = []
    existing: dict[str, set[str]] = {}
    for table, column, sql_type in MIGRATED_COLUMNS:
        if table not in existing:
            existing[table] = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column in existing[table]:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
        existing[table].add(column)
        added.append((table, column))
    return added


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Write transaction. Nested use becomes a savepoint."""
    if conn.in_transaction:
        name = f"sp_{next(_savepoints)}"
        conn.execute(f"SAVEPOINT {name}")
        try:
            yield conn
        except BaseException:
            if conn.in_transaction:  # SQLite may have rolled everything back already (disk full)
                conn.execute(f"ROLLBACK TO {name}")
                conn.execute(f"RELEASE {name}")
            raise
        conn.execute(f"RELEASE {name}")
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    try:
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def kv_get(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def kv_set(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO kv(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def _bump(conn: sqlite3.Connection, key: str) -> int:
    conn.execute(
        f"INSERT INTO kv(key, value) VALUES ('{key}', '2') "
        "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
    )
    return int(kv_get(conn, key, "1") or 1)


def faces_version(conn: sqlite3.Connection) -> int:
    return int(kv_get(conn, "faces_version", "1") or 1)


def bump_faces_version(conn: sqlite3.Connection) -> int:
    """Call inside the same transaction as ANY change to faces, bodies or persons (insert, delete,
    person_id/assigned_by change, person delete/merge). The in-memory face index reloads when this
    number changes."""
    return _bump(conn, "faces_version")


def search_version(conn: sqlite3.Connection) -> int:
    return int(kv_get(conn, "search_version", "1") or 1)


def bump_search_version(conn: sqlite3.Connection) -> int:
    """Call inside the same transaction as any write to item_embeddings or faces.torso_embedding,
    and whenever an item's visibility changes (delete, restore, nsfw)."""
    return _bump(conn, "search_version")


def next_item_id(conn: sqlite3.Connection) -> int:
    """Allocate an item id that was never used (call inside the insert's transaction)."""
    mark = int(kv_get(conn, "item_id_mark", "0") or 0)
    current = conn.execute("SELECT COALESCE(MAX(id), 0) FROM items").fetchone()[0]
    nxt = max(mark, int(current)) + 1
    kv_set(conn, "item_id_mark", nxt)
    return nxt


def one(conn: sqlite3.Connection, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
    return conn.execute(sql, params).fetchone()


def all_rows(conn: sqlite3.Connection, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def scalar(conn: sqlite3.Connection, sql: str, params: tuple | dict = (), default: Any = None) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row is not None and row[0] is not None else default


def chunked(values: list[Any], size: int = 500) -> Iterator[list[Any]]:
    for i in range(0, len(values), size):
        yield values[i:i + size]
