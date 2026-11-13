"""SQLite access.

- One connection per thread or request (`connect()`); never share a connection across threads.
- Connections are in autocommit mode. Group writes with `with tx(conn): ...`, which starts
  BEGIN IMMEDIATE (or a SAVEPOINT when already inside a transaction).
- Rows are sqlite3.Row (index by column name).
- init_db recreates items_fts when its stored tokenizer differs from FTS_TOKENIZE and sets kv
  fts_rebuild = 1; ingest.recover_on_startup then indexes every item again.
- init_db adds the MIGRATED_COLUMNS an older database lacks before it applies schema.sql, so a
  database from any earlier build upgrades in place at startup, and recreates the CHECKED_TABLES
  whose CHECK constraint changed (rows copied).
"""
from __future__ import annotations

import itertools
import re
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from app.config import settings

SCHEMA_VERSION = 1
# Columns added after the first release: (table, column, SQL type). init_db adds the ones an
# existing table lacks before schema.sql runs, so a database from any earlier build upgrades in
# place on the next start without losing data, and an index in schema.sql may cover any of them.
# A new column goes here and into its CREATE TABLE in schema.sql; tests/test_db_upgrade.py checks
# that a database from before the first of these ends up identical to a fresh one.
MIGRATED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("items", "content_id", "TEXT"),
    ("items", "companion_of", "INTEGER REFERENCES items(id) ON DELETE SET NULL"),
    ("items", "live_video_item_id", "INTEGER REFERENCES items(id) ON DELETE SET NULL"),
    ("items", "live", "INTEGER NOT NULL DEFAULT 0"),
    ("uploads", "pair_key", "TEXT"),
    ("sessions", "device_name", "TEXT"),
    ("persons", "kind", "TEXT NOT NULL DEFAULT 'person'"),
    ("persons", "species", "TEXT"),
    ("items", "animal_count", "INTEGER NOT NULL DEFAULT 0"),
    ("items", "place", "TEXT"),
    ("items", "place_city", "TEXT"),
    ("items", "place_admin", "TEXT"),
    ("items", "place_country", "TEXT"),
    ("items", "place_source", "TEXT"),
    ("items", "gps_checked", "INTEGER NOT NULL DEFAULT 0"),
    ("uploads", "session_id", "INTEGER"),
    ("uploads", "resource", "TEXT"),
    ("items", "received_via", "TEXT"),
    ("items", "forward", "INTEGER NOT NULL DEFAULT 0"),
    ("items", "forward_reasons", "TEXT"),
    ("items", "forward_source", "TEXT NOT NULL DEFAULT 'auto' CHECK (forward_source IN ('auto','user'))"),
    ("items", "forward_signals", "TEXT"),
    ("items", "forward_checked", "INTEGER NOT NULL DEFAULT 0"),
    ("animals", "species_source", "TEXT NOT NULL DEFAULT 'auto' CHECK (species_source IN ('auto','user'))"),
    ("animals", "species_scores", "TEXT"),
    ("animals", "source", "TEXT NOT NULL DEFAULT 'detector' CHECK (source IN ('detector','user'))"),
    ("item_tags", "relation", "TEXT NOT NULL DEFAULT 'in' CHECK (relation IN ('in','related'))"),
    ("faces", "clarity", "REAL"),
    ("faces", "sharpness", "REAL"),
    ("bodies", "source", "TEXT NOT NULL DEFAULT 'detector' CHECK (source IN ('detector','user'))"),
    ("faces", "source", "TEXT NOT NULL DEFAULT 'detector' CHECK (source IN ('detector','user'))"),
    ("faces", "by_guess", "INTEGER NOT NULL DEFAULT 0"),
    ("bodies", "by_guess", "INTEGER NOT NULL DEFAULT 0"),
    ("faces", "moved", "INTEGER NOT NULL DEFAULT 0"),
    ("bodies", "moved", "INTEGER NOT NULL DEFAULT 0"),
    ("animals", "moved", "INTEGER NOT NULL DEFAULT 0"),
    ("faces", "age_groups", "BLOB"),
    ("persons", "recognition_paused_at", "TEXT"),
)
_SCHEMA_FILE = Path(__file__).with_name("schema.sql")
# Full-text tokenizer of items_fts. unicode61 treats combining marks (Unicode Mn, Mc) as separators,
# which splits a Devanagari word at every vowel sign and virama ("मुंबईत" became "म" and "बईत").
# tokenchars adds every mark from U+0900 to U+0DFF (Devanagari through Sinhala), generated once with
# unicodedata (Unicode 15.0) and written out here and in schema.sql. Keep the two in step:
# tests/test_search_devanagari.py checks that schema.sql contains FTS_TOKENIZE.
FTS_TOKENCHARS = (
    "\u0900\u0901\u0902\u0903\u093a\u093b\u093c\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0945\u0946"
    "\u0947\u0948\u0949\u094a\u094b\u094c\u094d\u094e\u094f\u0951\u0952\u0953\u0954\u0955\u0956\u0957"
    "\u0962\u0963\u0981\u0982\u0983\u09bc\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c4\u09c7\u09c8\u09cb"
    "\u09cc\u09cd\u09d7\u09e2\u09e3\u09fe\u0a01\u0a02\u0a03\u0a3c\u0a3e\u0a3f\u0a40\u0a41\u0a42\u0a47"
    "\u0a48\u0a4b\u0a4c\u0a4d\u0a51\u0a70\u0a71\u0a75\u0a81\u0a82\u0a83\u0abc\u0abe\u0abf\u0ac0\u0ac1"
    "\u0ac2\u0ac3\u0ac4\u0ac5\u0ac7\u0ac8\u0ac9\u0acb\u0acc\u0acd\u0ae2\u0ae3\u0afa\u0afb\u0afc\u0afd"
    "\u0afe\u0aff\u0b01\u0b02\u0b03\u0b3c\u0b3e\u0b3f\u0b40\u0b41\u0b42\u0b43\u0b44\u0b47\u0b48\u0b4b"
    "\u0b4c\u0b4d\u0b55\u0b56\u0b57\u0b62\u0b63\u0b82\u0bbe\u0bbf\u0bc0\u0bc1\u0bc2\u0bc6\u0bc7\u0bc8"
    "\u0bca\u0bcb\u0bcc\u0bcd\u0bd7\u0c00\u0c01\u0c02\u0c03\u0c04\u0c3c\u0c3e\u0c3f\u0c40\u0c41\u0c42"
    "\u0c43\u0c44\u0c46\u0c47\u0c48\u0c4a\u0c4b\u0c4c\u0c4d\u0c55\u0c56\u0c62\u0c63\u0c81\u0c82\u0c83"
    "\u0cbc\u0cbe\u0cbf\u0cc0\u0cc1\u0cc2\u0cc3\u0cc4\u0cc6\u0cc7\u0cc8\u0cca\u0ccb\u0ccc\u0ccd\u0cd5"
    "\u0cd6\u0ce2\u0ce3\u0cf3\u0d00\u0d01\u0d02\u0d03\u0d3b\u0d3c\u0d3e\u0d3f\u0d40\u0d41\u0d42\u0d43"
    "\u0d44\u0d46\u0d47\u0d48\u0d4a\u0d4b\u0d4c\u0d4d\u0d57\u0d62\u0d63\u0d81\u0d82\u0d83\u0dca\u0dcf"
    "\u0dd0\u0dd1\u0dd2\u0dd3\u0dd4\u0dd6\u0dd8\u0dd9\u0dda\u0ddb\u0ddc\u0ddd\u0dde\u0ddf\u0df2\u0df3"
)
FTS_TOKENIZE = f"tokenize = 'unicode61 remove_diacritics 2 tokenchars ''{FTS_TOKENCHARS}'''"
FTS_REBUILD_KEY = "fts_rebuild"  # kv flag: items_fts was recreated, ingest.recover_on_startup refills it
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
    """Create the database and tables if needed, and upgrade an older one. Safe to call repeatedly."""
    conn = connect()
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        schema = _SCHEMA_FILE.read_text(encoding="utf-8")
        # Missing columns first: schema.sql has indexes on some of them (items_content on content_id,
        # for one), and CREATE INDEX on a column the table lacks fails the start.
        with tx(conn):
            _add_missing_columns(conn)
        conn.executescript(schema)
        from app.undo import install_triggers  # section 33: the journal follows the current columns

        with tx(conn):
            install_triggers(conn)
        with tx(conn):
            _upgrade_fts(conn, schema)
            _upgrade_checks(conn, schema)
            _mark_confirmed_guesses(conn)
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


def _upgrade_fts(conn: sqlite3.Connection, schema: str) -> bool:
    """Recreate items_fts when its stored definition lacks FTS_TOKENIZE (a database made before the
    Devanagari tokenizer: CREATE VIRTUAL TABLE IF NOT EXISTS never alters an existing table). The
    new table is empty; kv fts_rebuild = 1 tells ingest.recover_on_startup to index every item
    again. Runs inside init_db's transaction, so the drop, the create and the flag commit together.
    Returns True when the table was recreated."""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'items_fts'").fetchone()
    if row is not None and FTS_TOKENIZE in (row["sql"] or ""):
        return False
    m = re.search(r"CREATE VIRTUAL TABLE IF NOT EXISTS items_fts\b.*?\n\);", schema, re.DOTALL)
    if m is None:
        raise RuntimeError("schema.sql has no items_fts definition")
    conn.execute("DROP TABLE IF EXISTS items_fts")
    conn.execute(m.group(0))
    kv_set(conn, FTS_REBUILD_KEY, 1)
    return True


# Tables whose CHECK constraint grows over time (SQLite cannot alter a CHECK): recreated from
# schema.sql when the stored definition differs, rows copied, indexes made again.
CHECKED_TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("learning_events", ("CREATE INDEX IF NOT EXISTS learning_events_kind ON learning_events(kind, created_at)",)),
    ("guess_rejections", ()),   # kind 'animal' (section 22, not this pet)
)


def _check_clause(sql: str | None) -> str | None:
    m = re.search(r"CHECK \(kind IN \(([^)]*)\)\)", sql or "")
    return m.group(1) if m else None


def _upgrade_checks(conn: sqlite3.Connection, schema: str) -> list[str]:
    """Recreate each CHECKED_TABLES table whose stored CHECK on `kind` differs from schema.sql's (a
    new kind, for one), copying every row and creating its indexes again. Runs inside init_db's
    transaction. Returns the tables rebuilt."""
    rebuilt: list[str] = []
    for table, index_sql in CHECKED_TABLES:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
        m = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \(.*?\n\);", schema, re.DOTALL)
        if row is None or m is None:
            continue
        wanted, have = _check_clause(m.group(0)), _check_clause(row["sql"])
        if wanted is None or have == wanted:
            continue
        cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
        conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
        conn.execute(m.group(0))
        names = ", ".join(cols)
        conn.execute(f"INSERT INTO {table}({names}) SELECT {names} FROM {table}_old")
        conn.execute(f"DROP TABLE {table}_old")
        for stmt in index_sql:
            conn.execute(stmt)
        rebuilt.append(table)
    return rebuilt


def _add_missing_columns(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """ALTER TABLE ... ADD COLUMN for every MIGRATED_COLUMNS entry an existing table lacks. A table
    that is not there yet is skipped: schema.sql creates it with every column. Returns what it added."""
    added: list[tuple[str, str]] = []
    existing: dict[str, set[str]] = {}
    for table, column, sql_type in MIGRATED_COLUMNS:
        if table not in existing:
            existing[table] = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        columns = existing[table]
        if not columns or column in columns:  # no columns at all: the table does not exist yet
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
        columns.add(column)
        added.append((table, column))
    return added


@dataclass
class ActionContext:
    """The owner action a request belongs to (section 33, app.undo): tx() journals its writes."""

    id: str
    method: str
    path: str
    undo_of: str | None = None
    logged: bool = False  # some journaled row changed (the response then carries X-Action-Id)


current_action: ContextVar[ActionContext | None] = ContextVar("pcc_action", default=None)


def _begin_action(conn: sqlite3.Connection, act: ActionContext) -> None:
    conn.execute("INSERT OR REPLACE INTO undo_context(id, action_id) VALUES (1, ?)", (act.id,))
    conn.execute("INSERT OR IGNORE INTO undo_actions(id, method, path, created_at, undo_of) VALUES (?, ?, ?, ?, ?)",
                 (act.id, act.method, act.path, _now(), act.undo_of))


def _end_action(conn: sqlite3.Connection, act: ActionContext) -> None:
    if not act.logged:
        act.logged = conn.execute("SELECT 1 FROM undo_log WHERE action_id = ? LIMIT 1", (act.id,)).fetchone() is not None
        if not act.logged:  # a read, or a write the journal leaves out: nothing to undo
            conn.execute("DELETE FROM undo_actions WHERE id = ? AND undo_of IS NULL", (act.id,))
    conn.execute("DELETE FROM undo_context")


def _now() -> str:
    from app.util import now_iso

    return now_iso()


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Write transaction. Nested use becomes a savepoint. Inside an owner action (current_action) the
    transaction is journaled for undo: the undo_context row lives only inside it (section 33)."""
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
    act = current_action.get()
    try:
        if act is not None:
            _begin_action(conn, act)
        yield conn
        if act is not None:
            _end_action(conn, act)
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


def _mark_confirmed_guesses(conn: sqlite3.Connection) -> None:
    """Once (kv 'guess_marks_v1'): boxes whose person came from a guess the owner confirmed before faces.by_guess
    and bodies.by_guess existed. The guess learner's log keeps each confirmation's item and person; the owner's
    boxes of that person there (not drawn by the owner) are it. Call inside a tx."""
    if kv_get(conn, "guess_marks_v1") == "1":
        return
    pairs = conn.execute(
        "SELECT DISTINCT item_id, person_id FROM learning_events WHERE kind = 'guess' AND label = 1 "
        "AND item_id IS NOT NULL AND person_id IS NOT NULL").fetchall()
    for item_id, person_id in pairs:
        for table in ("faces", "bodies"):
            conn.execute(f"UPDATE {table} SET by_guess = 1 WHERE item_id = ? AND person_id = ? AND assigned_by = 'user' "
                         "AND source = 'detector' AND by_guess = 0", (item_id, person_id))
    kv_set(conn, "guess_marks_v1", "1")


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
