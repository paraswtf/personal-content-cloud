"""A database made by an earlier build upgrades in place when the server starts (db.init_db).

fixtures/schema_before_migrations.sql is schema.sql from the last build before the first entry of
db.MIGRATED_COLUMNS (commit ded0a39, 2026-11-07). The review server kept a database of that shape and
the first start of a later build died in init_db at CREATE INDEX items_content ON items(content_id):
schema.sql ran before the column was added. These tests build that database, with rows in it, and
check that init_db brings it to the shape of a fresh one without losing anything."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

OLD_SCHEMA = Path(__file__).parent / "fixtures" / "schema_before_migrations.sql"
T0 = "2026-11-07T14:25:00Z"


def _rows(conn: sqlite3.Connection, sql: str, *params) -> list[tuple]:
    return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def _shape(conn: sqlite3.Connection) -> dict[str, object]:
    """Everything structural about a database that code can depend on, order-free. Column order
    differs between an upgraded database (added columns come last) and a fresh one, so columns are
    a set; index SQL is compared with whitespace collapsed."""
    tables = sorted(
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table') AND name NOT LIKE 'sqlite_%'")
    )
    columns = {t: set(_rows(conn, f"PRAGMA table_info({t})")) for t in tables}
    columns = {t: {(c[1], c[2], c[3], c[4], c[5]) for c in cols} for t, cols in columns.items()}  # name, type, notnull, default, pk
    fks = {t: {(f[2], f[3], f[4], f[5], f[6]) for f in _rows(conn, f"PRAGMA foreign_key_list({t})")} for t in tables}
    indexes = {
        r[0]: re.sub(r"\s+", " ", r[1]).strip()
        for r in conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL")
    }
    return {"tables": tables, "columns": columns, "foreign_keys": fks, "indexes": indexes}


@pytest.fixture()
def old_db(tmp_path):
    """A data dir whose database has the pre-migration shape and a row in each migrated table, as a
    server from before 2026-11-07 22:25 would have left it. init_db has not run yet."""
    from app.config import reload_settings, settings

    reload_settings(data_dir=tmp_path / "data", frontend_dist=tmp_path / "dist", workers_enabled=False,
                    min_free_disk_bytes=0, chunk_size=256 * 1024, session_days=30)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.executescript(OLD_SCHEMA.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO items(id, uid, kind, original_filename, ext, mime, sha256, size_bytes, created_at, updated_at) "
        "VALUES (7, 'u7', 'image', 'a.jpg', '.jpg', 'image/jpeg', ?, 1234, ?, ?)",
        ("ab" * 32, T0, T0),
    )
    conn.execute(
        "INSERT INTO uploads(id, uid, original_filename, size_bytes, status, created_at, updated_at) "
        "VALUES (3, 'up3', 'a.jpg', 1234, 'stored', ?, ?)",
        (T0, T0),
    )
    conn.execute(
        "INSERT INTO sessions(id, token_hash, created_at, last_seen_at, expires_at) VALUES (5, 'h5', ?, ?, '2027-01-01T00:00:00Z')",
        (T0, T0),
    )
    conn.execute("INSERT INTO persons(id, name, name_key, created_at, updated_at) VALUES (2, 'Asha', 'asha', ?, ?)", (T0, T0))
    conn.execute("INSERT INTO kv(key, value) VALUES ('schema_version', '1')")
    conn.commit()
    conn.close()
    for reset in ("app.auth.passwords:reset_all", "app.analysis.index:reset_index", "app.search.service:reset_matrix"):
        mod, fn = reset.split(":")
        try:
            getattr(__import__(mod, fromlist=[fn]), fn)()
        except Exception:
            pass
    yield settings.db_path
    reload_settings()


def test_the_fixture_is_the_pre_migration_shape(old_db):
    """The fixture predates every migrated column, else the tests below prove less than they claim."""
    from app import db

    conn = sqlite3.connect(old_db)
    for table, column, _ in db.MIGRATED_COLUMNS:
        names = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        assert names, table
        assert column not in names, (table, column)
    conn.close()


def test_init_db_upgrades_a_database_from_before_the_migrated_columns(old_db):
    from app import db

    db.init_db()  # used to fail: no such column: content_id

    conn = db.connect()
    try:
        for table, column, _ in db.MIGRATED_COLUMNS:
            assert column in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}, (table, column)
        indexes = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert {"items_content", "items_companion", "items_gps", "uploads_asset", "uploads_pair"} <= indexes
        # the rows are still there, with the new columns at their defaults
        item = conn.execute("SELECT * FROM items WHERE id = 7").fetchone()
        assert item["uid"] == "u7" and item["content_id"] is None and item["live"] == 0
        assert item["animal_count"] == 0 and item["gps_checked"] == 0 and item["place"] is None
        up = conn.execute("SELECT * FROM uploads WHERE id = 3").fetchone()
        assert up["status"] == "stored" and up["pair_key"] is None and up["session_id"] is None and up["resource"] is None
        assert conn.execute("SELECT device_name FROM sessions WHERE id = 5").fetchone()[0] is None
        person = conn.execute("SELECT kind, species FROM persons WHERE id = 2").fetchone()
        assert tuple(person) == ("person", None)
        assert conn.execute("SELECT value FROM kv WHERE key = 'item_id_mark'").fetchone()[0] == "7"
        # a second start has nothing left to add
        assert db._add_missing_columns(conn) == []
    finally:
        conn.close()
    db.init_db()


def test_an_upgraded_database_has_the_shape_of_a_fresh_one(old_db, tmp_path):
    from app import db

    db.init_db()
    upgraded = sqlite3.connect(old_db)
    fresh = sqlite3.connect(tmp_path / "fresh.db")
    fresh.executescript(db._SCHEMA_FILE.read_text(encoding="utf-8"))
    try:
        a, b = _shape(upgraded), _shape(fresh)
        assert a["tables"] == b["tables"]
        for t in b["tables"]:
            assert a["columns"][t] == b["columns"][t], t
            assert a["foreign_keys"][t] == b["foreign_keys"][t], t
        assert a["indexes"] == b["indexes"]
    finally:
        upgraded.close()
        fresh.close()


def test_the_server_starts_on_an_old_database(old_db):
    """The whole startup path (lifespan: init_db, recovery, the site snapshot) on the old shape."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as c:
        r = c.get("/api/auth/state")
        assert r.status_code == 200, r.text
        assert r.json()["api_build"]
