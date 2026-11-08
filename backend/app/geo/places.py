"""Place names and the GPS backfill (docs/ARCHITECTURE.md section 25).

housekeeping(conn, budget_s) runs from the worker every ten minutes and, to completion, from
POST /api/map/refresh (start_refresh, on worker.run_in_background): first the items with GPS and no
place yet get their place columns from the gazetteer in batches of PLACE_BATCH and their FTS row
refreshed (the place goes into the labels column), then the GPS backfill re-reads the originals of
images and videos stored before positions were read (probe.read_gps) in batches of GPS_BATCH, marking
gps_checked either way. tag_item(conn, item_id) does one item. status(conn) feeds GET /api/map/status."""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from typing import Any

from app import storage
from app.db import tx
from app.geo import gazetteer
from app.media import probe

log = logging.getLogger(__name__)

PLACE_BATCH = 500
GPS_BATCH = 200
SOURCE_GEONAMES = "geonames"
SOURCE_NONE = "none"

_refresh_lock = threading.Lock()
_refresh_running = False


def _columns(place: gazetteer.Place | None) -> tuple[str | None, str | None, str | None, str | None, str]:
    if place is None:
        return None, None, None, None, SOURCE_NONE
    return place.display, place.city, place.admin, place.country, SOURCE_GEONAMES


def _tag_rows(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> list[int]:
    """Set the place columns of rows (id, gps_lat, gps_lon) in one transaction; returns the ids that
    got a place name (the ones whose FTS row needs a refresh)."""
    named: list[int] = []
    with tx(conn):
        for r in rows:
            place = gazetteer.place_for(r["gps_lat"], r["gps_lon"])
            display, city, admin, country, source = _columns(place)
            conn.execute(
                "UPDATE items SET place = ?, place_city = ?, place_admin = ?, place_country = ?, place_source = ? WHERE id = ?",
                (display, city, admin, country, source, int(r["id"])),
            )
            if place is not None:
                named.append(int(r["id"]))
    return named


def _refresh_fts(conn: sqlite3.Connection, item_ids: list[int]) -> None:
    from app.documents.extract import refresh_fts

    for item_id in item_ids:
        try:
            refresh_fts(conn, item_id)
        except Exception:  # noqa: BLE001 - the index is rebuilt by the next refresh of that item
            log.exception("places: could not refresh the search index of item %s", item_id)


def tag_item(conn: sqlite3.Connection, item_id: int) -> bool:
    """Look the item's place up now (True when it got a name). No-op without GPS or a gazetteer."""
    row = conn.execute("SELECT id, gps_lat, gps_lon FROM items WHERE id = ?", (item_id,)).fetchone()
    if row is None or row["gps_lat"] is None or row["gps_lon"] is None or not gazetteer.available():
        return False
    named = _tag_rows(conn, [row])
    _refresh_fts(conn, named)
    return bool(named)


def _tag_pending(conn: sqlite3.Connection, deadline: float) -> int:
    touched = 0
    while time.monotonic() < deadline:
        rows = conn.execute(
            "SELECT id, gps_lat, gps_lon FROM items WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL "
            "AND place_source IS NULL AND deleted_at IS NULL ORDER BY id LIMIT ?",
            (PLACE_BATCH,),
        ).fetchall()
        if not rows:
            break
        _refresh_fts(conn, _tag_rows(conn, rows))
        touched += len(rows)
        if len(rows) < PLACE_BATCH:
            break
    return touched


def _backfill_gps(conn: sqlite3.Connection, deadline: float) -> int:
    touched = 0
    tag = gazetteer.available()
    while time.monotonic() < deadline:
        rows = conn.execute(
            "SELECT id, uid, version, ext FROM items WHERE kind IN ('image', 'video') AND gps_lat IS NULL "
            "AND gps_checked = 0 AND deleted_at IS NULL ORDER BY id LIMIT ?",
            (GPS_BATCH,),
        ).fetchall()
        if not rows:
            break
        for r in rows:
            if time.monotonic() >= deadline:
                return touched
            path = storage.original_path(r["uid"], int(r["version"]), r["ext"])
            gps = None
            try:
                gps = probe.read_gps(path) if path.is_file() else None
            except probe.ToolFailure as e:
                # ffprobe could not run: leave the rest unchecked for a later round
                log.warning("places: GPS backfill paused, %s", e)
                return touched
            except Exception:  # noqa: BLE001 - one odd file never stops the backfill
                log.exception("places: could not read GPS of %s", path)
            with tx(conn):
                if gps is not None:
                    conn.execute(
                        "UPDATE items SET gps_lat = ?, gps_lon = ?, gps_alt = ?, gps_checked = 1 WHERE id = ?",
                        (gps[0], gps[1], gps[2], int(r["id"])),
                    )
                else:
                    conn.execute("UPDATE items SET gps_checked = 1 WHERE id = ?", (int(r["id"]),))
            if gps is not None and tag:
                _refresh_fts(conn, _tag_rows(conn, [conn.execute(
                    "SELECT id, gps_lat, gps_lon FROM items WHERE id = ?", (int(r["id"]),)).fetchone()]))
            touched += 1
        if len(rows) < GPS_BATCH:
            break
    return touched


def _round(conn: sqlite3.Connection, budget_s: float) -> int:
    """One round of tagging and back-filling within the budget; the number of items touched."""
    deadline = time.monotonic() + max(0.0, float(budget_s))
    touched = 0
    if gazetteer.available():
        touched += _tag_pending(conn, deadline)
    touched += _backfill_gps(conn, deadline)
    return touched


def housekeeping(conn: sqlite3.Connection, budget_s: float = 5.0) -> int:
    """The worker's entry: tag items with places and back-fill GPS within the time budget; returns the
    number of items touched. Does nothing while a full refresh (start_refresh) is running, so the two
    never work the same rows at once."""
    if refresh_running():
        return 0
    return _round(conn, budget_s)


def refresh_running() -> bool:
    return _refresh_running


def _refresh_all() -> None:
    """Run housekeeping to completion on a connection of its own (the background thread)."""
    global _refresh_running
    from app import db

    conn = db.connect()
    try:
        while _round(conn, budget_s=30.0) > 0:
            pass
    finally:
        conn.close()
        with _refresh_lock:
            _refresh_running = False


def start_refresh() -> bool:
    """Start the full refresh in the background; False when one is already running."""
    global _refresh_running
    from app import worker

    with _refresh_lock:
        if _refresh_running:
            return False
        _refresh_running = True
    started = worker.run_in_background("geo", _refresh_all)
    if started is None:
        with _refresh_lock:
            _refresh_running = False
        return False
    return True


def status(conn: sqlite3.Connection) -> dict[str, Any]:
    """Counts over the library's normal side (live, not a companion, not in the vault)."""
    row = conn.execute(
        "SELECT COALESCE(SUM(CASE WHEN gps_lat IS NOT NULL THEN 1 ELSE 0 END), 0) AS with_gps, "
        "COALESCE(SUM(CASE WHEN kind IN ('image', 'video') AND gps_lat IS NULL AND gps_checked = 0 THEN 1 ELSE 0 END), 0) AS unchecked, "
        "COALESCE(SUM(CASE WHEN gps_lat IS NOT NULL AND place_source IS NULL THEN 1 ELSE 0 END), 0) AS unplaced "
        "FROM items WHERE deleted_at IS NULL AND companion_of IS NULL AND nsfw = 0"
    ).fetchone()
    return {
        "gazetteer": gazetteer.available(),
        "with_gps": int(row["with_gps"]),
        "unchecked": int(row["unchecked"]),
        "unplaced": int(row["unplaced"]),
        "running": refresh_running(),
    }
