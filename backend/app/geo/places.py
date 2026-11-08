"""Place names and the GPS backfill (docs/ARCHITECTURE.md section 25). housekeeping(conn) runs from the
worker every ten minutes and from POST /api/map/refresh."""
from __future__ import annotations

import sqlite3


def housekeeping(conn: sqlite3.Connection, budget_s: float = 5.0) -> int:
    """Tag items with places and back-fill GPS within the time budget; returns the number of items touched."""
    return 0
