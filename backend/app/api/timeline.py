"""Timeline buckets (docs/ARCHITECTURE.md section 23).

GET /api/timeline/buckets -> TimelineBuckets{buckets: [{key, count, newest, oldest}], total}, newest first.
GET /api/timeline/bucket?key=YYYY-MM&cursor&limit -> ItemPage of that month in library order.
Both take every GET /api/items filter (library.item_filters, album included) and vault through
want_vault. A bucket is a calendar month of the sort key (COALESCE(taken_at, created_at) for
sort=taken, created_at for sort=added) in settings.local_timezone: the keys come from one indexed
query ordered newest first and are walked against month boundaries (util.local_month_start_utc,
UTC ISO strings in the stored format) compared as text, so the months agree with the from/to filters
and no strftime grouping of UTC strings is needed. Models live here so the API build id covers them.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.deps import get_db, require_user, want_vault
from app.api.library import _page, item_filters
from app.config import CATEGORIES, ITEM_KINDS, settings
from app.schemas import ItemPage
from app.util import local_month_start_utc

router = APIRouter(dependencies=[Depends(require_user)])

KEY_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
MAX_BUCKET_LIMIT = 1000


class TimelineBucket(BaseModel):
    key: str      # "YYYY-MM" in the local timezone
    count: int
    newest: str   # the sort key of the newest item in the month
    oldest: str   # the sort key of the oldest item in the month


class TimelineBuckets(BaseModel):
    buckets: list[TimelineBucket]
    total: int


def filter_params(
    category: Literal[CATEGORIES] | None = None,  # type: ignore[valid-type]
    kind: Literal[ITEM_KINDS] | None = None,  # type: ignore[valid-type]
    person: list[int] = Query(default=[]),
    tag: list[str] = Query(default=[]),
    live: bool | None = None,
    date_from: str | None = Query(default=None, alias="from", max_length=10),
    date_to: str | None = Query(default=None, alias="to", max_length=10),
    favorite: bool | None = None,
    misclicks: Literal["exclude", "include", "only"] = "exclude",
    forwards: Literal["exclude", "include", "only"] = "exclude",
    received: str | None = Query(default=None, max_length=20),
    dups: Literal["collapse", "all"] = "collapse",
    sort: Literal["taken", "added"] = "taken",
    album: int | None = Query(default=None, ge=1),
    vault: bool = Depends(want_vault),
) -> dict[str, Any]:
    """The GET /api/items filters, as keyword arguments for library.item_filters."""
    return dict(category=category, kind=kind, person=person, tag=tag, live=live, date_from=date_from,
                date_to=date_to, favorite=favorite, misclicks=misclicks, forwards=forwards, received=received, dups=dups, sort=sort, vault=vault,
                album=album)


def _tz() -> str:
    return settings.local_timezone or "UTC"


def _local_year_month(key: str, tz_name: str) -> tuple[int, int]:
    """The local calendar month of a stored UTC time string; the UTC month when it does not parse."""
    try:
        dt = datetime.fromisoformat(key.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        try:
            zone = ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError):
            zone = timezone.utc
        local = dt.astimezone(zone)
        return local.year, local.month
    except ValueError:
        try:
            return int(key[:4]), int(key[5:7])
        except ValueError:
            return 1, 1


def _prev_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def bucketize(keys: list[str], tz_name: str) -> list[TimelineBucket]:
    """Group sort keys (UTC ISO strings, sorted newest first) into local calendar months."""
    out: list[TimelineBucket] = []
    n = len(keys)
    if n == 0:
        return out
    i = 0
    year, month = _local_year_month(keys[0], tz_name)
    while i < n:
        start = local_month_start_utc(year, month, tz_name)
        floor = (year, month) <= (1, 1)
        if keys[i] < start and not floor:
            year, month = _prev_month(year, month)
            continue
        j = i
        while j < n and (keys[j] >= start or floor):
            j += 1
        out.append(TimelineBucket(key=f"{year:04d}-{month:02d}", count=j - i, newest=keys[i], oldest=keys[j - 1]))
        i = j
        year, month = _prev_month(year, month)
    return out


def _parse_key(key: str) -> tuple[int, int]:
    m = KEY_RE.match(key.strip())
    if m is None:
        raise HTTPException(status_code=422, detail="key must be YYYY-MM")
    return int(m.group(1)), int(m.group(2))


@router.get("/buckets", response_model=TimelineBuckets)
def buckets(
    filters: dict[str, Any] = Depends(filter_params),
    conn: sqlite3.Connection = Depends(get_db),
) -> TimelineBuckets:
    where, params, sort_expr = item_filters(conn, **filters)
    cond = " AND ".join(where) if where else "1"
    rows = conn.execute(f"SELECT {sort_expr} AS k FROM items i WHERE {cond} ORDER BY k DESC, i.id DESC", params)
    keys = [str(r["k"]) for r in rows]
    return TimelineBuckets(buckets=bucketize(keys, _tz()), total=len(keys))


@router.get("/bucket", response_model=ItemPage)
def bucket(
    key: str = Query(min_length=7, max_length=7),
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=500, ge=1, le=MAX_BUCKET_LIMIT),
    filters: dict[str, Any] = Depends(filter_params),
    conn: sqlite3.Connection = Depends(get_db),
) -> ItemPage:
    year, month = _parse_key(key)
    where, params, sort_expr = item_filters(conn, **filters)
    where.append(f"{sort_expr} >= ?")
    params.append(local_month_start_utc(year, month, _tz()))
    where.append(f"{sort_expr} < ?")
    params.append(local_month_start_utc(year, month + 1, _tz()))
    return _page(conn, where, params, sort_expr, cursor, limit)
