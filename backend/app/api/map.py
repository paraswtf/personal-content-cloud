"""Map and places (docs/ARCHITECTURE.md section 25): /api/map, behind the session, vault through
want_vault, every GET /api/items filter accepted (app.api.library.item_filters).

GET  /api/map/points?bbox=w,s,e,n   MapPoints{points: [[lat, lon, uid, k]], total, with_gps}; k: 0 image,
                                    1 video, 2 document, 3 other; five decimals; no limit
GET  /api/map/items?bbox=&cursor&limit  ItemPage of the items inside the box in library order
GET  /api/map/places                [PlaceOut] grouped by place, most photos first, at most 500
GET  /api/map/status                {gazetteer, with_gps, unchecked, unplaced, running}
POST /api/map/refresh               {started}: places and the GPS backfill to completion, in the background
"""
from __future__ import annotations

import sqlite3
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.deps import get_db, require_user, want_vault
from app.api.library import _page, item_filters
from app.config import CATEGORIES, ITEM_KINDS
from app.geo import places
from app.schemas import ItemPage

router = APIRouter(dependencies=[Depends(require_user)])

KIND_CODES = {"image": 0, "video": 1, "pdf": 2, "document": 2, "other": 3}
PLACES_LIMIT = 500


class MapPoints(BaseModel):
    points: list[tuple[float, float, str, int]]
    total: int
    with_gps: int


class PlaceOut(BaseModel):
    place: str
    place_city: str | None = None
    place_admin: str | None = None
    place_country: str | None = None
    count: int
    lat: float
    lon: float


class MapStatus(BaseModel):
    gazetteer: bool
    with_gps: int
    unchecked: int
    unplaced: int
    running: bool


class RefreshOut(BaseModel):
    started: bool


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
    dups: Literal["collapse", "all"] = "collapse",
    sort: Literal["taken", "added"] = "taken",
    album: int | None = Query(default=None, ge=1),
    vault: bool = Depends(want_vault),
) -> dict[str, Any]:
    """The library's filters, as keyword arguments for item_filters."""
    return dict(category=category, kind=kind, person=person, tag=tag, live=live, date_from=date_from,
                date_to=date_to, favorite=favorite, misclicks=misclicks, dups=dups, sort=sort, vault=vault, album=album)


def parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    """w,s,e,n in degrees; w > e means the box crosses the antimeridian. 422 when malformed."""
    if value is None or value.strip() == "":
        return None
    try:
        parts = [float(p) for p in value.split(",")]
    except ValueError:
        parts = []
    if len(parts) != 4 or any(p != p for p in parts):
        raise HTTPException(status_code=422, detail="bbox must be w,s,e,n")
    w, s, e, n = parts
    if not (-180.0 <= w <= 180.0 and -180.0 <= e <= 180.0 and -90.0 <= s <= 90.0 and -90.0 <= n <= 90.0) or s > n:
        raise HTTPException(status_code=422, detail="bbox must be w,s,e,n")
    return w, s, e, n


def bbox_sql(box: tuple[float, float, float, float] | None) -> tuple[list[str], list[Any]]:
    if box is None:
        return [], []
    w, s, e, n = box
    where = ["i.gps_lat >= ? AND i.gps_lat <= ?"]
    params: list[Any] = [s, n]
    if w <= e:
        where.append("i.gps_lon >= ? AND i.gps_lon <= ?")
        params += [w, e]
    else:
        where.append("(i.gps_lon >= ? OR i.gps_lon <= ?)")
        params += [w, e]
    return where, params


GPS_WHERE = "i.gps_lat IS NOT NULL AND i.gps_lon IS NOT NULL"


@router.get("/points", response_model=MapPoints)
def points(bbox: str | None = Query(default=None, max_length=80), filters: dict[str, Any] = Depends(filter_params),
           conn: sqlite3.Connection = Depends(get_db)) -> MapPoints:
    where, params, _sort = item_filters(conn, **filters)
    cond = " AND ".join(where)
    total = conn.execute(f"SELECT COUNT(*) FROM items i WHERE {cond}", params).fetchone()[0]
    with_gps = conn.execute(f"SELECT COUNT(*) FROM items i WHERE {cond} AND {GPS_WHERE}", params).fetchone()[0]
    bw, bp = bbox_sql(parse_bbox(bbox))
    rows = conn.execute(
        f"SELECT i.gps_lat, i.gps_lon, i.uid, i.kind FROM items i WHERE {' AND '.join([*where, GPS_WHERE, *bw])}",
        [*params, *bp],
    ).fetchall()
    pts = [(round(float(r["gps_lat"]), 5), round(float(r["gps_lon"]), 5), r["uid"], KIND_CODES.get(r["kind"], 3))
           for r in rows]
    return MapPoints(points=pts, total=int(total), with_gps=int(with_gps))


@router.get("/items", response_model=ItemPage)
def items(bbox: str | None = Query(default=None, max_length=80), cursor: str | None = Query(default=None, max_length=512),
          limit: int = Query(default=60, ge=1, le=200), filters: dict[str, Any] = Depends(filter_params),
          conn: sqlite3.Connection = Depends(get_db)) -> ItemPage:
    where, params, sort_expr = item_filters(conn, **filters)
    bw, bp = bbox_sql(parse_bbox(bbox))
    return _page(conn, [*where, GPS_WHERE, *bw], [*params, *bp], sort_expr, cursor, limit)


@router.get("/places", response_model=list[PlaceOut])
def place_list(filters: dict[str, Any] = Depends(filter_params), conn: sqlite3.Connection = Depends(get_db)) -> list[PlaceOut]:
    where, params, _sort = item_filters(conn, **filters)
    rows = conn.execute(
        "SELECT i.place, i.place_city, i.place_admin, i.place_country, COUNT(*) AS n, AVG(i.gps_lat) AS lat, AVG(i.gps_lon) AS lon "
        f"FROM items i WHERE {' AND '.join(where)} AND i.place IS NOT NULL AND {GPS_WHERE} "
        "GROUP BY i.place ORDER BY n DESC, i.place LIMIT ?",
        [*params, PLACES_LIMIT],
    ).fetchall()
    return [PlaceOut(place=r["place"], place_city=r["place_city"], place_admin=r["place_admin"], place_country=r["place_country"],
                     count=int(r["n"]), lat=round(float(r["lat"]), 5), lon=round(float(r["lon"]), 5)) for r in rows]


@router.get("/status", response_model=MapStatus)
def map_status(conn: sqlite3.Connection = Depends(get_db)) -> MapStatus:
    return MapStatus(**places.status(conn))


@router.post("/refresh", response_model=RefreshOut)
def refresh() -> RefreshOut:
    return RefreshOut(started=places.start_refresh())
