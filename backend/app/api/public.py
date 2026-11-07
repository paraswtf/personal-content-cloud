"""Public read API (no login; needs the access cookie only when ACCESS_KEY is set).

CONTRACT (implemented by the API agent). Router mounted at prefix /api, every route depends on
app.api.deps.require_access except GET /api/config. Hidden media and hidden people are never
returned here, even to admins (admins use /api/admin/...), EXCEPT GET /api/media/{uid}, which
returns hidden items to admins (so the admin can unhide them from the viewer).

GET /api/config -> ConfigOut
GET /api/people?q= -> PeopleOut
    publicly listed people (app.api.common.is_publicly_listed); q filters by name substring
    (case-insensitive, also matches "unnamed #12" style display names); named sorted by name,
    unnamed by media_count desc then id.
GET /api/people/{id} -> PersonOut (404 if hidden, missing or in no visible media; unnamed people
    below the public threshold are still returned by id, since links to them can exist).
    `links` holds the public kinds only (app.links.PUBLIC_KINDS), even for an admin; email and
    phone never appear on any route in this file except GET /api/media/{uid} for an admin.
GET /api/media?person=1&person=2&uploader=3&kind=image|video&sort=recent|taken&cursor=&limit=60 -> MediaPage
    person (repeatable) = item contains ALL these people (app.api.common.people_filter_sql);
    uploader = items this uploader contributed (media_contributors); sort recent = created_at desc,
    id desc; taken = COALESCE(taken_at, created_at) asc, id asc; keyset pagination with
    app.util.encode_cursor/decode_cursor; limit 1..200; total = count for the filter.
GET /api/media/{uid} -> MediaDetail (404 if missing or hidden for non-admins). Faces whose person
    is hidden show person=None for non-admins. `people` = distinct visible people from faces + tags.
    `geo` (capture position, place name, capture time label) when the item has a position;
    `geotag_available` for photos with one that are not hidden (app.api.common.geotag_available);
    `video_geotag_available` for videos with one that are not hidden (frames with the card and the
    geotag video below).
    `people_links` = linked profiles of `people`: public kinds for visitors, all kinds for a
    signed-in admin (that response is sent with Cache-Control: private, no-store).
POST /api/media/{uid}/geotag-video  GeotagVideoIn {"pos": "br"} -> GeotagVideoOut
    starts a copy of the whole video with the geotag card burned in at pos (app.media.geotag_video),
    or returns the job that exists for (uid, version, pos); a job in state error is retried. Needs
    the X-Requested-With: hub header (403 otherwise). 503 + Retry-After when too many renders
    wait. A render refused for disk space comes back as state error, "Not enough free disk space".
GET /api/media/{uid}/geotag-video?pos=br -> GeotagVideoOut  the job's state (polling); 404
    {"detail": "not_started"} when there is no job (never started, expired, server restarted):
    POST again. Both: 404 unless a video with a position and settings.geotag_enabled, hidden only
    for admins; url (when done) is /files/geotag-video/{uid}?v=&pos=&download=1. Sent with
    Cache-Control: no-store.
GET /api/uploaders?q= -> list[UploaderOut]  uploaders with >= 1 visible item, by name.
GET /api/search?q=<1-200 chars>&person=1&person=2&kind=image|video&limit=60&offset=0 -> SearchPage
    semantic search (app.search.service): items whose image embeddings match the text, by relevance
    (score 0-1) desc, only hits >= settings.search_min_relevance; videos carry match_time_s (time
    of the best-matching frame). person (repeatable) = item contains ALL these people; a hidden or
    unknown person gives an empty page. Hidden media never appear. 422 for an empty or blank q;
    503 {"detail": "search_unavailable"} when the model is missing, disabled or fails.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.api.common import (
    MEDIA_ITEM_COLUMNS,
    geo_out,
    geotag_available,
    is_publicly_listed,
    media_item,
    people_filter_sql,
    person_links,
    person_out,
    person_ref,
    person_stats,
    site_stats,
    place_label,
    uploader_ref,
    valid_cursor_scalar,
    video_geotag_available,
)
from app.api.deps import get_db, require_access
from app import storage
from app.auth import CSRF_HEADER, CSRF_VALUE, has_access, is_admin
from app.config import settings
from app.media import geotag_video
from app.schemas import (
    ConfigOut,
    FaceOut,
    GeotagPosition,
    GeotagVideoIn,
    GeotagVideoOut,
    MediaDetail,
    MediaPage,
    MediaUrls,
    PeopleOut,
    PersonLinks,
    PersonOut,
    PersonRef,
    SearchHit,
    SearchPage,
    SiteStats,
    UploaderOut,
    UploaderRef,
)
from app.media.video import stream_friendly
from app.search import indexer as search_indexer
from app.search import service as search_service
from app.urls import download_url, face_url, original_url, playable_video_url, preview_small_url, preview_url, thumb_url
from app.util import decode_cursor, encode_cursor

log = logging.getLogger(__name__)

router = APIRouter()
access = [Depends(require_access)]

# Value for <input type=file accept=...>. HEIC/HEIF and a few camera video containers are
# listed by extension because some browsers report no MIME type for them.
ACCEPT = "image/*,video/*,.heic,.heif,.avif,.webp,.mov,.m4v,.mkv,.3gp"

_SORT_KEYS = {
    # sort -> (key expression, order direction, keyset comparison operator)
    "recent": ("m.created_at", "DESC", "<"),
    "taken": ("COALESCE(m.taken_at, m.created_at)", "ASC", ">"),
}


# -------------------------------------------------------------------- config
@router.get("/config", response_model=ConfigOut)
def get_config(request: Request, conn: sqlite3.Connection = Depends(get_db)) -> ConfigOut:
    granted = has_access(request)
    stats = site_stats(conn) if granted else SiteStats(media=0, photos=0, videos=0, people=0, contributors=0)
    return ConfigOut(
        site_title=settings.site_title,
        max_upload_bytes=settings.max_upload_bytes,
        chunk_size=settings.chunk_size,
        accept=ACCEPT,
        is_admin=is_admin(request),
        access_granted=granted,
        stats=stats,
    )


# -------------------------------------------------------------------- people
@router.get("/people", response_model=PeopleOut, dependencies=access)
def list_people(q: str | None = Query(default=None, max_length=120),
                conn: sqlite3.Connection = Depends(get_db)) -> PeopleOut:
    rows = conn.execute("SELECT id, name, hidden, cover_face_id FROM persons WHERE hidden = 0").fetchall()
    stats = person_stats(conn, None)
    needle = (q or "").strip().casefold()
    named: list[PersonOut] = []
    unnamed: list[PersonOut] = []
    for row in rows:
        s = stats.get(row["id"])
        if not is_publicly_listed(row, s):
            continue
        p = person_out(row, s)
        if needle and needle not in p.display_name.casefold():
            continue
        (named if row["name"] else unnamed).append(p)
    named.sort(key=lambda p: ((p.name or "").casefold(), p.id))
    unnamed.sort(key=lambda p: (-p.media_count, p.id))
    return PeopleOut(named=named, unnamed=unnamed)


@router.get("/people/{person_id}", response_model=PersonOut, dependencies=access)
def get_person(person_id: int, conn: sqlite3.Connection = Depends(get_db)) -> PersonOut:
    row = conn.execute(
        "SELECT id, name, hidden, cover_face_id FROM persons WHERE id = ?", (person_id,)
    ).fetchone()
    if row is None or row["hidden"]:
        raise HTTPException(status_code=404, detail="Person not found")
    stats = person_stats(conn, [person_id]).get(person_id)
    if stats is None or stats.media_count == 0:
        # Only on hidden media (or on nothing): the name must not leak through the id.
        raise HTTPException(status_code=404, detail="Person not found")
    out = person_out(row, stats)
    out.links = person_links(conn, [person_id], include_private=False).get(person_id, [])
    return out


# --------------------------------------------------------------------- media
def _parse_cursor(cursor: str | None) -> tuple[str, int] | None:
    if not cursor:
        return None
    value = decode_cursor(cursor)
    if (
        value is None
        or len(value) != 2
        or not isinstance(value[0], str)
        or not isinstance(value[1], int)
        or isinstance(value[1], bool)
        or not valid_cursor_scalar(value[1])
    ):
        raise HTTPException(status_code=400, detail="Invalid cursor")
    return value[0], value[1]


@router.get("/media", response_model=MediaPage, dependencies=access)
def list_media(
    person: list[int] = Query(default=[]),
    uploader: int | None = None,
    kind: Literal["image", "video"] | None = None,
    sort: Literal["recent", "taken"] = "recent",
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=60, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_db),
) -> MediaPage:
    where = ["m.hidden = 0"]
    params: list = []
    if person:
        ids = sorted(set(person))
        ph = ",".join("?" * len(ids))
        visible = conn.execute(
            f"SELECT COUNT(*) FROM persons WHERE id IN ({ph}) AND hidden = 0", ids
        ).fetchone()[0]
        if visible != len(ids):
            # A hidden (or unknown) person must not reveal which photos they are in.
            return MediaPage(items=[], next_cursor=None, total=0)
        sql, p = people_filter_sql(ids)
        where.append(sql)
        params.extend(p)
    if uploader is not None:
        where.append("m.id IN (SELECT media_id FROM media_contributors WHERE uploader_id = ?)")
        params.append(uploader)
    if kind is not None:
        where.append("m.kind = ?")
        params.append(kind)

    key_expr, direction, op = _SORT_KEYS[sort]
    total = conn.execute(f"SELECT COUNT(*) FROM media m WHERE {' AND '.join(where)}", params).fetchone()[0]

    page_where, page_params = list(where), list(params)
    after = _parse_cursor(cursor)
    if after is not None:
        page_where.append(f"({key_expr}, m.id) {op} (?, ?)")
        page_params.extend(after)
    rows = conn.execute(
        f"SELECT {MEDIA_ITEM_COLUMNS}, {key_expr} AS sort_key FROM media m "
        f"WHERE {' AND '.join(page_where)} "
        f"ORDER BY {key_expr} {direction}, m.id {direction} LIMIT ?",
        [*page_params, limit + 1],
    ).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = encode_cursor([rows[-1]["sort_key"], rows[-1]["id"]]) if more and rows else None
    return MediaPage(items=[media_item(r) for r in rows], next_cursor=next_cursor, total=total)


@router.get("/media/{uid}", response_model=MediaDetail, dependencies=access)
def get_media(uid: str, request: Request, response: Response,
              conn: sqlite3.Connection = Depends(get_db)) -> MediaDetail:
    admin = is_admin(request)
    if admin:
        # Carries admin-only data (hidden items, email/phone links): never store it in a cache.
        response.headers["cache-control"] = "private, no-store"
    m = conn.execute(
        "SELECT m.*, u.name AS uploader_name FROM media m JOIN uploaders u ON u.id = m.uploader_id WHERE m.uid = ?",
        (uid,),
    ).fetchone()
    if m is None or (m["hidden"] and not admin):
        raise HTTPException(status_code=404, detail="Photo or video not found")

    contributors = [
        uploader_ref(r["id"], r["name"])
        for r in conn.execute(
            "SELECT u.id, u.name FROM media_contributors c JOIN uploaders u ON u.id = c.uploader_id "
            "WHERE c.media_id = ? ORDER BY (c.uploader_id = ?) DESC, c.first_upload_at, u.id",
            (m["id"], m["uploader_id"]),
        )
    ]

    people: dict[int, PersonRef] = {}
    faces: list[FaceOut] = []
    for f in conn.execute(
        "SELECT f.id, f.x, f.y, f.w, f.h, f.frame_time_s, f.spans, f.assigned_by, f.det_score, f.person_id, "
        "p.name AS person_name, p.hidden AS person_hidden "
        "FROM faces f LEFT JOIN persons p ON p.id = f.person_id "
        "WHERE f.media_id = ? ORDER BY COALESCE(f.frame_time_s, 0), f.x, f.id",
        (m["id"],),
    ):
        ref = None
        if f["person_id"] is not None:
            if not f["person_hidden"] or admin:
                ref = person_ref(f["person_id"], f["person_name"])
            if not f["person_hidden"]:
                people.setdefault(f["person_id"], person_ref(f["person_id"], f["person_name"]))
        faces.append(FaceOut(
            id=f["id"], x=f["x"], y=f["y"], w=f["w"], h=f["h"], frame_time_s=f["frame_time_s"],
            spans=_spans(f["spans"]), thumb_url=face_url(f["id"]), person=ref, assigned_by=f["assigned_by"],
            det_score=f["det_score"],
        ))

    tags: list[PersonRef] = []
    for t in conn.execute(
        "SELECT p.id, p.name, p.hidden FROM media_tags t JOIN persons p ON p.id = t.person_id "
        "WHERE t.media_id = ? ORDER BY p.name IS NULL, p.name_key, p.id",
        (m["id"],),
    ):
        if t["hidden"] and not admin:
            continue
        ref = person_ref(t["id"], t["name"])
        tags.append(ref)
        if not t["hidden"]:
            people.setdefault(t["id"], ref)

    ordered_people = sorted(people.values(), key=lambda p: (p.name is None, (p.name or "").casefold(), p.id))
    links = person_links(conn, [p.id for p in ordered_people], include_private=admin)
    people_links = [PersonLinks(person_id=p.id, links=links.get(p.id, [])) for p in ordered_people]
    uid_, v = m["uid"], m["version"]
    return MediaDetail(
        uid=uid_,
        kind=m["kind"],
        version=v,
        original_filename=m["original_filename"],
        mime=m["mime"],
        size_bytes=m["size_bytes"],
        width=m["width"],
        height=m["height"],
        duration_s=m["duration_s"],
        fps=m["fps"],
        video_codec=m["video_codec"],
        bitrate=m["bitrate"],
        taken_at=m["taken_at"],
        camera=m["camera"],
        created_at=m["created_at"],
        updated_at=m["updated_at"],
        uploader=UploaderRef(id=m["uploader_id"], name=m["uploader_name"]),
        contributors=contributors,
        urls=MediaUrls(
            thumb=thumb_url(uid_, v),
            preview=preview_url(uid_, v),
            preview_small=preview_small_url(uid_, v),
            original=original_url(uid_, v),
            download=download_url(uid_, v),
            video=_video_url(m),
        ),
        faces=faces,
        tags=tags,
        people=ordered_people,
        faces_status=m["faces_status"],
        preview_status=m["preview_status"],
        hidden=bool(m["hidden"]),
        geo=geo_out(m),
        geotag_available=geotag_available(m),
        video_geotag_available=video_geotag_available(m),
        people_links=people_links,
    )


# -------------------------------------------------------------- geotag video
def _geotag_video_row(conn: sqlite3.Connection, uid: str, request: Request, response: Response) -> sqlite3.Row:
    response.headers["cache-control"] = "no-store"
    m = conn.execute(
        "SELECT uid, kind, version, ext, hidden, duration_s, gps_lat, gps_lon, gps_alt, taken_at "
        "FROM media WHERE uid = ?", (uid,),
    ).fetchone()
    if m is None or (m["hidden"] and not is_admin(request)):
        raise HTTPException(status_code=404, detail="Photo or video not found")
    # Same rule as MediaDetail.video_geotag_available, except that admins may also use hidden items.
    if not video_geotag_available({**dict(m), "hidden": 0}):
        raise HTTPException(status_code=404, detail="No geotagged copy can be made of this item")
    return m


def _geotag_video_out(m: sqlite3.Row, pos: str, st: geotag_video.Status) -> GeotagVideoOut:
    done = st.state == "done"
    return GeotagVideoOut(
        state=st.state, progress=st.progress, error=st.error, bytes=st.bytes if done else None,
        url=f"/files/geotag-video/{m['uid']}?v={m['version']}&pos={pos}&download=1" if done else None,
    )


@router.post("/media/{uid}/geotag-video", response_model=GeotagVideoOut, dependencies=access)
def start_geotag_video(uid: str, body: GeotagVideoIn, request: Request, response: Response,
                       conn: sqlite3.Connection = Depends(get_db)) -> GeotagVideoOut:
    if request.headers.get(CSRF_HEADER) != CSRF_VALUE:
        raise HTTPException(status_code=403, detail="Missing X-Requested-With header")
    m = _geotag_video_row(conn, uid, request, response)
    lat, lon = float(m["gps_lat"]), float(m["gps_lon"])
    source = geotag_video.Source(
        uid=m["uid"], version=int(m["version"]), pos=body.pos,
        path=storage.original_path(m["uid"], m["version"], m["ext"]), duration_s=m["duration_s"],
        lat=lat, lon=lon, alt=m["gps_alt"], place=place_label(lat, lon), captured_at=m["taken_at"],
    )
    try:
        st = geotag_video.submit(source)
    except geotag_video.Busy:
        raise HTTPException(status_code=503, detail="Busy preparing other videos, please try again shortly",
                            headers={"Retry-After": str(geotag_video.RETRY_AFTER_S)}) from None
    return _geotag_video_out(m, body.pos, st)


@router.get("/media/{uid}/geotag-video", response_model=GeotagVideoOut, dependencies=access)
def geotag_video_status(uid: str, request: Request, response: Response, pos: GeotagPosition = "br",
                        conn: sqlite3.Connection = Depends(get_db)) -> GeotagVideoOut:
    m = _geotag_video_row(conn, uid, request, response)
    st = geotag_video.status(m["uid"], m["version"], pos)
    if st is None:
        raise HTTPException(status_code=404, detail="not_started", headers={"cache-control": "no-store"})
    return _geotag_video_out(m, pos, st)


def _video_url(m: sqlite3.Row) -> str | None:
    if m["kind"] != "video":
        return None
    friendly = stream_friendly(m["video_codec"], m["ext"], m["width"], m["height"], m["bitrate"])
    return playable_video_url(m["uid"], m["version"], m["preview_status"], friendly)


def _spans(raw: str | None) -> list[tuple[float, float]] | None:
    """faces.spans (JSON [[start, end], ...]) as pairs; None when absent or unreadable."""
    if not raw:
        return None
    try:
        return [(float(a), float(b)) for a, b in json.loads(raw)]
    except (ValueError, TypeError):
        log.warning("unreadable face spans %r", raw[:80])
        return None


# ----------------------------------------------------------------- uploaders
@router.get("/uploaders", response_model=list[UploaderOut], dependencies=access)
def list_uploaders(q: str | None = Query(default=None, max_length=120),
                   conn: sqlite3.Connection = Depends(get_db)) -> list[UploaderOut]:
    where, params = "", []
    needle = (q or "").strip().casefold()
    if needle:
        where, params = "WHERE instr(u.name_key, ?) > 0", [needle]
    rows = conn.execute(
        "SELECT u.id, u.name, COUNT(*) AS n FROM uploaders u "
        "JOIN media_contributors c ON c.uploader_id = u.id "
        "JOIN media m ON m.id = c.media_id AND m.hidden = 0 "
        f"{where} GROUP BY u.id ORDER BY u.name_key, u.id",
        params,
    ).fetchall()
    return [UploaderOut(id=r["id"], name=r["name"], media_count=r["n"]) for r in rows]


# -------------------------------------------------------------------- search
@router.get("/search", response_model=SearchPage, dependencies=access)
def search_media(
    q: str = Query(min_length=1, max_length=200),
    person: list[int] = Query(default=[]),
    kind: Literal["image", "video"] | None = None,
    limit: int = Query(default=60, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=100_000),
    conn: sqlite3.Connection = Depends(get_db),
) -> SearchPage:
    query = " ".join(q.split())
    if not query:
        raise HTTPException(status_code=422, detail="Type something to search for")
    try:
        result = search_service.search(conn, query, person_ids=person, kind=kind, limit=limit, offset=offset)
    except search_service.SearchUnavailable:
        raise HTTPException(status_code=503, detail="search_unavailable") from None
    model = search_service.current_model()
    return SearchPage(
        query=query,
        items=[
            SearchHit(**media_item(h.row).model_dump(), score=h.score, match_time_s=h.match_time_s)
            for h in result.hits
        ],
        total=result.total,
        next_offset=result.next_offset,
        indexed=search_service.indexed_count(conn, model, include_hidden=False),
        indexing=search_indexer.has_pending(conn, model),
    )
