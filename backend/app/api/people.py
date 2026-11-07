"""People, faces, bodies, guesses, merge suggestions and person tags (docs/ARCHITECTURE.md section 13).

Routes (all under /api, all require a logged-in session; writes need the CSRF header):
  GET    /people?q&filter=all|named|unnamed&sort=items|name|recent&cursor&limit -> PersonPage (keyset)
  POST   /people {name, face_ids?} -> PersonOut (an existing person with that name is reused)
  POST   /people/merge {target_id, source_ids} -> PersonOut
  GET    /people/{id} -> PersonOut;  PATCH {name?, cover_face_id?} -> PersonOut (renaming to a name another
         person has merges into that person and returns it);  DELETE -> Ok
  GET    /people/{id}/faces?cursor&limit -> FacePage;  GET /people/{id}/items?cursor&limit&vault -> ItemPage
  GET    /faces/unassigned -> FacePage;  POST /faces/assign -> AssignOut;  DELETE /faces/{id} -> Ok
  POST   /bodies/assign -> AssignOut;  DELETE /bodies/{id} -> Ok
  POST   /guesses/confirm {faces, bodies} -> BulkOut;  POST /guesses/reject -> BulkOut
  GET    /review/guesses?cursor&limit -> GuessPage
  GET    /suggestions?limit -> SuggestionList;  POST /suggestions/dismiss {a, b} -> Ok
  POST   /items/{uid}/tags {person_id | new_person_name} -> [PersonRef];  DELETE /items/{uid}/tags/{person_id}
  POST   /recluster -> Ok
Faces, bodies and counts in NSFW or trashed items are never shown here (person items take vault=1
with an unlocked session). Every write to faces, bodies or persons bumps faces_version; deleting a box
also bumps search_version (its torso embedding goes).
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app import storage
from app.analysis import cluster, guess
from app.api.common import (
    ITEM_CARD_COLUMNS,
    auto_cover,
    body_out,
    face_out,
    guess_out,
    item_card,
    person_ref,
    valid_cursor_scalar,
    visible_sql,
)
from app.api.deps import get_db, require_user, want_vault
from app.db import bump_faces_version, bump_search_version, tx
from app.schemas import (
    AssignBodiesIn,
    AssignFacesIn,
    AssignOut,
    BulkOut,
    DismissIn,
    FaceListItem,
    FacePage,
    GuessIdsIn,
    GuessItem,
    GuessPage,
    ItemPage,
    MergeIn,
    Ok,
    PersonCreate,
    PersonOut,
    PersonPage,
    PersonRef,
    PersonUpdate,
    Suggestion,
    SuggestionList,
    TagIn,
)
from app.urls import face_url, thumb_url
from app.util import clean_name, decode_cursor, encode_cursor, now_iso

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["people"], dependencies=[Depends(require_user)])

LIVE = "i.deleted_at IS NULL AND i.nsfw = 0"
_PRESENT = (
    "SELECT person_id, item_id FROM faces WHERE person_id IS NOT NULL UNION "
    "SELECT person_id, item_id FROM bodies WHERE person_id IS NOT NULL UNION "
    "SELECT person_id, item_id FROM item_tags"
)


# ------------------------------------------------------------------ helpers
def _like(q: str) -> str:
    esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{esc}%"


def _keyset(cursor: str | None, n: int) -> list[Any] | None:
    if not cursor:
        return None
    values = decode_cursor(cursor)
    if values is None or len(values) != n or not all(valid_cursor_scalar(v) for v in values):
        raise HTTPException(status_code=400, detail="Invalid cursor")
    return values


def _ids_sql(ids: list[int]) -> str:
    return ",".join("?" * len(ids))


def _person_row(conn: sqlite3.Connection, person_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT id, name, name_key, cover_face_id, created_at FROM persons WHERE id = ?",
                       (person_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return row


def _stats(conn: sqlite3.Connection, ids: list[int]) -> dict[int, tuple[int, int, int]]:
    """{person_id: (face_count, item_count, guess_count)} over live, non-NSFW items."""
    out: dict[int, list[int]] = {i: [0, 0, 0] for i in ids}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        ph = _ids_sql(chunk)
        for r in conn.execute(
            f"SELECT f.person_id, COUNT(*) FROM faces f JOIN items i ON i.id = f.item_id AND {LIVE} "
            f"WHERE f.person_id IN ({ph}) GROUP BY f.person_id", chunk,
        ):
            out[int(r[0])][0] = int(r[1])
        for r in conn.execute(
            f"SELECT p.person_id, COUNT(*) FROM ({_PRESENT}) p JOIN items i ON i.id = p.item_id AND {LIVE} "
            f"WHERE p.person_id IN ({ph}) GROUP BY p.person_id", chunk,
        ):
            out[int(r[0])][1] = int(r[1])
        for r in conn.execute(
            "SELECT g.pid, COUNT(*) FROM ("
            f"SELECT guess_person_id AS pid, item_id FROM faces WHERE guess_person_id IN ({ph}) AND person_id IS NULL "
            "UNION ALL "
            f"SELECT guess_person_id AS pid, item_id FROM bodies WHERE guess_person_id IN ({ph}) AND person_id IS NULL"
            f") g JOIN items i ON i.id = g.item_id AND {LIVE} GROUP BY g.pid", [*chunk, *chunk],
        ):
            out[int(r[0])][2] = int(r[1])
    return {k: (v[0], v[1], v[2]) for k, v in out.items()}


def _cover(conn: sqlite3.Connection, person_id: int, cover_face_id: int | None) -> str | None:
    if cover_face_id is not None:
        ok = conn.execute(
            f"SELECT 1 FROM faces f JOIN items i ON i.id = f.item_id AND {LIVE} WHERE f.id = ?", (cover_face_id,)
        ).fetchone()
        if ok is not None:
            return face_url(int(cover_face_id))
    fid = auto_cover(conn, person_id)
    return face_url(fid) if fid else None


def person_outs(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> list[PersonOut]:
    """PersonOut for persons rows (id, name, cover_face_id, created_at), counts over visible items only."""
    st = _stats(conn, [int(r["id"]) for r in rows]) if rows else {}
    out = []
    for r in rows:
        fc, ic, gc = st.get(int(r["id"]), (0, 0, 0))
        out.append(PersonOut(id=int(r["id"]), name=r["name"], cover_url=_cover(conn, int(r["id"]), r["cover_face_id"]),
                             cover_face_id=r["cover_face_id"],
                             face_count=fc, item_count=ic, guess_count=gc, created_at=r["created_at"]))
    return out


def _person_out(conn: sqlite3.Connection, person_id: int) -> PersonOut:
    return person_outs(conn, [_person_row(conn, person_id)])[0]


def _item_row(conn: sqlite3.Connection, uid: str) -> sqlite3.Row:
    row = conn.execute("SELECT id, uid, version, kind, nsfw, deleted_at FROM items WHERE uid = ?", (uid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return row


def _target_person(conn: sqlite3.Connection, person_id: int | None, new_name: str | None,
                   name_given: bool) -> int | None:
    """Resolve {person_id | new_person_name | neither}. Call inside a tx."""
    if person_id is not None and name_given:
        raise HTTPException(status_code=400, detail="Choose an existing person or a new name, not both")
    if person_id is not None:
        return int(_person_row(conn, person_id)["id"])
    if name_given:
        name = cluster.clean_person_name(new_name)
        if not name:
            raise HTTPException(status_code=400, detail="Enter a name")
        return cluster.find_or_create_person(conn, name)
    return None


def _items_of_person(conn: sqlite3.Connection, person_id: int) -> set[int]:
    return {int(r[0]) for r in conn.execute(
        "SELECT item_id FROM faces WHERE person_id = ? UNION SELECT item_id FROM bodies WHERE person_id = ? "
        "UNION SELECT item_id FROM item_tags WHERE person_id = ?",
        (person_id, person_id, person_id),
    )}


def _move_boxes(conn: sqlite3.Connection, table: str, ids: list[int], target: int | None) -> tuple[int, set[int]]:
    """Assign faces or bodies to target (None unassigns), locked as 'user'. Call inside a tx.
    Returns (rows updated, item ids). 404 when an id is unknown."""
    ids = sorted({int(i) for i in ids})
    ph = _ids_sql(ids)
    rows = conn.execute(f"SELECT id, item_id, person_id FROM {table} WHERE id IN ({ph})", ids).fetchall()
    if len(rows) != len(ids):
        raise HTTPException(status_code=404, detail="Some boxes no longer exist. Reload and try again.")
    previous = {int(r["person_id"]) for r in rows if r["person_id"] is not None}
    items = {int(r["item_id"]) for r in rows}
    clear = "guess_person_id = NULL, guess_score = NULL, guess_evidence = NULL"
    conn.execute(f"UPDATE {table} SET person_id = ?, assigned_by = 'user', {clear} WHERE id IN ({ph})", [target, *ids])
    now = now_iso()
    if table == "faces":
        # A body around the face follows it.
        conn.execute(f"UPDATE bodies SET person_id = ?, assigned_by = 'user', {clear} WHERE face_id IN ({ph})",
                     [target, *ids])
        conn.execute(
            f"UPDATE persons SET cover_face_id = NULL, updated_at = ? WHERE cover_face_id IN ({ph}) AND id IS NOT ?",
            [now, *ids, target],
        )
    if target is not None:
        conn.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (now, target))
    previous.discard(target)  # type: ignore[arg-type]
    cluster.prune_empty_persons(conn, previous)
    bump_faces_version(conn)
    return len(ids), items


def _tags_of(conn: sqlite3.Connection, item_id: int) -> list[PersonRef]:
    rows = conn.execute(
        "SELECT p.id, p.name, p.cover_face_id FROM item_tags t JOIN persons p ON p.id = t.person_id "
        "WHERE t.item_id = ? ORDER BY p.name_key IS NULL, p.name_key, p.id",
        (item_id,),
    ).fetchall()
    out = []
    for r in rows:
        ref = person_ref(conn, int(r["id"]), r["name"], r["cover_face_id"])
        if ref is not None:
            out.append(ref)
    return out


FACE_SELECT = (
    "SELECT f.id, f.person_id, f.assigned_by, f.det_score, f.size_px, f.quality, f.frame_time_s, f.created_at, "
    "f.guess_person_id, f.guess_score, f.guess_evidence, i.uid AS item_uid, i.kind AS item_kind, "
    "i.version AS item_version FROM faces f JOIN items i ON i.id = f.item_id"
)


def _face_item(conn: sqlite3.Connection, row: sqlite3.Row) -> FaceListItem:
    return FaceListItem(
        id=row["id"], item_uid=row["item_uid"], item_kind=row["item_kind"],
        thumb_url=thumb_url(row["item_uid"], int(row["item_version"])), url=face_url(int(row["id"])),
        det_score=row["det_score"], size_px=row["size_px"], quality=row["quality"],
        frame_time_s=row["frame_time_s"], person_id=row["person_id"], assigned_by=row["assigned_by"],
        guess=guess_out(conn, row), created_at=row["created_at"],
    )


def _after_people_change(conn: sqlite3.Connection, items: set[int], guesses: set[tuple[int, int | None]]) -> None:
    for pid, iid in sorted(guesses, key=lambda t: (t[0], -1 if t[1] is None else t[1])):
        try:
            guess.enqueue_guess(conn, pid, iid)
        except Exception as e:  # the user's action already happened; a missing follow-up is logged
            log.warning("could not queue guess job for person %s: %s", pid, e)
    guess.refresh_fts(conn, items)


# ------------------------------------------------------------------ people
@router.get("/people", response_model=PersonPage)
def list_people(
    q: str | None = Query(default=None, max_length=200),
    filter: Literal["all", "named", "unnamed"] = "all",
    sort: Literal["items", "name", "recent"] = "items",
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=100, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_db),
) -> PersonPage:
    conds = ["(x.item_count > 0 OR x.any_count = 0)"]
    params: list[Any] = []
    if filter == "named":
        conds.append("x.name IS NOT NULL")
    elif filter == "unnamed":
        conds.append("x.name IS NULL")
    needle = clean_name(q or "", 200).casefold()
    if needle:
        conds.append("(x.name_key LIKE ? ESCAPE '\\' OR ('unnamed #' || x.id) LIKE ? ESCAPE '\\')")
        params += [_like(needle), _like(needle)]
    base = (
        f"WITH present AS ({_PRESENT}), "
        f"vis AS (SELECT p.person_id, COUNT(*) AS n FROM present p JOIN items i ON i.id = p.item_id AND {LIVE} "
        "GROUP BY p.person_id), "
        "anyc AS (SELECT person_id, COUNT(*) AS n FROM present GROUP BY person_id), "
        "x AS (SELECT p.id, p.name, p.name_key, p.cover_face_id, p.created_at, COALESCE(vis.n, 0) AS item_count, "
        "COALESCE(anyc.n, 0) AS any_count, (p.name_key IS NULL) AS nf, COALESCE(p.name_key, '') AS nk "
        "FROM persons p LEFT JOIN vis ON vis.person_id = p.id LEFT JOIN anyc ON anyc.person_id = p.id) "
    )
    where = " AND ".join(conds)
    total = conn.execute(f"{base} SELECT COUNT(*) FROM x WHERE {where}", params).fetchone()[0]
    keyset_conds: list[str] = []
    kparams: list[Any] = []
    if sort == "items":
        order = "x.item_count DESC, x.id ASC"
        v = _keyset(cursor, 2)
        if v is not None:
            keyset_conds.append("(x.item_count < ? OR (x.item_count = ? AND x.id > ?))")
            kparams += [v[0], v[0], v[1]]
    elif sort == "name":
        order = "x.nf ASC, x.nk ASC, x.id ASC"
        v = _keyset(cursor, 3)
        if v is not None:
            keyset_conds.append("(x.nf > ? OR (x.nf = ? AND (x.nk > ? OR (x.nk = ? AND x.id > ?))))")
            kparams += [v[0], v[0], v[1], v[1], v[2]]
    else:
        order = "x.created_at DESC, x.id DESC"
        v = _keyset(cursor, 2)
        if v is not None:
            keyset_conds.append("(x.created_at < ? OR (x.created_at = ? AND x.id < ?))")
            kparams += [v[0], v[0], v[1]]
    full_where = " AND ".join([where, *keyset_conds])
    rows = conn.execute(f"{base} SELECT * FROM x WHERE {full_where} ORDER BY {order} LIMIT ?",
                        [*params, *kparams, limit + 1]).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    nxt = None
    if more and rows:
        last = rows[-1]
        if sort == "items":
            nxt = encode_cursor([int(last["item_count"]), int(last["id"])])
        elif sort == "name":
            nxt = encode_cursor([int(last["nf"]), last["nk"], int(last["id"])])
        else:
            nxt = encode_cursor([last["created_at"], int(last["id"])])
    return PersonPage(items=person_outs(conn, rows), next_cursor=nxt, total=int(total))


@router.post("/people", response_model=PersonOut)
def create_person(body: PersonCreate, conn: sqlite3.Connection = Depends(get_db)) -> PersonOut:
    name = cluster.clean_person_name(body.name)
    if not name:
        raise HTTPException(status_code=400, detail="Enter a name")
    items: set[int] = set()
    with tx(conn):
        pid = cluster.find_or_create_person(conn, name)
        assert pid is not None
        if body.face_ids:
            _, items = _move_boxes(conn, "faces", body.face_ids, pid)
        bump_faces_version(conn)
    _after_people_change(conn, items | _items_of_person(conn, pid), {(pid, None)})
    return _person_out(conn, pid)


@router.post("/people/merge", response_model=PersonOut)
def merge_people(body: MergeIn, conn: sqlite3.Connection = Depends(get_db)) -> PersonOut:
    sources = list(dict.fromkeys(int(s) for s in body.source_ids))
    if body.target_id in sources:
        raise HTTPException(status_code=400, detail="Cannot merge a person into themselves")
    ids = [body.target_id, *sources]
    found = conn.execute(f"SELECT COUNT(*) FROM persons WHERE id IN ({_ids_sql(ids)})", ids).fetchone()[0]
    if found != len(ids):
        raise HTTPException(status_code=404, detail="Person not found")
    try:
        cluster.merge_persons(conn, body.target_id, sources)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e) or "Cannot merge these people") from e
    _after_people_change(conn, _items_of_person(conn, body.target_id), {(body.target_id, None)})
    return _person_out(conn, body.target_id)


@router.get("/people/{person_id}", response_model=PersonOut)
def get_person(person_id: int, conn: sqlite3.Connection = Depends(get_db)) -> PersonOut:
    return _person_out(conn, person_id)


@router.patch("/people/{person_id}", response_model=PersonOut)
def update_person(person_id: int, body: PersonUpdate, conn: sqlite3.Connection = Depends(get_db)) -> PersonOut:
    fields = body.model_fields_set
    result = person_id
    renamed = False
    with tx(conn):
        _person_row(conn, person_id)
        now = now_iso()
        if "cover_face_id" in fields:
            if body.cover_face_id is not None:
                ok = conn.execute("SELECT 1 FROM faces WHERE id = ? AND person_id = ?",
                                  (body.cover_face_id, person_id)).fetchone()
                if ok is None:
                    raise HTTPException(status_code=400, detail="The cover must be one of this person's faces")
            conn.execute("UPDATE persons SET cover_face_id = ?, updated_at = ? WHERE id = ?",
                         (body.cover_face_id, now, person_id))
        if "name" in fields:
            name = cluster.clean_person_name(body.name)
            other = cluster.find_person_by_name(conn, name, exclude_id=person_id) if name else None
            if other is not None:
                cluster.merge_persons(conn, other, [person_id])
                result = other
            else:
                conn.execute("UPDATE persons SET name = ?, name_key = ?, updated_at = ? WHERE id = ?",
                             (name, name.casefold() if name else None, now, person_id))
            renamed = name is not None
        bump_faces_version(conn)
    if "name" in fields:
        _after_people_change(conn, _items_of_person(conn, result), {(result, None)} if renamed else set())
    return _person_out(conn, result)


@router.delete("/people/{person_id}", response_model=Ok)
def delete_person(person_id: int, conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    with tx(conn):
        _person_row(conn, person_id)
        items = _items_of_person(conn, person_id)
        cluster.clear_guesses_for(conn, [person_id])
        conn.execute("UPDATE faces SET person_id = NULL, assigned_by = 'user' WHERE person_id = ?", (person_id,))
        conn.execute("UPDATE bodies SET person_id = NULL, assigned_by = 'user' WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        bump_faces_version(conn)
    guess.refresh_fts(conn, items)
    return Ok()


@router.get("/people/{person_id}/faces", response_model=FacePage)
def person_faces(
    person_id: int,
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=200, ge=1, le=1000),
    conn: sqlite3.Connection = Depends(get_db),
) -> FacePage:
    _person_row(conn, person_id)
    conds = ["f.person_id = ?", LIVE]
    params: list[Any] = [person_id]
    total = conn.execute(f"SELECT COUNT(*) FROM faces f JOIN items i ON i.id = f.item_id WHERE {' AND '.join(conds)}",
                         params).fetchone()[0]
    v = _keyset(cursor, 2)
    if v is not None:  # ORDER BY quality DESC, id ASC
        conds.append("(f.quality < ? OR (f.quality = ? AND f.id > ?))")
        params += [v[0], v[0], v[1]]
    rows = conn.execute(f"{FACE_SELECT} WHERE {' AND '.join(conds)} ORDER BY f.quality DESC, f.id ASC LIMIT ?",
                        [*params, limit + 1]).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    nxt = encode_cursor([rows[-1]["quality"], int(rows[-1]["id"])]) if more and rows else None
    return FacePage(items=[_face_item(conn, r) for r in rows], next_cursor=nxt, total=int(total))


@router.get("/people/{person_id}/items", response_model=ItemPage)
def person_items(
    person_id: int,
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=60, ge=1, le=200),
    vault: bool = Depends(want_vault),
    conn: sqlite3.Connection = Depends(get_db),
) -> ItemPage:
    _person_row(conn, person_id)
    conds = [visible_sql(vault=vault, misclicks="exclude", dups="all"),
             "i.id IN (SELECT item_id FROM faces WHERE person_id = ? UNION SELECT item_id FROM bodies WHERE person_id = ? "
             "UNION SELECT item_id FROM item_tags WHERE person_id = ?)"]
    params: list[Any] = [person_id, person_id, person_id]
    total = conn.execute(f"SELECT COUNT(*) FROM items i WHERE {' AND '.join(conds)}", params).fetchone()[0]
    sort_key = "COALESCE(i.taken_at, i.created_at)"
    v = _keyset(cursor, 2)
    if v is not None:
        conds.append(f"({sort_key} < ? OR ({sort_key} = ? AND i.id < ?))")
        params += [v[0], v[0], v[1]]
    rows = conn.execute(
        f"SELECT {ITEM_CARD_COLUMNS}, {sort_key} AS sort_key FROM items i WHERE {' AND '.join(conds)} "
        f"ORDER BY {sort_key} DESC, i.id DESC LIMIT ?",
        [*params, limit + 1],
    ).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    nxt = encode_cursor([rows[-1]["sort_key"], int(rows[-1]["id"])]) if more and rows else None
    return ItemPage(items=[item_card(r) for r in rows], next_cursor=nxt, total=int(total))


# ------------------------------------------------------------------ faces and bodies
@router.get("/faces/unassigned", response_model=FacePage)
def unassigned_faces(
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=200, ge=1, le=1000),
    conn: sqlite3.Connection = Depends(get_db),
) -> FacePage:
    conds = ["f.person_id IS NULL", LIVE]
    params: list[Any] = []
    total = conn.execute(f"SELECT COUNT(*) FROM faces f JOIN items i ON i.id = f.item_id WHERE {' AND '.join(conds)}",
                         params).fetchone()[0]
    v = _keyset(cursor, 1)
    if v is not None:
        conds.append("f.id < ?")
        params.append(v[0])
    rows = conn.execute(f"{FACE_SELECT} WHERE {' AND '.join(conds)} ORDER BY f.id DESC LIMIT ?",
                        [*params, limit + 1]).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    nxt = encode_cursor([int(rows[-1]["id"])]) if more and rows else None
    return FacePage(items=[_face_item(conn, r) for r in rows], next_cursor=nxt, total=int(total))


def _assign(conn: sqlite3.Connection, table: str, ids: list[int], person_id: int | None, new_name: str | None,
            name_given: bool) -> AssignOut:
    with tx(conn):
        target = _target_person(conn, person_id, new_name, name_given)
        updated, items = _move_boxes(conn, table, ids, target)
    guesses = {(target, i) for i in items} if target is not None else set()
    _after_people_change(conn, items, guesses)  # type: ignore[arg-type]
    return AssignOut(updated=updated, person=_person_out(conn, target) if target is not None else None)


@router.post("/faces/assign", response_model=AssignOut)
def assign_faces(body: AssignFacesIn, conn: sqlite3.Connection = Depends(get_db)) -> AssignOut:
    return _assign(conn, "faces", body.face_ids, body.person_id, body.new_person_name,
                   "new_person_name" in body.model_fields_set and body.new_person_name is not None)


@router.post("/bodies/assign", response_model=AssignOut)
def assign_bodies(body: AssignBodiesIn, conn: sqlite3.Connection = Depends(get_db)) -> AssignOut:
    return _assign(conn, "bodies", body.body_ids, body.person_id, body.new_person_name,
                   "new_person_name" in body.model_fields_set and body.new_person_name is not None)


def _delete_box(conn: sqlite3.Connection, table: str, box_id: int) -> Ok:
    count_col = "face_count" if table == "faces" else "body_count"
    with tx(conn):
        row = conn.execute(f"SELECT id, item_id, person_id FROM {table} WHERE id = ?", (box_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Face not found" if table == "faces" else "Body not found")
        conn.execute(f"DELETE FROM {table} WHERE id = ?", (box_id,))
        conn.execute(f"UPDATE items SET {count_col} = (SELECT COUNT(*) FROM {table} WHERE item_id = ?) WHERE id = ?",
                     (row["item_id"], row["item_id"]))
        if row["person_id"] is not None:
            cluster.prune_empty_persons(conn, {int(row["person_id"])})
        bump_faces_version(conn)
        bump_search_version(conn)
    storage.safe_unlink(storage.face_crop_path(box_id) if table == "faces" else storage.body_crop_path(box_id))
    guess.refresh_fts(conn, {int(row["item_id"])})
    return Ok()


@router.delete("/faces/{face_id}", response_model=Ok)
def delete_face(face_id: int, conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    return _delete_box(conn, "faces", face_id)


@router.delete("/bodies/{body_id}", response_model=Ok)
def delete_body(body_id: int, conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    return _delete_box(conn, "bodies", body_id)


# ------------------------------------------------------------------ guesses
@router.post("/guesses/confirm", response_model=BulkOut)
def confirm_guesses(body: GuessIdsIn, conn: sqlite3.Connection = Depends(get_db)) -> BulkOut:
    return BulkOut(updated=guess.confirm(conn, body.faces, body.bodies))


@router.post("/guesses/reject", response_model=BulkOut)
def reject_guesses(body: GuessIdsIn, conn: sqlite3.Connection = Depends(get_db)) -> BulkOut:
    return BulkOut(updated=guess.reject(conn, body.faces, body.bodies))


@router.get("/review/guesses", response_model=GuessPage)
def review_guesses(
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=60, ge=1, le=200),
    person: int | None = Query(default=None, description="only guesses naming this person"),
    conn: sqlite3.Connection = Depends(get_db),
) -> GuessPage:
    person_sql = " AND g.guess_person_id = ?" if person is not None else ""
    base = (
        "SELECT g.* FROM ("
        "SELECT 'face' AS kind, f.id AS box_id, f.item_id, f.guess_person_id, COALESCE(f.guess_score, 0) AS score FROM faces f "
        "WHERE f.guess_person_id IS NOT NULL AND f.person_id IS NULL UNION ALL "
        "SELECT 'body' AS kind, b.id AS box_id, b.item_id, b.guess_person_id, COALESCE(b.guess_score, 0) AS score FROM bodies b "
        "WHERE b.guess_person_id IS NOT NULL AND b.person_id IS NULL"
        f") g JOIN items i ON i.id = g.item_id AND {LIVE}{person_sql}"
    )
    base_params: list[Any] = [person] if person is not None else []
    total = conn.execute(f"SELECT COUNT(*) FROM ({base})", base_params).fetchone()[0]
    conds: list[str] = []
    params: list[Any] = list(base_params)
    v = _keyset(cursor, 3)
    if v is not None:  # ORDER BY score DESC, kind ASC, box_id ASC
        conds.append("(g.score < ? OR (g.score = ? AND (g.kind > ? OR (g.kind = ? AND g.box_id > ?))))")
        params += [v[0], v[0], v[1], v[1], v[2]]
    where = f" WHERE {' AND '.join(conds)}" if conds else ""
    rows = conn.execute(f"{base}{where} ORDER BY g.score DESC, g.kind ASC, g.box_id ASC LIMIT ?",
                        [*params, limit + 1]).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    nxt = encode_cursor([rows[-1]["score"], rows[-1]["kind"], int(rows[-1]["box_id"])]) if more and rows else None
    item_ids = sorted({int(r["item_id"]) for r in rows})
    cards = {}
    if item_ids:
        for r in conn.execute(f"SELECT {ITEM_CARD_COLUMNS} FROM items i WHERE i.id IN ({_ids_sql(item_ids)})", item_ids):
            cards[int(r["id"])] = item_card(r)
    items: list[GuessItem] = []
    for r in rows:
        table = "faces" if r["kind"] == "face" else "bodies"
        box_row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (r["box_id"],)).fetchone()
        if box_row is None or int(r["item_id"]) not in cards:
            continue
        g = guess_out(conn, box_row)
        if g is None:
            continue
        box = face_out(conn, box_row) if r["kind"] == "face" else body_out(conn, box_row)
        items.append(GuessItem(kind=r["kind"], id=int(r["box_id"]), item=cards[int(r["item_id"])], box=box, guess=g))
    return GuessPage(items=items, next_cursor=nxt, total=int(total))


# ------------------------------------------------------------------ suggestions
@router.get("/suggestions", response_model=SuggestionList)
def suggestions(
    limit: int = Query(default=30, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_db),
) -> SuggestionList:
    pairs = cluster.merge_suggestions(conn, limit)
    if not pairs:
        return SuggestionList(items=[])
    ids = sorted({int(a) for a, _, _ in pairs} | {int(b) for _, b, _ in pairs})
    ph = _ids_sql(ids)
    rows = {int(r["id"]): r for r in conn.execute(
        f"SELECT id, name, cover_face_id, created_at FROM persons WHERE id IN ({ph})", ids)}
    outs = {o.id: o for o in person_outs(conn, list(rows.values()))}
    samples: dict[int, list[str]] = {}
    for r in conn.execute(
        "SELECT pid, fid FROM ("
        "  SELECT f.person_id AS pid, f.id AS fid, "
        "         ROW_NUMBER() OVER (PARTITION BY f.person_id ORDER BY f.quality DESC, f.id) AS rn "
        f"  FROM faces f JOIN items i ON i.id = f.item_id AND {LIVE} WHERE f.person_id IN ({ph})"
        ") WHERE rn <= 4 ORDER BY pid, rn",
        ids,
    ):
        samples.setdefault(int(r["pid"]), []).append(face_url(int(r["fid"])))
    items: list[Suggestion] = []
    for a, b, sim in pairs:
        oa, ob = outs.get(int(a)), outs.get(int(b))
        if oa is None or ob is None or oa.item_count == 0 or ob.item_count == 0:
            continue
        items.append(Suggestion(a=oa, b=ob, similarity=float(sim), a_samples=samples.get(oa.id, []),
                                b_samples=samples.get(ob.id, [])))
    return SuggestionList(items=items)


@router.post("/suggestions/dismiss", response_model=Ok)
def dismiss_suggestion(body: DismissIn, conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    if body.a == body.b:
        raise HTTPException(status_code=400, detail="Pick two different people")
    lo, hi = sorted((int(body.a), int(body.b)))
    found = conn.execute("SELECT COUNT(*) FROM persons WHERE id IN (?, ?)", (lo, hi)).fetchone()[0]
    if found != 2:
        raise HTTPException(status_code=404, detail="Person not found")
    with tx(conn):
        conn.execute("INSERT OR IGNORE INTO merge_dismissals(person_a, person_b, created_at) VALUES (?, ?, ?)",
                     (lo, hi, now_iso()))
    return Ok()


# ------------------------------------------------------------------ tags
@router.post("/items/{uid}/tags", response_model=list[PersonRef])
def add_tag(uid: str, body: TagIn, conn: sqlite3.Connection = Depends(get_db)) -> list[PersonRef]:
    name_given = body.new_person_name is not None
    if (body.person_id is None) == (not name_given):
        raise HTTPException(status_code=400, detail="Choose an existing person or a new name")
    with tx(conn):
        item = _item_row(conn, uid)
        item_id = int(item["id"])
        pid = _target_person(conn, body.person_id, body.new_person_name, name_given)
        assert pid is not None
        on_item = conn.execute(
            "SELECT 1 FROM faces WHERE item_id = ? AND person_id = ? UNION ALL "
            "SELECT 1 FROM bodies WHERE item_id = ? AND person_id = ? LIMIT 1",
            (item_id, pid, item_id, pid),
        ).fetchone()
        if on_item is None:
            conn.execute("INSERT OR IGNORE INTO item_tags(item_id, person_id, source, created_at) VALUES (?, ?, 'user', ?)",
                         (item_id, pid, now_iso()))
        conn.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (now_iso(), pid))
        bump_faces_version(conn)
    _after_people_change(conn, {item_id}, {(pid, item_id)})
    return _tags_of(conn, item_id)


@router.delete("/items/{uid}/tags/{person_id}", response_model=list[PersonRef])
def remove_tag(uid: str, person_id: int, conn: sqlite3.Connection = Depends(get_db)) -> list[PersonRef]:
    with tx(conn):
        item_id = int(_item_row(conn, uid)["id"])
        cur = conn.execute("DELETE FROM item_tags WHERE item_id = ? AND person_id = ?", (item_id, person_id))
        if cur.rowcount:
            cluster.prune_empty_persons(conn, {person_id})
            bump_faces_version(conn)
    guess.refresh_fts(conn, {item_id})
    return _tags_of(conn, item_id)


# ------------------------------------------------------------------ recluster
@router.post("/recluster", response_model=Ok)
def recluster(conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    from app import jobs

    jobs.enqueue(conn, "recluster", {}, unique=True)
    return Ok()
