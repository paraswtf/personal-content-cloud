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
with an unlocked session). With the vault locked, acting on a box or the tags of an NSFW item is
403 vault_locked, and a person seen only in vault items is 404 (as in the list). Every write to faces,
bodies or persons bumps faces_version; deleting a box also bumps search_version (its torso embedding goes).
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app import storage
from app.analysis import cluster, guess
from app.analysis import pets as pets_mod
from app.api.common import (
    ITEM_CARD_COLUMNS,
    animal_out,
    body_out,
    box_person_ids,
    face_out,
    guess_out,
    item_card,
    page_sql,
    person_refs,
    pet_cover_ids,
    valid_cursor_scalar,
    visible_sql,
)
from app.api.deps import get_db, require_user, want_vault
from app.auth.sessions import SessionInfo
from app.db import bump_faces_version, bump_search_version, tx
from app.schemas import (
    AnimalListItem,
    AnimalPage,
    AssignAnimalsIn,
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
    PersonSpeciesIn,
    PersonUpdate,
    SpeciesIn,
    Suggestion,
    SuggestionList,
    TagIn,
)
from app.urls import animal_url, face_url, thumb_url
from app.util import clean_name, decode_cursor, encode_cursor, now_iso

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["people"], dependencies=[Depends(require_user)])

LIVE = "i.deleted_at IS NULL AND i.nsfw = 0"
_PRESENT = (
    "SELECT person_id, item_id FROM faces WHERE person_id IS NOT NULL UNION "
    "SELECT person_id, item_id FROM bodies WHERE person_id IS NOT NULL UNION "
    "SELECT person_id, item_id FROM animals WHERE person_id IS NOT NULL UNION "
    "SELECT person_id, item_id FROM item_tags"
)
PERSON_COLUMNS = "id, name, name_key, cover_face_id, created_at, kind, species"


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
    row = conn.execute(f"SELECT {PERSON_COLUMNS} FROM persons WHERE id = ?", (person_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return row


def _vault_locked() -> HTTPException:
    return HTTPException(status_code=403, detail="vault_locked")


def _check_items_vault(conn: sqlite3.Connection, session: SessionInfo, item_ids: set[int] | list[int]) -> None:
    """403 vault_locked when the vault is locked and any of these items is NSFW."""
    ids = sorted({int(i) for i in item_ids})
    if session.vault_unlocked or not ids:
        return
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        if conn.execute(f"SELECT 1 FROM items WHERE id IN ({_ids_sql(chunk)}) AND nsfw = 1 LIMIT 1", chunk).fetchone():
            raise _vault_locked()


def _check_boxes_vault(conn: sqlite3.Connection, session: SessionInfo, table: str, ids: list[int]) -> None:
    """403 vault_locked when the vault is locked and any of these faces or bodies is in an NSFW item."""
    box_ids = sorted({int(i) for i in ids})
    if session.vault_unlocked or not box_ids:
        return
    for start in range(0, len(box_ids), 500):
        chunk = box_ids[start:start + 500]
        if conn.execute(
            f"SELECT 1 FROM {table} b JOIN items i ON i.id = b.item_id AND i.nsfw = 1 WHERE b.id IN ({_ids_sql(chunk)}) LIMIT 1",
            chunk,
        ).fetchone():
            raise _vault_locked()


def _hidden_person(conn: sqlite3.Connection, session: SessionInfo, person_id: int) -> bool:
    """A locked session does not see people whose only items are in the vault (the list hides them too)."""
    if session.vault_unlocked:
        return False
    row = conn.execute(
        "SELECT COALESCE(SUM(i.deleted_at IS NULL AND i.nsfw = 0), 0), COALESCE(SUM(i.nsfw = 1), 0) FROM items i "
        "WHERE i.id IN (SELECT item_id FROM faces WHERE person_id = ? UNION SELECT item_id FROM bodies WHERE person_id = ? "
        "UNION SELECT item_id FROM item_tags WHERE person_id = ?)",
        (person_id, person_id, person_id),
    ).fetchone()
    return int(row[0]) == 0 and int(row[1]) > 0


def _visible_person_row(conn: sqlite3.Connection, session: SessionInfo, person_id: int) -> sqlite3.Row:
    row = _person_row(conn, person_id)
    if _hidden_person(conn, session, int(row["id"])):
        raise HTTPException(status_code=404, detail="Person not found")
    return row


def _stats(conn: sqlite3.Connection, ids: list[int]) -> dict[int, tuple[int, int, int, int, int]]:
    """{person_id: (face_count, item_count, guess_count, animal_count, confirmed)} over live,
    non-NSFW items; confirmed = faces and crops the owner assigned."""
    out: dict[int, list[int]] = {i: [0, 0, 0, 0, 0] for i in ids}
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
        for r in conn.execute(
            f"SELECT a.person_id, COUNT(*) FROM animals a JOIN items i ON i.id = a.item_id AND {LIVE} "
            f"WHERE a.person_id IN ({ph}) GROUP BY a.person_id", chunk,
        ):
            out[int(r[0])][3] = int(r[1])
        for r in conn.execute(
            "SELECT pid, COUNT(*) FROM ("
            f"SELECT f.person_id AS pid FROM faces f JOIN items i ON i.id = f.item_id AND {LIVE} "
            f"WHERE f.person_id IN ({ph}) AND f.assigned_by = 'user' UNION ALL "
            f"SELECT a.person_id AS pid FROM animals a JOIN items i ON i.id = a.item_id AND {LIVE} "
            f"WHERE a.person_id IN ({ph}) AND a.assigned_by = 'user') GROUP BY pid", [*chunk, *chunk],
        ):
            out[int(r[0])][4] = int(r[1])
    return {k: (v[0], v[1], v[2], v[3], v[4]) for k, v in out.items()}


def _covers(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> dict[int, str | None]:
    """_cover for many persons rows with a few queries: the chosen cover when it is still visible,
    else the best visible face (the same order as common.auto_cover)."""
    out: dict[int, str | None] = {}
    chosen = {int(r["id"]): int(r["cover_face_id"]) for r in rows if r["cover_face_id"] is not None}
    fids = sorted(set(chosen.values()))
    visible: set[int] = set()
    for start in range(0, len(fids), 500):
        chunk = fids[start:start + 500]
        visible.update(int(r[0]) for r in conn.execute(
            f"SELECT f.id FROM faces f JOIN items i ON i.id = f.item_id AND {LIVE} WHERE f.id IN ({_ids_sql(chunk)})", chunk))
    for pid, fid in chosen.items():
        if fid in visible:
            out[pid] = face_url(fid)
    pets = sorted({int(r["id"]) for r in rows if r["kind"] == "pet"} - out.keys())
    for pid, aid in pet_cover_ids(conn, pets).items():
        out[pid] = animal_url(aid)
    rest = sorted({int(r["id"]) for r in rows} - out.keys() - set(pets))
    for start in range(0, len(rest), 500):
        chunk = rest[start:start + 500]
        for r in conn.execute(
            "SELECT pid, fid FROM ("
            "  SELECT f.person_id AS pid, f.id AS fid, "
            "         ROW_NUMBER() OVER (PARTITION BY f.person_id ORDER BY f.quality DESC, f.id) AS rn "
            f"  FROM faces f JOIN items i ON i.id = f.item_id AND {LIVE} WHERE f.person_id IN ({_ids_sql(chunk)})"
            ") WHERE rn = 1",
            chunk,
        ):
            out[int(r["pid"])] = face_url(int(r["fid"]))
    return out


def person_outs(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> list[PersonOut]:
    """PersonOut for persons rows (PERSON_COLUMNS), counts over visible items only."""
    st = _stats(conn, [int(r["id"]) for r in rows]) if rows else {}
    covers = _covers(conn, rows) if rows else {}
    out = []
    for r in rows:
        fc, ic, gc, ac, cf = st.get(int(r["id"]), (0, 0, 0, 0, 0))
        pet = r["kind"] == "pet"
        out.append(PersonOut(id=int(r["id"]), name=r["name"], cover_url=covers.get(int(r["id"])),
                             cover_face_id=None if pet else r["cover_face_id"],
                             face_count=fc, item_count=ic, guess_count=gc, created_at=r["created_at"],
                             kind=r["kind"] or "person", species=r["species"] if pet else None, animal_count=ac,
                             confirmed_faces=cf))
    return out


def _person_out(conn: sqlite3.Connection, person_id: int) -> PersonOut:
    return person_outs(conn, [_person_row(conn, person_id)])[0]


def _item_row(conn: sqlite3.Connection, uid: str) -> sqlite3.Row:
    row = conn.execute("SELECT id, uid, version, kind, nsfw, deleted_at FROM items WHERE uid = ?", (uid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return row


def _target_person(conn: sqlite3.Connection, person_id: int | None, new_name: str | None,
                   name_given: bool, session: SessionInfo, species: str | None = None) -> int | None:
    """Resolve {person_id | new_person_name | neither}. With `species` the target is a pet identity of
    that species (section 22): an existing one must match, a new name creates one. Call inside a tx."""
    if person_id is not None and name_given:
        raise HTTPException(status_code=400, detail="Choose an existing person or a new name, not both")
    if person_id is not None:
        row = _visible_person_row(conn, session, person_id)
        if species is not None:
            if row["kind"] != "pet":
                raise HTTPException(status_code=400, detail="That is a person, not a pet")
            if row["species"] and row["species"] != species:
                raise HTTPException(status_code=400, detail=f"That pet is a {row['species']}, this box is a {species}")
        elif row["kind"] == "pet":
            raise HTTPException(status_code=400, detail="That is a pet, not a person")
        return int(row["id"])
    if name_given:
        name = cluster.clean_person_name(new_name)
        if not name:
            raise HTTPException(status_code=400, detail="Enter a name")
        if species is not None:
            return cluster.find_or_create_person(conn, name, kind="pet", species=species)
        return cluster.find_or_create_person(conn, name)
    return None


def _items_of_person(conn: sqlite3.Connection, person_id: int) -> set[int]:
    return {int(r[0]) for r in conn.execute(
        "SELECT item_id FROM faces WHERE person_id = ? UNION SELECT item_id FROM bodies WHERE person_id = ? "
        "UNION SELECT item_id FROM animals WHERE person_id = ? UNION SELECT item_id FROM item_tags WHERE person_id = ?",
        (person_id, person_id, person_id, person_id),
    )}


def _move_boxes(conn: sqlite3.Connection, table: str, ids: list[int], target: int | None) -> tuple[int, set[int]]:
    """Assign faces, bodies or animal boxes to target (None unassigns), locked as 'user'. Call inside
    a tx. Returns (rows updated, item ids). 404 when an id is unknown."""
    ids = sorted({int(i) for i in ids})
    ph = _ids_sql(ids)
    rows = conn.execute(f"SELECT id, item_id, person_id FROM {table} WHERE id IN ({ph})", ids).fetchall()
    if len(rows) != len(ids):
        raise HTTPException(status_code=404, detail="Some boxes no longer exist. Reload and try again.")
    previous = {int(r["person_id"]) for r in rows if r["person_id"] is not None}
    if table == "bodies":
        _drop_body_tags(conn, [r for r in rows if r["person_id"] is not None and r["person_id"] != target])
    items = {int(r["item_id"]) for r in rows}
    clear = "" if table == "animals" else ", guess_person_id = NULL, guess_score = NULL, guess_evidence = NULL"
    conn.execute(f"UPDATE {table} SET person_id = ?, assigned_by = 'user'{clear} WHERE id IN ({ph})", [target, *ids])
    now = now_iso()
    if table == "faces":
        # A body around the face follows it.
        conn.execute(f"UPDATE bodies SET person_id = ?, assigned_by = 'user'{clear} WHERE face_id IN ({ph})",
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


def _drop_body_tags(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> None:
    """A confirmed faceless body tagged its item with source 'guess' (guess.confirm): the tag goes
    with the body's assignment. rows: bodies (item_id, person_id) losing their person."""
    for r in rows:
        conn.execute("DELETE FROM item_tags WHERE item_id = ? AND person_id = ? AND source = 'guess'",
                     (r["item_id"], r["person_id"]))


def _tags_of(conn: sqlite3.Connection, item_id: int) -> list[PersonRef]:
    rows = conn.execute(
        "SELECT p.id FROM item_tags t JOIN persons p ON p.id = t.person_id "
        "WHERE t.item_id = ? ORDER BY p.name_key IS NULL, p.name_key, p.id",
        (item_id,),
    ).fetchall()
    refs = person_refs(conn, [int(r["id"]) for r in rows])
    return [refs[int(r["id"])] for r in rows if int(r["id"]) in refs]


FACE_SELECT = (
    "SELECT f.id, f.person_id, f.assigned_by, f.det_score, f.size_px, f.quality, f.frame_time_s, f.created_at, "
    "f.guess_person_id, f.guess_score, f.guess_evidence, i.uid AS item_uid, i.kind AS item_kind, "
    "i.version AS item_version FROM faces f JOIN items i ON i.id = f.item_id"
)


def _face_items(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> list[FaceListItem]:
    """FaceListItems of a page of FACE_SELECT rows; the guessed people are looked up once."""
    refs = person_refs(conn, box_person_ids(rows))
    return [_face_item(conn, r, refs) for r in rows]


def _face_item(conn: sqlite3.Connection, row: sqlite3.Row, refs: dict[int, PersonRef] | None = None) -> FaceListItem:
    return FaceListItem(
        id=row["id"], item_uid=row["item_uid"], item_kind=row["item_kind"],
        thumb_url=thumb_url(row["item_uid"], int(row["item_version"])), url=face_url(int(row["id"])),
        det_score=row["det_score"], size_px=row["size_px"], quality=row["quality"],
        frame_time_s=row["frame_time_s"], person_id=row["person_id"], assigned_by=row["assigned_by"],
        guess=guess_out(conn, row, refs), created_at=row["created_at"],
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
    kind: Literal["person", "pet", "all"] = "all",
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=100, ge=1, le=500),
    conn: sqlite3.Connection = Depends(get_db),
) -> PersonPage:
    conds = ["(x.item_count > 0 OR x.any_count = 0)"]
    params: list[Any] = []
    if kind != "all":
        conds.append("x.kind = ?")
        params.append(kind)
    if filter == "named":
        conds.append("x.name IS NOT NULL")
    elif filter == "unnamed":
        conds.append("x.name IS NULL")
    needle = clean_name(q or "", 200).casefold()
    if needle:
        conds.append("(x.name_key LIKE ? ESCAPE '\\' OR ('unnamed #' || x.id) LIKE ? ESCAPE '\\')")
        params += [_like(needle), _like(needle)]
    # One pass over the presence union gives both counts (items always exist: foreign keys cascade).
    base = (
        f"WITH present AS ({_PRESENT}), "
        "cnt AS (SELECT p.person_id, COUNT(*) AS any_n, "
        "SUM(i.deleted_at IS NULL AND i.nsfw = 0) AS vis_n FROM present p JOIN items i ON i.id = p.item_id "
        "GROUP BY p.person_id), "
        "x AS (SELECT p.id, p.name, p.name_key, p.cover_face_id, p.created_at, p.kind, p.species, "
        "COALESCE(cnt.vis_n, 0) AS item_count, "
        "COALESCE(cnt.any_n, 0) AS any_count, (p.name_key IS NULL) AS nf, COALESCE(p.name_key, '') AS nk "
        "FROM persons p LEFT JOIN cnt ON cnt.person_id = p.id) "
    )
    where = " AND ".join(conds)
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
    # The total (without the keyset) comes from the same materialisation as the page.
    keyset_where = f" WHERE {' AND '.join(keyset_conds)}" if keyset_conds else ""
    rows = conn.execute(
        f"{base} SELECT * FROM (SELECT x.*, COUNT(*) OVER () AS total_n FROM x WHERE {where}) x{keyset_where} "
        f"ORDER BY {order} LIMIT ?",
        [*params, *kparams, limit + 1],
    ).fetchall()
    if rows:
        total = int(rows[0]["total_n"])
    else:
        total = conn.execute(f"{base} SELECT COUNT(*) FROM x WHERE {where}", params).fetchone()[0]
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
def create_person(body: PersonCreate, conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> PersonOut:
    name = cluster.clean_person_name(body.name)
    if not name:
        raise HTTPException(status_code=400, detail="Enter a name")
    items: set[int] = set()
    _check_boxes_vault(conn, session, "faces", body.face_ids or [])
    with tx(conn):
        pid = cluster.find_or_create_person(conn, name)
        assert pid is not None
        if body.face_ids:
            _, items = _move_boxes(conn, "faces", body.face_ids, pid)
        bump_faces_version(conn)
    _after_people_change(conn, items | _items_of_person(conn, pid), {(pid, None)})
    return _person_out(conn, pid)


@router.post("/people/merge", response_model=PersonOut)
def merge_people(body: MergeIn, conn: sqlite3.Connection = Depends(get_db),
                 session: SessionInfo = Depends(require_user)) -> PersonOut:
    sources = list(dict.fromkeys(int(s) for s in body.source_ids))
    if body.target_id in sources:
        raise HTTPException(status_code=400, detail="Cannot merge a person into themselves")
    ids = [body.target_id, *sources]
    found = conn.execute(f"SELECT COUNT(*) FROM persons WHERE id IN ({_ids_sql(ids)})", ids).fetchone()[0]
    if found != len(ids) or any(_hidden_person(conn, session, int(i)) for i in ids):
        raise HTTPException(status_code=404, detail="Person not found")
    try:
        _, followers = cluster.merge_and_follow(conn, body.target_id, sources)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e) or "Cannot merge these people") from e
    _after_people_change(conn, _items_of_person(conn, body.target_id), {(body.target_id, None)})
    out = _person_out(conn, body.target_id)
    out.followed = len(followers)
    return out


@router.get("/people/{person_id}", response_model=PersonOut)
def get_person(person_id: int, conn: sqlite3.Connection = Depends(get_db),
               session: SessionInfo = Depends(require_user)) -> PersonOut:
    return person_outs(conn, [_visible_person_row(conn, session, person_id)])[0]


@router.patch("/people/{person_id}", response_model=PersonOut)
def update_person(person_id: int, body: PersonUpdate, conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> PersonOut:
    fields = body.model_fields_set
    result = person_id
    renamed = False
    followed = 0
    with tx(conn):
        row = _visible_person_row(conn, session, person_id)
        now = now_iso()
        if row["kind"] == "pet":
            fields = fields - {"cover_face_id"}  # pets have no face cover: the best crop is the cover
        if "cover_face_id" in fields:
            if body.cover_face_id is not None:
                ok = conn.execute(
                    "SELECT i.deleted_at IS NULL AND i.nsfw = 0 AS usable FROM faces f JOIN items i ON i.id = f.item_id "
                    "WHERE f.id = ? AND f.person_id = ?",
                    (body.cover_face_id, person_id)).fetchone()
                if ok is None:
                    raise HTTPException(status_code=400, detail="The cover must be one of this person's faces")
                if not ok["usable"]:
                    raise HTTPException(status_code=422,
                                        detail="The cover must be a face from a photo in the library, "
                                               "not in the trash or the vault")
            conn.execute("UPDATE persons SET cover_face_id = ?, updated_at = ? WHERE id = ?",
                         (body.cover_face_id, now, person_id))
        if "name" in fields:
            name = cluster.clean_person_name(body.name)
            other = cluster.find_person_by_name(conn, name, exclude_id=person_id, kind=row["kind"] or "person") if name else None
            if other is not None and row["kind"] == "pet":
                other_row = _person_row(conn, other)
                if other_row["species"] and row["species"] and other_row["species"] != row["species"]:
                    other = None  # a cat and a dog may share a name
            if other is not None:
                _, followers = cluster.merge_and_follow(conn, other, [person_id])
                followed = len(followers)
                result = other
            else:
                conn.execute("UPDATE persons SET name = ?, name_key = ?, updated_at = ? WHERE id = ?",
                             (name, name.casefold() if name else None, now, person_id))
            renamed = name is not None
        bump_faces_version(conn)
    if "name" in fields:
        _after_people_change(conn, _items_of_person(conn, result), {(result, None)} if renamed else set())
    out = _person_out(conn, result)
    out.followed = followed
    return out


@router.delete("/people/{person_id}", response_model=Ok)
def delete_person(person_id: int, conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> Ok:
    with tx(conn):
        _visible_person_row(conn, session, person_id)
        items = _items_of_person(conn, person_id)
        cluster.clear_guesses_for(conn, [person_id])
        conn.execute("UPDATE faces SET person_id = NULL, assigned_by = 'user' WHERE person_id = ?", (person_id,))
        conn.execute("UPDATE bodies SET person_id = NULL, assigned_by = 'user' WHERE person_id = ?", (person_id,))
        conn.execute("UPDATE animals SET person_id = NULL, assigned_by = 'user' WHERE person_id = ?", (person_id,))
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
    session: SessionInfo = Depends(require_user),
) -> FacePage:
    _visible_person_row(conn, session, person_id)
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
    return FacePage(items=_face_items(conn, rows), next_cursor=nxt, total=int(total))


@router.get("/people/{person_id}/items", response_model=ItemPage)
def person_items(
    person_id: int,
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=60, ge=1, le=200),
    vault: bool = Depends(want_vault),
    conn: sqlite3.Connection = Depends(get_db),
    session: SessionInfo = Depends(require_user),
) -> ItemPage:
    _visible_person_row(conn, session, person_id)
    conds = [visible_sql(vault=vault, misclicks="exclude", dups="all"),
             "i.id IN (SELECT item_id FROM faces WHERE person_id = ? UNION SELECT item_id FROM bodies WHERE person_id = ? "
             "UNION SELECT item_id FROM animals WHERE person_id = ? UNION SELECT item_id FROM item_tags WHERE person_id = ?)"]
    params: list[Any] = [person_id, person_id, person_id, person_id]
    sort_key = "COALESCE(i.taken_at, i.created_at)"
    v = _keyset(cursor, 2)
    # The total comes with the first page only (clients read it there); later pages carry 0.
    total = 0 if v is not None else conn.execute(
        f"SELECT COUNT(*) FROM items i WHERE {' AND '.join(conds)}", params).fetchone()[0]
    sql, extra = page_sql(conds, sort_key, v)
    rows = conn.execute(sql, [*params, *extra, limit + 1]).fetchall()
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
    return FacePage(items=_face_items(conn, rows), next_cursor=nxt, total=int(total))


def _assign(conn: sqlite3.Connection, table: str, ids: list[int], person_id: int | None, new_name: str | None,
            name_given: bool, session: SessionInfo) -> AssignOut:
    _check_boxes_vault(conn, session, table, ids)
    followed: list[int] = []
    with tx(conn):
        target = _target_person(conn, person_id, new_name, name_given, session)
        leaving = _leaving(conn, table, ids, target) if table == "faces" else {}
        updated, items = _move_boxes(conn, table, ids, target)
        if leaving:
            followed, more_items = _follow_lookalikes(conn, leaving, target)
            items |= more_items
    guesses = {(target, i) for i in items} if target is not None else set()
    _after_people_change(conn, items, guesses)  # type: ignore[arg-type]
    if followed and target is None:
        # The look-alikes are loose now: let them find or start another person together (never the
        # one they just left: its rejected examples keep them away).
        try:
            cluster.assign_new_faces(conn, followed)
        except Exception:  # noqa: BLE001 - the owner's correction already happened
            log.exception("could not regroup %d look-alike faces", len(followed))
        guess.refresh_fts(conn, items)
    return AssignOut(updated=updated, person=_person_out(conn, target) if target is not None else None,
                     followed=len(followed))


def _leaving(conn: sqlite3.Connection, table: str, ids: list[int], target: int | None) -> dict[int, list[int]]:
    """{person_id: faces among `ids` that are leaving that person} (a move to the same person is no leave)."""
    ids = sorted({int(i) for i in ids})
    out: dict[int, list[int]] = {}
    for r in conn.execute(f"SELECT id, person_id FROM {table} WHERE id IN ({_ids_sql(ids)}) AND person_id IS NOT NULL", ids):
        if target is None or int(r["person_id"]) != int(target):
            out.setdefault(int(r["person_id"]), []).append(int(r["id"]))
    return out


def _follow_lookalikes(conn: sqlite3.Connection, leaving: dict[int, list[int]],
                       target: int | None) -> tuple[list[int], set[int]]:
    """When faces leave a person, the person's automatic faces that look like them go the same way
    (to the target person, or loose), and every face that left is remembered as not that person.
    Returns (the faces that followed, their items). Call inside a tx."""
    followed: list[int] = []
    items: set[int] = set()
    for pid, seeds in leaving.items():
        cluster.record_rejections(conn, seeds, pid)
        more = cluster.followers(conn, seeds, pid)
        if not more:
            continue
        cluster.record_rejections(conn, more, pid)
        marks = _ids_sql(more)
        rows = conn.execute(f"SELECT id, item_id FROM faces WHERE id IN ({marks})", more).fetchall()
        items |= {int(r["item_id"]) for r in rows}
        conn.execute(
            f"UPDATE faces SET person_id = ?, assigned_by = 'auto', guess_person_id = NULL, guess_score = NULL, "
            f"guess_evidence = NULL WHERE id IN ({marks})", [target, *more])
        conn.execute(
            f"UPDATE bodies SET person_id = ?, assigned_by = 'auto', guess_person_id = NULL, guess_score = NULL, "
            f"guess_evidence = NULL WHERE face_id IN ({marks}) AND assigned_by != 'user'", [target, *more])
        conn.execute(f"UPDATE persons SET cover_face_id = NULL, updated_at = ? WHERE cover_face_id IN ({marks})",
                     [now_iso(), *more])
        followed.extend(more)
        log.info("%d look-alike faces followed %d faces away from person %d", len(more), len(seeds), pid)
    if followed:
        cluster.prune_empty_persons(conn, set(leaving))
        bump_faces_version(conn)
    return followed, items


@router.post("/faces/assign", response_model=AssignOut)
def assign_faces(body: AssignFacesIn, conn: sqlite3.Connection = Depends(get_db),
                 session: SessionInfo = Depends(require_user)) -> AssignOut:
    return _assign(conn, "faces", body.face_ids, body.person_id, body.new_person_name,
                   "new_person_name" in body.model_fields_set and body.new_person_name is not None, session)


@router.post("/bodies/assign", response_model=AssignOut)
def assign_bodies(body: AssignBodiesIn, conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> AssignOut:
    return _assign(conn, "bodies", body.body_ids, body.person_id, body.new_person_name,
                   "new_person_name" in body.model_fields_set and body.new_person_name is not None, session)


ANIMAL_SELECT = (
    "SELECT a.id, a.person_id, a.assigned_by, a.det_score, a.species, a.species_source, a.frame_time_s, a.created_at, "
    "i.uid AS item_uid, i.kind AS item_kind, i.version AS item_version FROM animals a JOIN items i ON i.id = a.item_id"
)
_BOX_NAMES = {"faces": ("Face not found", "face_count"), "bodies": ("Body not found", "body_count"),
              "animals": ("Animal box not found", "animal_count")}
_CROP_PATHS = {"faces": storage.face_crop_path, "bodies": storage.body_crop_path, "animals": storage.animal_crop_path}


def _animal_item(row: sqlite3.Row) -> AnimalListItem:
    return AnimalListItem(
        id=row["id"], item_uid=row["item_uid"], item_kind=row["item_kind"],
        thumb_url=thumb_url(row["item_uid"], int(row["item_version"])), url=animal_url(int(row["id"])),
        det_score=row["det_score"], species=row["species"], frame_time_s=row["frame_time_s"],
        person_id=row["person_id"], assigned_by=row["assigned_by"], created_at=row["created_at"],
    )


@router.get("/people/{person_id}/animals", response_model=AnimalPage)
def person_animals(
    person_id: int,
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=200, ge=1, le=1000),
    conn: sqlite3.Connection = Depends(get_db),
    session: SessionInfo = Depends(require_user),
) -> AnimalPage:
    """A pet's crops (section 22), best detection first; empty for a person."""
    _visible_person_row(conn, session, person_id)
    conds = ["a.person_id = ?", LIVE]
    params: list[Any] = [person_id]
    total = conn.execute(f"SELECT COUNT(*) FROM animals a JOIN items i ON i.id = a.item_id WHERE {' AND '.join(conds)}",
                         params).fetchone()[0]
    v = _keyset(cursor, 2)
    if v is not None:  # ORDER BY det_score DESC, id ASC
        conds.append("(a.det_score < ? OR (a.det_score = ? AND a.id > ?))")
        params += [v[0], v[0], v[1]]
    rows = conn.execute(f"{ANIMAL_SELECT} WHERE {' AND '.join(conds)} ORDER BY a.det_score DESC, a.id ASC LIMIT ?",
                        [*params, limit + 1]).fetchall()
    more = len(rows) > limit
    rows = rows[:limit]
    nxt = encode_cursor([rows[-1]["det_score"], int(rows[-1]["id"])]) if more and rows else None
    return AnimalPage(items=[_animal_item(r) for r in rows], next_cursor=nxt, total=int(total))


@router.post("/animals/assign", response_model=AssignOut)
def assign_animals(body: AssignAnimalsIn, conn: sqlite3.Connection = Depends(get_db),
                   session: SessionInfo = Depends(require_user)) -> AssignOut:
    """Assign animal boxes to a pet (person_id), to a new or existing pet by name, or to nobody
    (section 22). All boxes must be of one species; the pet must be of that species."""
    ids = sorted({int(i) for i in body.animal_ids})
    rows = conn.execute(f"SELECT id, species FROM animals WHERE id IN ({_ids_sql(ids)})", ids).fetchall()
    if len(rows) != len(ids):
        raise HTTPException(status_code=404, detail="Some boxes no longer exist. Reload and try again.")
    species = {str(r["species"]) for r in rows}
    if len(species) > 1:
        raise HTTPException(status_code=400, detail="Choose boxes of one species at a time")
    _check_boxes_vault(conn, session, "animals", ids)
    name_given = "new_person_name" in body.model_fields_set and body.new_person_name is not None
    boxes_species = species.pop()
    followed = 0
    with tx(conn):
        if body.fix_species and body.person_id is not None:
            # section 22: the target pet (every crop of it) becomes the boxes' species first
            row = _visible_person_row(conn, session, body.person_id)
            if row["kind"] == "pet" and row["species"] and row["species"] != boxes_species:
                change = pets_mod.set_pet_species(conn, int(row["id"]), boxes_species)
                followed = change.followed
                if change.person_id is not None and change.person_id != body.person_id:
                    body.person_id = change.person_id
        target = _target_person(conn, body.person_id, body.new_person_name, name_given, session, species=boxes_species)
        updated, items = _move_boxes(conn, "animals", ids, target)
    _after_people_change(conn, items, set())
    return AssignOut(updated=updated, person=_person_out(conn, target) if target is not None else None, followed=followed)


@router.post("/animals/species", response_model=AssignOut)
def set_animal_species(body: SpeciesIn, conn: sqlite3.Connection = Depends(get_db),
                       session: SessionInfo = Depends(require_user)) -> AssignOut:
    """The owner's species for these boxes (section 22): locked as their choice, recorded for learning;
    a box in a pet of another species leaves it and is matched again; look-alike crops follow."""
    ids = sorted({int(i) for i in body.animal_ids})
    found = conn.execute(f"SELECT COUNT(*) FROM animals WHERE id IN ({_ids_sql(ids)})", ids).fetchone()[0]
    if found != len(ids):
        raise HTTPException(status_code=404, detail="Some boxes no longer exist. Reload and try again.")
    _check_boxes_vault(conn, session, "animals", ids)
    with tx(conn):
        change = pets_mod.set_species(conn, ids, body.species)
        bump_faces_version(conn)
        bump_search_version(conn)
    _after_people_change(conn, change.items, set())
    return AssignOut(updated=change.updated, followed=change.followed)


@router.post("/people/{person_id}/species", response_model=AssignOut)
def set_person_species(person_id: int, body: PersonSpeciesIn, conn: sqlite3.Connection = Depends(get_db),
                       session: SessionInfo = Depends(require_user)) -> AssignOut:
    """Every crop of a pet takes this species, as the owner's choice, and so does the pet (section 22). A
    pet of the same name and species already there takes the crops instead; the reply names the pet the
    change ends on."""
    row = _visible_person_row(conn, session, person_id)
    if row["kind"] != "pet":
        raise HTTPException(status_code=400, detail="That is a person, not a pet")
    with tx(conn):
        change = pets_mod.set_pet_species(conn, int(person_id), body.species)
        bump_faces_version(conn)
        bump_search_version(conn)
    final = change.person_id if change.person_id is not None else int(person_id)
    _after_people_change(conn, change.items | _items_of_person(conn, final), set())
    return AssignOut(updated=change.updated, person=_person_out(conn, final), followed=change.followed)


@router.delete("/animals/{animal_id}", response_model=Ok)
def delete_animal(animal_id: int, conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> Ok:
    return _delete_box(conn, "animals", animal_id, session)


def _delete_box(conn: sqlite3.Connection, table: str, box_id: int, session: SessionInfo) -> Ok:
    not_found, count_col = _BOX_NAMES[table]
    with tx(conn):
        row = conn.execute(f"SELECT id, item_id, person_id FROM {table} WHERE id = ?", (box_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=not_found)
        _check_items_vault(conn, session, [int(row["item_id"])])
        if table == "bodies" and row["person_id"] is not None:
            _drop_body_tags(conn, [row])
        if table == "faces":
            cluster.remove_faces(conn, [box_id])
        else:
            conn.execute(f"DELETE FROM {table} WHERE id = ?", (box_id,))
            if table == "bodies":
                cluster.drop_rejections(conn, "body", [box_id])
        conn.execute(f"UPDATE items SET {count_col} = (SELECT COUNT(*) FROM {table} WHERE item_id = ?) WHERE id = ?",
                     (row["item_id"], row["item_id"]))
        if row["person_id"] is not None:
            cluster.prune_empty_persons(conn, {int(row["person_id"])})
        bump_faces_version(conn)
        bump_search_version(conn)
    storage.safe_unlink(_CROP_PATHS[table](box_id))
    guess.refresh_fts(conn, {int(row["item_id"])})
    return Ok()


@router.delete("/faces/{face_id}", response_model=Ok)
def delete_face(face_id: int, conn: sqlite3.Connection = Depends(get_db),
                session: SessionInfo = Depends(require_user)) -> Ok:
    return _delete_box(conn, "faces", face_id, session)


@router.delete("/bodies/{body_id}", response_model=Ok)
def delete_body(body_id: int, conn: sqlite3.Connection = Depends(get_db),
                session: SessionInfo = Depends(require_user)) -> Ok:
    return _delete_box(conn, "bodies", body_id, session)


# ------------------------------------------------------------------ guesses
@router.post("/guesses/confirm", response_model=BulkOut)
def confirm_guesses(body: GuessIdsIn, conn: sqlite3.Connection = Depends(get_db),
                    session: SessionInfo = Depends(require_user)) -> BulkOut:
    _check_boxes_vault(conn, session, "faces", body.faces)
    _check_boxes_vault(conn, session, "bodies", body.bodies)
    return BulkOut(updated=guess.confirm(conn, body.faces, body.bodies))


@router.post("/guesses/reject", response_model=BulkOut)
def reject_guesses(body: GuessIdsIn, conn: sqlite3.Connection = Depends(get_db),
                   session: SessionInfo = Depends(require_user)) -> BulkOut:
    _check_boxes_vault(conn, session, "faces", body.faces)
    _check_boxes_vault(conn, session, "bodies", body.bodies)
    return BulkOut(updated=guess.reject(conn, body.faces, body.bodies))


@router.get("/review/guesses", response_model=GuessPage)
def review_guesses(
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=60, ge=1, le=200),
    person: int | None = Query(default=None, description="only guesses naming this person"),
    conn: sqlite3.Connection = Depends(get_db),
) -> GuessPage:
    person_sql = " AND g.guess_person_id = ?" if person is not None else ""
    # The box columns face_out / body_out need come with the page (no per-row re-read).
    base = (
        "SELECT g.* FROM ("
        "SELECT 'face' AS kind, f.id AS box_id, f.item_id, f.guess_person_id, COALESCE(f.guess_score, 0) AS score, "
        "f.id, f.x, f.y, f.w, f.h, f.det_score, f.size_px, f.quality, f.frame_time_s, f.spans, NULL AS face_id, "
        "f.person_id, f.assigned_by, f.guess_score, f.guess_evidence FROM faces f "
        "WHERE f.guess_person_id IS NOT NULL AND f.person_id IS NULL UNION ALL "
        "SELECT 'body' AS kind, b.id AS box_id, b.item_id, b.guess_person_id, COALESCE(b.guess_score, 0) AS score, "
        "b.id, b.x, b.y, b.w, b.h, b.det_score, NULL, NULL, b.frame_time_s, NULL, b.face_id, "
        "b.person_id, b.assigned_by, b.guess_score, b.guess_evidence FROM bodies b "
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
    refs = person_refs(conn, box_person_ids(rows))
    for r in rows:
        if int(r["item_id"]) not in cards:
            continue
        g = guess_out(conn, r, refs)
        if g is None:
            continue
        box = face_out(conn, r, refs) if r["kind"] == "face" else body_out(conn, r, refs)
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
        f"SELECT {PERSON_COLUMNS} FROM persons WHERE id IN ({ph})", ids)}
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
def add_tag(uid: str, body: TagIn, conn: sqlite3.Connection = Depends(get_db),
            session: SessionInfo = Depends(require_user)) -> list[PersonRef]:
    name_given = body.new_person_name is not None
    if (body.person_id is None) == (not name_given):
        raise HTTPException(status_code=400, detail="Choose an existing person or a new name")
    with tx(conn):
        item = _item_row(conn, uid)
        if item["nsfw"] and not session.vault_unlocked:
            raise _vault_locked()
        item_id = int(item["id"])
        pid = _target_person(conn, body.person_id, body.new_person_name, name_given, session)
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
def remove_tag(uid: str, person_id: int, conn: sqlite3.Connection = Depends(get_db),
               session: SessionInfo = Depends(require_user)) -> list[PersonRef]:
    with tx(conn):
        item = _item_row(conn, uid)
        if item["nsfw"] and not session.vault_unlocked:
            raise _vault_locked()
        item_id = int(item["id"])
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
