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
  POST   /items/{uid}/faces {x, y, w, h, t?, person_id | new_person_name} -> 201 FaceOut: a face box the owner
         draws (section 10, person_boxes.prepare_face); 422 no_face when no face shows inside, 409 when the face
         model is not loaded (models_unavailable) or the photo or frame cannot be opened
  PUT    /faces/{id}/box, /bodies/{id}/box, /animals/{id}/box {x, y, w, h} -> FaceOut | BodyOut | AnimalOut: the
         owner moves or resizes a box (section 10, app.analysis.box_edit); 422 for a bad box or no_face (a face
         box moved where no face is), 409 when the photo or frame cannot be opened
  POST   /guesses/confirm {faces, bodies} -> BulkOut;  POST /guesses/reject -> BulkOut
  POST   /guesses/correct {faces, bodies, person_id | new_person_name} -> AssignOut ("Someone else": the guess learned
         as wrong and the box given to the right person, one action)
  GET    /review/guesses?cursor&limit&person&basis=all|clothes|trait -> GuessPage (basis, section 34: clothes =
         guesses from clothes alone, trait = guesses that tattoos, mehndi or nails made or backed, face = guesses of
         a model trained on one person)
  Recognition trained on one person (section 10, app.learning.person_model; people only, 404 for a pet):
  GET    /people/{id}/recognition -> RecognitionOut (the model, its check, the last look, a run's progress)
  POST   /people/{id}/recognition/train {scan=true} -> 202 {started} (background; 409 with the reason when it
         cannot be trained, 409 busy while another person's run is going);  POST .../scan -> 202 (409 not_trained)
  GET    /people/{id}/recognition/conflicts?limit&cursor -> RecognitionConflictPage (a keyset cursor);
  GET    /people/{id}/recognition/elsewhere -> RecognitionElsewhereList;  POST .../leave {face_ids} -> BulkOut
         (those faces on other people are not this person: remembered, and off the list)
  POST   /people/{id}/recognition/withdraw -> BulkOut (its open guesses taken back, the model paused; the undo
         brings them back and un-pauses it);  DELETE /people/{id}/recognition -> Ok (the model forgotten, its
         guesses taken back; not undoable). Both 409 running while a run for this person is going.
  Body traits (section 34):
  GET    /people/{id}/traits -> PersonTraitsOut (tattoos by design, recent mehndi and nails; empty for a pet)
  PUT    /traits/{id} {hidden, verdict=not_trait|not_theirs} -> TraitOut: hide a trait, or show it again on this
         photo only (an unhide never removes a hide placed on another photo; a row still hidden as a look-alike
         gets a 'shown' hide here). Journalled like every owner action; the not_trait descriptors are copied
         outside the journal. 404 for a missing row or a trashed item.
  GET    /suggestions?limit -> SuggestionList;  POST /suggestions/dismiss {a, b} -> Ok
  POST   /items/{uid}/tags {person_id | new_person_name, kind?, species?, relation?} -> [PersonRef] (a person or a
         pet, section 22; relation in|related, section 10);  DELETE /items/{uid}/tags/{person_id}
  People on a selection (section 10; uids 1 to 1000, unknown and trashed items skipped, 403 vault_locked for a
  vault item while the vault is locked):
  POST   /items/people {uids} -> ItemsPeopleOut (the people and pets on them, unnamed faces counted)
  POST   /items/people/move {uids, from_person_id, person_id | new_person_name} -> PeopleBulkOut (their faces,
         bodies, pet boxes and tags there go to the target; a person only to a person, a pet only to a pet;
         boxes remembered as not them, look-alikes follow; a move to a person is learned like a merge)
  POST   /items/people/remove {uids, person_id} -> PeopleBulkOut (their boxes there unassigned, tags deleted)
  POST   /items/people/tag {uids, person_id | new_person_name, kind?, species?, relation?} -> PeopleBulkOut
  Related tags suggested (section 10, app.analysis.related):
  GET    /people/{id}/related-suggestions?limit&vault -> RelatedSuggestionList;  POST .../dismiss {uids} -> Ok
  GET    /items/{uid}/related-suggestions -> ItemRelatedSuggestions
  GET    /people/{id}/childhood -> ChildhoodOut (who an unnamed group of childhood photos may be, section 10)
  POST   /recluster -> Ok
Faces, bodies and counts in NSFW or trashed items are never shown here (person items take vault=1
with an unlocked session). With the vault locked, acting on a box or the tags of an NSFW item is
403 vault_locked, and a person seen only in vault items is 404 (as in the list). Every write to faces,
bodies or persons bumps faces_version; deleting a box also bumps search_version (its torso embedding goes).
Moving, drawing or deleting a person box wakes the traits runner after the commit (the photo is measured again).
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Literal, NamedTuple

from fastapi import APIRouter, Depends, HTTPException, Query

from app import storage
from app import undo as undo_mod
from app.analysis import cluster, guess
from app.analysis import animals as animals_mod
from app.analysis import box_edit, person_boxes, person_scan, trait_match
from app.analysis import traits as atraits
from app.analysis import related as related_mod
from app.analysis.index import get_index
from app.analysis import pets as pets_mod
from app.learning import not_faces as not_faces_mod
from app.learning import person_model
from app.learning import store as learned_store
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
    trait_out,
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
    ChildhoodCandidate,
    ChildhoodOut,
    DismissIn,
    GuessCorrectIn,
    ItemRelatedSuggestion,
    ItemRelatedSuggestions,
    RelatedDismissIn,
    RelatedSuggestionList,
    RelatedSuggestionOut,
    FaceListItem,
    FacePage,
    GuessIdsIn,
    GuessItem,
    GuessPage,
    ItemPage,
    ItemPersonOut,
    ItemsPeopleIn,
    ItemsPeopleOut,
    MergeIn,
    FaceDeleteOut,
    Ok,
    PersonCreate,
    PersonBoxIn,
    PersonBoxOut,
    PersonOut,
    PersonPage,
    PersonRef,
    PersonSpeciesIn,
    PersonUpdate,
    PeopleBulkOut,
    PeopleMoveIn,
    PeopleRemoveIn,
    PeopleTagIn,
    SpeciesIn,
    Suggestion,
    SuggestionList,
    TagIn,
    TraitHideIn,
    TraitOut,
    PersonTraitsOut,
    AnimalBoxIn,
    AnimalOut,
    BodyOut,
    BoxIn,
    FaceBoxIn,
    FaceOut,
    RedetectIn,
    RecognitionConflict,
    RecognitionConflictPage,
    RecognitionElsewhere,
    RecognitionElsewhereList,
    RecognitionLeaveIn,
    RecognitionOut,
    RecognitionStartIn,
    RecognitionStartOut,
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
    "SELECT person_id, item_id FROM item_tags WHERE relation = 'in'"
)  # who is in which item; a 'related' tag (section 10) is not presence
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


def _stats(conn: sqlite3.Connection, ids: list[int]) -> dict[int, tuple[int, int, int, int, int, int]]:
    """{person_id: (face_count, item_count, guess_count, animal_count, confirmed, related_count)} over
    live, non-NSFW items; confirmed = faces and crops the owner assigned; related_count = items tagged
    'related' to them (section 10), which item_count leaves out."""
    out: dict[int, list[int]] = {i: [0, 0, 0, 0, 0, 0] for i in ids}
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
        for r in conn.execute(
            f"SELECT t.person_id, COUNT(*) FROM item_tags t JOIN items i ON i.id = t.item_id AND {LIVE} "
            f"WHERE t.person_id IN ({ph}) AND t.relation = 'related' GROUP BY t.person_id", chunk,
        ):
            out[int(r[0])][5] = int(r[1])
    return {k: (v[0], v[1], v[2], v[3], v[4], v[5]) for k, v in out.items()}


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
        fc, ic, gc, ac, cf, rc = st.get(int(r["id"]), (0, 0, 0, 0, 0, 0))
        pet = r["kind"] == "pet"
        out.append(PersonOut(id=int(r["id"]), name=r["name"], cover_url=covers.get(int(r["id"])),
                             cover_face_id=None if pet else r["cover_face_id"],
                             face_count=fc, item_count=ic, guess_count=gc, created_at=r["created_at"],
                             kind=r["kind"] or "person", species=r["species"] if pet else None, animal_count=ac,
                             confirmed_faces=cf, related_count=rc))
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


_REJECTION_KIND = {"faces": "face", "bodies": "body", "animals": "animal"}


def _move_boxes(conn: sqlite3.Connection, table: str, ids: list[int], target: int | None) -> tuple[int, set[int]]:
    """Assign faces, bodies or animal boxes to target (None unassigns), locked as 'user', forgetting any
    earlier "not the target" for them (guess_rejections). Call inside a tx. Returns (rows updated, item ids).
    404 when an id is unknown."""
    ids = sorted({int(i) for i in ids})
    ph = _ids_sql(ids)
    rows = conn.execute(f"SELECT id, item_id, person_id FROM {table} WHERE id IN ({ph})", ids).fetchall()
    if len(rows) != len(ids):
        raise HTTPException(status_code=404, detail="Some boxes no longer exist. Reload and try again.")
    previous = {int(r["person_id"]) for r in rows if r["person_id"] is not None}
    if table == "bodies":
        _drop_body_tags(conn, [r for r in rows if r["person_id"] is not None and r["person_id"] != target])
    items = {int(r["item_id"]) for r in rows}
    # the owner's own say: no longer a guess they confirmed (by_guess, section 10)
    clear = "" if table == "animals" else ", by_guess = 0, guess_person_id = NULL, guess_score = NULL, guess_evidence = NULL"
    conn.execute(f"UPDATE {table} SET person_id = ?, assigned_by = 'user'{clear} WHERE id IN ({ph})", [target, *ids])
    if target is not None:
        # The owner says these boxes are the target: an older "not the target" for them is outdated.
        conn.execute(f"DELETE FROM guess_rejections WHERE kind = ? AND person_id = ? AND ref_id IN ({ph})",
                     [_REJECTION_KIND[table], int(target), *ids])
    now = now_iso()
    if table == "faces":
        # A body around the face follows it.
        conn.execute(f"UPDATE bodies SET person_id = ?, assigned_by = 'user'{clear} WHERE face_id IN ({ph})",
                     [target, *ids])
        if target is not None:
            conn.execute(
                "DELETE FROM guess_rejections WHERE kind = 'body' AND person_id = ? "
                f"AND ref_id IN (SELECT id FROM bodies WHERE face_id IN ({ph}))", [int(target), *ids])
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
    """The item's tags as PersonRefs carrying each tag's relation (section 10)."""
    rows = conn.execute(
        "SELECT p.id, t.relation FROM item_tags t JOIN persons p ON p.id = t.person_id "
        "WHERE t.item_id = ? ORDER BY p.name_key IS NULL, p.name_key, p.id",
        (item_id,),
    ).fetchall()
    refs = person_refs(conn, [int(r["id"]) for r in rows])
    return [refs[int(r["id"])].model_copy(update={"relation": r["relation"]}) for r in rows if int(r["id"]) in refs]


FACE_SELECT = (
    "SELECT f.id, f.person_id, f.assigned_by, f.source, f.by_guess, f.det_score, f.size_px, f.quality, f.frame_time_s, "
    "f.created_at, "
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
        source="user" if row["source"] == "user" else "detector", by_guess=bool(row["by_guess"]),
        guess=guess_out(conn, row, refs), created_at=row["created_at"],
    )


def _after_people_change(conn: sqlite3.Connection, items: set[int], guesses: set[tuple[int, int | None]]) -> None:
    for pid, iid in sorted(guesses, key=lambda t: (t[0], -1 if t[1] is None else t[1])):
        try:
            guess.enqueue_guess(conn, pid, iid)
        except Exception as e:  # the user's action already happened; a missing follow-up is logged
            log.warning("could not queue guess job for person %s: %s", pid, e)
    guess.refresh_fts(conn, items)


def _kick_traits() -> None:
    """A person box changed: the traits runner measures the photo again (section 34). After the commit."""
    try:
        atraits.kick()
    except Exception as e:  # the user's action already happened; the runner's own triggers catch up later
        log.warning("could not wake the traits runner: %s", e)


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
    for gone in [*sources, *followers]:
        person_model.forget(gone)  # the faces a model was trained on are the target's now
    _after_people_change(conn, _items_of_person(conn, body.target_id), {(body.target_id, None)})
    out = _person_out(conn, body.target_id)
    out.followed = len(followers)
    return out


@router.get("/people/{person_id}", response_model=PersonOut)
def get_person(person_id: int, conn: sqlite3.Connection = Depends(get_db),
               session: SessionInfo = Depends(require_user)) -> PersonOut:
    out = person_outs(conn, [_visible_person_row(conn, session, person_id)])[0]
    if out.kind == "person":
        out.trait_count = int(trait_match.trait_count(conn, person_id, vault_open=session.vault_unlocked) or 0)
    return out


@router.get("/people/{person_id}/traits", response_model=PersonTraitsOut)
def person_traits(person_id: int, conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> PersonTraitsOut:
    """Tattoos, mehndi and nails seen on this person (section 34). Vault photos only with the vault open."""
    row = _visible_person_row(conn, session, person_id)
    if row["kind"] == "pet":
        return PersonTraitsOut()
    return PersonTraitsOut.model_validate(
        trait_match.person_groups(conn, int(row["id"]), vault_open=session.vault_unlocked))


@router.patch("/people/{person_id}", response_model=PersonOut)
def update_person(person_id: int, body: PersonUpdate, conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> PersonOut:
    fields = body.model_fields_set
    result = person_id
    renamed = False
    followed = 0
    merged: list[int] = []
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
                merged = [person_id, *followers]
                result = other
            else:
                conn.execute("UPDATE persons SET name = ?, name_key = ?, updated_at = ? WHERE id = ?",
                             (name, name.casefold() if name else None, now, person_id))
            renamed = name is not None
        bump_faces_version(conn)
    for gone in merged:
        person_model.forget(gone)  # taking a name that exists is a merge: a model trained on them goes with them
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
        # section 10: a face a look took out of this (unnamed) group does not come back to a group the owner deleted
        person_scan.repoint(conn, {person_id: None})
        conn.execute("UPDATE faces SET person_id = NULL, assigned_by = 'user', by_guess = 0 WHERE person_id = ?", (person_id,))
        conn.execute("UPDATE bodies SET person_id = NULL, assigned_by = 'user', by_guess = 0 WHERE person_id = ?", (person_id,))
        conn.execute("UPDATE animals SET person_id = NULL, assigned_by = 'user' WHERE person_id = ?", (person_id,))
        conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        bump_faces_version(conn)
    person_model.forget(person_id)  # a model trained on them has nobody to recognise
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
    relation: Literal["in", "related", "all"] = "in",
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=60, ge=1, le=200),
    vault: bool = Depends(want_vault),
    conn: sqlite3.Connection = Depends(get_db),
    session: SessionInfo = Depends(require_user),
) -> ItemPage:
    """The person's items: `in` (faces, bodies, animal boxes and 'in' tags), `related` (the items about
    them without them in it, section 10) or `all`."""
    _visible_person_row(conn, session, person_id)
    boxes = ("SELECT item_id FROM faces WHERE person_id = ? UNION SELECT item_id FROM bodies WHERE person_id = ? "
             "UNION SELECT item_id FROM animals WHERE person_id = ? UNION ")
    if relation == "related":
        present = "SELECT item_id FROM item_tags WHERE person_id = ? AND relation = 'related'"
        params: list[Any] = [person_id]
    elif relation == "all":
        present = boxes + "SELECT item_id FROM item_tags WHERE person_id = ?"
        params = [person_id] * 4
    else:
        present = boxes + "SELECT item_id FROM item_tags WHERE person_id = ? AND relation = 'in'"
        params = [person_id] * 4
    conds = [visible_sql(vault=vault, misclicks="exclude", dups="all"), f"i.id IN ({present})"]
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
    """{person_id: boxes among `ids` that are leaving that person or pet} (a move to the same one is no leave)."""
    ids = sorted({int(i) for i in ids})
    out: dict[int, list[int]] = {}
    for r in conn.execute(f"SELECT id, person_id FROM {table} WHERE id IN ({_ids_sql(ids)}) AND person_id IS NOT NULL", ids):
        if target is None or int(r["person_id"]) != int(target):
            out.setdefault(int(r["person_id"]), []).append(int(r["id"]))
    return out


def _follow_lookalikes(conn: sqlite3.Connection, leaving: dict[int, list[int]],
                       target: int | None) -> tuple[list[int], set[int]]:
    """When faces leave a person, the person's automatic faces that look like them go the same way
    (to the target person, or loose), and every face that left is remembered as not that person. A
    look-alike the owner already said is not the target stays. Returns (the faces that followed, their
    items). Call inside a tx."""
    followed: list[int] = []
    items: set[int] = set()
    for pid, seeds in leaving.items():
        if conn.execute("SELECT 1 FROM persons WHERE id = ?", (pid,)).fetchone() is None:
            continue  # an unnamed group the move emptied is gone (prune_empty_persons): no look-alikes, no memory
        cluster.record_rejections(conn, seeds, pid)
        more = cluster.followers(conn, seeds, pid)
        if more and target is not None:
            refused = {int(r[0]) for r in conn.execute(
                f"SELECT ref_id FROM guess_rejections WHERE kind = 'face' AND person_id = ? AND ref_id IN ({_ids_sql(more)})",
                [int(target), *more])}
            more = [f for f in more if f not in refused]
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
    "SELECT a.id, a.person_id, a.assigned_by, a.source, a.det_score, a.species, a.species_source, a.frame_time_s, "
    "a.created_at, i.uid AS item_uid, i.kind AS item_kind, i.version AS item_version "
    "FROM animals a JOIN items i ON i.id = a.item_id"
)
_BOX_NAMES = {"faces": ("Face not found", "face_count"), "bodies": ("Body not found", "body_count"),
              "animals": ("Animal box not found", "animal_count")}
_CROP_PATHS = {"faces": storage.face_crop_path, "bodies": storage.body_crop_path, "animals": storage.animal_crop_path}


def _animal_item(row: sqlite3.Row) -> AnimalListItem:
    return AnimalListItem(
        id=row["id"], item_uid=row["item_uid"], item_kind=row["item_kind"],
        thumb_url=thumb_url(row["item_uid"], int(row["item_version"])), url=animal_url(int(row["id"])),
        det_score=row["det_score"], species=row["species"], frame_time_s=row["frame_time_s"],
        person_id=row["person_id"], assigned_by=row["assigned_by"], source=row["source"], created_at=row["created_at"],
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
        leaving = _leaving(conn, "animals", ids, target)
        updated, items = _move_boxes(conn, "animals", ids, target)
        crops: list[int] = []
        if leaving:
            # not this pet (section 22): remembered, and the pet's automatic look-alikes go the same way
            crops, crop_items = pets_mod.leave_pets(conn, leaving, target)
            items |= crop_items
    _after_people_change(conn, items, set())
    if crops and target is None:
        _regroup_pets(conn, crops, items)
    return AssignOut(updated=updated, person=_person_out(conn, target) if target is not None else None,
                     followed=followed + len(crops))


def _regroup_pets(conn: sqlite3.Connection, crops: list[int], items: set[int]) -> None:
    """Pet crops that followed a crop the owner unassigned are loose: let them find another pet or start
    one together (never the pet they left: its rejected crops keep them away)."""
    try:
        with tx(conn):
            marks = _ids_sql(crops)
            for item_id in sorted({int(r[0]) for r in conn.execute(
                    f"SELECT DISTINCT item_id FROM animals WHERE id IN ({marks})", crops)}):
                pets_mod.assign_new_animals(conn, item_id)
            bump_faces_version(conn)
    except Exception:  # noqa: BLE001 - the owner's correction already happened
        log.exception("could not regroup %d look-alike pet crops", len(crops))
    guess.refresh_fts(conn, items)


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


@router.post("/items/{uid}/animals", response_model=AnimalOut, status_code=201)
def add_animal_box(uid: str, body: AnimalBoxIn, conn: sqlite3.Connection = Depends(get_db),
                   session: SessionInfo = Depends(require_user)) -> AnimalOut:
    """A box the owner drew on a photo or a video frame (section 22, boxes the owner adds): crop, look
    embedding, species (the owner's, else the zero-shot's, else 400 species_needed), the pet match. A video
    takes `t`, the frame's time (400 frame_time_needed without it); other kinds answer 400."""
    item = _boxable_or_400(conn, session, uid)
    video = item["kind"] == "video"
    if video and body.t is None:
        raise HTTPException(status_code=400, detail="frame_time_needed")
    try:
        aid = animals_mod.add_box(conn, item, (body.x, body.y, body.w, body.h), species=body.species,
                                  t=body.t if video else None)
    except animals_mod.FrameTimeNeeded:
        raise HTTPException(status_code=400, detail="frame_time_needed") from None
    except animals_mod.SpeciesNeeded:
        raise HTTPException(status_code=400, detail="species_needed") from None
    except animals_mod.NoImage:
        raise HTTPException(status_code=409, detail="The photo or frame could not be opened") from None
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    with tx(conn):
        bump_faces_version(conn)
        bump_search_version(conn)
    _after_people_change(conn, {int(item["id"])}, set())
    return animal_out(conn, conn.execute("SELECT * FROM animals WHERE id = ?", (aid,)).fetchone())


@router.post("/items/{uid}/people-boxes", response_model=PersonBoxOut, status_code=201)
def add_person_box(uid: str, body: PersonBoxIn, conn: sqlite3.Connection = Depends(get_db),
                   session: SessionInfo = Depends(require_user)) -> PersonBoxOut:
    """A person box the owner drew on a photo or a video frame (section 10, app.analysis.person_boxes): the
    person for that occasion, and the face the detector finds inside it for the same person (counted by its
    face weight). A stored face found there is assigned the way a hand assignment is. A video takes `t`
    (400 frame_time_needed without it); the person is person_id or new_person_name (400 person_needed
    without either); other kinds answer 400, an unreadable photo or frame 409."""
    item = _boxable_or_400(conn, session, uid)
    video = item["kind"] == "video"
    if video and body.t is None:
        raise HTTPException(status_code=400, detail="frame_time_needed")
    name_given = body.new_person_name is not None and body.new_person_name.strip() != ""
    if body.person_id is None and not name_given:
        raise HTTPException(status_code=400, detail="person_needed")
    try:
        prep = person_boxes.prepare_box(item, (body.x, body.y, body.w, body.h), t=body.t if video else None)
    except animals_mod.FrameTimeNeeded:
        raise HTTPException(status_code=400, detail="frame_time_needed") from None
    except animals_mod.NoImage:
        raise HTTPException(status_code=409, detail="The photo or frame could not be opened") from None
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    with tx(conn):
        target = _target_person(conn, body.person_id, body.new_person_name if name_given else None, name_given, session)
        if target is None:
            raise HTTPException(status_code=400, detail="person_needed")
        drawn = person_boxes.store_box(conn, prep, int(target))
        bump_faces_version(conn)
        bump_search_version(conn)
    if drawn.existing_face and drawn.face_id is not None:
        _assign(conn, "faces", [drawn.face_id], int(target), None, False, session)
    _after_people_change(conn, {int(item["id"])}, {(int(target), int(item["id"]))})
    _kick_traits()
    body_row = conn.execute("SELECT * FROM bodies WHERE id = ?", (drawn.body_id,)).fetchone()
    face_row = conn.execute("SELECT * FROM faces WHERE id = ?", (drawn.face_id,)).fetchone() if drawn.face_id else None
    return PersonBoxOut(body=body_out(conn, body_row), face=face_out(conn, face_row) if face_row is not None else None)


@router.post("/items/{uid}/faces", response_model=FaceOut, status_code=201)
def add_face_box(uid: str, body: FaceBoxIn, conn: sqlite3.Connection = Depends(get_db),
                 session: SessionInfo = Depends(require_user)) -> FaceOut:
    """A face box the owner drew around a face the detector missed (section 10, person_boxes.prepare_face): the
    detector must find the face inside it (422 no_face otherwise; 409 models_unavailable without it); the face
    is the person's, as a hand assignment is, and a stored face there is assigned instead. A video takes `t`."""
    item = _boxable_or_400(conn, session, uid)
    video = item["kind"] == "video"
    if video and body.t is None:
        raise HTTPException(status_code=400, detail="frame_time_needed")
    name_given = body.new_person_name is not None and body.new_person_name.strip() != ""
    if body.person_id is None and not name_given:
        raise HTTPException(status_code=400, detail="person_needed")
    try:
        prep = person_boxes.prepare_face(item, (body.x, body.y, body.w, body.h), t=body.t if video else None)
    except animals_mod.FrameTimeNeeded:
        raise HTTPException(status_code=400, detail="frame_time_needed") from None
    except animals_mod.ModelsUnavailable:
        raise HTTPException(status_code=409, detail="models_unavailable") from None
    except animals_mod.NoImage:
        raise HTTPException(status_code=409, detail="The photo or frame could not be opened") from None
    except person_boxes.NoFace:
        raise HTTPException(status_code=422, detail="no_face") from None
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    with tx(conn):
        target = _target_person(conn, body.person_id, body.new_person_name if name_given else None, name_given, session)
        if target is None:
            raise HTTPException(status_code=400, detail="person_needed")
        drawn = person_boxes.store_face(conn, prep, int(target))
        bump_faces_version(conn)
        bump_search_version(conn)
    if drawn.existing:
        _assign(conn, "faces", [drawn.face_id], int(target), None, False, session)
    _after_people_change(conn, {int(item["id"])}, {(int(target), int(item["id"]))})
    return face_out(conn, conn.execute("SELECT * FROM faces WHERE id = ?", (drawn.face_id,)).fetchone())


@router.post("/items/{uid}/animals/detect", response_model=list[AnimalOut])
def redetect_animals(uid: str, body: RedetectIn | None = None, conn: sqlite3.Connection = Depends(get_db),
                     session: SessionInfo = Depends(require_user)) -> list[AnimalOut]:
    """The detector once more at a lower threshold (section 22): on a photo, or on a video's poster and up to
    REDETECT_VIDEO_FRAMES sampled frames; the boxes it adds, matched to pets; 409 models_unavailable without
    the detector; other kinds answer 400."""
    item = _boxable_or_400(conn, session, uid)
    if not animals_mod.available():
        raise HTTPException(status_code=409, detail="models_unavailable")
    threshold = body.threshold if body is not None else animals_mod.REDETECT_DEFAULT
    try:
        ids = animals_mod.redetect(conn, item, threshold)
    except animals_mod.ModelsUnavailable:
        raise HTTPException(status_code=409, detail="models_unavailable") from None
    except animals_mod.NoImage:
        raise HTTPException(status_code=409, detail="The photo or frames could not be opened") from None
    if ids:
        with tx(conn):
            bump_faces_version(conn)
            bump_search_version(conn)
        _after_people_change(conn, {int(item["id"])}, set())
    rows = conn.execute(f"SELECT * FROM animals WHERE id IN ({_ids_sql(ids)}) ORDER BY det_score DESC, id", ids).fetchall() if ids else []
    return [animal_out(conn, r) for r in rows]


def _boxable_or_400(conn: sqlite3.Connection, session: SessionInfo, uid: str) -> sqlite3.Row:
    """A live photo or video the session may see; other kinds answer 400 (section 22)."""
    item = conn.execute("SELECT * FROM items WHERE uid = ? AND deleted_at IS NULL", (uid,)).fetchone()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_items_vault(conn, session, [int(item["id"])])
    if item["kind"] not in ("image", "video"):
        raise HTTPException(status_code=400, detail="Boxes can be drawn on photos and videos only")
    return item


@router.delete("/animals/{animal_id}", response_model=Ok)
def delete_animal(animal_id: int, conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> Ok:
    return _delete_box(conn, "animals", animal_id, session)


def _delete_box(conn: sqlite3.Connection, table: str, box_id: int, session: SessionInfo) -> Ok:
    not_found, count_col = _BOX_NAMES[table]
    followed: list[int] = []
    touched: set[int] = set()
    with tx(conn):
        row = conn.execute(f"SELECT id, item_id, person_id FROM {table} WHERE id = ?", (box_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=not_found)
        _check_items_vault(conn, session, [int(row["item_id"])])
        touched.add(int(row["item_id"]))
        if table == "bodies" and row["person_id"] is not None:
            _drop_body_tags(conn, [row])
        if table == "faces":
            # "Not a face" (section 10): kept for the learner, and its unassigned look-alikes go with it.
            face = conn.execute("SELECT * FROM faces WHERE id = ?", (box_id,)).fetchone()
            not_faces_mod.record(conn, face)
            followed = not_faces_mod.followers(conn, face)
            if followed:
                touched |= {int(r[0]) for r in conn.execute(
                    f"SELECT DISTINCT item_id FROM faces WHERE id IN ({_ids_sql(followed)})", followed)}
            cluster.remove_faces(conn, [box_id, *followed])
        else:
            conn.execute(f"DELETE FROM {table} WHERE id = ?", (box_id,))
            if table in ("bodies", "animals"):
                cluster.drop_rejections(conn, _REJECTION_KIND[table], [box_id])
        for item_id in sorted(touched):
            conn.execute(f"UPDATE items SET {count_col} = (SELECT COUNT(*) FROM {table} WHERE item_id = ?) WHERE id = ?",
                         (item_id, item_id))
        if row["person_id"] is not None:
            cluster.prune_empty_persons(conn, {int(row["person_id"])})
        bump_faces_version(conn)
        bump_search_version(conn)
    for gone in (box_id, *followed):
        undo_mod.discard(_CROP_PATHS[table](gone))  # kept for undo (section 33)
    if table == "bodies":
        _kick_traits()
    guess.refresh_fts(conn, touched)
    if followed:
        log.info("not a face: face %s and %d look-alike(s) removed", box_id, len(followed))
    return FaceDeleteOut(followed=len(followed)) if table == "faces" else Ok()


@router.delete("/faces/{face_id}", response_model=FaceDeleteOut)
def delete_face(face_id: int, conn: sqlite3.Connection = Depends(get_db),
                session: SessionInfo = Depends(require_user)) -> Ok:
    return _delete_box(conn, "faces", face_id, session)


@router.delete("/bodies/{body_id}", response_model=Ok)
def delete_body(body_id: int, conn: sqlite3.Connection = Depends(get_db),
                session: SessionInfo = Depends(require_user)) -> Ok:
    return _delete_box(conn, "bodies", body_id, session)


# ------------------------------------------------------------------ moving a box (section 10)
def _move_box(conn: sqlite3.Connection, table: str, box_id: int, body: BoxIn, session: SessionInfo) -> sqlite3.Row:
    """The box at its new place (box_edit): the slow part before the transaction, the crop after it."""
    not_found, _count = _BOX_NAMES[table]
    row = conn.execute(f"SELECT item_id FROM {table} WHERE id = ?", (box_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=not_found)
    _check_items_vault(conn, session, [int(row["item_id"])])
    try:
        prep = box_edit.prepare(conn, table, box_id, (body.x, body.y, body.w, body.h))
    except LookupError:
        raise HTTPException(status_code=404, detail=not_found) from None
    except box_edit.NoFace:
        raise HTTPException(status_code=422, detail="no_face") from None
    except animals_mod.FrameTimeNeeded:
        raise HTTPException(status_code=400, detail="frame_time_needed") from None
    except animals_mod.NoImage:
        raise HTTPException(status_code=409, detail="The photo or frame could not be opened") from None
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    with tx(conn):
        if not box_edit.store(conn, prep):
            raise HTTPException(status_code=404, detail=not_found)
    box_edit.write_crop(prep)
    if table == "faces":
        get_index().reload_faces(conn, [box_id])
    return conn.execute(f"SELECT * FROM {table} WHERE id = ?", (box_id,)).fetchone()


@router.put("/faces/{face_id}/box", response_model=FaceOut)
def move_face(face_id: int, body: BoxIn, conn: sqlite3.Connection = Depends(get_db),
              session: SessionInfo = Depends(require_user)) -> FaceOut:
    return face_out(conn, _move_box(conn, "faces", face_id, body, session))


@router.put("/bodies/{body_id}/box", response_model=BodyOut)
def move_body(body_id: int, body: BoxIn, conn: sqlite3.Connection = Depends(get_db),
              session: SessionInfo = Depends(require_user)) -> BodyOut:
    row = _move_box(conn, "bodies", body_id, body, session)
    _kick_traits()
    return body_out(conn, row)


@router.put("/animals/{animal_id}/box", response_model=AnimalOut)
def move_animal(animal_id: int, body: BoxIn, conn: sqlite3.Connection = Depends(get_db),
                session: SessionInfo = Depends(require_user)) -> AnimalOut:
    return animal_out(conn, _move_box(conn, "animals", animal_id, body, session))


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


@router.post("/guesses/correct", response_model=AssignOut)
def correct_guesses(body: GuessCorrectIn, conn: sqlite3.Connection = Depends(get_db),
                    session: SessionInfo = Depends(require_user)) -> AssignOut:
    """"Someone else" on a guess (sections 10 and 18), in one action and so one undo: the guess is learned as wrong, as
    /guesses/reject does (its evidence goes to the guess learner as a no, and the box is remembered as not the guessed
    person, so that guess is not made again), and the box goes to the person the owner names, as /faces/assign and
    /bodies/assign do (the owner's own choice, not the system's find). A box guessed as that very person is a yes
    (/guesses/confirm); a box with no open guess is simply assigned."""
    name_given = body.new_person_name is not None
    if (body.person_id is None) == (not name_given):
        raise HTTPException(status_code=400, detail="Choose an existing person or a new name")
    faces = sorted({int(f) for f in body.faces})
    bodies = sorted({int(b) for b in body.bodies})
    if not faces and not bodies:
        raise HTTPException(status_code=400, detail="Choose a face or a person box")
    _check_boxes_vault(conn, session, "faces", faces)
    _check_boxes_vault(conn, session, "bodies", bodies)
    followed: list[int] = []
    with tx(conn):
        target = int(_target_person(conn, body.person_id, body.new_person_name, name_given, session))  # type: ignore[arg-type]

        def guessed_as_target(table: str, ids: list[int]) -> list[int]:
            if not ids:
                return []
            return [int(r[0]) for r in conn.execute(
                f"SELECT id FROM {table} WHERE id IN ({_ids_sql(ids)}) AND person_id IS NULL AND guess_person_id = ?",
                [*ids, target])]

        same_f, same_b = guessed_as_target("faces", faces), guessed_as_target("bodies", bodies)
        items: set[int] = set()
        for table, ids in (("faces", same_f), ("bodies", same_b)):
            if ids:
                items |= {int(r[0]) for r in conn.execute(
                    f"SELECT item_id FROM {table} WHERE id IN ({_ids_sql(ids)})", ids)}
        confirmed = guess.confirm(conn, same_f, same_b) if same_f or same_b else 0
        rest_f = [f for f in faces if f not in set(same_f)]
        rest_b = [b for b in bodies if b not in set(same_b)]
        # only the boxes with an open guess (of someone else) are touched; they go to the target right below
        guess.reject(conn, rest_f, rest_b, regroup=False)
        leaving = _leaving(conn, "faces", rest_f, target) if rest_f else {}
        updated = 0
        for table, ids in (("faces", rest_f), ("bodies", rest_b)):
            if ids:
                n, its = _move_boxes(conn, table, ids, target)
                updated += n
                items |= its
        if leaving:
            followed, more = _follow_lookalikes(conn, leaving, target)
            items |= more
    _after_people_change(conn, items, {(target, i) for i in items})
    return AssignOut(updated=updated + confirmed, person=_person_out(conn, target), followed=len(followed))


# Section 34: evidence.basis is 'trait' (traits alone), 'clothes+trait' (clothes backed by traits) or absent (clothes).
# Section 35: 'place' (the position alone) and 'clothes+place' (clothes backed by the position).
_BASIS_SQL = {
    "all": "",
    # CASE keeps json_extract away from malformed JSON (AND does not promise an order in SQLite)
    "clothes": " AND (CASE WHEN json_valid(g.guess_evidence) THEN json_extract(g.guess_evidence, '$.basis') END) IS NULL",
    "trait": " AND (CASE WHEN json_valid(g.guess_evidence) THEN json_extract(g.guess_evidence, '$.basis') END) "
             "IN ('trait', 'clothes+trait')",
    # section 10: the guesses of a model trained on one person
    "face": " AND (CASE WHEN json_valid(g.guess_evidence) THEN json_extract(g.guess_evidence, '$.basis') END) IS 'face'",
    # any guess with a position part: a trait guess that position agrees with keeps its basis and gains the part
    "place": " AND (CASE WHEN json_valid(g.guess_evidence) THEN json_extract(g.guess_evidence, '$.basis') "
             "IN ('place', 'clothes+place') OR json_extract(g.guess_evidence, '$.place') IS NOT NULL END)",
}


@router.get("/review/guesses", response_model=GuessPage)
def review_guesses(
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=60, ge=1, le=200),
    person: int | None = Query(default=None, description="only guesses naming this person"),
    basis: Literal["all", "clothes", "trait", "face", "place"] = Query(
        default="all", description="clothes alone, with traits, a model trained on one person, or with the position"),
    conn: sqlite3.Connection = Depends(get_db),
) -> GuessPage:
    person_sql = " AND g.guess_person_id = ?" if person is not None else ""
    person_sql += _BASIS_SQL.get(basis, "") if isinstance(basis, str) else ""  # a direct call gets the Query default
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


# ------------------------------------------------------------------ body traits (section 34)
_TRAIT_ROW = ", ".join(f"t.{c.strip()}" for c in atraits.ROW_COLUMNS.split(","))


def _trait_row(conn: sqlite3.Connection, trait_id: int) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT {_TRAIT_ROW}, i.nsfw FROM body_traits t JOIN items i ON i.id = t.item_id "
        "WHERE t.id = ? AND i.deleted_at IS NULL", (trait_id,)).fetchone()


@router.put("/traits/{trait_id}", response_model=TraitOut)
def hide_trait(trait_id: int, body: TraitHideIn, conn: sqlite3.Connection = Depends(get_db),
               session: SessionInfo = Depends(require_user)) -> TraitOut:
    """Hide a trait ("Not a tattoo", "Not this person's") or show it again (section 34). An owner action: the
    trait_hides rows are journalled (X-Action-Id undoes it). Unhiding is limited to this photo: it removes the
    hides here that cover the row, and when a look-alike of a hide on another photo still hides it, it adds a
    'shown' hide here instead of touching that photo. The guesses of the photo (and of photos citing the row)
    are worked out again after the commit."""
    row = _trait_row(conn, trait_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Trait not found")
    if row["nsfw"] and not session.vault_unlocked:
        raise _vault_locked()
    item_id = int(row["item_id"])
    region = (row["x"], row["y"], row["w"], row["h"])
    now = now_iso()
    with tx(conn):
        if conn.execute("SELECT 1 FROM body_traits WHERE id = ?", (trait_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="Trait not found")  # replaced by a rescan meanwhile
        if body.hidden:
            _insert_hide(conn, row, body.verdict, region, now)
        else:
            covering = [int(h) for h in trait_match.hides_covering(conn, trait_id)]
            if covering:
                conn.execute(f"DELETE FROM trait_hides WHERE item_id = ? AND verdict != 'shown' "
                             f"AND id IN ({_ids_sql(covering)})", [item_id, *covering])
            # After the delete: the row's own not_trait hide may list the row as its look-alike until it goes.
            if trait_match.hidden_by_lookalike(conn, trait_id):
                conn.execute(
                    "INSERT INTO trait_hides(item_id, kind, verdict, x, y, w, h, src_trait_id, created_at) "
                    "VALUES (?, ?, 'shown', ?, ?, ?, ?, ?, ?)", (item_id, row["kind"], *region, trait_id, now))
    try:
        # The look-alikes of the hide (up to VERIFY_PER_HIDE SIFT checks) and the re-guess run on the traits runner,
        # never in this request: the row is hidden at once (rules a and b need no verification), the guesses
        # follow within seconds. Guesses stay out of this action's journal (an undo re-queues through
        # traits.after_undo), so undoing an older hide is never blocked by guess columns a later one rewrote.
        from app.analysis import traits as atraits

        atraits.requeue(conn, [item_id])
        atraits.kick(conn)
    except Exception:  # the owner's change is saved; the guesses catch up on the next runner pass
        log.exception("traits: could not queue item %s after a hide change", item_id)
    fresh = _trait_row(conn, trait_id) or row
    return trait_out(fresh, trait_id in trait_match.hidden_ids(conn, [fresh]))


def _insert_hide(conn: sqlite3.Connection, row: sqlite3.Row, verdict: str, region: tuple, now: str) -> None:
    """A not_trait or not_theirs hide for the row (once per row and verdict). A not_trait hide of a tattoo or
    mehndi also keeps the row's descriptors in trait_hide_desc (not journalled, no FK: an undone unhide finds
    them again), so look-alikes on other photos can be found (rule c)."""
    trait_id, item_id = int(row["id"]), int(row["item_id"])
    if conn.execute("SELECT 1 FROM trait_hides WHERE item_id = ? AND src_trait_id = ? AND verdict = ?",
                    (item_id, trait_id, verdict)).fetchone():
        return
    box: tuple = (None, None, None, None)
    if verdict == "not_theirs":
        b = conn.execute("SELECT x, y, w, h FROM bodies WHERE id = ?", (int(row["body_id"]),)).fetchone()
        if b is not None:
            box = (b["x"], b["y"], b["w"], b["h"])
    hide_id = int(conn.execute(
        "INSERT INTO trait_hides(item_id, kind, verdict, x, y, w, h, bx, by, bw, bh, src_trait_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (item_id, row["kind"], verdict, *region, *box, trait_id, now)).lastrowid)
    if verdict == "not_trait" and row["kind"] in ("tattoo", "mehndi"):
        conn.execute("INSERT OR REPLACE INTO trait_hide_desc(hide_id, vec, detail, created_at) "
                     "SELECT ?, vec, detail, ? FROM body_traits WHERE id = ? AND vec IS NOT NULL "
                     "AND detail IS NOT NULL", (hide_id, now, trait_id))


# ------------------------------------------------------------------ suggestions
CHILD_SUGGEST_MAX = 5  # "Same child?" cards shown after the usual suggestions


@router.get("/suggestions", response_model=SuggestionList)
def suggestions(
    limit: int = Query(default=30, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_db),
) -> SuggestionList:
    """Merge suggestions, best first: people who look alike (cluster.merge_suggestions, up to `limit`), then up to
    CHILD_SUGGEST_MAX pairs whose childhood photos look alike (cluster.childhood_suggestions, kind 'childhood',
    section 10; their samples are those childhood photos)."""
    young = cluster.childhood_suggestions(conn, CHILD_SUGGEST_MAX)
    kinds = {(int(a), int(b)): "childhood" for a, b, _ in young}
    same = [t for t in cluster.merge_suggestions(conn, limit + len(kinds)) if (int(t[0]), int(t[1])) not in kinds][:limit]
    pairs = [*same, *young]  # the childhood pairs after the usual ones, not counted against `limit`: they are rare
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
    young_samples = _childhood_samples(conn, {p for a, b, _ in pairs if (int(a), int(b)) in kinds for p in (int(a), int(b))})
    items: list[Suggestion] = []
    for a, b, sim in pairs:
        oa, ob = outs.get(int(a)), outs.get(int(b))
        if oa is None or ob is None or oa.item_count == 0 or ob.item_count == 0:
            continue
        kind = kinds.get((int(a), int(b)), "same")
        pick = young_samples if kind == "childhood" else samples
        if kind == "childhood" and not (pick.get(oa.id) and pick.get(ob.id)):
            continue  # their childhood photos are in the vault or the trash: nothing to compare here
        items.append(Suggestion(a=oa, b=ob, similarity=float(sim), a_samples=pick.get(oa.id, []),
                                b_samples=pick.get(ob.id, []), kind=kind))
    return SuggestionList(items=items)


def _childhood_samples(conn: sqlite3.Connection, person_ids: set[int], n: int = 4) -> dict[int, list[str]]:
    """Up to n of each person's childhood photos (faces in the child band, section 10), best first."""
    out: dict[int, list[str]] = {}
    if not person_ids:
        return out
    idx = get_index()
    with idx.lock:
        idx.sync(conn)
        for pid in sorted(person_ids):
            rows = conn.execute(f"SELECT f.id FROM faces f JOIN items i ON i.id = f.item_id AND {LIVE} "
                                "WHERE f.person_id = ? ORDER BY f.quality DESC, f.id", (pid,)).fetchall()
            ids = [int(r[0]) for r in rows]
            flags = idx.child_flags(ids)
            out[pid] = [face_url(f) for f, young in zip(ids, flags) if young][:n]
    return out


@router.get("/people/{person_id}/childhood", response_model=ChildhoodOut)
def person_childhood(
    person_id: int,
    limit: int = Query(default=3, ge=1, le=10),
    conn: sqlite3.Connection = Depends(get_db),
    session: SessionInfo = Depends(require_user),
) -> ChildhoodOut:
    """Who an unnamed group of childhood photos may be as a child (section 10): named people ranked by how their
    adult photos look like the group's childhood photos (cluster.childhood_candidates). No bar: the owner picks
    (a merge, learned as a cross-age link) or ignores it."""
    _visible_person_row(conn, session, person_id)
    ranked = cluster.childhood_candidates(conn, person_id, limit)
    if not ranked:
        return ChildhoodOut(candidates=[])
    rows = conn.execute(f"SELECT {PERSON_COLUMNS} FROM persons WHERE id IN ({_ids_sql([p for p, _ in ranked])})",
                        [p for p, _ in ranked]).fetchall()
    outs = {o.id: o for o in person_outs(conn, rows)}
    return ChildhoodOut(candidates=[ChildhoodCandidate(person=outs[p], similarity=round(float(sv), 4))
                                    for p, sv in ranked if p in outs and not _hidden_person(conn, session, p)])


# ------------------------------------------------------------------ recognition trained on one person (section 10)
def _recognition_person(conn: sqlite3.Connection, session: SessionInfo, person_id: int) -> sqlite3.Row:
    row = _visible_person_row(conn, session, person_id)
    if row["kind"] != "person":
        raise HTTPException(status_code=404, detail="Person not found")
    return row


def _standing_conflicts(conn: sqlite3.Connection, person_id: int, raw: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The stored conflicts that still stand: the face is still on the person in a visible item and not the
    owner's own say, and the face it looks like still exists and is not the person's."""
    listed = [c for c in ((raw or {}).get("conflicts") or []) if isinstance(c, dict)]
    if not listed:
        return []
    mine: set[int] = set()
    other: set[int] = set()
    for chunk in _chunks(sorted({int(c["face_id"]) for c in listed})):
        mine |= {int(r[0]) for r in conn.execute(
            f"SELECT f.id FROM faces f JOIN items i ON i.id = f.item_id AND {LIVE} WHERE f.id IN ({_ids_sql(chunk)}) "
            "AND f.person_id = ? AND f.assigned_by != 'user'", [*chunk, person_id])}
    for chunk in _chunks(sorted({int(c["like_face_id"]) for c in listed})):
        other |= {int(r[0]) for r in conn.execute(
            f"SELECT id FROM faces WHERE id IN ({_ids_sql(chunk)}) AND person_id IS NOT ?", [*chunk, person_id])}
    return [c for c in listed if int(c["face_id"]) in mine and int(c["like_face_id"]) in other]


def _standing_elsewhere(conn: sqlite3.Connection, person_id: int, raw: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The last look's faces on other named people that look like the person, those still where they were."""
    listed = [e for e in (((raw or {}).get("scan") or {}).get("elsewhere") or []) if isinstance(e, dict)]
    if not listed:
        return []
    where: dict[int, int] = {}
    for chunk in _chunks(sorted({int(e["face_id"]) for e in listed})):
        for r in conn.execute(
            f"SELECT f.id, f.person_id FROM faces f JOIN items i ON i.id = f.item_id AND {LIVE} "
            "JOIN persons p ON p.id = f.person_id AND p.name IS NOT NULL AND p.name != '' "
            f"WHERE f.id IN ({_ids_sql(chunk)}) AND f.assigned_by = 'auto' AND NOT EXISTS (SELECT 1 FROM guess_rejections g "
            "WHERE g.kind = 'face' AND g.ref_id = f.id AND g.person_id = ?)", [*chunk, person_id],
        ):
            where[int(r[0])] = int(r[1])
    return [e for e in listed if where.get(int(e["face_id"])) == int(e["person_id"])]


def _recognition_out(conn: sqlite3.Connection, person_id: int) -> RecognitionOut:
    raw = person_model.stored(person_id)
    learning = learned_store.enabled(conn)
    reason = None
    if not learning:
        reason = "Learning is switched off in Settings."
    else:
        # the same count the training makes: the face weight of their faces outside the child band
        need = int(person_model.MIN_POSITIVE_WEIGHT)
        weight, faces = person_model.clear_weight(conn, person_id)
        if weight < need:
            reason = f"This person needs about {need} clear faces before recognition can be trained on them alone."
            if faces >= need:
                reason += (f" Their {faces} faces count as about {int(weight)}: a small, blurred or turned face counts "
                           "for less, and faces of a young child are left out.")
    st = person_scan.state(person_id, conn)
    busy_with = person_scan.busy_with()
    data: dict[str, Any] = {
        "available": reason is None, "reason": reason, "learning": learning, "running": st["running"],
        "progress": st["progress"], "error": st["error"], "busy": busy_with is not None and busy_with != person_id,
    }
    if raw is not None:
        scan = raw.get("scan") if isinstance(raw.get("scan"), dict) else None
        counts = raw.get("counts") or {}
        owner_faces, rejections = person_model.owner_answers(conn, person_id)
        data.update(
            trained=True, trained_at=raw.get("trained_at"),
            counts={k: v for k, v in counts.items() if isinstance(v, (int, float))},
            check=raw.get("check"), guess_p=raw.get("guess_p"), sure_p=raw.get("sure_p"),
            conflicts=len(_standing_conflicts(conn, person_id, raw)),
            elsewhere=len(_standing_elsewhere(conn, person_id, raw)),
            guesses=person_scan.standing_count(conn, person_id),
            withdrawn=person_id in person_model.paused(conn, [person_id]),
            answers_since=(max(0, owner_faces - int(counts.get("owner_faces", owner_faces)))
                           + max(0, rejections - int(counts.get("rejections", rejections)))),
            scan={**scan, "elsewhere": len(scan.get("elsewhere") or [])} if scan is not None else None,
        )
    return RecognitionOut.model_validate(data)


@router.get("/people/{person_id}/recognition", response_model=RecognitionOut)
def person_recognition(person_id: int, conn: sqlite3.Connection = Depends(get_db),
                       session: SessionInfo = Depends(require_user)) -> RecognitionOut:
    """Recognition trained on this person alone (section 10): whether it can be trained, the model and its check,
    the last look, and the progress of a run."""
    _recognition_person(conn, session, person_id)
    return _recognition_out(conn, person_id)


def _start_recognition(conn: sqlite3.Connection, person_id: int, *, train: bool, look: bool) -> RecognitionStartOut:
    started = person_scan.start(conn, person_id, train=train, look=look)
    if not started and person_scan.busy_with() not in (None, person_id):
        raise HTTPException(status_code=409, detail="busy")
    return RecognitionStartOut(started=started)


@router.post("/people/{person_id}/recognition/train", response_model=RecognitionStartOut, status_code=202)
def train_recognition(person_id: int, body: RecognitionStartIn | None = None,
                      conn: sqlite3.Connection = Depends(get_db),
                      session: SessionInfo = Depends(require_user)) -> RecognitionStartOut:
    """Train on the faces now on the person (the owner's say that they are right), then look again at every other
    face unless `scan` is false. Runs in the background; GET shows the progress. Nothing is tagged: what it finds
    becomes guesses."""
    _recognition_person(conn, session, person_id)
    out = _recognition_out(conn, person_id)
    if not out.available:
        raise HTTPException(status_code=409, detail=out.reason or "not_available")
    return _start_recognition(conn, person_id, train=True, look=body.scan if body is not None else True)


@router.post("/people/{person_id}/recognition/scan", response_model=RecognitionStartOut, status_code=202)
def scan_recognition(person_id: int, conn: sqlite3.Connection = Depends(get_db),
                     session: SessionInfo = Depends(require_user)) -> RecognitionStartOut:
    _recognition_person(conn, session, person_id)
    if person_model.load(person_id) is None:
        raise HTTPException(status_code=409, detail="not_trained")
    return _start_recognition(conn, person_id, train=False, look=True)


@router.get("/people/{person_id}/recognition/conflicts", response_model=RecognitionConflictPage)
def recognition_conflicts(
    person_id: int,
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=60, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_db),
    session: SessionInfo = Depends(require_user),
) -> RecognitionConflictPage:
    """Faces on the person that look the same as faces the owner took off them or gave to someone else, strongest
    first: the owner settles each with /faces/assign (to the person, to the other person, or to nobody). A settled
    one leaves the list, so the cursor is a keyset (the similarity and face of the last one read): what was not
    read yet is never skipped."""
    _recognition_person(conn, session, person_id)

    def order(c: dict[str, Any]) -> tuple[float, int]:
        return -float(c.get("sim") or 0.0), int(c["face_id"])

    standing = sorted(_standing_conflicts(conn, person_id, person_model.stored(person_id)), key=order)
    v = _keyset(cursor, 2)
    rest = standing
    if v is not None:
        if isinstance(v[0], bool) or not isinstance(v[0], (int, float)) or isinstance(v[1], bool) or not isinstance(v[1], int):
            raise HTTPException(status_code=400, detail="Invalid cursor")
        after = (-float(v[0]), int(v[1]))
        rest = [c for c in standing if order(c) > after]
    page = rest[:limit]
    ids = sorted({int(c["face_id"]) for c in page} | {int(c["like_face_id"]) for c in page})
    faces: dict[int, FaceListItem] = {}
    if ids:
        rows = conn.execute(f"{FACE_SELECT} AND {LIVE} WHERE f.id IN ({_ids_sql(ids)})", ids).fetchall()
        faces = {f.id: f for f in _face_items(conn, rows)}
    refs = person_refs(conn, [f.person_id for f in faces.values()])
    items = []
    for c in page:
        face = faces.get(int(c["face_id"]))
        if face is None:
            continue
        like = faces.get(int(c["like_face_id"]))
        items.append(RecognitionConflict(
            face=face, like=like, sim=float(c.get("sim") or 0.0),
            like_person=refs.get(like.person_id) if like is not None and like.person_id is not None else None))
    nxt = encode_cursor([float(page[-1].get("sim") or 0.0), int(page[-1]["face_id"])]) if len(rest) > limit else None
    return RecognitionConflictPage(items=items, next_cursor=nxt, total=len(standing))


@router.get("/people/{person_id}/recognition/elsewhere", response_model=RecognitionElsewhereList)
def recognition_elsewhere(person_id: int, conn: sqlite3.Connection = Depends(get_db),
                          session: SessionInfo = Depends(require_user)) -> RecognitionElsewhereList:
    """Faces the app put on other named people that the last look was sure are this person. They were not touched:
    the owner moves one with /faces/assign, or leaves it."""
    _recognition_person(conn, session, person_id)
    standing = _standing_elsewhere(conn, person_id, person_model.stored(person_id))
    ids = [int(e["face_id"]) for e in standing]
    faces: dict[int, FaceListItem] = {}
    if ids:
        rows = conn.execute(f"{FACE_SELECT} AND {LIVE} WHERE f.id IN ({_ids_sql(ids)})", ids).fetchall()
        faces = {f.id: f for f in _face_items(conn, rows)}
    refs = person_refs(conn, [int(e["person_id"]) for e in standing])
    return RecognitionElsewhereList(items=[
        RecognitionElsewhere(face=faces[int(e["face_id"])], person=refs[int(e["person_id"])], p=float(e.get("p") or 0.0))
        for e in standing if int(e["face_id"]) in faces and int(e["person_id"]) in refs])


@router.post("/people/{person_id}/recognition/leave", response_model=BulkOut)
def leave_recognition_elsewhere(person_id: int, body: RecognitionLeaveIn, conn: sqlite3.Connection = Depends(get_db),
                                session: SessionInfo = Depends(require_user)) -> BulkOut:
    """"Leave" on a face of the elsewhere list: the owner says this face on another person is not this person. It
    is remembered as any "not them" (guess_rejections), so the face leaves the list and its count, a later look
    does not bring it back, and the next training takes it as a face that is not the person. The face stays where
    it is. An owner action with its undo."""
    _recognition_person(conn, session, person_id)
    ids = sorted({int(f) for f in body.face_ids})
    _check_boxes_vault(conn, session, "faces", ids)
    with tx(conn):
        rows = conn.execute(
            f"SELECT id FROM faces WHERE id IN ({_ids_sql(ids)}) AND person_id IS NOT NULL AND person_id != ?",
            [*ids, person_id]).fetchall()
        if len(rows) != len(ids):
            raise HTTPException(status_code=400, detail="Only a face on another person can be left there.")
        known = conn.execute(
            f"SELECT COUNT(*) FROM guess_rejections WHERE kind = 'face' AND person_id = ? AND ref_id IN ({_ids_sql(ids)})",
            [person_id, *ids]).fetchone()[0]
        cluster.record_rejections(conn, ids, person_id)
    return BulkOut(updated=len(ids) - int(known))


def _recognition_idle(person_id: int) -> None:
    """409 `running` while a training or a look for this very person is going (`busy` means another person's)."""
    if person_scan.busy_with() == person_id:
        raise HTTPException(status_code=409, detail="running")


@router.post("/people/{person_id}/recognition/withdraw", response_model=BulkOut)
def withdraw_recognition(person_id: int, conn: sqlite3.Connection = Depends(get_db),
                         session: SessionInfo = Depends(require_user)) -> BulkOut:
    """Take the model's open guesses back and pause it: it makes no new ones until the next look or training. An
    owner action: its undo brings the guesses back and un-pauses the model (the pause is a journaled column)."""
    _recognition_person(conn, session, person_id)
    _recognition_idle(person_id)
    return BulkOut(updated=person_scan.withdraw(conn, person_id))


@router.delete("/people/{person_id}/recognition", response_model=Ok)
def forget_recognition(person_id: int, conn: sqlite3.Connection = Depends(get_db),
                       session: SessionInfo = Depends(require_user)) -> Ok:
    """Forget the model trained on this person and take its open guesses back. The faces and the owner's answers
    stay; the usual matching goes on as before. Not an undoable action: the model and its guesses are derived, and
    an undo could bring the guesses back but not the model behind them. Training again makes both again."""
    _recognition_person(conn, session, person_id)
    _recognition_idle(person_id)
    with undo_mod.not_undoable():
        person_model.forget(person_id)  # first: without the model nothing makes the guesses again
        person_scan.take_back(conn, person_id)
        person_model.resume(conn, person_id)
    return Ok()


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
def _tag_target(conn: sqlite3.Connection, body: TagIn, session: SessionInfo) -> int:
    """The person or pet a tag names: an existing id of either kind, or a new name; a new pet needs its
    species (400 species_needed, section 22). Call inside a tx."""
    if body.person_id is not None:
        return int(_visible_person_row(conn, session, body.person_id)["id"])
    if body.kind == "pet":
        if body.species is None:
            raise HTTPException(status_code=400, detail="species_needed")
        pid = _target_person(conn, None, body.new_person_name, True, session, species=body.species)
    else:
        pid = _target_person(conn, None, body.new_person_name, True, session)
    assert pid is not None
    return pid


# ------------------------------------------------------------------ related tags suggested (section 10)
def _cards_by_id(conn: sqlite3.Connection, ids: list[int]) -> dict[int, sqlite3.Row]:
    out: dict[int, sqlite3.Row] = {}
    ids = sorted(set(ids))
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        for r in conn.execute(f"SELECT {ITEM_CARD_COLUMNS} FROM items i WHERE i.id IN ({_ids_sql(chunk)})", chunk):
            out[int(r["id"])] = r
    return out


@router.get("/people/{person_id}/related-suggestions", response_model=RelatedSuggestionList)
def related_suggestions(
    person_id: int,
    limit: int = Query(default=24, ge=1, le=60),
    vault: bool = Depends(want_vault),
    conn: sqlite3.Connection = Depends(get_db),
    session: SessionInfo = Depends(require_user),
) -> RelatedSuggestionList:
    """Items that may be related to the person too (app.analysis.related): they look like an item the owner marked
    as related to them, or somewhat like it and were taken at the same moment or place. Best first. Vault items with
    vault=1 (unlocked), as the person's items."""
    _visible_person_row(conn, session, person_id)
    visible = visible_sql(vault=vault, misclicks="exclude", dups="collapse")
    # every related item the owner marked counts (a copy, a forward), on the side of the vault being listed
    seeds = "i.deleted_at IS NULL AND i.nsfw = " + ("1" if vault else "0")
    found = related_mod.suggest(conn, person_id, visible=visible, seed_visible=seeds, limit=limit)
    if not found:
        return RelatedSuggestionList(items=[])
    rows = _cards_by_id(conn, [s.item_id for s in found] + [s.because for s in found])
    return RelatedSuggestionList(items=[
        RelatedSuggestionOut(item=item_card(rows[s.item_id]), reason=s.reason, because_uid=str(rows[s.because]["uid"]))
        for s in found if s.item_id in rows and s.because in rows
    ])


@router.post("/people/{person_id}/related-suggestions/dismiss", response_model=Ok)
def dismiss_related(person_id: int, body: RelatedDismissIn, conn: sqlite3.Connection = Depends(get_db),
                    session: SessionInfo = Depends(require_user)) -> Ok:
    """"Not related to NAME": these items are not suggested for the person again (related_dismissals; undoable)."""
    with tx(conn):
        _visible_person_row(conn, session, person_id)
        item_ids = _selection(conn, session, body.uids)
        related_mod.dismiss(conn, person_id, item_ids)
    return Ok()


@router.get("/items/{uid}/related-suggestions", response_model=ItemRelatedSuggestions)
def item_related_suggestions(uid: str, conn: sqlite3.Connection = Depends(get_db),
                             session: SessionInfo = Depends(require_user)) -> ItemRelatedSuggestions:
    """The people this item may be related to (app.analysis.related.suggest_for_item), best first. Only items on the
    item's own side of the vault count as the owner's marks, and a vault item needs the vault unlocked."""
    item = _item_row(conn, uid)
    if item["nsfw"] and not session.vault_unlocked:
        raise _vault_locked()
    visible = "i.deleted_at IS NULL AND i.nsfw = " + ("1" if item["nsfw"] else "0")
    found = related_mod.suggest_for_item(conn, int(item["id"]), visible=visible)
    found = [(pid, reason, seed) for pid, reason, seed in found if not _hidden_person(conn, session, pid)]
    if not found:
        return ItemRelatedSuggestions(people=[])
    persons = {int(r["id"]): r for r in conn.execute(
        f"SELECT id, name, kind FROM persons WHERE id IN ({_ids_sql([p for p, _, _ in found])})",
        [p for p, _, _ in found])}
    uids = {int(r[0]): str(r[1]) for r in conn.execute(
        f"SELECT id, uid FROM items WHERE id IN ({_ids_sql([s for _, _, s in found])})", [s for _, _, s in found])}
    return ItemRelatedSuggestions(people=[
        ItemRelatedSuggestion(person=PersonRef(id=pid, name=persons[pid]["name"], kind=persons[pid]["kind"] or "person"),
                              reason=reason, because_uid=uids[seed])
        for pid, reason, seed in found if pid in persons and seed in uids
    ])


@router.post("/items/{uid}/tags", response_model=list[PersonRef])
def add_tag(uid: str, body: TagIn, conn: sqlite3.Connection = Depends(get_db),
            session: SessionInfo = Depends(require_user)) -> list[PersonRef]:
    """Tag a person or a pet on an item: relation 'in' (they are in it) or 'related' (it is about them,
    section 10); posting the other relation for someone already tagged switches it. Answers the item's
    tags with their relation."""
    name_given = body.new_person_name is not None
    if (body.person_id is None) == (not name_given):
        raise HTTPException(status_code=400, detail="Choose an existing person or a new name")
    with tx(conn):
        item = _item_row(conn, uid)
        if item["nsfw"] and not session.vault_unlocked:
            raise _vault_locked()
        item_id = int(item["id"])
        pid = _tag_target(conn, body, session)
        now = now_iso()
        on_item = conn.execute(
            "SELECT 1 FROM faces WHERE item_id = ? AND person_id = ? UNION ALL "
            "SELECT 1 FROM bodies WHERE item_id = ? AND person_id = ? UNION ALL "
            "SELECT 1 FROM animals WHERE item_id = ? AND person_id = ? LIMIT 1",
            (item_id, pid, item_id, pid, item_id, pid),
        ).fetchone()
        # A box of theirs already says 'in'; 'related' is recorded whatever the boxes say, and the
        # owner's relation replaces an earlier tag's.
        if body.relation == "related" or on_item is None:
            conn.execute(
                "INSERT INTO item_tags(item_id, person_id, source, relation, created_at) VALUES (?, ?, 'user', ?, ?) "
                "ON CONFLICT(item_id, person_id) DO UPDATE SET relation = excluded.relation, source = 'user'",
                (item_id, pid, body.relation, now),
            )
        conn.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (now, pid))
        kind = conn.execute("SELECT kind FROM persons WHERE id = ?", (pid,)).fetchone()[0]
        bump_faces_version(conn)
    # The guess job looks around the item for the person's other appearances: only when they are in it.
    guesses = {(pid, item_id)} if body.relation == "in" and kind == "person" else set()
    _after_people_change(conn, {item_id}, guesses)
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



# ------------------------------------------------------------------ people on a selection (section 10)
def _chunks(ids: list[Any], n: int = 500) -> list[list[Any]]:
    return [ids[i:i + n] for i in range(0, len(ids), n)]


def _selection(conn: sqlite3.Connection, session: SessionInfo, uids: list[str]) -> list[int]:
    """The ids of the selected items that exist and are not in the trash; 403 vault_locked when one of
    them is in the vault and the vault is locked."""
    wanted = list(dict.fromkeys(u for u in uids if u))
    ids: list[int] = []
    vault = False
    for chunk in _chunks(wanted):
        for r in conn.execute(
            f"SELECT id, nsfw FROM items WHERE uid IN ({_ids_sql(chunk)}) AND deleted_at IS NULL", chunk
        ):
            ids.append(int(r["id"]))
            vault = vault or bool(r["nsfw"])
    if vault and not session.vault_unlocked:
        raise _vault_locked()
    return sorted(ids)


def _box_ids(conn: sqlite3.Connection, table: str, item_ids: list[int], person_id: int) -> list[int]:
    out: list[int] = []
    for chunk in _chunks(item_ids):
        out += [int(r[0]) for r in conn.execute(
            f"SELECT id FROM {table} WHERE person_id = ? AND item_id IN ({_ids_sql(chunk)})", [person_id, *chunk])]
    return out


class _Moved(NamedTuple):
    changed: set[int]  # the selected items that changed
    faces: list[int]  # the source's faces there, now the target's (or loose)
    followed: list[int]  # the source's automatic look-alike faces that went along
    crops: list[int]  # the pet's automatic look-alike crops that went along
    touched: set[int]  # every item touched


def _move_on_items(conn: sqlite3.Connection, item_ids: list[int], source: int, target: int | None) -> _Moved:
    """On these items every face, body and pet box of `source` goes to `target` (None unassigns them), as
    the owner's (_move_boxes), and every tag of `source` moves to `target` with its relation ('in' winning
    over a tag the target already has there) or, with no target, is deleted. Every box taken from `source`
    is remembered as not `source` (guess_rejections: faces, bodies, and pet crops, section 22), and the
    automatic look-alikes of the faces and crops follow them (_follow_lookalikes, pets.leave_pets), as
    /faces/assign and /animals/assign do. Call inside a tx."""
    changed: set[int] = set()
    touched: set[int] = set()
    followed: list[int] = []
    crops: list[int] = []
    # every body of the source here, those around its faces included (they go with the faces)
    body_ids = _box_ids(conn, "bodies", item_ids, source)
    face_ids = _box_ids(conn, "faces", item_ids, source)
    if face_ids:
        leaving = _leaving(conn, "faces", face_ids, target)
        _, items = _move_boxes(conn, "faces", face_ids, target)
        changed |= items
        if leaving:
            followed, more = _follow_lookalikes(conn, leaving, target)
            touched |= more
    rest = _box_ids(conn, "bodies", item_ids, source)  # the faceless bodies
    if rest:
        _, items = _move_boxes(conn, "bodies", rest, target)
        changed |= items
    if body_ids:
        cluster.record_box_rejections(conn, "body", body_ids, source)
    animal_ids = _box_ids(conn, "animals", item_ids, source)
    if animal_ids:
        _, items = _move_boxes(conn, "animals", animal_ids, target)
        changed |= items
        crops, more = pets_mod.leave_pets(conn, {source: animal_ids}, target)
        touched |= more
    now = now_iso()
    for chunk in _chunks(item_ids):
        rows = conn.execute(
            f"SELECT item_id, relation FROM item_tags WHERE person_id = ? AND item_id IN ({_ids_sql(chunk)})",
            [source, *chunk],
        ).fetchall()
        for r in rows:
            item_id = int(r["item_id"])
            if target is not None:
                have = conn.execute("SELECT relation FROM item_tags WHERE item_id = ? AND person_id = ?",
                                    (item_id, target)).fetchone()
                relation = "in" if r["relation"] == "in" or (have is not None and have["relation"] == "in") else "related"
                conn.execute(
                    "INSERT INTO item_tags(item_id, person_id, source, relation, created_at) VALUES (?, ?, 'user', ?, ?) "
                    "ON CONFLICT(item_id, person_id) DO UPDATE SET relation = excluded.relation, source = 'user'",
                    (item_id, target, relation, now),
                )
            conn.execute("DELETE FROM item_tags WHERE item_id = ? AND person_id = ?", (item_id, source))
            changed.add(item_id)
    if target is not None and changed:
        conn.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (now, target))
    cluster.prune_empty_persons(conn, {source})
    if changed:
        bump_faces_version(conn)
    touched |= changed
    return _Moved(changed, face_ids, followed, crops, touched)


def _pet_species(conn: sqlite3.Connection, source: sqlite3.Row, item_ids: list[int]) -> str | None:
    """A pet's species: its own, else the one species of its boxes on these items."""
    if source["species"]:
        return str(source["species"])
    found: set[str] = set()
    for chunk in _chunks(item_ids):
        found |= {str(r[0]) for r in conn.execute(
            f"SELECT DISTINCT species FROM animals WHERE person_id = ? AND item_id IN ({_ids_sql(chunk)})",
            [int(source["id"]), *chunk])}
    return found.pop() if len(found) == 1 else None


def _pet_target(conn: sqlite3.Connection, session: SessionInfo, person_id: int | None, new_name: str | None,
                species: str | None) -> int:
    """The pet a pet's appearances move to: an existing pet (400 for a person, or a pet of another species),
    or a new name: a pet of that name and species (or of no species yet) when there is one, else a new pet
    of `species` (a cat and a dog may share a name). Call inside a tx."""
    if person_id is not None:
        row = _visible_person_row(conn, session, person_id)
        if row["kind"] != "pet":
            raise HTTPException(status_code=400, detail="That is a person, not a pet")
        if species and row["species"] and row["species"] != species:
            raise HTTPException(status_code=400, detail=f"That pet is a {row['species']}, not a {species}")
        return int(row["id"])
    name = cluster.clean_person_name(new_name)
    if not name:
        raise HTTPException(status_code=400, detail="Enter a name")
    for r in conn.execute("SELECT id, species FROM persons WHERE kind = 'pet' AND name_key = ? ORDER BY id",
                          (name.casefold(),)):
        if not species or not r["species"] or r["species"] == species:
            return int(r["id"])
    return cluster.new_person(conn, name, kind="pet", species=species)


@router.post("/items/people", response_model=ItemsPeopleOut)
def people_on_items(body: ItemsPeopleIn, conn: sqlite3.Connection = Depends(get_db),
                    session: SessionInfo = Depends(require_user)) -> ItemsPeopleOut:
    """The people and pets on the selected items (section 10): who is on them through faces, bodies, pet
    boxes and tags, most items first, named before unnamed; and the faces nobody is named for. `by_you`
    counts the items where the owner put them there (a box they tagged or drew, or a tag), the rest came
    from recognition alone."""
    item_ids = _selection(conn, session, body.uids)
    stats: dict[int, dict[str, Any]] = {}

    def entry(pid: int) -> dict[str, Any]:
        return stats.setdefault(pid, {"items": set(), "mine": set(), "faces": 0, "bodies": 0, "animals": 0,
                                      "tags": 0, "related": 0})

    unnamed = 0
    for chunk in _chunks(item_ids):
        ph = _ids_sql(chunk)
        for table, key in (("faces", "faces"), ("bodies", "bodies"), ("animals", "animals")):
            # The owner's: tagged by hand (not a guess they confirmed: by_guess), or a box they drew, moved or resized.
            mine = ("(assigned_by = 'user' OR source = 'user')" if table == "animals"
                    else "((assigned_by = 'user' AND by_guess = 0) OR source = 'user')")
            for r in conn.execute(
                f"SELECT person_id, item_id, {mine} AS mine FROM {table} "
                f"WHERE person_id IS NOT NULL AND item_id IN ({ph})", chunk
            ):
                e = entry(int(r["person_id"]))
                e[key] += 1
                e["items"].add(int(r["item_id"]))
                if r["mine"]:
                    e["mine"].add(int(r["item_id"]))
        # A tag added by hand is the owner's; one from a guess on a faceless body they confirmed is the system's find.
        for r in conn.execute(f"SELECT person_id, item_id, relation, source FROM item_tags WHERE item_id IN ({ph})", chunk):
            e = entry(int(r["person_id"]))
            e["related" if r["relation"] == "related" else "tags"] += 1
            e["items"].add(int(r["item_id"]))
            if r["source"] != "guess":
                e["mine"].add(int(r["item_id"]))
        unnamed += int(conn.execute(
            f"SELECT COUNT(*) FROM faces WHERE person_id IS NULL AND item_id IN ({ph})", chunk).fetchone()[0])
    rows: list[sqlite3.Row] = []
    for chunk in _chunks(sorted(stats)):
        rows += conn.execute(f"SELECT {PERSON_COLUMNS} FROM persons WHERE id IN ({_ids_sql(chunk)})", chunk).fetchall()
    covers = _covers(conn, rows) if rows else {}
    people = []
    for r in rows:
        pid = int(r["id"])
        e = stats[pid]
        pet = r["kind"] == "pet"
        people.append(ItemPersonOut(
            id=pid, name=r["name"], kind=r["kind"] or "person", species=r["species"] if pet else None,
            cover_url=covers.get(pid), items=len(e["items"]), faces=e["faces"], bodies=e["bodies"],
            animals=e["animals"], tags=e["tags"], related=e["related"], by_you=len(e["mine"]),
        ))
    people.sort(key=lambda p: (-p.items, p.name is None, (p.name or "").casefold(), p.id))
    return ItemsPeopleOut(items=len(item_ids), unnamed_faces=unnamed, people=people)


@router.post("/items/people/move", response_model=PeopleBulkOut)
def move_people_on_items(body: PeopleMoveIn, conn: sqlite3.Connection = Depends(get_db),
                         session: SessionInfo = Depends(require_user)) -> PeopleBulkOut:
    """On the selected items, everything of `from_person_id` goes to the target (section 10): a person only
    to a person, a pet only to a pet of its species. The boxes are remembered as not `from_person_id` and its
    look-alike faces and crops follow (as /faces/assign, /animals/assign); faces moved to a person are learned
    like a merge, pulling the unnamed look-alike groups along (`pulled`)."""
    name_given = body.new_person_name is not None
    if (body.person_id is None) == (not name_given):
        raise HTTPException(status_code=400, detail="Choose an existing person or a new name")
    with tx(conn):
        item_ids = _selection(conn, session, body.uids)
        source = _visible_person_row(conn, session, body.from_person_id)
        source_id = int(source["id"])
        is_pet = source["kind"] == "pet"
        if is_pet:
            target = _pet_target(conn, session, body.person_id, body.new_person_name,
                                 _pet_species(conn, source, item_ids))
        else:
            target = _target_person(conn, body.person_id, body.new_person_name, name_given, session)
        assert target is not None
        target = int(target)
        if target == source_id:
            raise HTTPException(status_code=400, detail="That is the same person")
        moved = _move_on_items(conn, item_ids, source_id, target)
    touched = set(moved.touched)
    guesses: set[tuple[int, int | None]] = set() if is_pet else {(target, i) for i in moved.changed}
    pulled: list[int] = []
    if not is_pet and moved.faces:
        # Learned like a merge (section 18): the similarity the owner accepted, and the unnamed look-alike
        # groups it justifies come along (cluster.accept_faces).
        try:
            with tx(conn):
                _, pulled = cluster.accept_faces(conn, target, moved.faces, moved.followed)
        except Exception:  # noqa: BLE001 - the owner's move already happened
            log.exception("could not learn from the move to person %d", target)
            pulled = []
        if pulled:
            touched |= _items_of_person(conn, target)
            guesses.add((target, None))
    _after_people_change(conn, touched, guesses)
    return PeopleBulkOut(updated=len(moved.changed), person=_person_out(conn, target),
                         followed=len(moved.followed) + len(moved.crops), pulled=len(pulled))


@router.post("/items/people/remove", response_model=PeopleBulkOut)
def remove_people_from_items(body: PeopleRemoveIn, conn: sqlite3.Connection = Depends(get_db),
                             session: SessionInfo = Depends(require_user)) -> PeopleBulkOut:
    """On the selected items, the person's faces, bodies and pet boxes are unassigned (remembered as not
    them, the look-alike faces and crops following and regrouping, as an unassign through /faces/assign and
    /animals/assign) and their tags there are deleted (section 10)."""
    with tx(conn):
        item_ids = _selection(conn, session, body.uids)
        source_id = int(_visible_person_row(conn, session, body.person_id)["id"])
        moved = _move_on_items(conn, item_ids, source_id, None)
    _after_people_change(conn, moved.touched, set())
    if moved.followed:
        # The look-alikes are loose now: let them find or start another person together (never the one
        # they just left: its rejected examples keep them away).
        try:
            cluster.assign_new_faces(conn, moved.followed)
        except Exception:  # noqa: BLE001 - the owner's correction already happened
            log.exception("could not regroup %d look-alike faces", len(moved.followed))
        guess.refresh_fts(conn, moved.touched)
    if moved.crops:
        _regroup_pets(conn, moved.crops, moved.touched)
    return PeopleBulkOut(updated=len(moved.changed), person=None, followed=len(moved.followed) + len(moved.crops))


@router.post("/items/people/tag", response_model=PeopleBulkOut)
def tag_people_on_items(body: PeopleTagIn, conn: sqlite3.Connection = Depends(get_db),
                        session: SessionInfo = Depends(require_user)) -> PeopleBulkOut:
    """POST /items/{uid}/tags on every selected item (section 10): a box of theirs already says 'in', a
    'related' tag is always recorded, and a new pet needs its species (400 species_needed)."""
    name_given = body.new_person_name is not None
    if (body.person_id is None) == (not name_given):
        raise HTTPException(status_code=400, detail="Choose an existing person or a new name")
    changed: set[int] = set()
    with tx(conn):
        item_ids = _selection(conn, session, body.uids)
        pid = _tag_target(conn, body, session)
        now = now_iso()
        for item_id in item_ids:
            on_item = conn.execute(
                "SELECT 1 FROM faces WHERE item_id = ? AND person_id = ? UNION ALL "
                "SELECT 1 FROM bodies WHERE item_id = ? AND person_id = ? UNION ALL "
                "SELECT 1 FROM animals WHERE item_id = ? AND person_id = ? LIMIT 1",
                (item_id, pid, item_id, pid, item_id, pid),
            ).fetchone()
            if body.relation != "related" and on_item is not None:
                continue  # a box of theirs already says they are in it
            have = conn.execute("SELECT relation, source FROM item_tags WHERE item_id = ? AND person_id = ?",
                                (item_id, pid)).fetchone()
            if have is not None and have["relation"] == body.relation and have["source"] == "user":
                continue
            conn.execute(
                "INSERT INTO item_tags(item_id, person_id, source, relation, created_at) VALUES (?, ?, 'user', ?, ?) "
                "ON CONFLICT(item_id, person_id) DO UPDATE SET relation = excluded.relation, source = 'user'",
                (item_id, pid, body.relation, now),
            )
            changed.add(item_id)
        conn.execute("UPDATE persons SET updated_at = ? WHERE id = ?", (now, pid))
        kind = conn.execute("SELECT kind FROM persons WHERE id = ?", (pid,)).fetchone()[0]
        if changed:
            bump_faces_version(conn)
    # The guess job looks around the item for the person's other appearances: only when they are in it.
    guesses = {(pid, i) for i in changed} if body.relation == "in" and kind == "person" else set()
    _after_people_change(conn, changed, guesses)
    return PeopleBulkOut(updated=len(changed), person=_person_out(conn, pid), followed=0)


# ------------------------------------------------------------------ recluster
@router.post("/recluster", response_model=Ok)
def recluster(conn: sqlite3.Connection = Depends(get_db)) -> Ok:
    from app import jobs

    jobs.enqueue(conn, "recluster", {}, unique=True)
    return Ok()
