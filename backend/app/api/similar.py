"""The name picker's order (docs/ARCHITECTURE.md section 10, "Naming by likeness"; app.analysis.similar). Session
required.

POST /api/people/similar  SimilarIn -> SimilarOut. What is being named, any of:
                            face_ids, body_ids, animal_ids   boxes (a person box with a face inside counts as the face)
                            uids + person_id                 that person's boxes on those items (Change to...)
                            person_id alone                  the whole person (Merge into)
                            around_uid                       an item: who was in the photos of the same hour and is
                                                             not in it (a box being drawn, a tag on the item)
                          `limit` (1..30, 8): how many people come with their details.
                          SimilarOut.items: the best `limit` named people (pets for pet boxes) the list would show
                          (people only in the vault or the trash are left out, as on the People page), each as
                          `person` {id, name, kind, species, cover_url} (no counts: they take longer than the
                          ranking), `score` (0 to 1, the order: a face's plain similarity), `likely` (the matching
                          or the guess path would take them), `basis` and a plain `reason`, and `first` (the box's
                          own open guess, listed first). SimilarOut.scores:
                          person id -> score for every such person ranked (at most MAX_SCORES), so a picker can
                          order the names it finds by typing. 400 for uids without person_id; 403 vault_locked when
                          a box or item named is in the vault and the vault is locked; 404 for a person_id the list
                          would not show. Unknown box ids and uids are skipped. It reads only: it is not an action.
"""
from __future__ import annotations

import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.analysis import similar
from app.api.common import pet_cover_ids
from app.api.deps import get_db, require_user
from app.api.people import _check_boxes_vault, _check_items_vault, _visible_person_row
from app.auth.sessions import SessionInfo
from app.schemas import PersonKind
from app.urls import animal_url, face_url

router = APIRouter(tags=["people"], dependencies=[Depends(require_user)])

MAX_SCORES = 500
Basis = Literal["guess", "model", "face", "clothes", "look", "person", "hour"]
REASONS: dict[str, str] = {
    "guess": "The current guess",
    "model": "Their trained face matches",
    "face": "Looks like them",
    "clothes": "Same clothes within the hour",
    "look": "Looks like them",
    "person": "Looks like them",
    "hour": "In photos from the same hour",
}


class SimilarIn(BaseModel):
    face_ids: list[int] = Field(default_factory=list, max_length=1000)
    body_ids: list[int] = Field(default_factory=list, max_length=1000)
    animal_ids: list[int] = Field(default_factory=list, max_length=1000)
    uids: list[str] = Field(default_factory=list, max_length=1000)
    person_id: int | None = None
    around_uid: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=8, ge=1, le=30)


class SimilarPerson(BaseModel):
    """Who they are, as the picker shows them: no counts (counting a person's photos takes longer than the ranking)."""
    id: int
    name: str | None = None
    kind: PersonKind = "person"
    species: str | None = None    # pets only
    cover_url: str | None = None


class SimilarOption(BaseModel):
    person: SimilarPerson
    score: float
    likely: bool
    basis: Basis
    reason: str
    first: bool = False


class SimilarOut(BaseModel):
    items: list[SimilarOption]
    scores: dict[int, float]


def _item_ids(conn: sqlite3.Connection, uids: list[str]) -> list[int]:
    out: list[int] = []
    clean = sorted({u for u in uids if u})
    for start in range(0, len(clean), 500):
        chunk = clean[start:start + 500]
        out += [int(r[0]) for r in conn.execute(
            f"SELECT id FROM items WHERE uid IN ({','.join('?' * len(chunk))})", chunk)]
    return out


LIVE_COVER = "i.deleted_at IS NULL AND i.nsfw = 0"


def _people(conn: sqlite3.Connection, ids: list[int]) -> dict[int, SimilarPerson]:
    """SimilarPerson for these ids: the chosen cover when its photo is in the library, else the best face there (one
    index walk per person, best quality first), and a pet's best crop (common.pet_cover_ids)."""
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT p.id, p.name, p.kind, p.species, p.cover_face_id, i.id IS NOT NULL AS cover_ok FROM persons p "
        f"LEFT JOIN faces f ON f.id = p.cover_face_id LEFT JOIN items i ON i.id = f.item_id AND {LIVE_COVER} "
        f"WHERE p.id IN ({marks})", ids).fetchall()
    pets = [int(r["id"]) for r in rows if r["kind"] == "pet"]
    pet_covers = pet_cover_ids(conn, pets) if pets else {}
    out: dict[int, SimilarPerson] = {}
    for r in rows:
        pid = int(r["id"])
        if r["kind"] == "pet":
            aid = pet_covers.get(pid)
            out[pid] = SimilarPerson(id=pid, name=r["name"], kind="pet", species=r["species"],
                                     cover_url=animal_url(aid) if aid else None)
            continue
        fid = int(r["cover_face_id"]) if r["cover_ok"] else None
        if fid is None:
            hit = conn.execute(
                f"SELECT f.id FROM faces f JOIN items i ON i.id = f.item_id AND {LIVE_COVER} WHERE f.person_id = ? "
                "ORDER BY f.quality DESC LIMIT 1", (pid,)).fetchone()
            fid = int(hit[0]) if hit else None
        out[pid] = SimilarPerson(id=pid, name=r["name"], kind="person", cover_url=face_url(fid) if fid else None)
    return out


def _shown(conn: sqlite3.Connection, ids: list[int], kinds: set[str]) -> set[int]:
    """Those of the people the People page lists: a name, a kind the source can be, and an item outside the vault
    and the trash (or no item at all), as list_people's (item_count > 0 OR any_count = 0)."""
    out: set[int] = set()
    if not ids or not kinds:
        return out
    live = "JOIN items i ON i.id = t.item_id AND i.deleted_at IS NULL AND i.nsfw = 0"
    kind_marks = ",".join("?" * len(kinds))
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        out.update(int(r[0]) for r in conn.execute(
            f"SELECT p.id FROM persons p WHERE p.id IN ({','.join('?' * len(chunk))}) AND p.name IS NOT NULL "
            f"AND p.kind IN ({kind_marks}) AND ("
            f"EXISTS (SELECT 1 FROM faces t {live} WHERE t.person_id = p.id) OR "
            f"EXISTS (SELECT 1 FROM bodies t {live} WHERE t.person_id = p.id) OR "
            f"EXISTS (SELECT 1 FROM animals t {live} WHERE t.person_id = p.id) OR "
            f"EXISTS (SELECT 1 FROM item_tags t {live} WHERE t.person_id = p.id AND t.relation = 'in') OR NOT ("
            "EXISTS (SELECT 1 FROM faces WHERE person_id = p.id) OR EXISTS (SELECT 1 FROM bodies WHERE person_id = p.id) "
            "OR EXISTS (SELECT 1 FROM animals WHERE person_id = p.id) "
            "OR EXISTS (SELECT 1 FROM item_tags WHERE person_id = p.id AND relation = 'in')))",
            [*chunk, *sorted(kinds)]))
    return out


@router.post("/api/people/similar", response_model=SimilarOut)
def similar_people(body: SimilarIn, conn: sqlite3.Connection = Depends(get_db),
                   session: SessionInfo = Depends(require_user)) -> SimilarOut:
    if body.uids and body.person_id is None:
        raise HTTPException(status_code=400, detail="Say whose boxes on these items")
    _check_boxes_vault(conn, session, "faces", body.face_ids)
    _check_boxes_vault(conn, session, "bodies", body.body_ids)
    _check_boxes_vault(conn, session, "animals", body.animal_ids)
    item_ids = _item_ids(conn, body.uids)
    _check_items_vault(conn, session, item_ids)
    around: int | None = None
    if body.around_uid:
        row = conn.execute("SELECT id FROM items WHERE uid = ?", (body.around_uid,)).fetchone()
        if row is not None:
            _check_items_vault(conn, session, [int(row[0])])
            around = int(row[0])
    kinds: set[str] = set()
    if body.person_id is not None:
        person = _visible_person_row(conn, session, int(body.person_id))
        kinds.add(str(person["kind"] or "person"))
    if body.face_ids or body.body_ids:
        kinds.add("person")
    if body.animal_ids:
        kinds.add("pet")
    if around is not None:
        kinds |= {"person", "pet"}
    if body.uids and not item_ids:
        return SimilarOut(items=[], scores={})
    ranked = similar.rank(conn, similar.Source(
        face_ids=list(body.face_ids), body_ids=list(body.body_ids), animal_ids=list(body.animal_ids),
        person_id=body.person_id, item_ids=item_ids, around_item_id=around))
    shown = _shown(conn, [o.person_id for o in ranked], kinds)
    ranked = [o for o in ranked if o.person_id in shown]
    top = ranked[:body.limit]
    people = _people(conn, [o.person_id for o in top])
    return SimilarOut(
        items=[SimilarOption(person=people[o.person_id], score=round(o.score, 4), likely=o.likely,
                             basis=o.basis, reason=REASONS.get(o.basis, ""), first=o.first)  # type: ignore[arg-type]
               for o in top if o.person_id in people],
        scores={o.person_id: round(o.score, 4) for o in ranked[:MAX_SCORES]},
    )
