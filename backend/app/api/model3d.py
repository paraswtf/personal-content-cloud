"""3D heads of a person (docs/ARCHITECTURE.md section 36, app.analysis.model3d). Session required on every route.

GET    /api/people/{id}/model3d   Model3dOut: whether a model can be made (`available`, with a plain `reason` when
                                  not), the running job's `progress`, the last `error`, and the model's `looks` with
                                  their mesh and picture URLs. Reading it never starts anything. 404 for a pet.
POST   /api/people/{id}/model3d   Model3dIn {max_looks 1..12 = 6, detail 1..3 = 2, framing 1.8..3.2 = 2.6}
                                  -> 202 {started: true}. 409 `busy` (one generation at a time), 409
                                  `generator_missing`, 422 `not_enough_photos`.
POST   /api/people/{id}/model3d/stop
                                  stops the person's running job -> {stopped}; the model in use, if any, stays.
                                  The job ends within about a second (`stopping` in the status until then).
DELETE /api/people/{id}/model3d   removes the model and stops the person's running job -> Ok. The model is not
                                  listed or served from that moment, also while the job is still on its way out.
GET    /files/model3d/{id}/{rev}/{name}
                                  a look's mesh (model/gltf-binary) or the picture it was made from (image/jpeg),
                                  only for the model in use and a look that is still listed (its source photo is in
                                  the library, not in the trash or the vault). The rev is in the path, so the
                                  response is cached for good ("private, max-age=31536000, immutable").
These writes are not owner edits of the library, so they are not undoable actions (app.undo.NOT_UNDOABLE).
"""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.analysis import looks, model3d
from app.api.deps import get_db, require_user
from app.auth.sessions import SessionInfo
from app.schemas import Ok

router = APIRouter(tags=["model3d"], dependencies=[Depends(require_user)])

IMMUTABLE = "private, max-age=31536000, immutable"
CSP = "default-src 'none'; sandbox"
MEDIA = {"glb": "model/gltf-binary", "jpg": "image/jpeg"}


class Model3dIn(BaseModel):
    max_looks: int = Field(default=looks.LOOKS_DEFAULT, ge=looks.LOOKS_MIN, le=looks.LOOKS_MAX)
    detail: int = Field(default=model3d.DETAIL_DEFAULT, ge=model3d.DETAIL_MIN, le=model3d.DETAIL_MAX)
    framing: float = Field(default=looks.FRAMING_DEFAULT, ge=looks.FRAMING_MIN, le=looks.FRAMING_MAX)


class Model3dProgress(BaseModel):
    step: str
    done: int
    total: int


class Look3dOut(BaseModel):
    key: str
    label: str
    age: float | None = None
    age_band: str
    hair: float | None = None
    hair_label: str | None = None
    facial_hair: float | None = None
    facial_hair_label: str | None = None
    photos: int
    taken_at: str | None = None
    item_uid: str
    picture_url: str
    mesh_url: str
    bytes: int
    vertices: int


class Model3dOptions(BaseModel):
    max_looks: int
    detail: int
    framing: float


class Model3dOut(BaseModel):
    available: bool
    reason: str | None = None
    generator: str | None = None
    usable_photos: int
    state: str  # none | running | done | error
    busy: bool = False  # another person's model is being generated
    stopping: bool = False  # Stop or Delete was pressed and the running job is on its way out
    progress: Model3dProgress | None = None
    error: str | None = None
    generated_at: str | None = None
    options: Model3dOptions | None = None
    looks: list[Look3dOut] = []


class Model3dStarted(BaseModel):
    started: bool


class Model3dStopped(BaseModel):
    stopped: bool


def _person(conn: sqlite3.Connection, session: SessionInfo, person_id: int) -> sqlite3.Row:
    """The person, as the people routes see them (404 when hidden or gone); 404 for a pet."""
    from app.api.people import _visible_person_row

    row = _visible_person_row(conn, session, person_id)
    if row["kind"] == "pet":
        raise HTTPException(status_code=404, detail="Person not found")
    return row


@router.get("/api/people/{person_id}/model3d", response_model=Model3dOut)
def get_model3d(person_id: int, conn: sqlite3.Connection = Depends(get_db),
                session: SessionInfo = Depends(require_user)) -> Model3dOut:
    _person(conn, session, person_id)
    return Model3dOut.model_validate(model3d.status(conn, person_id))


@router.post("/api/people/{person_id}/model3d", response_model=Model3dStarted, status_code=202)
def start_model3d(person_id: int, body: Model3dIn | None = None, conn: sqlite3.Connection = Depends(get_db),
                  session: SessionInfo = Depends(require_user)) -> Model3dStarted:
    _person(conn, session, person_id)
    options = (body or Model3dIn()).model_dump()
    try:
        model3d.start(conn, person_id, options)
    except model3d.Busy:
        raise HTTPException(status_code=409, detail="busy") from None
    except model3d.GeneratorMissing:
        raise HTTPException(status_code=409, detail="generator_missing") from None
    except model3d.NotEnoughPhotos:
        raise HTTPException(status_code=422, detail="not_enough_photos") from None
    return Model3dStarted(started=True)


@router.post("/api/people/{person_id}/model3d/stop", response_model=Model3dStopped)
def stop_model3d(person_id: int, conn: sqlite3.Connection = Depends(get_db),
                 session: SessionInfo = Depends(require_user)) -> Model3dStopped:
    _person(conn, session, person_id)
    return Model3dStopped(stopped=model3d.cancel(person_id))


@router.delete("/api/people/{person_id}/model3d", response_model=Ok)
def delete_model3d(person_id: int, conn: sqlite3.Connection = Depends(get_db),
                   session: SessionInfo = Depends(require_user)) -> Ok:
    _person(conn, session, person_id)
    model3d.remove(conn, person_id)
    return Ok()


@router.api_route("/files/model3d/{person_id}/{rev}/{name}", methods=["GET", "HEAD"])
def model3d_file(person_id: int, rev: str, name: str, conn: sqlite3.Connection = Depends(get_db),
                 session: SessionInfo = Depends(require_user)) -> FileResponse:
    _person(conn, session, person_id)
    path = model3d.file_path(person_id, rev, name)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    _doc, shown = model3d.listed_looks(conn, person_id)
    if not any(name in (lk.get("mesh"), lk.get("picture")) for lk in shown) or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type=MEDIA[name.rsplit(".", 1)[1]], headers={
        "cache-control": IMMUTABLE, "x-content-type-options": "nosniff", "content-security-policy": CSP})
