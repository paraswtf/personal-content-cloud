# Integration notes: analysis area

Files: `backend/app/analysis/{pipeline,quality,categories,faces,bodies,guess,cluster,index}.py`,
`backend/app/api/people.py`, `backend/tests/test_analysis_*.py`.

## Wiring the integrator has to do

1. `app/main.py`: `app.include_router(people.router)` with no extra prefix. The router already carries
   `prefix="/api"` and `dependencies=[Depends(require_user)]`, because it spans `/api/people`,
   `/api/faces`, `/api/bodies`, `/api/guesses`, `/api/review/guesses`, `/api/suggestions`,
   `/api/items/{uid}/tags` and `/api/recluster`. Mount it before any catch-all `/api/{rest}` route.
2. `GET /api/review/guesses` lives in `api/people.py`. `api/library.py` must not define it too.
3. Worker dispatch (already matches `app/worker.py` HANDLERS): `analyze` -> `pipeline.analyze_item(conn,
   item_id, version)`, `guess` -> `guess.propagate(conn, person_id, item_id_or_None)`, `recluster` ->
   `cluster.recluster_unassigned(conn)`. `pipeline.mark_failed(conn, item_id, version)` exists as well;
   `ingest.mark_analysis_failed` does the same job, either is fine.
4. Guess job payloads are always `{"person_id": int, "item_id": int | null}` in that key order (unique
   enqueue compares payload text).

## Contract points to check

- `items.category_scores` holds the group probabilities plus `explicit_p`, `doc_p`, `screenshot_p`,
  `text_density` (floats) and, when a model was missing, `skipped: [str]` (ARCHITECTURE.md section 8).
  `schemas.ItemDetail.category_scores` is `dict[str, float]`: any item analysed without a model has the
  `skipped` list, so every reader must drop non-float entries. `api/library.py` already does
  (`_float_scores`); search or documents code that builds an ItemDetail from the row must do the same.
  `quality_scores` (`dict[str, Any]`) also carries `skipped`.
- `PCC_NO_MODELS=1` turns every model off inside the analysis pipeline (`pipeline.models_disabled()`,
  `bodies.available()`), so worker.drain() in other areas' tests never loads SigLIP, ArcFace, NSFW or YOLOX.
  `app.ml.models.present()` does not check the variable; checking it centrally there would cover other
  callers too.
- `api/common.person_out` counts items and faces including NSFW items. `api/people.py` does not use it: it
  has its own `person_outs()` with visible-only counts (live and not NSFW), and the people list hides persons
  whose only items are in the vault or the trash. If other routers show PersonOut (search, item detail),
  consider switching them to `app.api.people.person_outs` so counts match.
- `app/persons.py` (core) still inserts into a `persons.hidden` column that the new schema does not have.
  Analysis does not import it (helpers are in `app.analysis.cluster`: `new_person`, `find_person_by_name`,
  `find_or_create_person`, `clean_person_name`, `prune_empty_persons`, `clear_guesses_for`). Delete it or fix
  it if anything else imports it.
- `PATCH /api/people/{id}` with a name another person already has merges this person into that one and
  returns the other person's PersonOut (different `id`). The person page should navigate to the returned id.
- `POST /api/people {name}` reuses an existing person with the same name instead of creating a duplicate.
- `FaceListItem.url` is the face crop, `thumb_url` is the item's thumbnail (FaceGrid uses `url`).
- FTS: label, tag and person-name changes call `app.documents.extract.refresh_fts(conn, item_id)` lazily
  (ImportError ignored). Renaming, merging or deleting a person refreshes every item of that person.
- Video bodies are not detected (person boxes run on photos only); video faces get torso embeddings from
  their best frame, so clothes guessing works for video faces.
- `faces`/`bodies` of `pdf`/`document` items are not detected (the documents area crops ID photos itself).
