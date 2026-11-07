# Integration notes: core

Decisions the core area took where the contract was open, and changes other areas should know about.
No shared contract file was edited.

## Interfaces that differ from or extend the listed signatures

- `app.media.probe.probe(path, filename=None)`. Staged uploads are `<uid>.part`, so the original file
  name is needed to recognise plain text (.txt/.md/.csv), RAW formats that share the TIFF header (NEF,
  ARW, PEF, SRW) and the extension of 'other' files. Without it those files come back as 'other'.
- `MediaInfo` gains `page_count`, `raw`, `reason` (why a known type was kept as 'other') and the
  properties `text_capable` and `visual`. `width`/`height` are `None` for 'other'.
- `probe` returns `None` only for an empty or unreadable file. Unknown files and files of a known type
  that do not decode (broken JPEG, damaged PDF or RAW) come back as kind 'other' and are stored byte for
  byte without previews. A sniffed type keeps its MIME and extension; unknown files get
  `application/octet-stream` and the uploaded suffix when it is `[a-z0-9]{1,8}`, else `.bin`.
- `app.media.video.extract_frame(...)` returns a BGR `np.ndarray` as the contract says;
  `extract_frame_image(...)` returns the PIL RGB image (used by fingerprints, posters and /files/frame).
- `app.media.documents`: every function takes an optional `ext=` (needed for files without a suffix).
  Extra helpers: `render_page(path, kind, uid, version, page_no, *, max_side, ext=None)`,
  `page_image(...)`, `logical_pages(path, ext=None)`, `text_page_image(text, max_side)`,
  `rendered_pages(uid, version)`, `probe_document(path, kind, ext)`, `sniff_document(...)`.
  `PDFIUM_LOCK` (an RLock) must be held around every direct pypdfium2 call: pdfium is not thread
  safe and the text worker, the main worker and request threads all render pages.
- Office renders and `text_layer` come from one pagination function, so page N's render shows page
  N's text. docx and plain text: about 3000 characters or 60 wrapped lines per page; xlsx: one per
  sheet; pptx: one per slide. At most 1000 logical pages; plain text is read up to 16 MB.
- `app.media.images.load_oriented(path, max_side=None, *, raw=None)` routes RAW files (by
  extension, or `raw=True`) through `app.media.raw.load_raw` (half-size demosaic).
- `app.media.dedup`: `find_similar` plus `link_into_group`, `regroup`, `group_member_ids`, `set_best`.
- `app.ingest.item_payload(item_id, version)` builds the analyze/text/video_preview payload
  `{"item_id": int, "version": int}`; guess is `{"person_id": int, "item_id": int | null}`. Unique
  enqueue compares the JSON text, so other areas should build payloads the same way.
- `app.ingest`: also `mark_analysis_failed`, `mark_text_failed`, `mark_preview_failed`,
  `requeue_failed_upload`, `expire_trash`, `sweep_orphan_files`.
- `app.worker.housekeeping(conn)` is public (used by POST /api/system/housekeeping).
- `app.persons.drop_empty_unnamed(conn, person_ids)` (new); persons rows no longer use `hidden`.

## Pipeline behaviour

- Ingest renders every page of a document up to `settings.page_render_max` (200) at
  `page_render_size`, because `documents.extract` reads pages 2..SCAN_PAGES from disk only. Cost: a
  200-page PDF holds the main queue for roughly 20 to 60 s. A render that is missing later is made on
  demand by `GET /files/page/{uid}/{n}` (striped lock) or `documents.render_page`, so `urls.pages`
  never 404s. A document whose renders fail (damaged but probeable) is stored without previews;
  only ffmpeg failures and a full disk make the ingest job retry.
- pdf and document items are inserted with `category='documents'` (rule 1 of section 9) so listings
  are right before analysis runs.
- 'other' items are inserted with `analysis_status='done'` and get no analyze job (nothing to analyse).
- After the item insert, ingest calls `app.documents.extract.refresh_fts(conn, item_id)` (lazily) so
  every item is findable by file name at once. `purge` calls `remove_fts` inside its transaction and
  falls back to `DELETE FROM items_fts WHERE rowid=?` when the documents module is missing.
- A re-upload of an exact duplicate that is in the trash answers 'duplicate' with a message saying so;
  it does not restore the item.
- Worker: a handler whose module or function cannot be imported raises `HandlerMissing`; the job is
  marked 'error' without retries and the item's status is left 'pending', so `recover_on_startup`
  queues it again on the next start. Every other exception retries with backoff and then marks the
  item's status 'error'.
- `/api/review/guesses` is not in library.py (people.py serves it).

## HTTP behaviour

- Routers are plain `APIRouter()`s without a prefix; main.py mounts them (auth at /api/auth, library
  and people at /api, documents at /api/documents, search at /api/search, uploads at /api/uploads,
  files at /files, system at /api/system). `main.mount_prefix` also accepts routers whose paths already
  carry their prefix. A router module that is missing or fails to import is logged and skipped.
- `/api` responses get `Cache-Control: private, no-store` unless they set their own (section 4).
- `/files`: `private, max-age=31536000, immutable` when `?v=` matches, else `private, no-cache`; crops
  `private, max-age=86400`; anything of an NSFW item `private, no-store` and 403 `vault_locked` while
  the vault is locked (thumbs, previews, originals, pages, doc photos, face and body crops, frames).
- Inline PDFs from `/files/original` get `X-Frame-Options: SAMEORIGIN`, an inline Content-Disposition
  and `default-src 'none'; sandbox allow-same-origin`, so the viewer can frame them.
- `/api/items?vault=1` lists only NSFW items (that is how `visible_sql` works); `/api/trash` includes
  NSFW items only with `vault=1`.
- `POST /api/items/{uid}/restore` returns the ItemDetail (as `frontend/src/lib/api.ts` expects); delete,
  reprocess, set-best, purge and housekeeping return `Ok`; bulk and trash/empty return `BulkOut`.
- `/api/stats.items` counts live, non-NSFW items; `nsfw` counts live NSFW items separately.
- GET+HEAD routes in files.py produce FastAPI "Duplicate Operation ID" warnings when the OpenAPI
  schema is built (harmless, same as the hub).

## Suggested contract changes (not applied)

- The owner's choice of a duplicate group's best (`POST /api/duplicates/{uid}/best`) is not sticky:
  a strictly better copy linked later becomes the best. A column such as `items.dup_best_source
  CHECK ('auto','user')` would let `dedup.regroup` keep a user choice.
- `DejaVuSans.ttf` is not installed on this machine; office page renders fall back to FreeSans
  (`/usr/share/fonts/truetype/freefont`), then Pillow's default font.

## Answers to other areas' notes

- people router (prefix "/api" on the router itself): mounted as is; `mount_prefix` sees the paths
  already start with /api. Routers are included before the `/api/{rest}` catch-all.
- frontend-shell: real files in `frontend/dist` (`/sw.js`, `/manifest.webmanifest`, `/icons/*`) are
  served as files, `sw.js` as `text/javascript` and the manifest as `application/manifest+json`, both
  with `Cache-Control: no-cache`. `POST /share-target` answers 303 to `/upload?shared=1`. No CSP or
  Permissions-Policy is added to app pages (only to /files responses).
- search: main.py, ingest.py no longer import `app.search.indexer`; `api/public.py` and
  `api/admin.py` are deleted.
- analysis: `app/persons.py` no longer uses `persons.hidden`.
- ml: the core system test sets `gpu_enabled="false"` and calls `app.ml.runtime.reset()` around
  `GET /api/system`, so it never starts CUDA.
