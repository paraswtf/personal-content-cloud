# Personal Content Cloud: architecture

A single-user content cloud for photos, videos, PDFs and other documents, derived from the
SICSR alumni hub (`docs/alumni-hub-reuse-map.md` maps what was reused). Everything runs on this
machine: models on the RTX 2080 through onnxruntime (CPU fallback), SQLite on local disk, nothing
sent to outside services. Originals are kept byte for byte; downloads return the original bytes.

Decisions taken with the owner on 2026-11-07:

| Topic | Decision |
|---|---|
| ML runtime | Fully local, GPU when present (onnxruntime CUDA provider, CPU fallback) |
| "Same location" without geotags | Same scene by sight: whole-image embeddings compared; EXIF GPS is extra evidence when present |
| NSFW | Locked vault: never shown in gallery, search, people or previews; a separate section asks for the password or a passkey again and stays unlocked 15 minutes |
| Login | Passkey (WebAuthn, discoverable) with password fallback; single account created on first run |
| Parsed document types | Indian IDs, money, education and work, health and insurance, event passes (with Apple Wallet export) |
| Mis-clicks | Blurry or shaken, too dark or blown out, pocket and ground shots, burst near-duplicates; flagged into a review bin, never deleted automatically |
| Ways in | Browser upload (drag and drop), PWA share target, in-app camera, office documents (text extraction and text-based preview) |
| Deployment | This server, `scripts/run.sh` with uv and a local Node build, declared in `.claude/devservers.json` |

Not in scope: bulk import from a server folder, watch folders, geotag cards and map downloads,
visitor uploads, profile links, exports with embedded metadata, the hub's change history and undo
engine. Deleted items go to a 30-day trash instead (soft delete, restore, purge).

## 1. Stack and process model

- Backend: Python 3.12 (uv), FastAPI, pydantic v2, SQLite (WAL, raw SQL, FTS5, JSON1), onnxruntime-gpu
  with CUDA 12 libraries from pip (`onnxruntime-gpu[cuda,cudnn]`), OpenCV, Pillow + pillow-heif, rawpy,
  pypdfium2, python-docx, openpyxl, python-pptx, rapidocr (PP-OCRv4 ONNX), zxing-cpp, webauthn (py_webauthn),
  argon2-cffi, cryptography, ffmpeg/ffprobe as subprocesses.
- Frontend: React 19, react-router 7, TanStack Query 5, Tailwind 4, Vite 7, TypeScript strict, lucide-react,
  hash-wasm. Installable PWA with a share target and an in-app camera.
- One uvicorn process (run exactly one worker: locks, indexes and rate limits live in process memory).
  Background threads inside it: `pcc-worker-main`, `pcc-worker-text`, `pcc-worker-video` (one per
  queue). Durable `jobs` table as in the hub. A flock on `DATA_DIR/.instance.lock` keeps a second process
  off the data directory.
- Every route except the auth bootstrap routes requires a logged-in session. There are no public pages.

## 2. Repository layout

```
backend/
  app/
    main.py            app factory, lifespan, middleware (security headers, gzip, SPA fallback)
    config.py          settings from env (section 15)
    db.py              connect(), tx(), kv helpers, init_db() applying schema.sql + migrations
    schema.sql         the whole schema (section 5)
    storage.py         data dir layout and atomic file helpers
    util.py            ids, names, cursors, time
    urls.py            URL builders for /files
    auth/
      passwords.py     argon2 hashing, rate limit
      account.py       the single account row
      sessions.py      session table, cookie, vault unlock window
      passkeys.py      WebAuthn registration and assertion ceremonies
    api/
      deps.py          get_db(), require_user(), optional_user(), require_vault(), want_vault(), check_csrf()
      auth.py          /api/auth/*
      library.py       /api/items, /api/trash, /api/review, /api/duplicates, /api/stats
      people.py        /api/people, /api/faces, /api/bodies, /api/guesses, /api/suggestions, tags
      documents.py     /api/documents/*
      search.py        /api/search
      uploads.py       /api/uploads/*
      files.py         /files/*
      system.py        /api/system/*
      common.py        row -> schema helpers, cursor validation, filters
    schemas.py         pydantic models (section 13 shapes)
    jobs.py            queue table, priorities, queue map
    worker.py          queue threads, dispatch, retry, housekeeping
    ingest.py          upload -> item (probe, hash, dedup, place, derivatives)
    media/             probe.py, images.py, video.py, hashing.py, dedup.py, raw.py, documents.py (pdf/office probing and rendering)
    ml/
      runtime.py       onnxruntime session factory: providers, GPU detection, thread settings
      models.py        model registry: names, files, sizes, sha256, download URLs
      siglip.py        SigLIP engine (from hub search/engine.py) + zero-shot helpers
      faces.py         SCRFD + ArcFace engine (from hub faces/engine.py)
      nsfw.py          ViT NSFW classifier
      persons.py       YOLOX person detector
      ocr.py           rapidocr wrapper
    analysis/
      pipeline.py      the analyze job: orchestrates the stages below for one item
      quality.py       blur, exposure, pocket shot, burst detection
      categories.py    zero-shot category scoring and the decision rules
      faces.py         face records, video tracking (from hub faces/pipeline.py)
      bodies.py        person boxes, torso crops, clothes embeddings
      guess.py         person guessing from the same hour + same scene + same clothes
      place*.py        photos taken moments apart lined up, people followed between them (section 35)
      cluster.py       face clustering, merge, suggestions (from hub faces/cluster.py)
      index.py         in-memory face index (from hub faces/index.py)
    documents/
      extract.py       text extraction (pdf text layer, OCR, office, plain text) -> pages + item_text + FTS
      detect.py        document type detection (keywords, id patterns, layout)
      fields.py        field models and helpers (dates, checksums: Verhoeff, MRZ)
      parsers/         one module per document type (aadhaar.py, pan.py, passport.py, ...)
      barcodes.py      zxing-cpp wrapper
      photo.py         ID photo crop
    search/
      service.py       semantic matrix + FTS query merge
    wallet/
      pkpass.py        Apple Wallet pass builder and signer
  tests/               pytest (section 16)
  pyproject.toml, uv.lock
frontend/              Vite app (section 14)
gen3d/                 the 3D head generator: a separate program with its own Python environment (section 36)
scripts/
  run.sh               sync env, download models, build frontend, start
  download_models.py   all model downloads, pinned and checksummed
docs/
  ARCHITECTURE.md      this file
  alumni-hub-reuse-map.md
.claude/devservers.json
```

## 3. Data directory

Two roots (section 31). `DATA_DIR` (default `<repo>/data`) holds the database and everything the app
makes; `ORIGINALS_DIR` (unset by default, then the same folder) holds the bytes the owner uploaded:
`originals/`, `staging/` and `failed/`. Shard `sh = uid[:2]`. Versions as in the hub (an item's
`version` only changes when the user replaces a file; near-duplicates never replace anything).

| Path | Content |
|---|---|
| `cloud.db` | SQLite database |
| `secret.key` | 32 random bytes, mode 0600, made on first use: signs the share unlock cookies (section 26); sessions use hashed random tokens instead |
| `.instance.lock` | flock |
| `originals/<sh>/<uid>.v<n><ext>` | untouched upload bytes (under `ORIGINALS_DIR`) |
| `thumbs/<sh>/<uid>.v<n>.webp` | 480 px thumbnail (page 1 for documents, poster for videos) |
| `previews/<sh>/<uid>.v<n>.webp`, `.sm.webp` | 2048 / 1280 px previews (images, video posters, document page 1); no `.sm.webp` is stored when the large preview already fits 1280 px (it would be the same bytes): `/files/preview-small` then serves the large file with the same cache headers, and makes a missing small preview from the large one only for bigger pictures |
| `pages/<sh>/<uid>.v<n>/<page>.webp` | document page renders, 1600 px long side (PDF pages; office documents get text-rendered pages) |
| `videos/<sh>/<uid>.v<n>.mp4` | 720p H.264 streaming copy |
| `faces/<id//1000>/<id>.webp` | face crops 192 px |
| `bodies/<id//1000>/<id>.webp` | person crops 256 px long side |
| `animals/<id//1000>/<id>.webp` | pet crops (section 22) |
| `docphoto/<sh>/<uid>.v<n>.webp` | photo cropped out of an ID document |
| `staging/<uid>.part` | uploads in progress (under `ORIGINALS_DIR`) |
| `failed/<uid>.part` | uploads whose ingest failed for good (under `ORIGINALS_DIR`); deleted after `TRASH_DAYS`; Settings > System > Retry failed uploads queues them again |
| `learned/` | the learned models (section 18) |
| `people3d/<person id>/` | a person's 3D heads (section 36): `current` names the folder in use, `<stamp>/model.json`, `<key>.glb` and `<key>.jpg` per look; `work/` while one is made, `error.json` after a failed run |
| `wallet/` | `cert.pem`, `key.pem`, `wwdr.pem` placed by the owner (never written by the app) |
| `tmp/` | wiped at startup |
| `site/` | copy of the frontend build taken at startup, the site that is served (section 14) |

Staging and originals share one filesystem (hard link placement); an original placed from a scratch
file on another disk is copied to a scratch name beside its destination and renamed from there
(`storage.move_into_place` handles `EXDEV`). `storage.rel(path)` names a file by its root and top folder
whichever root holds it, so `_is_referenced` works for both. The orphan sweeper walks originals, thumbs,
previews, pages, videos, faces, bodies, animals, docphoto (`storage.swept_bases()`: each under its root, plus
`DATA_DIR/originals` while files of the earlier single-folder layout are still being moved, section 31) and
deletes files older than an hour that no row references (names parsed by regex; `_is_referenced` guards
every delete). `storage.disk_usage()` and `Stats.disk` report the disk that receives uploads (the
originals root); `storage.has_room_for(n)` also asks the data root to keep `MIN_FREE_DISK_MB` for the
derived files when it is another disk.

Derived files are kept once (2026-11-08): a Live Photo clip (companion, section 20) keeps no thumb or
previews of its own, since the library never shows it; `/files/thumb`, `/files/preview` and
`/files/preview-small` of a companion serve its still's pictures and the upload list shows the still's
thumb. Purge removes face, body and pet crops. `ingest.tidy_derivatives` removes what an earlier version
stored (companion pictures, small previews that repeat the large one) once per `ingest.TIDY_VERSION` (kv
`derivatives_tidy`), from housekeeping. The 3D heads in `people3d/` are not part of the orphan sweep: housekeeping
calls `model3d.sweep`, which removes the folders of people who no longer exist (section 36).

## 4. Authentication and sessions

- One account row (`account`, id 1). `GET /api/auth/state` reports `setup_required` until it exists.
  `POST /api/auth/setup` creates it (username 1..64, password 10..200 chars, argon2id), only while no
  account exists (409 after).
- Password login: `POST /api/auth/login`. Rate limit: 10 failures per 10 minutes per IP (in memory).
- Passkeys (`webauthn` 3.x): registration from Settings (`resident_key=required`, `user_verification=preferred`),
  login with a discoverable credential (`POST /api/auth/passkeys/login/options` returns options and a
  `challenge_id`; the challenge is stored in `challenges` with a 5-minute expiry; `verify` creates the
  session). `RP_ID` and `ORIGIN` come from env; when unset they are derived from the request
  (`Host` without port; scheme from `X-Forwarded-Proto`). `ORIGIN` may list several origins, comma-separated:
  a ceremony from any of them verifies (`passkeys.expected_origins`), and the first is the server's own
  address (`passkeys.home_origin`). The dev server manifest sets `RP_ID` from `{host}` and `ORIGIN` from
  `{origins}` (its own URL, then its custom domains). The RP ID stays the review host, so existing passkeys
  keep working, and on a custom domain the browser uses them through WebAuthn related origins: the platform
  publishes `https://{host}/.well-known/webauthn` listing those domains, and the browser checks it before the
  ceremony. `GET /api/auth/rp` (no login) returns `{rp_id, home_url}`. On an address that is neither the RP ID
  nor under it, the sign-in, setup, vault unlock and add-passkey screens ask
  `PublicKeyCredential.getClientCapabilities()`; when the browser does not report `relatedOrigins`, they show
  a note linking to the same page at `home_url` instead of a passkey button that would fail
  (`usePasskeyPlace`, `PasskeyElsewhere`).
  Registering and removing a passkey need an unlocked vault (403 `vault_locked`): the session must have
  re-entered the password or used a passkey within `VAULT_UNLOCK_MIN`, so a stolen or left-open session cannot
  enrol its own authenticator. `passkeys/login/options` has its own limit of 30 calls per minute per IP (429
  `too_many_attempts`, apart from the login failure budget) and at most 500 open login challenges are kept.
- Sessions: `sessions` row with `token_hash = sha256(token)`; cookie `pcc_session` = token, HttpOnly,
  SameSite=Lax, Secure when https, path `/`, 30 days sliding (`last_seen_at` bumped at most once a
  minute; `expires_at = last_seen + SESSION_DAYS`). When the row slides, the response sets the cookie again
  with the same token and a fresh max-age (`SessionCookieMiddleware`), so the browser's copy slides too.
  Revocable one by one or all at once.
- CSRF: every non-GET request must carry `X-Requested-With: pcc` (403 otherwise). Same rule for
  chunk PUTs. The upload engine and `api.ts` send it on every request.
- Vault (NSFW): `POST /api/auth/vault/unlock` with `{password}` or `{credential}` (passkey assertion
  from `POST /api/auth/passkeys/assert/options`) sets `sessions.vault_until = now + VAULT_UNLOCK_MIN`.
  `require_vault()` answers 403 `{"detail":"vault_locked"}` when the window has passed. `POST
  /api/auth/vault/lock` clears it. Every list, detail, search and `/files` route hides NSFW items
  unless the request passes `vault=1` and the session is unlocked.
- 401 `{"detail":"login_required"}` on every protected route; the frontend redirects to `/login`.
  `GET /api/openapi.json` (the only schema route; there is no docs UI) needs a session too.
- Security headers on every response: nosniff, `Referrer-Policy: same-origin`, `X-Frame-Options: DENY`,
  and `Cache-Control: private, no-store` on `/api`. `/files` responses are `private, max-age=31536000,
  immutable` when `?v=` equals the plain current version of a non-vault item, else `private, no-cache`
  (vault items are `no-store`). Vault items' URLs carry `<version>n` as their cache key
  (`api.common.url_version`), so flagging an item changes its URLs and a thumb cached before the flag is
  never shown again; `/files` accepts any `v`.

## 5. Database schema

Schema changes after the first release are additive. A new column goes into its `CREATE TABLE` in
schema.sql and into `db.MIGRATED_COLUMNS`; `db.init_db` adds the columns an existing table lacks
(`ALTER TABLE ... ADD COLUMN`) before it applies schema.sql, so an index in schema.sql may cover a
migrated column and a database from any earlier build upgrades in place at startup. New tables and
indexes need nothing more than their `CREATE ... IF NOT EXISTS` in schema.sql.
`tests/test_db_upgrade.py` starts from the schema of the last build before the first migrated column
(`tests/fixtures/schema_before_migrations.sql`) and checks that the upgraded database has the same
tables, columns, foreign keys and indexes as a fresh one.

The full DDL is `backend/app/schema.sql`. Summary of tables and the columns code depends on:

- `kv(key, value)`: `schema_version`, `faces_version`, `search_version`, `item_id_mark`, `vmark:<id>`,
  `recluster_max_face_id`.
- `account(id=1, username, username_key, password_hash, created_at, updated_at)`.
- `passkeys(id, credential_id BLOB UNIQUE, public_key BLOB, sign_count, transports, name, aaguid, backup_eligible, created_at, last_used_at)`.
- `sessions(id, token_hash UNIQUE, created_at, last_seen_at, expires_at, user_agent, ip, vault_until, revoked_at)`.
- `challenges(id, kind CHECK ('register','login','assert'), challenge BLOB, created_at, expires_at, used INT)`.
- `uploads(id, uid UNIQUE, source CHECK ('web','share','camera','api'), client_id, original_filename, size_bytes,
  received_bytes, mime, sha256, status CHECK ('uploading','queued','processing','stored','duplicate','rejected',
  'error','cancelled'), message, item_id FK SET NULL, remote_addr, created_at, updated_at)`.
- `items`: identity (`id` never reused, `uid` UNIQUE, `kind CHECK ('image','video','pdf','document','other')`,
  `version`, `original_filename`, `ext`, `mime`, `sha256 UNIQUE`, `size_bytes`); picture and stream
  (`width`, `height`, `duration_s`, `bitrate`, `video_codec`, `fps`, `has_audio`, `page_count`, `animated`);
  capture (`taken_at` UTC, `exif_datetime`, `exif_subsec`, `camera`, `gps_lat`, `gps_lon`, `gps_alt`);
  fingerprints (`phash`, `dhash`, `sig`, `frames`); classification (`category`, `category_source CHECK
  ('auto','user')`, `category_scores` JSON, `doc_likeness` REAL, `nsfw` INT, `nsfw_score`, `nsfw_source`,
  `misclick` INT, `misclick_reasons` JSON, `misclick_source`, `quality_scores` JSON); forwards (section 29:
  `received_via`, `forward` INT, `forward_reasons` JSON, `forward_source`, `forward_signals` JSON, `forward_checked`); duplicates
  (`dup_group_id` = id of the group's best item, `dup_role CHECK ('best','copy')`, `dup_detail`);
  state (`analysis_status CHECK ('pending','done','error')`, `text_status CHECK ('none','pending','done','error')`,
  `preview_status CHECK ('none','pending','done','error')`, `face_count`, `body_count`, `favorite`,
  `deleted_at`, `created_at`, `updated_at`).
  Categories: `documents`, `screenshots`, `personal`, `places`, `food`, `pets`, `objects`, `misc`, `nsfw`.
- `item_embeddings(item_id, version, model, frame_index, frame_time_s, embedding BLOB float32[768], created_at, PK(item_id, model, frame_index))`:
  frame 0 is the whole image (page 1 for documents, poster for videos); videos add one row per sampled frame.
- `item_labels(item_id, label, score, PK(item_id,label))`: zero-shot labels for display and filters.
- `faces(id, item_id, person_id, assigned_by CHECK ('auto','user','guess'), x,y,w,h, det_score, size_px, quality,
  frame_time_s, spans, embedding BLOB float32[512], torso_embedding BLOB float32[768] NULL, guess_person_id,
  guess_score, guess_evidence JSON, created_at)`. `assigned_by='user'` is locked; `'guess'` means `person_id`
  is NULL and `guess_person_id` is set until confirmed or rejected.
- `bodies(id, item_id, face_id FK SET NULL, person_id, assigned_by, x,y,w,h, det_score, torso_embedding BLOB,
  guess_person_id, guess_score, guess_evidence, frame_time_s, created_at, source CHECK ('detector','user'), by_guess,
  moved)`. `source 'user'` is a box the owner drew; `by_guess = 1` means the person came from a guess the owner
  confirmed; `moved = 1` means the owner moved or resized the box, and a re-analysis keeps it (section 10).
- `persons(id, name, name_key, cover_face_id, created_at, updated_at)`; `recognition_paused_at` (section 10: the
  guesses of the recognition trained on the person were taken back) came through `MIGRATED_COLUMNS`.
- `item_tags(item_id, person_id, source CHECK ('user','guess'), relation CHECK ('in','related'), created_at,
  PK(item_id, person_id))`; `relation` ('in': they are in the item; 'related': the item is about them without them
  in it, section 10) came through `MIGRATED_COLUMNS`. A tag names a person or a pet (section 22).
- `guess_rejections(kind CHECK ('face','body','animal'), ref_id, person_id, created_at, PK(kind, ref_id, person_id))`.
- `merge_dismissals(person_a, person_b)`.
- `related_dismissals(item_id, person_id, created_at, PK(item_id, person_id))`: "Not related to NAME", a related-tag
  suggestion the owner turned down (section 10); journaled for undo, carried along by merges.
- `animals` (section 22) gains `species_source` (auto|user), `species_scores` (JSON) and `source` (detector|user: a box
  the owner drew) through `MIGRATED_COLUMNS`.
- `item_text(item_id PK, text, source CHECK ('pdf','ocr','office','plain','mixed'), lang, char_count, created_at)`.
- `pages(id, item_id, page_no, width, height, text, text_source CHECK ('layer','ocr','render','none'), ocr_boxes JSON, UNIQUE(item_id, page_no))`.
- `items_fts` (FTS5, `content=''` contentless is not used; external content table is not used either):
  standalone FTS5 with `rowid = item_id` and columns `filename, text, doc, people, labels`, tokenizer
  `unicode61 remove_diacritics 2 tokenchars '<every Mn/Mc mark of U+0900 to U+0DFF>'` (`db.FTS_TOKENIZE`), so a
  Devanagari word with its vowel signs is one token. Rows are rewritten whenever text, document fields, tags or
  labels change (`documents.extract.refresh_fts(conn, item_id)`); the indexed text has zero-width joiners
  removed and Devanagari digits written as ASCII (`extract.fold_for_index`; `search.service` and
  `extract.match_query` fold queries the same way, and search snippets are mapped back onto the original text
  by `extract.restore_snippet`).
  `db.init_db` drops and recreates the table when its stored definition lacks the current tokenizer and sets
  kv `fts_rebuild`; `ingest.recover_on_startup` then re-indexes every item and clears the flag.
- `documents(id, item_id UNIQUE, doc_type, doc_group, confidence, fields JSON, title, doc_date, issuer, person_name,
  barcode, barcode_format, has_photo INT, source CHECK ('auto','user'), created_at, updated_at)`.
  `fields` is `{"<key>": {"value": str, "label": str, "confidence": 0..1}}`.
- `jobs(id, type, queue CHECK ('main','text','video'), payload, status, attempts, last_error, priority, run_after, created_at, updated_at)`.

Tables and columns added by sections 23 to 28: `albums`, `album_items`, `shares`, `share_items`;
`items.place*`, `items.gps_checked`, the `items_gps` index; `uploads.session_id`, `uploads.resource`.

Tables added by section 34 (body traits). They are new tables only, with no CHECK constraints (kind, part, status and
verdict are checked in code) and nothing in `MIGRATED_COLUMNS`, so `test_db_upgrade` is unchanged:
- `body_traits(id AUTOINCREMENT, body_id FK CASCADE, item_id FK CASCADE, kind, part, x,y,w,h, quality, vec BLOB,
  detail BLOB, attrs JSON, crop BLOB, created_at)`: one row per trait region. `kind` is tattoo, mehndi or nails;
  `part` is `forearm_l/r`, `upper_arm_l/r`, `calf_l/r` or `hand_l/r`; the region is normalised. `vec` is
  float32[768] (the DINOv2 CLS of the crop, then of its mirror), `detail` the crop at native resolution as WebP
  quality 90 (at most 512 px, tattoo and mehndi only), `crop` a 192 px WebP for the UI. Ids are never reused.
- `trait_scans(body_id PK FK CASCADE, item_id, rev, box, item_boxes, status, bare JSON, scanned_at)`: one row per
  measured body, with the rev, the body's box and the item's box fingerprint it was measured with.
- `trait_hides(id AUTOINCREMENT, item_id FK CASCADE, kind, verdict, x,y,w,h, bx,by,bw,bh, src_trait_id, created_at)`:
  the owner's "Not a tattoo" (`not_trait`), "Not this person's" (`not_theirs`) and "Show again" (`shown`).
  Journalled, with no BLOBs.
- `trait_hide_desc(hide_id PK, vec, detail, created_at)`: the descriptors of a `not_trait` hide, copied from its row.
  No foreign key, so an undone unhide finds them again.
- `trait_hide_hits(trait_id FK CASCADE, hide_id FK CASCADE, PK)`: the look-alikes of a `not_trait` hide.
- `trait_requeue(item_id PK FK CASCADE, queued_at)`: items to guess again on the traits runner thread.
- `place_links(a_item FK CASCADE, b_item FK CASCADE, rev, kind, m JSON, inliers, spread, dt_s, computed_at, PK(a_item,
  b_item))` (section 35): one row per pair of a photo and one of its next photos, a the earlier one. `kind` is
  `still`, `moved`, `burst` or `none` (did not line up); `m` is the 2 x 3 matrix a -> b in normalised coordinates.
- `place_scans(item_id PK FK CASCADE, rev, sig, scanned_at)`: a photo whose pairs are stored; `sig` is the ids of its
  next photos, so a photo arriving later in the middle of a sequence makes its neighbours pending again.

## 6. Uploads

The hub's chunked, resumable protocol, unchanged except: no uploader name; `UploadCreate` gains
`source` (`web`, `share`, `camera`); `status` loses `replaced` and `lower_quality`; the client hashes
files up to 1 GiB first and a known `sha256` answers `duplicate` with the existing item before any
bytes move. Hashing at `complete` runs in a worker thread, not under the per-upload lock. Any file
type is accepted (`accept` is `*/*` plus the explicit document extensions for the picker). An extra
`POST /api/uploads/simple` (multipart, one file up to 256 MB) exists for tests and scripts; it must declare
`Content-Length` (411 `length_required`), and a declared length above `SIMPLE_UPLOAD_MAX_MB` is refused (413)
before any of the body is read. `UploadOut.item_nsfw` is true only when the session may see the vault and the
stored item is in it (the frontend then links it with `?vault=1`); a locked session gets `item_uid` and
`item_thumb_url` as null for such an item and no hint that it is private. Re-uploading a file whose item is
in the trash restores that item on every path (client hash, chunked complete, simple) and answers `duplicate`
with the message `Restored from the trash (identical file).`; staged bytes are never dropped while the only
stored copy sits in the trash.

## 7. Processing pipeline

Videos with a placeholder index (added 2026-11-08): some Apple encoders leave a `hoov` atom (or a `free`
atom) holding an `mvhd` before the media data, with the real `moov` after the `mdat`. ffmpeg's lenient default
takes the placeholder for the index, finds no packets and the clip used to be kept as kind `other` ("There is no
preview for this kind of file"). `probe.input_flags(path)` reads the top-level atoms (cached by path, size and
mtime) and returns `["-strict", "strict"]` for such a file; every ffmpeg and ffprobe call on an original passes
it before the input, so the real index is read. `MediaInfo.strict_input` marks them and
`video.needs_web_preview` always gives them a web copy (browsers trip over the placeholder as ffmpeg does). A
non-zero ffmpeg or ffprobe exit whose stderr names a resource problem (no memory, too many open files, no
space, an I/O error) is a `ToolFailure`, not a verdict on the file. `probe.PROBE_VERSION` (3) is bumped when
the probe learns to read files it used to keep as `other`; at startup, when kv `probe_version` is lower,
`ingest.reprobe_plain` runs once on a background thread over the items kept as `other` whose MIME says image,
video or PDF, or whose extension is `.jxl`: one that now probes as a real kind (with the same storage extension)
gets its derivatives, its row's media columns, fingerprint and statuses, its analysis, text and web-copy jobs and
a Live Photo pairing; the kv is set when the pass completes (a `ToolFailure` stops it for the next start).

JPEG XL and ProRAW DNGs (2026-11-09, version 3). JPEG XL (`.jxl`, `image/jxl`: a bare codestream starting
`FF 0A`, or the `JXL ` box container) is a photo like any other, decoded by `pillow-jxl-plugin` (registered with
Pillow by `media.images` as pillow-heif is); its orientation lives in the codestream and the decoder applies it,
so an Exif box's orientation is ignored (the format's rule; turning it again would put the picture on its side).
An iPhone ProRAW DNG from the iPhone 17 Pro on stores its raw image JPEG XL compressed (DNG 1.7, compression
52546), which the LibRaw inside `rawpy` cannot open at all; such a file is read from its largest embedded JPEG
preview instead (`raw.embedded_preview`: IFD0, the chained IFDs and their SubIFDs, at least 640 pixels on its long
side; an iPhone keeps a full-size one in IFD0), turned upright once by its own EXIF orientation or else the
file's, with the dates, camera and place from the file's EXIF. DNGs LibRaw reads are demosaiced as before.
Version 3 turned the owner's stored ones (five DNGs and a `.jxl` kept as plain files) into photos.

Every job that ran to a verdict leaves a `job_runs` row (type, queue, finished_at, duration_ms, ok) for the
processing speed charts (section 32); deferrals (full disk, ffmpeg not running) are not runs. Housekeeping
deletes rows older than 14 days.

Jobs, queues and priorities: `ingest` (main, 10), `analyze` (main, 50), `video_preview` (video, 50),
`text` (text, 60), `guess` (main, 70), `recluster` (main, 90). The queues are serial; `main` runs
all GPU work except OCR so two heavy models never contend. Handlers are idempotent and re-check state.

1. `ingest` (`ingest.process_upload`): probe (`media.probe.probe` extended with `media.documents.probe_document`
   for PDF, DOCX, XLSX, PPTX, TXT, MD, CSV and `media.raw` for camera RAW), sha256, fingerprint, near-duplicate
   search (`media.dedup.find_similar`: same thresholds as the hub; the result links, never replaces: the new item
   gets `dup_group_id` of the best item in the group, and the group's best item is re-chosen by the hub's quality
   ranking), place original, derivatives (thumb + previews; page 1 render for documents; poster for videos),
   insert item (`analysis_status='pending'`, `text_status='pending'` for `pdf` and `document`, else `'none'`,
   the file-level forward signals and `received_via` of section 29), enqueue `analyze`, plus `video_preview` when `needs_web_preview`, plus `text` for documents.
2. `analyze` (`analysis.pipeline.analyze_item`): load the analysis image (oriented original at most 2048 px; page 1
   render for documents; for videos, frames 5 a second for the faces (section 10, "Video faces through the whole
   clip") of which SigLIP, NSFW and pets keep 1 per second, at most 150); SigLIP embeddings (whole image and
   frames) stored in `item_embeddings`; category scores and labels; NSFW score; quality scores and mis-click
   reasons (not for documents); faces (SCRFD multi-scale, ArcFace) and video face tracks; person boxes (YOLOX-S
   class 0) matched to faces by containment; torso crops (3.2 face widths wide, from y-0.25h to y+4h; for a body
   without a face the upper 60 percent of the box) embedded with SigLIP into `torso_embedding`; write faces and
   bodies with crops; `cluster.assign_new_faces`; `guess.guess_item` for still-unassigned faces and bodies;
   decide category and nsfw, then the forward decision (section 29); `analysis_status='done'`; enqueue `text` when `doc_likeness >= DOC_LIKENESS_MIN`
   and `text_status='none'`. A reprocess that runs without one of the models keeps that model's earlier rows
   (embeddings and labels, faces, bodies and their covers, the nsfw decision and the scene category) instead of
   deleting them. A reprocess requested while a run is in progress is not lost: the run commits its results
   with `analysis_status='pending'`, so the job it queued runs in full.
3. `text` (`documents.extract.extract_item`): PDF text layer per page via pypdfium2, OCR with rapidocr on pages
   without a layer (and on images; every detected line is read by the bundled Latin recogniser and, when the
   `ocr_devanagari` model is present, by the PP-OCRv5 Devanagari recogniser too, and the better reading is
   kept: `ml.ocr_devanagari.choose`); office files via python-docx / openpyxl / python-pptx (pages are logical:
   one per docx section of ~3000 chars, one per sheet, one per slide; the renders are text drawn with Pillow);
   write `pages`, `item_text`, FTS; then `documents.detect.detect(text, pages, item)` -> doc type and
   `parsers.<type>.parse(...)` -> fields; barcodes via zxing-cpp on page renders; ID photo crop from the largest
   face in the page; write `documents`; `text_status='done'`. Items whose text is under 20 characters and that
   match no type get no `documents` row.
4. `video_preview`: the hub's transcode.
5. `guess` (`{person_id, item_id}`): after the user names, confirms or tags someone, propagate to unassigned faces
   and bodies in items within the same window (section 10).
6. `recluster`: the hub's `recluster_unassigned`.

Statuses the UI polls: `analysis_status`, `text_status`, `preview_status`.

Failure and recovery rules: a job is claimed at most `MAX_ATTEMPTS` (3) times; one found running after a
crash, or by the stale sweep, with its attempts used up goes to `error` and gets the final-failure handling
(upload bytes to `failed/`, item status `error`), so a poison file never crash-loops the server. A clean
shutdown gives the attempt back to jobs still running. A `ToolFailure` (ffmpeg or ffprobe timeout, signal or
missing tool) uses no attempt: the job is deferred after 1, 5 and 30 minutes, then hourly, and fails for good
after 48 hours (`last_error` starts with `[tool failure N since <iso>]`). `POST /api/system/retry-failed`
queues failed uploads whose bytes are kept and errored jobs again. Restoring an item from the trash queues
the stages still pending (`analyze`, `text`, `video_preview`, unique). Startup recovery enqueues in one
transaction (`jobs.enqueue_many`, partial index `jobs_dedup`). Housekeeping every 10 minutes on the main
worker thread covers stale uploads, `failed/` retention, trash expiry and session purge; the orphan file
sweep runs on its own low-priority thread, first an hour after start and then daily. A `preview` reprocess
of a document renders on a worker-owned background thread into a scratch directory and swaps the pages in
one by one, so a running text job never sees a missing page.

Body traits run outside these jobs (section 34). One runner thread (`app.analysis.traits`, `worker.run_in_background`
key `traits`, nice 19, its own connection) reads tattoos, mehndi and nails from the people boxes of finished photos,
newest item first, and guesses those photos again. It never runs inside `analyze_item`, so the analysis transaction
and `Result` are unchanged and face ages, `doc_likeness` and `category` exist by the time it looks at a photo. It is
kicked by `pipeline._after` (after `guess_item`), by `ingest.recover_on_startup` (`traits.start`, then
`trait_match.apply_settings_change`), by the routes that move, draw or delete a person box (after their commit), by
`guess.propagate`, by `undo.undo` (`traits.after_undo`) and by housekeeping (`traits.kick`, then `traits.prune`). The
item is the unit: each `trait_scans` row keeps the item's box fingerprint (every body box of the photo, written and
compared with SQLite's own `printf`), so moving, drawing or deleting any box of a photo, or undoing that, makes every
body of the photo pending again. Photos to guess again wait in `trait_requeue`, which survives a restart; the runner
drains it between measuring steps and needs no model for that. There is no new `jobs.type`.

## 8. ML engines and models (`app/ml`)

`ml.runtime.session(path, *, threads)` builds an `InferenceSession` with providers
`[CUDAExecutionProvider, CPUExecutionProvider]` when `GPU_ENABLED` is `auto`/`true` and the CUDA
provider is available, else CPU. One lock per session. `ml.runtime.status()` reports the provider in
use for `/api/system`. Models (`ml.models.REGISTRY`, downloaded by `scripts/download_models.py`,
pinned and checksummed, skipped when present):

| Key | Files | Source | License |
|---|---|---|---|
| `faces` | `buffalo_l/det_10g.onnx`, `buffalo_l/w600k_r50.onnx` | InsightFace release v0.7 zip | non-commercial research (personal use) |
| `siglip` | `search/ViT-B-16-SigLIP__webli/{visual,textual}/model.onnx`, tokenizer | immich-app on Hugging Face, commit c3b037a | Apache 2.0 |
| `nsfw` | `nsfw/model.onnx` (+ `config.json`, `preprocessor_config.json`) | onnx-community/nsfw_image_detection-ONNX, pinned commit | Apache 2.0 (Falconsai ViT) |
| `ages` | `ages/onnx/model_int8.onnx` (87 MB, + `config.json`, `preprocessor_config.json`): nine FairFace age groups per face crop (`app/ml/ages.py`, section 10) | onnx-community/fairface_age_image_detection-ONNX, commit fd04cdd (int8 export of dima806/fairface_age_image_detection) | Apache 2.0 |
| `persons` | `yolox/yolox_s.onnx` | Megvii YOLOX release 0.1.1rc0 (35.9 MB) | Apache 2.0 |
| `ocr` | rapidocr's bundled models (`PP-OCRv6_det_small`, `PP-OCRv6_rec_small`, `ch_ppocr_mobile_v2.0_cls`; installed with the package, nothing downloaded at run time) | RapidAI | Apache 2.0 |
| `ocr_devanagari` | `ocr/devanagari_PP-OCRv5_rec_mobile.onnx` (7.9 MB, Hindi and Marathi recogniser, character list in the ONNX metadata) and `fonts/NotoSansDevanagari-Regular.ttf` (page renders and test fixtures) | RapidOCR model zoo (modelscope, release v3.10.0); Noto fonts | Apache 2.0; SIL OFL 1.1 |
| `pose` | `pose/end2end.onnx` (71.8 MB): DWPose-m whole-body pose, 133 COCO-WholeBody points per people box, input 192 x 256 (`app/ml/pose.py`, section 34) | Hugging Face Tau-J/RTMPose at commit cd4d709, the `rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192` zip (`archive_url`, member `end2end.onnx`, looked up by basename) | code Apache 2.0; the weights are trained on COCO-WholeBody, which is research and non-commercial use only (acceptable for this personal, non-commercial repository) |
| `hands` | `hands/handpose.onnx` (4.1 MB): MediaPipe hand landmarks, 21 points on a 224 px hand crop (`app/ml/hands.py`) | Hugging Face opencv/handpose_estimation_mediapipe at commit 4b2a0b4 (OpenCV Zoo 2023feb) | Apache 2.0 |
| `dino` | `dino/dinov2_small.onnx` (88.5 MB, fp32): DINOv2-small CLS descriptors of tattoo and mehndi crops (`app/ml/dino.py`) | Hugging Face onnx-community/dinov2-small at commit 8b1f705 | Apache 2.0 |

Every engine is lazy (loaded on first use), lives once per process, and falls back to CPU when the
CUDA provider fails to initialise (logged once). Without a model file the dependent stage marks
its part as skipped (`quality_scores.skipped`, `category_scores.skipped`) and the item still completes.

GPU memory: onnxruntime's per-session arena grows with every new input shape and never shrinks on its own (a
server in ordinary use was measured at 7.7 GB of the card's 8 GB). `runtime.session` therefore returns CUDA
sessions wrapped so that every `run()` carries the `memory.enable_memory_arena_shrinkage` option
(`runtime.wrap_session`, `GPU_ARENA_SHRINK`, default on); `app.ml.ocr` wraps rapidocr's sessions the same way.

The body traits engines (section 34) follow `app/ml/ages.py`: a lazy singleton, one lock, a session from
`runtime.session(path, threads=TRAITS_THREADS)`, and `available()`, `loaded()`, `provider()`, `reset()`.
- `pose.estimate(bgr, boxes_px)` returns one (133, 3) array per box (x and y in image pixels, a score). The box grows
  by 1.25, is widened or heightened to 192:256 and cut with an affine warp (RGB, mean 123.675 / 116.28 / 103.53, std
  58.395 / 57.12 / 57.375); the SimCC outputs (384 x bins, 512 y bins) decode as argmax / 2, the score as
  0.5 x (max_x + max_y). Boxes of one photo run in batches of 8. About 20 ms a box on 4 CPU threads.
- `hands.landmarks(crops_rgb224)` returns, per crop, the 21 landmarks in crop pixels, a confidence and a handedness
  value read as the probability of a right hand. The graph's four outputs are told apart by shape at load time; the
  handedness polarity is not checked on labelled hands. One crop at a time, about 3 ms each.
- `dino.embed(crops_rgb)` returns L2-normalised CLS vectors (n, 384) of crops resized to 224 px (ImageNet mean and
  std), in batches of 8, about 78 ms a crop on 4 CPU threads.
- SigLIP is reached through the shared `siglip.get_engine()` (it runs at `SEARCH_THREADS`). Crops go to it as PIL RGB
  images (`traits.siglip_images`): `_to_pil_rgb` reads a raw numpy array as BGR and would swap red and blue, which
  matters for the henna and nail gates.

## 9. Categories, NSFW and mis-clicks (`app/analysis`)

Zero-shot scoring: `siglip.zero_shot(image_embedding, PROMPTS)` returns softmax probabilities over
prompt groups (several prompts per group, averaged). Prompt groups live in `analysis/categories.py`
(`CATEGORY_PROMPTS`, `DOCUMENT_PROMPTS`, `SCREENSHOT_PROMPTS`, `ACCIDENT_PROMPTS`).

Decision order for an item (user overrides win: `*_source='user'` is never changed by the worker):
1. `kind in (pdf, document)` -> `documents`.
2. NSFW (`categories.vault_decision`): the classifier score `nsfw_score` and three SigLIP zero-shot signals
   (`explicit_p`, `suggestive_p` for lingerie, shirtless and intimate selfies, `safe_p` for beach, sport, baby,
   medical and museum skin), all stored in `category_scores`. In the vault when the classifier is very sure
   (`>= 0.90`) and the photo is not ordinary skin (`safe_p < 0.60`), when the classifier (`>= 0.60`) agrees with
   explicit (`>= 0.30`) or suggestive (`>= 0.40`), when explicit is `>= 0.60` on its own (unless no person was
   found and the classifier is under `0.10`: a skin-coloured pattern), or when suggestive is `>= 0.70` on its
   own in a photo with a person; ordinary skin blocks the classifier-only and suggestive-only paths. `VAULT_SENSITIVITY` (env) and the kv flag `vault_sensitivity` (Settings > Learning) shift every cut by
   0.10: `low` puts fewer photos in the vault, `high` more. The vault learner (section 18) replaces the rule
   with weights fitted over the four signals once the owner has corrected enough decisions; the sensitivity
   shifts its cut too (0.35/0.50/0.65 on the learned probability, plus or minus 0.10 on the calibrated
   classifier cut), never a labelled look-alike's own answer. A sensitivity change re-sorts the automatically
   decided items at once from the stored signals (`analysis/vault.resort`: the reply's `vault_moved` counts
   `checked`, `moved_in`, `moved_out` and `stale`, the items analysed before the signals existed, which need
   the "apply to existing items" re-analysis). NSFW items get `category='nsfw'`. Without the zero-shot
   signals (SigLIP missing) the old rule applies: `nsfw_score >= 0.7`, or `>= 0.4` with explicit `>= 0.5`.
3. Screenshot: PNG or WebP without camera EXIF whose dimensions match a phone or desktop screen, or screenshot
   zero-shot `>= 0.6` -> `screenshots`.
4. `doc_likeness` = max(document zero-shot, text-density heuristic from edges); `>= DOC_LIKENESS_MIN` (0.55) -> `documents`.
5. At least one good face or body -> `personal`.
6. Best of `places`, `food`, `pets`, `objects` when its probability `>= 0.35`, else `misc`.
7. Forwards (section 29, after the category and the vault): `forwards.apply` decides `received_via` and `forward`
   from the file's signals and the `meme_p` zero-shot; a `forward_source='user'` row keeps the owner's decision.

Mis-click reasons (`quality.py`, images only, documents and screenshots excluded; a video only gets `dark`):
- `blurry`: variance of the Laplacian on the grey image scaled to 1024 px long side `< BLUR_VAR_MIN` (40),
  or motion blur zero-shot `>= 0.6` combined with variance `< 2 * BLUR_VAR_MIN`;
- `dark`: mean luminance `< 28` and 95th percentile `< 70` (0..255); `blown_out`: mean `> 235` or more than 60
  percent of pixels at `>= 250`;
- `pocket`: accident zero-shot (floor, ceiling, pocket, hand over lens, sky only) `>= 0.6` and no good face;
- `burst`: another image of the same camera taken within `BURST_WINDOW_S` (3 s) with pHash distance `<= 12`;
  the sharpest of the group is `burst_best` (not flagged) and the rest get `burst` with `best_uid` in the reasons.
  Documents and screenshots never join a burst group, and a group stays on one side of the vault.
Flagged items (`misclick=1`) are excluded from the library by default and listed under Review.
Scores are stored in `quality_scores` (`blur_var`, `mean_lum`, `p95_lum`, `clipped_frac`, `accident_p`,
`motion_blur_p`) so thresholds can be tuned later without re-running the models.

## 10. People: faces, bodies, guessing

Faces follow the hub (`buffalo_l`, clustering thresholds 0.45 good / 0.52 other, `faces_version`,
locked `user` assignments, merge suggestions). Bodies come from YOLOX-S person boxes (score `>= 0.5`,
NMS 0.45); a body whose box contains a face centre takes `face_id` and shares its person.

One face, one box (2026-11-09). Faces are detected at two input sizes (640 and 1280, `detect_multiscale`) and the
results merged by non-maximum suppression: a box goes when its IoU with a better one is above 0.4, and also when at
least 80 percent of the smaller of the two lies inside the other (`app.ml.faces.CONTAIN_THRESH`), as a tight and a
loose box around one hooded or turned face overlap too little for IoU alone. Photos analysed before that keep
such twins until a one-time pass at startup (`face_upkeep.drop_nested_twins`, kv `nested_faces_v1`): two detector
boxes of one photo from the same analysis, one at least 80 percent inside the other, leave one face; a face the
owner tagged stays over one they did not, then the better detector score, and two faces the owner tagged as
different people both stay. A body around the face that goes takes the one that stays (or goes when that one
already has a body), and an unnamed group left empty goes. It is housekeeping, not an owner action (no undo).

The reference for a person is every face they have, never one picture: a new face scores a person by
the weighted mean of its top-3 cosine similarities to that person's faces (`index._topk_means`), the
owner's own assignments weighing 1.5 and automatic ones 1.0, so the reference sharpens as faces are added
and confirmed; each person also learns their own threshold from confirmations and rejections (section 18).
`PersonOut.confirmed_faces` says how many anchors the owner has given.

Corrections propagate (`people._follow_lookalikes`). When the owner takes faces away from a person
(`POST /api/faces/assign` to another person or to nobody), the faces that left are written to
`guess_rejections` for that person, and the person's automatic faces that look like them follow: closer to
a removed face (cosine `>= FOLLOW_SIM` 0.55) than to the person's other faces (top-3 mean, removed faces
left out), the owner's own assignments never. Followers go to the same target (assigned `auto`) or, when
the faces were just unassigned, are regrouped at once (`cluster.assign_new_faces`) into another or a new
unnamed person; bodies around them follow. `AssignOut.followed` counts them for the toast. Matching
(`cluster._score`) then leaves out any person whose rejected examples look more like a face than the
person's own faces do, so look-alikes of a rejected face never return to that person, in new photos or in a
recluster.

Merging pulls look-alikes along (`cluster.merge_and_follow`, behind `POST people/merge` and behind renaming a
person to a name that exists). The similarity the owner accepted is measured first (`learn_accepted`: for each
merged person, the mean over its faces of the top-3 mean to the target's faces, band by band; the least of those;
under 0.30 or across the child band nothing is learned, see "Ages and childhood photos" below), the merge
happens, and then every unnamed person whose faces are all automatic and who matches the merged person at
`>= max(accepted, 0.30, the person's learned threshold)`, ahead of any other person by 0.05, with the rejected
look-alikes rule applied, is merged in too (`follow_merge`): never a named person, one with a face, body or tag
the owner assigned, one dismissed against the target (`merge_dismissals`), or one seen in the same photo. A quick
pass (2026-11-09) settles most of the unnamed groups with one matrix product: a face's score for a person is a
weighted mean of its best similarities to the person's faces, so a group with any face whose best similarity to
the target is under the cut cannot follow (`FaceIndex.least_best_match`); the full check runs for the rest. It
gives the same pulls as before (checked against every named person on the owner's library) in a fraction of the
time: a merge or "Change to..." on a library of about 1,300 unnamed groups took 12 seconds, now under one.
`PersonOut.followed` counts them for the toast. The merge is a learning event (section 18, kind `merge`), so the
person is matched down to that similarity from then on; taking a face away again (`POST faces/assign`) records a
rejection that raises the threshold above it and keeps its look-alikes out of the next pull.

Guessing (`analysis/guess.py`). For an unassigned face or body D in item I:
- Candidates: faces (and bodies) of named or unnamed persons with `assigned_by in ('user','auto')` in items J
  taken within `GUESS_WINDOW_H` (1 h) before or after I, never further whatever the calendar day (clothes are
  evidence only close in time), excluding persons already present in I and pairs in `guess_rejections`.
- Scene: `cos(emb(I), emb(J)) >= GUESS_SCENE_SIM` (0.80), or both items have GPS within `GUESS_GPS_M` (250 m).
- Clothes: `cos(torso(D), torso(F)) >= GUESS_CLOTHES_SIM` (0.85).
- Face: when D has a face embedding, it must not match any other person at `>= 0.40`; a weak match with the
  candidate (`>= 0.20`) raises the score.
- Score `= 0.5 * clothes + 0.3 * scene + 0.2 * face_or_0.5` until the owner has answered enough guesses; from then
  on the learned guess probability replaces this mix for every candidate (section 18: `learning.people.guess_probability`
  over clothes, scene term, face term, same day and GPS, once `MIN_GUESS_EVENTS` answers exist).
  The best candidate wins when it leads the runner-up by `>= 0.05` on that scale; the result is stored as
  `guess_person_id`, `guess_score`, `guess_evidence` (`{clothes, scene_term, scene, face, ref_face_id, ref_body_id,
  ref_item_uid, same_day, minutes_apart, gps_m, score}`; `scene_term` is the scene term the score used, `score` is set
  on the winner) with `assigned_by='guess'`.
- Body traits (section 34). When the library's traits are read and a box of the photo has some, `_compute` fuses each
  box's clothes result with its tattoo, mehndi and nails evidence (`_compute_fused`, the same inputs and the same one
  person per photo loop); otherwise `_compute_clothes` runs, exactly the clothes path. `guess_evidence.basis` marks
  the result: `trait` for a guess from traits alone (a tattoo, under the admission rule of section 34),
  `clothes+trait` for a clothes guess that traits agreed with or broke a tie for, and no `basis` for clothes alone.
  An open face box gets the traits of the body that holds it (`bodies.face_id`).
- Position (section 35). When the photo is lined up with a photo taken moments before or after it, each box's result
  is then fused with where people stood (`place_match.fuse`, after the trait step, inside the same one person per
  photo loop): `clothes+place` for a clothes guess that the position agrees with or broke a tie for, `place` for a
  guess from the position alone, and no guess when the position contradicts a clothes winner. `_compute` takes the
  fused loop when a trait context or a place context exists; with neither, `_compute_clothes` runs, and the scene
  loop is one function for both (`_clothes_inputs`).
- Confirming sets `person_id`, `assigned_by='user'` and enqueues `guess` for that person; rejecting writes
  `guess_rejections` and clears the guess. A confirmed body without a face adds an `item_tags` row with
  `source='guess'`, removed again when that body is moved or deleted.
- Propagation (`guess` job) runs the same scoring from the newly tagged face outward over the window. It also
  queues, for the traits runner, the photos at any date whose open boxes carry a tattoo (or, within its window,
  mehndi) close to the person's by DINOv2 cosine (`trait_match.trait_targets`, at most `TRAIT_PROPAGATE_MAX`, nothing
  verified there), so the serial queue stays fast (section 34).
- A guess card for a box in a video shows the frame at the box's time (`/files/frame`, preview size) with the
  box outlined, says "at 0:42", and opens the video there; the poster frame is a different moment.
- A changed `GUESS_WINDOW_H` takes effect at the next start: `guess.apply_window_change` (from
  `ingest.recover_on_startup`) withdraws every open guess whose reference photo lies outside the window now and
  queues a `guess` job for its item; `kv guess_window_h` records the value in force. Guesses with basis `trait` have
  no window and are kept, and so are guesses with basis `place` (their own limits are seconds, section 35).
- A person is guessed for at most one box per photo (the best score wins, the other boxes are scored again
  without that person; videos are exempt), and confirming skips a box whose person is already in the item.
  Rejections follow a face across a reprocess (carried to the new face, see "Carry-over" below). Vault items are never
  references, and NSFW items get no guesses. Reclustering and merging refresh the FTS people column.
- People routes and the vault: with the vault locked, tagging or untagging an NSFW item and box actions on
  NSFW boxes answer 403 `vault_locked`; a person seen only in vault items answers 404 by id outside the vault.

Whole-video editing (2026-11-08). A video's faces are tracks (one row per run of frames, `spans`), and a
person who leaves the frame and comes back, or whose track breaks, is several tracks; naming them one by one
is tedious. So the item detail of a video groups its tracks and the viewer edits a group at once:
- `FaceOut.group`, `BodyOut.group` and `AnimalOut.group` (an integer, videos only, null for photos) computed
  on read (`library.item_detail` through `common.video_groups(faces, bodies, animals)`): tracks assigned to a
  person share that person's group; among the rest (unassigned and guessed), tracks join the group of the
  most similar earlier track when the cosine between their embeddings is at least `VIDEO_GROUP_SIM` (0.5)
  (faces by face embedding, bodies by torso, animals by crop embedding, never across kinds), else start one.
  Groups are numbered in order of first appearance.
- The viewer's strip ("People and pets in this video") shows one chip per group with the number of tracks
  ("Priya, 4 tracks"; "Not tagged, 3 tracks"), seeks to the group's first track, and its edit control opens
  the editor in group scope: assign to a person, name, not this person, confirm or reject a guess and remove
  apply to every track of the group (`POST /api/faces/assign`, `/api/bodies/assign`, `/api/animals/assign`,
  the guess routes and the delete routes already take lists or are called per id). The editor says "Applies to
  all N tracks in this video" with a switch "Only this track" that narrows it to the one the chip was opened
  from; a group chip expands to its tracks ("Show tracks") so a mis-detected track can be edited, seeked to or
  removed alone. Photos are unchanged.
- Tests: `video_groups` on hand-made embeddings (assigned tracks grouped by person, look-alikes joined at 0.5,
  strangers apart, kinds never mixed, photos get null), the item detail of a video carrying the groups, and
  the assign route with several face ids of a group.

Face weight (2026-11-08). Only clear faces should teach who someone is, so every face counts by how clear and
visible it is, from 0 to 1 (`faces.face_weight`). A blurred, masked, helmeted or turned-away face, or something
that is not a face at all, would otherwise pull a person's examples towards "unclear", and other unclear faces
would start to match them.
- Measures, stored per face: `faces.clarity`, the ArcFace feature norm before normalising (it falls for blur,
  low resolution, a face turned away or covered), and `faces.sharpness`, the Laplacian variance over the centre
  of the aligned 112 px face (`app.ml.faces.sharpness`); both migrated columns, NULL for faces analysed before
  them. Measured on 617 faces from the owner's photos: below a norm of about 14 they were turned away, masked or
  not faces (textures, a sign, the back of a head), from about 22 clear; below a sharpness of about 8 blurred,
  dark or tiny.
- The weight is the product of four linear ramps, each 0 at its first value and 1 at its second: norm 13 to 22,
  sharpness 5 to 30, detector score 0.5 to 0.8, size `FACE_MIN_SIZE_PX` to `FACE_GOOD_MIN_SIZE_PX` source pixels.
  An unmeasured norm or sharpness counts as 1, so old faces are weighed on detector score and size until they are
  measured.
- Matching: a person's faces are their examples by weight (`index._topk_means`: weight times 1.5 for the owner's
  own), faces of weight 0 are no example at all, and a person whose faces all weigh 0 is never scored; centroids
  (merge suggestions, merge similarity) are weighted the same way. A good face's own cut rises linearly from
  `CLUSTER_MATCH_THRESHOLD` (weight 1) to `CLUSTER_MATCH_THRESHOLD_LOW` (weight 0); any other face keeps
  `CLUSTER_MATCH_THRESHOLD_LOW` as before, so the weight only ever makes a cut stricter. Only a clear face
  (`is_clear_face`: the good-face rule and weight `>= CLEAR_WEIGHT` 0.5) starts a new person.
- Learning (section 18): a face moves a learned threshold by its weight.
- A person the owner tags on an unclear face (a helmet, a mask, their back, or a drawn person box with no face,
  below) stays theirs on that item and is a reference for the clothes guesses within the hour like any tagged
  face or body, but teaches nothing about their face.
- Existing faces: photos analysed before the measures get them on a background pass at startup
  (`face_upkeep.measure_photo_faces`: the photo is detected again as the analysis saw it, and each stored face
  takes the measures of the detection over it at IoU `>= 0.5`; the stored embedding stays; kv `face_measure_cursor`
  resumes it). Videos get them from their re-tracking (below).

Video faces through the whole clip (2026-11-08). Faces are followed through every part of a video by where they
are and how they look, so a profile, a blurred frame or a look away stays on the right person, and a person is
known from all their good views.
- Sampling (`faces.video_plan`): `VIDEO_FACE_INTERVAL_S` (0.2 s, 5 frames a second), at most
  `VIDEO_FACE_MAX_FRAMES` (1500; the step grows on clips over 5 minutes). Clips over 10 minutes are read by
  seeking, one ffmpeg run per frame, so they keep the plain sampling (`VIDEO_SAMPLE_INTERVAL_S`,
  `VIDEO_MAX_SAMPLES`). The single decode pass already decoded every frame, so the cost is the detection and the
  embedding: measured on the owner's 4K clips on the RTX 2080, the face pass takes about 0.4 to 0.5 times the
  clip's length. SigLIP, NSFW and the pet detector keep the plain cadence (`on_frame` sees the first frame and
  then each one at least `max(1 s, duration / 150)` after the last, at most 150).
- Every face of at least `FACE_MIN_SIZE_PX` source pixels is detected at `FACE_VIDEO_DET_SIZE` and embedded,
  with its clarity and sharpness.
- Linking (`faces.FaceTracker`): each open tracklet predicts its box (last box moved on by its velocity). A
  detection continues it by motion (seen at most 1 s ago, IoU `>= 0.3` with the prediction, and looking alike
  at `>= 0.2` against its latest views or all of them) or by look alone (`>= VIDEO_TRACK_THRESHOLD` 0.5).
  Measured on the owner's clips at 5 frames a second: the same face in consecutive frames looks alike at 0.43 or
  more (1st percentile), two people in one frame at 0.26 or less (99th percentile). Pairs are taken best first
  (a motion link scores its similarity plus its IoU), one detection per tracklet per frame; a tracklet unseen
  for `max(1 s, 2.5 steps)` ends.
- Tracklets of one person are then joined by look (closest groups first, weighted identities, `>= 0.5`), never
  two seen in the same frame: two faces on screen at once are two people. Tracklets in fewer frames than cover
  0.3 s are flickers (a detector blink, a blurred pan) and are dropped first on dense sampling; a person is kept
  when seen in frames covering 0.8 s (at least 2), or with a good face.
- Each person gives one face row as before: the best view's box, crop, torso and time (the best view is the
  clearest, then the highest quality), the identity (`faces.identity`: up to 8 of the best views at least 0.4 s
  apart, averaged by face weight, leaving out views unlike the rest at `< 0.3`), and the spans, gaps up to the
  lost time closed and never past the clip's end.
- Existing videos: once per `faces.TRACKER_VERSION` (kv `video_tracker_version`), every analysed video that is not
  a Live Photo's clip or in the trash is analysed again, its job at priority 80 so new uploads come first
  (`face_upkeep.retrack_videos`, from `ingest.recover_on_startup`; `jobs.enqueue` takes a `priority`). It reads
  "pending" until its job runs; its faces, names and choices stay until then.
- Carry-over (`cluster.plan_carry`, `carry_over`): re-detection pairs old and new faces one to one (box IoU
  `>= 0.3` with similarity `>= 0.3`, then similarity `>= 0.6`, then for the owner's faces `>= 0.4`, or IoU
  `>= 0.1` with `>= 0.3`), the owner's locked faces through all three passes before any other face takes a new
  one. A video tracked again has fewer, longer tracks, so old faces left without a pair follow the new face they
  match by the first two passes. Every old face that pairs with or follows a new face is one of its claimants: the
  new face takes the person of its best claimant (one the owner locked first, then one with a named person, then
  the most alike) and the rejected guesses of all of them; an automatic person carried onto a face that now holds
  a rejection of that person is dropped, and the face is assigned afresh. A face the owner named whose person
  reached no new face stays as it is (`analyze_item`), so a re-analysis never loses the owner's choice.

Not a face (2026-11-08). "Not a face" (`DELETE /api/faces/{id}`) teaches detection instead of only deleting
(`app/learning/not_faces.py`):
- The removed face is kept in `not_faces` (item, detector score, size, clarity, sharpness, embedding).
- Detection drops a new face (`not_faces.Filter.keep`, in the pipeline before bodies or anything else use the
  faces, photos and videos alike) that looks like a marking at `>= 0.5`, or that the learned rule (section 18)
  calls no face at `P >= 0.9`. Measured on the owner's photos: things that are not faces look alike to the face
  model (0.52 median to the nearest other one, up to 0.73), while clear faces stay at 0.34 or below against them.
  A clear face (weight `>= 0.5`) is never dropped, nor one where the owner already assigned a face on the item
  (IoU `>= 0.5` or similarity `>= 0.6`). The markings apply whether learning is on or off, like rejected guesses.
- The unassigned automatic faces that look like the removed one the same way go with it (`not_faces.followers`,
  through the face index), with their crops and counts; they are not markings themselves. The answer is
  `FaceDeleteOut {ok, followed}`.

Person boxes the owner draws (2026-11-08), like the pet boxes of section 22. `POST /api/items/{uid}/people-boxes`
(`PersonBoxIn {x, y, w, h, t?, person_id | new_person_name}`, 201 `PersonBoxOut {body, face}`;
`app/analysis/person_boxes.py`):
- The slow part runs before any write (`prepare_box`: the photo or the frame at `t` as the analysis sees it, the
  SigLIP vector of the upper box as its clothes, and the face detector inside the box padded by 15 percent); then
  one transaction makes the person (a new name) and stores the box (`store_box`).
- The box is a `bodies` row: `source 'user'` (a migrated column, `BodyOut.source`), `assigned_by 'user'`, the
  person, det_score 1.0, the clothes vector, `frame_time_s = t` on a video, and its crop. It holds the person for
  that occasion: the item lists them, and the guesses use the clothes within the hour.
- A face found inside becomes a face row of the same person (`assigned_by 'user'`, clarity and sharpness, so its
  weight decides how much it teaches; a video's spans `t +- 0.5 s`) and the body points at it. On a photo a stored
  face already there (IoU `>= 0.4`) is used instead and assigned through `_assign`, as a hand assignment is. No
  face (a helmet, a back, someone far away): the box alone, never their face.
- 400 `frame_time_needed` (a video without `t`), 400 `person_needed`, 400 for documents, 404, 409 when the photo or
  frame cannot be opened, 422 for a box outside the frame or under half a percent of it. Deleting is `DELETE
  /api/bodies/{id}` ("Not a person").
- Re-analysis keeps the drawn boxes and the faces they point at, and drops new detector boxes and faces over them
  (IoU `>= 0.5`, photos) as the same person (`person_boxes.drop_over_drawn`).
- Tests: `tests/test_video_tracking.py`, `tests/test_face_weights.py`, `tests/test_not_faces.py`,
  `tests/test_person_boxes.py`, `tests/test_face_upkeep.py`.

Face boxes the owner draws (2026-11-09), around a face the detector missed: `POST /api/items/{uid}/faces`
(`FaceBoxIn {x, y, w, h, t?, person_id | new_person_name}`, 201 `FaceOut`; `person_boxes.prepare_face` and
`store_face`):
- The slow part runs first: the photo or the frame at `t`, the box checked (inside the picture, at least 12 pixels
  a side in the analysed frame), and the face detector inside the box padded by 15 percent. The region is enlarged
  for the detector, so a small or soft face it missed in the whole picture is often found; of several, the one most
  inside the box (`pick_face`). It must find one: 422 `no_face` otherwise (a helmet, a back or someone far away is a
  person box), 409 `models_unavailable` without the face model.
- Tilted and overlapping faces (2026-11-09, `person_boxes.find_faces`): the detector misses faces tilted much past
  30 degrees (a head lying sideways, two faces cheek to cheek in a selfie; IMG_5192 scored 0.28 upright against the
  0.5 cut). When the upright search finds no face whose centre lies in the box, copies of the padded region turned
  by 30, 60 and 90 degrees each way are searched too (`_faces_turned`), and the faces found are mapped back: the box
  becomes the upright bounds of the turned face box, and the vector, made from the turned copy where the face
  stands upright, is as good as an upright face's. The same face found at several angles is kept once (best
  score). On a photo, a found face that is a neighbour's stored face is left out: a stored face the drawn box does
  not surround (IoU under 0.4 and neither box at least 80 percent inside the other, `surrounds`), with the found
  face over it (IoU `>= 0.4`) and its vector alike (cosine `>= 0.4`, `NEIGHBOUR_SIM`). So a box drawn over the
  second of two overlapping faces never takes the first one's face, and two faces that overlap heavily are told
  apart by their vectors. Moving a face box (`box_edit`) uses the same search, with the item's other faces as the
  neighbours. A person box the owner draws gets the turned search for the face inside it too.
- The row keeps the owner's box with `source 'user'` (drawn, `FaceOut.source`), `assigned_by 'user'` for the person
  (a new name makes them),
  the found face's vector, score, size, quality, clarity and sharpness, the head-and-torso vector with SigLIP, and
  its crop; on a video `frame_time_s = t` and spans `t +- 0.5 s`. On a photo a stored face under it (IoU `>= 0.4`
  with the found face, the drawn box around it as above, and the two vectors alike) is assigned instead, as a hand
  assignment is. A new face inside a person box the detector
  found with no face of its own (the face's centre in the box's upper half; not a box the owner drew or tagged)
  becomes that box's face, as a detected one would, and the box follows its person (its unnamed group goes when
  nothing else holds it). items.face_count follows, and the person's guesses run as after any tag. Undo takes the
  face and its crop away; a re-analysis keeps it on a photo (a moved or drawn face, below).
- 400 `frame_time_needed`, 400 `person_needed`, 400 for documents, 404, 409, 422. Tests: `tests/test_face_boxes.py`.

Moving or resizing a box (2026-11-09, `app/analysis/box_edit.py`): `PUT /api/faces/{id}/box`,
`/api/bodies/{id}/box` and `/api/animals/{id}/box` with `BoxIn {x, y, w, h}` answer the box (`FaceOut`, `BodyOut`,
`AnimalOut`), for any face, person or pet box, a guess's too.
- The slow part runs before the write (`box_edit.prepare`): the photo, or a video box's frame at its
  `frame_time_s`, as the analysis sees it; the box checked (inside the picture, at least 12 pixels a side in that
  frame unless it is no smaller than the old box, so a small detector box can still be nudged); the new crop. A face
  box also gets the face detector inside it (padded like a drawn person box): a face whose centre lies inside the
  box gives the new face vector, detector score, size, quality, clarity and sharpness (of several, the one most
  inside the box, then the one it overlaps most: `person_boxes.pick_face`, so a box enlarged around Asha that takes
  in part of Ravi's face still means hers); none found keeps the old vector while the box is still around the old
  face (overlapping it by IoU `>= 0.3`, covering half of it, or holding its centre: `box_edit.near_old`) or when the
  detector is not loaded, and otherwise answers 422 `no_face` (a box where no face is belongs to a person box).
  With SigLIP a face's head-and-torso
  vector, a person box's clothes vector and a pet box's look vector are taken again.
- The write stores the owner's box (not the detector's) with `moved = 1` (a migrated column on faces, bodies and
  animals; who put the person there does not change by fixing the box, so `source` stays and the origin icon with
  it), the new vectors, and then the crop: the old one is copied to the
  action's undo folder (`undo.keep_copy`) and the new one written; the face index takes the face again
  (`FaceIndex.reload_faces`, as a plain sync only loads vectors of faces it has not seen). Names and guesses stay
  (no new guess job: the owner may be fixing the box before answering), and a pet box nobody is on is matched to
  pets again.
- Crop URLs carry the crop file's time (`/files/face/{id}?v=...`, `app/urls.py`), so a moved box's new crop shows at
  once wherever its crop is (and an undo, which brings back the old file with its old time, brings back the old
  URL).
- Re-analysis keeps the moved boxes like drawn ones (`source 'user'` or `moved = 1`): person and pet boxes, and on
  a photo the faces, dropping new detector faces over them; a new detector body around such a twin takes the
  kept face instead (`person_boxes.relink_twin_bodies`; dropping the twin would clear its `face_id` and leave a
  faceless person box around the kept face), or goes when a body already holds it. A video's faces are tracked
  again (their names carry over to the new tracks), as a re-track cannot keep a box from an older run.
- 400 `frame_time_needed`, 403 `vault_locked`, 404, 409 when the photo or frame cannot be opened, 422 for a bad box
  or `no_face`. Undo puts the box, its vectors and its crop back (section 33). Origins (section 14) do not change: a
  moved box shows who put the person there, as before.
- Tests: `tests/test_box_move.py`.

Related items (2026-11-08). A tag can also say that an item is about someone without them in it (a pet's
toy, a person's handwriting), so it shows under the person and in a search for the name:
- `item_tags.relation TEXT NOT NULL DEFAULT 'in' CHECK (relation IN ('in','related'))` (a migrated column).
  `POST /api/items/{uid}/tags` takes `relation` (default `in`); posting the other relation for a person
  already tagged switches it. `ItemDetail.people` entries carry `relation`; the viewer lists `related` ones
  under "Related to" with a link mark instead of a face mark.
- `GET /api/people/{id}/items?relation=in|related|all` (default `in`, the behaviour so far); `PersonOut`
  gains `related_count`; the person page shows a "Related" tab with that count. The library's `person`
  filter, the timeline and the people pages keep meaning "in"; search finds related items by the name
  because `refresh_fts` writes every tagged name, whatever the relation, into the `people` column.
- "Tag someone" gets a switch "Not in the picture, related to them"; the chip's menu offers "Make this
  'in'" and "Make this 'related'" (the same POST with the other relation) and Remove.
- Tests: the column upgrade (tests/test_db_upgrade.py stays green), the tag route with both relations and the
  switch, the person items filter and count, the viewer's people list carrying the relation, and a search for
  the name finding the related item.

People on a selection (2026-11-08). On every page that selects items (the library, search results, an album,
the vault, and the Items and Related tabs of a person's or a pet's page) the owner can change who the
selected items show: move one person's or pet's appearances in them to another (an existing one or a new
name), remove someone from them, tag someone on all of them, or rename someone found in them. Routes
(`app/api/people.py`, mounted at `/api`; every route requires a session; `uids` 1 to 1000 item uids; items not
found or in the trash are skipped; a selection holding a vault item while the vault is locked answers 403
`vault_locked`):
- `POST /api/items/people` `{uids}` -> `ItemsPeopleOut{items, unnamed_faces, people: [ItemPersonOut]}`: the
  people and pets on those items, from faces, bodies, pet boxes and tags. `ItemPersonOut{id, name, kind,
  species, cover_url, items, faces, bodies, animals, tags, related, by_you}`: `items` counts the selected items
  they are on (any way), `tags` and `related` their tags by relation, `by_you` (2026-11-09) the items where the
  owner put them there: a face, person box or pet box with `assigned_by = 'user'` that is not a confirmed guess
  (`by_guess`), a box the owner drew (`source = 'user'`), or a tag added by hand (a tag a confirmed
  guess on a faceless body made, `source 'guess'`, is the system's find). Sorted by
  `items` (most first), then named before unnamed, then by name. `unnamed_faces` counts the faces in the selection
  that nobody is named for.
- `POST /api/items/people/move` `{uids, from_person_id, person_id | new_person_name}` ->
  `PeopleBulkOut{updated, person, followed}`. In those items every face, body and pet box of `from` goes to
  the target, locked as the owner's (`assigned_by = 'user'`, as `/faces/assign`), and every tag of `from`
  is re-pointed to the target with its relation (when the target already has a tag there, `in` wins).
  Faces leaving `from` are recorded as not `from` and its automatic look-alikes follow them
  (`_leaving`/`_follow_lookalikes`, section 10's merge learning), counted in `followed`. A person moves only
  to a person and a pet only to a pet (400 otherwise); a pet's new name creates a pet of `from`'s species,
  and an existing pet of another species is refused (400). The same person as target is 400. `updated`
  counts the selected items where something changed; `person` is the target's `PersonOut`. `from` is pruned
  when it ends up empty and unnamed (`cluster.prune_empty_persons`). Guess jobs and the search index follow
  (`_after_people_change`).
- `POST /api/items/people/remove` `{uids, person_id}` -> `PeopleBulkOut{updated, person: null, followed}`: in
  those items the person's faces, bodies and pet boxes are unassigned (faces recorded as not them, look-alikes
  follow and regroup, as an unassign through `/faces/assign`), and their tags on those items are deleted.
- `POST /api/items/people/tag` `{uids, person_id | new_person_name, kind: person|pet = person, species?,
  relation: in|related = in}` -> `PeopleBulkOut{updated, person, followed: 0}`: the per-item tag route
  (`POST /api/items/{uid}/tags`) on every item: a box of theirs already says `in`, `related` is always
  recorded, and a new pet needs its species (400 `species_needed`).
- Renaming someone found in the selection is the existing `PATCH /api/people/{id} {name}` (it renames them
  everywhere, and merges into an existing person or pet of that name).

Learning from these choices (the owner's "learn from not this person choices and merge data"):
- Not this person: faces leaving `from` are remembered (`guess_rejections` kind `face`) and its automatic
  look-alikes follow, as above; bodies leaving `from` get kind `body` rows, so guesses never propose `from` for
  them again; an owner's box assignment forgets an earlier rejection of that pair; look-alikes the owner
  already refused for the target stay where they are.
- Not this pet: `guess_rejections.kind` gains `animal` (old databases rebuild the table at startup through
  `db.CHECKED_TABLES`). Crops leaving a pet (these routes and `POST /api/animals/assign`) are remembered; the
  pet's automatic crops of that species at least `pets.PET_FOLLOW_SIM` (0.86) like a removed crop, and closer
  to it than to the pet's other crops, go along (counted in `followed`, remembered too, matched again but never
  back to that pet); `pets.assign_new_animals` never puts a crop that like a rejected one on that pet; the
  pet's learned threshold rises above its closest rejected crop (+0.02, at most 0.97; faces stay at most 0.90).
  Re-analysis moves a crop's rejections to the new box; deleting a crop drops them.
- Merge data: a bulk move to a person is learned like a merge (`cluster.accept_faces`): `accepted` is the
  moved faces' mean top-k similarity to the faces the target already had (None without any); a `merge`
  learning event `{accepted, faces, followed}` loosens the target as merges do, and the unnamed look-alike
  groups join the target (`follow_merge`), counted in `PeopleBulkOut.pulled`. An `accepted` below
  `PULL_FLOOR` (0.30, one badly matched face in a small selection) is recorded as `below_floor`: no
  loosening and no pull; the faces still count as the owner's examples. Pet moves record no merge event.

Frontend: `components/people/PeopleBulkAction.tsx`, a "People" button for selection bars with props `{uids,
vault, busy?, onDone, from?: {id, name, kind, species}}`. It opens a sheet "People in N items": with `from`
(a person's or pet's page) a first block "Not NAME?" with "Move these to..." (`PersonChoiceModal` of the same
kind; a pet keeps its species) and "Not NAME" (`/items/people/remove` behind a confirm that says what happens:
the boxes stay, unassigned and remembered as not them, look-alikes go along, their tags here go; the same as
"Not this person" and "Not this pet" in the box editors, so it is not styled as a deletion); then the list from
`POST /api/items/people`, each row with the avatar, the name ("Unnamed person" or "Unnamed pet") and "in X of
N", and the actions "Change to...", "Rename" (an inline name field; "Name" for unnamed ones) and "Not this
person" or "Not this pet" (the same confirm); the unnamed faces count as a note; and "Tag someone in these"
(the picker of both kinds with the related switch, a new pet asking its species). A move, a "Not NAME" or a
tag closes the sheet, clears the selection, toasts the outcome ("12 items moved to Meera", "13 items marked as
not Chutney", "+3 look-alike faces followed") and refreshes the library, the
person pages, the people list and open item details; a rename keeps the sheet open and reloads its list.
`BulkBar` (library, search, albums, vault) gets the button; the person and pet pages' Items and Related tabs
get a Select button, selectable tiles and a selection bar with the button (`from` = that person or pet).

Tests (`tests/test_people_bulk.py`): the list with faces, bodies, pet boxes and both relations; a move to an
existing person and to a new name, with look-alikes following and the tags re-pointed (relation kept, `in`
winning); a pet moved to a pet and refused to a person or another species; remove; a bulk tag of both kinds
and relations with `species_needed`; vault items with the vault locked; the same person as target.

### Related tags suggested (2026-11-09)

An item the owner marked as related to someone (`item_tags` relation `related`, source `user`: about them, without
them in it) makes other items candidates for the same mark (`app/analysis/related.py`). SigLIP whole-image similarity
alone cannot tell the same scene from a merely similar one: on the owner's library random pairs reach 0.77 at the
99th percentile, a dark red-lit video was 0.83 to 0.90 alike to unrelated dark scenes, and the same candle-and-rose
decoration from other angles was 0.84 to 0.90. So each candidate is measured against the related item's own baseline:
z = (s - median) / (1.4826 x median absolute deviation) of the related item's best-frame similarity to every item
(the search matrix, `app.search.service.get_matrix`). The decoration stood at 5.0 to 6.0, the dark scenes at 3.4 to
4.3.

- Reasons: `alike` (z >= `Z_ALIKE` 4.5, anywhere), `moment` (z >= `Z_CONTEXT` 3.0 and the same moment: capture times
  within 3 hours, or, when either has none, saved in the same batch: created within 120 s and at most 30 ids apart,
  as a camera-roll sync saves them in order) and `place` (z >= 3.0 and GPS within 100 m). Place alone is not enough:
  the owner's home holds most of the library. The best related item per candidate wins (score z + 0.5 for the moment,
  + 0.3 for the place).
- Never suggested: an item already tagged with them (either relation), one they are in through a face, body or pet
  box, one the owner turned down (`related_dismissals`), a related item itself, and anything a listing would not show
  (trash, mis-clicks, forwards, collapsed duplicates; the vault only with `vault=1` unlocked). Every related item the
  owner marked counts as a source (a copy or a forward too), on the side of the vault being listed, the
  `related.MAX_SEEDS` (200) most recently marked; all of them are scored in one matrix product, and a related item's
  baseline is cached by its own vectors and the library's size in steps of 512 rows.
- `GET /api/people/{id}/related-suggestions?limit=24&vault -> {items: [{item: ItemCard, reason, because_uid}]}`,
  best first; `POST /api/people/{id}/related-suggestions/dismiss {uids}` ("Not related", an owner action, undoable);
  `GET /api/items/{uid}/related-suggestions -> {people: [{person: PersonRef, reason, because_uid}]}` (the related
  items on the item's own side of the vault count; a vault item needs the vault unlocked; the related items'
  baselines are cached per state of the search matrix). Adding uses the existing tag routes with `relation:
  'related'`. On the owner's library: the three other decoration photos for both people the decoration was marked
  for, and for one of them a photo from the same evening; nothing from the dark video.
- UI: the person page's Related tab shows "Maybe related to NAME" above the grid (tiles with the reason, "Add",
  "Not related", and "Add all"); the viewer's people section shows "Maybe related to" under "Related to" (the person,
  why, and Add and No buttons). Accepted items become related items themselves, so the next suggestions build on them.
- Tests: `tests/test_related_suggestions.py`.

### Ages and childhood photos (2026-11-09)

The owner keeps childhood photos of people who are in the library as adults, mostly re-shot prints and forwards, so
their capture dates (all 2026 on the owner's library) say nothing about when they were taken. The age comes from the
face. Measured on the owner's 9,456 faces: InsightFace's own age model read children of 10 to 12 as 24 to 32 and
blurred adults as children; SigLIP zero-shot called about 1 adult in 10 young; the FairFace ViT kept 405 of 418
sampled faces in the adult groups and found every baby and child that could be seen, so it is the model.

- Storage: `faces.age_groups` BLOB, float32[9], the FairFace probabilities of the face's crop (0-2, 3-9, 10-19, 20-29,
  ..., 70+); NULL = not measured. Written for every face analysis finds (`pipeline`: from the in-memory crop, outside
  the write transaction), every face the owner draws or a person box finds (`person_boxes`), and a moved face box
  (`box_edit`: the new crop's, NULL without the model, never the old crop's; undo puts the old one back, it is a
  `FACE_VECTOR_COLUMNS` entry). `app/analysis/ages.py` reads it (`youth`: P(under 10), `band`, `expected_age`).
- Backfill: `ages.measure_stored_faces` on a startup thread (`face_upkeep.start`, when the model is present and a face
  has no age) reads the stored 192 px crops, 32 at a time, one short transaction each, about 70 to 90 ms a face on
  this CPU (about 13 minutes for the owner's library). Not an owner action. When no face is left it sets kv
  `face_ages_ready` to `ages.MODEL_REV` and moves `faces_version` once (a later pass, for faces stored while the
  model was missing, moves it too); an empty or fully measured library is marked at startup. Until then nothing
  below applies, so the child band switches on for the whole library at once. A crop the model cannot take is
  skipped (the batch is read again one crop at a time); a model that fails on a plain grey crop stops the pass until
  the next start.
- The child band (`ages.is_child_band`, `in_child_band`): P(under 10) >= 0.5 on a face of face weight >= 0.35.
  On the owner's library: 24 faces, all babies and young children but one toy. Children of about 10 to 14 read 0.3 to
  0.5 and stay out on purpose: the face model still links them to their adult selves (0.49 to 0.68 on five such
  photos), which is the cross-age match the owner wants. `settings.face_age_matching` (FACE_AGE_MATCHING) turns it off.
- Matching (`FaceIndex._child`, `person_scores_many(child=)`, `cluster._score`): a face in the child band is scored
  against each person's child-band faces only (a mask, so one tagged childhood photo is enough to draw the next one,
  where today's top-3 mean would dilute it with adult faces). Its cut is `settings.cluster_child_threshold` (0.48)
  plus the usual low-quality extra (so 0.48 to 0.55), or the follow cut when that is higher; a learned threshold only
  ever raises it (it was learned from adults: a person loosened to 0.30 cannot take in a stranger's toddler); and it
  needs a lead of `settings.child_lead` (0.08) over the next person (young children, siblings above all, look alike to
  the face model), checked after the rejected look-alikes rule, and it keeps that one option only (a best that is
  taken in the photo never falls through to an ambiguous second). Same-child leave-one-out scores on the library were 0.41 to 0.64, different children at most
  0.37. A clear child face with no option starts its own group, as any face. Every other face is scored exactly as
  before: on a copy of the library, 300 random adult faces kept identical options; 3 of the 24 child faces lost an
  automatic join (to a blurred one-photo group, or a lead under 0.08 against the same baby split in two groups).
  Guesses pass the band too.
- Learning (`cluster.learn_accepted`, used by `merge_and_follow` and `accept_faces`): what a merge or a move taught,
  band by band. Adult faces against the target's adult faces (`accepted`, which loosens the person, section 18);
  child-band faces against its childhood photos (`child_accepted`, which pulls childhood groups only). A band with
  nothing on the target side is a cross-age link (`cross_age`): the owner's say, nothing learned, no adult look-alikes
  pulled. An adult value under `PULL_FLOOR` (0.30) is `below_floor` and teaches nothing, for merges as for bulk moves
  (merges lacked this; on the owner's library nine people had been loosened to the 0.30 floor by merges at 0.0 to
  0.30, and faces scoring under the default cut had joined some of them, a monkey's face among them). When childhood photos
  were added, `follow_childhood` pulls unnamed groups whose faces are all in the child band and match the target's
  childhood photos first at >= max(child_accepted, the child cut), with PULL_MARGIN and the child lead. The event
  records `accepted`, `below_floor`, `child_accepted`, `cross_age`.
- "Same child?" (`cluster.childhood_suggestions`): pairs of people whose childhood-photo centroids are >=
  `settings.child_suggest_threshold` (0.40) alike, without dismissed or co-occurring pairs. `GET /api/suggestions`
  adds up to `CHILD_SUGGEST_MAX` (5) of them after the usual suggestions (not counted against `limit`), as `kind:
  'childhood'` with childhood photos as samples (a pair in both lists shows as a childhood one; a pair whose
  childhood photos are all in the vault or the trash shows no card). Accepting one is a merge. Two cards on the
  owner's library.
- "Who is this as a child?" (`cluster.childhood_candidates`, `GET /api/people/{id}/childhood -> {candidates:
  [{person, similarity}]}`): for an unnamed group with at least half its face weight in the child band, up to 3
  named people: first those whose own childhood photos look like the group's (>= 0.40), then the rest by the group's
  childhood centroid against their adult centroid. No bar and no percentage shown: the face model links a young
  child to their adult self only weakly (0.2 to 0.35; on the library the right person still ranked first), so the
  owner decides. Empty for anyone else.
- UI: `SuggestionsPanel` shows a childhood pair with the badge "Same child?" and, when one side is named, "Compared
  with the childhood photos already on NAME."; `PersonPage` shows an unnamed childhood group the card "These look like
  childhood photos." with "Who is this as a child?" (the candidates, the first marked "Most alike", and "Someone
  else" for the usual picker); picking one opens the usual merge confirm. No ages are shown anywhere.
- Tests: `tests/test_face_ages.py` (storage, backfill, readiness, drawn and moved boxes, undo),
  `tests/test_age_matching.py` (adults bit-identical, the mask, cut, lead and strict-only learned thresholds,
  cross-age and below-floor merges, old merges under the floor, the learner, childhood follow, suggestions and
  candidates, the off switch).

### Recognition trained on one person (2026-11-13)

The owner cleans one person's faces and says they are right. Recognition is then trained on that person alone and
looks again at every face that is not theirs (`app/learning/person_model.py`, `app/analysis/person_scan.py`). What it
finds becomes guesses: nothing is tagged until the owner confirms.

Why the usual score is not enough, measured on the owner's library for the person with the most faces (5,096, 72
tagged by hand, 706 faces taken off them): their learned cut had risen to 0.7555, because the removed faces raise it
(section 18), so about a quarter of their own clear faces would not be tagged today. About 148 of the removed faces
still score 0.5 to 0.88 against the person by the top-3 mean; 123 of them sit on two other named people whom the face
model can hardly tell from the person (2,597 of the person's faces have a removed face at cosine 0.6 or more). No cut
on the top-3 score separates them (it finds 19 percent of the person's faces at 1 mistake in 1,000 once the removed
faces count as negatives). A logistic regression on the 512 values, trained with those faces as negatives, does.

- The model (`person_model.train(conn, person_id)`, pure numpy, about 35 s and 410 MB on that person): positives are
  the person's faces of weight > 0 outside the child band, each by its face weight (times 1.5 when the owner assigned
  it). Trusted negatives: faces the owner took off the person (`guess_rejections`), faces the owner assigned to someone
  else, and the other faces of photos (not videos) the person has a face in, except ambiguous ones (top-3 against the
  person `>= 0.5`: a mirror, a print or a collage shows one person twice) unless such a face sits on another named
  person it matches better by 0.05 (an unnamed group may hold the person's own faces, the scan takes them out of
  such groups, so it never makes a face a negative). Never the faces the app put on other people elsewhere, nor
  loose faces: those are what the scan looks at.
- Conflicts (`person_model._conflicting`): a positive the owner did not lock whose nearest explicit negative is at
  cosine `>= 0.55` is a suspect. A suspect is cleared only by trusted faces: the owner's own faces, the faces that
  are no suspects, and each suspect already cleared (its top-3 to the trusted faces reaches its cosine to that
  negative); this repeats until no more suspect is cleared (at most 50 rounds). So suspects never vouch for one
  another: several faces of one look-alike on the person, each other's best matches, would otherwise clear
  themselves and be trained as the person (the first version measured each suspect against all of the person's
  faces; with 27 such faces in the test world 14 were trained as positives). The plain rule of that version (nearer
  to the negative than the top-3 to all the person's other faces) still always counts, and it alone counts when the
  rule would set aside more face weight than it keeps: the trusted faces do not stand for the person then (someone
  whose every face resembles a face taken off them, with none locked). Conflicts are left out of the training and of
  the reference and listed for the owner, strongest first. On that person 3,537 of the 5,000 faces the owner did
  not lock are suspects (two other named people can hardly be told from them), 1,516 faces are trusted at the start,
  and after five rounds 38 conflicts are left, each at cosine 0.67 to 0.99 to a face that now sits on another
  person. The first version's rule found 22 of them; judging every suspect against the first trusted faces only,
  with no further rounds, would set 893 aside.
- Fit and check: five folds by the local calendar day of the photo; a photo without a date folds by itself, and when
  the days do not spread the examples (a person seen on one day) every photo does. The check records which it ran
  (`check.folds`: `day` or `photo`) and the card words its sentence from it; the per-photo check is the weaker test,
  because photos of one day share clothes, light and angle. Out-of-fold regression logit (weighted logistic loss,
  L2 0.001, each side half the weight, 400 momentum steps from zero, so deterministic) and out-of-fold top-3
  (reference: the positives of the other folds). A second logistic regression stacks `[top3, logit / 10, w, top3 *
  (1 - w), (logit / 10) * (1 - w)]` (w the face weight) into one probability; the check fits it on four folds of
  out-of-fold rows and scores the fifth. `guess_p` is 0.5; `sure_p` is the lowest cut on a 0.01 grid whose checked
  precision is 0.995 or more. On that person (folds by day): at 0.5 it found 4,895 of 5,015 held-out faces with 33
  of 5,474 negatives mistaken, at 0.9 4,754 with 11, and `sure_p` came out at 0.59 (4,876 and 22); of 331 unclear
  faces (weight under 0.5) it found 263 at 0.5. With the first version's 22 conflicts the same check read 4,892 of
  5,031 with 50 mistaken at 0.5, and `sure_p` 0.67: the 16 further faces set aside were teaching the model the
  look-alikes.
- Storage: `learned/person_<id>.json` (`store.save`): `{rev, person_id, trained_at, faces_version, w[512], b,
  stack{w[5], b}, guess_p, sure_p, ref_ids, counts, check{folds, positives, negatives, unclear_*, cuts[{p, found,
  mistaken, unclear_found, unclear_mistaken}], claim{the same at 0.8}}, conflicts[{face_id, like_face_id,
  like_person_id, sim}], scan}`.
  Not in `train.LEARNERS`: only the owner trains it. A learning reset removes it; with learning off `load()` answers
  None and nothing below runs. Deleting the person, or merging them into someone else (also by giving them a name
  that exists, and the unnamed look-alike groups a merge pulls along), forgets it: `person_model.forget` hands the
  file to `undo.discard`, so the undo of that action brings the model back with the person (section 33). Whether
  the model is paused is not in the file: `persons.recognition_paused_at`, a journaled column.
- Scoring a face (`Model.probability`): its top-3 against the reference (`ref_ids` still on the person, weights as the
  face index), the regression and the stacker. Two vetoes in `person_scan`: a face the owner took off the person is
  nearer than the person's own faces (cosine `>= max(top3, 0.6)`), and another named person scores the face at least
  as high as its top-3 here (unless that person's own model refuses it).
- The scan (`person_scan.scan`): candidates are faces with `assigned_by` auto or guess, weight > 0, outside the child
  band, in live items outside the vault, not turned down for the person, in items the person is not in yet (a face,
  a person box or a tag; videos too, since a confirmation is refused there), and not in an unnamed group the owner
  said is not the person (a merge suggestion turned down, `merge_dismissals`; `merge_item` drops the guess of a face
  that came out of such a group when the owner says so later). A loose face or one in an unnamed group
  with probability `>= guess_p` is picked; `guess_item` then writes the guess for every picked photo, so one code
  path decides (one person per photo, clothes agreeing). A face in an unnamed group is taken out of the group first
  (`person_id` NULL, the person box around it follows, an emptied group is pruned), but only when `guess_item` would
  keep the guess (it is tried with the face added to the photo's open boxes, nothing written); its evidence remembers
  the group (`was_person_id`). Faces the app put on other named people are never touched: those at `sure_p` or more
  are listed (`scan.elsewhere`). A guess of an earlier scan that the model no longer makes is cleared. On that person:
  23,110 faces looked at in about 10 s, 15 guesses (8 at `sure_p` or more, 4 in videos, 7 out of unnamed groups, 13
  of them under the old 0.7555 cut), 2 faces on other people listed; a second scan writes nothing, and taking the
  guesses back leaves every face as it was before the look.
- The guess: `assigned_by 'guess'`, `guess_score` the probability, `guess_evidence {"basis": "face", "score", "face"
  (the top-3), "p", "model": {"rev", "trained_at"}, "was_person_id"?, "also"?, "place"?, "place_llr"?}`.
  `guess.guess_item` calls `person_scan.merge_item` after the clothes, trait and position paths: the same person by
  both keeps the higher score and notes the other (`also` on a model guess, which also keeps the position part of
  the guess it agrees with, `place` and `place_llr`, so the Position filter, the two crops and the sweep for changed
  position rules still find it; `p` on a clothes, tattoo or position guess); a different person: the higher score
  wins, except that
  a candidate for an unnamed person never displaces the model's; then one person per photo again. So the guess is
  recomputed on every run, and a new photo gets one when it is analysed. Without a model the result is the same
  object, untouched. A guess whose model has no say any more (paused, forgotten, learning off) stays until the owner
  answers it. `apply_window_change` keeps basis `face`; `record_outcome` records `{basis, p, face, score}` and the
  clothes weights leave these out (`people.RULES_REV` 4).
- Answers: confirming is the usual confirm (the face becomes the owner's, an example for the next training).
  Turning down records the rejection. `withdraw` clears the model's open guesses (those in the trash and the vault
  too, so none comes back with its photo), pauses the model (`persons.recognition_paused_at`, so the undo of the
  action un-pauses it and a redo pauses it again) until the next look or training, and has `guess_item` run again on
  those photos so guesses of other evidence come back. The photos where a guess of other evidence carried the
  model's `p` are run again as well, so no card goes on saying "trained face NN%" for a model that is paused or gone.
- A face that leaves an unnamed group in a look, or goes back into one, may be the face a position guess in a photo
  taken moments apart was followed from (section 35): the photos lined up with its photo are queued for a fresh
  guess (`place.requeue_around` over the full chain length).
- Back into the group (`person_scan.regroup`): a face that a scan took out of an unnamed group sits in that group
  again whenever its guess goes without the owner's answer for that face. Every path that clears such a guess asks
  first which faces remember a group (`left_groups`) and regroups them in the same transaction: turning down
  (`guess.reject`), taking back, `guess_item` (the model no longer names the face, the person is in the photo now,
  the group was dismissed, the photo went to the vault), a guess that cannot be confirmed (`guess.confirm`), the
  guess window (`apply_window_change`), the trait rules (`trait_match.apply_settings_change`), the person deleted
  (`cluster.clear_guesses_for`) and a merge that drops the guess (`cluster.merge_persons`). `guess_item` also keeps
  the memory on whatever guess the face holds, and `recluster_unassigned` leaves a face with a `face` guess or with
  that memory to the owner. The group when it still exists unnamed; a group that went when the scan left it empty
  comes back as one new unnamed person for all its faces, and the open guesses that still remember it are
  re-pointed to the new one (written as `guess_item` writes evidence, so the next run changes nothing), so answers
  given one at a time rebuild one group, not one group per face; a group merged into another person is remembered as
  that person (`merge_persons` re-points); a group that has a name now is never joined (a clear face starts a new
  unnamed group for the faces that return with it, an unclear one stays loose); a group the owner deleted is
  forgotten as a place to return to. Never a matching step, never a named person. What is lost when a scan
  empties a group: its id, and with it the merge suggestions the owner turned down for that group against other
  people (the rows cascade with the person).
- The automatic matching of new faces (`cluster._score`, default path, faces outside the child band) asks the
  trained models twice.
  - The gate (`_drop_gated`): a model that gives a face less than 0.2 (`GATE_P`, the option's score as the top-3)
    takes its person off that face's options. Only that option goes, and the face takes its next option exactly as
    if the person had scored under their cut. So the gate can change who a face is tagged as, not only whether it
    is tagged: a look-alike's own new face whose best raw score is the trained person goes to the look-alike, which
    is what the gate is for, and a face of the person that the model misses goes to its runner-up too (out of fold,
    55 of that person's 5,015 faces fall under 0.2). Cutting the list at the refused person instead was tried and
    dropped: the look-alike's face then had no option left and started a one-face unnamed person each time. A
    paused model (the owner took its guesses back) has no say here either, as in the claim: the plain rules
    decide until the next look or training.
  - The claim (`_claim`): every model at work (learning on, not paused, its person there) gives each face its
    probability from its own top-3, whether or not the person is among the face's options. A face is kept for
    that person at 0.8 or more (`CLAIM_P`), and only when the guess path would offer it to them as well
    (`person_scan.offerable`: the two refusals of `merge_item`, asked through the same functions): no face the
    owner took off the person is as near as the person's own faces, and no other named person scores the face as
    high as the person's top-3 (unless that person's own model refuses it). The second test is what keeps the
    claim from tagging the wrong person: someone named after the training was no negative for the model, so it
    was sure of their new faces, which matched them at 0.92 and the trained person at 0.71, and without the test
    those faces were tagged as the trained person. A face that fails either test is not kept: it goes where the
    plain rules send it. A kept face's options shrink to that person alone when they are its best option, and to
    none when they are not: their learned cut or the rejected look-alikes rule kept them out, or someone else
    scores the face higher. That someone is an unnamed person (a named one fails the second test): a look-alike
    the owner never named sits in an unnamed group, the model never saw them as a negative, and the second test
    asks about named people only. In a test library such a face matched its own group at 0.81 and the trained
    person at 0.61 to 0.71, the model gave it 0.80 to 0.97, and with the person among its options it was tagged
    as them. Now it waits as a guess, and after a no it joins its group at the next recluster. Two models sure of
    one face: the higher probability has it. A face the owner said is not the person is never kept for them, nor
    is a face of weight 0 (no guess is made for one). The claim removes other people's options. It never tags a
    face that the plain rules would give to someone else or refuse the person: where the model and the plain
    rules disagree, the face becomes a guess. The check stores what this cut found and mistook (`check.claim`):
    on that person 4,820 of 5,015 held-out faces (96 percent) with 14 of 5,474 negatives mistaken.
  - A kept face that the options gave nobody (`_apply_item`, `cluster.Kept`) stays loose and starts no unnamed
    person: a guess names the person for it (`merge_item`, basis `face`), and the owner confirms. The analysis
    makes that guess for its own item (`pipeline._after`); on every other path that assigns faces (the look-alikes
    that follow an unassign) `assign_new_faces` queues a `guess` job for the item, or the face would wait with no
    person, no group and no guess. It waits only where that guess can be made and kept: the item takes guesses
    (not in the vault or the trash), the person is not in it (through a face, also one assigned in the same pass,
    a person box or a tag: a guess never names someone who is in the item, and a confirmation names a person once
    per item, clearing their other guesses there), and no other face of the item is kept for them already (the one
    the model is surest of waits). A kept face that fails one of these goes back to the plain rules without the
    named people: it joins an unnamed person among the options it had, or starts one when it is clear (in
    `recluster_unassigned`, which starts nobody, it may only join). It never takes a named person the model kept
    it from. A clip is where this matters: the tracks of one person are several faces, and once one of them is
    theirs no guess can name them on another.
  - Waiting is a forecast, and it can be wrong by the time the guess is made: the same analysis puts a new photo
    in the vault after it assigned the faces, a person box of the photo holds a stronger guess for the person
    (one person per photo), sibling tracks of a clip went to a named look-alike in the same pass and that person
    now scores the kept track as high, or the guess failed. So after its guess the analysis settles what it left
    waiting (`cluster.release_unguessed`, from `pipeline._after`): a face that still has no person and no guess
    joins the unnamed person among the options it had when it was kept (if that person is still unnamed), or
    starts one when it is clear. Never a named person. The paths that queue the guess as a job (the look-alikes
    that follow an unassign) do not settle: a face whose guess does not come there stays loose, among the
    unassigned faces, and so does a kept face whose guess the owner later turns down, until a recluster joins it
    to a match.
  - Why the claim: that person's learned cut had risen to 0.7555 and the rejected look-alikes rule removes them
    as an option for faces that resemble a removed face, so a look-alike with a lower cut won their new faces.
    Measured by scoring existing automatic faces as if they were new (each batch of 64 left out of the reference,
    the regression has seen them): of that person's 5,006 automatic faces the plain matching sent 3,292 back to
    them, 1,277 to someone else and 437 to nobody, and the gate changed none of that. With the claim 3,292 still go
    to them, 1,499 are kept loose for a guess, 111 go to someone else and 104 to nobody. The model was sure of 21
    more that it does not keep, because a face taken off the person is as near as the person's own: they go to
    someone else, as the plain rules send them. No face was refused because another named person scored it as
    high. Of the 12,130 automatic faces on other named people the model is sure of none, so the claim keeps none
    and the gate takes the person off none: nothing changes for them.
  - The same items taken whole (every face of an item unassigned, left out of the reference and scored together,
    `_apply_item` deciding): of the 4,406 such faces in 4,401 photos, 3,458 are tagged as the person, 835 wait for
    a guess and 9 are kept where no guess can follow (the person got another face of the photo 5, a second kept
    face 2, the vault 2). Of the 600 in 530 clips, 433 are tagged, 77 wait and 31 cannot be guessed (the person
    is on another track 29, a second kept track 2). Of those 40, 19 joined an unnamed person, 10 started one and
    11 stayed loose, not clear enough to start a person and matching no unnamed one (as any such face does); none
    went to a named person. Before the rule above all 40 stayed loose, with no group and no guess.
  - `recluster_unassigned` scores through the same rules, so a face that waits for its guess keeps waiting
    there, and it also leaves faces that hold a `face` guess, or any guess on a face a scan took out of a group.
    It scores before its write transaction, so the unnamed people a kept face may join are read again inside it:
    one that got a name in between is not joined.
    The person's learned cut is unchanged, so a new face of theirs under it becomes a guess, not a tag.
- A run (`person_scan.start`): training then the scan on one background thread (`worker.run_in_background`, nice 19,
  own connection, nothing journaled), progress in module state; inline with `WORKERS_ENABLED` off. The state of a
  run is lost with the process, so a run leaves a mark in `kv` (`recognition_run:<person>`, `{train, look,
  started_at}`) from its start to its end. A mark without a run means the app was restarted under it: the status
  then reports an error ("The training stopped before it finished, because the app was restarted", or that it was
  trained but the look stopped), and the next run replaces the mark.
- Tests: `tests/test_person_model.py`.

## 11. Documents (`app/documents`)

Text extraction order: text layer (pdf) -> OCR (rapidocr; English, Hindi and Marathi: each detected line is
read by the Latin and the Devanagari recogniser and the better reading wins, section 7 step 3) -> office
extractors -> plain text. `item_text.text` is the concatenation of pages separated by form feeds. FTS columns:
`filename`, `text`, `doc` (document title, type label, field values), `people` (person names), `labels`.

Hindi and Marathi: `fields.normalize_text` folds Devanagari before parsing (digits to ASCII, danda to a full
stop, zero-width joiners dropped, nukta letters decomposed); `find_dates`, `parse_date` and `parse_month`
know the Hindi and Marathi month names; the shared label lists (`*_LABELS_LOCAL`, `STOP_LABELS_LOCAL`,
`LOCAL_GENDER`, `LOCAL_RELATIONS`) and `detect.LOCAL_RULES` carry the Hindi and Marathi words for names,
parents, dates of birth, validity, amounts, addresses and document headings, and labels match whole words of
their own script. The ID parsers (aadhaar, pan, passport, driving_licence, voter_id, vehicle_rc), marksheet,
invoice, receipt, salary_slip and the generic fallback read them. A document that carries a name in both
scripts keeps the Latin spelling in `name` and the Devanagari one in `name_local` ("Name (local script)");
`extract.analyze_text` adds a `language` field ("Hindi", "Marathi", "Hindi or Marathi", each optionally
followed by ", English"; `fields.languages_of`, which needs at least two Devanagari words to decide) when the
text has Devanagari words; the generic parser
adds `kind` ("Rent agreement") when it recognises one. Page renders of office and plain-text documents draw
Devanagari runs with Noto Sans Devanagari (else FreeSerif) and shrink a line that would overflow the page
(`media.documents._draw_line`). Search: a query without Latin letters skips the semantic engine
(`search.service.text_only_query`) and the query keeps Devanagari words whole; a nukta letter is searched in
both its precomposed and decomposed spelling.

Type detection (`detect.py`): each registered type has keyword sets, identifier regexes with
checksums where they exist, and negative keywords. Score = weighted hits; the best type wins when
its score `>= 0.5`. Groups and types:

| Group | Types | Key fields |
|---|---|---|
| `id` | `aadhaar` (12 digits, Verhoeff), `pan` (`[A-Z]{5}[0-9]{4}[A-Z]`), `passport` (MRZ TD3, check digits), `driving_licence`, `voter_id` (EPIC `[A-Z]{3}[0-9]{7}`), `vehicle_rc`, `id_card` (college, school, employee and membership cards: organisation from the heading lines, holder, number from a label, the barcode or the longest digit run, course or designation, validity, blood group) | name, number, dob, gender, address, issue/expiry, father/spouse, issuing authority, photo |
| `money` | `bank_statement`, `invoice`, `receipt`, `salary_slip`, `traffic_challan`, `parking_receipt` (section 30) | account/IFSC, period, opening/closing balance, invoice number, GSTIN, total, parties, net pay; challan number, vehicle, violations, amount, payment status, pay portal; receipt number, vehicle, in and out times, amount, operator |
| `education_work` | `marksheet`, `degree`, `certificate`, `offer_letter` | institution, course, roll number, grade, date, employer, designation, CTC |
| `health` | `insurance_policy`, `prescription`, `lab_report`, `vaccination` | insurer, policy number, sum insured, validity, doctor, patient, tests, vaccine, beneficiary id |
| `pass` | `event_pass`, `boarding_pass` | event, venue, date, time, seat/section/gate, booking id, barcode; flight, pnr, from/to |
| `other` | `other` | title, date (OCR text only) |

Parsers return `ParsedDocument(doc_type, confidence, fields, title, doc_date, issuer, person_name)`. A new
rule set (`ingest.DOC_RULES_VERSION`, kv `doc_rules_version`) queues a reparse at startup for the automatic
`other` documents whose text now scores for a type it added (section 30).
Every parser is pure (text + boxes in, fields out) and unit-tested on synthetic renders.

Virtual documents: `GET /api/documents/{uid}` plus the viewer's document panel render the type's
card: the ID photo (`/files/doc-photo/{uid}`), the fields in the type's order, confidence shading,
inline editing (`PATCH` with `source='user'`), the barcode payload, and for passes the Wallet button.

Apple Wallet (`wallet/pkpass.py`): builds an `eventTicket` (or `boardingPass`) from the document
fields and the item's thumbnail, signs `manifest.json` with a detached DER PKCS#7 signature
(`cryptography` `PKCS7SignatureBuilder`, SHA-256, signer cert + key, WWDR certificate added) and
streams a `.pkpass` zip. Needs `WALLET_PASS_TYPE_ID`, `WALLET_TEAM_ID`, `WALLET_ORG_NAME` and the three
PEM files in `DATA_DIR/wallet/`; `GET /api/system` reports `wallet.configured` and the Settings page
explains what to put where. Unconfigured: 409 `{"detail":"wallet_not_configured"}`.

Limits (`media/documents.py`): PDFs are read up to `PDF_MAX_PAGES` (2000) pages for the count, the text layer,
renders and the lazy page route, and `PDFIUM_LOCK` is taken per page. Office zips are refused and the file is
kept as kind `other` (reason logged) beyond `ZIP_MAX_*`: 10000 entries, a 2 MB central directory, zip64,
256 MB uncompressed in total, a 64 MB `.xml`/`.rels` part, or an entry of 1 MB or more that expands over
200x. XLSX scanning stops at 100000 rows per sheet (empty rows included), 1000 cells per row and 2000000
cells per workbook (a cut sheet ends with `...`). Plain text is read up to 16 MB: `page_count`, logical pages
and renders stop at 1000, and the text past that is appended to the last page's text so `item_text` and the
FTS index hold the whole file; lines longer than 5400 characters are split into blocks. Typed-document
detection and parsing read at most the first 400000 characters or 150 pages (`extract.ANALYZE_MAX_*`).

## 12. Search

`GET /api/search?q=` runs two engines and merges:
- semantic: SigLIP text embedding against `item_embeddings` (whole image and frames) and
  `faces.torso_embedding` (person crops), with the hub's relevance gate, contrast gate and colour check;
- text: FTS5 `MATCH` over `items_fts` with `bm25` ranking and snippets.
`mode=auto` returns both lists merged (semantic order first when the query looks visual, text first
when FTS has strong hits: any bm25 hit whose terms all matched); `mode=semantic` / `mode=text` force one.
A query whose letters are all non-Latin (Hindi, Marathi) skips the semantic engine in `mode=auto`
(`service.text_only_query`; SigLIP is English-trained) and returns the text hits alone.
The contrast gate's threshold is capped at an absolute relevance of 0.25, so a subject that fills the library
still matches. Offset pages after the first come from a short-lived in-process copy of the merged list (8
queries, 10 minutes; the visibility of the page's ids is re-checked), and snippets are built for the
returned page only.
Filters (`category`, `kind`, `person`, `from`, `to`, `vault`) apply in SQL. NSFW and deleted items
are excluded unless `vault=1` with an unlocked session.

## 13. HTTP API

All JSON. Errors are `{"detail": "<message or code>"}`. Codes used by the frontend:
`login_required` (401), `vault_locked` (403), `csrf` (403), `setup_required` (409, login before setup),
`already_set_up` (409), `invalid_credentials` (401), `too_many_attempts` (429, one budget shared by password
login, passkey login, vault unlock and password change), `challenge_invalid` and `registration_failed` (400),
`credential_exists` (409), `password_or_passkey_required` (400), `offset_mismatch` (409),
`search_unavailable` (503, only when the one engine asked for cannot answer), `wallet_not_configured` (409),
`not_a_pass` (422), `length_required` (411, simple upload without `Content-Length`), `Invalid cursor` (400).
`vault_locked` is also the answer of passkey registration and removal while the vault is locked.

Auth (`/api/auth`): `GET state`; `POST setup`; `POST login`; `POST logout`; `POST logout-all`;
`GET sessions`; `DELETE sessions/{id}`; `POST password {current_password,new_password}`;
`POST passkeys/register/options`; `POST passkeys/register/verify {name, credential}`; `GET passkeys`;
`PATCH passkeys/{id} {name}`; `DELETE passkeys/{id}`; `POST passkeys/login/options` -> `{challenge_id, options}`;
`POST passkeys/login/verify {challenge_id, credential}`; `POST passkeys/assert/options` -> `{challenge_id, options}`;
`POST vault/unlock {password} | {challenge_id, credential}`; `POST vault/lock`.
`AuthState = {setup_required, authenticated, username, passkeys, vault_until, session_id, site_title}`.

Library (`/api`): `GET items` (query: `category`, `kind`, `person` repeated, `from`, `to` (ISO dates),
`favorite`, `misclicks=exclude|include|only` (default exclude), `forwards=exclude|include|only` (default exclude) and
`received=<app>|any` (section 29), `dups=collapse|all` (default collapse), `vault`,
`sort=taken|added` (default taken desc), `cursor`, `limit` 1..200 default 60) -> `ItemPage{items, next_cursor, total}`;
`GET items/{uid}` -> `ItemDetail`; `PATCH items/{uid} {category?, nsfw?, misclick?, forward?, favorite?, filename?}`;
`DELETE items/{uid}` (trash); `POST items/{uid}/restore`; `POST items/bulk {uids, action: delete|restore|keep|category|nsfw|favorite|unfavorite|forward|unforward, value?}`;
`POST items/{uid}/reprocess {stages?: [analyze, text, preview]}`; `GET items/{uid}/pages`; `GET items/{uid}/text`;
`GET trash` -> `ItemPage`; `POST trash/empty`; `DELETE trash/{uid}` (purge one);
`GET review/misclicks` -> `ItemPage` (with `misclick_reasons`); `GET review/forwards` -> `ItemPage` (the forwards set apart, section 29); `GET review/guesses?person=&basis=all|clothes|trait|face|place` -> `GuessPage` (served by the people router; `basis=clothes` keeps guesses with no `basis`, `basis=trait` keeps `trait` and `clothes+trait`, section 34, `basis=face` the guesses of recognition trained on one person, section 10; `basis=place` keeps every guess whose evidence has a `place` part, so `place`, `clothes+place` and a trait guess that position agrees with, section 35; `evidence.place` carries `ref_crop_url`, the crop of the box the person was followed from, while that photo is live);
`GET duplicates` -> `DupGroupPage{groups:[{best: ItemCard, copies: [ItemCard]}]}`; `POST duplicates/{uid}/best`;
`GET stats` -> `Stats{items, by_kind, by_category, people, documents, misclicks, forwards, guesses, trash, queue{queued, running, failed, by_queue}, disk{free_bytes, total_bytes, originals_bytes}, analysis{pending, done, error}, text{pending, done}}`.
Listing rules: `total` is counted on the first page only (cursor pages carry `total: 0`) for `items`, `trash`,
`review/misclicks` and `people/{id}/items`; pages come from the `items_order` and `items_added` indexes
(`common.page_sql`, card columns joined for the page rows only). Duplicate collapse (`dups=collapse`, shared
with search through `visible_sql`): an item of a group is listed when it is the best, or when the best is not
a live item on its own nsfw side, so a normal copy under a vault best stays in the library and a vault copy
under a normal best stays in the vault; `dup_count` counts the other live members on the same side and
`dup_group` is null when the best is on the other side. `GET duplicates` lists copies on the best item's
side only.

`ItemCard = {uid, kind, version, filename, width, height, duration_s, taken_at, created_at, category, nsfw, misclick,
misclick_reasons, favorite, dup_group, dup_count, page_count, doc_type, urls{thumb, preview, preview_small}}`.
`ItemDetail = ItemCard + {mime, size_bytes, bitrate, fps, video_codec, has_audio, camera, gps{lat, lon, alt} | null, exif_datetime,
category_source, category_scores, nsfw_score, quality_scores, labels[], faces[FaceOut], bodies[BodyOut], tags[PersonRef],
people[PersonRef], document: DocumentOut | null, text_status, analysis_status, preview_status, deleted_at,
urls{thumb, preview, preview_small, original, download, video, pages[]}}`.
`FaceOut = {id, x, y, w, h, det_score, size_px, quality, frame_time_s, spans, person: PersonRef | null, assigned_by, guess: GuessOut | null, url}`;
`BodyOut = {id, x, y, w, h, det_score, face_id, person, assigned_by, guess, url, traits: TraitOut[] | null}` (the
body's trait rows, hidden ones too with `hidden: true`; never a BLOB column, section 34);
`TraitOut = {id, kind, part, x, y, w, h, label, crop_url, hidden, hex, item_uid, taken_at}` (`crop_url` is
`/files/trait/{id}`; `hex` for nails; `item_uid` and `taken_at` on the person traits rows);
`GuessOut = {person: PersonRef, score, evidence}` (for trait evidence `evidence.traits[]` entries gain `crop_url` and
`ref_crop_url`, built from the ids without a query); `PersonRef = {id, name, cover_url, kind, relation}` (`relation`
`in` | `related` in an item's `people` and `tags`, section 10; a reference from a box is always `in`).

People (`/api`): `GET people?q&filter=all|named|unnamed&sort=items|name|recent&cursor&limit` -> `PersonPage`;
`POST people {name, face_ids?}`; `GET|PATCH|DELETE people/{id}` (`PATCH {name?, cover_face_id?}`; 422 when the
cover face's photo is in the trash or the vault; `PersonRef.cover_url` skips such a chosen cover and uses the
automatic one);
`POST people/merge {target_id, source_ids}` (PersonOut with `followed`: the look-alikes merged along, section 10);
`POST items/{uid}/animals {x, y, w, h, species?, t?}` (an owner-drawn pet box on a photo or, with `t` seconds, on a
video's frame, 201 AnimalOut; 400 `species_needed` without a species or a model, 400 `frame_time_needed` for a
video without `t`, 422 for a box outside the frame or under half a percent of it) and
`POST items/{uid}/animals/detect {threshold?}` (the detector once more at a lower threshold, the new boxes; on a
video over the poster and up to 12 sampled frames; 409 `models_unavailable`), section 22; `GET people/{id}/faces`;
`GET people/{id}/items?relation=in|related|all` (default `in`: faces, bodies, animal boxes and 'in' tags; `related`:
the items about them, section 10);
`GET faces/unassigned`; `POST faces/assign {face_ids, person_id | new_person_name | null}`; `DELETE faces/{id}`;
`POST bodies/assign {body_ids, person_id | new_person_name | null}`; `DELETE bodies/{id}`;
`POST guesses/confirm {faces: [id], bodies: [id]}` (the box is locked as the owner's, `assigned_by 'user'`, and
marked `by_guess` so it still shows as the system's find, section 14; any later hand assignment clears the mark; a
re-analysis carries it; confirmations from before the mark existed are marked once at start from the guess
learner's log, kv `guess_marks_v1`); `POST guesses/reject {faces, bodies}`;
`POST guesses/correct {faces, bodies, person_id | new_person_name}` -> `AssignOut` ("Someone else", 2026-11-09: in one
action, so one undo, the guess is learned as wrong exactly as `guesses/reject` does (a `guess` learning event with
label 0 for the guessed person, and a `guess_rejections` row so that guess is not made again) and the box goes to the
person named as `faces/assign` and `bodies/assign` do, the owner's own choice with `by_guess` 0; a box guessed as that
very person is a yes, as `guesses/confirm`; a box with no open guess is simply assigned. Before this the owner had to
say yes then change it, which taught the guess learner a wrong yes, or no then tag it, two steps. The guess cards
(person Guesses tab, review page) have "Someone else" with a name field; the viewer's guess chips have a third
button that opens the box editor at its name field (an editor already open for that box stays open and moves to
the field); in the box editor a name picked for boxes holding an open guess (the box, or any track of a whole-video
group) goes through this route instead of a plain assignment);
Recognition trained on one person (section 10; people only, 404 for a pet): `GET people/{id}/recognition` ->
`RecognitionOut {available, reason, learning, trained, trained_at, counts, check{folds: day|photo, ..., claim}, guess_p,
sure_p, conflicts, elsewhere, guesses, withdrawn, scan{at, looked, guesses, sure, detached, elsewhere}, running:
train|scan|null, progress{done, total, step}, error, answers_since, busy}` (`available` follows the count the
training makes: the face weight of the person's faces outside the child band must reach 20, and `reason` says what
their faces count as; `error` is why the last run failed, or that it stopped when the app was restarted);
`POST people/{id}/recognition/train {scan = true}` -> 202 `{started}` (409 with the reason when learning is off or the
person has too few clear faces, 409 `busy` while another person's run is going; the run is in the background and
writes no undo action); `POST people/{id}/recognition/scan` -> 202 (409 `not_trained`);
`GET people/{id}/recognition/conflicts?limit&cursor` -> `{items: [{face: FaceListItem, like: FaceListItem | null,
like_person: PersonRef | null, sim}], next_cursor, total}` (those still standing, strongest first; the cursor is a
keyset on `(sim, face id)`, so settling a conflict never makes the next page skip one);
`GET people/{id}/recognition/elsewhere` -> `{items: [{face, person, p}]}`; `POST people/{id}/recognition/leave
{face_ids}` -> `BulkOut` ("Leave": those faces on other people are not this person, kept as `guess_rejections` rows;
400 for a face that is not on another person; an owner action with its undo); `POST people/{id}/recognition/withdraw`
-> `BulkOut` (an owner action: its undo brings the guesses back and un-pauses the model); `DELETE
people/{id}/recognition` -> `Ok` (runs outside the undo journal: the model and its guesses are derived, and training
again makes both again). Withdraw and forget answer 409 `running` while a run for that very person is going. Conflicts
are settled with `faces/assign` (to the person: locked as the owner's; to another person or to nobody), and so is
"This is NAME" on a face elsewhere.
`GET people/{id}/traits` -> `PersonTraitsOut {tattoos: TraitGroupOut[], mehndi: TraitOut[], nails: TraitOut[]}`
(tattoos grouped by part: `{kind, part, label, photos, first_seen, last_seen, best, item_uids}`; mehndi and nails of the
last 60 days, at most 12 each; vault rows only while the vault is open); `PUT traits/{id} {hidden, verdict:
not_trait|not_theirs}` -> `TraitOut` ("Not a tattoo", "Not this person's", "Show again": an owner action, journalled,
undone through `X-Action-Id`; showing a row again is limited to its photo, and when a look-alike of a hide on another
photo still hides it a `shown` hide is added here instead; 404 for a missing row or a trashed item, 403
`vault_locked` for a vault photo while the vault is locked; the photo's guesses are worked out again after the
commit, outside the action's journal); `GET people/{id}` fills `PersonOut.trait_count`; the routes that move, draw or
delete a person box kick the traits runner after their commit (section 34); `GET /files/trait/{id}` serves a
trait's 192 px WebP crop from `body_traits.crop` (its own query, as the crop lives in the database: 404 for a missing
row or a trashed item, 403 `vault_locked` for a vault photo while locked, `no-store` for the vault and the crop cache
otherwise, with the usual `nosniff` and CSP headers; no `?v=`, since a rescan writes new ids);
`GET suggestions`; `POST suggestions/dismiss {a, b}`; `POST items/{uid}/tags {person_id | new_person_name, kind?,
species?, relation?}` (a person or a pet: `kind 'pet'` with `species` names a new pet, 400 `species_needed`
without one, section 22; `relation` `in` | `related`, default `in`, the other relation for someone already tagged
switches it, section 10; answers the item's tags as `PersonRef[]` with `relation`);
`DELETE items/{uid}/tags/{person_id}`; `POST recluster`.
`PersonOut = {id, name, cover_url, cover_face_id, face_count, item_count, guess_count, created_at, kind, species,
animal_count, confirmed_faces, followed, related_count}` (counts over live, non-vault items; `item_count` means
'in', `related_count` the items related to them, section 10).

Documents (`/api/documents`): `GET ?type&group&q&cursor&limit` -> `DocumentPage{items:[{item: ItemCard, document: DocumentOut}], next_cursor, total}`;
`GET types` -> `[{type, group, label, count}]`; `GET {uid}`; `PATCH {uid} {doc_type?, fields?, title?, doc_date?, person_name?}`;
`POST {uid}/reparse`; `GET {uid}/wallet.pkpass`.
`DocumentOut = {doc_type, doc_group, label, confidence, fields{key:{value,label,confidence}}, field_order[], title, doc_date,
issuer, person_name, barcode, barcode_format, photo_url, source, wallet_available, updated_at}`.

Search: `GET /api/search?q&mode&category&kind&person&from&to&vault&limit&offset` -> `SearchPage{query, mode, items:[SearchHit], total, next_offset, indexed, indexing}`;
`SearchHit = ItemCard + {score, match_time_s, snippet, source}`.

Uploads: as the hub (`POST /api/uploads`, `PUT /api/uploads/{id}/chunk?offset`, `POST .../complete`, `GET`, `GET ?ids`, `GET mine?client_id`, `DELETE`), plus `POST /api/uploads/simple`.
`UploadOut = {id, filename, size, received_bytes, status, message, chunk_size, item_uid, item_thumb_url, item_nsfw, created_at}`.

Files (`/files`, GET and HEAD): `thumb/{uid}`, `preview/{uid}`, `preview-small/{uid}`, `original/{uid}?download=1`,
`video/{uid}`, `frame/{uid}?t`, `page/{uid}/{n}`, `face/{id}`, `body/{id}`, `doc-photo/{uid}`. Inline PDFs are
served from `original` with `content-security-policy: default-src 'none'; sandbox allow-same-origin` so the
browser's viewer works.

System (`/api/system`): `GET` -> `{version, gpu{available, name, provider}, models[{key, present, loaded, provider}],
queues, disk, wallet{configured, missing[]}, timezone, data_dir_free_bytes, forwards{enabled, unchecked, forwards, received,
running} (section 29), storage{tiered, tiers[{key, path, holds, free_bytes, total_bytes, shared_with}], migration{active,
running, pending_files, pending_bytes, moved_files, moved_bytes, error, started_at, finished_at}}, backup{enabled, dir, hour,
keep, running, progress, last, snapshots[], next_run_at, free_bytes, total_bytes} (section 31)}`; `POST models/download`
(starts the downloader in a thread; progress via `GET`); `POST housekeeping` (no orphan sweep); `POST retry-failed` ->
`{requeued}` (failed uploads whose bytes are kept, errored jobs, item statuses back to `pending`); `POST backup` ->
`{started, running}` (a snapshot now, section 31); `GET throughput?range=hour|day|week` -> `ThroughputOut` (finished
jobs per bucket by type, section 32; 422 for another range).

3D heads (section 36; schemas in `api/model3d.py`): `GET people/{id}/model3d` -> `Model3dOut {available, reason,
generator, usable_photos, state: 'none' | 'running' | 'done' | 'error', busy, stopping, progress {step, done, total}
| null, error, generated_at, options {max_looks, detail, framing} | null, looks: Look3dOut[]}` with `Look3dOut =
{key, label, age, age_band, hair, hair_label, facial_hair, facial_hair_label, photos, taken_at, item_uid,
picture_url, mesh_url, bytes, vertices}` (reading it starts nothing; `available` false comes with a plain `reason`;
`busy` means another person's job is running; `stopping` means Stop or Delete was pressed and the job is on its way
out; 404 for a pet); `POST people/{id}/model3d {max_looks 1..12 = 6, detail 1..3 = 2, framing 1.8..3.2 = 2.6}` ->
202 `{started}` (409 `busy`, 409 `generator_missing`, 422 `not_enough_photos`); `POST people/{id}/model3d/stop` ->
`{stopped}` (the model in use stays); `DELETE people/{id}/model3d` (removes the model, stopping the person's job
first); `GET /files/model3d/{person_id}/{rev}/{name}` serves a look's mesh (`model/gltf-binary`) or its picture
(`image/jpeg`) for the model in use and a look that is still listed, cached with `private, max-age=31536000,
immutable` because the rev is in the path. These writes are not owner edits of the library, so they are not undoable
actions (section 33).

Timeline, albums, map and places, share links, video encoders and device uploads: sections 23 to 28.

## 14. Frontend

Routes: `/setup`, `/login`, `/s/:token` (shared link, section 26; no shell); everything else inside
`AppShell` (session guard: redirects to `/login?next=`, `/setup` when `setup_required`): `/` Library
(the timeline grid, section 23), `/people`, `/people/:id`, `/documents`, `/albums`, `/albums/:id`
(section 24), `/map` (section 25), `/review` (tabs Mis-clicks, Forwards (section 29), Duplicates, Guesses), `/vault`, `/upload`,
`/trash`, `/settings` (Account and passkeys, Sessions, System, Learning, Sharing, Wallet, Appearance),
`/m/:uid` (standalone viewer), `*`.

Library: filter bar (category chips, kind, people, dates, favourites, mis-clicks and forwards included on request), search box (one box; results page
shows the semantic and text hits with a mode switch), `MediaGrid` with document tiles (page 1 render in
a paper frame, type badge), video duration, mis-click, forward and duplicate badges, multi-select with bulk actions
(section 29: the viewer's info panel names the app an item came through, with the forward reasons and a keep or mark button).
Viewer: the hub's viewer with a document mode (a page strip of 30 pages either side of the current one plus the first and last, text, the virtual document card), face and
body boxes with guess chips (confirm, reject, assign), an info panel (category with override, quality
scores, labels, camera, dates), download original. A gallery page opens it with `?m=<uid>` (`useViewerParam`,
prev and next over the page's list). An item that drops out of that list while it is open (2026-11-09: the owner
took the person off it on their page, kept a mis-click on the review, changed what a filter matches) stays open at
its old place, still counted in "3 of 10", with prev and next going to the items now either side of it; once the
owner moves to another item it is gone from the list. Deleting an item or moving it to the vault leaves it from
the viewer itself (the next item, else close), as before.
Vault: locked screen asking for password or passkey; once unlocked the same Library component with
`vault=1`; a countdown shows the remaining minutes.
Upload: DropZone, camera button (in-app `CameraCapture` using getUserMedia for photos and MediaRecorder
for clips, with `<input capture>` fallback), queue and outcomes; files shared through the PWA appear here.
PWA: `public/manifest.webmanifest` (name, icons, `display: standalone`, `share_target` POST multipart
with `files`), `public/sw.js` handling `POST /share-target` by storing the files in IndexedDB and
redirecting to `/upload?shared=1`; the upload page drains that store into the upload engine. The service
worker caches nothing else.
Theme: neutral tokens in `index.css` with light and dark (`prefers-color-scheme` and a toggle stored in
localStorage), 16 px gutters, 40 px tap targets, no em dashes or exclamation marks in copy.

Errors and versions. React unmounts the whole tree on an uncaught render error, so three error
boundaries (`components/ErrorBoundary.tsx`) keep that from showing as a blank page: one around the
router in `App.tsx`, one around the routed page in `AppShell` (keyed by the location, so navigating
away clears it; it sits outside Suspense, so a lazy chunk that fails to load is caught too) and one
around the viewer (shows the error over the page with Close). The panel shows the message, Reload,
Try again and a link to the library. The site and the server each carry an API build id: sha256 of
`backend/app/schemas.py`, `urls.py` and `api/*.py` (first 12 hex digits), computed by
`app/build_id.py` at import (so it describes the code the process runs) and by
`frontend/scripts/api-build.mjs` at build time (`__API_BUILD__` from `vite.config.ts`).
`GET /api/auth/state` reports the server's as `api_build`; when both are present and differ, the shell
shows `BuildBanner` ("restart the server from the portal, then reload"). The server serves a copy of
the build taken at startup (`DATA_DIR/site`, `app.main.snapshot_site`), so a later `npm run build`
cannot swap a newer site under an older running server; the banner covers the dev server (Vite
serving sources against a stale API) and any other mix.

Challan and parking receipt cards (section 30): `docMeta.ts` carries the two types' labels, icons
(`TYPE_ICONS`, used by the card header and the tile), field order and labels (`TYPE_FIELDS`, the server's
`field_order` wins) and tile fields (`KEY_FIELDS`); timestamp fields (`in_time`, `out_time`) are shown like
other dates. `VirtualDocument` shows a challan's `payment_status` as a chip (Unpaid in the warning colour,
anything else plain), hides `pay_url` from the grid and offers "Pay or check status" (copies the challan
number, opens `pay_url`, else the Parivahan portal, in a new tab) and "Mark as paid" (`PATCH` fields
`payment_status` "Paid" and `paid_on` today).

Pets, species (section 22): the animal box popover (`viewer/AnimalEditor.tsx`) has "Not a {species}" with
the ten species to pick from (POST `animals/species`; the toast counts the look-alike crops that followed),
and picking a pet of another species in "Move to another pet" shows "Move and make {pet} a {species}"
(assign with `fix_species`). The pet's page has a "These are {species}" picker (POST `people/{id}/species`).
A photo without an animal box offers "Add a pet box" (draw mode on the photo stage: drag over the pet, or
"Whole photo", then "What is it?" picks the species; POST `items/{uid}/animals`) and "Find pets again"
(POST `items/{uid}/animals/detect`); a photo with boxes keeps both under "More" in the People and pets section,
after "Add a person box", which is always shown.

The video strip ("People and pets in this video") shows one chip per group of tracks (section 10, whole-video
editing): the name or species with "N tracks", seeking to the first track; its edit control opens the face or pet
editor in group scope ("Applies to all N tracks of this person in the video", a switch "Only this track"), so a
person who leaves and comes back is named, moved, unassigned, confirmed or removed in one go; a chevron shows the
single tracks for a mis-detection.

Pets in videos and related items (sections 10 and 22): the "Tag someone" button opens "Tag a person or pet", whose
picker (`PersonPicker` with `kind="all"`) lists people and then a Pets section, and offers a typed name as "Create"
(a person) and "New pet"; the latter asks "What is {name}?" with the ten species before posting `kind: 'pet'` and
`species`. The form's switch "Not in the picture, related to them" posts `relation: 'related'`; such tags show under
"Related to" with a link mark. The person and pet pages have a "Related" tab (`items?relation=related`,
count `related_count`; the Items tab keeps sending no relation). "Add a pet box" and "Find pets again" also appear
for videos: draw mode pauses the player and keeps it paused, the box is drawn over the picture area of the frame on
screen ("Whole frame" takes it all) and is posted with `t`, the viewer's `currentFrameTime`; the new box shows as a
track chip under the video.

People in the info panel (2026-11-08, `viewer/InfoPanel.tsx`, `viewer/PersonEdit.tsx`): a person's badge does three
things (2026-11-09). A tap on the face and name shows them on the stage (`onShowPerson`, the viewer's `showPerson`): on
a video it jumps to them exactly as their chip under the video does (the first of their tracks, the box drawn on the
paused frame), and under "In the frame" to the appearance on screen now, once the video can play; on a photo their
boxes light up for four seconds (a playing Live Photo clip stops, and a second tap lights them again). The boxes are
turned on for it, the editors close, and on a phone the details sheet goes down so the moment is in view, the focus
waiting on the details button; the badge of the person the stage shows is marked (`aria-current`, a ring). Someone
with no box here (a tag alone, "Related to") has a plain name. A guess under "Guessed from the same hour, scene and
clothes" works the same way (2026-11-09, `onShowBox`, the viewer's `showBox`): a tap on the guessed name shows that
box, on a video seeking to its frame and drawing it as its chip under the video does, on a photo lighting it for four
seconds; the face beside it still opens the face editor. The arrow beside it ("Open {name}'s page") goes to the person's
page, and the pencil ("Edit {name}") opens the menu (arrow keys and Escape) with the changes the face boxes make, for
everything of theirs on the item at once, without clicking a box. "Change to..." and "Rename" (or "Name" for an unnamed person) open a light card under the chips, which scrolls
itself into view in the phone sheet: "Change {name} to..." holds a `PersonPicker` (people for a person, pets of the
same species for a pet, a typed name creates one) and posts `/api/items/people/move` with this item alone (their
faces, bodies, pet boxes and tags here go to the one picked, remembered as not them, learned like a move from a face
box); "Rename" sends `PATCH /api/people/{id}` (everywhere; a name someone else has merges them). "Not {name}" posts
`/api/items/people/remove` with this item: their boxes here stay, unnamed and remembered as not them
(guess_rejections), and their tags here go. Someone here through a tag alone gets "Remove from this item" (the tag's
DELETE) instead, and a tag the owner made also offers "Make this 'in'" or "Make this 'related'". Like the box editors,
the menu also takes their boxes here away (2026-11-09), each item shown only when they have such boxes and each one
undoable action: "Not a face" (`DELETE /api/faces/{id}` for each of their faces here, kept for the not-a-face learner,
look-alikes going with them), "Not a person" (their person boxes without a face, `DELETE /api/bodies/{id}`) and, for a
pet, "Not a pet" (`DELETE /api/animals/{id}`) and "Species..." (a card with the species buttons of the pet box editor;
like a rename it changes the pet everywhere, `POST /api/people/{id}/species`, its boxes following, so the pet stays
on the item, or joins a pet of the same name and species when there is one, which the message names; the box editor
still changes one box). A face taken away from inside a person box of theirs leaves them on the item through that
box, and the message says so. A box already gone counts as removed (a retry after a failure half way), and the item
is read again either way. After a change that can take the chip away, the focus goes to their pencil, or when they
left to the first pencil in the list, else to "Tag someone". The chips under
"In the frame" (videos) have no pencil. Escape closes the card (not the viewer) and puts the focus back on the
chip's pencil.

Who tagged it (2026-11-09, `lib/origin.ts`, `people/OriginIcon.tsx`): every person or pet name on a box or a badge
carries a small icon, a person (lucide `UserRound`) when the owner put them there and a bot (`Bot`) when the system
did, with the words as a tooltip ("Tagged by you", "Drawn by you", "Tagged automatically", "Suggested
automatically") and in the label of the control around it ("Show Asha in the photo, tagged by you", "Edit face
tagged as Ravi automatically"); `data-origin` is `you` or `auto`. A box (`boxOrigin`) is the owner's when
`assigned_by` is 'user' with a person on it, "Drawn by you" when the owner drew it (`source` 'user') and the system
matched the person on it (a pet box), the system's when recognition or grouping
put the person there, and a guess waiting for an answer shows the bot of a suggestion, also on a box the owner
placed. A guess the owner confirmed stays the system's find (2026-11-09): the box is locked as the owner's for
learning (`assigned_by 'user'`) but marked `by_guess` (`FaceOut`, `BodyOut`, `FaceListItem`; a tag a confirmed
faceless guess made carries `by_guess` in `ItemDetail.tags`), and shows the bot with "Guessed automatically,
confirmed by you" (origin 'confirmed'); tagging that box by hand afterwards makes it the owner's. A box the detector
found with nobody on it has no icon (the
person page keeps a lock on a face the owner took off someone, "Kept unassigned by you"). Naming an unnamed group
or merging people does not make the faces the owner's: the system grouped them, so they keep the bot until the
owner tags one. A badge that stands for several boxes takes one answer: a person on an item (`itemOrigins`, used
by every section of the info panel so their chips agree) is the owner's when a tag or any box of theirs there is
(then drawn by them, then a confirmed guess, then the system's),
and a video's group chip is the owner's when any of its tracks is (the opened tracks show their own). Shown on the
photo's box labels and the label of the picked video track, the chips under a video, the info panel's chips ("In
the frame", the people, "Related to") and guesses, the face and pet editors' headers, the person page's face and
pet crop tiles (pet crops carry `source`), the review's guess cards and the selection's people list, which shows
the person icon for the items `by_you` counts and the bot for the rest ("Tagged or drawn by you on 2 items", "Found
automatically on 1 item", and "2 by you" in the line under the name when both apply). An older server without `by_you` gets no icon there.

Person boxes (2026-11-08, `viewer/PersonBoxCard.tsx`, section 10 for the server side): the People section of the
info panel has "Add a person box" next to the pet buttons, on photos and videos, for someone in a helmet, seen from
behind or far away, or a face the detector missed. Draw mode (`Viewer` keeps which kind is drawn, 'pet' or
'person') works as for a pet box: drag over the person or take the whole photo or frame; on a video the player pauses
and the box goes with `t`. The drawn box (sky blue, like other person boxes) then opens "Who is it?", a card placed
like the face editor (next to the box on wide screens, a bottom sheet on phones) and rendered into the viewer rather
than the stage, holding a `PersonPicker` (a typed name creates the person) with "Draw again" and "Cancel". Picking
someone posts `items/{uid}/people-boxes` (`person_id` or `new_person_name`); the toast says when a face inside the
box was tagged too. The new box shows with the name like any person box (a video's as a chip under it), and its
editor's "Not a person" removes it.

Face boxes (2026-11-09, section 10 for the server side): "Add a face box" sits before "Add a person box" in the
People section, on photos and videos, for a face the detector missed. Draw mode works as for a person box (no
"Whole photo": a face is never the whole picture), the drawn box is white like a face box, and the same "Who is it?"
card (`PersonBoxCard` with `kind 'face'`, "Face box" over the title) posts `items/{uid}/faces`. It says what went
wrong in words: no face in the box ("Draw it closer around the face, or add a person box when the face does not
show") or the face model not loaded. The new face shows with the name and the person icon like any face the owner
tagged.

Moving or resizing a box (2026-11-09, `viewer/BoxAdjuster.tsx`, section 10 for the server side): on photos and videos
the face and pet editors offer "Move or resize", for every face, person and pet box, guesses included (a video's box
on its own frame: the player goes there and stays paused, the handles sit over the picture and cover the player's
controls, and the chips under the video step aside until it is saved or cancelled). The editor closes
(on a phone the details go down too) and the box gets corner handles over a dimmed photo, the other boxes stepping
back: drag the box to move it, a corner to resize it (touch too: the layer is `data-no-swipe` and `touch-none`, so
a drag never turns into a swipe to the next item). The box itself takes the focus: the arrows move it (Shift for
small steps), + and - resize it around its middle, Enter saves and Escape cancels; the viewer's own keys and the
previous and next arrows stay out of the way meanwhile. Save and Cancel sit under the box, else over it, else
inside its bottom edge, centred on it but always inside the picture (measured, so a box at a phone's edge keeps
them on screen). Adjust mode ends when its box can no longer show: the boxes hidden, a Live Photo playing, drawing
started, or a badge in the details jumping elsewhere. Save sends `PUT .../box` and shows "Face box moved" (person box, pet box) with Undo; nothing moved closes
without a request; `no_face` keeps the handles with "No face there. For someone whose face does not show, use
Add a person box instead." The focus then goes back to the box's own button (on a video, its chip).

Header (2026-11-08, `components/shell/Header.tsx`, `Nav.tsx`, `HeaderMenu.tsx`): on md and up the bar
holds the site name (from lg), Library and People as links, a Documents menu (All documents and the six
document groups, `/documents?group=<id>`) and a More menu in two groups (Browse: Albums, Map, Vault with its
locked or open state; Tidy up: Review with its waiting count, Trash; a dot on the trigger while Review has
items), then on the right the vault countdown while it is open, an Upload button and the account menu (the
owner's initial): "Signed in as", a Settings submenu with the eight sections (`settings/sections.ts`, shared
with the Settings page), a Theme submenu (System, Light, Dark as radio items) and Sign out. Triggers show
icons only below lg. `HeaderMenu` is an ARIA menu: Enter, Space or ArrowDown opens it on the first item,
arrows, Home and End move, ArrowRight or Enter opens a submenu on its first item, ArrowLeft or Escape goes
back, Escape closes and focuses the trigger, Tab closes; a submenu opens on hover with the mouse and on tap,
and flies out to the left when there is no room on the right; a click outside or a route change closes it.
Phones keep the bottom tab bar; its More sheet lists the same groups under headings (Browse, Tidy up,
Account), then the theme switch and Sign out.

Undo (2026-11-09; the server side is section 33). Every write `lib/api.ts` sends carries `X-Action-Id`: the
MutationCache in `main.tsx` gives each mutation one id for all its requests (a bulk loop or a whole-video group
edit undoes as one), and a write outside a mutation gets its own. An answer with `X-Action-Id` means the change
can be undone, and `lib/undo.ts` records it in the undo history, which lives in this browser only
(localStorage `pcc.undo.v1`, newest first, at most 200, shared between tabs). A success toast that describes the
action just recorded (the running mutation's, or one that ended under 2.5 s ago) labels it and shows Undo; the
viewer's own messages do the same. Undo (`lib/undoRun.ts`) posts `/api/actions/{id}/undo`, refreshes every
query and says "Undone: ..." with Redo, which undoes the undo; refusals read as plain sentences (later changes
to the same things, offering to undo those first when this browser made them; too old; vault locked; something
in the way). Ctrl+Z or Cmd+Z undoes the newest change and Ctrl+Shift+Z, Cmd+Shift+Z or Ctrl+Y redoes the change
undone last, never while typing or while a dialog other than the viewer is open. "Undo history" in the account
menu, and in the More sheet on phones, opens a dialog listing every change with its time, Undo or Redo, and
"Clear history" (this browser only).

Guess keys (2026-11-10, `pages/ReviewPage.tsx`, `people/GuessCard.tsx` with `keys`, `Viewer.tsx` with `guessKeys`):
on a computer (a fine pointer), Review > Guesses focuses the first card's Yes, with a ring, once per filter (not when
the person pressed Tab while the list loaded). Enter on the focused Yes confirms (the button's own click; a held Enter
answers once); Space on any of the card's three buttons opens the "Someone else" name box with its field focused and
selected, scrolled into view (its keydown and keyup are prevented, so it never confirms, Firefox included, and holding
it types nothing); X rejects; Esc in an empty name box closes it and puts focus back on Yes (with text, Esc clears it
first). After a decision, once the card has left the list, focus moves to the Yes of the card now in its place, or
back to the same card when the request failed (in the render that enables its buttons again). A line under the intro
names the keys. The viewer opened from a card selects that card's guess (else the first open guess of the item;
focusing a chip's button selects that one), rings its check in the photo and in "Guesses and why", opens the details
once per item, and takes Enter (only when the viewer itself has focus: a focused button keeps its own click), Space
(the box editor at its name field; closing that editor returns focus to the viewer, so the next Enter confirms) and X.
Closing it after its card was answered there focuses the card now in that card's place. None of these keys fire while
typing, with Ctrl, Cmd or Alt, as a held repeat, while a dialog, the tag picker or a Live Photo clip is open, while a
decision is being saved, or in the viewer until the item has been fetched again after a change (a viewer decision
stays pending until then), so a quick second key never answers the same guess twice. The person page and the other
viewers do not take them, and touch screens get no automatic focus and no ring.

Trash keys (2026-11-10, `pages/TrashPage.tsx`, `trash/TrashPreview.tsx`): on a computer the first tile's Restore is
focused once the list is in, with a ring on the tile (not after a Tab while it loads). Arrows move between tiles (Up
and Down by row, nearest in x), Home and End to the ends, and the next page loads near the end; Enter (the focused
Restore's own click) or R restores, and Ctrl+Z undoes it as before; Delete or Backspace opens "Delete forever" with its
confirm button focused, so Enter deletes for good and Esc cancels; Space opens a larger preview (the preview image, a
video with controls, a document's first page; private items stay locked) where Left and Right go to the previous and
next item, Enter or R restores, Delete asks, and Space or Esc closes back to that item's tile. An answered tile's
buttons do nothing until a list fetched after the answer is in (aria-disabled, so focus stays and the arrows keep
moving; an undo before that fetch makes it usable again); when it leaves with the focus on it, the tile now in its
place takes focus (counting earlier answered tiles that left). Loading the next page never cancels the list fetch
that follows an answer. Nothing fires while typing, with Ctrl, Cmd or Alt, or as a held repeat; touch screens get
no automatic focus, ring or hint.

Recognition trained on one person (section 10; `people/RecognitionCard.tsx`, `lib/recognition.ts`): a card under
the header of a named person's page (hidden for pets, unnamed groups, people whose faces count for less than 20 clear
ones and with learning off). Not trained: what it will do in three sentences and "Train and look again". Running: the
step in words and a progress bar, polled every 2 s. Trained: when and on how many faces, the check with its counts as
they are ("On days it was not trained on, it found 4,892 of NAME's 5,031 faces and mistook 50 of 5,474 faces of other
people for NAME"; "On photos it was not trained on" when the check could not hold whole days out, `check.folds`), the
last look with a link to the Guesses tab, "N answers since the training", and "Look again", "Train again", "Take back
its guesses" and "Forget" (the last two behind a confirm). A run that ends with an error says so in a toast and on the
card, also the run the app was restarted under (the status reports it, so it is never taken for a success). 409
`running` reads "A training or a look for NAME is running", 409 `busy` "Another person's training is running". Two
lists open in the card: faces on NAME that look the same as faces taken off NAME (pairs of crops with "This is NAME",
"Not NAME" and "Move to OTHER") and faces the app put on other people ("This is NAME", "Leave"). Every answer is an
owner action with its undo: the usual face assignment, and for "Leave" `recognition/leave`, which keeps the answer, so
the face leaves the list and its count for good. A guess with basis `face`
reads "By face, NN% sure" with a face icon and the line "Looks like NAME's photos, 0.61" (", and the same clothes"
when clothes agree, a tattoo or the nails likewise); when position agrees too, the card adds the position reason
with its two crops and the footprints icon, and the guess also sits under the "Position" chip. A clothes, tattoo or
position guess the model agrees with adds "trained face NN%". Review > Guesses has a "Trained face" chip
(`basis=face`).

Position (section 35; `people/evidence.ts`, `PlaceCompare.tsx`): a guess whose evidence has a `place` part shows
footprints beside the label and its reason in words, "In the same spot as NAME 8 s earlier" (`placeReason`,
`gapPhrase`: seconds under 90, then minutes, "in the same second"), "Followed across 2 photos to NAME, 12 s earlier"
when the reference is more than one photo away, and ", and 2 other people stayed in place" when known people kept
their places. Labels: "By position" (`place`), "By clothes and position" (`clothes+place`), and "position" added to
a trait label that also has the part. A guess card shows the box's crop and the crop of the box the person was
followed from side by side (`PlaceCompare`, the reference a button that opens its photo); the viewer's details and
box editor show the reason line, and the guess chip's tooltip a short form ("same spot 8 s earlier"). A guess from
the position alone shows a percentage: its score is the share of right answers measured for that gap (section 35),
not a learned value. Review > Guesses has a "Position" filter (`basis=place`), and Settings > Learning a "Position"
card with the progress of the lining up ("Lining up photos taken together: N of M.", badge Ready, Working or Off).

Body traits (section 34; `people/evidence.ts`, `TraitCompare.tsx`, `TraitsTab.tsx`, `useTraitActions.ts`):
- Guess labels follow `evidence.basis` (`guessBasis`, `guessLabel`): "Face" and "By clothes" as before, "By" and the
  kinds that counted for a trait-only guess ("By tattoo", "By tattoo and nails", "By mehndi and nails"), and "By
  clothes and tattoo" (or mehndi, nails) for `clothes+trait`. Icons: a shirt for clothes, a pen for a tattoo, a hand
  for mehndi or nails. A trait-only guess shows "strong match" in words, never a percentage
  (`guessSure`, also in the viewer's guess chip tooltip): its stored score is a posterior that is not calibrated
  until the owner has answered such guesses. `clothes+trait` keeps the clothes percentage.
- `traitReasons` writes the reason, for example "Same tattoo on the forearm as NAME's photo from 14 March 2026 and 2
  other photos" (", probably a mirror photo" when the match needed the mirrored crop), "Same blue nail art, long, 2
  days earlier" or "Same mehndi on the hand, the same day"; `daysPhrase` says how far apart, and the time-window words
  of clothes guesses never appear for trait evidence. `againstReasons` reads "No tattoo on the upper arm here, where
  NAME has one".
- `TraitCompare` puts the two crops side by side ("This photo" and the person's photo with its date), 56 px on
  phones and 72 px from sm, the reference crop a button that opens its photo; a crop that no longer loads (a rescan
  replaced the row, or the vault locked) reads "Photo changed"; nails add a colour swatch (`aria-hidden`, the colour
  is named in words). Guess cards show one compare row and the rest under "N more signs".
- In the viewer, the box editor shows "Seen on this person": the box's trait chips (from the body, or the body that
  holds the face). Each chip's menu has "Not a tattoo" (or nails, mehndi), "Not this person's" and, on a hidden
  chip, "Show again" (this photo only); hidden chips are struck through. These are owner actions with Undo
  (`useTraitActions`). The photo stage draws dashed violet rings around the selected box's trait regions; the info
  panel gives each guess chip its reason line and a small compare.
- The person page has a "Tattoos and nails" tab when `trait_count > 0`: tattoos by part ("In 14 photos, March 2025 to
  October 2026", the best crop and the photos), "Mehndi lately" and "Nails lately" (the last 60 days), in a grid of
  one, two or three columns by width.
- Review has the chips "All", "Clothes" and "Tattoos and nails" (`basis`); Settings > Learning shows the traits
  line ("Tattoos and nails: N outcomes, thresholds from N pairs", or the defaults note), the reading progress
  ("Reading people boxes: N of M"), "Reading tattoos and nails is turned off." when `TRAITS_ENABLED` is off, and
  "Paused: a model is missing (...)" when a model file is missing.

3D heads (section 36; `lib/model3d.ts`, `people/Model3dTab.tsx`, `people/Head3dViewer.tsx`):
- A person (not a pet) has a "3D" tab (`?tab=3d`), a lazy chunk. Query key `['model3d', id]`, outside
  `LIBRARY_KEYS`, read again every 2 s while the person's job runs and every 5 s while another person's job is in
  the way.
- Before a model exists: what it does, how many face photos can be used, the options as sliders (Looks, Detail,
  Framing) and "Generate"; when nothing can be made, the server's reason and a disabled button. While the server
  works: the step, a bar and "Stop" (an older model stays below it); after Stop or Delete the card reads
  "Stopping" until the job is gone, which takes about a second. After a failed or shortened run: its reason.
- With a model: the head, a line that says which photo it came from and when, with a link to the photo, the looks
  as a row of their pictures, and the controls. Look: sliders Age, Hair and Facial hair. Their stops are the values
  that exist among the looks (age bands; hair and facial hair labels in the order of their levels), each with a
  tick and its value in words. Moving one decides that axis first; among the looks at the new stop, the one nearest
  to the head on screen on the other axes is shown and the other sliders follow it. So the three sliders always
  describe the head on screen and every stop leads to a look (`lookAxes`, `pickLook`). An axis with one stop is
  shown disabled with its value; one that was not read is left out. View: Turn, Tilt and Zoom (also drag, wheel and
  pinch). Light: direction, height and brightness. Switches "Colour" (off shows the bare shape) and "Spin slowly";
  "Reset view"; "Generate again" (the options, starting from the model's own) and "Delete this model" (confirmed).
- `Head3dViewer` holds all the three.js code and is a lazy chunk of its own, loaded only when a model exists (three
  0.187, MIT; 657 kB, 166 kB gzipped; the tab itself is 15 kB). Its `HeadStage` draws only when something changed,
  fetches a look's mesh when it is first wanted and keeps the one on screen and four more, caps the pixel ratio at
  2, takes its background from the theme, and lets go of everything on unmount, the WebGL context included.
  (`useTheme` hands React the choice and the theme in effect as one value, so a system switch under "System"
  re-renders its users: with the choice alone it did not, and the 3D background kept the old colour.) From one look
  to the next it cross-fades two whole pictures over 300 ms (the old look is drawn and copied to a plain canvas on
  top, the new one is drawn underneath, the copy fades out), at once under `prefers-reduced-motion`. The view
  belongs to the viewer, because the owner also drags it: it reports it (`onView`, at most every 60 ms) and takes
  changes as commands (`Head3dHandle.setView`), so a slider never fights a drag. The lamp hangs on the camera and
  keeps its place for the viewer while the head turns. Without WebGL the tab says so and shows the look's photo, and
  the look sliders still work. The wrapper carries `data-ready`, `data-look` and `data-fade` for checks.
- `components/ui.tsx` gains `Slider` (the native range input, styled once in `index.css`: a 44 px tall touch
  target, a thin track coloured up to the thumb through `--fill`, tick marks, the value in words as
  `aria-valuetext`) and `Switch` (moved from `LearningTab`).

## 15. Configuration (env)

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `<repo>/data` | database and files |
| `MODELS_DIR` | `<repo>/models` | all models |
| `PORT`, `HOST` | `41000`, `127.0.0.1` | listen address |
| `SITE_TITLE` | `Personal Content Cloud` | |
| `RP_ID`, `ORIGIN`, `RP_NAME` | derived from the request / site title | WebAuthn relying party; `ORIGIN` may be a comma-separated list, the first being the server's own address (section 4) |
| `SESSION_DAYS` | 30 | sliding session length |
| `VAULT_UNLOCK_MIN` | 15 | NSFW unlock window |
| `TRASH_DAYS` | 30 | purge after |
| `LOCAL_TIMEZONE` | `Asia/Kolkata` | naive capture times and "same day" |
| `GPU_ENABLED` | `auto` | `auto`, `true`, `false` |
| `ORT_THREADS` | `min(6, cpus-2)` | CPU threads per session when on CPU |
| `MAX_UPLOAD_MB`, `CHUNK_SIZE_MB`, `MIN_FREE_DISK_MB`, `STAGING_TTL_HOURS` | 16384, 8, 2048, 48 | uploads |
| `THUMB_SIZE`, `PREVIEW_SIZE`, `PREVIEW_SMALL_SIZE`, `PREVIEW_QUALITY`, `PAGE_RENDER_SIZE` | 480, 2048, 1280, 80, 1600 | derivatives |
| `VIDEO_*` | as the hub | previews, sampling, limits |
| `FACE_*`, `CLUSTER_*`, `SUGGEST_THRESHOLD` | as the hub | faces |
| `BODY_DET_THRESHOLD` | 0.5 | YOLOX person score |
| `NSFW_THRESHOLD` | 0.7 | |
| `DOC_LIKENESS_MIN` | 0.55 | triggers OCR for images |
| `BLUR_VAR_MIN`, `BURST_WINDOW_S` | 40, 3 | mis-clicks |
| `GUESS_WINDOW_H`, `GUESS_SCENE_SIM`, `GUESS_CLOTHES_SIM`, `GUESS_GPS_M` | 1, 0.80, 0.85, 250 | person guessing: the hours before and after a photo whose tagged people are references |
| `FORWARDS_ENABLED`, `FORWARD_MAX_SIDE`, `FORWARD_JPEG_QUALITY_MAX`, `FORWARD_MEME_MIN` | true, 1600, 82, 0.5 | forwards (section 29) |
| `CHALLAN_PORTAL_URL` | the Maharashtra e-challan portal | the portal a challan is paid at when its issuer, number or location says Maharashtra (section 30) |
| `SEARCH_*` | as the hub | semantic search gates |
| `OCR_ENABLED`, `OCR_MAX_PAGES` | true, 50 | |
| `WALLET_PASS_TYPE_ID`, `WALLET_TEAM_ID`, `WALLET_ORG_NAME`, `WALLET_DIR`, `WALLET_KEY_PASSWORD` | unset, unset, site title, `DATA_DIR/wallet`, unset | Apple Wallet (the key password only when key.pem is encrypted) |
| `WORKERS_ENABLED` | true (false in tests) | |
| `ORIGINALS_DIR` | unset (= `DATA_DIR`) | the folder (another disk) for `originals/`, `staging/` and `failed/` (section 31) |
| `BACKUP_DIR`, `BACKUP_HOUR`, `BACKUP_KEEP` | unset (backups off), 3, 14 | nightly snapshots: where, the local hour, how many are kept (section 31) |
| `VIDEO_ENCODER`, `VAAPI_DEVICE` | `auto`, `/dev/dri/renderD128` | web copy encoder: `auto`, `h264_nvenc`, `h264_vaapi`, `h264_qsv`, `libx264` (section 27) |
| `VIDEO_PREVIEW_MAX_HEIGHT`, `VIDEO_PREVIEW_MAXRATE_KBPS`, `VIDEO_PREVIEW_CRF`, `VIDEO_STREAM_MAX_KBPS` | 1080, 6000, 23, 8000 | web copy rendition and the bitrate up to which H.264 originals stream as they are |
| `GEO_ENABLED` | true | download and use the GeoNames gazetteer for place names (section 25) |
| `FORWARDS_ENABLED`, `FORWARD_MAX_SIDE`, `FORWARD_JPEG_QUALITY_MAX`, `FORWARD_MEME_MIN` | true, 1600, 82, 0.5 | media received through messaging apps (section 29) |
| `MAP_TILES`, `MAP_TILE_STYLE` | `offline`, OpenFreeMap liberty | default basemap and the online style URL (section 25) |
| `TRAITS_ENABLED`, `TRAIT_MATCHING` | true, true | read tattoos, mehndi and nails from people boxes; use them in guesses (off: guesses exactly as before) (section 34) |
| `TRAIT_ONLY_GUESSES`, `TRAIT_ONLY_MEHNDI` | true, false | a guess from a tattoo alone under the admission rule; the same from mehndi alone |
| `TRAITS_MEHNDI` | false | read mehndi at all (off: SigLIP read plain hands as mehndi on the owner's library; section 34) |
| `TRAITS_THREADS` | 2 | CPU threads per traits model session (pose, hands, dino) |
| `PLACE_ENABLED`, `PLACE_ONLY_GUESSES` | true, true | line up photos taken together and use where people stood in guesses (off: guesses exactly as before); a guess from the position alone under the admission rule (section 35) |
| `PLACE_THREADS` | 1 | OpenCV threads while a photo is lined up: 1 keeps the work on the runner's nice 19 thread, more is faster but runs on OpenCV's pool at normal priority; process-wide while it measures (section 35) |
| `TRAIT_MIN_BODY_PX`, `TRAIT_MIN_LIMB_PX`, `NAIL_MIN_HAND_PX` | 200, 64, 96 | smallest box long side and limb segment (pixels of the 2048 px preview), smallest hand span for nails (pixels of the source used) |
| `TATTOO_GATE_P` | 0.80 | SigLIP tattoo cut (0.60 up to it is "unclear"; measured on the owner's library, section 34) |
| `TATTOO_MIN_INLIERS`, `MEHNDI_MIN_INLIERS` | 12, 30 | SIFT inliers a verified tattoo or mehndi pair needs |
| `TRAIT_MARGIN_LLR`, `TRAIT_PRIOR_PEOPLE`, `TRAIT_MIN_P` | 2.0, 20, 0.90 | lead over the runner-up (log likelihood ratio) for a trait-only guess; prior odds 1:N for its stored score; floor of the owner-outcome tightening |
| `NAILS_MAX_DAYS`, `MEHNDI_MAX_DAYS` | 14, 21 | nails and mehndi are compared within this many days |
| `TRAIT_PROPAGATE_MAX` | 400 | photos one tagging queues for a trait re-guess |
| `GEN3D_DIR`, `GEN3D_PYTHON` | `MODELS_DIR/gen3d`, `GEN3D_DIR/venv/bin/python` | the 3D head generator's own environment (`venv/`, `weights/`, `src/`) and its Python (section 36) |
| `GEN3D_GENERATOR` | `auto` | `auto` (the first installed backend), a backend's name, or `fake` (the stand-in: it needs no model and runs with the server's own Python) |
| `GEN3D_SANDBOX` | `auto` | run the generator inside `bwrap`, where it sees only the system's libraries, its own environment and the job's work folder (section 36): `auto` (the real generator, when `bwrap` is installed), `on` (always, and nothing is generated without it), `off` |
| `GEN3D_MAX_RSS_MB`, `GEN3D_LOOK_TIMEOUT_S` | 5000, 600 | the generator is stopped above this much resident memory (its whole process tree), or when one head takes longer than this |
| `GEN3D_FAKE` | false | use the stand-in generator with the server's own Python (tests, or a look at the tab before the model is installed) |

## 16. Testing

pytest with a fresh `DATA_DIR` per test, `WORKERS_ENABLED=false`, `worker.drain()` to run jobs,
`TestClient` with a logged-in session fixture (`user_client`, created through `/api/auth/setup` and
`/api/auth/login` with the CSRF header). Model-backed tests skip when the model files are missing
(`ml.models.present(key)`); pure modules (parsers, quality metrics on synthetic images, dedup, cursors,
auth, uploads, FTS) run everywhere. Fixtures: the hub's public-domain face photos, synthetic renders of
each document type (text drawn with Pillow, valid check digits), a two-page PDF with a text layer and a
scanned-style PDF, generated blurry and dark images, a tiny ffmpeg-generated video. The frontend
builds with `tsc` and is checked in a real browser with `ws-check`.

Body traits (section 34) run without models: `tests/helpers.py` adds `make_body`, `make_scan`, `make_trait` (a row
with a synthetic `detail` crop and its scan row), `set_traits_ready` and `adjusted_params`, and `tattoo_crop(seed,
angle, scale, mirror, small, variant)` draws a synthetic glyph on skin at 448 px; SIFT runs for real. The fake
SigLIP engine asserts that it receives PIL RGB images. Files: `test_trait_sift.py` (verification, caches,
symmetry), `test_traits_extract.py` (the runner, statuses, ownership, the gate, nails, the fingerprint, undo),
`test_trait_match.py` (tiers, admission, shared designs, absence, mehndi, nails, near-duplicates, hides, budgets),
`test_trait_guess.py` (fusion with clothes guesses, the window exemption, learning, propagation),
`test_learning_traits.py`, `test_traits_api.py`, `test_traits_integration.py`; `test_ml_traits.py` needs the real
models (run alone with `ORT_THREADS=2`). `tests/conftest.py` resets the verification caches for every test
database, since trait ids restart at 1 in each (and the picture features of section 35, cached by item id). `scripts/trait_eval.py` checks the whole path on a snapshot of a
library (never the live one): stage timings, the gate sheets, the per-query verification table, the watched bodies
and a leave-one-out run.

Recognition trained on one person (section 10) runs without models: `tests/test_person_model.py` builds a person, a
look-alike identity at cosine 0.7 and strangers from hand-made vectors and covers the training (examples, conflicts,
also a group of look-alike faces that would vouch for each other, the check and which folding it ran, determinism),
the scan (what may become a guess, what is never touched, a group the owner said is not the person), `guess_item`
recomputing the guesses, the review actions, a face going back into its group on every path that clears its guess
(and an emptied group coming back as one), the automatic matching (the gate and what it costs, a face the model is
sure of kept for the person in photos and videos, the cut, the child band, two models, no model; a face another
named person matches better or that looks like one taken off the person is not kept; the guess job for a face kept
outside the analysis; a kept face no guess can reach, in a photo or clip the person is in, in the vault, as a
second face of one item, and in a recluster), the routes (a
lost run, the keyset cursor, "Leave", whose run is going), and what an undo brings back (the pause, the model of a
deleted or merged person).

Position (section 35) runs without models too; SIFT runs for real on synthetic textured pictures. Files:
`test_place_align.py` (a shifted, a zoomed and a rolled picture, another size and shape, an unrelated one, a plain
wall, the burst rule, a person who moved in front of a still background, the cache and its bound),
`test_place.py` (which photos are pending and with which next photos, the sub-second order, the runner's writes,
pairs that do not line up, a photo arriving in the middle, duplicates and Live pairs, bursts, the re-guess queue,
the switch, deleted photos, failures), `test_place_match.py` (head points, mutual matches, the bounds, the
arrangement check, clothes floors, the child band, the tiers, chains and hops, people present) and
`test_place_guess.py` (the four fusion rules, guesses from the position alone and their switch, the last tier and
a burst, people present and rejected, the vault, one person per photo, the face veto, the child band, a chain
across two photos, outcomes and the people learner, the window change, changed rules, the Review list with the
reference crop, the learning status). The guess tests of sections 10 and 34 have no links, so they also show that
nothing changes without them. Each defect a review found has its test in these files (a trait veto that position
must not undo, three-photo chains, a chain through a photo that says no, a child through a faceless box, rules
switched off and on again, photos a chain can reach, the trash, a document, a shared cache, a walker in front of a
plain wall, the original when there is no preview, failures in a row, the thread count), and `test_place_eval.py`
checks that the script refuses a live data folder from a worktree too. `scripts/place_eval.py` checks the whole
path on a snapshot of a library (never the live one): the lining up with its timings, the kinds of link by gap,
and a leave-one-out by mode (position, clothes, both), kind of box, gap and photos walked, also with the person
taken off the photos around (`--around`).

3D heads (section 36) are tested without PyTorch. `test_gen3d.py`: the mesh clean-up, the GLB writer read back
(linear 16-bit colours that return to the same sRGB bytes), the stand-in backend, a backend module that offers
`create()`, `python -m gen3d` end to end with its exit codes and a clean stop on SIGTERM, the check with and
without `--device` (a stand-in `torch` on the path shows that the plain check never imports it), and the viewer
check below. `test_looks.py`: the square around a face, another face inside it, the head's turn from five points,
the faults, ages from the birth year, a day sharing its levels, the bins and the choice of looks, the sample over
the years and its limit (also for someone who is a child in every photo), the reading's reports before the models
load and after every photo, and the plan and the picture on hand-made rows with no models. `test_model3d.py`: the
job end to end with the stand-in generator; one job at a time; Stop and Delete in the child's step and while looks
are found; a person deleted during either; a child that fails, that cannot start, that passes its memory limit or
its time limit; heads that were finished before a limit or a crash; a look that fails among others; looks whose
photo left the library or whose face is no longer the person's; the sweep, also of what an interrupted run left;
the check asked once, renewed in the background and given a folder of its own; the child's environment; the
routes and the files route. Three tests need `bwrap` and skip without it. One runs the stand-in job through the
sandbox and then asks from inside it, with the app's own Python: the work folder can be written and nothing else
can, a file and a listening unix socket outside it are not there, `/run` is empty, a TCP connection fails, a
variable of the server's environment is absent, and of the home folder and the repository only the way to what
is shown exists.

`frontend/scripts/check-head3d.mjs` checks the viewer without a browser. It bundles the real `HeadStage` and the
look sliders' logic with esbuild, gives them three.js with a stand-in renderer and a stand-in page, and lets
GLTFLoader read files that the Python writer has just written: the first look and its colours, the cross-fade (two
draws and one copy a frame), a change of mind during a fade, reduced motion, the camera for Turn, Tilt and Zoom,
the directions of drag and wheel, the spin, the lamp, a file of any size and place, a file that cannot be read,
which meshes are kept, a new model with the same keys, a lost context, the theme's value under a system switch,
dispose and a browser without WebGL.
`test_gen3d.py` runs it and skips when node or the frontend's packages are missing. What the pictures look like is
not something it can tell: that takes a browser.

## 17. Build notes

The areas below were built in parallel on 2026-11-07; each area's decisions and deviations from this
contract are recorded in `docs/integration-notes/<area>.md`. Where a note and this file disagree, the
note describes the code as built. Known deviations folded in: `probe(path, filename)` takes the original
file name; MediaInfo has `page_count`, `raw`, `reason`; the people router carries its own `/api` prefix and
serves `/api/review/guesses`; `pdfium` calls hold `app.media.documents.PDFIUM_LOCK`; worker job payloads are
`{"item_id", "version"}` and `{"person_id", "item_id"}` in that key order; `SearchResult.semantic_used`;
items whose text stage finds a typed document move to `documents` unless the owner chose a category.

Review round (2026-11-07): an adversarial review of the first build (75 findings, 72 confirmed by two
independent verifiers) was fixed area by area with regression tests in `tests/test_*_regressions.py`
(backend) and the frontend follow-ups; the behaviour changes are folded into the sections above.
`extract_item` sets `text_status='error'` for problems with the file itself, while database, memory and
disk-full errors propagate so the worker retries or defers the text job. Frontend: every `Modal` traps Tab
and Shift+Tab; Settings > Passkeys asks for the vault unlock in a dialog when adding or removing a passkey
answers `vault_locked`, then retries once (`components/vault/UnlockForm.tsx` is shared with the vault page);
Settings > System has a Retry failed uploads button; the upload queue links vault items with `?vault=1`.

Scale round (2026-11-08): sections 23 to 28 were built in parallel by one implementer per feature on a
shared scaffold (tables, settings, router stubs, routes, slots), each followed by an adversarial review and
a fix pass; the integrator kept the shared files. Deviations folded into the sections: the timeline loads a
month page by page and the viewer position is the library index; album member counts are planned from
the membership side; share links skip mis-clicks in album shares and items of kind other; a hardware
encoder is retired only on device errors or two failures in a row, and requeue scope all leaves done
copies alone; the map falls back to libx264 on this server (no NVENC library, no VA display). The iOS
background uploader was written against the contract but never compiled (no Mac).

### Ownership for parallel implementation

Each area owns its files; shared contracts (`schema.sql`, `config.py`, `jobs.py`, `urls.py`,
`schemas.py`, `lib/types.ts`, `lib/api.ts`) are written first and changed only by the integrator.

| Area | Owns |
|---|---|
| core | `main.py`, `db.py`, `storage.py`, `util.py`, `worker.py`, `ingest.py`, `media/*`, `api/uploads.py`, `api/files.py`, `api/common.py` |
| auth | `auth/*`, `api/auth.py`, tests `test_auth*.py` |
| ml | `ml/*`, `scripts/download_models.py` |
| analysis | `analysis/*`, `api/people.py`, `api/library.py` (review, guesses), tests |
| documents | `documents/*`, `wallet/*`, `api/documents.py`, tests |
| search | `search/*`, `api/search.py`, FTS helpers |
| frontend-shell | app shell, auth pages, settings, upload, PWA, theme |
| frontend-library | library, viewer, people, documents, review, vault pages |

## 18. Learning from your choices (`app/learning`)

Every correction the owner makes is a training example for the next automatic decision of the same kind.
Nothing is sent anywhere and no deep model is retrained: small models are fitted on top of the frozen
embeddings and scores the pipeline already produces, from the choices the database records. Learned
models live in `DATA_DIR/learned/*.json` (small, rebuilt from the database at any time) and are read by
the pipeline with an mtime cache. `LEARNING_ENABLED` (env, default true) and the kv flag
`learning_enabled` (Settings switch, default on) turn it off: the learners then answer "nothing learned"
and training is skipped. Resetting deletes the learned files only, never the choices themselves.

Faces count by their face weight (section 10) wherever a learner reads them: a blurred, masked or turned-away
face moves a person's threshold little or not at all.

Signals (read from existing tables; `learning_events` adds what is otherwise lost):
- category: `items.category_source = 'user'` -> (SigLIP whole-image embedding, category);
- vault: `items.nsfw_source = 'user'` -> (embedding, nsfw) and the classifier's `nsfw_score`;
- mis-clicks: `misclick_source = 'user' AND misclick = 0` with the reasons still stored = kept (negative),
  `misclick = 1` with `misclick_source = 'user'`, or trashed while flagged automatically = confirmed
  (positive); features are `quality_scores`;
- document type: `documents.source = 'user'` -> (document text, doc_type); a field edit counts as a
  confirmation of the type;
- people: `faces.assigned_by = 'user'` (and bodies) per person, `guess_rejections`, and the confirm and
  reject outcomes of guesses with their evidence (`learning_events`, kind `guess`).

`learning_events(id, kind, label, features JSON, item_id, person_id, created_at)` is written by the API and
the guess code through `learning.events.record(conn, kind, label, features, item_id=None, person_id=None)`:
kind `guess` (label 1 confirm / 0 reject, features = the guess evidence), `category`, `vault`, `misclick`
and `doc_type` (label None, features `{"from": .., "to": ..}` plus the scores at the time), and `merge` (label 1,
features `{accepted, sources, followed}`, `person_id` the person merged into), and `forward` (label 1 set apart / 0
kept, features = the forward signals, section 29), and `species` (label None, features `{from, to, score, followed}`,
`item_id`, `person_id` the pet the crop was in; section 22). Events also feed
the status page ("learned from N corrections").

Learners (one module each, all pure numpy, all with a `status()` and a `train(conn)` that writes the model):
- `doc_type.py`: TF-IDF exemplars over `fields.word_tokens(normalize_text(text))` (unigrams and bigrams,
  Devanagari-aware); `predict(text) -> (doc_type, similarity) | None` returns the nearest user-labelled
  exemplar when cosine >= 0.35. `extract.analyze_text` uses it: the learned type wins when similarity
  >= 0.5, or >= 0.35 when the rules say `other`; the detection score becomes max(rule score, similarity).
- `images.py`: nearest-neighbour over SigLIP embeddings of user-labelled items. `predict_category(emb) ->
  (category, score) | None` (weighted vote of neighbours with cosine >= 0.80; needs the best neighbour >=
  0.85 and the winner holding >= 0.6 of the vote) overrides the rule category in `categories.decide` except
  `documents` by kind and the vault; `predict_nsfw(emb, nsfw_score) -> bool | None` answers from a
  user-labelled neighbour at cosine >= 0.85, else from a calibrated threshold (the cut on `nsfw_score`
  that best separates the user's vault labels, searched in [0.5, 0.95], used only with at least 8 labels
  and at least 3 on each side), else None (the rules decide). `forward_vote(emb) -> bool | None` (section 29)
  answers from the owner's kept or set-apart look-alikes with the same neighbour rule.
- `misclick.py`: per reason (`blurry` on blur_var, `dark` on mean_lum, `blown_out` on clipped_frac,
  `pocket` on accident_p) the cut that best separates kept from confirmed mis-clicks, searched within
  [0.25x, 4x] of the default and used only with at least 5 examples on each side; `thresholds() ->
  {"blur_var_min": .., "dark_mean": .., "clipped_frac": .., "accident_min": ..}` with the defaults filled
  in, read by `quality.decide`.
- `images.py` also keeps the crops whose species the owner set (`animals.species_source = 'user'`) as the
  neighbours of `species_vote(emb) -> str | None` (section 22).
- `people.py`: `person_threshold(person_id) -> float | None`: a per-person face threshold from the
  similarities of the person's user-confirmed faces to the person (loosened to their 10th percentile
  minus 0.02, floor 0.30), from the merges the owner made into the person (loosened to the least
  similarity accepted, `learning_events` kind `merge`, minus 0.02, floor 0.30) and raised above the
  closest rejected face (+0.02; the reference is the confirmed faces, else the person's 200 best faces;
  a rejection wins over any loosening). A merge event whose `accepted` is under `cluster.PULL_FLOOR` (0.30, also
  those recorded before merges checked it) teaches nothing, and a cross-age link records no adult value; once the
  library's ages are read, faces in the child band are left out of the confirmed, rejected and reference faces (a
  baby photo says nothing about the adult, section 10). `people.RULES_REV` and kv `face_ages_ready` are part of
  `train.signature`, so a new rule or the ages becoming ready retrains at the next housekeeping. `cluster._score` uses the learned threshold in place of the
  good-face default (a weaker face keeps its extra margin on top), so a person the owner confirmed or
  merged often is matched more readily and one they corrected is matched more carefully. Every face counts by
  its face weight (section 10): the centroid and the 10th percentile are weighted, `MIN_USER_FACES` counts the
  total weight, faces of weight 0 are left out, and a rejected face raises the threshold by its weight's share
  of the way to its similarity + 0.02. `guess_probability(features) -> float | None`: a logistic
  regression on [clothes, scene_term, face_term, same_day, has_gps] fitted on `guess` events (needs 30
  events with both outcomes); `guess.guess_score` returns it instead of the fixed 0.5/0.3/0.2 mix. Outcomes of
  guesses made from body traits alone (features `basis: 'trait'`, section 34) are left out of this fit and of its
  event count, as their terms are not the clothes terms; `clothes+trait` outcomes keep the clothes features and add
  `basis` and `trait_llr`. Outcomes of guesses with basis `face` (below) and of guesses made from the position alone
  (`basis: 'place'`, section 35) are left out the same way (`NOT_CLOTHES`), and `clothes+place` outcomes keep the
  clothes features and add `basis` and `place_llr`. `RULES_REV` is 4 (3, 2026-11-09: the trait outcomes left out; 4,
  2026-11-13: the trained-face and the position outcomes too).
- `person_model.py` (section 10, "Recognition trained on one person"): one model per person the owner asked for,
  `learned/person_<id>.json`: a logistic regression on the face vectors stacked with the top-3 score and the face
  weight, checked on days (or, when the days do not spread the faces, photos) it was not trained on. Not in
  `LEARNERS` and not in `signature`: only the owner trains it (`POST /api/people/{id}/recognition/train`). It obeys
  the learning switch and a reset removes it. Its file goes through `undo.discard` when the person is deleted or
  merged, so the undo of that action brings it back (section 33). Besides making guesses it has two says in the
  automatic matching of new faces (`cluster._score`): under `GATE_P` 0.2 the person is no option for the face,
  and at `CLAIM_P` 0.8 or more nobody else is, provided the guess path would offer the face to them (section 10):
  so a face the person's learned cut (above) keeps off them stays loose for a guess instead of going to a
  look-alike with a lower cut. The person's learned cut itself is not touched by the model.
- `traits.py` (section 34): `params() -> TraitParams` (the tattoo and mehndi tiers, `tattoo_min_inliers`,
  `tattoo_single_min`, `single_pair_ok`, `nail_color_de`, `admission_p`, `source`), the numbers
  `analysis/trait_match.py` reads. Tiers are not fitted in this version: the library has one heavily tattooed
  person, and fitting would need 30 or more positives from 3 or more people. `train(conn)` reads only the
  calibration sample (kv `trait_calib`, taken on the traits runner thread, never under the training lock) and the
  owner's answers to trait-only guesses; it verifies nothing. Three adjustments, each only ever stricter: when 3 or
  more pairs of different named people in different photos verify at the floor, `tattoo_min_inliers` becomes one
  above the largest such count (at most 30; pairs inside one photo are reported but do not count, as they are mostly
  one arm inside two boxes); `nail_color_de` becomes `clamp(min(q90 of same-person dE, q10 of hard-negative dE), 6,
  10)` with 20 or more of each; and with 20 or more answered trait-only guesses, precision under 0.95 raises
  `tattoo_single_min` to 30 and under 0.85 turns single-pair admission off. When the params change, the photos of the
  open `trait` and `clothes+trait` guesses are queued for a re-guess. `status()` reports the outcomes, the hard
  negatives and the adjustments; Settings > Learning adds the reading progress (`traits.progress`).
- `not_faces.py` (section 10, "Not a face"): the faces the owner removed as not a face (`not_faces`) against
  the faces the owner assigned, clear or not (so the helmets and masks the owner tags keep being found); a
  logistic regression on [detector score, clarity / 30, log1p(sharpness) / 5, log(size) / 6, similarity to
  the nearest marking (the marking itself left out)], fitted once 10 measured markings and 10 measured owner
  faces exist; detection uses it at `P(not a face) >= 0.9`, never on a clear face. `status()` gives the number
  of markings and whether a rule exists.
- `forwards.py` (section 29): `probability(signals) -> float | None`, a logistic regression over the forward
  signals fitted on the owner's choices (`forward_source='user'` rows and `forward` events), used once 10
  examples with both labels exist; `forwards.decide` takes it in place of its rule for the second step.
- `train.py`: `train_all(conn) -> dict` (every learner, writes `learned/status.json` with `trained_at`,
  per-learner example counts and what was learned), `maybe_train(conn)` (trains when a choice is newer
  than `trained_at`; called by worker housekeeping every 10 minutes), `pending(conn) -> int` (choices
  newer than the last training), `reset()`. `LEARNERS` includes `traits`. `signature` also holds kv
  `traits_ready`, kv `trait_calib_at` (it changes only when a calibration sample is taken, which happens only once
  the library is read, so the backfill itself never retrains) and `learning.traits.RULES_REV` (2: hard negatives of
  two named people only), so a change to the traits learner's rules recomputes its stored adjustments.

API (`/api/system/learning`, session required, CSRF on writes): `GET` ->
`LearningStatus{enabled, trained_at, pending, learners{doc_type{examples, labels[]}, category{examples,
labels[]}, vault{examples, threshold}, misclick{examples, thresholds{}}, people{persons, events,
weights{} | null}, place{scan{scanned, eligible, ready, rev, last_error, enabled}}}, events{category, vault,
misclick, doc_type, guess}}` (`place` is the progress of section 35: nothing is learned there); `PATCH {enabled}` (kv flag);
`POST train` -> `LearningStatus` (trains now); `POST apply` -> `{"items": n, "documents": m}` (queues a
fresh analysis for live items whose category, vault or mis-click decision was automatic and a fresh text
stage for documents typed automatically, so the learned models reach existing items); `DELETE` (reset:
learned files removed, status cleared).

Frontend: Settings gets a Learning tab (`components/shell/settings/LearningTab.tsx`): the switch, what each
learner has learned in plain words with its counts, the time of the last training, buttons Train now,
Apply to existing items (confirm; shows the counts queued) and Reset (confirm), and a short note that
corrections are made on items, documents and people as before.

Ownership for the build: `learning/*`, `schema.sql` (the `learning_events` table), `schemas.py` and
`api/system.py` (the learning routes), the worker hook and `docs/integration-notes/learning.md` belong to
the learning builder; the call sites in `analysis/*`, `documents/extract.py`, `api/library.py`,
`api/documents.py` and `api/people.py` belong to the integration builder; `lib/types.ts`, `lib/api.ts`,
the tab and `SettingsPage.tsx` belong to the frontend builder. The stub package in `app/learning` with
the exact signatures exists before the build starts.

## 19. Tags the owner adds, and that the system learns (`app/api/tags.py`, `app/learning/tags.py`)

Free-form tags ("wedding", "Goa trip", "claim") on any item. The owner adds them; the tag learner then
suggests them on look-alike photos and videos (SigLIP embeddings) and on similar documents (text), and a
removed suggestion is a counter-example. Person tags (`item_tags`) stay what they are.

Schema: `tags(id, name 1..40 chars as typed, name_key UNIQUE = util.name_key(name), created_at)`;
`item_user_tags(item_id, tag_id, source user|auto, score REAL for auto rows, created_at, PRIMARY KEY
(item_id, tag_id))` with an index on `(tag_id, item_id)`; `tag_rejections(item_id, tag_id, created_at,
PRIMARY KEY (item_id, tag_id))` for suggestions the owner removed. Both item tables cascade on item
and tag deletion.

API (`/api`, session required, CSRF on writes, vault items only when the vault is open):
- `GET tags?q=` -> `[TagOut{id, name, items, auto_items}]` (counts over live items the session may see;
  sorted by items desc, then name); `POST tags {name}` -> `TagOut` (an existing key returns that tag);
  `PATCH tags/{id} {name}` (a rename onto an existing key merges into it; user rows win over auto rows);
  `DELETE tags/{id}`.
- `POST items/{uid}/user-tags {tag_id | name}` -> `[ItemTagOut{id, name, source, score}]` (creates the tag
  when `name` is new; an auto row becomes a user row, which also counts as a confirmation; any rejection
  of that pair is cleared); `DELETE items/{uid}/user-tags/{tag_id}` -> the list (removing an auto row
  writes `tag_rejections` and a `learning_events` row kind `tag`, label 0; removing a user row writes
  nothing else).
- `POST items/bulk {action: tag | untag, value: name}` on the selected items.
- `GET items?tag=<name>` and `GET search?tag=<name>` (repeated, every tag required; `common.tags_filter_sql`).
- `ItemDetail.user_tags: [ItemTagOut]` (user rows first, then auto rows by score). `ItemCard` is unchanged.
- Tag names go into the `labels` column of `items_fts` (`refresh_fts`), so search finds them.

Learner (`learning/tags.py`, trained with the others by `learning.train`): per tag, positives are the items
with a user row, negatives the rejections; examples are SigLIP whole-image embeddings (frame 0) for images
and videos and TF-IDF vectors of `item_text` for pdf and office documents (the same tokenisation as the
document type learner, kept in `learning/textvec.py`). A tag is learned once it has 3 positive examples
of a kind. `predict(emb, text, kind) -> [(tag_id, score)]` answers for each learned tag when the best
positive cosine is >= 0.80 (images) or >= 0.45 (text) and beats the best negative cosine by 0.02; the
score is that cosine. The pipeline writes auto rows after analysis (images and videos) and after the text
stage (documents), skipping tags already on the item and rejected pairs. `apply_all(conn) -> int` predicts
for every live item from stored embeddings and text (no model runs) and is part of
`POST /api/system/learning/apply` (`{"items", "documents", "tags"}`). `LearningStatus.learners.tags =
{tags, examples, rejections}`; `events.tag` counts the removals.

Frontend: the viewer's info panel gets a Tags section (`components/viewer/TagsPanel.tsx`): user tags as
solid chips with a remove button, suggested tags as dashed chips with an accept (adds as user) and a remove
button, an input with existing tags offered as you type and Enter to add a new one. The library filter
bar offers the tags with their counts (URL `tag=`), the selection bar has Add tag and Remove tag, and the
Learning tab shows the tags card.

## 20. Live Photos and motion photos (`app/media/live.py`, ingest pairing)

An iPhone Live Photo arrives as two files, the still (HEIC or JPEG) and a short MOV; Android motion
photos (Google, Samsung) are one JPEG with an MP4 appended. Both become one item that shows the still and
can play its clip. Every uploaded byte is kept: the MOV stays a stored item of its own (a hidden
companion) and the JPEG keeps its embedded clip.

Schema (`items`): `content_id TEXT` (Apple's ContentIdentifier read from the still's MakerNote and from
the MOV's `com.apple.quicktime.content.identifier` tag; indexed), `companion_of INTEGER REFERENCES
items(id)` (set on the MOV: the still it belongs to; a companion is hidden everywhere), `live_video_item_id
INTEGER REFERENCES items(id)` (set on the still: its MOV item), `live INTEGER NOT NULL DEFAULT 0` (the
still has a clip: a paired MOV or an embedded motion clip). `uploads.pair_key TEXT` (optional, from the
client: the iOS app sends the asset's local identifier with both files so they pair without metadata).

Pictures: the companion keeps no thumb or previews. Ingest writes them like any video's before the
pairing commits, then `_drop_companion_pictures` deletes them (whichever half arrived second); reprocess
never remakes them; the file routes serve the still's pictures for a companion's uid (section 3). Its web
copy stays: it is the clip that plays.

Pairing (ingest, after the item row of the second file exists, in the same transaction): a still and a
MOV pair when they share a `content_id`, or a `pair_key` on their uploads, or the same file stem
(`IMG_1234`) with the MOV under 4 seconds and both uploaded within one hour of each other. Whichever
arrives second finds the other (`live.find_partner`). The MOV gets `companion_of`, the still gets
`live_video_item_id` and `live = 1`. A companion gets no analysis of its own (`analysis_status` done, no
embeddings, no faces), no FTS row, no dedup group, and `visible_sql` hides it (`companion_of IS NULL`), so
it never shows in listings, search, review, duplicates or stats; its original, thumb and web preview are
still produced (the viewer plays the preview). Trashing or purging the still trashes or purges the
companion with it; restoring restores both. Unpairing is not offered.

Motion photos (ingest of a JPEG): `live.extract_motion_clip(path) -> bytes | None` finds the embedded
MP4 through the XMP `GCamera:MicroVideoOffset` / `Container:Directory` (`Item:Semantic="MotionPhoto"`,
`Item:Length`) tags or the Samsung `MotionPhoto_Data` trailer, and the clip is written to
`videos/<shard>/<uid>.v<version>.live.mp4` (`storage.live_clip_path`, a derivative, rebuilt by a preview
reprocess); the still gets `live = 1`. The clip is served as is (these are H.264 MP4 files).

API: `ItemCard.live: bool`; `ItemDetail.live_video: {uid | null, url, duration_s} | null` (the companion's
uid and web preview URL, or the extracted clip's URL `/files/live/{uid}?v=`) and `urls.live`; the
download menu offers the still and the clip (`/files/original/{companion uid}?download=1` for a paired
MOV, `/files/live/{uid}?download=1` for an extracted clip). `/files/live/{uid}` serves the extracted clip
with the usual vault and cache rules. `GET /items?live=1` lists the live items.

Frontend: tiles show a LIVE badge; the viewer shows the badge as a button that plays the clip in place
of the still (loops while held or until tapped again, muted), and a Live Photo opened from the review or
people pages behaves the same; the download menu lists both halves.

## 21. Devices and the iOS app (`ios/`, device tokens, sync check)

Device tokens: a device session is a `sessions` row with `device_name` set, created by `POST /api/auth/devices
{name}` (needs the vault open, like enrolling a passkey; the raw token is returned once), listed by
`GET /api/auth/devices` and ended by `DELETE /api/auth/devices/{id}`. The token goes in
`Authorization: Bearer <token>`; `sessions.authenticate` accepts it in place of the cookie, the session
slides by `DEVICE_DAYS` (365), the vault starts locked, and the CSRF header is not required for bearer
requests (no cookie, so no browser can send it on the owner's behalf). `SessionOut.device_name` tells
devices and browsers apart in Settings > Sessions > Devices.

Sync check: `POST /api/uploads/check {sha256: [...], asset_ids: [...]}` -> `{known: {sha256: uid}, assets:
{asset_id: uid}}`, where `asset_ids` are the `pair_key` values the device sent with its uploads (the iOS
app uses the asset's local identifier for both halves of a Live Photo, section 20). Trashed items count
as known, so a device never re-sends what the owner deleted.

The iOS app (`ios/`, SwiftUI, iOS 17, built with XcodeGen from `ios/project.yml` on a Mac): signs in with
a device token, watches the photo library (PhotoKit), and uploads new photos, videos and both halves of
Live Photos through the chunked upload protocol (section 6: create with `sha256` and `pair_key`, PUT
chunks at offsets, complete, poll the status) with `source` `api`; it keeps a local record per asset
(uid, hash, bytes) and asks the sync check before uploading, so a reinstall does not re-send the
library. Sync runs when the app is open, on a library change, and in a BGProcessingTask when iOS allows
it (Wi-Fi only and charging only are options). The "Delete synced" screen lists assets the server
confirms it holds (a sync check right before) and deletes them from the library through PhotoKit, which
shows the system confirmation. The app cannot be built or run on this server; `ios/README.md` has the
steps for a Mac.

## 22. Pets (`app/analysis/animals.py`, `app/analysis/pets.py`, `animals` table)

Individual pets are recognised by their look the way people are recognised by their face: the detector
finds the animal, a crop embedding describes it, crops are grouped into identities the owner names, and
corrections teach each pet its own threshold.

Detection: the YOLOX-S person detector (`app/ml/persons.py`, `detect_classes`) answers for the person class
and the COCO animal classes (`ANIMAL_CLASSES`: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
giraffe; `ANIMAL_DET_THRESHOLD` 0.45) from one run per image. `analysis/animals.py` turns the boxes into
`AnimalRecord`s (at most 6 per species and 12 per item, boxes under 24 px dropped) with a crop padded by 10
percent for SigLIP and a crop for storage. Videos run the detector on the poster and 6 sampled frames and keep
the 12 best boxes with their `frame_time_s`. The pipeline stores each box in `animals(id, item_id, person_id,
assigned_by auto|user, x, y, w, h, det_score, species, frame_time_s, embedding BLOB float32[768] = SigLIP of
the padded crop, created_at)` with the crop at `animals/<id // 1000>/<id>.webp` (`storage.animal_crop_path`),
sets `items.animal_count`, and makes a photo with an animal box covering 5 percent of the frame and no person
a `pets` photo (`categories.decide(has_animal=...)`). A re-analysis keeps user assignments on the
best-overlapping new box of the same species (IoU >= 0.5, `pets.carry_over`). When the detector did not run,
the old boxes stay (like bodies). `PETS_ENABLED=false` turns the animal classes off.

Identities: `persons.kind` is `person` (default) or `pet`, and a pet carries `species` (migrated columns).
`pets.assign_new_animals(conn, item_id)` compares each new box to every pet of the same species through the
centroid of the pet's assigned crops (cosine >= `PET_MATCH_THRESHOLD` 0.82, or the pet's learned threshold
from `learning.people`, which fits pets from their owner-assigned crops with the pet default), assigns the
best, and otherwise pairs the box with an unassigned box of that species from another item at >=
`PET_NEW_THRESHOLD` 0.86 into a new unnamed pet (other close unassigned boxes join it; two boxes of one
item never pair, two dogs in one photo are two dogs). Names: `cluster.find_person_by_name(kind=...)` keeps
people and pets apart, so a dog and a person may both be called Bruno; a renamed pet merges only into a pet
of the same species. `cluster.merge_persons` refuses a pet and a person, or two species. Deleting a pet
unassigns its boxes; pruning removes unnamed pets without boxes.

Routes: `GET /api/people?kind=person|pet|all` (default `all`); `PersonOut.kind`, `species`, `animal_count`
(for a pet `face_count` is 0, `cover_face_id` null and `cover_url` its best crop, user-assigned first);
`PersonRef.kind`; `GET /api/people/{id}/animals` (`AnimalPage` of `AnimalListItem`, best detection first, with
`source`: 'user' for a box the owner drew);
`GET /api/people/{id}/items` and the `person` filter of the library and search include animal boxes;
`ItemDetail.animals: [AnimalOut{id, x, y, w, h, det_score, species, frame_time_s, person, assigned_by, url}]`
and `ItemDetail.people` lists pets with kind `pet`; `POST /api/animals/assign {animal_ids, person_id |
new_person_name}` (one species per call; a person target or a pet of another species is 400; a new name
creates or reuses a pet of the boxes' species; null unassigns; `fix_species` first makes a pet of another species
the boxes' species); `DELETE /api/animals/{id}`; `POST /api/animals/species {animal_ids, species}` and
`POST /api/people/{id}/species {species}` (species corrections; `AssignOut.followed` counts the look-alike crops
that followed);
`/files/animal/{id}` serves the crop with the face-crop rules (403 `vault_locked` for vault items). Pet names
go into the `people` column of the search index. Pets have no guesses and no merge suggestions.

Frontend: the viewer draws animal boxes with a paw mark (name or species, dashed when unassigned) and an
assign popover that lists pets of that species first and never people; the item's people chips show pets
with the paw; the People page requests `kind=person` and shows a Pets section below; the person page of a
pet shows its species and a crops grid (select, move, unassign, remove) instead of faces and guesses; the
library's person filter lists pets too.

Species corrections (2026-11-08). The detector's class is the only source of a crop's species, and it is
wrong at times (a white cat on a lap read as a dog). Three things change that:

- Schema: `animals.species_source TEXT NOT NULL DEFAULT 'auto' CHECK (species_source IN ('auto','user'))` and
  `animals.species_scores TEXT` (JSON `{species: probability}` from the zero-shot check), both through
  `MIGRATED_COLUMNS`.
- The zero-shot check (`animals.species_check(embedding, detected) -> (species, scores, changed)`): SigLIP
  zero-shot over the crop's embedding with one prompt group per species in `SPECIES` ("a photo of a cat",
  "a photo of a dog", ...); when the best species differs from the detector's, its probability is at least
  `SPECIES_SWAP_MIN` (0.6) and the detector's species gets at most 0.25, the crop takes the zero-shot species
  (`species_source` stays `auto`; the scores are stored either way). The check runs in the analyze stage after
  detection and in worker housekeeping as a backfill for crops without `species_scores`
  (`animals.housekeeping(conn, budget_s)`, 500 crops per run from the stored embeddings, never a `user`
  row; a changed crop is unassigned from a pet of another species and offered to the pets of the new species
  through `pets.assign_new_animals`). Without SigLIP the detector's class stands.
- More kinds than the detector has classes (2026-11-09): `SPECIES` (and `schemas.SPECIES_NAMES`, the app's
  `Species`) holds the detector's ten and fifteen more: rabbit, hamster, guinea_pig ("Guinea pig"), turtle, fish,
  goat, pig, donkey, camel, alpaca, llama, deer, monkey, chicken and duck, each with its zero-shot prompts ("a photo
  of a bunny" with the rabbit's, "a photo of a tortoise" with the turtle's). The detector finds such an animal as
  its nearest class (a rabbit as a cat, an alpaca as a sheep) and the zero-shot check names it; one it does not
  find is drawn with "Add a pet box". Every species picker lists all 25, pets first, then farm animals, then wild
  ones. Once per `SPECIES_LABELS_VERSION` (2, kv `species_labels_version`) housekeeping clears the scores of the
  crops the system labelled outside a named pet (not the owner's species, not a box the owner assigned), so the
  backfill checks them against the new kinds; a named pet keeps its species and its crops (the owner changes it
  with Species..., for a rabbit saved as a cat).
- The owner's correction. `POST /api/animals/species {animal_ids, species}` (one call may carry boxes of
  several species): each box takes the species with `species_source='user'`; a box assigned to a pet of
  another species leaves it (an unnamed pet left empty is pruned; a named one stays for the owner) and is
  matched to the pets of the new species like a new box (`pets.assign_new_animals` for that item, an
  unnamed pet is made when a look-alike exists); a learning event kind `species` (label None, features
  `{"from": .., "to": .., "score": det_score}`, item_id, person_id = the pet it was in) is recorded per box.
  Look-alikes follow: every automatic box with the old species whose crop embedding is within
  `SPECIES_FOLLOW_SIM` (0.85) of a corrected crop takes the new species too (source `auto`, a `species` event
  with `"followed": true`); `AssignOut.followed` counts them. `POST /api/people/{id}/species {species}` does
  the same for every crop of a pet and sets `persons.species` (a pet named by the owner keeps its name and
  becomes a pet of the new species; a pet of that name and species already there gets the crops instead,
  through `cluster.merge_persons`). The vote (`learning/images.py::species_vote(emb) -> str | None`, the
  owner-labelled crops as the neighbours, cosine 0.85 or more, 60 percent of the weight) answers before the
  zero-shot check for every new crop, so the owner's choices carry to crops that look like them.
- Carry-over on re-analysis (`pets.carry_over`): a `user` species survives: the old box's species and source
  go to the best-overlapping new box (IoU `>= CARRY_IOU`) whatever the detector said this time.
- Frontend: the animal box's popover in the viewer gains "Not a {species}" with a species picker (the
  `SPECIES` list with labels); the pet page gains the same picker beside the species ("These are cats"); the
  toast says "Changed to cat" and how many look-alike crops followed; `ItemDetail.animals[]` carries
  `species_source`. Settings > Learning counts species corrections among the corrections recorded.
- Tests: the zero-shot swap on a hand-made score table (swap, no swap when the detector is confident, no
  change without the model), the route on boxes of two species (events, the pet left behind, the follower
  within 0.85 and the stranger outside it), the pet-level route (name kept, merge into an existing pet of the
  new species), carry-over of a user species, and the backfill never touching a user row.
- One box per animal (`animals.from_boxes`): the detector answers with several classes for one animal at
  times (a cat read as both `cat` and `dog`); boxes of different species whose IoU is at least
  `CROSS_SPECIES_IOU` (0.6) are one animal, and the better-scored box stays (the zero-shot check then
  settles its species). The backfill applies the same rule to stored boxes (the dropped box's assignment
  moves to the kept one when the kept one has none).
- The face detector fires on animal faces: a face whose box lies inside an animal box (at least
  `FACE_IN_ANIMAL` 0.8 of the face's area) with no person box over it is the animal's face, not a person's
  (`faces.py` / the analyze stage drops it before clustering). The backfill removes such faces from stored
  items (their crops and index entries too, through the same path `DELETE /api/faces/{id}` uses, with a
  rejection for the person they were assigned to so the look-alike rule keeps them out), and the viewer
  stops showing a person's name on a cat.
- The assign popover: moving a box to a pet of another species is refused today ("That pet is a dog, this
  box is a cat"); `POST /api/animals/assign` gains `fix_species: true`, which first makes the target pet
  (and every crop in it) the box's species (`species_source='user'`, events) and then moves the box; the
  popover offers it as "Move and make {pet} a {species}". A pet identity's species comes from its crops:
  when every crop of a pet is corrected, `persons.species` follows.

Boxes the owner adds (2026-11-08). The detector misses an animal at times (a cat filling the frame), and
without a box there is nothing to name:
- `POST /api/items/{uid}/animals {x, y, w, h, species?}` (a normalised box inside [0, 1] covering at least
  0.5 percent of the frame; photos only, a video answers 400): the server cuts the padded crop from the
  original (`animals.padded_crop`), writes the crop file (`storage.animal_crop_path`), embeds it with SigLIP
  when the model is loaded (`pipeline._model("siglip")`, `embed_images`), and decides the species: the
  owner's `species` when given (`species_source='user'`), else the zero-shot answer over `SPECIES_PROMPTS`
  on the embedding, else 400 `species_needed`. The row is stored with `det_score 1.0`, `assigned_by 'auto'`
  and `animals.source = 'user'` (a new migrated column, `CHECK (source IN ('detector','user'))`, default
  `detector`); `items.animal_count` follows; `pets.assign_new_animals(conn, item_id)` matches the box to an
  existing pet of that species. The reply is the item's `AnimalOut` for the new box (with `person` when it
  matched) and the viewer opens the popover on it so the owner names the pet.
- `POST /api/items/{uid}/animals/detect {threshold?}` (default 0.2, floor 0.1; photos only): the detector
  runs once more on the photo with that threshold for the animal classes (`persons.detect_classes` with
  `animals.thresholds()` scaled), boxes that overlap no stored box (IoU under 0.5) are kept, stored like the
  pipeline stores them (`source 'detector'`, species by the detector then the zero-shot check, SigLIP
  embedding), assigned, and returned; 409 `models_unavailable` when the detector is not loaded.
- `pets.carry_over` keeps an owner-drawn box across a re-analysis untouched (the detector never produced
  it), with its assignment; `DELETE /api/animals/{id}` removes one as before.
- Frontend: the viewer's "People and pets" section (and the photo stage) offers "Add a pet box" and "Find
  pets again" whenever a photo has no animal box, and under a "More" control otherwise. "Add a pet box"
  puts the stage in draw mode (hint "Drag over the pet"; Escape cancels): a drag with mouse or touch draws a
  rectangle (at least 24 px), "Whole photo" takes the frame minus a 3 percent margin; on release a species
  picker ("What is it?", Cat first) confirms and posts; the new box appears and its popover opens. "Find pets
  again" calls the detect route and toasts how many boxes were added ("No pet found" when none, "The
  models are not loaded" on 409).
- Tests: the create route with models off (crop file written, species from the body, 400 without one, the
  box matched to a pet of that species when a look-alike pet exists, `animal_count`), the detect route with
  models off (409), a user-drawn box surviving `carry_over`, the box inside the frame and large enough.

Pets in videos (2026-11-08, later). A video's animal is often missed and "Tag someone" knew people only:
- `POST /api/items/{uid}/tags` takes a pet too: `person_id` of a pet, or `new_person_name` with `kind: 'pet'`
  and `species` (400 `species_needed` without one); a pet tag means the pet is in the item, like a person
  tag, and `ItemDetail.people` lists it with `kind 'pet'`. The "Tag someone" form lists people and pets (the
  picker's pet section) and asks for the species when a new pet is named.
- `POST /api/items/{uid}/animals` takes `t` (seconds) for a video: the frame at `t` is extracted
  (`video.extract_frame_image` on the original, else the web copy), the box is cut from it and stored with
  `frame_time_s = t`; a video without `t` answers 400 `frame_time_needed`. "Add a pet box" in a video draws
  over the paused frame and posts the player's current time; the new box becomes a track chip.
- `POST /api/items/{uid}/animals/detect` on a video runs the detector at the lower threshold over the poster
  and up to `REDETECT_VIDEO_FRAMES` (12) frames sampled evenly (`video.frames_at`), keeps per frame the boxes
  over no stored box of that frame (IoU under 0.5), and stores them with their `frame_time_s`.
- Tests: a pet tag with a new name and species, the species-less 400, the video box with `t` (frame cut from a
  generated clip, `frame_time_s` stored), the 400 without `t`, the video detect route with a faked detector
  storing boxes with times and skipping a frame's stored box.

## 23. Timeline (`app/api/timeline.py`, `components/timeline/`)

The library at scale: the browse grid renders only what is on screen, knows the whole library's
shape before any card is loaded, and jumps to any month. Search results, people pages, review and
trash keep `MediaGrid`.

Buckets. A bucket is one calendar month of the sort key (`COALESCE(taken_at, created_at)` for
`sort=taken`, `created_at` for `sort=added`) in `settings.local_timezone`, so the month headers agree
with the `from`/`to` filters and the day logic elsewhere (section 9 of the hub, `local_day_start_utc`).
`GET /api/timeline/buckets` takes every `GET /api/items` filter (`category`, `kind`, `person`, `tag`,
`live`, `from`, `to`, `favorite`, `misclicks`, `dups`, `sort`, `vault`, `album`) and answers
`TimelineBuckets = {buckets: [{key: "2024-03", count, newest, oldest}], total}` newest first, where
`newest` and `oldest` are the sort keys at the bucket's ends. It runs one indexed query for the sort
keys (`SELECT <sort_expr> AS k ... ORDER BY k DESC`, the `items_order` or `items_added` index) and
walks them against month boundaries computed in Python (`util.local_month_start_utc(year, month)`,
the first instant of the local month as a UTC ISO string in the stored format, compared as text,
the same trick as `local_day_start_utc`). No `strftime` grouping: the stored strings are UTC and the
local offset is not constant across years. 100 000 items answer in well under a second.
`GET /api/timeline/bucket?key=YYYY-MM` with the same filters plus `cursor` and `limit` (1..1000,
default 500) returns an `ItemPage` of that month in library order (`_page` with the month's
`[start, end)` added to the where list; `total` is the bucket count on the first page). Both share
the filter builder with `GET /api/items` (`library.item_filters(...) -> (where, params, sort_expr)`),
so a filter added later applies to the grid, the buckets, the map and the shares alike. A `key` that
is not `YYYY-MM` is 422.

Frontend (`components/timeline/TimelineGrid.tsx`, used by `Library.tsx` whenever `filters.q` is
empty). Tiles keep `MediaGrid`'s look: square thumbs, the same column counts per breakpoint (3, 4, 5,
6, 7), the same badges and hover checkbox. The grid measures its width with a `ResizeObserver`,
derives the column count and tile size, and lays the buckets out top to bottom: a month header row
(`MMMM YYYY` and the count, sticky under the app header while its month is on screen) followed by
`ceil(count / columns)` rows. Every bucket's height is known from its count alone, so the page has
its full height at once and the scrollbar is honest. The window scrolls (as everywhere in the app);
the grid renders only the buckets that intersect the viewport plus one viewport above and below, and
within a bucket only the rows in that range, so a month with 5 000 photos costs as much as one screen.
Bucket cards load on demand through `['timeline', 'bucket', query, key]` (react-query, pages of 500
fetched in order until the visible rows are covered); rows not loaded yet draw placeholder squares
of the final size, so nothing jumps when they arrive. `['timeline', 'buckets', query]` is the
skeleton; both keys are in `LIBRARY_KEYS`, so bulk actions and viewer edits refetch them, and the
scroll position stays because bucket heights only change when counts do. The header count comes
from `total`. The empty state is `MediaGrid`'s.

Scrubber. A thin rail on the right edge (always on desktop, shown while scrolling and for two seconds
after on phones) marks the years at their share of the total height; dragging or tapping it scrolls
to the bucket under the pointer and a floating label names the month. Keyboard users get the month
headers in the tab order with `Home` and `End` on the grid.

Selection and viewer. Selection works as before over loaded tiles; each month header has a Select
month toggle for its loaded tiles; Select all in the bulk bar selects every loaded tile. The viewer
opens through `useViewerParam` over the concatenation of the loaded buckets' cards in timeline order;
its `loadMore` loads the next bucket after the last loaded one, so prev and next walk the whole
library given time, and `total` is the buckets' total. Opening `?m=<uid>` for a card in a month that
is not loaded yet scrolls to and loads that month first (the bucket key is derived from the card's
sort key when the card is cached, else from `GET /api/items/{uid}`).

## 24. Albums (`app/api/albums.py`, `albums` and `album_items` tables, `components/albums/`)

Schema: `albums(id, name, name_key, description, cover_item_id REFERENCES items ON DELETE SET NULL,
created_at, updated_at)` with `albums_name(name_key)`; `album_items(album_id REFERENCES albums ON
DELETE CASCADE, item_id REFERENCES items ON DELETE CASCADE, added_at, PRIMARY KEY (album_id,
item_id))` with `album_items_item(item_id)`. `name_key` is `util.name_key` of the name (casefolded,
collapsed spaces); two albums cannot share it (409 `album_exists`). An album lists in time order,
the library's order; there is no manual ordering. Trashing an item hides it from its albums and
restoring brings it back (membership stays); purging removes the row through the cascade.

API (`/api/albums`, session): `GET ?q&sort=updated|name|created&cursor&limit` ->
`AlbumPage{items: [AlbumOut], next_cursor, total}` (`q` matches `name_key` as a substring);
`POST {name, description?, item_uids?}` -> `AlbumOut` (201); `GET {id}`; `PATCH {id} {name?,
description?, cover_item_uid?}` (`null` clears the chosen cover; a cover must be a member); `DELETE
{id}` (the items stay); `POST {id}/items {uids}` -> `{added, skipped}` (skipped = already in, unknown, trashed or a companion;
vault items are added or removed only while the vault is open);
`DELETE {id}/items {uids}` -> `{removed}`. `AlbumOut = {id, name, description, count, vault_count,
cover_url, newest, oldest, created_at, updated_at}`: `count` is the live members on the normal side,
`vault_count` the live members in the vault (a number only; their cards show only in an unlocked
vault), `cover_url` the chosen cover's thumb or the newest normal-side member's, `newest` and
`oldest` the sort keys of the normal-side members. Items in an album: `GET /api/items?album=<id>`
(and the timeline buckets with `album=`), so the album page is the library with one more filter;
an unknown album id is 404. `ItemDetail.albums: [{id, name}]`. Bulk: `POST /api/items/bulk {action:
"album" | "unalbum", value: "<album id>"}` adds or removes the selection (the album's `updated_at`
moves on every change). Vault items may be members; nothing about them shows outside an unlocked
vault, and an album share (section 26) never includes them.

Frontend: `/albums` (`pages/AlbumsPage.tsx`: cards with cover, name, count and date range; sort menu;
New album dialog; empty state) and `/albums/:id` (`pages/AlbumPage.tsx`: `<Library album={id}>` with
the album header above the grid: name, description, count and date range, and a menu with Rename and
description, Set cover (from the current selection or the open item), Share album (section 26) and
Delete album; the bulk bar offers Remove from album here). Everywhere else the bulk bar has Add to
album, which opens `AlbumPicker` (search, recent albums, Create new); the viewer's info panel has an
Albums section with chips linking to each album and the same picker. Navigation gets Albums (desktop
links and the phone More sheet; the phone tabs stay as they are). Query keys `['albums', ...]` and
`['album', id]` are in `LIBRARY_KEYS`.

## 25. Map and places (`app/geo/`, `app/api/map.py`, `components/map/`)

What exists: `items.gps_lat/gps_lon/gps_alt` are read at ingest from EXIF and QuickTime tags
(section 7) and shown in `ItemDetail.gps`; nothing else used them. This section adds an index, place
names, a points API, a backfill for items ingested before GPS was read, and the map page.

Schema: `items_gps ON items(gps_lat, gps_lon) WHERE gps_lat IS NOT NULL AND deleted_at IS NULL`;
columns `items.place TEXT` (the display name, "Panaji, Goa, India"), `place_city`, `place_admin`,
`place_country` (TEXT), `place_source TEXT` (NULL until looked up, then `geonames` or `none` when
nothing is within reach), `gps_checked INTEGER NOT NULL DEFAULT 0` (the original was re-read for GPS
by the backfill). All migrated through `MIGRATED_COLUMNS`.

Gazetteer (`app/geo/gazetteer.py`): GeoNames `cities15000.zip`, `admin1CodesASCII.txt` and
`countryInfo.txt` (CC BY 4.0; the attribution is in the README, the map footer and Settings >
System), fetched by `scripts/download_models.py` into `MODELS_DIR/geo/` next to the models (about
3 MB; `GEO_ENABLED=false` skips the download and every lookup). Loaded lazily into numpy arrays;
`place_for(lat, lon) -> Place | None` is the nearest city by haversine within 100 km, with the
admin-1 name and the country name resolved. Without the files every lookup answers None and
`GET /api/map/status` says so.

Places (`app/geo/places.py`): `tag_item(conn, item_id)` sets the place columns from the gazetteer;
`housekeeping(conn, budget_s=5)` runs from the worker's housekeeping (every 10 minutes, section 7) and
from `POST /api/map/refresh` (which runs it to completion on `worker.run_in_background("geo", ...)`):
first items with GPS and `place_source IS NULL` (batches of 500, then `refresh_fts` for them), then
the GPS backfill, `probe.read_gps(original)` for images and videos with `gps_lat IS NULL AND
gps_checked = 0` (batches of 200, `gps_checked = 1` either way). Place names go into the FTS
`labels` column (`documents/extract.refresh_fts`), so "goa" finds the photos taken there in text
search. Items analysed later keep getting places from housekeeping within ten minutes of ingest.

API (`/api/map`, session, vault through `want_vault`, every `GET /api/items` filter accepted):
`GET points?bbox=w,s,e,n?` -> `MapPoints{points: [[lat, lon, uid, k]], total, with_gps}` where `k`
is 0 image, 1 video, 2 document, 3 other, coordinates have five decimals, `total` counts the
filtered items and `with_gps` those with coordinates; no limit, the response is compact and gzipped
(100 000 points are about 3 MB raw). `GET items?bbox=w,s,e,n&cursor&limit` -> `ItemPage` of the items
inside the box in library order (for the cluster panel). `GET places` -> `[{place, place_city,
place_admin, place_country, count, lat, lon}]` grouped by place, most photos first, at most 500.
`GET status` -> `{gazetteer: bool, with_gps, unchecked, unplaced, running}`; `POST refresh` -> `{started}`.
Every query goes through `visible_sql`, so vault items, companions, mis-clicks and duplicate copies
follow the library's rules.

Frontend (`/map`, `pages/MapPage.tsx`, MapLibre GL JS and supercluster, new dependencies). Basemap:
by default an offline one, Natural Earth 110m land and country boundaries bundled as
`frontend/public/map/basemap.json` (about half a megabyte) drawn as fill and line layers on a plain
background, so the map works with no internet and nothing leaves the server; a Detailed map toggle
on the page switches to online vector tiles (`MAP_TILE_STYLE`, default OpenFreeMap's `liberty`
style, no key) and remembers the choice in localStorage (`pcc.mapTiles`), with a one-line note that
tiles come from that service and photos never do. `GET /api/config` carries `map: {tiles, style_url}`
with the defaults (`MAP_TILES` is `offline` or `online`). Points are clustered client side (radius
60 px, counts on the cluster circles, a thumb on single points when zoomed in); clicking a cluster
zooms into it, clicking a single point or a small cluster opens the side panel (desktop) or bottom
sheet (phone) with those items in a `MediaGrid` (`GET /api/map/items` with the cluster's bounds) and
the viewer. The library's filter bar is reused in a compact form (dates, people, category,
favourites, kind; no search), kept in the URL like the library so a filtered map is a link, and a
Show on map button in the library's filter panel opens `/map` with the same parameters. A Places
list (from `GET /api/map/places`) sits beside the map on desktop and in a sheet on phones; choosing
one flies there. The viewer's info panel shows a Location row (place name, else coordinates) with a
Show on map link that opens `/map?at=<lat>,<lon>&z=14&m=<uid>`. Navigation gets Map. The footer
credits Natural Earth and GeoNames, and OpenFreeMap and OpenStreetMap when online tiles are on.

## 26. Sharing (`app/api/share.py`, `shares` and `share_items` tables, `/s/:token`)

A share link shows a chosen set of photos and videos to someone without an account: an album, or
a hand-picked set. Vault items are never shared (creating a share with one answers 422 `vault_item`;
an album share skips them), trashed items drop out of a share while in the trash, items of kind `other`
(no thumb or preview) are left out at creation and in the listing (a set with nothing left is 422
`no_items`), an album share also leaves mis-clicks out as the album page does, and a link can
carry a password, an expiry and a download switch, and can be revoked.

Schema: `shares(id, token UNIQUE (16 random bytes, base64url, 22 chars), kind 'album'|'items',
album_id REFERENCES albums ON DELETE CASCADE, title, password_hash (argon2id, NULL when open),
expires_at, allow_download INTEGER DEFAULT 1, created_at, revoked_at, view_count, last_viewed_at)`;
`share_items(share_id REFERENCES shares ON DELETE CASCADE, item_id REFERENCES items ON DELETE
CASCADE, position, PRIMARY KEY (share_id, item_id))`.

Owner API (`/api/shares`, session): `GET` -> `[ShareOut]` (newest first, revoked ones last 30 days
included with `revoked_at` set); `POST {kind: "album" | "items", album_id?, item_uids?, title?,
password?, expires_days?, allow_download?}` -> `ShareOut` (201); `PATCH {id} {title?, password?,
expires_days?, allow_download?}` (an empty password clears it, `expires_days: null` means never);
`DELETE {id}` revokes (the row stays for the list; a revoked link answers 404). `ShareOut = {id, token,
path ("/s/<token>"), kind, album: {id, name} | null, title, count, has_password, expires_at,
allow_download, created_at, revoked_at, view_count, last_viewed_at}`; the frontend shows
`location.origin + path`.

Public API (no session; the router carries its full paths and is registered before the SPA
catch-all): `GET /api/s/{token}` -> `SharePublic{title, kind, count, allow_download, expires_at,
items: [ShareItem{uid, kind, width, height, duration_s, taken_at, filename, urls{thumb, preview,
video, download}}]}` in library order, or 403 `share_password_required`, 404 `share_not_found`
(unknown or revoked), 410 `share_expired`; it bumps `view_count` and `last_viewed_at`.
`POST /api/s/{token}/unlock {password}` checks the password (403 `invalid_password`; ten failures per
IP per ten minutes answer 429 `too_many_attempts`, a limiter of its own next to the login one) and sets
the cookie `pcc_share_<token>` (an `itsdangerous.TimestampSigner` signature of the token with the
app secret, HttpOnly, SameSite=Lax, 7 days) that the other public routes accept. Public files, outside
`/api` so they are not gzipped and Range works: `GET|HEAD /files/s/{token}/thumb/{uid}`,
`/preview/{uid}`, `/video/{uid}` (the web copy when done, else the original) and `/original/{uid}`
(only with `allow_download`; `?download=1` adds the attachment disposition); each checks the token,
the password cookie and that the item is a live, non-vault member, serves with the usual
`content-security-policy` and `private, max-age=3600`, and never answers 401 (the owner's open tab
would log out). Share URLs appear in nobody's logs beyond the access log.

Frontend: `/s/:token` (`pages/SharePage.tsx`, outside the shell, its own light chrome: title,
count, "Shared from <site title>", a password form, a simple responsive grid, a lightbox with prev
and next, keyboard arrows and Escape, a video element, and a Download button when allowed; expired
and not-found states). Owner side: `ShareModal` (title, password, expiry never, 1 day, 7 days, 30
days, allow download; afterwards the link with Copy) opened from the bulk bar's Share, the album
page's Share album and the viewer's Share (one item). Settings > Sharing (`SharingTab.tsx`) lists the
links with views, expiry and status, and offers Copy, Edit (password, expiry, download) and Revoke.
`['shares']` is in `LIBRARY_KEYS`.

## 27. Video transcoding with hardware encoders (`app/media/encoders.py`, `app/media/video.py`)

The web copy (`video_preview` job, section 7) exists since the first build: libx264 on the CPU, 720p,
only for originals that browsers cannot stream (`needs_web_preview`). This section makes it use the
machine's encoder, raises the rendition, and shows what is going on.

Encoders (`app/media/encoders.py`): candidates in order from `VIDEO_ENCODER` (`auto` means
`h264_nvenc`, `h264_vaapi`, `h264_qsv`, `libx264`; a named encoder means that one, then `libx264`).
`detect() -> EncoderInfo{name, hw, device, reason, checked_at}` runs a three-frame test encode per
candidate (`ffmpeg -f lavfi -i color=size=320x240:rate=25 -frames:v 3 -c:v <enc> <opts> -f null -`,
20 s timeout; VAAPI with `-vaapi_device $VAAPI_DEVICE`, default `/dev/dri/renderD128`) and keeps the
first that works; the result is cached for the process, computed lazily by the first transcode on
the video worker thread (never at startup), and re-run by `POST /api/system/video/detect`.
`transcode_preview` builds the command for the chosen encoder: software scaling to
`VIDEO_PREVIEW_MAX_HEIGHT` (`scale=-2:min(ih,H)`; VAAPI scales on the device after `hwupload`),
`-profile:v high`, `yuv420p` (nv12 on the device), a keyframe every 2 s, AAC audio, `+faststart`,
and the encoder's quality knob at an equivalent of CRF `VIDEO_PREVIEW_CRF` (`-cq` for NVENC,
`-global_quality` for QSV, `-qp` for VAAPI) capped by `VIDEO_PREVIEW_MAXRATE_KBPS`. A hardware
attempt that fails falls back to libx264 inside the same job (logged, no attempt used); a libx264
failure raises as before. Software encodes keep `nice`.

Defaults move with the hardware: `VIDEO_PREVIEW_MAX_HEIGHT` 1080 (was 720),
`VIDEO_PREVIEW_MAXRATE_KBPS` 6000 (was 2500), `VIDEO_PREVIEW_CRF` 23 (was 27), `VIDEO_STREAM_MAX_KBPS`
8000 (was 3000: H.264 originals up to that bitrate stream as they are). Copies made earlier stay as
they are; `POST /api/system/video/requeue {scope: "failed" | "all"}` -> `{requeued}` queues the web
copy again for videos whose copy failed (`failed`), or (`all`) for every video that needs a copy and
has none yet: `none` and `error` rows, and `pending` rows without a live job. Finished copies are
never redone by this route, so nothing stops playing while the queue works. A hardware run that fails
on one file falls back to libx264 for that copy only; the encoder is skipped for later copies when the
failure is about the device or driver (or after two failures in a row) until Test encoders runs again,
and a finished file is probed (one video stream, a duration) before it replaces anything.

Status: `GET /api/system` gains `video: {encoder, hw, device, reason, checked_at, queued, running,
failed, done, pending}` (`encoder` null until detected). Settings > System gets a Video card with the
encoder in use (for example "h264_nvenc on NVIDIA GeForce RTX 2080" or "libx264 (software)"), the
counts, Test encoders and Re-run failed copies.

## 28. Device uploads: resumable sessions, a ledger and background transfers (`api/uploads.py`, `ios/`)

The chunked protocol (section 6) already resumes by offset within one upload id and the sync check
(section 21) already stops re-sends. What a phone needs on top: an interrupted upload must continue
after the app was killed, the server must confirm per asset what it holds, and the transfer must run
while the app is in the background.

Backend. `uploads` gains `session_id INTEGER` (the session that created the upload) and `resource
TEXT` (`still`, `video` or `file`; the half of a Live Photo), both migrated, with an index
`uploads_asset(session_id, pair_key, resource)`. `POST /api/uploads` from a bearer session with a
`pair_key` is idempotent: an `uploading` row of the same session, `pair_key`, `resource` and `size`
is returned as it is (its `received_bytes` tells the client where to continue) instead of opening
a second staging file; a row of another size is cancelled first. `POST /api/uploads/check` answers
`{known, assets, pending: {asset_id: [{upload_id, resource, received_bytes, size}]}, trashed:
[asset_id]}`: `pending` lists the caller's unfinished uploads for those assets, `trashed` the assets
whose stored item is in the trash (they still count as known, and the app shows them apart so the
owner decides before deleting them from the phone). `UploadCreate.pair_key` reaches the web types
too (the PWA has no use for it beyond typing).

iOS (`ios/Sources/Services/`): a background `URLSession` (identifier `in.rivant.pcc.upload`, never
`isDiscretionary`: iOS may otherwise hold a transfer back for many minutes, even during a sync the
owner started by hand; `allowsCellularAccess` from the Wi-Fi option, and the engine does not start a
run against the Wi-Fi or charging options) carries every chunk as an upload task from a temporary file; the delegate schedules
the next chunk of the same file when one completes, so a file keeps flowing while the app is
suspended and the system relaunches the app into the delegate when it has to. The upload state
(upload id, resource, next offset, chunk file) lives in the `SyncRecord`; on launch and on every run
the engine first asks the sync check and continues the `pending` uploads from the server's offset
(a lost chunk reply is resolved by `GET /api/uploads/{id}`), then hashes and creates the rest.
Create, chunk and complete calls retry with backoff (1, 2, 4, 8, 16, 30 s, at most eight attempts),
`offset_mismatch` resyncs, and a 507 or 413 fails the asset with the server's message. While the engine
waits for an asset, the uploader watches the session every 30 s: a lost hand-off (no request of the asset
in flight) is sent again, at most three times before the asset stalls for this run, and a request with no
progress for 120 s is cancelled and tried again with the backoff, so one asset can never hold a run.
Parallel transfers (2026-11-08): the engine exports and hashes assets one after another but lets up to
`AppSettings.parallelUploads` of them (Settings > Sync > Files at a time, 1 to 4, default 3) transfer at
once. `UploadSlots` watches each started or resumed asset with `wait(for:)` on a task of its own; the run
waits for a free slot before the next export, so at most that many assets sit exported on the phone, and
waits for the rest at the end ("Finishing N uploads"). Every piece of transfer state is per asset (the
record, the uploader's `sending` entry and its stall clock), the session allows
`AppSettings.maxParallelUploads` connections per host, and the Sync screen lists each file being sent
with its own progress. Stop cancels every task in flight; the deadline of a background run ends the
watching, never the transfers. The server needs nothing new: uploads are independent rows with their own
locks, and the ingest queue stores them one at a time as before. The web upload queue already sends three
files at once (`MAX_IN_FLIGHT` in `lib/uploader.ts`). With several assets in flight: while the session is
replaced (a Wi-Fi or charging option change; `finishTasksAndInvalidate` waits for the other assets'
requests) an idle asset is not kicked and `sessionInvalidated()` sends its next request, skipping assets
whose chunk is being cut (their own task hands it over); on a relaunch into the delegate the completion
handler is called only once every chunk being cut is in the session; the engine asks the session again
for each interrupted asset and never resumes one this process is still moving (`isDriving`); and Stop
during an export no longer puts that asset on the wire. Tests: `tests/test_uploads_parallel.py` (four
assets from one device token, chunks interleaved from four connections, every byte in place, the sync
check lists all four).
Assets sent in an earlier run that the sync check now lists are marked synced before anything else, so
the "sent, waiting for the server" count is right from the start of a run. The sync
check sends asset ids, and hashes for assets without a record (after a reinstall or when the photo
was uploaded from the web), so nothing is sent twice. Delete synced lists only assets the server
confirms as stored and not trashed; trashed ones sit under their own heading. The PWA's drop zone
waits for the picker (section 14): iOS hands files over late when originals live in iCloud.

Web: `DropZone` polls the file input after the picker opens (90 s at most) and shows a waiting notice,
because WebKit may deliver the files seconds after the picker closes or without a change event
while iCloud originals download.

## 29. Forwards: media received through messaging apps (`app/analysis/forwards.py`, `app/learning/forwards.py`)

WhatsApp and the other messaging apps re-encode what they deliver and strip the camera's metadata, and a
library synced from a phone fills with memes, greetings, posters and screenshots of posts that are not the
owner's photos. A friend's photo of the owner, received the same way, is a real photo. So: media that came
through an app is recognised (`received_via`), the made-for-sharing part of it is set apart (`forward`), a
real photo received through an app stays in the library with its source noted, and the owner's corrections
train a learner that replaces the rule.

Measured on the owner's library (1,592 items, 2026-11-08): no filename carried an app's pattern (the iPhone
Photos app renames saved media to `IMG_nnnn`, the PWA sends UUID names), 138 JPEGs had neither camera nor
EXIF time, 122 of them with a longest side of at most 1600 px (WhatsApp's cap) against 3 of 900 camera JPEGs;
the JPEG luma quantisation of those sat at 23 to 45 (quality 60 to 80, WhatsApp's range) while every camera
JPEG sat at 6.6 to 8.6 (quality 92 to 96). Filenames matter for Android exports and web uploads; on an iPhone
library the metadata and content signals do the work.

Schema (`items`, every column through `MIGRATED_COLUMNS`):
- `received_via TEXT`: NULL, or the app: `whatsapp`, `telegram`, `facebook`, `snapchat`, `messenger`, or `app`
  when the file carries a messaging app's marks without a known name;
- `forward INTEGER NOT NULL DEFAULT 0` (set apart), `forward_reasons TEXT` (JSON: the signals that spoke for
  it, see below), `forward_source TEXT NOT NULL DEFAULT 'auto' CHECK (forward_source IN ('auto','user'))`;
- `forward_signals TEXT` (JSON of every signal, so a rule change, the learner or the backfill re-decides
  without opening the file again); `forward_checked INTEGER NOT NULL DEFAULT 0` (the signals were computed);
- index `items_forward ON items(forward, deleted_at)`.

Signals (`forwards.file_signals(kind=, ext=, width=, height=, camera=, exif_datetime=, gps_lat=, bitrate=,
filename=, path=) -> dict` at ingest from the probe result and the file header, stored in `forward_signals`;
`forwards.model_signals(meme_p=, doc_likeness=, screenshot_p=, people=) -> dict` in the analyze stage):
- `name_app`: from the original filename: `IMG-YYYYMMDD-WAnnnn`, `VID-`, `AUD-`, `PTT-`, `DOC-` and
  `WhatsApp Image|Video|Audio YYYY-MM-DD at ...` give `whatsapp`; `photo_YYYY-MM-DD_HH-MM-SS` and `video_...`
  give `telegram`; `FB_IMG_<digits>` gives `facebook`; `Snapchat-<digits>` gives `snapchat`;
  `received_<digits>` gives `messenger`; else null;
- `stripped`: an image (JPEG or WebP; PNG is screenshots, HEIC from a phone always has EXIF) with no camera
  make or model and no EXIF capture time;
- `small`: an image whose longest side is at most `FORWARD_MAX_SIDE` (1600); `wa_size`: the longest side is one
  of WhatsApp's outputs (1600, 1280, 1024, 960, 800);
- `jpeg_quality`: the IJG quality estimated from the luma quantisation table (`s = mean(table) / 58.3`, the
  standard table's mean; `q = 100 - 50 s` when `s <= 1`, else `50 / s`), read from the file header only; null
  for anything but JPEG. WhatsApp sends 60 to 80, a phone camera 92 to 96, a screenshot saved as JPEG 90+;
- `video_small`: a video with a longest side of at most 1280 px, no GPS and a bitrate of at most 2,500 kbps
  (WhatsApp re-encodes to 480p or 720p H.264; phones record 1080p and more, with GPS; screen recordings are
  taller than 1280 px);
- `meme_p`: SigLIP zero-shot on the frame-0 embedding (`categories.binary_p`, prompts `MEME_PROMPTS` against
  `NOT_MEME_PROMPTS`): "a meme", "a greeting image such as good morning with text", "a motivational quote on a
  picture", "a poster, flyer or advertisement", "a screenshot of a social media post or a news clipping", "an
  infographic or chart", "a cartoon or comic" against "a photo taken with a phone camera", "a photo of people",
  "a selfie", "a landscape or street photo", "a photo of food"; stored in `category_scores['meme_p']`; null
  without the model;
- `text`: `doc_likeness >= DOC_LIKENESS_MIN` or `screenshot_p >= 0.6` (section 9);
- `people`: at least one good face or body in the item (sections 9 and 10).

Decision (`forwards.decide(signals, *, learned=None, vote=None) -> Decision(received_via, forward, reasons)`,
applied by `forwards.apply(conn, item_id, signals, emb=)` at the end of the analyze stage after the category and
the vault; a row with `forward_source='user'` keeps its forward and reasons; documents and other files only get
`received_via`, the forward rule is for images and videos):
1. `received_via` = `name_app`, else `app` for an image with `stripped` and `small` and (`jpeg_quality` at
   most `FORWARD_JPEG_QUALITY_MAX` (82) or `wa_size`), else `app` for a video with `video_small`, else null.
2. With `received_via` set: `forward` when (`text` and not `people`), or `meme_p >= FORWARD_MEME_MIN` (0.5)
   and not `people`, or `meme_p >= 0.7` whatever is in the picture (posters with faces). Everything else
   received stays a normal library item with its source noted.
3. `forward_reasons` lists the signals that decided (`{name: app, stripped, small, jpeg_quality, meme_p,
   text}`), shown in the viewer like mis-click reasons.
4. The learner (below) replaces step 2 once it has enough examples; step 1 stays a rule.

Learning (`learning/forwards.py`, registered in `train.LEARNERS`): a logistic regression over
`[name_app set, stripped, small, wa_size, jpeg_quality / 100 (1.0 when null), meme_p (0 when null), text,
people, video_small]` fitted on the owner's choices: rows with `forward_source = 'user'` (label = `forward`)
and `learning_events` kind `forward` (label 1 when the owner marked a forward, 0 when they kept one; features
= the signals at the time). Used once `MIN_EVENTS` (10) examples with both labels exist; then an item received
through an app is a forward when the learned probability is at least 0.5. Before that, and on top of it, the
look-alike vote of `learning/images.py` applies to forwards as it does to categories: an item whose frame-0
embedding has a neighbour at cosine 0.85 or more among the items the owner kept or marked, with the winner
holding 60 percent of the weight, follows that answer (`images.forward_vote(embedding) -> bool | None`).
`learning.events.record(conn, "forward", label, signals, item_id=...)` is written by the item and bulk
actions when the owner changes the flag. `train.status()` gains `forwards: {examples, learned, weights}`.

Backfill: items analysed before this section have `forward_checked = 0`. Worker housekeeping
(`forwards.housekeeping(conn, budget_s)`) computes their signals in batches of 200 from the stored metadata,
the file header (JPEG quantisation) and the stored frame-0 embedding (`meme_p` through the SigLIP engine
when it is loaded; null when models are off), decides, and sets `forward_checked = 1`; `GET /api/system`
reports `forwards: {unchecked, running}` and Settings > Learning shows it.

API (section 13):
- `GET /api/items`, the timeline buckets, the map, albums and people items take `forwards=exclude|include|only`
  (default `exclude`) through `visible_sql`, and `received=<app>|any` to list what came through an app;
- `ItemOut` and `ItemDetail` carry `received_via`, `forward`, `forward_reasons`; `ItemDetail` also
  `forward_source`;
- `PATCH /api/items/{uid}` takes `forward: bool` (`forward_source = 'user'`; reasons cleared when false; a
  `forward` learning event); `POST /api/items/bulk` actions `forward` and `unforward`;
- `GET /api/review/forwards` lists the flagged items (newest first, same shape as `review/misclicks`);
  `stats.forwards` counts them.

Frontend (section 14): the Filter bar gets a "Forwards" chip like "Mis-clicks" (hidden by default; the chip
includes them); Review gets a "Forwards" tab (Keep = not a forward, which teaches the learner; Delete = trash;
a count badge from stats); the viewer's info panel shows "Received on WhatsApp" (or "through a messaging app")
with the reasons and a "Not a forward" / "Mark as forward" button; the bulk bar offers "Not forwards" and
"Forwards"; Settings > Learning shows the learner (examples, learned or rule) and the backfill progress.

Configuration (section 15): `FORWARDS_ENABLED` (true), `FORWARD_MAX_SIDE` (1600), `FORWARD_JPEG_QUALITY_MAX`
(82), `FORWARD_MEME_MIN` (0.5).

Tests: filename patterns per app; the JPEG quality estimate on PIL-written JPEGs at quality 75 and 95; the
decision table (received photo of people stays, meme goes apart, text without people goes apart, a camera
JPEG is never received); the learner (rule until 10 examples, then the fitted cut; the look-alike vote); the
filters, the review bin and the bulk actions through the API; the backfill marks rows checked and never
touches a user decision.

## 30. Traffic challans and parking receipts (`documents/parsers/traffic_challan.py`, `documents/parsers/parking_receipt.py`)

Two document types the owner meets often, both in group `money`, detected by `detect.RULES` like every
other type and parsed by a pure parser (text and boxes in, `ParsedDocument` out), so the document learner
of section 18 (`learning/doc_type.py`, the owner's relabels as TF-IDF exemplars) covers them as it covers
the rest: a type the owner sets on one document carries to the ones that read alike.

`traffic_challan` ("Traffic challan"): an e-challan page or its printout (state traffic police portals,
the national Parivahan portal, a city police challan). Detection: "challan" (0.5), "e-challan" (0.5),
"traffic police" (0.3), "challan no" (0.3), "challan details" (0.2), "comp. fee" or "compounding fee" (0.2),
"payment status" (0.2), "violation" (0.1), "MVA" or "motor vehicles act" (0.1), a challan number (an
identifier, 0.3: two to six capitals then 11 to 20 digits, as in `PNCCC26002934519`), a vehicle number
(0.1, `[A-Z]{2} ?\d{1,2} ?[A-Z]{1,3} ?\d{4}`); negative "parking" (0.5). Fields, in this order:
`challan_number`, `date` (the challan date and time), `vehicle_number`, `vehicle_owner`, `owner_address`,
`violations` (every row of the violations table joined as "Sec 122/177 MVA: description (Rs 500)"),
`amount` (Comp. Fee or the fines' total), `payment_status` (as printed: Unpaid, Paid, Disposed, or the
owner's own value), `location`, `officer` (name and designation), `mobile`, `evidences` (count of image
links). Title "Challan {challan_number}" with the vehicle number appended when known; `doc_date` the
challan date; `issuer` the heading ("Maharashtra Traffic Police", "Pune City Police"); `person_name` the
vehicle owner. Labels read from "Label   Value" rows and from the table rows the OCR flattens
(`fields.after_label` with the challan's labels, local-script labels for Marathi and Hindi portals).

`parking_receipt` ("Parking receipt"): a thermal or printed parking ticket. The OCR of such prints is
rough (the example reads "U.No" for "V.No", "PUT. LTD" for "PVT. LTD", "OUNERS RISK"), so detection rests on
several weak marks: "parking" (0.5), "parking charges" or "parking fee" (0.3), "in dt", "in tm", "out dt",
"out tm", "in time", "out time" (0.3 together), "vehicle type" (0.2), "2 wheeler", "4 wheeler", "2w", "4w"
(0.25), a tariff line "... hours rs N" or "rs N per hour" (0.3), "owner's risk" matched loosely (`o[uw]ners?.?s?
risk`, 0.2), "pass not allowed" (0.1), "valet" (0.2); negatives "challan" (0.5), "traffic police" (0.5).
Fields: `receipt_number` (BILL NO, TICKET NO, RECEIPT NO), `vehicle_number` (V.No, U.No, VEH NO, VEHICLE NO;
often only the last four digits), `vehicle_type`, `in_time` (IN DT plus IN TM, as one ISO timestamp), `out_time`
(OUT DT plus OUT TM when printed), `duration` (when both are known), `amount` (TOTAL, AMOUNT, PAID), `operator`
(the company lines above the bill number: "P ONE INFRA PVT. LTD", "MANAGED BY ..."), `location` (a site name
or address when printed), `tariff` (the rate lines joined, optional). Title "Parking {operator or location}
{date}"; `doc_date` the in date; `issuer` the operator.

Paying a challan and its status: the Maharashtra portal (`https://mahatrafficechallan.gov.in/payechallan/PaymentService.htm`)
and the national one (`https://echallan.parivahan.gov.in/index/accused-challan`) take the challan number in
a form behind a captcha; there is no public API, and the paid lookups that exist are business services. So
the challan card offers "Pay or check status": it copies the challan number to the clipboard and opens the
portal that matches the challan (Maharashtra when the issuer or the number says so, else Parivahan), the
status shown is the one printed on the challan (`payment_status`), and the owner marks it paid on the card
(`PATCH /api/documents/{uid}` fields `payment_status` = "Paid" and `paid_on`, `source='user'`). The backend
exposes the portal choice as `ParsedDocument.fields['pay_url']` (label "Pay at", `traffic_challan.portal_for`),
never fetches anything from the portals, and `CHALLAN_PORTAL_URL` (env) overrides the Maharashtra link for
another state's portal.

Existing documents: `kv doc_rules_version` records the rule set the automatic documents were detected with
(`ingest.DOC_RULES_VERSION`, 2 for this section); at startup, when it is lower, every automatic `other`
document whose stored text scores for a new type (`ingest.DOC_RULES_NEW_TYPES`) is queued for a `text` job
(`ingest.recover_on_startup`, inside its one transaction), so the challans and parking tickets already in the
library get their type without a click; the owner's own types (`documents.source = 'user'`) are never touched.

Frontend (section 14): `docMeta.ts` gains both types (labels, lucide icons `TrafficCone` and
`SquareParking`, field order and labels, key fields); the type picker lists them; the challan card shows the
status as a chip (Unpaid in the warning colour, Paid plain), the "Pay or check status" button, "Mark as paid";
the Documents page groups them under Money like receipts.

Tests: both parsers on anonymised copies of the two OCR texts measured on 2026-11-08 (the Maharashtra
Traffic Police challan page and the thermal parking bill), the detection scores of each against the other
and against a receipt and an invoice, the pay-portal choice, and the startup reparse of an automatic `other`
document.

## 31. Storage tiers and nightly backups (`app/storage.py`, `app/tiers.py`, `app/backup.py`)

Three disks, three jobs. A fast disk holds the bytes the owner uploaded; the disk the app runs from holds
the database and everything made from the originals (previews, thumbnails, page renders, streaming copies,
crops, the served site); a large slow disk holds a snapshot a night. Nothing changes when the two settings
are unset: one folder, as before.

**Tiers.** `ORIGINALS_DIR` is the originals root: `originals/`, `staging/` and `failed/` live there
(`storage.TIER_DIRS`); every other folder stays under `DATA_DIR` (`storage.DERIVED_DIRS`). The same path
pattern applies under either root, so `storage.rel`/`from_rel`, `_is_referenced` and the sweeper only need
the top folder name. `ORIGINALS_DIR` equal to `DATA_DIR` means one tier. `GET /api/system` reports
`storage.tiers` with each root's free and total bytes and what it holds; a tier on the same filesystem as an
earlier one carries `shared_with` so the Settings page counts a disk once. `disk` keeps meaning the disk that
receives uploads.

**Switching an existing library.** Set `ORIGINALS_DIR` and restart. At startup, before recovery and the
workers, `tiers.move_staged()` moves every `staging/*.part` and `failed/*.part` left under `DATA_DIR`
(copy beside the destination, size check, fsync, rename, unlink; synchronous, because chunk writes must
never race it; a file already present under the new root wins). The originals follow on a nice-19 thread
(`tiers.start()` -> `migrate_originals`): one file at a time, a destination that already exists with the same
size and mtime (a copy made by hand before the switch) is kept and only the old file removed; empty shard
folders and `DATA_DIR/originals` itself go at the end. While a legacy file remains, `storage.original_path`
returns it, so viewing, downloading and analysis keep working during the move, and the sweeper walks the
legacy folder too. `GET /api/system` reports `storage.migration` (pending and moved counts and bytes, the
last error); the move resumes at the next start when it was interrupted. Setting `ORIGINALS_DIR` back is not
automatic: move the three folders by hand.

**Backups.** With `BACKUP_DIR` set, `backup.start()` runs a nice-19 thread that takes one snapshot a day at
`BACKUP_HOUR` (default 3) local time. Layout: `BACKUP_DIR/snapshots/<YYYY-MM-DD>T<HHMM>/` holds `cloud.db` (an
online copy through SQLite's backup API that passed `PRAGMA quick_check`), `secret.key`, `learned/`,
`wallet/` and `originals/<sh>/<file>`; an original whose size and mtime match the previous snapshot's copy is a
hard link to it, so a night costs only what changed. `BACKUP_DIR/latest` points at the newest complete
snapshot; `snapshots/.partial-<name>` is the one being written and is deleted at the next run if left over.
After each run the newest `BACKUP_KEEP` (default 14) snapshots are kept. Not in a snapshot: derived files
(rebuilt from the originals), staging and failed uploads, `tmp/`, `site/`, and `.env` (outside `DATA_DIR`).
The body traits of section 34 keep their crops in the database, not in files: `body_traits.crop` (192 px WebP),
`detail` (tattoo and mehndi crops at native resolution, WebP, at most 512 px) and `vec`, and the descriptors of
the owner's hides in `trait_hide_desc`. So they are in `cloud.db` and in every snapshot, and the orphan file sweep
never sees them. The estimate before the first measurement was 100 to 130 MB for the owner's library; it is not
measured yet.
A run never starts while the originals are still being moved, a missed night (the server was off) is made
up five minutes after start when the last complete snapshot is older than a day, a first snapshot waits for
the hour, and a failed run is tried again an hour later. The outcome is kept in kv `backup_last` (started_at,
finished_at, ok, error, snapshot, files, bytes_copied, bytes_linked, duration_s, reason) and shown by
`GET /api/system` as `backup` (with `next_run_at`, the snapshot names newest first, the backup disk's free
space and, while a run goes, `progress{phase, files_total, files_done, bytes_copied, bytes_linked}`).
`POST /api/system/backup` takes one now (Settings > System > Back up now); a run already going is reported,
not doubled.

**Restore.** Stop the server; copy `snapshots/<name>/cloud.db` to `DATA_DIR/cloud.db` (remove a stale
`cloud.db-wal` and `cloud.db-shm`), `secret.key`, `learned/` and `wallet/` into `DATA_DIR`, and `originals/`
into the originals root; start. Previews, thumbnails and streaming copies of items that lost them are
rebuilt with "Process again" (section 13) or `POST /api/system/video/requeue`.

Frontend (section 14): Settings > System shows a Storage card (each disk once, with its free space and what
it holds; the move's progress while legacy originals remain) and a Backups card (the last snapshot and its
outcome, the next run, snapshots kept, the backup disk, "Back up now", the progress bar while a run goes;
with backups off, how to turn them on).

Tests: paths under both roots, `rel`/`from_rel`, the legacy fallback, `move_into_place` across filesystems,
an upload landing on the originals root and being served, `move_staged`, `migrate_originals` (fresh, already
copied, damaged copy, interrupted and resumed), the sweeper over both roots, `has_room_for` over both disks,
snapshot contents and hard links, retention, a failed run leaving no partial snapshot, legacy originals in a
snapshot, the schedule (`next_run`), and "Back up now" through the API.

## 32. Processing speed (`app/throughput.py`, Settings > System)

Every job that ran to a verdict leaves a row in `job_runs` (`type`, `queue`, `finished_at`, `duration_ms`,
`ok`); a job deferred because the disk is full or ffmpeg could not run did not run. Housekeeping deletes
rows older than `throughput.KEEP_DAYS` (14). `GET /api/system/throughput?range=hour|day|week` sums them into
buckets aligned to the local clock (`LOCAL_TIMEZONE`), the last bucket holding now (so partial): the hour as
60 one-minute buckets, the day as 24 one-hour buckets, the week as 28 six-hour buckets. The reply lists every
job type in a fixed order with a label (`ingest` "Files stored", `analyze` "Photos and videos analysed", `text`
"Documents read", `video_preview` "Video web copies", `guess` "Person guesses", `recluster` "Face clustering
runs"), its total and failed counts, the mean duration in ms, and per bucket the finished and the failed
counts; `per_minute` gives each type's finished runs per minute over the last ten minutes.

Frontend (section 14): a "Processing speed" card in Settings > System with a range switch (Last hour, 24 hours,
7 days), one small column chart per job type (one hue for finished runs, the danger colour stacked for
failures, a legend, hover and focus tooltips with the bucket's time, counts and mean duration) plus the
type's total, mean duration and current rate; "Show as table" lists the same numbers. The card polls every
30 seconds while the tab is open.

Tests: bucketing of the last hour (placement, failed counts, mean duration, the ten-minute rate), day and
week buckets aligned to Asia/Kolkata hours, pruning, the worker recording completed and failed runs, and the
route (including 422 for an unknown range).

## 33. Undo (`app/undo.py`, `app/api/actions.py`)

Every change the owner makes can be undone, and the undo redone (2026-11-09). The history the owner sees is
kept in the browser (section 14); what each action changed is kept by the server, so an undo puts back
exactly what happened, including the look-alikes a move pulled along, the faces a "Not a face" took with it,
and the rows a foreign key cleared.

- An action is one API write (POST, PUT, PATCH, DELETE under `/api/`) outside the routes that are not owner
  content or cannot come back: `/api/auth`, `/api/uploads`, `/api/system`, `/api/actions`, `/api/trash` (a
  purge deletes the originals), `/api/search`, `/api/s`, reprocess, recluster, map refresh and document
  reparse; the bulk purge runs inside `undo.not_undoable()`. `ActionMiddleware` names the action from the
  client's `X-Action-Id` header (`[A-Za-z0-9_-]{8,64}`; every request of one action in the app carries the
  same id, so a bulk loop undoes as one), else a new id, and keeps it in `db.current_action` (a ContextVar
  that reaches the endpoint threads). The response carries `X-Action-Id` when the action changed something.
- Journal: `db.tx` writes the `undo_context` row at BEGIN of every write transaction of an action and removes
  it before COMMIT, so no other connection (the worker) ever sees it. Triggers on the journaled tables (items,
  faces, bodies, animals, persons, item_tags, guess_rejections, merge_dismissals, learning_events, not_faces,
  albums, album_items, shares, share_items, tags, item_user_tags, tag_rejections, documents; made from the
  current columns by `undo.install_triggers` at every start) log each row change into `undo_log` with the row
  before and after as JSON (BLOB columns as hex), built 30 columns per function call because SQLite before
  3.48 allows 127 arguments and cannot open a database whose trigger has more. A delete keeps the whole row,
  an update only the columns that changed plus the key and `item_id` (moving a face logs about 650 bytes
  instead of 21 KB of embeddings), an insert every column but the BLOBs (the undo's own delete keeps the
  whole row, ready for a redo). Foreign-key actions (ON DELETE SET NULL, CASCADE) fire them
  too. `kv`, jobs, uploads, sessions and the derived tables are not journaled: a version counter put back
  would leave the face index stale. Measured on the API tests: no owner write happens outside a transaction.
- Recognition trained on one person (section 10): the pause of a model is `persons.recognition_paused_at`, a
  persons column like any other, so the undo of "Take back its guesses" brings the guesses back and un-pauses the
  model, and a redo takes both away again. The model's file goes and comes with its persons row
  (`undo.ROW_FILES`, beside the crops of boxes): `person_model.forget` hands it to `undo.discard`, so the undo of
  deleting the person, of a merge, or of a rename that merged puts it back from the action's stash, and a redo moves
  it to the undo's stash. Training and looking again run on a background thread outside any action, and
  forgetting a model runs inside `undo.not_undoable()`: a model and its guesses are derived, and an undo could
  bring the guesses back but not the model behind them.
- Body traits (section 34): `trait_hides` (the owner's "Not a tattoo", "Not this person's" and "Show again") is
  journaled and holds no BLOBs, so its journal stays small. The derived trait tables (`body_traits`,
  `trait_scans`, `trait_hide_hits`, `trait_requeue`) are not journaled, as cascades would hex-log their crops
  (nor are `place_links` and `place_scans` of section 35, derived data that the runner rebuilds);
  `trait_hide_desc` (a hide's descriptors) is not journaled either and has no foreign key, so an undone unhide
  finds its descriptors again, and `traits.prune` drops it only when the hide is gone, it is older than
  `undo.KEEP_DAYS` and the hide's removal is no longer in `undo_log`. After its transaction, `undo.undo()` hands the
  tables and items it replayed to `traits.after_undo`: the photos of hides put back or taken away are queued for the
  traits runner (which recomputes their look-alikes and guesses them again, never inside the request), and boxes put
  back change the photo's box fingerprint, so the runner is kicked. A failure there is logged, never raised. The hide
  route queues the photo the same way, so its guesses are rewritten outside the action's journal and an older hide
  can still be undone after a later one rewrote the same guess columns.
- `POST /api/actions/{id}/undo` -> `UndoOut {undo_id, changes, partial, items}`. Newest change first, in one
  transaction with foreign keys deferred: a row the action inserted is deleted, a row it deleted is inserted
  again, an update puts back only the columns it changed and only where the column still holds what the
  action wrote (when the analysis changed one since, it stays and `partial` is true). Bookkeeping columns
  (`updated_at` and the items' face, body and animal counts) never block an undo and are not put back; the
  counts of the items touched are counted again. The undo is an action
  itself (`undo_actions.undo_of`), so undoing `undo_id` is a redo. Answers: 404 `not_found`, 410 `expired`
  (older than 30 days), 409 `already_undone`, 409 `{detail: "later_changes", blocking: [ids]}` when a later
  action still in effect changed the same rows and columns (undo those first), 409 `conflict` when the old
  row no longer fits (a name taken since), 403 `vault_locked` when the action touched a vault item and the
  vault is locked. Afterwards `faces_version` and `search_version` move and the items' full-text rows are
  rewritten.
- Files: a crop the action removes (`undo.discard`, used by the box deletes) moves to `DATA_DIR/undo/<action>/`
  instead of being deleted; an undo that puts the row back puts the crop back, and the crops of rows an undo
  removes go to the undo's own folder, ready for a redo. A crop an action replaces (a moved box,
  `undo.keep_copy`) is copied there first; an undo that puts the box's x, y, w or h back swaps the crops (the one
  on disk goes to the undo's folder, the action's copy comes back), so a redo swaps them again. Faces whose vector
  or weight columns come back are taken again by the face index (`FaceIndex.reload_faces`).
- Housekeeping drops actions older than 30 days with their journal and files (`undo.prune`).
- Tests: `tests/test_undo.py` (a face named then undone and redone, "Not a face" with a look-alike and their
  crops, a merge and a person delete, a drawn person box, category, trash and album, a later change blocking,
  the analysis' change kept with `partial`, one action id over two requests, routes that are not actions,
  the vault, expiry and pruning, counters that never block, the triggers of every table and their argument
  count, the size of an update's journal, and BLOBs put back by an undo and a redo).

## 34. Body traits: tattoos, nails and mehndi (`app/analysis/traits.py`, `trait_regions.py`, `trait_nails.py`, `trait_sift.py`, `trait_match.py`)

People boxes whose face does not show (a back, a helmet, a turned head) and unclear faces get guesses from the clothes
of photos taken within the hour (section 10). Body traits reach further: a tattoo is the same person months later, and
nail polish or mehndi is the same person a few days later. They are read from the people boxes of photos and used as
evidence for guesses, never as an assignment: the owner confirms or corrects every guess (2026-11-09).

| Kind | Term | Matched against |
|---|---|---|
| `tattoo` | long | references at any date, geometrically verified |
| `mehndi` | short, off unless `TRAITS_MEHNDI` | references within `MEHNDI_MAX_DAYS` (21), full weight for 3 days |
| `nails` | short, support only | references within `NAILS_MAX_DAYS` (14), full weight for 2 days: polish colour, finish (plain, french, glitter, art, multi) and length (short, medium, long) |

Parts: `forearm_l/r`, `upper_arm_l/r`, `calf_l/r`, `hand_l/r`. Matching compares part families (the part without its
side), because a mirror photo swaps the sides. Mehndi is read only with `TRAITS_MEHNDI` (off by default): on the
owner's library SigLIP's henna group won on plain hands (249 of 255 mehndi rows were hands, and a random 40 showed no
mehndi), so it waits for a gate measured on real mehndi. With it off, a crop SigLIP calls henna is no evidence at all
(never a tattoo) and does not stop nails. Not read yet: the neck (necklaces and mangalsutras read as ink), nail
shape (fingertips of 10 to 25 px), hair colour (lighting dominates, and faces cover it), piercings, jewellery and
watches (small, removable and common; watches and bangles are negatives in the gate), scars, feet, and boxes in
videos (scanned with status `video`). An open face gets traits only through the body that holds it.

### Reading a photo (`trait_regions.measure_item`, on the runner thread)

The item is the unit: every body of the photo is measured together, outside any transaction, and written in one short
transaction that first checks the photo's box fingerprint and bodies did not change meanwhile (else nothing is written
and the photo stays pending). Statuses set without decoding, first match wins: `video` (not an image, or a box at a
video time), `vault` (`nsfw = 1`), `doc` (category screenshots or documents, or `doc_likeness >= DOC_LIKENESS_MIN`),
`small` (the box's long side under `TRAIT_MIN_BODY_PX`, 200, in pixels of the 2048 px preview) and `child` (the ages
are read and the body's face is in the child band, section 10). The rest are measured: `ok`, `no_pixels` when neither
the preview nor the original reads, or `error` when our own code failed on that body (it keeps no rows). `doc`,
`child`, `no_pixels` and `error` are checked again after a day; `vault` as soon as the photo leaves the vault.

1. Pose (`ml/pose.py`, section 8) runs once over the measured bodies of the photo, on the 2048 px preview
   (`storage.preview_path`; `animals.open_photo` when there is none).
2. Limb crops along the keypoints: the forearm from the elbow to 80 percent of the way to the wrist (so watches,
   kadas and bangles stay out), the upper arm and the calf whole. A limb counts with both keypoint scores at 0.5 or
   more and a segment of at least `TRAIT_MIN_LIMB_PX` (64). Hands: the 21 DWPose hand points with a mean score of
   0.4 or more and a span of 64 px or more (the gate was measured on hands that size or larger).
3. Ownership drops a region, never flags it. It must lie at least 80 percent inside its box grown by 10 percent, no
   other body's pose may hold the same family of limb (scores 0.3 or more) within half its length, and it may lie over
   another box of the photo by half at most (people standing close overlap a little; before, 15 percent dropped
   regions no one else claimed); a side whose forearm and upper arm lengths are out of proportion (ratio
   outside 0.5 to 1.6) is dropped. Boxes at IoU 0.7 or more with the body are its twin and ignored. This is the
   case of one person's tattooed arm lying inside a relative's box in a group photo.
4. The gate, on 224 px crops of the preview: a limb as a square of 1.15 x the segment, upright, with the sides outside
   the limb strip set to grey (the strip is 0.6 of the segment for a forearm, 0.7 for an upper arm, 0.55 for a calf); a
   hand as 1.4 x its span, wrist down. SigLIP zero-shot (sent as PIL RGB images) over five groups: tattoo, bare skin,
   cloth (a sleeve, trousers, printed fabric), henna and jewel (a wristwatch, a bracelet or bangle, a phone in the
   hand). Skin (YCrCb within Cr 133 to 173 and Cb 77 to 127, or the face's own skin plus or minus 10), ink (inside the
   hull of the skin, not skin, and 20 or more below the skin L* or 25 or more away in chroma) and sharpness (Laplacian
   variance) are read on the strip.
5. The verdict, first rule that matches:
   - henna (forearm or hand): p(henna) 0.50 or more, or the tattoo rule passes and the darkest pixels are brown to
     orange (median Lab hue 30 to 75 degrees, chroma 15 or more): `mehndi` with `TRAITS_MEHNDI`, else no evidence.
     Henna is never a tattoo.
   - `tattoo`: p(tattoo) at least `TATTOO_GATE_P` (0.80), p(cloth) under 0.10, p(jewel) under 0.15 and sharpness 30
     or more. The skin and ink fractions are not used for tattoos: on the owner's library they kept 345 of the 874
     crops that passed the SigLIP cuts and dropped mostly real sleeves (dense ink leaves little skin), while the
     sharpness floor alone keeps 852. They still decide bare limbs.
   - `unclear`: p(tattoo) from 0.60 up to the gate, or the tattoo rule failing only on p(jewel). No row, and not bare.
   - `bare`: p(bare) 0.80 or more, skin 0.60 or more, ink under 0.01, sharpness 60 or more, keypoint scores 0.6 or
     more and a segment of at least 2 x `TRAIT_MIN_LIMB_PX`. Written into the scan's `bare` as `{part: {p, len}}`, the
     only source of "tattoo absent".
6. Every tattoo or mehndi crop is cut again from the original at native resolution (the original decoded once per
   photo, only when needed, with `images.load_oriented` and no size limit): the same rectangle with the keypoints
   scaled by original side / preview side, `clamp(1.15 x segment x scale, 224, 512)` px square, `INTER_AREA`. When the
   original does not read, or is under 1.2 times the preview, the preview stands in (`attrs.src` `preview`, else
   `original`). The row keeps that crop as `detail` (WebP quality 90), a 192 px `crop` for the UI, and `vec`: the
   DINOv2 CLS of the crop at 224 px and of its mirror. `attrs` hold the five group probabilities `p`, the SIFT
   keypoint count `kp` (a crop with fewer than 60 is `small` and never counts), the strip `share`, the side `out`,
   `src`, `skin`, `ink`, `sharp` and the segment `len`. `quality` (for display order) is min(1, short side / 96) x the
   keypoint score x min(1, sharpness / 150).
7. Nails, on hands with no tattoo or mehndi and p(jewel) under 0.50, whose span is at least `NAIL_MIN_HAND_PX` (96)
   in the source used (the original when the photo's long side is 3072 px or more, else the preview):
   `trait_nails.describe_hand` on a 224 px crop of 1.7 x the span, wrist down, with the MediaPipe landmarks
   (`ml/hands.py`). No row unless every gate holds: landmark confidence 0.8 or more; the back of the hand faces the
   camera (the sign of `cross(p5 - p0, p17 - p0)` for the person's right or left hand, from DWPose); at least 2
   fingers whose nail strip (DIP to TIP, from 0.5 to 1.0 of it) differs from their own skin strip (PIP to DIP) by 12
   or more (CIELAB distance) and whose colour does not run on past the tip for more than 0.6 of DIP to TIP (a held
   cup, phone or pole); SigLIP on the fingertips padded by 35 percent says painted nails at 0.5 or more; the polished
   nails agree within 15 (or form two groups of 2 or more, 20 or more apart: `multi`); the fingertips' sharpness is 80
   or more. The colour is kept after a von Kries balance in linear RGB that maps the hand's own skin to one target
   skin colour, so coloured light shifts skin and nail together and cancels; it is named from 14 colour names and
   kept as `lab` and `hex`. Finish by zero-shot (counted only at 0.7 or more); length from the median run past the
   tip (under 0.15 short, under 0.35 medium, else long). A row stores no `vec` or `detail`.

An exception inside a model call (pose, hands, DINOv2, SigLIP, including a failed session or the GPU running out of
memory) stops the whole run with nothing written for that photo, which stays pending; the message is shown in
Settings > Learning and the next kick tries again. When a model fails on the same photo 3 times in a row
(`ABORTS_PER_ITEM`), that photo's bodies get `error` instead, so one photo cannot stop the backfill. Any other
exception while a photo is measured or written marks its bodies `error` (or leaves it pending when the write failed)
and the run goes on.

### The runner and readiness (`traits.py`)

The runner and its triggers are described in section 7. Each round: the sweep when a new rev became ready
(`trait_match.sweep_open` queues the photos whose open boxes carry a usable tattoo or mehndi row), the re-guess queue,
then the pending photos in 30 s steps, newest first, draining the queue between steps, until nothing is pending, a
model fails, or the process grew by 2 GB since the loop started or passed 7 GB (the review server's limit is 8 GB; the
next kick resumes). It sleeps 2 s and asks again before it exits, because kicks are dropped while it runs.

`traits.wanted_rev()` is `TRAITS_REV` plus the kinds measured (`2:nails,tattoo`; with `TRAITS_MEHNDI`,
`2:mehndi,nails,tattoo`), never the models that happened to load. Rev 2 is the reviewed gate and ownership; switching
mehndi on or off changes the rev, so the library is measured again by itself. kv `traits_ready` is set to it when nothing is pending (an empty library is ready at start), and
matching uses only rows whose scan has the ready rev, so a missing model never switches matching off: without all four
engines (pose, hands, dino, SigLIP) or with `TRAITS_ENABLED` off, nothing is measured, written or deleted, and
`progress()` reports `paused` with the missing keys. A usable row has an `ok` scan of the ready rev measured on the
body's current box, on a live photo outside the vault that is not a screenshot or document. After a photo is written
the runner checks the new rows against the owner's "Not a tattoo" hides (look-alikes, below), guesses the photo again
and queues for a re-guess the photos whose open boxes look like the new rows of named people (at most 50).

### Matching (`trait_match.py`)

`trait_match.context` returns nothing (and guessing takes the clothes path unchanged) unless `TRAITS_ENABLED` and
`TRAIT_MATCHING` are on, the library is ready, the photo is an image with a capture time, and an open box has a
usable row or bare limbs. Queries are the rows of an open faceless body, or of the body that holds an open face.
References are bodies with a person assigned by the owner or automatically: strong when the body or its face was
assigned by the owner, weak otherwise. People present in the photo and rejected people are left out, and so are people
an open face contradicts: another named person scores the face at 0.40 or more, or an unnamed group does while the
face scores the person under 0.20. (A tattooed person's photos split into an unnamed group must not veto that person:
on the owner's library such a group of 8 faces blocked a box whose tattoo verified with the person at 59 inliers.)

Near-duplicates give no evidence and count for nothing: the same dup group, a Live pair (same `content_id`,
`companion_of`, `live_video_item_id`), pHash within 6 bits, or a burst (scene cosine 0.97 or more within 120 s).
Short-term kinds need an EXIF capture time on both photos and no forward (section 29). With the ages read, a body
holding a child-band face is never a query or reference for nails or mehndi, and never a tattoo reference.

Tattoo, long term:
- Retrieval: DINOv2 cosine of the query crop against each reference crop and its mirror, 0.45 or more, same family,
  never the query's own photo or a near-duplicate of it; the top 20, at most 8 per person (the probe's figures came
  from the top 20 regardless of person).
- Verification (`trait_sift.verify`): SIFT (1,500 features, contrast threshold 0.02) on the CLAHE (2.0, 8 x 8) grey of
  the native crops, limited to the limb strip; plain against plain, mirrored against plain and plain against
  mirrored; ratio test 0.8, at least 6 good matches, `estimateAffinePartial2D` with RANSAC at 6 px per 224 px of the
  pair's mean side (3,000 iterations), a fitted scale between 0.5 and 2.0. A pair verifies (`passes`) with at least
  `TATTOO_MIN_INLIERS` (12) inliers and an inlier share of the good matches of 0.25 or more. Results are cached by row
  id in both directions (8,192 results, 256 feature sets), and one `guess_item` call spends at most 120 verifications.
  Cosine alone never counts.
- Each verified pair gives a log likelihood ratio by its inliers: 12 to 19 give 2.5, 20 to 29 give 3.0, 30 or more give
  3.5, times min(1, query keypoints / 300); a `small` crop on either side gives 0.
- Same photo: a query row that verifies with another body's row of its own photo counts 0 (whose arm it is cannot be
  told), and a reference photo where two bodies verify is dropped.
- Shared design: when several people verify with one query row, the strongest keeps 0.75 of its value if its strength
  (independent verifying photos, plus 1 with a strong reference) is at least twice every other's and no other has a
  strong reference; otherwise the row counts 0 for everyone.
- Per person: the best pair per family, summed over families, plus 0.5 for each further verifying photo on another date
  and in another dup group (at most 1.0); halved without the reference bar (one strong reference, verifying photos
  from 2 dates, or 3 separate verifying photos of a named person, not near-duplicates of each other, on any dates:
  one event seen from several sides); at most 6.0. A pair worth 0 (a `small` crop, a shared design) counts for
  nothing: no term, no support and no bar.

Tattoo absence: a person's track (their rows of one family linked by verified pairs, at most 60 rows) seen on dates 30 or
more days apart with a median ink share of 0.15 or more gives -0.7 when the query's scan lists both sides of that
family as bare (each segment at least 2 x `TRAIT_MIN_LIMB_PX`), the query has no tattoo row there, and the query was
taken after the track was first seen. Nothing is inferred from photos before a tattoo appeared. All negatives together
are at least -1.4.

Mehndi, short term: the same retrieval, verification and rules within `MEHNDI_MAX_DAYS` at `MEHNDI_MIN_INLIERS` (30;
dense repeated patterns verify too easily), 30 to 59 inliers giving 1.0 and 60 or more 1.5, times a decay (1 for 3
days, then halving every 5 days, 0 past the window); at most 4.0, and no absence.

Nails, support only, within `NAILS_MAX_DAYS`: the colour distance weighs lightness half (`sqrt((dL/2)^2 + da^2 +
db^2)`). Rarity f is the share of other people's polished hands within the colour distance (default 10) in the same
window (the library's share with fewer than 20 there, 0.25 with fewer than 50 in the library); the colour term is
`ln(1/max(f, 0.02)) / ln(50)` x (1 - distance / 10), plus 0.5 for the same art or multi finish, 0.4 glitter, 0.3
french (both at 0.7 or more), plus 0.2 for the same length; times a decay (1 for 2 days, then halving every 4 days);
further dates add 0.25 x their decay; at most 1.5. It counts 0 for everyone when another named person, or another body
in the query's or the reference's photo, matches the colour. Both polished, clearly different (more than 25) and 2
days or less apart gives -0.5, used only on the trait-only path.

`L(P)` is the sum of the tattoo, mehndi and nails terms and the negatives. The stored score of a trait-only guess is
`sigmoid(L - ln TRAIT_PRIOR_PEOPLE)` (prior odds 1:20): a sort key, not a calibrated probability, so the UI shows
"strong match" in words.

### Fusion with guesses (`trait_match.fuse`, `guess._compute_fused`)

The clothes path runs as before and each box's result goes through `fuse` with that box's traits, inside the same
greedy one-person-per-photo loop. The first rule that applies wins:
1. Absence veto: the clothes winner's absence term is at the cap (-1.4): no guess.
2. Conflict: traits admit a different person by the rule below: no guess.
3. Agreement: the clothes winner has `L > 0` (without the nails negative): the clothes guess stays, with its learned
   score, as basis `clothes+trait`.
4. Otherwise a clothes winner stays as it is.
5. Tie-break: no clothes winner only because of the 0.05 margin, and among the people within the margin exactly one
   has a tattoo pair with `tattoo_single_min` (20) or more inliers or verified photos on 2 dates (or a mehndi pair at
   its floor), and the others have `L <= 0`: that person wins with the clothes score, basis `clothes+trait`. Nails never break a tie.
6. Admission of a trait-only guess, basis `trait`, when there is no clothes candidate, or when the clothes candidates
   tied and the admitted person is one of them. All of these must hold: `TRAIT_ONLY_GUESSES` is on; the person has a verified tattoo pair with
   `tattoo_single_min` (20) or more inliers, or verified pairs from 2 or more photos on different dates that are not
   near-duplicates of each other; no one else verifies with the box's rows after the same-photo and shared-design
   rules; the reference bar holds; the tattoo term is 2.0 or more and at least the nails term; the lead over the
   runner-up, both without nails, is at least `TRAIT_MARGIN_LLR` (2.0); and the face
   veto, child band, vault, documents, rejections, people present and one person per photo allow it. Mehndi admits
   alone only with `TRAIT_ONLY_MEHNDI` (60 or more inliers, or 2 photos). Nails never lead and never admit.

Evidence. A trait-only guess stores `{basis: "trait", score, trait_llr, ref_item_uid, ref_body_id, days_apart, traits,
against, terms}`, and never the clothes keys. A `clothes+trait` guess stores the clothes evidence plus `basis`,
`trait_llr`, `traits` and `score`. `traits` holds at most 3 entries, strongest first: for tattoo and mehndi `{kind,
part, ref_part, mirrored, trait_id, ref_trait_id, ref_body_id, ref_item_uid, ref_taken_at, days_apart, ref_before, cos,
inliers, support, strong, llr}`, for nails `{kind, part, trait_id, ref_trait_id, ref_body_id, ref_item_uid,
ref_taken_at, days_apart, ref_before, color, hex, finish, length, de, decay, llr}`; `against` lists `{kind, part, llr,
why}` (`bare` for a tattoo seen before, `color` for nails). No URLs are stored; the API adds the crop URLs. Floats are
rounded and lists sorted, so the same evidence gives the same JSON. `guess.apply_window_change` keeps basis `trait`
guesses; `record_outcome` records their own terms (`trait_match.outcome_terms`), which the people learner leaves out
(section 18).

### Corrections

`PUT /api/traits/{id}` (section 13) writes `trait_hides`: `not_trait` ("Not a tattoo", "Not nails", "Not mehndi"),
`not_theirs` ("Not this person's", with the body's box) and `shown` ("Show again"). A row is hidden when (a) a
`not_trait` hide of its kind in its photo covers its region at IoU 0.3 or more, (b) a `not_theirs` hide covers its region
at 0.3 and its body at 0.5, or (c) it is a look-alike of a `not_trait` hide elsewhere and no `shown` hide in its own
photo covers it. Look-alikes: a `not_trait` hide of a tattoo or mehndi keeps its row's `vec` and `detail` in
`trait_hide_desc`, and rows of that kind anywhere at DINOv2 cosine 0.45 or more (the retrieval cut; verification
decides, since even one person's native crops rarely pass 0.7) that verify against it (200 verifications at most) are
written to `trait_hide_hits`; new rows are checked against every such hide when written. A
printed shirt repeats and verifies; a real tattoo that only looks similar does not. Showing a row again is limited to
its photo. Hidden rows are left out of matching, the person page and the counts, and show struck through in the box
editor. The route saves the hide (the row is hidden at once: rules a and b need no verification) and queues the photo;
the runner then finds the look-alikes, guesses the photo again, and queues open guesses citing rows that are now
hidden and up to 10 look-alikes, so a tap never waits on SIFT. Undo of a hide works as for every owner action
(section 33) and queues the photo the same way.

### Propagation, settings changes and calibration

- Naming someone (`guess.propagate`) queues up to `TRAIT_PROPAGATE_MAX` (400) photos, at any date for tattoos and
  within its window for mehndi, whose open boxes carry rows close to the person's newest 200 rows by cosine; the
  runner thread verifies them within its budget. Photos in the clothes window whose open boxes carry trait rows go to
  the runner too (`trait_match.deferred`): the serial guess job never runs SIFT.
- `trait_match.apply_settings_change`, at startup, hashes `TRAITS_REV` and the trait settings into kv `trait_rules`
  (always, ready or not). When the hash changed, open `trait` guesses are withdrawn if matching or trait-only guesses
  are now off, and the photos of the other `trait` and `clothes+trait` guesses are queued for a re-guess.
- Calibration (`trait_match.calibration_sample`, on the runner thread once the library is ready, then after 200 new
  rows and at most every 6 hours; 1,500 verifications at most): the newest 300 usable named rows of each verified kind
  against other named people (hard negatives, with their inlier counts; pairs inside one photo marked apart), the
  reference bodies the gates admit per person (strong and weak), and the nail colour distances of the same person
  within 3 days and of different people in one photo or within a day. Stored in kv `trait_calib`. The traits learner
  (section 18) turns it into stricter params only; tiers are not fitted. A hard negative raises the tattoo floor only
  when both people are named, at least one side is confirmed by the owner, and it is in another photo; 3 or more such
  pairs from 2 or more pairs of people raise the floor to one above the third largest count. (On the owner's library
  all 17 tattoo hard negatives were one person against an unnamed group that was that person split in two, and the
  earlier rule had raised the floor from 12 to 30.)

### Measured on the owner's library (2026-11-09, read-only probes)

The gate (step 0): with the five groups above and the shortened forearm, 32,501 crops of every photo people box
were scored. At p(tattoo) 0.80 or more with cloth under 0.10 and jewel under 0.15, 706 limb crops remained; a random
48 of those not belonging to the most tattooed person were mostly real tattoos (sleeves, small text, a small mark on
an upper arm), and the misses were jersey prints and a bare hand; the shared-design rule and the owner's "Not a
tattoo" hides deal with prints that repeat. Before the jewel group and the shorter forearm, a watch on a sunlit arm gave one person 40 crops at the cut;
after, 12. The 0.60 to 0.80 band (105 crops) was mostly bare arms, so it stays unclear.

Verification. On 224 px crops from the 2048 px previews SIFT did not separate people (20 or more inliers: 14 of 800
same-person pairs against 6 of 1,500 different-person pairs). On crops cut from the originals at native resolution
(median segment 421 px, 90th percentile 763 px), with different-person pairs taken from photos without the most
tattooed person (so no arm inside another's box):

| Inliers at least | Same person, other day, of 900 | Different people, of 1,500 | ln LR |
|---|---|---|---|
| 8 | 45 | 2 | 3.2 |
| 12 | 36 | 2 | 3.0 |
| 20 | 21 | 2 | 2.5 |
| 30 | 8 | 2 | 1.5 (too few negatives to say more) |

Per query, as guessing works (the top 20 references from other days by DINOv2 cosine, then SIFT): 56 of the most
tattooed person's 90 crops verified only with that person at 12 or more inliers (43 at 20 or more) and none with anyone
else; none of 66 crops of other people verified with anyone at 8 or more; 12 of 60 crops on unnamed boxes verified with
that person at 24 to 51 inliers, days apart from the references. The tiers and the admission rule above follow these
numbers.

End to end with the shipped code (rev 2, `scripts/trait_eval.py` on a snapshot of the owner's library, 2026-11-09):
the scan read 11,901 people boxes of 5,344 photos in 33 minutes (0.17 s a box with the GPU, peak memory 2.8 GB), and
the database grew by 21 MB (29.5 MB of crops). It stored 659 tattoo rows and 102 nails rows. Per query, 384 of 601
tattoo crops on boxes with a person verified only with that person at 12 or more inliers; the 13 that also verified
with someone else were nearly all one person against an unnamed group that was the same person split in two, which the
shared-design rule handles. Leave-one-out over 126 boxes (6 confirmed by the owner, 120 whose clear face put them in a
named person's group), each made faceless for the trial (the box's face deleted inside the rolled-back transaction,
so no face check can decide for the traits): with no photo in the clothes window, the tattoo path alone named 95 of
the 126 boxes, all right (Wilson 95 percent: 0.96 to 1.0), 9 of them against photos from other days; with the clothes
path as well, every guess was right and no clothes winner changed. The 40 crops the gate is surest of are all real
tattoos.

The limit of these numbers: they cover one heavily tattooed person over about 8 days, and most of the different-person
crops were the gate's own false positives rather than other people's real tattoos. They say nothing about two people
with similar real designs, or about a match across months. Nails: of 1,500 sampled boxes, 482 hands were usable, the
best gate kept 37, and only a third to a half of those were real polish (inked fingers, coloured light and held objects
were the rest), which is why nails only support.

### Not verified

- The back-of-hand sign in `trait_nails.is_back`: derived from geometry (a right hand from the back, fingers up, image
  y down, gives a positive `cross(p5 - p0, p17 - p0)`), not measured on labelled photos. The handedness output of
  `ml/hands.py` is not used.
- The nail cuts (12 from the skin, spread 15, run past the tip 0.6, the length cuts, sharpness 80) and the white
  balance on real hands.
- The skin and ink cuts of bare limbs (the YCrCb ranges, the ink rule, skin 0.60, ink under 0.01, sharpness 60) and the
  mehndi hue rule, on real photos.
- Retrieval cosine 0.45, `MEHNDI_MIN_INLIERS` 30, the absence share 0.15, the hide thresholds, pHash 6 and the burst
  rule, the shared-design factor 0.75 and ratio 2.
- Faceless child bodies are not child-gated; with `FACE_AGE_MATCHING` off there is no child protection.

## 35. Following people between photos (`app/analysis/place.py`, `place_align.py`, `place_match.py`)

Photos taken moments apart at one place show the same background, moved by however the camera moved in between.
Lining the backgrounds up tells where each box of one photo would be in the next, and a box found there is the same
person: also when the face is turned away, when the camera panned or zoomed, and when the clothes vector is under the
clothes cut. This is evidence for a guess, along with clothes, never an assignment: the owner confirms or corrects
every guess (2026-11-13). With `PLACE_ENABLED` off, guesses are exactly as before.

### Lining two photos up (`place_align`)

1. The picture (`load_gray`): the small preview (1280 px) when there is one, else the 2048 px preview, else the photo
   as the analysis saw it (`animals.open_photo`); grey, the long side reduced to `ALIGN_SIDE` (1024). The three have
   the same geometry, as boxes are normalised to the upright picture.
2. Features (`describe`): CLAHE (2.0, 8 x 8), then SIFT (`SIFT_FEATURES` 3,000, contrast 0.02), descriptors kept as
   uint8. A keypoint is flagged when it lies inside a face or person box of its photo (boxes grown by 10 percent).
   `features(key, loader)` keeps them in an LRU of `FEATURE_CACHE_BYTES` (96 MB, about 230 photos) keyed by
   (item id, version), behind a lock; each thread has its own SIFT, CLAHE and matcher objects.
3. The fit (`align`): brute-force matching with the ratio test (0.75), then a similarity (rotation, one scale, a
   shift: `cv2.estimateAffinePartial2D` with `USAC_MAGSAC`, 3 px at 1024) on the background matches only (neither end
   inside a people box). When that gives fewer than `MIN_INLIERS` (20) inliers, once more on every match: people who
   stood still are good anchors, and people who moved are outliers. That second fit must not be one person's doing:
   it counts with at least `MIN_BG_INLIERS` (6) of its inliers outside the people boxes, or with inliers on at
   least two people in both photos (`MIN_OWNER_INLIERS`, 3, each; a face counts with the person box around it), or
   when each photo shows one person only. In front of a plain wall it would otherwise lock onto one person who
   walked, report a camera move that did not happen and put everybody else in the wrong place. A fit is kept with
   20 or more inliers, a scale within 0.4 to 2.5 and inliers whose convex hull covers at least `MIN_SPREAD` (0.02)
   of the first picture.
4. The result is the 2 x 3 matrix from the first photo to the second in normalised coordinates (so a portrait and a
   landscape photo line up too), with kind `still` (shift under 2 percent of the picture, scale within 2 percent of
   1, roll under 1 degree) or `moved`.
5. Without a fit (`burst`): two photos taken within `BURST_S` (3 s) whose whole-picture vectors are at least
   `BURST_COS` (0.97) alike count as the identity, kind `burst` (a plain wall or a night shot has no keypoints, but
   nobody moved).

Not tried: a mirrored pair (a front camera's photo saved as its mirror image), and anything but photos.

### The links and the runner (`place`)

A photo takes part when it is a live, dated image outside the vault, analysed, not a Live Photo's clip, not a
screenshot or a document, and has at least one face or person box (`ELIGIBLE_SQL`). Such photos are ordered by
capture time, then the EXIF sub-second, then id, and each is paired with its next `AHEAD` (3) photos taken at most
`MAX_GAP_S` (300 s) after it.

- `place_links` holds one row per pair, the earlier photo as `a_item` (section 5). A pair that did not line up is
  stored too (kind `none`), so it is not tried again. Two photos of one duplicate group or one Live Photo are stored
  as `none` without reading pixels: the same moment is no movement. `dt_s` is the gap with the sub-seconds.
- `place_scans` marks a photo as done; `sig` is the ids of its next photos, so a photo that arrives later in the
  middle of a sequence makes its neighbours pending again (`pending_items`: no scan row, another rev or another sig;
  one window-function query over the eligible photos, 0.1 s on the owner's library).
- The runner (`kick`, `_loop`; `worker.run_in_background("place")`, its own connection, nice 19) works in rounds of
  30 s by capture time, newest first, so a photo's next photos were just described and each photo costs one read. It
  measures outside any transaction and writes one short transaction per photo, which first checks that the items
  still exist. A photo whose measuring raises stays pending; after 3 failures in a row its pairs are stored as
  `none`. The thread ends when nothing is pending, and stops for now when the process grew by 2 GB or holds 7 GB.
  It is started at startup (`place.start` from `ingest.recover_on_startup`), after each analysis and by
  housekeeping. Nothing runs with the workers off (tests call `run_pending`).
- Priority. The runner's thread is nice 19, but OpenCV does its work on a pool of its own threads, which stay at
  normal priority, and its thread count is one number for the whole process. So while a photo is measured the count
  is set to `PLACE_THREADS` (1) and put back after (`place._cv_threads`). At 1, OpenCV works on the calling thread:
  all of the lining up then runs at nice 19 on one core (measured: no time on any other thread). At 2, half of the
  work ran on pool threads at normal priority. While a photo is measured, every other OpenCV call of the process
  runs with that count too.
- After each batch of 16 photos, the photos with an open box among the photos just linked and the photos within 2
  more links of them are queued for a fresh guess on the traits runner's queue (`place.requeue_around`,
  `traits.requeue`: one queue for every re-guess). A guess walks 3 links, so a new link changes guesses that far
  away, not only in its own two photos. The same is done, 3 links around, when a photo goes to the trash or comes
  back (`ingest.soft_delete`, `ingest.restore`): a guess may rest on it.
- kv `place_ready` holds the rev (`PLACE_REV`) the whole library was last lined up with, and `place.progress` gives
  `{scanned, eligible, ready, rev, last_error, enabled}` for Settings > Learning. kv `place_rules` holds the rules the
  open position guesses were made under (`place_match.rules_sig`: the two settings and the constants below). When
  it differs at startup (`apply_settings_change`), every photo with a usable link and an open box is guessed again,
  and every photo whose open guess has a position part under any basis: a looser rule or the feature switched back
  on brings guesses back, a stricter one withdraws them, and with `PLACE_ENABLED` off a clothes guess that position
  had withdrawn comes back.

### Following boxes (`place_match`: arithmetic on the stored links and the photos' current boxes, no pixels)

A unit is one person as a photo shows them: a face (with the person box around it), or a person box without a face.
Its head point is the face's centre, or for a faceless box the top centre lowered by `HEAD_DOWN` (0.25) of the box's
width. Its width is `FACE_WIDTHS` (3.2) face widths, or `BOX_WIDTH` (0.875) of the box's width, so the two agree at
the median.

For a lined-up pair, the head points of the earlier photo are put into the later one through the link
(`match_pair`). Two units are the same person standing there when all of these hold:

- each is the other's nearest, and the distance over their mean width (the offset) is at most `RESIDUAL_MAX` (0.25),
  or `RESIDUAL_MAX_BOX` (0.35) when one of them is a faceless box, whose head point is only estimated;
- their widths, after the link's scale, differ by a factor within `WIDTH_RATIO` (0.6 to 1.67);
- the second nearest on either side is clearly worse: the offset is at most `SECOND_RATIO` (0.75) of it;
- their age bands do not differ (section 10: a child next to a parent);
- their clothes do not say no: when both have a clothes vector, its cosine is at least `CLOTHES_NO` (0.80), or
  `CLOTHES_NO_MIXED` (0.60) between a face's crop and a faceless box's crop, which are cut differently. Two boxes
  that both have a person are not held to this.

The arrangement check: when a named person of one photo sits on a different named person of the other, people moved
around, and the whole pair gives no evidence (`PairMatch.swapped`). An unnamed group sitting on anyone else is no
evidence either way for that one match (the group may be the same person). Each person found in their own place
(`kept`) is support for the others.

For an open box, the links are walked both ways from its photo, at most `HOPS` (3) photos away, through boxes nobody
is named for, to boxes that have a person (`assigned_by` user or auto, as clothes references are). The evidence for a
person is their best chain: its strength is its weakest link, minus `HOP_COST` (0.4) for each hop after the first.
Per hop the strongest path to each box is kept, so a weak direct link never hides a stronger longer chain. A chain
is void for a person whom a photo on the way shows elsewhere (the box walked through is then someone else) or whom
the owner rejected for a box on the way, and it never joins a child's face to an adult's, also not through a
faceless box in between. Photos are checked again at every guess: one that has since become a screenshot, a
document or a Live Photo's clip, or went to the trash or the vault, takes no part (`load_info`). A
link's strength (`link_llr`) is its tier by the gap, plus `CLOTHES_BONUS` (0.5) when the clothes cosine is 0.85 or
more, plus `KEPT_BONUS` (0.5) for each other person found in place (1.0 at most), minus `FACELESS_COST` (0.5) when
either end is a faceless box:

| Gap between the two photos | Strength | Share of right answers, pair by pair (two faces) |
|---|---|---|
| up to 3 s | 3.9 | 0.985 |
| up to 10 s | 3.7 | 0.985 |
| up to 60 s | 2.9 | 0.963 |
| up to 300 s | 1.2 | 0.83 |
| a burst (no fit) | 1.2 | not measured |

The strengths are log odds near the lower end of each interval of the pair-by-pair measurement below; they order the
candidates. A guess from the position alone needs `ADMIT_LLR` (2.9, about 0.95) and a lead of `LEAD_LLR` (2.0) over
the runner-up, and two caps, set from the end-to-end runs below, decide which chains can reach that:

- a link with a gap over `ALONE_S` (3 s), or without a fit (a burst), is never stronger than `FAR_LLR` (2.7). Up to
  3 s the position alone was right 0.99 of the time; from 3 to 10 s 0.93 for faces and 0.89 for faceless boxes,
  under the bar of 0.95;
- a chain of more than `ALONE_HOPS` (2) photos is never stronger than `LONG_CHAIN_LLR` (2.7): three-photo chains
  were measured on too few decisions to pass the bar.

So the position alone names someone only from a photo taken at most 3 s away, or through one photo in between with
both steps at most 3 s. Every other chain agrees with clothes and breaks ties.

Left out for a box: people already in the photo, pairs the owner rejected, and for a face anyone its own face
contradicts (the trait path's veto, section 34).

### Fusion (`place_match.fuse`, in `guess._compute_fused` after the trait step)

`guess._compute` asks for a place context when the photo has a usable link (`place_match.context`). Each box's
result, after `trait_match.fuse`, goes through `place_match.fuse` inside the same one person per photo loop. What the
traits withdrew stays withdrawn: when the trait step drops a clothes winner (a tattoo of theirs is absent, or the
traits admit someone else), the box gets no guess, and a person the traits rule out for the box is no candidate for
position either. A drain of the re-guess queue shares one cache; links written meanwhile are read again (the cache
is checked against the largest rowid and the count of `place_links` at every guess). First rule that applies:

1. A winner from clothes or traits, while the position admits another person (the admission rule, so only with
   `PLACE_ONLY_GUESSES`; with it off, position only agrees and breaks ties): no guess.
2. A winner that the position agrees with keeps its score and gains the `place` part; its basis becomes
   `clothes+place` when it was plain clothes (a basis with `trait` in it stays).
3. No winner, and a clothes tie inside `MARGIN` that the position backs for exactly one of the tied (strength
   `TIE_LLR`, 1.0, or more, and nothing for the others): that person, `clothes+place`.
4. No winner: a guess from the position alone, basis `place`, under the admission rule and `PLACE_ONLY_GUESSES`, and
   only when the clothes path ranked nobody or ranked that person. Its score is sigmoid(strength), 0.99 at most:
   near the share of right answers measured within 3 s, which is why it is shown as a percentage.

The evidence part, in a fixed key order (a guess is rewritten only when its JSON differs): `place: {ref_item_uid,
ref_face_id, ref_body_id, seconds_apart, ref_before, hops, offset, kind, others, clothes, llr}`: the box the person
was followed from, the time between the two photos and which came first, the photos walked, the offset of the box's
own first link in widths, how the chain's weakest link lined up, the other people found in place, and the clothes
cosine of a direct match. A `place` guess also has `basis`, `score`, `place_llr`, `ref_item_uid`, `ref_face_id`,
`ref_body_id` and `seconds_apart` at the top. `guess.apply_window_change` keeps `place` guesses (their limits are
seconds, not the clothes window); `record_outcome` stores a `place` outcome with `place_match.outcome_terms` and adds
`basis` and `place_llr` to a `clothes+place` outcome; the people learner leaves `place` outcomes out
(`learning.people.NOT_CLOTHES`, section 18). Nothing is learned from position yet: the outcomes are kept for it.
In the API a guess counts as a position guess when its evidence has the `place` part, whatever its basis
(`basis=place`), and the part carries `ref_crop_url` only while the photo the person was followed from is neither
in the trash nor in the vault (else no crop and no `ref_item_uid`: the card says "Photo changed").

### Measured on the owner's library (2026-11-13, a snapshot, read-only, pictures at 1024 px)

Two kinds of numbers follow. The first kind comes from `scripts/place_eval.py` and can be made again on a snapshot.
The second kind was measured once, with scripts that are not in the repository; the tier strengths, the head point
and the clothes floors rest on those.

#### With `scripts/place_eval.py` (the newest 1,734 photos lined up, the rules as shipped)

The library: 12,668 photos take part, in 28,763 pairs of a photo with one of its next 3 photos within 5 minutes.

Cost. With `PLACE_THREADS` 1 the runner took 0.59 s a photo (0.32 s a pair), so lining this library up takes about
two hours, once, on one core at nice 19; a new photo costs one description and at most 3 matches. With 4 threads it
took 0.25 s a photo (53 minutes for the library), most of it at normal priority. The machine was busy with other
work during both runs. The process held 0.35 to 0.41 GB.

How often a pair lined up, of 3,159 pairs:

| Gap | Lined up |
|---|---|
| up to 1 s | 143 of 150 (95%) |
| 1 to 3 s | 498 of 573 (87%) |
| 3 to 10 s | 661 of 961 (69%) |
| 10 to 60 s | 472 of 1,121 (42%) |
| 1 to 5 min | 64 of 354 (18%) |

The camera usually moves: 253 of the 1,836 fits were `still`, and 2 pairs counted as a burst. 308 fits came from the
second pass (people as anchors), and 125 second-pass fits were refused as one person's doing. Asking for background
inliers in every second-pass fit, which was tried first, refused 333 and cost one alignment in six.

Leave-one-out over 800 boxes that have a person, in the lined-up photos; each made open inside a rolled-back
transaction, its face evidence and the body traits off. From the position alone, clothes path off:

| Box | Right of decided | Wilson 95% | Decided | Where both people are named |
|---|---|---|---|---|
| face | 429 of 436 (0.984) | 0.967 to 0.992 | 436 of 800 (55%) | 280 of 280 |
| faceless | 331 of 336 (0.985) | 0.966 to 0.994 | 336 of 675 (50%) | 246 of 246 |

The three-second rule comes from the same run made without it, by gap:

| Box | Up to 3 s | 3 to 10 s | 10 to 60 s |
|---|---|---|---|
| face | 416 of 422 (0.986) | 137 of 148 (0.926, Wilson 0.872 to 0.958) | 26 of 26 |
| faceless | 316 of 320 (0.988) | 115 of 129 (0.891, Wilson 0.826 to 0.934) | capped already |

The bar for a guess from the position alone is 0.95 on 20 or more decisions of a tier. The 3 to 10 s tier is under
it for both kinds of box (an earlier sample of the same library had given 0.951 and 0.954, at the edge), so beyond
3 s the position never names alone, and `PLACE_ONLY_GUESSES` stays on for what is left. Every wrong name had a group
without a name on at least one side; by the faces, 1 of the 12 was the same person split over two groups.

Clothes and position together, against clothes alone, on the same boxes:

| Box | Clothes alone | With position | Among named people |
|---|---|---|---|
| face | 546 right of 580 decided (73% of boxes) | 650 of 685 (86%) | 397 of 397, then 437 of 437 |
| faceless | 468 of 490 (73%) | 548 of 578 (86%) | 390 of 390, then 417 of 418 |

By basis with both on: `clothes+place` 578 of 592 (faces) and 449 of 462 (faceless), most of them clothes ties that
the position decided; `place` alone 7 of 7 and 8 of 8. Over all boxes the share of right answers stays about where
the clothes path has it (faces 0.941 alone and 0.949 with position, faceless 0.955 and 0.948), with 13 more boxes in
100 decided.

Chains and people who are not known nearby (`--around 1`: the person is taken off every photo one link around the
box as well, so only a chain of two photos can name them, and a guess from one photo away cannot be right):

| Box | Two-photo chains right | Named from one photo away though they are not there |
|---|---|---|
| face | 19 of 19 | 10 of 800 |
| faceless | 18 of 18 | 8 of 675 |

Before the three-second rule the same run named 30 of 800 and 25 of 675 that way, and two-photo chains were right
202 times of 207 and 142 of 149. With `--around 2`: 10 of 800 and 9 of 675, and no chain of three photos names
anyone alone any more (they had been right 17 times of 18, too few decisions).

#### One-off measurements (2026-11-13, not in the repository)

How often a pair lined up among the 2,300 pairs of the whole library with a known person in both photos: 97% under
1 s, 95% up to 3 s, 84% up to 10 s, 60% up to 60 s, 28% up to 5 minutes (20, 30 or 50 inliers gave similar results up
to 60 s). Describing a photo took 0.18 s and matching a pair 0.10 s on 2 threads.

The same spot, pair by pair (whole library, a box whose person is also in the other photo: is the box in the same
spot that person?), two faces, no clothes. The tier strengths come from this table. It leaves out the boxes whose
person is not in the other photo, which is why the end-to-end runs above decide who may be named alone:

| Gap | Right of decided | Wilson 95% | Decided of chances |
|---|---|---|---|
| under 1 s | 0.988 of 1,620 | 0.981 to 0.992 | 85% |
| 1 to 3 s | 0.982 of 1,484 | 0.974 to 0.988 | 77% |
| 3 to 10 s | 0.985 of 960 | 0.976 to 0.991 | 61% |
| 10 to 60 s | 0.963 of 482 | 0.942 to 0.976 | 47% |
| 1 to 5 min | 0.833 of 72 | 0.731 to 0.902 | 23% |

With the clothes floor of 0.80: 0.991, 0.987, 0.991, 0.984 and 0.941. With the box treated as faceless (bound 0.35):
0.986, 0.980, 0.979, 0.918 and 0.758.

- Movement. Of 855 people seen in three photos in a row with gaps up to 10 s, 727 were found standing still. 39 had
  moved by more than a quarter of their width between the first two photos, and a straight-line prediction put 1 of
  them within 0.25 widths of where they were (up to 3 s: 197, 163, 9 and 1). Predicting movement is not built.
- The head point of a faceless box. In 21,668 person boxes that hold a face, the face's centre is at the middle
  across (median 0.005 of the width off) and 0.25 of the box's width below the top (half within 0.16 widths, nine
  in ten within 0.34). By the width, because the height depends on how much of the body is in the frame.
- Clothes on lined-up pairs. Two face crops of one person are under 0.80 in 6 pairs of 100, of different people in
  63. A face crop against a box crop: one person under 0.60 in 1 of 100, under 0.70 in 4, under 0.80 in 18;
  different people in 5, 24 and 72. A floor of 0.70 for that mixed pair removed 15 right answers and 1 wrong one of
  473, so it is 0.60.
- Support. Decisions with other people found in place were not more often right (faces: 260 of 273 with them, 317
  of 324 without, in the first version). The bonuses are small and decide nothing alone.
- A rule against faceless boxes cut by the left or right edge of the picture (their width is not the person's) was
  tried and dropped: it removed 160 decisions, 2 of them wrong.
- Real open boxes, which no leave-one-out contains. The lined-up photos held 884 open boxes (255 faces, 629
  faceless) in 464 photos. Position changed little there: 9 new guesses, 14 clothes guesses backed, none withdrawn,
  4 guesses from the position alone; 20 of the 23 guesses with a position part name a group without a name. In
  the first version 12 new guesses were looked at by eye: 5 were the same person, 1 was someone who had stepped
  in close to the camera where another person had stood, and 6 were too small or too dark to tell.
- Open faces are the unclear ones. Of the 5,570 open faces in dated photos, 89 are clear by the clustering's own
  rule (`faces.is_clear_face`), and each of those carries a rejection (the owner took it off a person or said no
  to a guess). A clear face that matches nobody starts a person of its own and is no longer open.

### Limits

- A box of someone who is not known in the photos around it: a stranger who stands where a known person stood up
  to 3 s before gets that person as a guess. With the person taken off the photos around, that happened to 10 of 800
  faces and 8 of 675 faceless boxes. A guard on the face cannot help: the open faces are the unclear ones (above),
  and an unclear face says little about who it is not; a faceless box has no face at all. The clothes floor is what
  stands in the way, and the real share of right answers will show in the outcomes of confirmed and rejected
  guesses (`basis: place`).
- Groups without a name. Guesses can name one, as clothes guesses can, and most guesses with a position part on
  real open boxes do. Over all boxes the leave-one-out counts one person split over two groups as a wrong name.
- People who walk, and gaps over 3 s: the position only supports there.
- People who swap places: the arrangement check sees it only between two named people who are both in both photos.
- Someone close to the camera whose box the frame cuts: the width check that keeps a near person from a far one
  means nothing for a cut box.
- A person alone in front of a plain wall: the fit may follow them and not the camera. They map onto themselves, so
  nobody is named wrongly, but it is no evidence that the camera stood still.
- Two cameras at one event have different clocks, so their photos interleave wrongly.
- Photos only: videos track faces inside a clip already, and have no person boxes.
- The numbers cover the newest 1,734 photos of the library; the pair table covers the whole library.

### Not verified

- The strength of the tiers beyond 3 s (they only order candidates now) and the burst rule (2 pairs).
- `CLOTHES_BONUS`, `KEPT_BONUS`, `HOP_COST`, `LEAD_LLR`, `TIE_LLR`, `MIN_BG_INLIERS` and `MIN_OWNER_INLIERS` are not
  fitted.
- A faceless box as the reference (a back view the owner named): the runs made the open box faceless, never the
  reference.
- Mirrored pairs (a front camera's photo saved as its mirror image) are not tried.
- The runner inside the running server (the timings are from the script), and how much the process-wide thread
  count slows other OpenCV work while photos are lined up.

## 36. 3D heads of a person, one per look (`gen3d/`, `app/analysis/looks.py`, `app/analysis/model3d.py`, `app/api/model3d.py`)

A person's page has a "3D" tab. Its "Generate" button builds one 3D head for each look found in that person's
photos, on the server, with a local model. A look is how someone looked for a while: an age band, a hair length and
an amount of facial hair. The viewer shows one head at a time; sliders pick the look (Age, Hair, Facial hair), turn
and tilt the head, zoom, and move the light. Nothing is read, loaded or generated before the button is used:
`GET people/{id}/model3d` only reads what is stored, and the generator's Python and its model are never inside the
server's process.

### Two programs

- The app finds the looks with the models it already has (SigLIP, the face detector, FairFace ages, DWPose; numpy and
  onnxruntime only), cuts a head-and-shoulders picture for each, runs the generator, stores what it made and serves
  it.
- The generator is the `gen3d` package at the repo root: a separate program with its own Python environment, started
  as `python -m gen3d --job JOB.json` for the length of one job. The app never imports it. PyTorch and the model's
  code live in that environment only: `GEN3D_DIR` (default `MODELS_DIR/gen3d`) holds `venv/` (its Python), `weights/`
  (the model files, with `weights/hf` as `HF_HOME` and `weights/u2net` as `U2NET_HOME`) and `src/` (the model's
  code); `scripts/setup_3d.sh` makes it. So the server's memory and the GPU are free the rest of the time, and a
  generator that crashes or runs out of memory cannot take the server with it.

`gen3d/`: `__main__.py` (the job runner and `--check`), `backends/__init__.py` (the interface and the registry
`{"fake", "triposr"}`; `ORDER = ("triposr",)` is what "auto" tries), `backends/fake.py` (a stand-in that needs no
model: an egg-shaped head on a neck, 1,900 triangles, coloured from the picture, numpy and Pillow only),
`backends/triposr.py` and `prep.py` (the real backend and its background cut), `mesh.py` and `glb.py`.

### The contract between them

Job file: `{"out_dir", "weights_dir", "generator": "auto" | a backend's name, "device": "auto", "options":
{"detail": 1..3, "seed": 0}, "looks": [{"key", "image"}]}`; `image` is a square picture of the head and shoulders
with the face in the middle. Output: `OUT/<key>.glb` for every look that worked and `OUT/result.json` `{"generator":
{"name", "version"}, "looks": [{"key", "ok", "mesh", "vertices", "faces", "seconds", "error"}]}`; on stdout, one
line per event and flushed at once, `progress: N/M` and `look: <key> ok|error <seconds>`. A look that fails does not
stop the others. Exit codes: 0 when at least one head was built, 3 when none was, 2 for a bad job file, 4 when the
backend cannot load, 5 when it was told to stop (SIGTERM) or its parent went away (`PR_SET_PDEATHSIG` and a watch
on the parent's pid). The last line on stderr is the reason, and the owner reads it in the tab. `python -m gen3d
--check --weights-dir DIR` prints one JSON line `{"backends": {name: installed}, "auto", "device", "torch"}`. It
only looks at files: `device` and `torch` are null unless `--device` is added, which imports PyTorch and takes
seconds.

A backend's module offers `BACKEND`, or a `create()` that returns it, and must be cheap to import (no PyTorch at
import time). The backend exposes `name`, `version`, `available(weights_dir)` (cheap as well: the check runs for
status reads), `load(weights_dir, device, options)`, `generate(image, options) -> Mesh` and `close()`. `generate`
gets an RGB picture, or RGBA with the background cut out when `gen3d.prep.prepare(image, options)` ran first (it runs
for every backend but the stand-in, when that module is there); `options` are the job's plus `out_dir` and the look's
`key`.
The mesh must have +Y up and +Z towards the viewer, at any scale; its colours, when it has any, are one sRGB byte
triple per vertex. `mesh.finish` then checks it, drops loose parts with fewer than 2 percent of the largest part's
triangles, centres it, scales it into a box one unit across, turns an inside-out mesh the right way round and
computes the normals. `glb.write` stores binary glTF 2.0: positions and normals as float32, `COLOR_0` as four
normalised 16-bit values in linear light (glTF vertex colours are linear, and 8 linear bits would band in dark hair),
uint32 indices and one rough, double-sided material. Any glTF viewer shows the file as the app does.

### The sandbox, and why

The generator runs model code written by others on the owner's face photos. So the real generator runs inside
`bwrap`, in a system made of exactly these parts (`model3d.sandbox_prefix`):

- read-only: `/usr` and the loader's files from `/etc` (`ld.so.cache`, `ld.so.conf`, `ld.so.conf.d`,
  `alternatives`), with `/bin`, `/lib` and their like as the links or folders they are on the host; `/sys`; the
  Python environment that runs it, under the name it has and where it really is, and the installation its
  interpreter comes from, under every name on the link's way (uv links an environment to an installation named by
  its minor version, and that name is a link to the exact one); `GEN3D_DIR` (weights and model code), as named and
  where it really is; this repository's `gen3d` package;
- its own `/proc`, a new `/dev` (null, zero, random, a terminal, shared memory) with the machine's NVIDIA device
  nodes added, an empty `/tmp` and an empty `/run`;
- read and write: the job's work folder, and nothing else that lasts. `/tmp` and `/dev/shm` are memory and go with
  it; the folders that only lead to the parts above are read-only too;
- namespaces of its own for users, process ids, IPC, the host name and the network, where nothing but its own
  loopback exists; it dies with its parent;
- an environment of its own (`--clearenv`): `PATH`, `HOME=/tmp`, `LANG`, `PYTHONUNBUFFERED`, the cache folders
  under `/tmp`, `HF_HOME`, `U2NET_HOME`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and from the server only
  `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH` and `GEN3D_THREADS`, when they are set.

What that hides: the home folder with its credentials, `DATA_DIR` and `ORIGINALS_DIR` (the library, the vault, the
database), the rest of the repository and its `.env`, every other model, the rest of `/etc`, the server's
environment, the network, the other processes, and every socket of the machine. The rootless Docker socket and the
session bus live in `/run` and in the home folder, and a read-only view of them is not enough, because connecting
to a socket is not a write. The first version of this sandbox showed the whole machine read-only and passed the
server's environment on; a review connected to the Docker socket from inside it, which would have let that code
start a container with a network and the home folder. What it does not hide: the system's programs and libraries
(read-only), what `/proc` and `/sys` say about the machine (processor, memory, kernel, devices), the GPU through
its device nodes (the driver is a large surface, and the card is shared with the server's own models), the
generator's own weights and code, and the pictures of this one job's looks, which are the point. It is no limit on
processor time or memory: those are the watch below. A flaw in the kernel or in the GPU driver is outside what it
can stop.

`GEN3D_SANDBOX`: `auto` (the default: the sandbox for the real generator when `bwrap` is installed; the stand-in is
this repository's own code and runs without it), `on` (always; without `bwrap` nothing is generated and the tab says
why) or `off`. The check that asks what is installed runs the generator's Python too, so it runs the same way, with
an empty folder of its own as the only place to write. Whatever the setting: the child never gets the server's
environment (the same short list, with its home and caches inside the work folder when there is no sandbox, since
a machine's `/tmp` may be shared); it is its own process group with `oom_score_adj` 1000, so a memory limit takes it
and not the server; once a second the resident memory of its whole process tree is added up, and the child is
stopped above `GEN3D_MAX_RSS_MB` (5000), when one head takes longer than `GEN3D_LOOK_TIMEOUT_S` (600), on Stop or
Delete, when its person is deleted or merged away, and when the server exits. One job runs at a time for the whole
app, as there is one GPU.

### Finding the looks (`looks.py`)

Source faces: the person's faces with a weight of 0.6 or more (section 10) and 120 px or more on the shorter side,
on items that are in the library (not in the trash, not in the vault). At most 240 are read per plan, whoever
the person is, spread over the years: every year keeps at least 12, and a year's child-band faces are a group of
their own beside that year's other faces, so that childhood prints photographed again in a busy year are not
crowded out. For each face:

- from a square 2.6 face sizes wide around it in the 2048 px preview, which is what the generator will see: the hair
  as a level from 0 to 1 (zero-shot with SigLIP over a ladder of prompts from "a bald head" to "hair below the
  shoulders"; the reading is the probability-weighted level), a covered head and a hand or an object at the face the
  same way, and the head's turn, nod and tilt from the face detector's five points on that crop;
- from the stored face crop, where the tight view shows it better: the facial hair as a level (a ladder from
  "clean-shaven" to "a long beard"), sunglasses, and the light on the face;
- from the pose of the person box around the face (DWPose, section 34): a wrist at the face, or a wrist or an elbow
  up beside the head;
- the age: FairFace's expected age, and for photos with a camera time that are not forwards the year minus the
  person's birth year, which is the median of (year minus expected age) over at least 10 such adult faces. A face
  read as 13 or younger keeps its read age, since a print photographed again carries today's date.

A photo counts against itself as an input (its `faults`) when another face of the photo is inside the square
(`other_face`), the square leaves the picture by more than 15 percent (`cut_off`), the head is turned or tilted far
(`turned`), a hand or an object is at the face (`covered_face`), an arm is up by the head (`raised_hand`), the eyes
are behind sunglasses, the head is covered, or the light on the face is coloured or very dark (`odd_light`). Looks
are made from photos without a fault, because a turned head or a second person in the crop also bends the hair and
facial hair readings. The photos of one day share that day's median levels, since hair does not change within a day.

The bins are (age band, facial hair, hair). Age bands: under 10, 10 to 17, 18 to 24, 25 to 34, 35 to 44, 45 to 59,
60 and over. Facial hair: "little or no facial hair" under 0.44, "stubble or a short beard" under 0.53, "a beard"
above. Hair: "shaved or very short hair" under 0.30, "short hair" under 0.52, "medium hair" under 0.76, "long hair"
above. A bin is a look when it holds 3 photos without a fault (1 in the youngest band, where photos are few). The
fullest bin of every age band is taken first, then the fullest of the rest, up to the "Looks" option (1 to 12,
default 6). When no bin has enough such photos, the one fullest bin of all the photos that can be placed gives a
single look, so a person with few photos still gets a head. A look's input is its best photo (face weight, size,
sharpness, how frontal it is, shoulders in view, light), with three runners-up in case its pixels cannot be read.
The picture is cut from the original at 1024 px: a square "Framing" face sizes wide (1.8 to 3.2, default 2.6).
Without SigLIP the hair and facial hair are unknown and the bins are age bands only.

### The job (`model3d.py`)

`start` refuses when a job is running (`Busy`), when the generator is not installed (`GeneratorMissing`) and when
the person has fewer than 3 source faces (`NotEnoughPhotos`). The job then runs in a background thread: "Finding
looks" (the plan), "Preparing photos" (the pictures, into `people3d/<id>/work/`), "Building heads" (the child; its
`progress:` lines move the bar) and "Saving". The result goes to `DATA_DIR/people3d/<person id>/<stamp>/`:
`model.json` (the manifest: generator, options, birth year, and per look its key, label, age, levels and labels,
photo count, source face with its box and item, faults, file names, bytes, vertices and faces), `<key>.glb`, and
`<key>.jpg` (the input picture at 512 px). Writing `current` with the new stamp makes it the model in use; the
older folders are removed after that.

How a job ends early. Stop and Delete take effect within about a second in every step: between two photos while
looks are found (the reading reports before the models load and after every photo, and the job looks there),
between two looks while the pictures are cut, and in the child's watch. Until the job is gone the status says
`stopping`. Stop keeps the model in use and leaves nothing new. Delete takes the model out of use at once (it is not
listed or served from that moment), and its folder goes when the job has ended. A person who is deleted or merged
into someone else takes their job along: no route can reach a job whose person is gone, so the job looks for its
person itself, once a second and before a head is built or saved, and stops. A run that fails leaves `error.json`
and the older model as it was. A child that was stopped by the memory or the time limit, or that broke part of the
way, may have finished some heads: those become the new model (a mesh is written whole before it is reported), and
`error.json` says why the rest is missing.

`status` reports `none`, `running` (with the step and its counts), `done` or `error`, whether a model can be made
and why not, and the looks. A look is neither listed nor served once its source photo is deleted, in the trash or
in the vault, or once its source face is no longer this person's: moved to someone else, unassigned or removed. A
photo that was processed again has new face rows; its look stays when one of this person's faces there overlaps
the look's face by half or more. Whether the generator is installed is asked from it with `--check`, which only
looks at files: measured with the real environment inside the sandbox, 0.4 s the first time and 0.13 s after. The
answer is remembered for 5 minutes (30 seconds while it is missing). A status read waits for it once; after that an
old answer is given at once and renewed in the background, and `start` asks again itself. The stand-in generator
runs with the server's own Python however it is asked for (`GEN3D_FAKE`, `GEN3D_GENERATOR=fake`).

A deleted or merged person's model stops being served at once, because every route asks for the person first, and
`model3d.sweep` (housekeeping) removes the folder later; that delay also leaves the model in place for an undone
delete. While no job runs, the sweep also removes what an interrupted run left behind for everyone else: a work
folder (a server that stops in the middle of a job never gets to remove it) and model folders that `current` does
not name.

### Measured (2026-11-13, a snapshot of the owner's library, one person)

- 3,907 source faces over five years. 237 were read in 196 s on a CPU (about 0.8 s a face: two SigLIP passes, a
  face detection and, for 234 of them, a pose), with a peak of 1.5 GB.
- Faults among the 237: another face in the square 65, the square leaving the picture 57, a turned head 50,
  something at the face 48, odd light 37, a raised hand 22, sunglasses 4. 38 had none.
- Three looks came out, all in one age band: short hair with a beard (3 photos), medium hair with stubble (27) and
  long hair with stubble (4). The chosen photos, looked at by eye, are frontal, evenly lit and have nothing at the
  face.
- Against 320 clear faces of 185 other people: the facial hair ladder on the face crop puts 87 percent of them under
  that bearded person's 10th percentile (70 percent on the head crop, 53 to 75 percent for two-way and three-way
  prompt sets). Short noun prompts ("a hat", "sunglasses") separate where sentences did not: the first covered-head
  set flagged 52 of 397 bare-headed photos, the final one none of them and 17 of the 320 others, all with a hat, cap,
  turban, helmet or hood. The zero-shot "at the face" flag misses raised arms and a hand under the chin, which is
  what the pose check is for.
- The generator's picture, on 60 source faces (46 JPEG, 7 HEIC, 6 DNG, 1 PNG): every original was read, the square
  cut from it showed the same thing as the square cut from the preview (correlation of 64 px greys 0.997 or more),
  and it was 1,229 to 5,024 source pixels wide (median 2,484), so the 1024 px picture never enlarges anything.
- The generator (TripoSR on an RTX 2080, four runs on private test pictures, each the backend and the background
  cut alone, before the sandbox was rebuilt): 3 to 6 s a head at detail 2 with 2.4 GB of GPU memory, 3.1 GB at
  detail 3, 9 to 13 s to load, the background cut about 1 s on the CPU, at most 3.7 GB resident.

### Limits

- One head per look, each from one photo: it is the model's reading of that photo, not a scan. The back of the head
  is a guess, and the expression is the photo's.
- Hair and facial hair levels are zero-shot readings and can be wrong; the cuts between the bins were set on one
  person against 185 others. Ages are estimates.
- The person boxes are used to frame the picture and for the raised-arm check only: no body is built.
- On a CPU the looks of a person with many photos take about three minutes to find. The generator's speed without a
  GPU was not measured.

### Not verified

- The viewer in a real browser: the pictures themselves, the copy of the WebGL canvas that the cross-fade uses, and
  touch gestures. Its logic, and three.js reading the written files, are checked in node (section 16).
- The cuts for sunglasses, a covered head and odd light on libraries other than the owner's.
- A whole generation by the real generator started by the app. Checked one by one: the backend and the cut on real
  pictures (above), the app's job with the stand-in through the same command line and sandbox (the tests), the
  mesh finishing and the writer on a mesh of the real size (136,000 triangles: 0.1 s, a 3.7 MB file), and, inside
  the rebuilt sandbox, the plain check and a probe (PyTorch sees the card and computes on it; no home folder, no
  Docker socket, no network; only the work folder takes a write). The first Generate on a server is the first run
  of all of it together; a failure shows as the error in the 3D tab.
