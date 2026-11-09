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
`derivatives_tidy`), from housekeeping.

## 4. Authentication and sessions

- One account row (`account`, id 1). `GET /api/auth/state` reports `setup_required` until it exists.
  `POST /api/auth/setup` creates it (username 1..64, password 10..200 chars, argon2id), only while no
  account exists (409 after).
- Password login: `POST /api/auth/login`. Rate limit: 10 failures per 10 minutes per IP (in memory).
- Passkeys (`webauthn` 3.x): registration from Settings (`resident_key=required`, `user_verification=preferred`),
  login with a discoverable credential (`POST /api/auth/passkeys/login/options` returns options and a
  `challenge_id`; the challenge is stored in `challenges` with a 5-minute expiry; `verify` creates the
  session). `RP_ID` and `ORIGIN` come from env; when unset they are derived from the request
  (`Host` without port; scheme from `X-Forwarded-Proto`). The dev server manifest sets them from `{host}`.
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
- `bodies(id, item_id, face_id FK SET NULL, person_id, assigned_by, x,y,w,h, det_score, torso_embedding BLOB, guess_person_id, guess_score, guess_evidence, frame_time_s, created_at)`.
- `persons(id, name, name_key, cover_face_id, created_at, updated_at)`.
- `item_tags(item_id, person_id, source CHECK ('user','guess'), relation CHECK ('in','related'), created_at,
  PK(item_id, person_id))`; `relation` ('in': they are in the item; 'related': the item is about them without them
  in it, section 10) came through `MIGRATED_COLUMNS`. A tag names a person or a pet (section 22).
- `guess_rejections(kind CHECK ('face','body'), ref_id, person_id, created_at, PK(kind, ref_id, person_id))`.
- `merge_dismissals(person_a, person_b)`.
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
| `persons` | `yolox/yolox_s.onnx` | Megvii YOLOX release 0.1.1rc0 (35.9 MB) | Apache 2.0 |
| `ocr` | rapidocr's bundled models (`PP-OCRv6_det_small`, `PP-OCRv6_rec_small`, `ch_ppocr_mobile_v2.0_cls`; installed with the package, nothing downloaded at run time) | RapidAI | Apache 2.0 |
| `ocr_devanagari` | `ocr/devanagari_PP-OCRv5_rec_mobile.onnx` (7.9 MB, Hindi and Marathi recogniser, character list in the ONNX metadata) and `fonts/NotoSansDevanagari-Regular.ttf` (page renders and test fixtures) | RapidOCR model zoo (modelscope, release v3.10.0); Noto fonts | Apache 2.0; SIL OFL 1.1 |

Every engine is lazy (loaded on first use), lives once per process, and falls back to CPU when the
CUDA provider fails to initialise (logged once). Without a model file the dependent stage marks
its part as skipped (`quality_scores.skipped`, `category_scores.skipped`) and the item still completes.

GPU memory: onnxruntime's per-session arena grows with every new input shape and never shrinks on its own (a
server in ordinary use was measured at 7.7 GB of the card's 8 GB). `runtime.session` therefore returns CUDA
sessions wrapped so that every `run()` carries the `memory.enable_memory_arena_shrinkage` option
(`runtime.wrap_session`, `GPU_ARENA_SHRINK`, default on); `app.ml.ocr` wraps rapidocr's sessions the same way.

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
person to a name that exists). The similarity the owner accepted is measured first (`merge_similarity`: for each
merged person, the mean over its faces of the top-3 mean to the target's faces; the least of those), the merge
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
- Score `= 0.5 * clothes + 0.3 * scene + 0.2 * face_or_0.5`; the best candidate wins when it leads the runner-up
  by `>= 0.05`; the result is stored as `guess_person_id`, `guess_score`, `guess_evidence` (`{clothes, scene,
  face, ref_face_id, ref_item_uid, same_day, minutes_apart, gps_m}`) with `assigned_by='guess'`.
- Confirming sets `person_id`, `assigned_by='user'` and enqueues `guess` for that person; rejecting writes
  `guess_rejections` and clears the guess. A confirmed body without a face adds an `item_tags` row with
  `source='guess'`, removed again when that body is moved or deleted.
- Propagation (`guess` job) runs the same scoring from the newly tagged face outward over the window.
- A guess card for a box in a video shows the frame at the box's time (`/files/frame`, preview size) with the
  box outlined, says "at 0:42", and opens the video there; the poster frame is a different moment.
- A changed `GUESS_WINDOW_H` takes effect at the next start: `guess.apply_window_change` (from
  `ingest.recover_on_startup`) withdraws every open guess whose reference photo lies outside the window now and
  queues a `guess` job for its item; `kv guess_window_h` records the value in force.
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
  ("Paras, 4 tracks"; "Not tagged, 3 tracks"), seeks to the group's first track, and its edit control opens
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
- The row keeps the owner's box with `source 'user'` (drawn, `FaceOut.source`), `assigned_by 'user'` for the person
  (a new name makes them),
  the found face's vector, score, size, quality, clarity and sharpness, the head-and-torso vector with SigLIP, and
  its crop; on a video `frame_time_s = t` and spans `t +- 0.5 s`. On a photo a stored face under it (IoU `>= 0.4`
  with the found face) is assigned instead, as a hand assignment is. A new face inside a person box the detector
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
not Chikki", "+3 look-alike faces followed") and refreshes the library, the
person pages, the people list and open item details; a rename keeps the sheet open and reloads its list.
`BulkBar` (library, search, albums, vault) gets the button; the person and pet pages' Items and Related tabs
get a Select button, selectable tiles and a selection bar with the button (`from` = that person or pet).

Tests (`tests/test_people_bulk.py`): the list with faces, bodies, pet boxes and both relations; a move to an
existing person and to a new name, with look-alikes following and the tags re-pointed (relation kept, `in`
winning); a pet moved to a pet and refused to a person or another species; remove; a bulk tag of both kinds
and relations with `species_needed`; vault items with the vault locked; the same person as target.

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
`GET review/misclicks` -> `ItemPage` (with `misclick_reasons`); `GET review/forwards` -> `ItemPage` (the forwards set apart, section 29); `GET review/guesses?person=` -> `GuessPage` (served by the people router);
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
`BodyOut = {id, x, y, w, h, det_score, face_id, person, assigned_by, guess, url}`;
`GuessOut = {person: PersonRef, score, evidence}`; `PersonRef = {id, name, cover_url, kind, relation}` (`relation`
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

## 15. Configuration (env)

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `<repo>/data` | database and files |
| `MODELS_DIR` | `<repo>/models` | all models |
| `PORT`, `HOST` | `41000`, `127.0.0.1` | listen address |
| `SITE_TITLE` | `Personal Content Cloud` | |
| `RP_ID`, `ORIGIN`, `RP_NAME` | derived from the request / site title | WebAuthn relying party |
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

## 16. Testing

pytest with a fresh `DATA_DIR` per test, `WORKERS_ENABLED=false`, `worker.drain()` to run jobs,
`TestClient` with a logged-in session fixture (`user_client`, created through `/api/auth/setup` and
`/api/auth/login` with the CSRF header). Model-backed tests skip when the model files are missing
(`ml.models.present(key)`); pure modules (parsers, quality metrics on synthetic images, dedup, cursors,
auth, uploads, FTS) run everywhere. Fixtures: the hub's public-domain face photos, synthetic renders of
each document type (text drawn with Pillow, valid check digits), a two-page PDF with a text layer and a
scanned-style PDF, generated blurry and dark images, a tiny ffmpeg-generated video. The frontend
builds with `tsc` and is checked in a real browser with `ws-check`.

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
  a rejection wins over any loosening). `cluster._score` uses the learned threshold in place of the
  good-face default (a weaker face keeps its extra margin on top), so a person the owner confirmed or
  merged often is matched more readily and one they corrected is matched more carefully. Every face counts by
  its face weight (section 10): the centroid and the 10th percentile are weighted, `MIN_USER_FACES` counts the
  total weight, faces of weight 0 are left out, and a rejected face raises the threshold by its weight's share
  of the way to its similarity + 0.02. `guess_probability(features) -> float | None`: a logistic
  regression on [clothes, scene_term, face_term, same_day, has_gps] fitted on `guess` events (needs 30
  events with both outcomes); `guess.guess_score` returns it instead of the fixed 0.5/0.3/0.2 mix.
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
  newer than the last training), `reset()`.

API (`/api/system/learning`, session required, CSRF on writes): `GET` ->
`LearningStatus{enabled, trained_at, pending, learners{doc_type{examples, labels[]}, category{examples,
labels[]}, vault{examples, threshold}, misclick{examples, thresholds{}}, people{persons, events,
weights{} | null}}, events{category, vault, misclick, doc_type, guess}}`; `PATCH {enabled}` (kv flag);
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
