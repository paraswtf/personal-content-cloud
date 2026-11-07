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
      guess.py         person guessing from same day + same scene + same clothes
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

`DATA_DIR` (default `<repo>/data`). Shard `sh = uid[:2]`. Versions as in the hub (an item's
`version` only changes when the user replaces a file; near-duplicates never replace anything).

| Path | Content |
|---|---|
| `cloud.db` | SQLite database |
| `secret.key` | 32 random bytes, mode 0600 (cookie signing, challenge HMAC) |
| `.instance.lock` | flock |
| `originals/<sh>/<uid>.v<n><ext>` | untouched upload bytes |
| `thumbs/<sh>/<uid>.v<n>.webp` | 480 px thumbnail (page 1 for documents, poster for videos) |
| `previews/<sh>/<uid>.v<n>.webp`, `.sm.webp` | 2048 / 1280 px previews (images, video posters, document page 1) |
| `pages/<sh>/<uid>.v<n>/<page>.webp` | document page renders, 1600 px long side (PDF pages; office documents get text-rendered pages) |
| `videos/<sh>/<uid>.v<n>.mp4` | 720p H.264 streaming copy |
| `faces/<id//1000>/<id>.webp` | face crops 192 px |
| `bodies/<id//1000>/<id>.webp` | person crops 256 px long side |
| `docphoto/<sh>/<uid>.v<n>.webp` | photo cropped out of an ID document |
| `staging/<uid>.part` | uploads in progress |
| `failed/<uid>.part` | uploads whose ingest failed for good; deleted after `TRASH_DAYS`; Settings > System > Retry failed uploads queues them again |
| `wallet/` | `cert.pem`, `key.pem`, `wwdr.pem` placed by the owner (never written by the app) |
| `tmp/` | wiped at startup |

Staging, originals, tmp share one filesystem (hard link placement). The orphan sweeper walks
originals, thumbs, previews, pages, videos, faces, bodies, docphoto and deletes files older than an
hour that no row references (names parsed by regex; `_is_referenced` guards every delete).

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
  `misclick` INT, `misclick_reasons` JSON, `misclick_source`, `quality_scores` JSON); duplicates
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
- `item_tags(item_id, person_id, source CHECK ('user','guess'), created_at, PK(item_id, person_id))`.
- `guess_rejections(kind CHECK ('face','body'), ref_id, person_id, created_at, PK(kind, ref_id, person_id))`.
- `merge_dismissals(person_a, person_b)`.
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

Jobs, queues and priorities: `ingest` (main, 10), `analyze` (main, 50), `video_preview` (video, 50),
`text` (text, 60), `guess` (main, 70), `recluster` (main, 90). The queues are serial; `main` runs
all GPU work except OCR so two heavy models never contend. Handlers are idempotent and re-check state.

1. `ingest` (`ingest.process_upload`): probe (`media.probe.probe` extended with `media.documents.probe_document`
   for PDF, DOCX, XLSX, PPTX, TXT, MD, CSV and `media.raw` for camera RAW), sha256, fingerprint, near-duplicate
   search (`media.dedup.find_similar`: same thresholds as the hub; the result links, never replaces: the new item
   gets `dup_group_id` of the best item in the group, and the group's best item is re-chosen by the hub's quality
   ranking), place original, derivatives (thumb + previews; page 1 render for documents; poster for videos),
   insert item (`analysis_status='pending'`, `text_status='pending'` for `pdf` and `document`, else `'none'`),
   enqueue `analyze`, plus `video_preview` when `needs_web_preview`, plus `text` for documents.
2. `analyze` (`analysis.pipeline.analyze_item`): load the analysis image (oriented original at most 2048 px; page 1
   render for documents; sampled frames for videos: 1 per second, at most 150); SigLIP embeddings (whole image and
   frames) stored in `item_embeddings`; category scores and labels; NSFW score; quality scores and mis-click
   reasons (not for documents); faces (SCRFD multi-scale, ArcFace) and video face tracks; person boxes (YOLOX-S
   class 0) matched to faces by containment; torso crops (3.2 face widths wide, from y-0.25h to y+4h; for a body
   without a face the upper 60 percent of the box) embedded with SigLIP into `torso_embedding`; write faces and
   bodies with crops; `cluster.assign_new_faces`; `guess.guess_item` for still-unassigned faces and bodies;
   decide category and nsfw; `analysis_status='done'`; enqueue `text` when `doc_likeness >= DOC_LIKENESS_MIN`
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

Guessing (`analysis/guess.py`). For an unassigned face or body D in item I:
- Candidates: faces (and bodies) of named or unnamed persons with `assigned_by in ('user','auto')` in items J
  with `|taken_at(I) - taken_at(J)| <= GUESS_WINDOW_H` (12 h) or the same local calendar date (`LOCAL_TIMEZONE`),
  excluding persons already present in I and pairs in `guess_rejections`.
- Scene: `cos(emb(I), emb(J)) >= GUESS_SCENE_SIM` (0.80), or both items have GPS within `GUESS_GPS_M` (250 m).
- Clothes: `cos(torso(D), torso(F)) >= GUESS_CLOTHES_SIM` (0.85).
- Face: when D has a face embedding, it must not match any other person at `>= 0.40`; a weak match with the
  candidate (`>= 0.20`) raises the score.
- Score `= 0.5 * clothes + 0.3 * scene + 0.2 * face_or_0.5`; the best candidate wins when it leads the runner-up
  by `>= 0.05`; the result is stored as `guess_person_id`, `guess_score`, `guess_evidence` (`{clothes, scene,
  face, ref_face_id, ref_item_uid, same_day, gps_m}`) with `assigned_by='guess'`.
- Confirming sets `person_id`, `assigned_by='user'` and enqueues `guess` for that person; rejecting writes
  `guess_rejections` and clears the guess. A confirmed body without a face adds an `item_tags` row with
  `source='guess'`, removed again when that body is moved or deleted.
- Propagation (`guess` job) runs the same scoring from the newly tagged face outward over the window.
- A person is guessed for at most one box per photo (the best score wins, the other boxes are scored again
  without that person; videos are exempt), and confirming skips a box whose person is already in the item.
  Rejections follow a face across a reprocess (carried to the new face by position). Vault items are never
  references, and NSFW items get no guesses. Reclustering and merging refresh the FTS people column.
- People routes and the vault: with the vault locked, tagging or untagging an NSFW item and box actions on
  NSFW boxes answer 403 `vault_locked`; a person seen only in vault items answers 404 by id outside the vault.

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
| `id` | `aadhaar` (12 digits, Verhoeff), `pan` (`[A-Z]{5}[0-9]{4}[A-Z]`), `passport` (MRZ TD3, check digits), `driving_licence`, `voter_id` (EPIC `[A-Z]{3}[0-9]{7}`), `vehicle_rc` | name, number, dob, gender, address, issue/expiry, father/spouse, issuing authority, photo |
| `money` | `bank_statement`, `invoice`, `receipt`, `salary_slip` | account/IFSC, period, opening/closing balance, invoice number, GSTIN, total, parties, net pay |
| `education_work` | `marksheet`, `degree`, `certificate`, `offer_letter` | institution, course, roll number, grade, date, employer, designation, CTC |
| `health` | `insurance_policy`, `prescription`, `lab_report`, `vaccination` | insurer, policy number, sum insured, validity, doctor, patient, tests, vaccine, beneficiary id |
| `pass` | `event_pass`, `boarding_pass` | event, venue, date, time, seat/section/gate, booking id, barcode; flight, pnr, from/to |
| `other` | `other` | title, date (OCR text only) |

Parsers return `ParsedDocument(doc_type, confidence, fields, title, doc_date, issuer, person_name)`.
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
`favorite`, `misclicks=exclude|include|only` (default exclude), `dups=collapse|all` (default collapse), `vault`,
`sort=taken|added` (default taken desc), `cursor`, `limit` 1..200 default 60) -> `ItemPage{items, next_cursor, total}`;
`GET items/{uid}` -> `ItemDetail`; `PATCH items/{uid} {category?, nsfw?, misclick?, favorite?, filename?}`;
`DELETE items/{uid}` (trash); `POST items/{uid}/restore`; `POST items/bulk {uids, action: delete|restore|keep|category|nsfw|favorite|unfavorite, value?}`;
`POST items/{uid}/reprocess {stages?: [analyze, text, preview]}`; `GET items/{uid}/pages`; `GET items/{uid}/text`;
`GET trash` -> `ItemPage`; `POST trash/empty`; `DELETE trash/{uid}` (purge one);
`GET review/misclicks` -> `ItemPage` (with `misclick_reasons`); `GET review/guesses?person=` -> `GuessPage` (served by the people router);
`GET duplicates` -> `DupGroupPage{groups:[{best: ItemCard, copies: [ItemCard]}]}`; `POST duplicates/{uid}/best`;
`GET stats` -> `Stats{items, by_kind, by_category, people, documents, misclicks, guesses, trash, queue{queued, running, failed, by_queue}, disk{free_bytes, total_bytes, originals_bytes}, analysis{pending, done, error}, text{pending, done}}`.
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
`GuessOut = {person: PersonRef, score, evidence}`; `PersonRef = {id, name, cover_url}`.

People (`/api`): `GET people?q&filter=all|named|unnamed&sort=items|name|recent&cursor&limit` -> `PersonPage`;
`POST people {name, face_ids?}`; `GET|PATCH|DELETE people/{id}` (`PATCH {name?, cover_face_id?}`; 422 when the
cover face's photo is in the trash or the vault; `PersonRef.cover_url` skips such a chosen cover and uses the
automatic one);
`POST people/merge {target_id, source_ids}`; `GET people/{id}/faces`; `GET people/{id}/items` (faces, bodies and tags);
`GET faces/unassigned`; `POST faces/assign {face_ids, person_id | new_person_name | null}`; `DELETE faces/{id}`;
`POST bodies/assign {body_ids, person_id | new_person_name | null}`; `DELETE bodies/{id}`;
`POST guesses/confirm {faces: [id], bodies: [id]}`; `POST guesses/reject {faces, bodies}`;
`GET suggestions`; `POST suggestions/dismiss {a, b}`; `POST items/{uid}/tags {person_id | new_person_name}`;
`DELETE items/{uid}/tags/{person_id}`; `POST recluster`.
`PersonOut = {id, name, cover_url, cover_face_id, face_count, item_count, guess_count, created_at}` (counts over live, non-vault items).

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
queues, disk, wallet{configured, missing[]}, timezone, data_dir_free_bytes}`; `POST models/download` (starts the
downloader in a thread; progress via `GET`); `POST housekeeping` (no orphan sweep); `POST retry-failed` ->
`{requeued}` (failed uploads whose bytes are kept, errored jobs, item statuses back to `pending`).

## 14. Frontend

Routes: `/setup`, `/login` (no shell); everything else inside `AppShell` (session guard: redirects to
`/login?next=`, `/setup` when `setup_required`): `/` Library, `/people`, `/people/:id`, `/documents`,
`/review` (tabs Mis-clicks, Duplicates, Guesses), `/vault`, `/upload`, `/trash`, `/settings`
(Account and passkeys, Sessions, System, Wallet), `/m/:uid` (standalone viewer), `*`.

Library: filter bar (category chips, kind, people, dates, favourites), search box (one box; results page
shows the semantic and text hits with a mode switch), `MediaGrid` with document tiles (page 1 render in
a paper frame, type badge), video duration, mis-click and duplicate badges, multi-select with bulk actions.
Viewer: the hub's viewer with a document mode (a page strip of 30 pages either side of the current one plus the first and last, text, the virtual document card), face and
body boxes with guess chips (confirm, reject, assign), an info panel (category with override, quality
scores, labels, camera, dates), download original.
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
| `GUESS_WINDOW_H`, `GUESS_SCENE_SIM`, `GUESS_CLOTHES_SIM`, `GUESS_GPS_M` | 12, 0.80, 0.85, 250 | person guessing |
| `SEARCH_*` | as the hub | semantic search gates |
| `OCR_ENABLED`, `OCR_MAX_PAGES` | true, 50 | |
| `WALLET_PASS_TYPE_ID`, `WALLET_TEAM_ID`, `WALLET_ORG_NAME`, `WALLET_DIR`, `WALLET_KEY_PASSWORD` | unset, unset, site title, `DATA_DIR/wallet`, unset | Apple Wallet (the key password only when key.pem is encrypted) |
| `WORKERS_ENABLED` | true (false in tests) | |

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
and `doc_type` (label None, features `{"from": .., "to": ..}` plus the scores at the time). Events also feed
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
  and at least 3 on each side), else None (the rules decide).
- `misclick.py`: per reason (`blurry` on blur_var, `dark` on mean_lum, `blown_out` on clipped_frac,
  `pocket` on accident_p) the cut that best separates kept from confirmed mis-clicks, searched within
  [0.25x, 4x] of the default and used only with at least 5 examples on each side; `thresholds() ->
  {"blur_var_min": .., "dark_mean": .., "clipped_frac": .., "accident_min": ..}` with the defaults filled
  in, read by `quality.decide`.
- `people.py`: `person_threshold(person_id) -> float | None`: a per-person face threshold from the
  similarities of the person's user-confirmed faces to the person (loosened to their 10th percentile
  minus 0.02, floor 0.30) and raised above the closest rejected face (+0.02); `cluster._score` uses
  max(face threshold, learned) so a person the owner confirmed often is matched more readily and one
  they corrected is matched more carefully. `guess_probability(features) -> float | None`: a logistic
  regression on [clothes, scene_term, face_term, same_day, has_gps] fitted on `guess` events (needs 30
  events with both outcomes); `guess.guess_score` returns it instead of the fixed 0.5/0.3/0.2 mix.
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
`PersonRef.kind`; `GET /api/people/{id}/animals` (`AnimalPage` of `AnimalListItem`, best detection first);
`GET /api/people/{id}/items` and the `person` filter of the library and search include animal boxes;
`ItemDetail.animals: [AnimalOut{id, x, y, w, h, det_score, species, frame_time_s, person, assigned_by, url}]`
and `ItemDetail.people` lists pets with kind `pet`; `POST /api/animals/assign {animal_ids, person_id |
new_person_name}` (one species per call; a person target or a pet of another species is 400; a new name
creates or reuses a pet of the boxes' species; null unassigns); `DELETE /api/animals/{id}`;
`/files/animal/{id}` serves the crop with the face-crop rules (403 `vault_locked` for vault items). Pet names
go into the `people` column of the search index. Pets have no guesses and no merge suggestions.

Frontend: the viewer draws animal boxes with a paw mark (name or species, dashed when unassigned) and an
assign popover that lists pets of that species first and never people; the item's people chips show pets
with the paw; the People page requests `kind=person` and shows a Pets section below; the person page of a
pet shows its species and a crops grid (select, move, unassign, remove) instead of faces and guesses; the
library's person filter lists pets too.
