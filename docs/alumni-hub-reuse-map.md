# Alumni hub reuse map

Source project: `/home/ws-paraswtf/projects/sicsr-allumni-meet-portal` (the `SICSR Alumni Meet - Content HUB`). Every path below is relative to that root. Backend code is in `backend/app`, frontend code in `frontend/src`, the spec in `docs/SPEC.md`, and tests in `backend/tests`.

This document merges eleven subsystem maps: uploads, ingest-worker, media, faces, search, db-auth-api, admin-api-export, changelog-undo, frontend-shell, frontend-viewer-people, frontend-admin and tests. Nothing was run and no file was modified. Section 7 lists the places the readers did not read line by line. A copy of this document is at `/home/ws-paraswtf/.tmp/claude-2010/-home-ws-paraswtf-projects-personal-content-cloud/92a7cf6e-37b5-4902-9309-7ade33d23575/scratchpad/alumni-hub-reuse-map.md`.

The target is a single-user personal content cloud covering photos, videos, PDFs and other documents. The owner has every permission and must log in for everything. Originals are kept byte for byte. It needs categories (documents, personal, nsfw), mis-click detection, faces in photos and video frames, person guessing, OCR with full-text search, and ID document parsing. Geotag downloads and map cards are not needed, although EXIF GPS may still be read as a signal.

---

## 1. Stack and conventions

### 1.1 Stack

- **Backend:** Python 3.12, managed with `uv` (`[tool.uv] package=false`). It uses FastAPI and pydantic v2. Storage is SQLite in WAL mode with raw SQL and no ORM. Inference runs on onnxruntime CPU for InsightFace buffalo_l and SigLIP ViT-B/16. Image work uses OpenCV (`cv2.dct`), Pillow with pillow-heif, numpy and `tokenizers`. Sessions use itsdangerous. ffmpeg and ffprobe run as subprocesses.
- **Process model:** one uvicorn process; run a single worker. The background work runs on threads inside it:
  - `hub-worker-main` (serial queue `main`);
  - `hub-worker-video` (queue `video`);
  - `hub-gps-backfill` (optional);
  - `hub-search-indexer`.

  A flock on `DATA_DIR/.instance.lock` keeps a second process off the same data dir.
- **Frontend:** package `sicsr-content-hub-web`, ESM. It uses React 19.1, react-router 7.6 (imported from `react-router`, not `-dom`), @tanstack/react-query 5.80, Tailwind 4.1.8 through `@tailwindcss/vite`, Vite 7, TypeScript ~5.8.3 in strict mode, lucide-react 0.511, clsx, `@fontsource-variable/inter` and hash-wasm 4.12 (client SHA-256).
- **Scripts:**
  - `scripts/download_models.py` fetches buffalo_l.
  - `scripts/download_search_model.py` fetches SigLIP.
  - `scripts/run.sh` calls the SigLIP downloader on every start, and only warns if it fails. This file was not mapped.

### 1.2 Routing

`backend/app/main.py` holds `create_app()` and `lifespan(app)`. Routers live in `app/api/*`:

| Router file | Mount | Notes |
|---|---|---|
| `api/public.py` | `/api` | config, people, media, search, uploaders, geotag-video |
| `api/uploads.py` | `/api/uploads` | router deps `Depends(require_access)`, `Depends(upload_change, scope='function')` |
| `api/admin.py` | `/api/admin` | public part: login, logout, session. `protected = APIRouter(dependencies=[require_admin, _no_store, admin_change(scope='function')])`, included last |
| `api/exports.py` | `/api/admin/exports` | zip export |
| `api/history.py` | `/api/admin` (prefix applied at mount) | `dependencies=[Depends(require_admin), Depends(_no_store)]` |
| `api/files.py` | `/files` | file serving, GET and HEAD |

The app is built with `docs_url=None`, and OpenAPI is served at `/api/openapi.json`. An unknown `/api/*` path returns a JSON 404, and any other path falls back to the SPA `index.html` (sent with `no-cache`). Route order matters in uploads: `/mine` is declared before `/{upload_id}`.

**Lifespan startup order:**

1. `_acquire_instance_lock` (`LOCK_EX|LOCK_NB`).
2. `storage.ensure_dirs`.
3. `db.init_db`.
4. `images.register_heif` (a failure is logged).
5. Warn if ffmpeg or ffprobe is missing.
6. `precompress_assets`.
7. On a connection: `trash.recover`, declare the changelog change `maintenance.startup`, then `ingest.recover_on_startup`.
8. `geotag_video.startup`.
9. If `workers_enabled`: `worker.start()`, then `search_indexer.start()`.

Shutdown runs `geotag_video.shutdown`, then `search_indexer.stop`, then `worker.stop`, then releases the lock.

### 1.3 Auth and CSRF (`app/auth.py`, `app/api/deps.py`, `main.py`)

- **Admin account:** one hardcoded account in `app/config.py`, `ADMIN_USERNAME='Admin'` and `ADMIN_PASSWORD='Admin@1.12132040'`. These are plaintext constants, not env vars, and the README, the SPEC and `tests/conftest.py` all print them. `check_credentials(username, password)` uses `hmac.compare_digest` on the casefolded, stripped username and the exact password.
- **Session cookie:** `hub_admin` holds `TimestampSigner(secret_key(), salt='hub-admin-session').sign(b'admin:<ADMIN_USERNAME>:<epoch>')`. It is HttpOnly, SameSite=Lax and `path=/`, with `max_age` 7 days (`SESSION_MAX_AGE`). It is marked Secure when `X-Forwarded-Proto` is https or the scheme is https. The session is fixed-length, not sliding.
- **Epoch check:** `is_admin(request)` unsigns the cookie and compares the epoch with kv `admin_session_epoch`, opening one DB connection per call. `end_all_sessions()` bumps the epoch, which logs out every device.
- **Secret key:** `secret_key()` keeps 32 random bytes in `DATA_DIR/secret.key`, created with O_EXCL and mode 0600, and caches them under a lock.
- **CSRF:** `CSRF_HEADER='x-requested-with'`, `CSRF_VALUE='hub'`. `require_admin` returns 401 `'Admin login required'`, and 403 `'Missing X-Requested-With header'` on any method other than GET, HEAD or OPTIONS. Login and logout also require the header. The frontend sends `X-Requested-With: hub` on every request, chunk PUTs included.
- **Login rate limit:** in-memory, per IP and per process, lost on restart. `_MAX_FAILURES=10` per `_WINDOW_S=600`, tracking at most `_MAX_TRACKED=5000` IPs. `claim_login_attempt(ip)` counts the attempt as a failure before the password check, and `reset_login_failures(ip)` runs on success. Too many attempts return 429.
- **Visitor gate (optional):** settings `access_key`, env `ACCESS_KEY`. `AccessKeyAndHeadersMiddleware` is pure ASGI. When `?key=` matches (`hmac.compare_digest`) it sets cookie `hub_access` (an HMAC-SHA256 of the key, kept 365 days). An HTML GET is then answered with a 303 redirect to the same URL without the key, with leading slashes collapsed to avoid an open redirect. `has_access(request)` is True when no key is set, the cookie matches, or `is_admin`. `require_access` returns 401 detail `'access_key_required'`.
- **Security headers on every response:** `nosniff`, `Referrer-Policy: same-origin` and `X-Frame-Options: DENY`.
- **Client IP:** `client_ip(request)` is `request.client.host[:64]`. It never reads `X-Forwarded-For`; it relies on `uvicorn --proxy-headers --forwarded-allow-ips 127.0.0.1`.

### 1.4 Error shape

- Errors are raised as `HTTPException` with a short human `detail` string, for example `'Person not found'` or `'Invalid cursor'`.
- These details are fixed machine strings: `'access_key_required'`, `'search_unavailable'`, `'not_started'`, `'Invalid cursor'` and `'offset_mismatch'`.
- Status codes and their structured bodies:
  - 409 for the chunk offset mismatch, body `{detail:'offset_mismatch', received_bytes}`;
  - 409 for an undo conflict, with a fresh `UndoPreviewOut` in `detail`.
- Other status codes:
  - 413: too large (file, chunk or export edits).
  - 507: out of storage. The client never retries it.
  - 503 plus `Retry-After`: busy.
  - 422: validation.
  - 410: export source changed.
  - 416: range unsatisfiable.
- On the frontend, `ApiError(status, message, detail, body)` takes its message from a string `detail` or from `detail[0].msg`.

### 1.5 DB access pattern (`app/db.py`)

- **`connect()`:** opens `settings.db_path` (`DATA_DIR/portal.db`) with `timeout=30`, `isolation_level=None` (autocommit), `check_same_thread=False` and `row_factory=sqlite3.Row`. It sets `PRAGMA foreign_keys=ON`, `busy_timeout=30000`, `synchronous=NORMAL` and `temp_store=MEMORY`. It attaches `conn.hub = capture.Recorder()` and calls `capture.register`, plus `capture.install` when `changelog_eager`. Use one connection per request or thread and never share it.
- **Per request:** `get_db()` is a generator dependency: connect, yield, close. Every endpoint is a sync `def` running in the threadpool.
- **`tx(conn)`:** the outermost call runs `BEGIN IMMEDIATE`, and a nested call becomes `SAVEPOINT sp_N`. It calls changelog hooks around begin and commit: `context.before_begin/after_begin/before_commit/after_commit/after_rollback` and `trash.savepoint/rollback_to/after_commit/after_rollback`. It handles SQLite's auto-rollback (disk full). It is not a standalone helper until those hooks are removed.
- **Helpers:** `kv_get(conn,key,default)`, `kv_set` (upsert, values stored via `str()`), `one`, `all_rows`, `scalar`, `faces_version`, `bump_faces_version`.
- **`init_db()`:** sets `PRAGMA journal_mode=WAL` and runs `executescript(schema.sql)`. Inside a tx it then:
  - applies `_add_missing_columns` (`MIGRATED_COLUMNS`);
  - seeds kv `schema_version` (`SCHEMA_VERSION=6`, only ever bumped up), `faces_version='1'` and `media_id_mark=MAX(media.id)`;
  - sets `gps_backfill_end/cursor` the first time `gps_lat` is added.

  Changelog triggers are uninstalled while it runs.
- **SQL style:** raw SQL with `?` parameters only. Rows are accessed by column name. `IN` lists are chunked at 500, and `LIKE` searches escape `% _ \` with `ESCAPE '\\'`.
- **Heavy work outside the tx:** decode, hash, ffmpeg and inference run before the write transaction. The tx does only short writes and re-checks state, for example `media.version`.

### 1.6 Worker queue pattern (`app/jobs.py`, `app/worker.py`)

- **Queue:** a durable SQLite table `jobs`, with in-process `wake[queue]` Events.
  - `enqueue(conn, type_, payload, *, unique=False)` sets the priority from `PRIORITY` and the queue from `QUEUE_OF`, and copies `change_id` from `changelog.context.job_change_id(conn)`.
  - `unique=True` skips an identical queued `(type, payload text)`; for `recluster` it also checks running jobs.
- **Claiming and finishing:**
  - `claim_next(conn, queue)`: `BEGIN IMMEDIATE`, then the lowest `(priority, id)` job that is queued with `run_after` null or due; sets `running` and `attempts+1`.
  - `complete` deletes the row, so no job history is kept.
  - `fail(..., retry=True)`: while `attempts < MAX_ATTEMPTS=3`, requeues with `run_after = now + 15*2^(attempts-1)` s (15 s, then 30 s). After that the status becomes `'error'` and `last_error` is truncated to 4000.
  - `defer(conn,id,error,delay_s)` gives the attempt back.
- **Maintenance calls:** `requeue_running`, `requeue_stale(conn, queue, older_than_s, exclude_ids)` and `counts(conn) -> {queued,running,failed,video_queued}`.
- **Priorities and queues:** `PRIORITY={ingest:10, restore:20, faces:50, video_preview:50, recluster:90}`. `QUEUE_OF` puts `ingest`, `restore`, `faces` and `recluster` on `main`, and `video_preview` on `video`. The `main` queue is strictly serial, so dedup decisions never race.
- **Dispatch:** `worker.run_job(conn, job)` switches on `job['type']`:

  | Type | Payload |
  |---|---|
  | `ingest` | `{upload_id}` |
  | `faces` | `{media_id, version, people?}` |
  | `video_preview` | `{media_id, version}` |
  | `restore` | `{superseded_id}` |
  | `recluster` | `{}` |

  An unknown type raises `ValueError`. A new job kind must be added in `run_job`, `_final_failure`, `PRIORITY`, `QUEUE_OF` and the `jobs.type` CHECK.
- **`_handle(conn, job)`:** runs `_declare_job_change` (the job is skipped and completed if its change was undone), then `run_job`. On an exception it rolls back. A disk-full error anywhere in the cause chain (ENOSPC, EDQUOT or SQLite 'disk is full') calls `jobs.defer(120 s)`; anything else calls `jobs.fail`, and then `_final_failure` when no retry is left. `complete` and `fail` are themselves retried with waits of 2, 4, 8 and 16 s (`BOOKKEEPING_DELAYS_S`). `changelog_context.finish` always runs.
- **`_loop(queue)`:** waits with `wake.wait(1.0)` when idle. Every 600 s (main thread only) it runs `requeue_stale` for both queues (stale means older than 3600 s and not held by a live thread), then `_housekeeping`, which is `cleanup_stale_uploads`, `retention.expire` and `retention.purge_for_space`.
- **Recluster trigger:** when the main queue empties, `max(faces.id) > kv recluster_max_face_id`, and some faces are unassigned `auto` faces, a unique `recluster` is enqueued.
- **Tests:** `drain(queue=None, max_jobs=10000)` runs jobs synchronously, main before video.
- **Idempotency:** every handler re-checks DB state first. A terminal upload, a version mismatch, `faces_status='done'` or `preview_status!='pending'` each mean a no-op.

### 1.7 File serving (`app/api/files.py`, `app/urls.py`)

- **`_send(path, media_type, cache, extra=None)`:** returns a `FileResponse`, which gives Range/206 and ETag. It answers 404 when the file is missing and always sets `x-content-type-options: nosniff` and `content-security-policy: "default-src 'none'; sandbox"`.
- **`_media(conn, uid, request)`:** performs the access check and the lookup. It answers 404 when the item is missing, or hidden and the caller is not admin.
- **`_cache(row, v)`:**
  - hidden: `'private, no-cache'`;
  - `v == version`: `IMMUTABLE='public, max-age=31536000, immutable'`;
  - otherwise: `'no-cache'`;
  - face crops: `FACE_CACHE='public, max-age=86400'`.
- **Downloads:** `content_disposition(filename, ext)` builds `attachment; filename="<ascii fallback>"; filename*=UTF-8''<pct>` from `util.safe_filename` and `_ascii_fallback` (NFKD). The download name comes from `util.media_file_name(original_filename, ext)`.
- **URL builders:** every file URL is built in `urls.py` with `?v=<version>` for cache busting, for example `thumb_url(uid,v)=f'/files/thumb/{uid}?v={v}'`. The builders are:
  - `thumb_url`, `preview_url`, `preview_small_url`, `original_url`;
  - `download_url` (adds `&download=1`);
  - `geotag_url`, `video_url`;
  - `playable_video_url(uid,v,preview_status,stream_friendly)`, which is None while the preview is pending and the original is not stream friendly;
  - `face_url(id)='/files/face/{id}'`;
  - `superseded_thumb_url`, `superseded_file_url`.
- **Gzip:** only `/api` responses are gzipped (min 1024 B, level 6). `/files/` is never gzipped, so Range keeps working. `.gz` siblings (level 9, mtime 0) are precompressed for js, css, svg, json, map, html and txt under `frontend_dist/assets`.
- **Concurrency guards:** striped `threading.Lock` tables (64 stripes, `crc32(uid)%64`), a `BoundedSemaphore`, and 503 plus Retry-After when busy.

### 1.8 Naming, ids and time

- **Upload id:** `util.new_uid(length=14)` draws from the alphabet `abcdefghijkmnopqrstuvwxyz23456789` via `secrets`, about 70 bits. It doubles as the capability for that upload, and a new media item reuses it as `media.uid`.
- **Internal ids:** integer primary keys; `uid` strings appear in URLs and file names. `faces` and `persons` ids are AUTOINCREMENT. Media ids are never reused (kv `media_id_mark`), nor are versions (kv `vmark:<id>`), because undo restores rows under their old ids.
- **Time:** `util.now_iso(delta_s=0.0)` returns `'YYYY-MM-DDTHH:MM:SS.mmmZ'`, which sorts as text. `taken_at` is normalised to UTC with milliseconds and a trailing Z.
- **Cursors:** `util.encode_cursor(list)` and `decode_cursor(str|None)` are base64url JSON keyset cursors. Search uses offset paging instead.
- **Statuses:** string enums enforced by SQL CHECK constraints.
- **Error text for users:**
  - `ingest.short_error(exc)` keeps the first line, replaces the data dir with `<data>` and cuts to 160 chars;
  - `uploads.message` is capped at 300.
- **Code style:**
  - Python files start with `from __future__ import annotations`, use PEP 604 unions, define `log = logging.getLogger(__name__)` and never print. Modules open with a `CONTRACT` docstring.
  - Frontend files start with a one-line comment. Pages and components are default exports; hooks and helpers are named exports.
- **Copy rules:** SPEC section 2 and the machine policy ban em dashes, emoji and exclamation marks in UI copy.

---

## 2. Storage layout and schema summary

### 2.1 Data directory (`app/storage.py`)

Everything lives under `settings.data_dir` (env `DATA_DIR`, then `MEDIA_ROOT`, default `<repo>/data`). `DIRS` holds originals, thumbs, previews, videos, faces, staging, failed, superseded, geotag and tmp. `trash/` and `portal.db`, `secret.key` and `.instance.lock` sit there too. The shard is `sh = uid[:2]`.

| Helper | Path |
|---|---|
| `original_path(uid, version, ext)` | `originals/<sh>/<uid>.v<version><ext>` (untouched bytes) |
| `thumb_path(uid, v)` | `thumbs/<sh>/<uid>.v<v>.webp` |
| `preview_path(uid, v)` | `previews/<sh>/<uid>.v<v>.webp` |
| `preview_small_path(uid, v)` | `previews/<sh>/<uid>.v<v>.sm.webp` |
| `video_preview_path(uid, v)` | `videos/<sh>/<uid>.v<v>.mp4` |
| `face_crop_path(face_id)` | `faces/<face_id//1000>/<face_id>.webp` |
| `staging_path(uid)` | `staging/<uid>.part` (also the ingest input) |
| `failed_path(uid)` | `failed/<uid>.part` (never swept) |
| `superseded_path(uid, ext)`, `superseded_thumb_path` | `superseded/...` |
| `geotag_path`, `geotag_files` | `geotag/...` |
| `tmp_path(suffix='')` | `tmp/...` (wiped at startup) |
| trash | `trash/<change id>/<relpath>`, journal `trash/.pending/<token>.json` |

**File operations:**
- `ensure_dirs()`;
- `rel(path)` returns a path relative to the data dir, and `from_rel(relpath)` raises `ValueError` when the result escapes it;
- `move_into_place(src, dst)` (os.replace, creating parents) and `write_bytes_atomic(path, data)` (tmp name then os.replace, no fsync);
- `safe_unlink(path|None)`;
- `disk_usage() -> (free, total)` and `has_room_for(nbytes)`, which is true when `free - n >= settings.min_free_disk_bytes`.

Staging, originals, failed, tmp and trash must share one filesystem, because placement uses hard links and `os.replace`. `root()` returns `data_dir` un-resolved while `rel` and `from_rel` call `resolve()`, so `data_dir` must already be absolute.

### 2.2 Tables

All tables are defined in `backend/app/schema.sql`, plus the `MIGRATED_COLUMNS` added by `db.init_db`. No reader mapped `schema.sql` on its own; the columns below are the union of what the readers saw.

**Uploads and ingest**

- **`uploads`:**
  - `id` INTEGER PK; `uid` TEXT NOT NULL UNIQUE;
  - `uploader_id` INTEGER NOT NULL FK `uploaders`; `client_id` TEXT;
  - `original_filename` TEXT NOT NULL; `size_bytes` INTEGER NOT NULL (as declared); `received_bytes` INTEGER NOT NULL DEFAULT 0;
  - `mime` TEXT (the browser's claim, untrusted); `sha256` TEXT;
  - `status` TEXT NOT NULL CHECK IN (uploading, queued, processing, stored, duplicate, replaced, lower_quality, rejected, error, cancelled); `message` TEXT;
  - `media_id` FK `media` ON DELETE SET NULL; `remote_addr` TEXT;
  - `admin` INTEGER NOT NULL DEFAULT 0 (migrated, schema v5); `removed` INTEGER NOT NULL DEFAULT 0 (migrated, v6);
  - `created_at`, `updated_at`.
  - Indexes: `uploads_status(status, updated_at)`, `uploads_client(client_id, id)`, `uploads_created(created_at, id)`.
  - `ingest.TERMINAL` = stored, duplicate, replaced, lower_quality, rejected, error, cancelled. `ACTIVE` = uploading, queued, processing.
- **`uploaders`:** `id`, `name`, `name_key` UNIQUE (casefold; upsert `ON CONFLICT(name_key) DO NOTHING`), `created_at`. The DDL was inferred from INSERTs.
- **`known_hashes`:** `sha256` TEXT PK; `media_id` NOT NULL FK `media` ON DELETE CASCADE; `relation` CHECK IN ('current','replaced','lower_quality'); `created_at`. Index `known_hashes_media(media_id)`.
- **`media_contributors`:** `media_id` FK CASCADE, `uploader_id` FK `uploaders`, `first_upload_at`; PK `(media_id, uploader_id)`; index on `uploader_id`.
- **`superseded`:** `id`, `media_id`, `reason` ('replaced'|'lower_quality'), `path`, `thumb_path` (relative to the data dir; NULL after purge), `sha256`, `original_filename`, `ext`, `mime`, `size_bytes`, `width`, `height`, `duration_s`, `bitrate`, `uploader_id`, `upload_id`, `detail`, `created_at`.
- **`jobs`:**
  - `id` INTEGER PK AUTOINCREMENT;
  - `type` CHECK IN ('ingest','faces','video_preview','recluster','restore');
  - `queue` DEFAULT 'main' CHECK IN ('main','video');
  - `payload` TEXT DEFAULT '{}' (JSON);
  - `status` DEFAULT 'queued' CHECK IN ('queued','running','error');
  - `attempts` INT DEFAULT 0; `last_error` TEXT; `priority` INT DEFAULT 100; `run_after` TEXT;
  - `change_id` INT (migrated); `created_at`, `updated_at`.
  - Index `jobs_pick(queue, status, priority, id)`.

**Media**

- **`media`:**
  - Identity and file:
    - `id` PK (never reused); `uid` UNIQUE;
    - `kind` CHECK ('image','video'); `version` INT DEFAULT 1;
    - `original_filename`; `ext` (dotted, lowercase); `mime`;
    - `sha256` UNIQUE; `size_bytes`.
  - Picture and stream properties:
    - `width`, `height` (display orientation);
    - `duration_s`, `bitrate`, `video_codec`, `fps`, `has_audio`.
  - Capture metadata:
    - `taken_at` (UTC `...mmmZ`);
    - `exif_datetime`, `exif_subsec`, `camera`;
    - `gps_lat`, `gps_lon`, `gps_alt` (migrated).
  - Fingerprints:
    - `phash` (16 hex), `dhash` (16 hex);
    - `sig` BLOB (4096 B, 64x64 grey);
    - `frames` (video JSON list of 12 hex pHashes, `''` when a frame was not decoded).
  - State:
    - `uploader_id` NOT NULL FK; `hidden` INT DEFAULT 0;
    - `faces_status` CHECK ('pending','done','error');
    - `preview_status` CHECK ('none','pending','done','error');
    - `face_count`; `created_at`, `updated_at`.
  - Indexes: `media_created(created_at,id)`, `media_taken(taken_at,id)`, `media_uploader`, `media_kind(kind,hidden)`.

**Faces and people**

- **`faces`:**
  - `id` PK AUTOINCREMENT; `media_id` NOT NULL FK `media` ON DELETE CASCADE; `person_id` FK `persons` ON DELETE SET NULL;
  - `assigned_by` NOT NULL DEFAULT 'auto' CHECK ('auto','admin'). `'admin'` means locked, including admin-unassigned rows (person_id NULL with 'admin').
  - `x,y,w,h` REAL normalised to [0,1] in display orientation;
  - `det_score`, `size_px` (shorter side in source pixels), `quality`;
  - `frame_time_s` (video), `spans` TEXT JSON `[[start,end],...]` (migrated);
  - `embedding` BLOB NOT NULL float32[512], L2-normalised; `created_at`.
  - Indexes: `faces_media(media_id)`, `faces_person(person_id, quality)`.
- **`persons`:** `id` PK AUTOINCREMENT; `name` (NULL means unnamed, shown as `'Unnamed #id'`); `name_key` (casefold); `cover_face_id` FK `faces` ON DELETE SET NULL; `hidden` INT NOT NULL DEFAULT 0; `created_at`, `updated_at`. Index `persons_name(name_key)`.
- **`media_tags`:** `media_id` FK CASCADE, `person_id` FK CASCADE, `created_at`; PK `(media_id, person_id)`; index `media_tags_person`. These are manual tags without a face box.
- **`person_links`:** `id` PK; `person_id` FK CASCADE; `kind` CHECK IN (linkedin, instagram, website, email, phone); `value`; `label`; `position` INT DEFAULT 0; `created_at`, `updated_at`; UNIQUE `(person_id, kind, value)`.
- **`merge_dismissals`:** `person_a`, `person_b` FK `persons` CASCADE, `created_at`; PK `(person_a, person_b)`; CHECK `person_a < person_b`.

**Search**

- **`media_embeddings`:**
  - `media_id` NOT NULL FK `media` ON DELETE CASCADE; `version` INT NOT NULL;
  - `model` TEXT NOT NULL; `frame_index` INT NOT NULL DEFAULT 0; `frame_time_s` REAL (NULL for photos);
  - `embedding` BLOB NOT NULL (float32[768], L2-normalised, 3 KB); `created_at`.
  - PK `(media_id, model, frame_index)`; index `media_embeddings_model(model, media_id)`.
- **`person_embeddings`:** `face_id` PK FK `faces` CASCADE; `model`; `embedding` BLOB (NULL when the face is narrower than 20 px; the row still marks it done); `created_at`. Upsert `ON CONFLICT(face_id)`.

**Change history (changelog)**

- **`changes`:**
  - `id` PK AUTOINCREMENT; `created_at`, `updated_at`;
  - `actor` CHECK ('admin','visitor','system'); `actor_label`, `device`, `ip`, `session`;
  - `kind` (dotted); `summary` NOT NULL; `details` (JSON lines);
  - `maintenance` INT 0; `log_only` INT 0; `log_only_reason`;
  - `state` CHECK ('active','undone') DEFAULT 'active'; `undone_by` INT.
  - Index `changes_created(created_at, id)`.
- **`change_rows`:**
  - `id` PK; `change_id` (NULL until adopted); `tbl`;
  - `op` CHECK I/U/D/C; `pk` (JSON of the pk columns); `media_id`;
  - `data` (`{"old":{},"new":{}}`, where an UPDATE stores only changed columns). `{"error":true}` and `{"erased":true,"on":day}` mark rows that cannot be reverted.
  - Indexes: `(change_id,id)`, `(tbl,pk,id)`, `(media_id,tbl,id)` partial, `(id) WHERE change_id IS NULL`.
- **`change_refs`:** `ref`, `change_id`; PK `(ref, change_id)` WITHOUT ROWID. Refs look like `m:<uid>`, `p:<id>` or `u:<upload uid>`.
- **`change_undos`:** `undo_id`, `change_id`; PK WITHOUT ROWID; index on `change_id`.
- **`change_files`:** `id` PK; `change_id`; `relpath`; `trash_path`; `size_bytes`; `state` CHECK ('trashed','restored','purged'); `created_at`, `updated_at`. Indexes on `change_id`, `(state, created_at)` and `(relpath, state)`.

**Key/value: `kv`** (`key` TEXT PK, `value` TEXT NOT NULL)

| Key | Owner and meaning |
|---|---|
| `schema_version` | db.init_db (6) |
| `faces_version` | db, faces index reload counter; bump in the same tx as any change to faces or persons |
| `media_id_mark` | ingest and db; media ids are never reused |
| `vmark:<media_id>` | ingest `next_version`; versions are never reused |
| `admin_session_epoch` | auth; logout everywhere |
| `search_version` | search service; bumped in every tx that writes the embedding tables |
| `recluster_max_face_id` | worker `RECLUSTER_MARK` |
| `gps_backfill_end`, `gps_backfill_cursor` | ingest GPS backfill (drop) |
| `geotag_cache_signature` | ingest geotag cache (drop) |
| `video_face_times_v2`, `video_face_spans_v3` | ingest one-time `VIDEO_FACE_RERUNS` (drop) |

### 2.3 Configuration (`app/config.py`, not mapped directly)

Settings are read through `app.config.settings` and `reload_settings(**overrides)`, which the tests use.

| Env | Attr | Default | Consumer |
|---|---|---|---|
| `DATA_DIR` (fallback `MEDIA_ROOT`) | `data_dir`, `db_path` property | `<repo>/data`, `portal.db` | everything |
| `ACCESS_KEY` | `access_key` | unset | main middleware, auth |
| `SITE_TITLE` | `site_title` | brand title | config, exports, page title |
| `MAX_UPLOAD_MB` | `max_upload_bytes` | 16384 | uploads |
| `CHUNK_SIZE_MB` | `chunk_size` | 8 | uploads (tests 256 KiB) |
| `MIN_FREE_DISK_MB` | `min_free_disk_bytes` | 2048 | uploads, retention, download script |
| (n/a) | `staging_ttl_hours` | not stated by readers | `cleanup_stale_uploads` |
| `KEEP_SUPERSEDED` | `keep_superseded` | true | ingest `_keep_lower_quality` |
| `TRASH_DAYS` | `trash_days` | 30 | retention |
| `CHANGELOG_EAGER` | `changelog_eager` | false | db.connect |
| (n/a) | `workers_enabled` | true in prod, False in tests | worker, indexer |
| (n/a) | `frontend_dist` | `frontend/dist` | main |
| `EVENT_TIMEZONE` | `event_timezone` | `Asia/Kolkata` (UTC with a warning if invalid) | probe, exports, frame EXIF |
| `THUMB_SIZE` | `thumb_size` | 480 | images |
| `PREVIEW_SIZE` | `preview_size` | 2048 | images, indexer, frontend `preview.ts` |
| `PREVIEW_SMALL_SIZE` | `preview_small_size` | 1280 | images, files, frontend |
| `PREVIEW_QUALITY` | `preview_quality` | 80 (WebP) | images |
| `VIDEO_PREVIEWS` | `video_previews` | true | ingest gate for video_preview |
| `VIDEO_PREVIEW_MAX_HEIGHT` | `video_preview_max_height` | 720 (short side) | video |
| `VIDEO_PREVIEW_CRF` | `video_preview_crf` | 27 | video |
| `VIDEO_PREVIEW_MAXRATE_KBPS` | `video_preview_maxrate_kbps` | 2500 (bufsize 2x) | video |
| `VIDEO_PREVIEW_AUDIO_KBPS` | `video_preview_audio_kbps` | 96 | video |
| `VIDEO_TRANSCODE_THREADS` | `video_transcode_threads` | 4 | video |
| `VIDEO_STREAM_MAX_KBPS` | `video_stream_max_kbps` | 3000 | video `stream_friendly` |
| `VIDEO_TRANSCODE_MAX_S` | `video_transcode_max_s` | 10800 | video |
| `VIDEO_FACES_MAX_S` | `video_faces_max_s` | 900 | `iter_frames` ceiling |
| `VIDEO_SAMPLE_INTERVAL_S` | `video_sample_interval_s` | 1.0 | faces pipeline, people meta |
| `VIDEO_MAX_SAMPLES` | `video_max_samples` | 150 | faces pipeline |
| `VIDEO_TRACK_THRESHOLD` | (same) | 0.5 | FaceTracker |
| `MODELS_DIR` | `models_dir` | `<repo>/models` | faces engine, download_models |
| `ORT_THREADS` | `ort_threads` | `max(1, min(6, cpus-2))` | faces engine |
| `FACE_DET_SIZES` | `face_det_sizes` | 640,1280,1920 | analyze_image |
| `FACE_VIDEO_DET_SIZE` | `face_video_det_size` | 640 | analyze_video |
| `FACE_DET_THRESHOLD` | `face_det_threshold` | 0.5 | engine |
| `FACE_MIN_SIZE_PX` | `face_min_size_px` | 24 | pipeline |
| `FACE_GOOD_MIN_SIZE_PX`, `FACE_GOOD_DET_SCORE` | same | 48, 0.7 | `is_good_face`, tracker |
| `CLUSTER_MATCH_THRESHOLD` | same | 0.45 | cluster (good faces) |
| `CLUSTER_MATCH_THRESHOLD_LOW` | same | 0.52 | cluster (other faces) |
| `CLUSTER_TOPK` | same | 3 | index scoring |
| `SUGGEST_THRESHOLD` | `suggest_threshold` | 0.35 | merge_suggestions |
| `PUBLIC_MIN_MEDIA` | same | 2 | public people list |
| `SEARCH_ENABLED` | `search_enabled` | true | indexer, service |
| `SEARCH_MODELS_DIR` | `search_models_dir` | `<repo>/models` | search engine, download script |
| `SEARCH_THREADS` | `search_threads` | 2 | search engine |
| `SEARCH_MIN_RELEVANCE` | `search_min_relevance` | 0.001 | service |
| `SEARCH_CONTRAST` | `search_contrast` | 20.0 (0 disables) | service |
| `SEARCH_COLOR_CHECK` | `search_color_check` | true | service |
| `SEARCH_VIDEO_FRAMES` | `search_video_frames` | 6 | indexer |
| `SEARCH_PERSON_CROPS` | `search_person_crops` | true | indexer |
| `SEARCH_PERSON_MIN_RELEVANCE` | same | 0.4 | service |
| `GEOTAG_*` (`geotag_enabled`, `geotag_jpeg_quality`, `geotag_max_concurrent`, `geotag_map_km`, `geotag_place_name/lat/lon/radius_m`) | (same) | not stated | geotag only (drop); `geotag_jpeg_quality` is also used for full-size frame downloads |
| `BACKEND_URL` (frontend dev) | n/a | `http://127.0.0.1:41001` | vite proxy |

**Module constants:**

| File | Constants |
|---|---|
| `api/uploads.py` | `RESERVE_IDLE_S=3600`, `MAX_CHUNKS_IN_FLIGHT=64`, `MAX_IDS=200`, `MINE_LIMIT=300`, `_HASH_BLOCK=1 MiB` |
| `ingest.py` | `ORPHAN_STAGING_AGE_S=3600`, `CROP_QUALITY=85`, `GPS_BACKFILL_BATCH=20`, `TERMINAL`, `ACTIVE` |
| `worker.py` | `IDLE_POLL_S=1.0`, `CLEANUP_INTERVAL_S=600`, `BOOKKEEPING_DELAYS_S=(2,4,8,16)`, `STALE_RUNNING_S=3600`, `DISK_FULL_DELAY_S=120`, `GPS_BACKFILL_START_DELAY_S=15`, `GPS_BACKFILL_PAUSE_S=0.5`, `GPS_BACKFILL_RETRY_S=300`, `RECLUSTER_MARK` |
| `jobs.py` | `MAX_ATTEMPTS=3`, `PRIORITY`, `QUEUE_OF` |
| `main.py` | `SECURITY_HEADERS`, `API_GZIP_MIN_BYTES=1024`, `API_GZIP_LEVEL=6`, `IMMUTABLE`, `INSTANCE_LOCK='.instance.lock'` |
| `persons.py` | `NAME_MAX=120` |
| `media/probe.py` | `FFPROBE_TIMEOUT_S=60`, `DECODE_CHECK_TIMEOUT_S=60`, `SNIFF_BYTES=65536`, `MAX_VIDEO_PIXELS=8192*4320`, `MAX_VIDEO_FPS=300`, `ACCEPT_ATTR` |
| `media/images.py` | `Image.MAX_IMAGE_PIXELS=125_000_000` (process-wide), `MAX_PIXELS=250_000_000` |
| `media/video.py` | `FRAME_TIMEOUT_S=60`, `SEEK_SAMPLING_MIN_S=600`, `KEYFRAME_INTERVAL_S=2`, `NICE_LEVEL=10` |
| `media/hashing.py` | `SIG_SIZE=64`, `VIDEO_FP_POSITIONS` (12 points from 5% to 95%) |
| `media/xmp.py` | `MAX_PACKET=4 MiB`, `MAX_NODES=100000`, `MAX_DEPTH=256`, `MATCH_IOU=0.3` |
| `faces/*` | `CROP_SIZE=192`, `VIDEO_MAX_SIDE=1280`, `TRACK_EMBED_TOPN=5`, `MIN_SIM=0.2`, `_SCORE_BATCH=64`, `_SUGGEST_CHUNK=1024`, `_RECLUSTER_CHUNK=200`, `CARRY_IOU=0.3`, `CARRY_IOU_MIN_SIM=0.3`, `CARRY_SIM=0.6`, `CARRY_ADMIN_SIM=0.4`, `CARRY_ADMIN_IOU=0.1`, `DET_MODEL='buffalo_l/det_10g.onnx'`, `REC_MODEL='buffalo_l/w600k_r50.onnx'`, `EMBEDDING_DIM=512`; `index.DIM=512` |
| `search/*` | `BATCH_ITEMS=8`, `IDLE_POLL_S=3.0`, `BATCH_PAUSE_S=0.25`, `START_DELAY_S=10.0`, `TOOL_RETRY_S=300.0`, `IMAGE_SHORT_SIDE=448`, `PERSON_BATCH_FACES=24`, `PERSON_MIN_FACE_PX=20`, `VIDEO_FRAME_MAX_SIDE=640`, `NICE_INCREMENT=10`, `PROGRESS_LOG_EVERY=100`, `_TEXT_CACHE_SIZE=256`, `_IN_CHUNK=500`, `LOGIT_SCALE=117.33076`, `LOGIT_BIAS=-12.932437`, `SEARCH_PROMPT='a photo of {}'`, context 64, image 224, `MODEL_ID='ViT-B-16-SigLIP__webli'` |
| `auth.py` | `ADMIN_COOKIE='hub_admin'`, `ACCESS_COOKIE='hub_access'`, `SESSION_MAX_AGE=7*24*3600`, `CSRF_HEADER`, `CSRF_VALUE`, `_MAX_FAILURES=10`, `_WINDOW_S=600`, `_MAX_TRACKED=5000` |
| `api/files.py` | `IMMUTABLE`, `FACE_CACHE`, CSP, `FRAME_MAX_CONCURRENT=2`, `FRAME_WAIT_S=1.0`, `FRAME_PREVIEW_SIDE=1280`, `FRAME_PREVIEW_QUALITY=82`, `FRAME_CACHE_ENTRIES=8`, `GEOTAG_WAIT_S=20.0`, `EXACT_SEEK_EXTS={.mts,.m2ts,.ts}` |
| `api/public.py` | `ACCEPT='image/*,video/*,.heic,.heif,.avif,.webp,.mov,.m4v,.mkv,.3gp'` |
| `api/exports.py` | `MAX_UIDS=10000`, `MAX_EXPORTS=20`, `MAX_AGE_S=43200`, `MAX_STORE_BYTES=128 MiB`, `MAX_STREAMS=3`, `RETRY_AFTER_S=10`, `ID_BYTES=18`, `SQL_CHUNK=500` |
| `export/people.py` | `HUB_VERSION=1`, `MAX_JSON_CHARS=4 MiB`, `MAX_ITEMS=5000`, `MAX_NAME_CHARS=1000`, `BOX_TOL=0.01`, `BOX_DECIMALS=5`, `PHOTO_IOU=0.5`, `VIDEO_IOU=0.3`, `PHASH_MAX=10`, `DHASH_MAX=12`, `FP_BITS=128`, `FP_DIM=512`, `FP_MAX=24` |
| `export/zipstream.py` | `ZIP64_LIMIT32=0xFFFFFFFF`, `ZIP64_LIMIT16=0xFFFF`, `CRC_CHUNK=1 MiB` |
| `links.py` | `PUBLIC_KINDS=(linkedin,instagram,website)`, `PRIVATE_KINDS=(email,phone)`, `MAX_LINKS_PER_PERSON=20`, `LABEL_MAX=80`, `WEBSITE_MAX=300` |
| `changelog/*` | `undo.MAX_SET=50`, `REGROUP_KINDS={'faces.recluster','job.recluster'}`, `JOB_STATE={'faces_status','preview_status'}`, `ALLOWED_RESETS={('persons','cover_face_id'),('uploads','media_id'),('superseded','upload_id')}`, `ROWLESS_DIRS=('failed','staging')`, `_DERIVED=('thumbs','previews','videos')`, `retention.MIN_AGE_FOR_SPACE_S=86400`, `trash.LOW_DISK_MARGIN=1 GiB`, `describe.MAX_DETAILS=30` |

### 2.4 HTTP routes

All routes use the auth listed. "access" means `require_access` or `has_access`; "admin" means `require_admin`, with the CSRF header required on writes and `cache-control: private, no-store`.

**Public (`api/public.py`)**

| Method | Path | Auth | Request | Response |
|---|---|---|---|---|
| GET | `/api/config` | none (stats zeroed unless access) | none | `ConfigOut{site_title,max_upload_bytes,chunk_size,accept,is_admin,access_granted,stats:SiteStats{media,photos,videos,people,contributors}}` |
| GET | `/api/people` | access | `q` (max 120) | `PeopleOut{named,unnamed}`. Hidden people are excluded, and so are unnamed people with fewer than `PUBLIC_MIN_MEDIA` items. Named are sorted by name_key, unnamed by media_count desc then id. |
| GET | `/api/people/{id}` | access | none | `PersonOut` with public link kinds; 404 if hidden or in no visible media |
| GET | `/api/media` | access | `person[]` (AND), `uploader`, `kind`, `sort=recent|taken`, `cursor` (max 512 chars), `limit` 1..200 default 60 | `MediaPage{items,next_cursor,total}`. Hidden items are always excluded. Keyset paging: recent is `created_at DESC, id DESC`; taken is `COALESCE(taken_at,created_at) ASC, id ASC`; fetches limit+1; `total` is a separate COUNT; a bad cursor gives 400 `'Invalid cursor'`. |
| GET | `/api/media/{uid}` | access (hidden items for admin only, then `private, no-store`) | none | `MediaDetail{uid,kind,version,original_filename,mime,size_bytes,width,height,duration_s,fps,video_codec,bitrate,taken_at,camera,created_at,updated_at,uploader,contributors,urls:MediaUrls{thumb,preview,preview_small,original,download,video},faces:[FaceOut],tags,people,faces_status,preview_status,hidden,geo,geotag_available,video_geotag_available,people_links}` |
| POST/GET | `/api/media/{uid}/geotag-video` | access (+CSRF on POST) | `GeotagVideoIn{pos}` / `pos` | `GeotagVideoOut{state,progress,error,url,bytes}`, 404 `'not_started'` (drop) |
| GET | `/api/uploaders` | access | `q` | `[UploaderOut{id,name,media_count}]` (drop) |
| GET | `/api/search` | access | `q` (1..200, whitespace collapsed), `person[]`, `kind`, `limit` 1..200 default 60, `offset` 0..100000 | `SearchPage{query,items:[SearchHit=MediaItem+score+match_time_s],total,next_offset,indexed,indexing}`; a blank q gives 422 `'Type something to search for'`; `SearchUnavailable` gives 503 `{"detail":"search_unavailable"}` |

**Uploads (`api/uploads.py`, all `require_access`, no admin requirement)**

| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/api/uploads` | `UploadCreate{filename(1..255), size>0, mime?(<=128), uploader_name(1..80), client_id?(<=64), sha256?(^[0-9a-f]{64}$)}` | 201 `UploadOut`; 400 bad name, 413, 507 |
| PUT | `/api/uploads/{id}/chunk?offset=N` | raw octet-stream | `ChunkOut{received_bytes}`; 409 `{detail:'offset_mismatch',received_bytes}`; 409 when the status is not uploading; 413; 400 on an empty body; 503 with Retry-After 2; 507 |
| POST | `/api/uploads/{id}/complete` | `{}` | `UploadOut` (idempotent); 409 when the size or staging file does not match |
| GET | `/api/uploads/{id}` | none | `UploadOut` |
| GET | `/api/uploads?ids=a,b` | up to 200 ids, order kept, deduped, unknown ids omitted | `UploadList{uploads}` |
| GET | `/api/uploads/mine?client_id=` | 1..64 chars | latest 300 by id DESC |
| DELETE | `/api/uploads/{id}` | none | `UploadOut`; acts only on `uploading` (sets cancelled and deletes staging) |

`UploadOut = {id, filename, size, received_bytes, status, removed=False, message, chunk_size, media_uid, media_thumb_url, created_at}`. It is built by `common.upload_out(row)` from `UPLOAD_SELECT = 'SELECT u.*, m.uid AS media_uid, m.version AS media_version, m.hidden AS media_hidden FROM uploads u LEFT JOIN media m ON m.id = u.media_id'`. `media_uid` is None when the media is hidden, and `message` defaults from `UPLOAD_MESSAGES[status]`.

**Files (`api/files.py`, GET and HEAD; `has_access`; hidden items admin only)**

| Path | Query | Response |
|---|---|---|
| `/files/thumb/{uid}` | `v` | image/webp |
| `/files/preview/{uid}` | `v` | image/webp (large preview or video poster) |
| `/files/preview-small/{uid}` | `v` | image/webp built lazily by `_ensure_small_preview(uid, version)`; falls back to the large preview with `no-cache` |
| `/files/original/{uid}` | `v`, `download=0|1` | original bytes with `media.mime`; `download=1` adds Content-Disposition |
| `/files/video/{uid}` | `v` | the mp4 streaming copy if `preview_status=='done'` and the file exists, else the original; 404 for images |
| `/files/frame/{uid}` | `t` (required, finite), `v`, `pos=none|tl|tc|tr|bl|bc|br`, `size=full|preview`, `download` | image/jpeg of one frame; 422 if `t` is not finite; 404 for photos; 503 Retry-After 10 when busy |
| `/files/face/{face_id}` | none | image/webp crop, `FACE_CACHE` |
| `/files/geotag/{uid}`, `/files/geotag-video/{uid}` | `v,pos,size,download` | geotag renders (drop) |
| `/files/superseded/{id}/thumb`, `/files/superseded/{id}/original` | `download` | admin only, PRIVATE |

**Frame extraction:**
- **Time handling:** `ms = floor(round(t*1000,3))`, clamped to the duration. `_shown_ms` snaps to the start of the displayed frame via `video.shown_frame_time`, and `extract_frame(path, sec, max_side, exact_seek, gentle=True)` grabs it. If that fails, it retries `max(0.1, 1.5/fps)` s earlier, then 1000 ms earlier.
- **Concurrency:** a semaphore of 2 with a 1 s wait.
- **Preview frames:** long side 1280, q82. They come from the streaming copy when it is done, and an 8-entry LRU caches them by `(uid,version,source,ms)`.
- **Full frames:** taken from the original with quality `geotag_jpeg_quality`. The EXIF has Orientation 1, DateTimeOriginal in `event_timezone`, and GPS.
- **Download name:** `<stem>-<m>m<ss.s>s.jpg`.
- **Reusable helpers:** `_write_atomic(path,data)` and `_original_exif` (EXIF with Orientation=1 plus ICC).

**Admin (`api/admin.py`, under `/api/admin`)**

| Method | Path | Request | Response |
|---|---|---|---|
| POST | `/login` (open, CSRF) | `LoginIn{username,password}` (max 200) | `SessionOut{is_admin,username}`; 401, 429 |
| POST | `/logout` (CSRF) | none | `SessionOut`; ends all sessions only if the caller is admin |
| GET | `/session` (open, never 401) | none | `SessionOut` |
| GET | `/stats` | none | `AdminStats{media,photos,videos,hidden_media,faces,unassigned_faces,persons,named_persons,hidden_persons,uploaders,uploads_by_status,superseded,queue{queued,running,video_queued,failed},search{indexed,total,model},disk{free_bytes,total_bytes,originals_bytes,superseded_bytes}}` |
| GET | `/persons` | `q, filter=all|named|unnamed|hidden, sort=faces|name|recent, cursor, limit` (100, max 500) | `PersonPage` (offset cursor, computed in Python) |
| POST | `/persons` | `PersonCreate{name,face_ids}` | `PersonOut` |
| POST | `/persons/merge` | `MergeIn{target_id,source_ids}` | `PersonOut` |
| GET/PATCH/DELETE | `/persons/{id}` | `PersonUpdate{name?,hidden?,cover_face_id?}` (applied via `model_fields_set`) | `PersonOut` / `{ok}` |
| GET | `/persons/{id}/faces` | limit 200 (max 1000), quality desc, then id | `FacePage` |
| GET | `/persons/{id}/media` | limit 60 (max 200), faces union manual tags | `MediaPage` |
| GET/POST, PATCH/DELETE, POST | `/persons/{id}/links`, `/persons/{id}/links/{link_id}`, `/persons/{id}/links/reorder` | `{kind,value,label}`, `PersonLinkReorder{ids}` | `PersonLink` (drop) |
| GET | `/faces/unassigned` | cursor, limit 200 | `FacePage`; `AdminFace{id,media_uid,media_kind,thumb_url,det_score,size_px,quality,frame_time_s,person_id,assigned_by,created_at}` |
| POST | `/faces/assign` | `AssignFacesIn{face_ids, person_id | new_person_name | neither (unassign)}` | `AssignOut{updated,person}` |
| DELETE | `/faces/{id}` | none | `{ok}`; the crop goes to trash |
| GET | `/suggestions?limit=30` (max 200) | none | `SuggestionList{items:[{a,b,similarity,a_samples,b_samples}]}` (top 4 crops via ROW_NUMBER) |
| POST | `/suggestions/dismiss` | `DismissIn{a,b}` | `{ok}` |
| GET | `/media` | `hidden, cursor, limit` | `MediaPage` (keyset on `created_at DESC, id DESC`) |
| PATCH/DELETE | `/media/{uid}` | `MediaUpdate{hidden}` | `MediaItem` / `{ok}` |
| POST | `/media/{uid}/reprocess` | none | `{ok}`; sets `faces_status='pending'` and enqueues `faces` |
| POST, DELETE | `/media/{uid}/tags`, `/media/{uid}/tags/{person_id}` | `TagIn{person_id | new_person_name}` | `list[PersonRef]` |
| GET | `/uploads` | `status, q, cursor, limit` (100, max 500) | `UploadLogPage` (`UploadLogItem`: id, filename, size, received_bytes, status, message, removed, can_retry, uploader.name, remote_addr, media_uid, media_thumb_url, created_at, updated_at) |
| POST, DELETE | `/uploads/{uid}/retry`, `/uploads/{uid}/file` | none | `{ok}` |
| GET | `/duplicates?reason=replaced|lower_quality` | cursor, limit 60 | `SupersededPage` |
| POST, DELETE | `/duplicates/{id}/restore`, `/duplicates/{id}` | none | `{ok}` |
| POST | `/recluster` | none | `{ok}` (enqueues a unique `recluster`) |

**Exports (`api/exports.py`)**

- `POST /api/admin/exports`: request `ExportCreate{uids? | all?, hidden?}`; response `ExportOut{id,url,filename,items,photos,videos,hidden,bytes,with_people,missing,unknown}`. It returns 400 when nothing is selected and 413 when the edits exceed 128 MiB.
- `GET|HEAD /api/admin/exports/{id}/{filename:path}`: answers 200, 206, 404 (expired), 410 (originals changed), 416 or 503 with retry-after 10. It sends `etag '"<id>"'`, `accept-ranges`, `content-disposition` and `private, no-store`.

**History (`api/history.py`, under `/api/admin`)**

| Method | Path | Notes |
|---|---|---|
| GET | `/changes` | `actor` (admin|visitor|system), `q` (max 200; LIKE over summary, actor_label and details), `media`, `person`, `upload`, `maintenance` (default false), `cursor`, `limit` (30, 1..100). Returns `ChangePage{items,next_cursor,total,trash:TrashInfo{files,bytes,days},since}`, newest id first. |
| GET | `/changes/{id}` | `ChangeOut{id,created_at,actor,actor_label,device,ip,session,kind,summary,details[],state,undone_by,group_size,undo_of[],maintenance,log_only,log_only_reason,pending,can_undo,action(undo|redo|null),superseded_by,why_not,items[ChangeItem{uid,name,thumb_url,exists}] (max 6),people[ChangePerson{id,name,exists}] (max 8),trashed_files}` |
| POST | `/changes/{id}/undo-preview` | `UndoPreviewOut{ok,ids,changes,problems,visible,notes,why,blocked_by,files_restored,files_trashed}` (via `undo.closure`) |
| POST | `/changes/undo` | `UndoIn{ids:1..50, confirm_visible=false}` gives `UndoOut{undo_id,summary}`; 409 with an `UndoPreviewOut` in detail |
| POST | `/changes/{id}/erase` | `retention.erase(full=True)`; 409 when undone, held by a newer chain entry, or nothing was deleted |
| POST | `/trash/empty` | `TrashInfo` |
| GET | `/change-files/{file_id}` | webp from trash only when `state='trashed'` and the relpath is under `thumbs` or `faces` (or ends `.thumb.webp`); never originals |

---

## 3. Processing pipeline as it exists today

### Stage 0: client upload (`frontend/src/lib/uploader.ts`, `lib/api.ts`)

`UploadEngine.process()` runs these steps for each file:

1. Phase `checking`.
2. `tryResume`, using the `localStorage['hub.pending']` map keyed by `fileKeyOf(file)='name|size|lastModified'`. It resumes when the server status is uploading, queued or processing.
3. Hash with hash-wasm SHA-256 in 4 MiB slices, but only when the size is at most `HASH_LIMIT=1 GiB`.
4. `POST create` (via `withRetry`).
5. `sendChunks` through `uploadChunk`: an XHR PUT with a 60 s stall watchdog (`CHUNK_STALL_MS`, polled every 5 s; a tick gap over 15 s resets it).
6. Phase `finishing`: `POST complete`. A 409 here triggers up to 3 rounds of re-GET and resend.
7. `applyServer`. Queued or processing items move to phase `processing` and are polled.

**Engine constants:**
- `MAX_IN_FLIGHT=3`, `MAX_ATTEMPTS=8`, `BACKOFF_S=[1,2,4,8,16,30]`;
- `POLL_MS=2000`, `POLL_BATCH=200`, `MAX_RESYNCS=50`;
- `MIN_SLICE=256 KiB`: a 413 halves the slice down to this size and the limit sticks for the session;
- `PENDING_MAX_AGE_MS=14 days`, `EMIT_THROTTLE_MS=100`.

**Retry policy:** network errors, status 0, 429 and 5xx are retried, except 507. Offline time does not consume attempts.

### Stage 1: upload API (`api/uploads.py`)

- **`create_upload`:**
  - 413 when `size > max_upload_bytes`;
  - `clean_name(uploader_name)` else 400; `safe_filename`; mime trimmed to 128.
  - A `sha256` found in `known_hashes` creates a row with status `duplicate`. `media_id` is set in a separate UPDATE so the changelog records it, a `media_contributors` row is added, and no bytes are sent.
  - Otherwise, when the disk lacks room, it calls `retention.purge_for_space`. It touches an empty `staging/<uid>.part` and, inside the tx, requires `has_room_for(size + _reserved_bytes)` else 507.
  - `_reserved_bytes` is `SUM(size_bytes - received_bytes)` over uploading rows updated within `RESERVE_IDLE_S=3600`.
- **`put_chunk`** (async):
  - `_precheck_chunk` in the threadpool: the status must be `uploading`, and `offset == received_bytes` else 409.
  - `limit = min(chunk_size, size - offset)`; a larger `Content-Length` gives 413.
  - Beyond 64 chunks in flight it returns 503 with Retry-After 2.
  - The body is buffered into a bytearray, then `_write_chunk(uid, offset, data)` runs in the threadpool.
- **`_write_chunk`:**
  - Runs under `_upload_lock(uid)`, a refcounted per-uid `threading.Lock`.
  - Re-verifies the row. `_chunk_fits` requires `free - n >= min_free_disk_bytes//2` else 507.
  - Opens the staging file `r+b` (recreated with `w+b` only when offset is 0), then seek, write, flush, fsync. ENOSPC or EDQUOT truncates to the offset and returns 507.
  - Finishes with a compare-and-set: `UPDATE uploads SET received_bytes ... WHERE status='uploading' AND received_bytes=offset`.
- **`complete_upload`** (sync):
  - Returns 409 unless the received and staged sizes match.
  - `_sha256_file` reads 1 MiB blocks in the request thread.
  - A known hash becomes `duplicate` and the staging file is deleted.
  - Otherwise it sets `queued` plus `sha256` and calls `jobs.enqueue(conn,'ingest',{'upload_id': row['id']})` in the tx, then `jobs.wake[jobs.QUEUE_OF['ingest']].set()` after commit.

### Stage 2: ingest job (`ingest.process_upload(conn, upload_id)`)

1. A terminal upload unlinks the staged file and returns. Status `uploading` is a no-op. Otherwise the status becomes `processing`.
2. `_locate_upload_file` looks for `staging_path(uid)`, falling back to unreferenced `originals/<uid>.v1.*` or `superseded/<uid>.*`. If none exists, the upload goes to `error`.
3. `media.probe.probe(src)`: None leads to `_reject` (status `rejected`). A `ToolFailure` propagates so the job retries.
4. `sha = uploads.sha256 or sha256_file`. A hit in `known_hashes` sets `duplicate`, adds a contributor and drops the staged file.
5. `_fingerprint`: a `ValueError` rejects with `'The file could not be decoded...'`.
6. `dedup.find_duplicate(conn, info, fp, size)` returns `DedupDecision(action new|replace|lower_quality, target_id, target_uid, detail)`. A replace or lower_quality whose file carries a valid signed people packet for an item in `known_hashes` becomes `duplicate` (`_exported_copy`).
7. The decision dispatches to one of:
   - **`_store_new(conn, up, src, info, sha, fp, img, size)`:** the media uid is the upload uid. `_place` puts `src` at `original_path(uid,1,ext)`; `_place(src, dest)` hard-links via `tmp_path('.link')` plus `move_into_place`, and falls back to a copy. It writes the derivatives (Stage 3). One tx then runs `_insert_media` (34 columns), `media_contributors`, `known_hashes 'current'` and uploads `stored`. It calls `_enqueue_followups(conn, media_id, version, preview_status, people=False)`: a unique `faces` job `{media_id,version:1,people:1}`, `video_preview` when pending, and `search.indexer.notify()`. On error `_cleanup_unreferenced(written)` runs; after commit, `_drop_source`.
   - **`_replace`:** `version = next_version` (max of kv `vmark:<id>` and the current version, plus 1). It writes a new original and derivatives at v, adds a `superseded(reason 'replaced')` row, and marks the old hash `replaced` and the new one `current`. It UPDATEs `media` file fields; `COALESCE` keeps `taken_at`, EXIF and GPS when the new copy lacks them. It resets `faces_status` to `pending`, calls `trash.discard` on the old previews, and re-enqueues `faces`.
   - **`_keep_lower_quality`:** when `keep_superseded`, the file goes to `superseded/<uid><ext>` with a thumb, a `superseded(reason 'lower_quality')` row and a `known_hashes 'lower_quality'` row are added, and uploads becomes `lower_quality`.
8. Final failure (`_final_failure`) runs `mark_upload_failed(conn, upload_id, message)`: status `error`, and the bytes go to `failed/<uid>.part`. `requeue_failed_upload` moves them back to staging and enqueues a fresh ingest; it deletes the errored job by an exact payload JSON match. `discard_failed_upload` sends the file to trash.

**Probe details (`media/probe.py`):**
- **`sniff(path) -> (kind, mime, ext) | None`:** reads the first 64 KiB by magic bytes and needs at least 12 bytes. It knows:
  - images: JPEG, PNG, GIF, WebP, HEIC/HEIF/AVIF (ftyp brands; stills win over video), TIFF, BMP (DIB header size check);
  - video: MP4/M4V/3GP/MOV (ftyp or atoms), WebM/MKV (EBML DocType), AVI, MPEG-TS/M2TS (0x47 every 188 or 192 bytes).

  PDF, Office, SVG, HTML, archives, RAW (crx), jp2 and M4A all return None.
- **`probe_image`:** loads with Pillow (JPEG via draft 1/8) and rejects images over `MAX_PIXELS`. It reads EXIF orientation 0x0112, Make and Model, DateTimeOriginal 0x9003 (falling back to 0x9004), SubSecTimeOriginal 0x9291, OffsetTimeOriginal 0x9011 and the GPS IFD 0x8825, and detects animation.
- **`probe_video`:**
  - ffprobe JSON, using the first non-attached-pic video stream;
  - rejects frames over 8192x4320 or an average fps over 300, and requires ffmpeg to decode one frame (framemd5);
  - rotation comes from displaymatrix side data (negated) or the legacy `rotate` tag;
  - bitrate falls back to `size*8/duration`;
  - `taken_at` comes from `com.apple.quicktime.creationdate`, `creation_time` or `date`; years before 1980 are ignored.
- **Time and GPS helpers:**
  - `parse_taken_at` and `normalize_taken_at` read naive times in `event_timezone`.
  - `read_gps`, `exif_gps`, `parse_iso6709`, `video_gps` and `valid_gps` handle GPS. `valid_gps` rejects (0,0), non-finite and out-of-range values, drops an altitude beyond ±100000 m, and rounds to 7 dp (altitude 2 dp). The video tags tried are `com.apple.quicktime.location.iso6709`, `location` and `location-eng`.
- **`MediaInfo`:** `kind`, `mime`, `ext`, `width`, `height` (display-oriented), `duration_s`, `bitrate`, `video_codec`, `audio_codec`, `fps`, `has_audio`, `rotation`, `taken_at`, `exif_datetime`, `exif_subsec`, `camera`, `orientation`, `animated`, `gps_lat/lon/alt`.
- **Error taxonomy:** `ValueError` means a bad file; `ToolFailure(RuntimeError)` means the tools could not run, so retry.

**Fingerprints (`media/hashing.py`):**
- `sha256_file` reads 1 MiB blocks.
- `phash`: grey 32x32 LANCZOS, `cv2.dct`, top-left 8x8, each bit set when it exceeds the median excluding [0,0].
- `dhash`: grey 9x8, left > right.
- `signature`: 64x64 grey, 4096 bytes.
- `video_fingerprint`: pHash at 12 positions plus a 50% frame for `phash`, `dhash` and `sig`.
- `hamming` returns 64 for bad input. `sig_diff` returns `(mean_abs, max_8x8_block_mean)` after equalising brightness, and `(255,255)` on bad input.

**Dedup thresholds (`media/dedup.py`):**

| Constant | Value | Meaning |
|---|---|---|
| `PHASH_MAX` / `DHASH_MAX` | 10 / 12 | image hash gate |
| `SIG_MEAN_MAX` / `SIG_BLOCK_MAX` | 4.0 / 12.0 | pixel signature check |
| `IMAGE_ASPECT_TOL` / `VIDEO_ASPECT_TOL` | 0.02 / 0.03 | candidate filter (images filtered by aspect in SQL) |
| `RES_MARGIN` / `BYTES_MARGIN` | 1.05 / 1.10 | image quality ranking |
| `VIDEO_DURATION_ABS_S` / `VIDEO_DURATION_REL` | 1.0 / 0.02 | video candidate |
| `VIDEO_MIN_FRAMES` | 8 | decoded frames needed |
| `VIDEO_TAKEN_TOL_S` | 2.0 | capture-time guard |
| `VIDEO_FRAME_MEDIAN_MAX` / `VIDEO_FRAME_CLOSE` / `VIDEO_FRAME_CLOSE_SHARE` | 10 / 14 / 0.75 | per-frame pHash medians |
| `VIDEO_BITRATE_MARGIN` / `VIDEO_BYTES_MARGIN` | 1.15 / 1.15 | video quality ranking |

**Quality ranking:**
- Images rank by pixels (beyond 5%), then camera EXIF present, then size (beyond 10%) within a format family: jpeg; heif+heic+avif; png/tif/tiff/bmp; webp; gif.
- Videos rank by pixels (5%), then bitrate (15%), then size (15%), else the existing copy is kept.

**Guards:** the EXIF shot guard applies, but missing EXIF is never treated as evidence of a different shot. Hidden media count as candidates. All videos with `frames` are scanned in Python.

### Stage 3: derivatives (`media/images.py`, `media/video.py`)

- **`load_oriented(path, max_side=None)`:** applies `ImageOps.exif_transpose` and flattens to RGB (transparency on white; 16-bit grey shifted right by 8). It takes the first animation frame, uses JPEG draft when `max_side` is set, then a LANCZOS thumbnail.
- **`make_image_derivatives(img, thumb_path, preview_path, preview_small_path)`:** large preview 2048, then small 1280 resized from the large one, then thumb 480 from the small one. All are WebP, q80, method 4, atomic (hidden temp `.<name>.<uuid>.tmp` then os.replace), and never upscaled. `save_webp(img, path, max_side, quality)` and `to_bgr(img)` return a contiguous HxWx3 BGR array.
- **Video posters:** `make_video_derivatives` takes the frame at `poster_time(info)`, which is `min(3 s, 10% of duration)` (0 when under 1 s), and feeds it to `make_image_derivatives`. It raises `ValueError` when no frame decodes.
- **Frame helpers:**
  - `extract_frame(path, t, max_side=None, *, exact_seek=False, gentle=False)` pipes BMP with a 60 s timeout.
  - `frames_at(path, info, times, max_side=256)` clamps to the duration minus `max(0.1, 1.5/fps)` and retries 1 s earlier.
  - `shown_frame_time(path, t)` uses ffprobe packet times.
- **Subprocess rules:** every ffmpeg call uses argument lists with `-nostdin`, stdin DEVNULL and a timeout. Heavy runs go through `nice -n 10` (`_low_priority`).

### Stage 4: faces job (`ingest.process_faces(conn, media_id, version, people=False)`)

1. Skip if the media is gone, the version differs, or `faces_status=='done'`.
2. For an image: `images.load_oriented` then `pipeline.analyze_image(img)`. For a video: `pipeline.analyze_video(original, MediaInfo)`.
3. Encode crops as WebP q85 (`CROP_QUALITY`), each `CROP_SIZE=192` square.
4. Read the people packet only when `people=True` and the item has no faces yet.
5. Run `get_index().sync(conn)` before the tx.
6. In a single tx:
   1. INSERT the faces rows (`person_id NULL`, `assigned_by 'auto'`, box, `det_score`, `size_px`, `quality`, `frame_time_s`, `spans`, embedding bytes) and write `faces/<id>.webp`.
   2. `bump_faces_version`, then `cluster.carry_over(old_faces, new_ids)` and `_repoint_covers`.
   3. Delete the old faces and trash their crops.
   4. `people_meta.apply_to_new_faces` (in a savepoint), then `idx.sync_media`.
   5. `cluster.assign_new_faces(conn, remaining, exclude_face_ids=set(old_ids))`, then `_drop_empty_people`.
   6. Set `faces_status='done'` and `face_count`.
7. A `_Stale` exception is skipped quietly; any other error runs `idx.invalidate()` and re-raises.

**Engine (`faces/engine.py`, `FaceEngine(models_dir, threads=4)`):**
- **Models:** SCRFD-10GF `det_10g.onnx` (9 outputs: strides 8, 16 and 32 with 2 anchors) plus ArcFace R50 `w600k_r50.onnx` (512-d). Both run under `CPUExecutionProvider` with one lock per model.
- **`detect(img_bgr, det_size=640, threshold=0.5, nms_thresh=0.4)`:** pads to a multiple of 32 at top-left and normalises with `(x-127.5)/128`. It drops boxes under 2 px and returns `[]` for images under 8 px.
- **`detect_multiscale(img_bgr, sizes=(640,1280), ...)`:** runs the larger sizes only when the longest side is at least `0.75*size`, then merges with NMS 0.4.
- **`embed`:** umeyama alignment to 112x112 with `ARCFACE_DST`, `(x-127.5)/127.5`, batches of 32, then L2-normalise.
- **`analyze(img_bgr, sizes, threshold, min_size)`** chains detection and embedding. `get_engine()` is a singleton.
- **Output:** `DetectedFace` has `bbox`, `kps (5,2)`, `det_score`, `embedding`, and `width`, `height`, `size_px`.

**Pipeline (`faces/pipeline.py`):**
- **Record and crop:** `FaceRecord(x,y,w,h,det_score,size_px,quality,frame_time_s,embedding,crop,spans)`. `face_quality = det_score*min(size_px,160)/160`. `crop_face(img, box, size=192)` makes a square 1.6x the longer box side, shifted inside the image when it can be, else padded black.
- **`analyze_image`:** uses `face_det_sizes`, `face_det_threshold` and `face_min_size_px`.
- **`analyze_video`:**
  - Frame sampling: `iter_frames(path, info, video_sample_interval_s, video_max_samples, 1280)`, detected at `face_video_det_size`; a frame's min size is `face_min_size_px / src_scale`, with `src_scale` at least 1.
  - Tracking: `FaceTracker(threshold=0.5)` links greedily by cosine against each track's mean embedding, with no time or position term.
  - Output: one record per track, holding the best detection's box, crop and `frame_time_s`.
    - The embedding is the normalised sum of the top 5 qualities (`TRACK_EMBED_TOPN`).
    - `spans` come from `visible_spans(times, samples, end)` (midpoint rule, 3 dp).
  - Filtering: tracks with fewer than 2 detections are dropped unless they are good faces.

**`iter_frames(path, info, interval_s, max_frames, max_side)`:**
- **Output:** yields `(exact pts seconds, BGR array)` from one ffmpeg process using `select+showinfo`; a thread reads the pts from stderr.
- **Time limit:** `min(max(120, 4*duration), video_faces_max_s)`.
- **Long clips:** clips over 600 s use `_seek_frames`, one ffmpeg per timestamp with `-copyts`.
- **Fallback:** `extract_frame(0)` when nothing decodes.

### Stage 5: clustering (`faces/cluster.py`, `faces/index.py`)

- **`FaceIndex`:** an in-memory numpy index holding `_ids`, `_person` (-1 means unassigned), `_media`, `_admin` and `_emb (N,512)`, guarded by an RLock and resynced from kv `faces_version`.
  - Methods: `person_scores_many(embs, topk, exclude)` (the mean of the top-k cosine per person, for persons with at least one face scoring `MIN_SIM` 0.2 or more), `centroids`, `embeddings_of`, `set_person`, `sync_media(conn, media_id, base_version)`, `invalidate`, and a `cache_get`/`cache_put` keyed by generation.
  - A sync made inside an uncommitted tx is "tentative".
  - `get_index()` is a singleton keyed by `settings.db_path`.
- **`assign_new_faces(conn, face_ids, exclude_face_ids=None)`:** runs inside `tx` plus `idx.lock`.
  - Only faces with `person_id NULL` and `assigned_by 'auto'` are eligible.
  - Items are handled best quality first. Photos allow one person per photo (the best (face, person) pairs win); videos allow repeats.
  - A face joins a person at a top-k mean of 0.45 or more for good faces, or 0.52 for others. An unmatched good face creates a new unnamed person.
  - It bumps `faces_version` when anything changed.
- **`is_good_face(det_score, size_px)`:** `size_px >= 48` and `det_score >= 0.7`.
- **`recluster_unassigned(conn) -> int`:** works in steps of about 200 faces (whole items). It scores without a tx and applies in a short tx with a re-check. It never creates people.
- **`carry_over(conn, old_faces, new_face_ids)`:** three passes, best first:
  1. IoU ≥ 0.3 gated by sim ≥ 0.3;
  2. sim ≥ 0.6;
  3. admin-locked faces only: sim ≥ 0.4, or IoU ≥ 0.1 with sim ≥ 0.3.
- **`merge_persons(conn, target_id, source_ids)`:** moves faces, media_tags, merge_dismissals and person_links. The target inherits the first source's name when it has none, then the sources are deleted. Errors raise `ValueError`.
- **`merge_suggestions(conn, limit=30) -> [(a,b,sim)]`:** centroid cosine ≥ 0.35. It excludes hidden persons, dismissed pairs, and persons who appear together in a photo. It works in 1024-row chunks with a heap.

### Stage 6: video preview (`process_video_preview(conn, media_id, version)`, queue `video`)

- **Encoder:** `video.transcode_preview(src, dst, info, max_height)` encodes libx264 veryfast with `CRF video_preview_crf`. It sets maxrate and bufsize, high profile, yuv420p, and a short side of at most 720 (never upscaled).
- **Stream settings:** a keyframe every 2 s, AAC stereo, `-map_metadata -1`, `-fps_mode passthrough` and `+faststart`.
- **Process:** runs under nice 10 with a temp file plus os.replace. The timeout is `min(max(120, 4*duration), video_transcode_max_s)`.
- **Status:** `preview_status` is set to `done` only when the version still matches and the status is still pending; otherwise the output is deleted.
- **When a preview is needed:** `needs_web_preview(info, ext)` and `stream_friendly(codec, ext, w, h, bitrate)`. A file counts as stream friendly when it is h264 in .mp4/.m4v/.mov, its short side is at most 1.1×720, and its known bitrate is at most 3000 kbps.

### Stage 7: recluster (`worker._maybe_enqueue_recluster`, `_recluster`)

`_recluster` calls `cluster.recluster_unassigned` and stores the max face id captured before the run. Faces added during a run trigger another run.

### Stage 8: search indexer (`search/indexer.py`, thread `hub-search-indexer`)

- **Start condition:** `start()` requires `workers_enabled`, `search_enabled` and `service.engine_available()`. `notify()` sets `_wake`.
- **Loop:** each pass embeds media lacking a `media_embeddings` row for `(model, media.version)`, newest first, 8 per batch, with a 0.25 s pause; it polls every 3 s. After loading the model it lowers its own priority with nice +10.
- **Inputs:**
  - Photos: `load_photo(row)` shrinks the large preview to a 448 px short side, falling back to the oriented original.
  - Videos: `load_video_frames(row)` takes `search_video_frames` (6) frames via `iter_frames` at 640 px.
- **Person crops:** with no media pending, it embeds head-and-torso crops of photo faces, 24 per batch, into `person_embeddings`. `person_box(x,y,w,h,width,height)` is 3.2 face widths wide and runs from `y-0.25h` to `y+4.0h`; it returns None under 20 px.
- **Failures:** undecodable items go to `_failed` until restart. ffmpeg or engine errors go to `_deferred` for 300 s.
- **Store:** `_store` deletes all rows for the media_id (any model) and inserts the new ones in a short tx that re-checks the version.
- **Helpers:** `index_pending(conn)` (sync), `progress()`, `has_pending()`, `pending_items()`, `pending_faces()`.
- **Engine (`search/engine.py`):**
  - Sessions: `SearchEngine(models_dir, threads=2)` loads the visual and textual ONNX sessions (static batch 1) plus `tokenizer.json`.
  - Preprocessing: images are squashed to 224 bicubic and normalised with `(x/255-0.5)/0.5`. Text goes through `normalize_text`, then `search_prompt` (prefix `'a photo of {}'`), padded to 64 with id 1.
  - Methods: `embed_images(list)` returns (n,768); `embed_text(text, prompt=True)` uses an LRU of 256; `relevance(cos) = sigmoid(cos*117.33076 - 12.932437)`.
  - Files: `REQUIRED_FILES` are `visual/model.onnx`, `textual/model.onnx` and `textual/tokenizer.json` under `<root>/search/ViT-B-16-SigLIP__webli/`.

### Stage 9: search query (`search/service.py`)

- **`search(conn, q, person_ids=None, kind=None, limit=60, offset=0, include_hidden=False) -> SearchResult(hits, total, next_offset)`:**
  1. Embed `q` and sync `EmbeddingMatrix`, then compute `cos = emb @ query`.
  2. Keep the best row per media (`_best_rows`).
  3. Threshold: `max(search_min_relevance, search_contrast * median(rel))`.
  4. Colour check: `COLORS` = red, orange, yellow, green, blue, purple, pink, brown, black, white and grey, plus synonyms gray, violet, maroon and navy. A row passes when the named colour wins among the 11 variants.
  5. Person-crop pass with floor `max(threshold, 0.4)`, merged by the higher score.
  6. Order by `(-score, id)` and filter in SQL with `_filter_rows` (hidden, kind and `people_filter_sql`).
  7. Slice. `total` counts all hits after filtering, and the whole list is recomputed on every page.
- **`EmbeddingMatrix.sync(conn, model)`:** the signature is `(kv search_version, count media_embeddings, count non-null person_embeddings)`. On change it loads incrementally in IN-chunks of 500 and keeps the most common dimension.
- **Helpers:** `bump_search_version(conn)` (call it in the writing tx), `indexed_count`, and `engine_available`/`get_engine`/`current_model`, which load lazily so tests can monkeypatch them. `FALLBACK_MODEL_ID` names the default model.

### Stage 10: housekeeping and recovery

- **`cleanup_stale_uploads(conn)`:** cancels uploading rows idle longer than `staging_ttl_hours`, deletes their `.part` files and orphan staging files older than 3600 s, then runs `sweep_orphan_files`.
- **`sweep_orphan_files(conn, min_age_s=3600)`:** walks originals, thumbs, previews, videos, superseded, faces and geotag. It deletes files older than an hour that no row references and whose uid has no active upload. Names are parsed with the regexes `_ORIGINAL_NAME`, `_DERIVED_NAME`, `_FACE_NAME` and `_GEOTAG_NAME`, and `_is_referenced(conn, path)` guards every delete.
- **`recover_on_startup(conn)`:**
  - normalises `taken_at`, clears the stale geotag cache and runs `jobs.requeue_running`;
  - applies the one-time `VIDEO_FACE_RERUNS`;
  - enqueues the missing ingest, faces, video_preview and streaming-copy backfill jobs;
  - then wipes `data/tmp`.
- **`delete_media(conn, media_id)`:** one tx that deletes faces, media_tags, contributors, known_hashes and superseded rows, sets `uploads.media_id NULL` and `removed=1`, deletes the media row, keeps `media_id_mark`, drops empty people and bumps `faces_version`. It runs `trash.discard` on every file.
- **Retention:**
  - `retention.expire(conn)` handles items past `trash_days`.
  - `purge_for_space(conn, need_bytes)` purges oldest first, only trash older than 24 h, counting files with `st_nlink<=1`.
  - `empty_trash`, `erase(conn, change_id, full=False, reason)`.
  - `trash.recover(conn)` at startup reverses journaled moves.

### What does not exist anywhere

None of these exist in any subsystem:

- document kinds (PDF, Office, text) and PDF page renders;
- OCR and a full-text index (FTS5);
- ID document parsing;
- categories and an NSFW classifier;
- blur, dark or pocket-shot scoring;
- clothing or scene evidence for person guessing;
- per-frame video face boxes (only the best frame plus spans per track are kept);
- a password hash or a configurable login.

The closest existing hooks:

- `DetectedFace.kps/bbox`;
- `FaceRecord.quality`;
- `MediaInfo.taken_at`, `gps_*` and `camera`;
- the `person_embeddings` torso crops;
- SigLIP zero-shot prompts via `embed_text` and `relevance`;
- the 64x64 `sig`.

---

## 4. Frontend structure

### 4.1 Build and entry

- **`frontend/vite.config.ts`:** plugins `react()` and `tailwindcss()`. The dev host is 127.0.0.1, and `/api` plus `/files` are proxied to `BACKEND_URL` (default `http://127.0.0.1:41001`). Build output goes to `dist` with no sourcemaps and `chunkSizeWarningLimit` 900. There are no path aliases.
- **`tsconfig.json`:** ES2022, bundler resolution, `jsx react-jsx`, strict, `noUnusedLocals/Parameters`, `noFallthroughCasesInSwitch`, `isolatedModules`, `noEmit`.
- **Scripts:** `build` runs `tsc --noEmit -p tsconfig.json && vite build`. The frontend has no tests.
- **`index.html`:** theme-color `#C4161C`, favicon `/brand/favicon.svg`, title `SICSR Alumni Meet - Content HUB`. The brand assets are `public/brand/{emblem,lockup,lockup-light,favicon}.svg`.
- **`main.tsx`:** `StrictMode > QueryClientProvider > App` and imports Inter. The QueryClient uses `staleTime 15_000` and `refetchOnWindowFocus false`. Its retry rule never retries an `ApiError` 4xx and retries other errors at most twice.

### 4.2 Routes (`App.tsx`, BrowserRouter, Suspense with PageSpinner, React.lazy pages)

| Path | Page | Layout |
|---|---|---|
| `/` | HomePage | `Layout` (public shell) |
| `/people/:id` | PersonPage | Layout |
| `/photos` | PhotosPage (`?people=1,2&uploader=&kind=&sort=&q=&m=&t=`) | Layout |
| `/m/:uid` | MediaPage (standalone Viewer; closes to /photos) | Layout |
| `/upload` | UploadPage | Layout |
| `*` | NotFoundPage | Layout |
| `/admin/login` | AdminLoginPage | none |
| `/admin` + `people`, `people/:id`, `suggestions`, `faces`, `uploads`, `duplicates`, `media`, `history` | AdminDashboard, AdminPeoplePage, AdminPersonPage, AdminSuggestionsPage, AdminFacesPage, AdminUploadsPage, AdminDuplicatesPage, AdminMediaPage, AdminHistoryPage | `AdminLayout` (guards itself) |

- **Public shell:** `Layout.tsx` gets its access decision from `useConfig()`. While pending it shows PageSpinner; on error, an ErrorState. When `!access_granted && !is_admin` it shows `AccessGate`, an invite form that sets `?key=` and reloads. Otherwise it renders the skip-link, Header (People, All photos, Admin, Upload), `<main id=main><Outlet/>` and Footer.
- **`AdminLayout`:** wraps everything in `ToastProvider` and `Shell`. Shell guards with `useQuery(['admin','session'])` (staleTime 60 s, `retry false`) and redirects to `/admin/login?next=<path+search>`. It subscribes to the QueryCache and MutationCache; any 401 sets the session to `{is_admin:false}`. Logout clears every admin query except `session`, plus `media-detail`. The tab badges come from `api.admin.stats` (staleTime 15 s).
- **`AdminLoginPage`:** `safeNext()` allows only `/admin...` targets, rejecting `//` and `/admin/login`.

### 4.3 Data fetching

- **`lib/api.ts`:** `request<T>(method, url, body?, signal?)` sends `X-Requested-With: hub`, `Accept: json` and `credentials: same-origin`. Its helpers are `get`, `post` (which sends `{}` when there is no body), `patch` and `del`. It exports:
  - `qs(params)`, which skips null, undefined and `''` and repeats array keys;
  - `ApiError`, `OffsetMismatchError(receivedBytes)` and `uploadChunk`;
  - `frameSeconds(t)` (floored to ms, `toFixed(3)`) and `frameUrl(uid, version, t, pos, size='full', download=false)`;
  - `geotagUrl`, `geotagVideoUrl`;
  - the query types `MediaQuery{person?,uploader?,kind?,sort?,cursor?,limit?}`, `SearchQuery{q,person?,kind?,limit?,offset?}` and `ChangesQuery`;
  - the `api` object with namespaces `config`, `people`, `person`, `media`, `search`, `mediaDetail`, `uploaders`, `uploads.{create,chunk,complete,get,many,mine,cancel}` and `admin.*`.
- **`lib/types.ts`:** a hand-written mirror of `backend/app/schemas.py`, with no codegen.
- **Hooks:**
  - `lib/hooks.ts`:
    - `useConfig()`, query `['config']` with staleTime 30 s;
    - `useIsAdmin()`;
    - `useInfiniteSentinel(onVisible, enabled)`: an IntersectionObserver with `rootMargin '800px 0px'`;
    - `usePageTitle(page)`.
  - `lib/useMediaList.ts`: `useMediaList(query, enabled)` wraps `useInfiniteQuery(['media', query])` with cursor paging and dedupes by uid. `total` comes from the first page.
  - `lib/useSearchList.ts`: `useSearchList(query, enabled)` uses key `['search', query]` with offset paging. It exposes `unavailable` (503), `indexed` and `indexing`, and never retries an ApiError.
  - `components/admin/adminHooks.ts`:
    - `useCursorList(key, fetchPage, {poll?, enabled?})` dedupes by id or uid, polls every 8000 ms, and takes `total` from the last page;
    - `useSelection`, `useRefreshAll` (invalidates admin, people, person, media, media-detail, uploaders and config), `runBulk` (sequential, rethrows 401), `useDebounced(300)`, `useStable`, `errorMessage`, `isUnauthorized`;
    - a second `useViewerParam`.
- **Query keys:**
  - Public: `['config']`, `['people', q]`, `['person', id]`, `['media', query]`, `['search', query]`, `['media-detail', uid]`, `['uploaders', q]`, `['uploads-mine', clientId]`, `['uploads-interrupted']`.
  - Admin, all starting with `'admin'`: `session`, `stats`, `persons`, `person`, `person-faces`, `person-links`, `faces-unassigned`, `suggestions`, `media`, `uploads`, `duplicates`, `changes`, `undo-preview`, `person-picker`.
- **URL state:**
  - Filters live in search params set with `replace:true`.
  - `components/public/params.ts` provides `parseIds('1,2,x,2') -> [1,2]` and `withParam(params,key,value)`, which also strips `m` and `t`.
  - `components/public/useViewerParam.ts` `useViewerParam(items, hasMore, loadMore, total?)` drives the viewer with `?m=<uid>&t=<sec>` and pushes history state `hubViewer:true`. Close uses `navigate(-1)` when it pushed the entry. It prefetches within 3 items of the end and auto-closes when the uid leaves the list.

### 4.4 Shared components

- **`components/ui.tsx`:**
  - `buttonClasses(variant,size,extra)`: variants primary, secondary, ghost and danger; sizes sm h-8, md h-10 and lg h-12; always rounded-full.
  - Basics: `Button{loading}`, `Spinner`, `PageSpinner`, `EmptyState{icon,title,children,action}`, `ErrorState{error,onRetry}`, `Badge{tone neutral|brand|ok|warn}`, `Avatar{src,name,size=96}`, `Card`.
  - Overlays: `lockScroll()` (ref-counted), `Modal{open,onClose,title,children,footer,wide}` (a portal; Escape closes; becomes a bottom sheet on mobile), `ConfirmModal`.
- **`components/Select.tsx`:** `Select{value,onChange,options:SelectOption{value,label,hint?}[],label,showLabel,placeholder,size,disabled,className,searchable}` is a portal listbox. It is searchable by default above 8 options, with `MENU_MAX 320` and `ROW_H 40`.
- **`components/MediaGrid.tsx`:** `MediaGrid{items,onOpen(uid,index),hasMore,loadMore,loading,selectable?,selected?,onToggleSelect?}`.
  - Layout: grid `cols 3/4/5/6/7` with square `object-cover` tiles, lazy thumbs and 18 skeletons.
  - Tile overlays: video duration pill, `at m:ss` search pill, Hidden badge.
  - Paging: an infinite sentinel plus a Load more button.
- **`components/public/`:**
  - `Segmented<T>{label,value,options,onChange}`;
  - `CopyLinkButton{url,label}`;
  - `params.ts` and `useViewerParam.ts` (described in 4.3).
- **People components:**
  - `PersonCard{person,size,to,selected,onSelect}`.
  - `PersonPicker{onPick(PersonChoice),placeholder,allowCreate=true,excludeIds,autoFocus}`: a 200 ms debounce over `api.admin.persons` (limit 30).
  - `PeopleFilter{value,onChange,exclude}`: loads all of `api.people()` client side, up to 60 options.
  - `PersonLinks` exports `LINK_KINDS`, `LINK_META`, `safeHref` and `PersonLinkIcons`.
- **Viewer (`components/Viewer.tsx`):**
  - Props: `{uid,onClose,onPrev?,onNext?,position?,startAt?}`. It renders as a portal at z-40 on `#111`.
  - Detail polling: `['media-detail', uid]` refetches every 5000 ms while `preview_status==='pending'` and every 4000 ms while `faces_status==='pending'`.
  - Keyboard and swipe: Esc, arrows (ignored when the video has focus), `i`, `f`, a Tab focus trap; swipe at dx ≥ 50 and ≥ 1.5×|dy|.
  - Panels: InfoPanel, FaceEditor, DownloadMenu and the local `VideoFaces` chips.
  - Remembered state: the module-level `rememberFaces` and `rememberInfo`.
- **Viewer parts (`components/viewer/`):**
  - Stages:
    - `PhotoStage`: face boxes in %, srcSet, aspect re-derived from the preview.
    - `VideoStage`: `posterTime`, `currentFrameTime` and `useFrameClock` (requestVideoFrameCallback).
    - `VideoPreparing`: shown while the streaming copy is not ready.
  - Video helpers:
    - `videoSeek.ts`: `frameSeekTime` is the frame time plus `0.5/fps` (`FALLBACK_FPS=30`); `seekToFrame` is abortable; `boxInVideo` handles letterboxing.
    - `useOnScreenFaces.ts`: `facesAt` and `useOnScreenFaces` use the face spans.
  - Panels and menus:
    - `InfoPanel` (has local `PersonChip`, `Section` and `Row`).
    - `FaceEditor`: assign, unassign, not-a-face, and group rename or merge.
    - `DownloadMenu`: plain `<a download>` links.
    - `GeotagDialog` and `VideoGeotagDialog`.
  - `shared.ts`: `darkButton`, `iconButton`, `useInvalidateAfterAdmin(uid)` and `findCachedItem(qc, uid)`.
- **Upload components (`components/upload/`):**
  - `DropZone{accept,maxLabel,onFiles}`.
  - `QueueRow`: local previews for images up to 25 MiB.
  - `SummaryBar`.
  - `status.tsx`: `OUTCOMES`, `StatusBadge`, `FailedBadge`, `CancelledBadge`.
  - `EarlierUploads`, `NameStep`.
- **Admin components (`components/admin/`):**
  - `AdminModal` and `Confirm` (stable onClose).
  - `fields.tsx`: `inputClass`, `PageHeader`, `SearchInput`, `TextInput`, `SelectField`, `Segmented`, `SelectionBar`, `LoadMore`, `ListSkeleton`.
  - `Toast.tsx`: `ToastProvider` and `useToast()`, keeping 4 toasts that last 5 s (errors 9 s).
  - `StatusBadge`.
  - `FaceGrid`: shows size_px or a film icon with time, plus a lock for `admin`.
  - `MergeModal`, `PersonChoiceModal`, `PersonTile`, `PersonLinksCard`.
  - `ChangeEntry` (`canErase`, `whoLine`, `DELETING_KINDS`), `UndoDialog`.

### 4.5 Styling tokens (`index.css`, Tailwind v4 `@theme`, light only)

| Token | Value |
|---|---|
| `--font-sans` | Inter Variable stack |
| `brand` / `brand-dark` / `brand-tint` | #c4161c / #9f1116 / #f9e3e1 |
| `ink` / `ink-soft` / `ink-faint` | #2c3135 / #5b6268 / #8a9096 |
| `cream` / `cream-soft` / `paper` | #f3e9dd / #f8f2ea / #fffdfa |
| `line` | #e3d6c6 |
| `ok` / `warn` | #2f7d4f / #a86411 |
| `--radius-card` | 1rem |
| `--shadow-card` | card shadow |

The base layer sets `color-scheme: light`, body `bg-cream text-ink min-h 100dvh`, a brand outline on focus-visible, themed form controls (with a custom SVG check), an autofill override placed outside the layers on purpose, the `.select-pop` animation and a reduced-motion clamp. The viewer's `#111` is hardcoded. Design rules from the SPEC: 360 px minimum width, 16 px gutters (`px-4`), `max-w-7xl`, tap targets of at least 40 px.

---

## 5. Reuse verdicts

These are the readers' verdicts. Rows marked "not mapped" were referenced but never assigned to a reader; their verdict is the one the readers implied.

### Uploads

| File | Verdict | Why |
|---|---|---|
| `backend/app/api/uploads.py` | adapt | Keep the chunk protocol, CAS, locks, reservation and 507/503 handling; drop uploader, contributors, changelog and `require_access`, and move the complete-time hashing off the request thread. |
| `backend/app/storage.py` | adapt | Keep the sharding, versioned names, atomic moves and path guard; drop geotag/superseded, add document dirs, fix resolve(). |
| `backend/app/util.py` | adapt | Copy most helpers verbatim; extend `_EXT_ALIASES` in `media_file_name` for documents. |
| `frontend/src/lib/uploader.ts` | adapt | Best client asset; drop the uploader name and invite 401 text, add a document kind, rename the `hub.*` keys. |
| `frontend/src/pages/UploadPage.tsx` | adapt | Keep the layout and invalidation; remove NameStep and the alumni copy. |
| `frontend/src/components/upload/DropZone.tsx` | copy | Generic; change the copy to cover all files. |
| `frontend/src/components/upload/QueueRow.tsx` | adapt | Add a document icon and category; update the View link. |
| `frontend/src/components/upload/SummaryBar.tsx` | copy | Generic. |
| `frontend/src/components/upload/status.tsx` | adapt | Keep `OUTCOMES`; update the states and copy. |
| `frontend/src/components/upload/EarlierUploads.tsx` | adapt | Becomes a plain recent log, or drop it. |
| `frontend/src/components/upload/NameStep.tsx` | drop | Visitor-only concept. |

### Ingest and worker

| File | Verdict | Why |
|---|---|---|
| `backend/app/jobs.py` | adapt | Generic queue; remove the changelog `change_id` coupling and extend the types, priorities and queues. |
| `backend/app/worker.py` | adapt | Keep the thread loop, retry, defer and stale sweep; drop GPS backfill and the changelog declare. |
| `backend/app/ingest.py` | adapt | Keep `_place`, `_is_referenced`, the store and faces skeletons and cleanup; drop the replace/superseded, people-meta and GPS/geotag parts. |
| `backend/app/main.py` | adapt | Keep the lock, lifespan, headers, gzip and SPA fallback; replace ACCESS_KEY with a mandatory login. |
| `backend/app/persons.py` | copy | 48 lines, depends only on util. |

### Media

| File | Verdict | Why |
|---|---|---|
| `backend/app/media/probe.py` | adapt | Keep sniff, MediaInfo and the EXIF/GPS/time parsing; add a document branch and kind. |
| `backend/app/media/images.py` | copy | Self-contained derivatives and loaders. |
| `backend/app/media/video.py` | copy | `iter_frames`, frame and transcode helpers are needed as is. |
| `backend/app/media/hashing.py` | copy | Exact and perceptual hashes; blur and dark scoring must be added elsewhere. |
| `backend/app/media/dedup.py` | adapt | Keep the algorithm and thresholds; turn replace/lower_quality into duplicate linking. |
| `backend/app/media/xmp.py` | drop | Hub export namespace and signature; rewriting files conflicts with byte-for-byte originals. |
| `backend/app/media/embed.py` | not mapped (drop or optional) | XMP-into-copies for exports; `EMBED_EXTS` are JPEG, classic TIFF/DNG, PNG, WebP and MP4/M4V/3GP/MOV. |
| `backend/app/media/geotag_video.py` | not mapped (drop) | Geotag video renders, 528 lines. |
| `backend/app/geo/*` (places, overlay) | not mapped (drop) | Reverse geocoding and geotag cards. |

### Faces

| File | Verdict | Why |
|---|---|---|
| `backend/app/faces/engine.py` | copy | Standalone buffalo_l runner. |
| `backend/app/faces/pipeline.py` | adapt | Reusable; consider tracker time/position terms and per-frame persistence. |
| `backend/app/faces/cluster.py` | adapt | Keep the algorithm; drop `claim` and person_links, and rename `admin` to user-confirmed. |
| `backend/app/faces/index.py` | copy | Generic, concurrency-safe in-memory index. |
| `backend/app/faces/__init__.py` | copy | Empty marker. |
| `scripts/download_models.py` | adapt | Change the User-Agent; generalise for more models. |

### Search

| File | Verdict | Why |
|---|---|---|
| `backend/app/search/engine.py` | copy | Self-contained SigLIP engine, usable for zero-shot categories too. |
| `backend/app/search/indexer.py` | adapt | The right loop shape; add document loaders and drop the hidden and uploader assumptions. |
| `backend/app/search/service.py` | adapt | Keep the matrix and gates; replace the `api.common` imports, add filters, and add FTS separately. |
| `backend/app/search/__init__.py` | copy | Marker. |
| `scripts/download_search_model.py` | copy | Pinned and checksummed; template for other models. |
| `frontend/src/lib/useSearchList.ts` | adapt | Change the query shape; keep the 503 and indexing hints. |

### DB, auth and API

| File | Verdict | Why |
|---|---|---|
| `backend/app/db.py` | adapt | Keep connect/tx/kv/helpers; strip or keep the changelog hooks and start the schema version fresh. |
| `backend/app/auth.py` | adapt | Keep the signed cookie, epoch, CSRF, rate limit and secret key; hash the password, drop the access key, cache the epoch. |
| `backend/app/config.py` | not mapped (adapt) | Holds every setting in section 2.3 plus the hardcoded credentials, which must go. |
| `backend/app/schema.sql` | not mapped (adapt) | Drop uploaders, contributors, links and superseded; add document, category and analysis tables; widen `media.kind`. |
| `backend/app/api/deps.py` | adapt | Keep `get_db`; replace `require_access` with one global `require_user`. |
| `backend/app/api/common.py` | adapt | Keep `media_item`, cursor validation, people filter and person stats; drop the geotag and visibility code. |
| `backend/app/api/public.py` | adapt | Template for the library API (keyset paging, detail, search); drop uploaders and geotag. |
| `backend/app/api/files.py` | adapt | Keep `_send`, disposition, version caching, lazy small preview and frames; require login and use `private` cache. |
| `backend/app/schemas.py` | adapt | Reuse the media, face, person, upload and change models; drop uploader and geo; add document/category models. |
| `backend/app/urls.py` | copy | Pure builders; drop geotag/superseded and add document URLs. |
| `backend/app/links.py` | drop | Alumni profile links. |

### Admin, export and changelog

| File | Verdict | Why |
|---|---|---|
| `backend/app/api/admin.py` | adapt | Persons, faces, tags, media and recluster routes become the owner API; drop uploaders, links and the public/admin split. |
| `backend/app/api/exports.py` | adapt | Keep streaming, Range and store; generalise folders and kinds and make XMP planning optional. |
| `backend/app/export/zipstream.py` | copy | Dependency-free ranged zip64 writer. |
| `backend/app/export/people.py` | adapt | Keep `face_fp` and the orientation maps; the signing and trust model is mostly unnecessary. |
| `backend/app/export/__init__.py` | copy | Marker. |
| `backend/app/changelog/capture.py` | copy | Table-driven recorder; change `CAPTURED` and `CLAIMABLE`. |
| `backend/app/changelog/trash.py` | copy | Transactional file trash with crash recovery. |
| `backend/app/changelog/context.py` | adapt | Drop the visitor actor and uploads-log special case. |
| `backend/app/changelog/actors.py` | adapt | Keep `device_name` and `session_tag`; drop `visitor_spec`. |
| `backend/app/changelog/retention.py` | adapt | Update `CONTENT`, the layout prefixes and the wording. |
| `backend/app/changelog/describe.py` | adapt | Rewrite the kind switch for new domain kinds. |
| `backend/app/changelog/undo.py` | adapt | Keep the engine; replace the hub specifics (`REGROUP_KINDS`, `ALLOWED_RESETS`, `_fixups`, `row_files`, regeneration). |
| `backend/app/changelog/__init__.py` | copy | Docstring only. |
| `backend/app/api/history.py` | adapt | Keep the endpoints and button chain; drop the visitor wording. |

### Frontend shell

| File | Verdict | Why |
|---|---|---|
| `frontend/package.json` | adapt | Right stack; rename it. |
| `frontend/vite.config.ts` | copy | Change the default backend port. |
| `frontend/tsconfig.json` | copy | Generic strict config. |
| `frontend/index.html` | adapt | Rebrand and add a noindex meta. |
| `frontend/src/main.tsx` | adapt | Add global 401 handling. |
| `frontend/src/App.tsx` | adapt | Gate every route and add `/login`. |
| `frontend/src/index.css` | adapt | Re-theme the tokens; form-control rules are reusable. |
| `frontend/src/components/Layout.tsx` | adapt | Keep the structure; delete AccessGate, branding and the admin branching. |
| `frontend/src/components/ui.tsx` | copy | Domain-free primitives. |
| `frontend/src/components/Select.tsx` | copy | Generic accessible dropdown. |
| `frontend/src/components/MediaGrid.tsx` | adapt | Add NSFW masking, badges and a document tile; drop the Hidden badge. |
| `frontend/src/lib/api.ts` | adapt | Keep request/qs/ApiError/uploadChunk/frameUrl; drop geotag and uploader. |
| `frontend/src/lib/types.ts` | adapt | Keep the core shapes; add category, document and quality fields. |
| `frontend/src/lib/hooks.ts` | adapt | `useSession` replaces `useConfig`/`useIsAdmin`. |
| `frontend/src/lib/format.ts` | copy | Generic formatters. |
| `frontend/src/lib/preview.ts` | adapt | Constants track the backend. |
| `frontend/src/lib/useMediaList.ts` | adapt | Widen the query. |
| `frontend/src/pages/HomePage.tsx` | drop | Hub hero; reuse only the `SectionHeader` idea. |
| `frontend/src/pages/PhotosPage.tsx` | adapt | Best library template. |
| `frontend/src/pages/MediaPage.tsx` | adapt | Drop the uploader title. |
| `frontend/src/pages/NotFoundPage.tsx` | adapt | Update the copy and links. |
| `frontend/src/components/public/CopyLinkButton.tsx` | drop | No sharing in a single-user app. |
| `frontend/src/components/public/Segmented.tsx` | copy | Generic. |
| `frontend/src/components/public/params.ts` | copy | URL helpers. |
| `frontend/src/components/public/useViewerParam.ts` | copy | Generic lightbox hook. |
| `frontend/public/brand/*` | drop | SICSR logos. |

### Frontend viewer and people

| File | Verdict | Why |
|---|---|---|
| `frontend/src/components/Viewer.tsx` | adapt | Keep the overlay, keys, swipe and face chips; remove the admin branches and geotag; add a document view. |
| `frontend/src/components/viewer/shared.ts` | adapt | Update the invalidation keys. |
| `frontend/src/components/viewer/PhotoStage.tsx` | adapt | Drop the visitor branch; add an NSFW blur. |
| `frontend/src/components/viewer/VideoStage.tsx` | copy | Generic frame-aware player. |
| `frontend/src/components/viewer/videoSeek.ts` | copy | Pure seek math. |
| `frontend/src/components/viewer/useOnScreenFaces.ts` | copy | Pure hook. |
| `frontend/src/components/viewer/VideoPreparing.tsx` | copy | Still needed. |
| `frontend/src/components/viewer/InfoPanel.tsx` | adapt | Drop Shared by and Location; add sections for category, quality, OCR and ID fields. |
| `frontend/src/components/viewer/FaceEditor.tsx` | adapt | Drop the admin gating; add confirm and reject for guesses. |
| `frontend/src/components/viewer/DownloadMenu.tsx` | adapt | Remove the geotag item. |
| `frontend/src/components/viewer/GeotagDialog.tsx` | drop | Geotag. |
| `frontend/src/components/viewer/VideoGeotagDialog.tsx` | drop | Geotag. |
| `frontend/src/pages/PersonPage.tsx` | adapt | Remove the public wording and links. |
| `frontend/src/components/PersonCard.tsx` | adapt | Change the hidden badge and paths. |
| `frontend/src/components/PersonPicker.tsx` | adapt | Point it at the single persons endpoint. |
| `frontend/src/components/PeopleFilter.tsx` | adapt | Point it at the authenticated endpoint; it could merge with PersonPicker. |
| `frontend/src/components/PersonLinks.tsx` | drop | Profile links. |

### Frontend admin

| File | Verdict | Why |
|---|---|---|
| `frontend/src/components/admin/adminHooks.ts` | copy | Generic hooks. |
| `frontend/src/components/admin/fields.tsx` | copy | Form atoms. |
| `frontend/src/components/admin/Toast.tsx` | copy | Self-contained. |
| `frontend/src/components/admin/AdminModal.tsx` | copy | Focus fix wrapper. |
| `frontend/src/components/admin/StatusBadge.tsx` | adapt | Change the status enum. |
| `frontend/src/components/admin/FaceGrid.tsx` | copy | Handles photo and video faces. |
| `frontend/src/components/admin/MergeModal.tsx` | copy | Merge is still needed. |
| `frontend/src/components/admin/PersonChoiceModal.tsx` | copy | Needs PersonPicker. |
| `frontend/src/components/admin/PersonTile.tsx` | adapt | Drop the visitor-hidden semantics. |
| `frontend/src/components/admin/PersonLinksCard.tsx` | drop | Profile links. |
| `frontend/src/components/admin/ChangeEntry.tsx` | adapt | Only if history is kept; drop the visitor actor. |
| `frontend/src/components/admin/UndoDialog.tsx` | adapt | Drop "What visitors will notice". |
| `frontend/src/pages/admin/AdminLayout.tsx` | adapt | Becomes the app-wide auth shell. |
| `frontend/src/pages/admin/AdminLoginPage.tsx` | adapt | Move to `/login`, widen `safeNext`, never ship credentials. |
| `frontend/src/pages/admin/AdminDashboard.tsx` | adapt | Keep the cards and polling; switch to new counts. |
| `frontend/src/pages/admin/AdminPeoplePage.tsx` | adapt | Becomes the People section. |
| `frontend/src/pages/admin/AdminPersonPage.tsx` | adapt | Drop links and the public page link. |
| `frontend/src/pages/admin/AdminSuggestionsPage.tsx` | adapt | Template for person-guess review. |
| `frontend/src/pages/admin/AdminFacesPage.tsx` | copy | Works as is. |
| `frontend/src/pages/admin/AdminUploadsPage.tsx` | adapt | Ingest log without the uploader. |
| `frontend/src/pages/admin/AdminDuplicatesPage.tsx` | adapt | Depends on the duplicate policy. |
| `frontend/src/pages/admin/AdminMediaPage.tsx` | adapt | Main library with bulk actions and export; add category filters. |
| `frontend/src/pages/admin/AdminHistoryPage.tsx` | adapt | Only if history is kept. |

### Tests and scripts

| File | Verdict | Why |
|---|---|---|
| `backend/tests/conftest.py` | adapt | Keep the pattern; replace the admin login. |
| `backend/tests/helpers.py` | adapt | Remove the uploader parts; add document factories. |
| `backend/tests/__init__.py` | copy | Marker. |
| `backend/tests/fixtures/*.jpg` + README | copy | Public-domain faces with measured cosines. |
| `backend/tests/fixtures/search/*` + README | copy | Search and category fixtures. |
| `test_media.py`, `test_faces.py`, `test_worker.py`, `test_search_engine.py`, `test_zipstream.py` | copy | Generic modules they cover carry over. |
| `test_cluster.py`, `test_dedup.py`, `test_ingest.py`, `test_api_uploads.py`, `test_search.py`, `test_search_e2e.py`, `test_gps.py` (EXIF parts), `test_security.py`, `test_api_public.py`, `test_api_admin.py`, `test_api_exports.py`, `test_video_downloads.py` (frame parts) | adapt | Follow their modules. |
| `test_changelog_core.py`, `test_changelog_admin.py`, `test_changelog_uploads.py` | adapt | Only if undo is kept. |
| `test_changelog_review.py`, `test_embed.py`, `test_xmp.py`, `test_people_meta.py`, `test_reupload.py`, `test_geo.py`, `test_api_geotag.py`, `test_links.py` | drop | Hub-specific features. |
| `backend/pyproject.toml` | adapt | Pytest config `testpaths=['tests']`, `pythonpath=['.']`; dev dependencies pytest>=8 and httpx>=0.27. |
| `scripts/run.sh` | not mapped (adapt) | Calls the search model downloader on every start. |

---

## 6. Pitfalls and gotchas

### Security and auth

- The credentials are hardcoded in plaintext (`app/config.py`, README, SPEC section 3, `tests/conftest.py`). Never carry them over.
- `files._check_access` returns True when `ACCESS_KEY` is unset. Copied unchanged, it makes every `/files/*` route public. `/api/config` is intentionally unauthenticated.
- `is_admin()` opens a SQLite connection on every call, so a grid of 60 thumbnails opens 60 or more extra connections.
- Thumbnails and previews are sent with `public, max-age=31536000, immutable`, which is wrong for a private app behind a login; use `private`.
- The `?key=` invite flow puts a secret in the URL and the browser history.
- The login rate limit and the session epoch check are per process. A 7-day session does not slide, and logout ends every device.
- CSRF protection is only the `X-Requested-With: hub` header plus SameSite=Lax. Any new middleware that checks it must change in step with `api.ts` and `uploader.ts`.
- `client_ip` relies on uvicorn's proxy settings. If they are wrong, every request shares one rate-limit bucket.
- The CSP `default-src 'none'; sandbox` on originals may stop browsers from viewing PDFs inline.
- Cached previews stay in the browser after logout. The viewer's module-level preferences are not reset either.

### Upload and storage

- `complete_upload` hashes a possibly multi-GB file synchronously while holding the per-upload lock.
- Each chunk is fully buffered (8 MiB by default) with up to 64 in flight, about 512 MiB in the worst case. The docstring says it streams; the code buffers.
- `_locks`, `_chunks_in_flight`, the export store, the face index, the search matrix and the login limiter all live in process memory. Run exactly one uvicorn worker.
- The create-time duplicate shortcut trusts the client-supplied sha256 after only a pattern check.
- A file over 1 GiB is never hashed by the client, so it always transfers in full before dedup.
- `tryResume` works only in the same browser while the localStorage entry exists. Abandoned staging files depend on `cleanup_stale_uploads`.
- Placement relies on hard links and `os.replace`, so staging, originals, tmp and trash must share one filesystem. Otherwise `_place` falls back to copying, which doubles the IO.
- `media_file_name` keeps suffix aliases only for image and video extensions, and the extension must be passed lowercase with its dot.
- PDFs are rejected in three places: `kindOf`/`matchesAccept` in the client, `probe()` returning None, and the `media.kind` CHECK.

### Pipeline

- **The orphan sweeper** deletes any file older than an hour under the swept dirs unless a known regex matches it and a row references it. New derivative kinds must be referenced, match the patterns, or live outside those dirs.
- **Single serial `main` queue:** a big video faces job (or future OCR) blocks every ingest.
- **Faces jobs:** a `faces_status='done'` item is a hard skip. Reprocessing requires resetting it to `pending`. Any write to faces or persons must call `bump_faces_version` in the same tx. `idx.invalidate()` must run on failure.
- **Unique enqueue** compares payload text exactly. Key order and the extra `people:1` flag make different jobs. `jobs.complete` deletes the row, so no job history is kept.
- **Video face sampling:** limited to 1 frame per second, at most 150 frames and 900 s of decode. Truncation is only logged. Only the best frame and the spans per track are kept.
- **FaceTracker** links on embedding alone. Lookalikes merge into one track.
- **Cluster thresholds:** `CLUSTER_MATCH_THRESHOLD_LOW` (0.52) is stricter than the good-face threshold (0.45) on purpose.
- **`merge_suggestions`** skips pairs that appear together in one photo.
- **Swapping the face model** means retuning every threshold and re-embedding every face. The buffalo_l weights are licensed for non-commercial research only. SigLIP is Apache 2.0.
- **Search engine:**
  - Throughput: SigLIP has a static batch of 1, about 0.18 s per photo and 1.1 s per 6-frame video on 2 threads.
  - Load cost: about 1 GB of RAM per process.
  - Contrast gate: calibrated on 221 event photos plus 252 items with video frames. A library of mostly scans shifts the median.
- **Search failure state:** `_failed` and `_deferred` are in-memory and lost on restart. `_store` deletes rows of every model for the media_id. The pending-faces query needs SQLite JSON1.
- **Capture times:** naive EXIF times are read in `event_timezone`, which can move "same day" grouping around midnight. Apple's `creationdate` carries an offset; `creation_time` is usually UTC.
- **`video_fingerprint`** spawns 13 ffmpeg processes per video and decodes the middle frame twice.
- **`Image.MAX_IMAGE_PIXELS`** is set process-wide when `images.py` is imported, which affects PDF page rendering with Pillow.
- **`load_oriented`** flattens transparency to white and keeps only the first frame.
- **JPEG draft mode** decodes smaller than the reported size.
- **TIFF sniffing** accepts DNG-like files, and only the first page is decoded.
- **The frame endpoint** runs ffmpeg per request with 2 slots, so heavy scrubbing gets 503s.

### Changelog

- Triggers are TEMP and per connection, installed on the first `tx()`. A write outside `tx()` is unrecorded, or adopted later under `system.other`.
- The `pk` JSON must match byte for byte between SQL `json_object` and `capture.dumps`.
- Automatic analysis writes many rows per upload. Large blobs (embeddings, OCR text) and derived columns would bloat `change_rows`. Each derived column needs an explicit decision, as `face_count`, `faces_status` and `preview_status` had.
- Deletes must go through `trash.discard` inside the tx. The sweep must skip `trash/` and hold `trash.FILES_LOCK`.
- `erase()` makes a change impossible to undo (`log_only=1`).
- `undo._preconditions` has many persons and faces special cases. The hub's randomised tests guard them.

### Frontend

- `api.ts` has no global 401 handling; only `AdminLayout` redirects.
- `useToast` throws outside `ToastProvider`, which exists only inside `AdminLayout`.
- `MediaGrid` crops square with `object-cover`, which cuts off documents. NSFW thumbnails would show in the grid, the search results and the viewer posters.
- Two `useViewerParam` implementations exist:
  - `components/public/useViewerParam.ts` handles `m`, `t`, `hubViewer` and prefetch.
  - `components/admin/adminHooks.ts` handles `m` only.

  Both auto-close when the uid leaves the list. That fires wrongly when a filter changes, for example hiding NSFW, while the viewer is open. The faces pages pass no items (`NO_ITEMS`) on purpose.
- `Segmented` also exists twice (`components/public/Segmented.tsx` and `admin/fields.tsx`).
- `useMediaList` takes `total` from the first page; `useCursorList` takes it from the last.
- The PhotosPage SearchBox uses a ref to avoid URL feedback loops. Copying only part of it causes typing glitches.
- `preview.ts` sizes (2048 and 1280) must equal the backend settings, and `posterTime` mirrors `video.poster_time`.
- The dev proxy forwards only `/api` and `/files`.
- `types.ts` is a hand mirror of `schemas.py`, so drift is silent.
- Admin "Select all N" pages through the whole list on the client. Bulk actions send one request per item, creating one history entry each.
- `PeopleFilter` loads every person client side.
- The 30-day trash text is hardcoded in three confirm bodies, while History reads `trash.days`.
- `ExportDialog` posts from a `useEffect` guarded by a ref, to survive StrictMode.
- The Uploads and History pages poll forever if the server never clears `pending` or active states.
- The autofill CSS must stay outside `@layer`.
- The theme is light only, and a dark theme means redefining the tokens.

### Tests

- The face, ingest, reupload and changelog-upload tests need real buffalo_l and ffmpeg.
- The SigLIP model tests skip silently without `SEARCH_MODELS_DIR`.
- The autouse leak guard needs `app.changelog.capture.OUTSIDE_TX`.
- `make_media` assumes `uploader_id`.
- There are no fixtures for PDFs, OCR, ID documents, NSFW content or blur.

---

## 7. Open questions and reader disagreements

1. **Face crop size.** The ingest map says "160-style crops", while the faces map gives `CROP_SIZE=192` in `faces/pipeline.py`. 160 is the size cap inside `face_quality`, so the crop is most likely 192. The ingest reader did not read `pipeline.py`.
2. **Frame endpoint wait.** The db-auth-api reader saw `FRAME_WAIT_S=1.0` and then 503 with Retry-After 10. The viewer reader quoted SPEC 18.1 as "20 s wait then 503". 20 s matches `GEOTAG_WAIT_S`. Check `api/files.py` before relying on either.
3. **Two accept strings.** `api/public.py ACCEPT` includes `.webp` but not `.mts/.m2ts`, and it is the string served by `/api/config`. `media/probe.py ACCEPT_ATTR` is the reverse. Which one the UI uses depends on `/api/config`.
4. **`removed` upload status.** The frontend `UploadStatus` and `TERMINAL_UPLOAD_STATUSES` include `'removed'`, but the DB CHECK does not. In the DB it is the separate `uploads.removed` column, mapped to display.
5. **Login rate limit wording.** The auth reader says 10 attempts per 600 s, counted as failures before the password check and reset on success. The admin reader quotes the docstring as "10 failures / 10 min".
6. **SigLIP load cost.** The docstring says about 3 s and 1 GB; SPEC says 1.2 s.
7. **Face cosine ranges.** README says the same person scores 0.6-0.8 and different people below 0.3. The fixtures README measured 0.75-0.80 and 0.12 or lower.
8. **`event_timezone` default.** The code default is `Asia/Kolkata`, falling back to UTC only when the zone is invalid. The exports map phrased it as "default UTC fallback".
9. **Keep or strip the changelog and undo.** The changelog reader judged it valuable (undoing deletes, wrong tags, mis-sorting) and the core reusable. Other readers recommend stripping it from `db.tx`, `jobs`, `worker`, `uploads` and `ingest` to simplify. This is a product decision, and it is the largest coupling in the codebase.
10. **Near-duplicate policy.** `dedup`'s `replace` and `lower_quality` branches, `superseded`, `known_hashes.relation` and `media.sha256 UNIQUE` all assume one copy wins. That conflicts with "everything uploaded is retained". Readers agreed it must change, but not how: link as a duplicate group, keep both with a best-copy cover, or keep the superseded store.
11. **Hidden flags.** Some readers suggest repurposing `media.hidden` and `persons.hidden` as an NSFW vault or "ignored person"; others suggest dropping them.
12. **Unknown defaults and DDL:**
    - the `staging_ttl_hours` default was never stated;
    - the `uploaders` DDL was inferred;
    - `media_tags` and `person_links` columns came partly from usage;
    - the response shape of `POST /persons/{id}/links/reorder` is unknown;
    - the `GEOTAG_*` defaults were not read.
13. **Not read line by line:**
    - `changelog/undo.py`, roughly lines 420-880 (`_collateral`, `_fixups`, `_files`, `row_files`);
    - `pages/admin/AdminPersonPage.tsx`, roughly lines 420-487 (end of FacesTab, PhotosTab);
    - `viewer/GeotagDialog.tsx` and `viewer/VideoGeotagDialog.tsx`;
    - `media/xmp.py` lines 200-459;
    - most test file bodies, so test counts are approximate, and parts of `tests/helpers.py`;
    - `app/media/embed.py`, `app/media/geotag_video.py`, `app/geo/*`, `app/config.py`, `app/schema.sql` and `scripts/run.sh`.
14. **Ingest behaviour on staging files.** The uploads reader did not verify it; the ingest reader describes it, including the fallback locate and the hard link.
15. **Person scoring with few faces.** `person_scores` averages "top-k" over fewer than 3 values for persons with few faces. Whether this biases new-person matching on a personal library was not measured.