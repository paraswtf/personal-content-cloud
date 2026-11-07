-- Personal Content Cloud schema (SQLite, WAL). See docs/ARCHITECTURE.md section 5.
-- Timestamps are ISO 8601 UTC strings from app.util.now_iso(). File paths are never stored for
-- items: they derive from (uid, version, ext) via app.storage.

CREATE TABLE IF NOT EXISTS kv (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- The single owner account (always id 1).
CREATE TABLE IF NOT EXISTS account (
  id            INTEGER PRIMARY KEY CHECK (id = 1),
  username      TEXT NOT NULL,
  username_key  TEXT NOT NULL,                 -- casefolded
  password_hash TEXT NOT NULL,                 -- argon2id
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL
);

-- WebAuthn credentials (discoverable passkeys).
CREATE TABLE IF NOT EXISTS passkeys (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  credential_id   BLOB NOT NULL UNIQUE,
  public_key      BLOB NOT NULL,               -- COSE key bytes as returned by py_webauthn
  sign_count      INTEGER NOT NULL DEFAULT 0,
  transports      TEXT,                        -- JSON list of transport hints
  name            TEXT NOT NULL,               -- owner's label ("iPhone", "YubiKey")
  aaguid          TEXT,
  backup_eligible INTEGER NOT NULL DEFAULT 0,
  created_at      TEXT NOT NULL,
  last_used_at    TEXT
);

-- Browser sessions. The cookie holds the raw token; only its sha256 is stored.
CREATE TABLE IF NOT EXISTS sessions (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  token_hash   TEXT NOT NULL UNIQUE,
  created_at   TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  expires_at   TEXT NOT NULL,
  user_agent   TEXT,
  ip           TEXT,
  vault_until  TEXT,                           -- NSFW vault unlocked until this time (NULL = locked)
  revoked_at   TEXT
);
CREATE INDEX IF NOT EXISTS sessions_expires ON sessions(expires_at);

-- One-time WebAuthn challenges (registration, login, re-authentication for the vault).
CREATE TABLE IF NOT EXISTS challenges (
  id         TEXT PRIMARY KEY,                 -- random id handed to the client
  kind       TEXT NOT NULL CHECK (kind IN ('register','login','assert')),
  challenge  BLOB NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  used       INTEGER NOT NULL DEFAULT 0
);

-- One row per upload attempt (chunked, resumable).
CREATE TABLE IF NOT EXISTS uploads (
  id                INTEGER PRIMARY KEY,
  uid               TEXT NOT NULL UNIQUE,      -- returned to the client; a new item reuses it as items.uid
  source            TEXT NOT NULL DEFAULT 'web' CHECK (source IN ('web','share','camera','api')),
  client_id         TEXT,                      -- random id kept in the browser (resume, "mine")
  original_filename TEXT NOT NULL,
  size_bytes        INTEGER NOT NULL,          -- declared size
  received_bytes    INTEGER NOT NULL DEFAULT 0,
  mime              TEXT,                      -- as declared by the browser (not trusted)
  sha256            TEXT,                      -- computed by the server on completion
  status            TEXT NOT NULL CHECK (status IN ('uploading','queued','processing','stored','duplicate',
                                                    'rejected','error','cancelled')),
  message           TEXT,
  item_id           INTEGER REFERENCES items(id) ON DELETE SET NULL,
  remote_addr       TEXT,
  created_at        TEXT NOT NULL,
  updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS uploads_status  ON uploads(status, updated_at);
CREATE INDEX IF NOT EXISTS uploads_client  ON uploads(client_id, id);
CREATE INDEX IF NOT EXISTS uploads_created ON uploads(created_at, id);

-- One row per stored file. The original is kept byte for byte.
CREATE TABLE IF NOT EXISTS items (
  id                INTEGER PRIMARY KEY,       -- never reused (kv item_id_mark)
  uid               TEXT NOT NULL UNIQUE,      -- id in URLs and file names
  kind              TEXT NOT NULL CHECK (kind IN ('image','video','pdf','document','other')),
  version           INTEGER NOT NULL DEFAULT 1,
  original_filename TEXT NOT NULL,
  ext               TEXT NOT NULL,             -- storage extension incl. dot, lowercase ('.jpg', '.pdf')
  mime              TEXT NOT NULL,
  sha256            TEXT NOT NULL UNIQUE,
  size_bytes        INTEGER NOT NULL,
  width             INTEGER,                   -- display orientation; page 1 for documents; NULL for 'other'
  height            INTEGER,
  duration_s        REAL,
  bitrate           INTEGER,
  video_codec       TEXT,
  fps               REAL,
  has_audio         INTEGER,
  page_count        INTEGER,                   -- documents
  animated          INTEGER NOT NULL DEFAULT 0,
  taken_at          TEXT,                      -- capture time, UTC
  exif_datetime     TEXT,
  exif_subsec       TEXT,
  camera            TEXT,
  gps_lat           REAL,
  gps_lon           REAL,
  gps_alt           REAL,
  phash             TEXT,                      -- 16 hex (video: middle frame; documents: page 1)
  dhash             TEXT,
  sig               BLOB,                      -- 64x64 grey signature
  frames            TEXT,                      -- video: JSON list of per-frame phash hex
  category          TEXT NOT NULL DEFAULT 'misc' CHECK (category IN ('documents','screenshots','personal','places',
                                                                      'food','pets','objects','misc','nsfw')),
  category_source   TEXT NOT NULL DEFAULT 'auto' CHECK (category_source IN ('auto','user')),
  category_scores   TEXT,                      -- JSON {category: probability, ...}
  doc_likeness      REAL,                      -- 0..1, triggers text extraction for images
  nsfw              INTEGER NOT NULL DEFAULT 0,
  nsfw_score        REAL,
  nsfw_source       TEXT NOT NULL DEFAULT 'auto' CHECK (nsfw_source IN ('auto','user')),
  misclick          INTEGER NOT NULL DEFAULT 0,
  misclick_reasons  TEXT,                      -- JSON {reason: detail, ...}: blurry, dark, blown_out, pocket, burst
  misclick_source   TEXT NOT NULL DEFAULT 'auto' CHECK (misclick_source IN ('auto','user')),
  quality_scores    TEXT,                      -- JSON {blur_var, mean_lum, p95_lum, clipped_frac, accident_p, motion_blur_p}
  dup_group_id      INTEGER REFERENCES items(id) ON DELETE SET NULL,  -- the group's best item (self for the best)
  dup_role          TEXT CHECK (dup_role IN ('best','copy')),
  dup_detail        TEXT,
  analysis_status   TEXT NOT NULL DEFAULT 'pending' CHECK (analysis_status IN ('pending','done','error')),
  text_status       TEXT NOT NULL DEFAULT 'none' CHECK (text_status IN ('none','pending','done','error')),
  preview_status    TEXT NOT NULL DEFAULT 'none' CHECK (preview_status IN ('none','pending','done','error')),
  face_count        INTEGER NOT NULL DEFAULT 0,
  body_count        INTEGER NOT NULL DEFAULT 0,
  favorite          INTEGER NOT NULL DEFAULT 0,
  deleted_at        TEXT,                      -- in the trash since (NULL = live)
  created_at        TEXT NOT NULL,
  updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_created   ON items(created_at, id);
CREATE INDEX IF NOT EXISTS items_taken     ON items(taken_at, id);
-- The library's two orders over live items, led by the vault side every listing filters on, so
-- a page reads an index in order instead of sorting every visible row (the planner picks these
-- over items_flags with or without statistics).
CREATE INDEX IF NOT EXISTS items_order     ON items(nsfw, COALESCE(taken_at, created_at), id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS items_added     ON items(nsfw, created_at, id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS items_kind      ON items(kind, deleted_at);
CREATE INDEX IF NOT EXISTS items_category  ON items(category, deleted_at);
CREATE INDEX IF NOT EXISTS items_flags     ON items(deleted_at, nsfw, misclick);
CREATE INDEX IF NOT EXISTS items_dup       ON items(dup_group_id);
CREATE INDEX IF NOT EXISTS items_analysis  ON items(analysis_status);
CREATE INDEX IF NOT EXISTS items_text      ON items(text_status);

-- Image-text embeddings (SigLIP): frame 0 is the whole image (page 1 for documents, poster for
-- videos); videos add one row per sampled frame. The frame 0 row is also the scene embedding.
CREATE TABLE IF NOT EXISTS item_embeddings (
  item_id      INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  version      INTEGER NOT NULL,
  model        TEXT NOT NULL,
  frame_index  INTEGER NOT NULL DEFAULT 0,
  frame_time_s REAL,
  embedding    BLOB NOT NULL,                  -- float32[dim], L2-normalised
  created_at   TEXT NOT NULL,
  PRIMARY KEY (item_id, model, frame_index)
);
CREATE INDEX IF NOT EXISTS item_embeddings_model ON item_embeddings(model, item_id);

-- Zero-shot labels kept for display and filters.
CREATE TABLE IF NOT EXISTS item_labels (
  item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  label   TEXT NOT NULL,
  score   REAL NOT NULL,
  PRIMARY KEY (item_id, label)
);

-- A person = a cluster of faces (and confirmed bodies). name NULL means not named yet.
CREATE TABLE IF NOT EXISTS persons (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT,
  name_key      TEXT,
  cover_face_id INTEGER REFERENCES faces(id) ON DELETE SET NULL,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS persons_name ON persons(name_key);

CREATE TABLE IF NOT EXISTS faces (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,   -- never reused (crop file names)
  item_id         INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  person_id       INTEGER REFERENCES persons(id) ON DELETE SET NULL,
  assigned_by     TEXT NOT NULL DEFAULT 'auto' CHECK (assigned_by IN ('auto','user','guess')),
                                                -- 'user' = locked; 'guess' = person_id NULL, guess_* set
  x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,  -- normalised [0,1], display orientation
  det_score       REAL NOT NULL,
  size_px         INTEGER NOT NULL,
  quality         REAL NOT NULL,
  frame_time_s    REAL,
  spans           TEXT,                         -- video: JSON [[start, end], ...]
  embedding       BLOB NOT NULL,                -- float32[512], L2-normalised (ArcFace)
  torso_embedding BLOB,                         -- float32[768], L2-normalised (SigLIP crop of head and torso)
  guess_person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
  guess_score     REAL,
  guess_evidence  TEXT,                         -- JSON {clothes, scene, face, ref_face_id, ref_item_uid, same_day, gps_m}
  created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS faces_item   ON faces(item_id);
CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id, quality);
CREATE INDEX IF NOT EXISTS faces_guess  ON faces(guess_person_id);

-- Person boxes (YOLOX): a body may carry the face inside it; faceless bodies are matched by clothes.
CREATE TABLE IF NOT EXISTS bodies (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id         INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  face_id         INTEGER REFERENCES faces(id) ON DELETE SET NULL,
  person_id       INTEGER REFERENCES persons(id) ON DELETE SET NULL,
  assigned_by     TEXT NOT NULL DEFAULT 'auto' CHECK (assigned_by IN ('auto','user','guess')),
  x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
  det_score       REAL NOT NULL,
  torso_embedding BLOB,
  guess_person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
  guess_score     REAL,
  guess_evidence  TEXT,
  frame_time_s    REAL,
  created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS bodies_item   ON bodies(item_id);
CREATE INDEX IF NOT EXISTS bodies_person ON bodies(person_id);
CREATE INDEX IF NOT EXISTS bodies_guess  ON bodies(guess_person_id);

-- People in an item without a usable face box (manual tags, confirmed faceless bodies).
CREATE TABLE IF NOT EXISTS item_tags (
  item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  source     TEXT NOT NULL DEFAULT 'user' CHECK (source IN ('user','guess')),
  created_at TEXT NOT NULL,
  PRIMARY KEY (item_id, person_id)
);
CREATE INDEX IF NOT EXISTS item_tags_person ON item_tags(person_id);

-- "That is not this person": a rejected guess is never suggested again for the same box.
CREATE TABLE IF NOT EXISTS guess_rejections (
  kind       TEXT NOT NULL CHECK (kind IN ('face','body')),
  ref_id     INTEGER NOT NULL,
  person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  PRIMARY KEY (kind, ref_id, person_id)
);

-- "These two are different people": hides a merge suggestion.
CREATE TABLE IF NOT EXISTS merge_dismissals (
  person_a   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  person_b   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  PRIMARY KEY (person_a, person_b),
  CHECK (person_a < person_b)
);

-- Extracted text of an item (all pages, separated by form feeds).
CREATE TABLE IF NOT EXISTS item_text (
  item_id    INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
  text       TEXT NOT NULL,
  source     TEXT NOT NULL CHECK (source IN ('pdf','ocr','office','plain','mixed')),
  lang       TEXT,
  char_count INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

-- Pages of a document (PDF pages, logical pages of office files, the single page of an image with text).
CREATE TABLE IF NOT EXISTS pages (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id     INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  page_no     INTEGER NOT NULL,               -- 1-based
  width       INTEGER,                        -- of the render
  height      INTEGER,
  text        TEXT,
  text_source TEXT NOT NULL DEFAULT 'none' CHECK (text_source IN ('layer','ocr','render','none')),
  ocr_boxes   TEXT,                           -- JSON [[x0,y0,x1,y1,text,score], ...] normalised to the render
  UNIQUE (item_id, page_no)
);

-- Full-text search. rowid = items.id. Rewritten by app.documents.extract.refresh_fts.
-- tokenchars lists every combining mark (Unicode Mn and Mc) from U+0900 to U+0DFF (Devanagari
-- through Sinhala: vowel signs, virama, nukta, anusvara, visarga, candrabindu), so a Hindi or
-- Marathi word is one token instead of being split at each vowel sign. The same literal is
-- app.db.FTS_TOKENIZE; app.db.init_db rebuilds the table when a database has another definition.
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
  filename, text, doc, people, labels,
  tokenize = 'unicode61 remove_diacritics 2 tokenchars ''ऀँंःऺऻ़ािीुूृॄॅॆेैॉॊोौ्ॎॏ॒॑॓॔ॕॖॗॢॣঁংঃ়ািীুূৃৄেৈোৌ্ৗৢৣ৾ਁਂਃ਼ਾਿੀੁੂੇੈੋੌ੍ੑੰੱੵઁંઃ઼ાિીુૂૃૄૅેૈૉોૌ્ૢૣૺૻૼ૽૾૿ଁଂଃ଼ାିୀୁୂୃୄେୈୋୌ୍୕ୖୗୢୣஂாிீுூெேைொோௌ்ௗఀఁంఃఄ఼ాిీుూృౄెేైొోౌ్ౕౖౢౣಁಂಃ಼ಾಿೀುೂೃೄೆೇೈೊೋೌ್ೕೖೢೣೳഀഁംഃ഻഼ാിീുൂൃൄെേൈൊോൌ്ൗൢൣඁංඃ්ාැෑිීුූෘෙේෛොෝෞෟෲෳ'''
);

-- Parsed "virtual document" of an item.
CREATE TABLE IF NOT EXISTS documents (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id        INTEGER NOT NULL UNIQUE REFERENCES items(id) ON DELETE CASCADE,
  doc_type       TEXT NOT NULL,                -- aadhaar, pan, passport, ... , event_pass, other
  doc_group      TEXT NOT NULL CHECK (doc_group IN ('id','money','education_work','health','pass','other')),
  confidence     REAL NOT NULL,
  fields         TEXT NOT NULL,                -- JSON {key: {value, label, confidence}}
  title          TEXT,
  doc_date       TEXT,                         -- ISO date when known
  issuer         TEXT,
  person_name    TEXT,
  barcode        TEXT,
  barcode_format TEXT,
  has_photo      INTEGER NOT NULL DEFAULT 0,
  source         TEXT NOT NULL DEFAULT 'auto' CHECK (source IN ('auto','user')),
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_type ON documents(doc_type);

-- Tags the owner adds, and that the system learns (docs/ARCHITECTURE.md section 19).
CREATE TABLE IF NOT EXISTS tags (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  name       TEXT NOT NULL,                  -- as typed, 1..40 chars
  name_key   TEXT NOT NULL UNIQUE,           -- util.name_key(name)
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS item_user_tags (
  item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  tag_id     INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
  source     TEXT NOT NULL DEFAULT 'user' CHECK (source IN ('user','auto')),
  score      REAL,                           -- the learner's score for auto rows
  created_at TEXT NOT NULL,
  PRIMARY KEY (item_id, tag_id)
);
CREATE INDEX IF NOT EXISTS item_user_tags_tag ON item_user_tags(tag_id, item_id);
CREATE TABLE IF NOT EXISTS tag_rejections (   -- a suggestion the owner removed: a counter-example
  item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  tag_id     INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  PRIMARY KEY (item_id, tag_id)
);

-- Corrections and outcomes the learners need beyond the main tables (docs/ARCHITECTURE.md section 18).
CREATE TABLE IF NOT EXISTS learning_events (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  kind       TEXT NOT NULL CHECK (kind IN ('guess','category','vault','misclick','doc_type','tag')),
  label      INTEGER,                       -- 1 / 0 for outcomes, NULL for a plain correction
  features   TEXT NOT NULL DEFAULT '{}',    -- JSON: evidence terms, or {from, to, scores}
  item_id    INTEGER,                       -- kept after the item is gone
  person_id  INTEGER,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS learning_events_kind ON learning_events(kind, created_at);

-- Durable background jobs. Handlers must be idempotent.
CREATE TABLE IF NOT EXISTS jobs (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  type       TEXT NOT NULL CHECK (type IN ('ingest','analyze','text','video_preview','guess','recluster')),
  queue      TEXT NOT NULL DEFAULT 'main' CHECK (queue IN ('main','text','video')),
  payload    TEXT NOT NULL DEFAULT '{}',
  status     TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','error')),
  attempts   INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  priority   INTEGER NOT NULL DEFAULT 100,
  run_after  TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_pick ON jobs(queue, status, priority, id);
-- enqueue(unique=True) looks for a live job with the same type and payload.
CREATE INDEX IF NOT EXISTS jobs_dedup ON jobs(type, payload) WHERE status IN ('queued', 'running');
