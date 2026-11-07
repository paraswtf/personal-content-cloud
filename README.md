# Personal Content Cloud

A private cloud for one person's photos, videos, PDFs and other documents. Everything runs on this
machine: files stay on local disk, the models run on the local GPU (or CPU), and nothing is sent to
outside services. Every upload is kept byte for byte and downloads return the original file; the
browser sees generated previews.

What it does with each file:

- **Any format.** JPEG, PNG, HEIC, WebP, camera RAW, GIF, video (MP4, MOV, MKV and more), PDF, DOCX,
  XLSX, PPTX and plain text. Anything else is stored and listed without a preview.
- **Previews.** WebP thumbnails and large previews, 720p H.264 streaming copies of videos, page
  renders of documents.
- **Categories.** Each item is sorted automatically into documents, screenshots, personal (people),
  places, food, pets, things, other, or the private (NSFW) vault. Any category can be overridden.
- **Mis-clicks.** Blurry, shaken, too dark, blown out, pocket and ground shots and burst duplicates
  are flagged into a review bin. Nothing is deleted on its own.
- **Faces and people.** Faces are found in photos and in video frames (about one per second),
  grouped into people, and named once. People are also guessed from earlier photos of the same day,
  the same scene and the same clothes; guesses wait for a confirm or reject.
- **Documents.** Text is extracted (text layer or OCR, in English, Hindi and Marathi) and searchable.
  Aadhaar, PAN, passport, driving licence, voter ID, vehicle RC, bank statements, invoices, receipts,
  salary slips, marksheets, degrees, certificates, offer letters, insurance policies, prescriptions,
  lab reports, vaccination certificates, event passes and boarding passes get a parsed "virtual
  document" card with their fields, from English, Hindi or Marathi labels. A document in Devanagari
  shows its language, and a name printed in both scripts keeps both. Event and boarding passes can be
  exported to Apple Wallet.
- **Search.** One box searches by meaning ("beach at sunset", "person in a red shirt") and by text
  (OCR, file names, people, document fields), with filters for category, kind, people and dates.
- **Vault.** NSFW items never appear in the library, search, people pages or previews. The vault
  asks for the password or a passkey again and stays open for 15 minutes.
- **Ways in.** Drag and drop in the browser, the camera inside the app, and the phone's Share menu
  (install the app from the browser menu first).
- **Learning from your choices.** Corrections to categories, the vault, mis-clicks, document types
  and people teach small local models, so the next automatic decision of the same kind follows them.
- **Your own tags.** Add free-form tags such as "wedding" or "Goa trip" to any item, and the same
  tags are suggested on look-alike photos, videos and similar documents for you to accept or remove.

### Learning from your choices

Every correction you make (a category or vault change on an item, a kept or deleted mis-click, a
document type or field fix, a face assignment, a confirmed or rejected guess) is a training example.
No deep model is retrained and nothing leaves the machine: small models are fitted on top of the
embeddings and scores the pipeline already has, and kept in `DATA_DIR/learned/`. Training runs every 10
minutes when there are new corrections, or at once from Settings, Learning, which also shows what was
learned, applies it to items already in the library and resets it. Reset deletes only the learned
models, never your choices. The switch there, or `LEARNING_ENABLED=false`, turns learning off.

## Running it

From the portal's **Dev servers** tab, start **web** (declared in `.claude/devservers.json`). On a
fresh checkout the first start installs dependencies, downloads the models and builds the frontend,
which takes a few minutes. Manually:

```bash
./scripts/run.sh                 # PORT=41010 HOST=127.0.0.1 by default
```

Requirements: Python 3.12 via [uv](https://docs.astral.sh/uv/), Node 20+, `ffmpeg` and `ffprobe` on
the PATH. An NVIDIA GPU with a current driver is used when present (CUDA libraries come from pip);
otherwise everything runs on the CPU.

### First run

Open the site and create the account (username and password). Then add a passkey under Settings
so the phone and laptop can sign in with a fingerprint or face. Passkeys need HTTPS (the dev server's
public URL) or localhost. Adding or removing a passkey asks for the password (or an existing passkey)
again: that is the vault unlock, and it stays open for 15 minutes.

## Configuration

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `./data` | Database and all uploaded files. Put this on a large disk. |
| `MODELS_DIR` | `./models` | Downloaded models. |
| `PORT`, `HOST` | `41010`, `127.0.0.1` | Where the server listens. |
| `SITE_TITLE` | `Personal Content Cloud` | |
| `RP_ID`, `ORIGIN` | derived from the request | WebAuthn relying party (set them when behind a proxy). |
| `SESSION_DAYS` | `30` | Sliding session length. |
| `VAULT_UNLOCK_MIN` | `15` | How long the vault stays open. |
| `TRASH_DAYS` | `30` | Deleted items wait this long before they are purged. |
| `LOCAL_TIMEZONE` | `Asia/Kolkata` | Time zone for camera times without an offset and for "same day". |
| `GPU_ENABLED` | `auto` | `auto`, `true` or `false`. |
| `GPU_ARENA_SHRINK` | `true` | Give unused GPU memory back after every model run (keeps the server from growing to the card's size). |
| `MAX_UPLOAD_MB`, `CHUNK_SIZE_MB`, `MIN_FREE_DISK_MB` | `16384`, `8`, `2048` | Upload limits and the free-space floor. |
| `NSFW_THRESHOLD` | `0.7` | Classifier probability at which an item goes to the vault. |
| `DOC_LIKENESS_MIN` | `0.55` | Images at or above this are sent to OCR. |
| `BLUR_VAR_MIN`, `BURST_WINDOW_S` | `40`, `3` | Mis-click thresholds. |
| `GUESS_WINDOW_H`, `GUESS_SCENE_SIM`, `GUESS_CLOTHES_SIM`, `GUESS_GPS_M` | `12`, `0.80`, `0.85`, `250` | Person guessing. |
| `WALLET_PASS_TYPE_ID`, `WALLET_TEAM_ID`, `WALLET_ORG_NAME` | unset | Apple Wallet passes (see below). |
| `OCR_ENABLED`, `OCR_MAX_PAGES` | `true`, `50` | OCR runs in English, Hindi and Marathi when the Devanagari model is present. |

The full list, with the face, video, preview and search settings carried over from the alumni hub,
is in `backend/app/config.py` and `docs/ARCHITECTURE.md`.

### Apple Wallet

Passes are signed with an Apple Pass Type ID certificate from a paid developer account. Put
`cert.pem` (the Pass Type ID certificate), `key.pem` (its private key) and `wwdr.pem` (Apple's WWDR
G4 certificate) in `DATA_DIR/wallet/`, set `WALLET_PASS_TYPE_ID` (`pass.com.example.cloud`),
`WALLET_TEAM_ID` and `WALLET_ORG_NAME`, and restart. Settings shows what is still missing.

## Models

All models are downloaded once by `scripts/download_models.py` (run by `scripts/run.sh` on every
start, a no-op when they are present) into `MODELS_DIR`:

| Model | Use | Source | License |
|---|---|---|---|
| InsightFace buffalo_l (SCRFD-10GF, ArcFace R50) | face detection and recognition | InsightFace release v0.7 | non-commercial research |
| SigLIP ViT-B/16 (WebLI), ONNX export by Immich | search, categories, scene and clothes similarity | Hugging Face immich-app | Apache 2.0 |
| Falconsai NSFW ViT, ONNX export by onnx-community | vault classifier | Hugging Face | Apache 2.0 |
| YOLOX-S | person boxes for clothes matching | Megvii YOLOX release | Apache 2.0 |
| PP-OCRv6 small models bundled with rapidocr | OCR, English | RapidAI | Apache 2.0 |
| PP-OCRv5 Devanagari mobile recogniser | OCR, Hindi and Marathi | RapidOCR model zoo | Apache 2.0 |
| Noto Sans Devanagari | page renders of Hindi and Marathi text | Noto fonts | SIL OFL 1.1 |

## Data, backups and disk

All state lives in `DATA_DIR`: `cloud.db` (SQLite, WAL mode), `originals/` (untouched files),
`thumbs/`, `previews/`, `pages/`, `videos/`, `faces/`, `bodies/`, `docphoto/` (generated, rebuildable),
`staging/` and `failed/` (uploads), `wallet/` (your certificates), `tmp/`. Deleted items stay in the
trash for `TRASH_DAYS` days. To back up, stop the server and copy `DATA_DIR`, or use
`sqlite3 cloud.db ".backup copy.db"` plus a copy of `originals/` while it runs.

## Development

```bash
cd backend && uv sync && PCC_NO_MODELS=1 .venv/bin/python -m pytest -q   # fast tests, no models
cd backend && ORT_THREADS=2 .venv/bin/python -m pytest -q                 # with the models
cd frontend && npm ci && npm run build                                    # typecheck + production build
BACKEND_URL=http://127.0.0.1:41010 npm run dev                            # Vite dev server against a running API
```

Layout: `backend/app` (FastAPI app; `api/` routes, `auth/`, `media/`, `ml/`, `analysis/`, `documents/`,
`search/`, `wallet/`, `ingest.py`, `worker.py`), `frontend/src` (React pages and components),
`docs/ARCHITECTURE.md` (the contract), `docs/alumni-hub-reuse-map.md` (what came from the alumni hub).
