# Personal Content Cloud

Source: <https://github.com/paraswtf/personal-content-cloud>

A private cloud for one person's photos, videos, PDFs and other documents. Everything runs on this
machine: files stay on local disk, the models run on the local GPU (or CPU), and nothing is sent to
outside services. Every upload is kept byte for byte and downloads return the original file; the
browser sees generated previews.

What it does with each file:

- **Any format.** JPEG, PNG, HEIC, WebP, camera RAW, GIF, video (MP4, MOV, MKV and more), PDF, DOCX,
  XLSX, PPTX and plain text. Anything else is stored and listed without a preview.
- **Previews.** WebP thumbnails and large previews, page renders of documents, and 1080p H.264 web
  copies of videos the browser cannot play as they are, made with the machine's hardware encoder
  (NVENC, VAAPI or Quick Sync) when there is one.
- **Categories.** Each item is sorted automatically into documents, screenshots, personal (people),
  places, food, pets, things, other, or the private (NSFW) vault. Any category can be overridden.
- **Mis-clicks.** Blurry, shaken, too dark, blown out, pocket and ground shots and burst duplicates
  are flagged into a review bin. Nothing is deleted on its own.
- **Faces and people.** Faces are found in photos and in video frames (about one per second),
  grouped into people, and named once. People are also guessed from earlier photos of the same day,
  the same scene and the same clothes; guesses wait for a confirm or reject.
- **Tattoos, nails and mehndi.** People boxes whose face does not show are also guessed from a tattoo
  seen in their other photos at any date (checked point by point, not just by look), and helped by the
  same nail polish or mehndi a few days apart. Each guess says why and shows the two crops side by side;
  "Not a tattoo" and "Not this person's" correct it, and a person's page lists their tattoos.
- **Recognition trained on one person.** Once you have gone through a person's faces, their page can
  train a small model on that person alone: their faces as examples, the faces you took off them as
  counterexamples. It then looks at every other face and offers the ones that look like them as guesses,
  and a new photo of them is offered the same way instead of going to a look-alike. Nothing is tagged
  until you confirm. The card says how the model did on days it was not trained on, and lists the faces
  still on the person that look like ones you removed.
- **Following people between photos.** Photos taken moments apart at one place are lined up by their
  background, so someone who stays in the same spot is followed from one photo to the next, also when the
  camera moved or they turned away. It counts as a reason for a guess, along with clothes.
- **3D heads.** A person's page can build a 3D head for each look found in their photos (an age, a hair
  length, facial hair or none), with models that run on this machine, and only when you press Generate.
  The front of a head is the photo itself, laid over a smoothed shape whose nose, lips and chin are read
  from the photo by a depth model. Sliders pick the look, turn the head and move the light. Each head
  comes from one photo: past about 50 degrees the sides and the back are filled-in colours, so it is a
  reading of that photo, not a scan.
- **Documents.** Text is extracted (text layer or OCR, in English, Hindi and Marathi) and searchable.
  Traffic challans and parking receipts are read into their own cards: a challan shows its number,
  vehicle, violations, amount and printed payment status with a button that opens the right portal to pay
  or check it (the portals sit behind a captcha, so the status is not fetched automatically).
  Aadhaar, PAN, passport, driving licence, voter ID, vehicle RC, bank statements, invoices, receipts,
  salary slips, marksheets, degrees, certificates, offer letters, insurance policies, prescriptions,
  lab reports, vaccination certificates, event passes and boarding passes get a parsed "virtual
  document" card with their fields, from English, Hindi or Marathi labels. A document in Devanagari
  shows its language, and a name printed in both scripts keeps both. Event and boarding passes can be
  exported to Apple Wallet.
- **Search.** One box searches by meaning ("beach at sunset", "person in a red shirt") and by text
  (OCR, file names, people, document fields), with filters for category, kind, people and dates.
- **Vault.** NSFW items never appear in the library, search, people pages or previews. The vault
  asks for the password or a passkey again and stays open for 15 minutes. A sensitivity setting in
  Settings, Learning (low, normal or high) decides how readily photos go there, and moving photos
  into or out of the vault tunes it further.
- **Pets.** Cats, dogs, birds and other animals get a box like faces do; crops of the same animal become a
  pet you can name, and the People page lists your pets with their photos.
- **Ways in.** Drag and drop in the browser, the camera inside the app, and the phone's Share menu
  (install the app from the browser menu first).
- **Forwards.** Media that came through WhatsApp or another messaging app is recognised (the app's
  filename, stripped camera metadata, WhatsApp's sizes and JPEG quality, and a model's view of the content),
  memes, greetings, posters and screenshots of posts are set apart under Review > Forwards, and a real photo
  received the same way stays in the library with its source noted. Keeping or marking one teaches the rule.
- **Learning from your choices.** Corrections to categories, the vault, mis-clicks, document types
  and people teach small local models, so the next automatic decision of the same kind follows them.
  Taking a face away from a person takes the faces that look like it along, and they never come back
  to that person; a person is matched against every face they have, your own assignments counting more.
- **Your own tags.** Add free-form tags such as "wedding" or "Goa trip" to any item, and the same
  tags are suggested on look-alike photos, videos and similar documents for you to accept or remove.
- **A timeline that scales.** The library is laid out by month with the whole library's shape known up
  front, renders only what is on screen, and has a scrubber to jump to any month; a hundred thousand
  photos scroll like a hundred.
- **Albums.** Hand-picked sets with a cover, a description and the library's filters inside them;
  add from any selection or from the viewer.
- **Map and places.** Photos with GPS appear on a map (an offline basemap by default, detailed
  online tiles when you switch them on), clustered, with a list of places; place names come from an
  offline gazetteer and work in text search ("Goa").
- **Share links.** An album or a hand-picked set gets a link for someone without an account, with an
  optional password, an expiry and a download switch; vault items are never shared.
- **Phone sync that survives.** The iOS app uploads in the background, continues an interrupted
  upload where it stopped after a restart, and only offers to delete from the phone what the server
  confirms it holds.

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
git clone https://github.com/paraswtf/personal-content-cloud.git
cd personal-content-cloud
./scripts/run.sh                 # PORT=41010 HOST=127.0.0.1 by default
```

The server serves a copy of the frontend build taken when it started (`DATA_DIR/site`), so building
the frontend again does not change a running server. After pulling or editing code, restart the
server from the portal: it rebuilds the frontend when sources changed. If a page ever shows
"This site and the server were started from different versions", that restart is what it needs.

Requirements: Python 3.12 via [uv](https://docs.astral.sh/uv/), Node 20+, `ffmpeg` and `ffprobe` on
the PATH. An NVIDIA GPU with a current driver is used when present (CUDA libraries come from pip);
otherwise everything runs on the CPU.

### First run

Open the site and create the account (username and password). Then add a passkey under Settings
so the phone and laptop can sign in with a fingerprint or face. Passkeys need HTTPS (the dev server's
public URL) or localhost. Adding or removing a passkey asks for the password (or an existing passkey)
again: that is the vault unlock, and it stays open for 15 minutes.

## Configuration

Private settings go in `.env` at the repository root, which git ignores and `scripts/run.sh` loads before
the server starts: the Apple Wallet values (`WALLET_PASS_TYPE_ID`, `WALLET_TEAM_ID`, `WALLET_ORG_NAME`),
another state's `CHALLAN_PORTAL_URL`, and anything else you would not want in a public repository. Keep
them out of `.claude/devservers.json` and `ios/project.yml` (set your Apple team in Xcode under Signing).
`.gitignore` also keeps out `data/` (the database, the originals, the share-link key, the Wallet
certificates), `models/`, every key, certificate, provisioning profile and pass file, local editor and
tool state, and the generated Xcode project.

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `./data` | The database and everything made from your files (previews, thumbnails, streaming copies). |
| `ORIGINALS_DIR` | same as `DATA_DIR` | Where the uploaded files themselves live. Point it at a fast or large disk; an existing library is moved over in the background after a restart. |
| `BACKUP_DIR`, `BACKUP_HOUR`, `BACKUP_KEEP` | off, `3`, `14` | A snapshot of the database, the originals and the learned models every night at that local hour, on another disk; the newest `BACKUP_KEEP` are kept and unchanged files are shared between them. |
| `MODELS_DIR` | `./models` | Downloaded models. |
| `PORT`, `HOST` | `41010`, `127.0.0.1` | Where the server listens. |
| `SITE_TITLE` | `Personal Content Cloud` | |
| `RP_ID`, `ORIGIN` | derived from the request | WebAuthn relying party (set them when behind a proxy). `ORIGIN` may list several origins, comma-separated, so passkeys work on custom domains too; the first is the server's own address. |
| `SESSION_DAYS` | `30` | Sliding session length. |
| `VAULT_UNLOCK_MIN` | `15` | How long the vault stays open. |
| `TRASH_DAYS` | `30` | Deleted items wait this long before they are purged. |
| `LOCAL_TIMEZONE` | `Asia/Kolkata` | Time zone for camera times without an offset and for "same day". |
| `GPU_ENABLED` | `auto` | `auto`, `true` or `false`. |
| `GPU_ARENA_SHRINK` | `true` | Give unused GPU memory back after every model run (keeps the server from growing to the card's size). |
| `MAX_UPLOAD_MB`, `CHUNK_SIZE_MB`, `MIN_FREE_DISK_MB` | `16384`, `8`, `2048` | Upload limits and the free-space floor. |
| `NSFW_THRESHOLD` | `0.7` | Classifier cut of the fallback rule (used only without the SigLIP signals). |
| `VAULT_SENSITIVITY` | `normal` | `low`, `normal` or `high`: how readily photos go to the vault (also set in Settings > Learning). |
| `DOC_LIKENESS_MIN` | `0.55` | Images at or above this are sent to OCR. |
| `BLUR_VAR_MIN`, `BURST_WINDOW_S` | `40`, `3` | Mis-click thresholds. |
| `GUESS_WINDOW_H`, `GUESS_SCENE_SIM`, `GUESS_CLOTHES_SIM`, `GUESS_GPS_M` | `1`, `0.80`, `0.85`, `250` | Person guessing by clothes: only photos taken within this many hours before or after a tagged one. |
| `TRAITS_ENABLED`, `TRAIT_MATCHING` | `true`, `true` | Read tattoos, mehndi and nails from people boxes, and use them in guesses (off: guesses exactly as before). |
| `TRAIT_ONLY_GUESSES`, `TRAIT_ONLY_MEHNDI` | `true`, `false` | Allow a guess from a tattoo alone (under strict rules); the same from mehndi alone. |
| `TRAITS_MEHNDI` | `false` | Read mehndi on hands and forearms. Off by default: the detector mistook plain hands for mehndi. |
| `TRAITS_THREADS` | `2` | CPU threads per traits model. |
| `PLACE_ENABLED`, `PLACE_ONLY_GUESSES` | `true`, `true` | Line up photos taken moments apart and use where people stood in guesses, and allow a guess from the position alone. |
| `PLACE_THREADS` | `1` | OpenCV threads while photos are lined up. 1 keeps that work at low priority on one core; more is faster and competes with the rest. |
| `TRAIT_MIN_BODY_PX`, `TRAIT_MIN_LIMB_PX`, `NAIL_MIN_HAND_PX` | `200`, `64`, `96` | Smallest people box, limb and hand (pixels) that is read. |
| `TATTOO_GATE_P`, `TATTOO_MIN_INLIERS`, `MEHNDI_MIN_INLIERS` | `0.80`, `12`, `30` | How sure the tattoo check must be, and how many matching points a tattoo or mehndi pair needs. |
| `TRAIT_MARGIN_LLR`, `TRAIT_PRIOR_PEOPLE`, `TRAIT_MIN_P` | `2.0`, `20`, `0.90` | The lead a guess from a tattoo alone needs over the next person, the prior used for its sort score, and the floor of the tightening from your answers. |
| `NAILS_MAX_DAYS`, `MEHNDI_MAX_DAYS` | `14`, `21` | Nails and mehndi are only compared within this many days. |
| `TRAIT_PROPAGATE_MAX` | `400` | Photos one tagging queues for a new look by tattoos. |
| `GEN3D_DIR`, `GEN3D_GENERATOR` | `MODELS_DIR/gen3d`, `auto` | The 3D head generator's own environment (see Models, 3D heads) and which backend it uses. |
| `GEN3D_SANDBOX` | `auto` | Run the generator in a sandbox with no network (`bwrap`): `auto` when it is installed, `on` to insist, `off`. |
| `GEN3D_MAX_RSS_MB`, `GEN3D_LOOK_TIMEOUT_S` | `5000`, `600` | The generator is stopped above this much memory (near it, a head is finished without Face relief instead), or when one head takes longer than this many seconds. |
| `FORWARDS_ENABLED`, `FORWARD_MAX_SIDE`, `FORWARD_JPEG_QUALITY_MAX`, `FORWARD_MEME_MIN` | `true`, `1600`, `82`, `0.5` | Forwards: the longest side and JPEG quality that mark an image as received through an app, and the content score that sets it apart. |
| `PETS_ENABLED`, `ANIMAL_DET_THRESHOLD` | `true`, `0.45` | Pet identities (section 22) and the animal detection cut. |
| `PET_MATCH_THRESHOLD`, `PET_NEW_THRESHOLD` | `0.82`, `0.86` | Cosine cuts for matching a crop to a known pet and for pairing two crops into a new pet. |
| `WALLET_PASS_TYPE_ID`, `WALLET_TEAM_ID`, `WALLET_ORG_NAME` | unset | Apple Wallet passes (see below). |
| `OCR_ENABLED`, `OCR_MAX_PAGES` | `true`, `50` | OCR runs in English, Hindi and Marathi when the Devanagari model is present. |
| `VIDEO_ENCODER`, `VAAPI_DEVICE` | `auto`, `/dev/dri/renderD128` | Web copy encoder: `auto` tries `h264_nvenc`, `h264_vaapi`, `h264_qsv`, then `libx264`; Settings > System shows the one in use. |
| `VIDEO_PREVIEW_MAX_HEIGHT`, `VIDEO_PREVIEW_MAXRATE_KBPS`, `VIDEO_PREVIEW_CRF`, `VIDEO_STREAM_MAX_KBPS` | `1080`, `6000`, `23`, `8000` | The web copy rendition, and the bitrate up to which H.264 originals play as they are. |
| `GEO_ENABLED` | `true` | Download the GeoNames gazetteer and name the places of photos with GPS. |
| `MAP_TILES`, `MAP_TILE_STYLE` | `offline`, OpenFreeMap liberty | The map's default basemap and the online style used when detailed tiles are switched on. |

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
| GeoNames cities15000, admin1 and country tables | place names for photos with GPS (`MODELS_DIR/geo`) | geonames.org | CC BY 4.0 |

The map's offline basemap (`frontend/public/map/basemap.json`) is Natural Earth 1:110m land, lakes and
boundaries (public domain). Detailed online tiles, when switched on, come from OpenFreeMap (OpenMapTiles
with OpenStreetMap data); photos never leave the server either way.

### 3D heads (optional)

The 3D heads use a generator that is a separate program with its own Python environment, so PyTorch stays
out of the server. Until it is set up, the 3D tab says that the generator is not installed and nothing else
changes. Set it up once:

```bash
scripts/setup_3d.sh
```

It installs PyTorch and a few libraries into their own environment (about 6 GB), fetches the TripoSR code
at a pinned commit and downloads its files, each checked against its checksum: the TripoSR weights (1.7
GB, MIT licence, Tripo AI and Stability AI), the U2-Net background cutter (176 MB, Apache 2.0), one small
settings file of the image encoder (facebook/dino-vitb16, Apache 2.0), and the two DAViD depth models that
shape the face (449 MB each, MIT licence, Microsoft). About 9.4 GB in all; to keep it on another disk,
make `models/gen3d` a link to a folder there first. The depth models are optional: when their download
fails the script says so and goes on, and heads are made without Face relief until you run it again. If
you set the generator up before this version, run the script once more to fetch them; finished steps are
skipped. `scripts/setup_3d.sh --check` says what is installed.

A head is built in two steps. The image-to-3D model makes a rough shape: on an RTX 2080, 3 to 6 seconds a
head and 2.4 GB of GPU memory. Then the shape is finished on the processor, in about 10 seconds a head: it
is smoothed, a depth model shapes the nose, lips and chin from the photo, and the photo itself is laid
over the front as the head's texture. Generate has options for this: Smoothing, Photo texture (off, or the
photo at up to 1024 px, up to 2048 px or at its own size; it is never enlarged) and Face relief. In the
viewer, the Photo switch shows the plain colours under the photo.

A head comes from one photo. From the front and up to about 40 degrees to either side it is that photo;
further round, the sides and the back are filled in from its colours, without detail. Glasses, or a hand
in front of the face, are painted onto the surface. Heads made before this version keep showing, with a
line that says so: press "Generate again" on the person's 3D tab to get new ones.

A head file holds the photo's own pixels. It is served only to a signed-in browser, which asks the server
before every use, and it follows its photo: a head stops showing when its photo goes to the trash or the
vault, and its file is removed once the photo is gone for good.

The generator lives in `MODELS_DIR/gen3d` (`GEN3D_DIR`). It runs only while a model is being made, inside
a sandbox (`bwrap`), because it runs model code written by others on your face photos. Inside it the
generator sees the system's libraries, its own files, the GPU and the pictures of the job in hand, and
nothing else: no network, no home folder, no library, none of the server's settings.
`GEN3D_SANDBOX=off` turns the sandbox off;
`docs/ARCHITECTURE.md` section 36 has the details.

## Data, backups and disk

Three places, three jobs. `DATA_DIR` holds `cloud.db` (SQLite, WAL mode) and everything that can be rebuilt
from your files: `thumbs/`, `previews/`, `pages/`, `videos/`, `faces/`, `bodies/`, `docphoto/`, the 3D
heads in `people3d/`, the learned models in `learned/`, your Wallet certificates in `wallet/`, the served
copy of the frontend in `site/` and `tmp/`. `ORIGINALS_DIR` (the same folder unless you set it) holds the uploaded files themselves in
`originals/` and the uploads on their way in (`staging/`, `failed/`); put it on your fast or biggest disk.
`BACKUP_DIR`, when set, gets a snapshot every night at `BACKUP_HOUR`: the database (a consistent online
copy), the originals, the share-link key, the learned models and the Wallet certificates, as
`snapshots/<date>T<time>/` folders where unchanged originals are hard links to the night before, so each
night costs only what changed; the newest `BACKUP_KEEP` snapshots are kept and `latest` points at the
newest. Settings > System shows each disk's free space, the last and the next backup, and has a "Back up
now" button. Deleted items stay in the trash for `TRASH_DAYS` days.

To restore, stop the server, copy `cloud.db`, `secret.key`, `learned/` and `wallet/` from a snapshot into
`DATA_DIR` (delete a stale `cloud.db-wal` and `cloud.db-shm` first) and `originals/` into `ORIGINALS_DIR`,
then start it; previews that are missing are rebuilt when you choose "Process again" on an item.

## Development

```bash
cd backend && uv sync && PCC_NO_MODELS=1 .venv/bin/python -m pytest -q   # fast tests, no models
cd backend && ORT_THREADS=2 .venv/bin/python -m pytest -q                 # with the models
cd frontend && npm ci && npm run build                                    # typecheck + production build
BACKEND_URL=http://127.0.0.1:41010 npm run dev                            # Vite dev server against a running API
```

Layout: `backend/app` (FastAPI app; `api/` routes, `auth/`, `media/`, `ml/`, `analysis/`, `documents/`,
`search/`, `wallet/`, `ingest.py`, `worker.py`), `frontend/src` (React pages and components), `gen3d`
(the 3D head generator, a program of its own),
`docs/ARCHITECTURE.md` (the contract), `docs/alumni-hub-reuse-map.md` (what came from the alumni hub).

## License

[PolyForm Noncommercial License 1.0.0](LICENSE). You may use, copy, change and share this software
for any noncommercial purpose: personal use, research, education, hobby projects, and use by
charities, educational institutions and other noncommercial organizations. Commercial use of any
kind needs the author's written permission. The models and data the server downloads (section
Models above) keep their own licences, as do the GeoNames and Natural Earth data.
