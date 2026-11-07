# Documents area: integration notes

Files: `backend/app/documents/{extract,detect,fields,barcodes,photo}.py`, `backend/app/documents/parsers/*.py`
(21 types plus `_base.py`), `backend/app/wallet/pkpass.py`, `backend/app/api/documents.py`,
`backend/tests/test_documents_{parsers,extract,api,ocr}.py`, `backend/tests/test_wallet_pkpass.py`,
`backend/tests/fixtures/documents/` (three PNGs and `make_fixtures.py`, which regenerates them).

## Changes suggested to shared contract files

1. `config.py`: add `wallet_key_password` (env `WALLET_KEY_PASSWORD`). Apple pass certificate keys
   exported from Keychain are usually passphrase-protected. Until then `wallet/pkpass.py` reads
   `os.environ["WALLET_KEY_PASSWORD"]` directly.
2. `ml/models.py` (ml area): `present()` ignores `PCC_NO_MODELS`. `documents/extract.py` checks the variable
   itself (`_models_allowed()`), so OCR and face detection stay unloaded in parallel test runs. If
   `models.present` learns to honour it, the local check can go.
3. `api/common.py`: `DOC_TYPE_LABELS` and `DOC_GROUP_OF` must stay in sync with
   `documents.parsers.REGISTRY`. `tests/test_documents_parsers.py::test_registry_matches_api_labels`
   checks this. They could be derived from the registry instead.

## Wiring (nothing to change, recorded for the integrator)

- `api/documents.py` declares `APIRouter(prefix="/api/documents")`. Mount it without another prefix.
  `main.mount_prefix` already detects this.
- The worker's `text` handler calls `extract_item(conn, item_id, version)`. Errors from the file are
  logged, and the function sets `text_status='error'` and returns normally, so the job does not retry.
  Database errors, MemoryError and a full disk are raised instead (also when recording 'error' fails),
  so the worker retries with backoff or defers, and `_final_failure` marks `text_status='error'`. A stale
  version or a trashed item is skipped.
- `refresh_fts(conn, item_id)` works for any item (images with tags or labels only, items without text)
  and nests inside an open transaction (savepoint). `remove_fts(conn, item_id)` drops the row.
- `extract.match_query(q)` builds a safe FTS5 MATCH expression (every word quoted, prefix match). The
  search area may reuse it.
- `extract.reparse(conn, item_id, forced_type=None)` re-parses stored text without OCR. The API uses it,
  and so does PATCH when `doc_type` changes. PATCH with a `doc_type` on an item that has no documents row
  creates one.
- The documents list shows NSFW items only with `vault=1` and an unlocked vault. It lists near-duplicate
  copies too (`dups="all"`), so one document is never hidden behind another photo of the same layout, and
  it ignores mis-click flags.
- `wallet_available` is true only when the wallet is configured, the group is `pass` and a barcode was
  decoded (rule in `api/common.document_out`). `build_pass` also works without a decoded barcode: it uses
  the booking id or PNR as a QR message.

## Behaviour decisions to review

- User edits: a field edited through PATCH carries `"source": "user"` in the fields JSON, and these
  fields survive re-extraction and reparse. Once a row has `source='user'`, its doc_type and its non-empty
  title, doc_date and person_name are kept. A field the user deleted (empty string) comes back on the next
  reparse if the parser finds it again. Deletions are not remembered.
- Category: when the text stage finds a typed document in an image, it does not change `items.category`
  (analysis owns categories). If such images should move to `documents`, the analysis area or the
  integrator should add that rule after `extract_item`.
- Detection threshold 0.5. With no OCR text, a BCBP barcode alone classifies an image as a boarding pass.
- ID photo: crops are made only for the `id` group, from the image or from page renders 1 and 2. When no
  face is found, a stale `docphoto` file is removed.
