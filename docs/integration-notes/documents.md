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

## Hindi and Marathi (Devanagari)

What the documents area does with Devanagari text (tests: `backend/tests/test_documents_hindi.py`;
fixtures are drawn with `tests/helpers.devanagari_text_image`, which needs
`models/fonts/NotoSansDevanagari-Regular.ttf` and skips without it).

- `fields.normalize_text` folds Devanagari before anything else (`dev_fold`): digits U+0966 to U+096F
  become ASCII digits, the danda, double danda and abbreviation sign become '.', U+200C and U+200D are
  dropped, and the precomposed nukta letters U+0958 to U+095F are split into letter + nukta (what NFC
  does for them), so both spellings of 'फ़रवरी' compare equal. Gaps and dashes behave as before; the
  function stays idempotent.
- Dates and months: Hindi and Marathi month names (`fields.LOCAL_MONTHS`, with the usual spelling
  variants) work in `find_dates`, `parse_date` and `parse_month`; a month word must stand alone, so the
  Marathi 'मे' (May) does not match inside Hindi 'में'. `month_label` still writes English month names.
- Labels: every parser appends Hindi and Marathi labels after its English ones (shared lists in
  fields.py: `NAME_LABELS_LOCAL`, `FATHER_LABELS_LOCAL`, `DOB_LABELS_LOCAL`, `ISSUE_LABELS_LOCAL`,
  `DATE_LABELS_LOCAL`, `AMOUNT_LABELS_LOCAL`, `ADDRESS_LABELS_LOCAL`, ...), so an English label still
  wins on a bilingual document. `after_label` guards a label by its own script (a Devanagari label does
  not match inside a longer Devanagari word: 'नाम' not in 'उपनाम'), skips the other half of a bilingual
  label ('जन्म तिथि/DOB:', 'Name / नाम :'; at the end of a line the other half must end in a label word,
  so 'Name / राहुल शर्मा' gives the name), and with anchored=True accepts 'नाम / Name' as a line start.
  The generic date words दिनांक, तारीख, तिथि ignore a hit right after 'जन्म' (birth). `STOP_LABELS_LOCAL`
  ends a value at a later Devanagari label.
- Names: `plausible_name`, `nice_name` and `_base.is_name_line` count combining marks as letters. A
  value in both scripts keeps its Latin spelling; a Devanagari-only name is kept as written (no
  casing). `name_after` prefers a Latin spelling found after any of its labels; `local_name_after`
  reads the Devanagari spelling. New field `name_local` ("Name (local script)") on aadhaar, pan,
  voter_id, driving_licence, vehicle_rc and marksheet, set only when it differs from the main name. It
  is not a key field, so English confidences do not change.
- Gender (`लिंग`; पुरुष, पुरूष, महिला, स्त्री, अन्य) and relations (पुत्र, पुत्री, पत्नी, पति, मुलगा, मुलगी) give
  the English values ('Male', 'Son of'); the relative's name stays in Devanagari. Marksheet results
  (उत्तीर्ण, प्रथम श्रेणी, विशेष प्राविण्य ...) and receipt payment modes (नकद, रोख, धनादेश ...) are given in
  English too. Amounts accept रु., रू., रुपये.
- Detection: `detect.LOCAL_RULES` adds Hindi and Marathi phrases with weights like the English ones and
  matching negatives (a marksheet or a rent agreement is not an invoice or a receipt; 'पगार पावती' is a
  salary slip, not a receipt). A phrase matches whole words of its script, which also stops the old
  'आधार' keyword from firing on 'आधारित' ("based on"). There is no rent agreement type (the registry
  must stay in sync with `api/common.py`): a rent agreement stays `other`, and the other parser sets a
  new field `kind` = "Rent agreement" for an English, Hindi or Marathi rent agreement heading. The
  other parser takes a Devanagari line as the title only when no mostly-Latin line exists (a Hindi line
  that only carries an ID such as a PAN does not count as Latin); `largest_text` and
  `invoice.seller_line` also try Latin first. Aadhaar and PAN have the rent phrases as negatives, since
  a rent agreement prints both parties' Aadhaar and PAN numbers.
- Language: `extract.analyze_text` adds a field `language` (label "Language", confidence 0.9) when the
  text has Devanagari letters: "Hindi", "Marathi", "Hindi or Marathi" (the stop words do not decide),
  each followed by ", English" when the text has two or more Latin words of three letters or more.
  `fields.languages_of(text)` decides with stop words and common document words (`HINDI_WORDS`,
  `MARATHI_WORDS`) and the Marathi genitive endings -चे, -ची, -च्या; it needs two Devanagari words.
  English-only documents get no language field.
- Search box: `extract.match_query` splits with `fields.word_tokens` (a token is a run of word
  characters, combining marks or letters U+0900 to U+0DFF; dandas excluded; U+200C and U+200D dropped),
  so 'मुंबईत' stays one quoted word. The query first gets the index fold (`extract.fold_for_index`:
  U+200C and U+200D dropped, Devanagari digits as ASCII), so '१९८८' and '1988' both find a receipt that
  says either. The search area applies the same rule in `search/service.py` (its own change in this round).

### For other areas

- Search: `item_text` keeps the raw page text; every `items_fts` column is stored through
  `extract.fold_for_index` (joiners dropped, Devanagari digits as ASCII, nothing else; nukta letters and
  dandas stay as written). A query needs exactly that fold, not `fields.dev_fold`, which also decomposes
  nukta letters. `search/service.py` has its own copy (`_JOINERS`, `_QUERY_DIGITS`); importing
  `fold_for_index` would keep the two from drifting.
- Search snippets: `snippet(items_fts, ...)` returns folded text ('15' for '१५', a Marathi eyelash-ra
  word without its ZWJ). `extract.snippet_originals(conn, ids)` gives the original text and doc columns,
  and `extract.restore_snippet(raw, original, open, close)` maps a raw snippet (markers still in) back
  onto the original; it returns the raw snippet when the fragment is not found. `search/service._snippets`
  has to call them before `_render_snippet`.
- Frontend: the new fields (`name_local`, `language`, `kind`) arrive in the fields JSON with their
  labels like any other field; nothing in `lib/types.ts` needs to change unless the UI lists field keys.
- Behaviour change for English text: the voter ID issue date no longer reads the 'Date of Birth' line
  through the generic 'Date' label (it did before on cards with an English 'Date of Birth' line and no
  issue date).
