# Integration notes: search

Files: `backend/app/search/service.py`, `backend/app/api/search.py`, `backend/tests/test_search_{support,text,semantic,api}.py`.
`backend/app/search/indexer_hub_reference.py` is deleted (no indexer thread any more).

## Mounting

`app.api.search.router` has no prefix and declares `GET /search`. Mount it with
`app.include_router(search.router, prefix="/api")`. The router depends on `require_user`; the route
also uses `want_vault` and `get_db` from `app.api.deps`.

## Hub leftovers to remove (owned by core)

These hub copies still import the deleted `app.search.indexer` and must not survive integration:
- `app/main.py` (imports `app.search.indexer`, starts and stops it in the lifespan),
- `app/ingest.py` line ~298 (`from app.search.indexer import notify`),
- `app/api/public.py` and `app/api/admin.py` (hub routers, replaced by library/system; they also call the
  old `search_service.search(..., include_hidden=...)`, `indexed_count(..., include_hidden=False)` and
  `MEDIA_ITEM_COLUMNS`, none of which exist any more).
Nothing needs to wake search after an upload: the analysis stage writes `item_embeddings` and
`torso_embedding` and bumps `search_version`; the matrix picks the rows up on the next query.

## Contract details the other areas rely on

- `tests/conftest.py` calls `app.search.service.reset_matrix()`: the name is kept.
- Embeddings are read with `model = app.ml.siglip.MODEL_ID` (fallback `"ViT-B-16-SigLIP__webli"`)
  and only rows whose `version` equals `items.version` count. The analysis stage must store
  item_embeddings under that model id and the item's current version.
- Person crops: `faces.torso_embedding` plus `bodies.torso_embedding` of bodies with `face_id IS NULL`
  (bodies with a face are covered by the face's crop). Writers must bump `search_version` in the same
  tx (the row counts also trigger a refresh, but an in-place rewrite of an existing row needs the bump).
- `service.engine_available()` is False when `settings.search_enabled` is off; otherwise it calls
  `app.ml.siglip.available()`. The engine is used through `get_engine().embed_text(text)` and
  `.relevance(cos)` only.
- `service.indexed_count(conn, model=None, *, vault=None)` and `service.indexing_pending(conn)` are
  available for `/api/stats` or `/api/system` if wanted.
- `SearchResult` has one extra field beyond the contract: `semantic_used: bool`.

## Behaviour choices (documented in the service docstring)

- Visibility: `visible_sql(vault=..., misclicks="include", dups="collapse")`. Mis-click shots are
  searchable (the card carries `misclick`), duplicate copies collapse onto the group's best item.
- `mode=semantic` with the model unavailable answers 503 `search_unavailable` (the only engine asked
  for cannot answer). `auto` and `text` never 503 unless FTS itself fails and semantic did too.
- `indexing` is true while any live item has `analysis_status='pending'` or `text_status='pending'`
  (text extraction feeds FTS, so documents waiting for OCR count as indexing too).
- Snippets are HTML-escaped with only `<b>` and `</b>` added, safe to render as HTML.
- Contrast gate: `search_contrast x median` is capped at `service.CONTRAST_CAP` (0.25 relevance), so a
  one-item library or a subject that fills the library still gets its clear matches.
- Paging: page 1 (offset 0) computes the merged list; later pages reuse it from an in-process cache
  (keyed by query, mode and filters, 8 entries, 10 minutes) and re-check visibility of the page ids.
  Snippets are made only for the returned page. `reset_matrix()` also clears that cache.
- `to` is inclusive (`COALESCE(taken_at, created_at) < to + 1 day`), dates are compared on the UTC
  ISO strings (no local-timezone shift).

## Frontend

`SearchHit.snippet` may contain `<b>` markup (escaped otherwise). `SearchHit.source` is
`semantic | text | both`. `match_time_s` is set for video hits (seek target).

## After assembly

`tests/test_search_api.py::test_end_to_end_through_the_real_app` skips while `app.main`, the auth
area or the mount are missing. After integration it must run and pass, not skip: a skip there then
means a real mount or auth problem.

## Hindi and Marathi (Devanagari)

- Tokenizer: `items_fts` uses `unicode61 remove_diacritics 2 tokenchars '...'`, where the list holds
  every combining mark (Unicode Mn, Mc) from U+0900 to U+0DFF (Devanagari through Sinhala). Plain
  `unicode61` treats vowel signs and the virama as separators, so "मुंबईत" was indexed as "म" and "बईत".
  The literal lives in `schema.sql` and in `app.db.FTS_TOKENIZE` (`FTS_TOKENCHARS` holds the marks);
  `tests/test_search_devanagari.py` checks that the two agree. Danda (।) stays a separator.
- Migration: `db.init_db` compares the stored `items_fts` definition with `FTS_TOKENIZE`. When it differs,
  it drops the table, creates it from `schema.sql` and sets kv `fts_rebuild = 1`, all in one transaction.
  `ingest.recover_on_startup` then calls `documents.extract.refresh_fts` for every item (trashed ones
  too), 500 per transaction, logs the count and deletes the flag. An interrupted rebuild runs again on
  the next start. Search finds nothing between the two steps, which run back to back at startup.
- Query side: Indic combining marks stay inside a piece, U+200C and U+200D are removed, so "राहुल" is
  one term and "मुंबई*" a prefix query. A piece with a Devanagari nukta letter is searched in both
  spellings, precomposed (U+0929, U+0931, U+0934, U+0958..U+095F, which the OCR model emits) and base
  letter + U+093C: `("बड़ा" OR "बड़ा")`. The strict query then joins terms with an explicit `AND`,
  because FTS5 allows the implicit AND only between phrases. Queries without such letters are unchanged.
- Snippets highlight whole tokens, so a Hindi word is wrapped whole: `<b>कुमार</b>`.
- Semantic engine: SigLIP is trained on English text. In mode `auto` a query whose letters are all
  non-Latin (`service.text_only_query`) skips the semantic pass and answers with the text hits alone
  (`semantic_used` false). A mixed query ("राहुल passport") and a query of digits alone run both
  engines as before. `mode=semantic` still asks the engine. The colour check is Latin only.
- Not done here (other owners): `documents.extract.match_query` (used by `/api/documents?q=`) still
  splits words with `\w+`, which drops the marks, so a Hindi word there becomes several one-letter
  prefix terms. `refresh_fts` indexes text as it comes: removing U+200C/U+200D there too would make
  words typed with a joiner (Marathi eyelash ra) match. Devanagari digits and ASCII digits are
  different tokens: "१९८८" does not find "1988".
