# Integration notes: frontend-library

Files: `frontend/src/pages/{LibraryPage,MediaPage,PeoplePage,PersonPage,DocumentsPage,ReviewPage,VaultPage}.tsx`,
`frontend/src/components/{MediaGrid,Viewer}.tsx`, `frontend/src/components/{library,people,documents,viewer}/*`,
`frontend/src/lib/{hooks,format,preview,useItemList,useSearchList}.ts`.

## Contract changes worth making (worked around locally)

1. `GET /api/review/guesses` has no person filter. PersonPage's Guesses tab pages through all guesses
   (limit 200 per page, query key `['review','guesses','all']`) and keeps the ones for that person.
   Suggested: `person` query param on the endpoint and `api.review.guesses(cursor, limit, person?)`.
2. `PersonOut` has no `cover_face_id`. PersonPage marks the cover face by parsing `/files/face/<id>` out of
   `cover_url`. Suggested: add `cover_face_id: int | None` to `PersonOut` and `types.ts`.
3. `api.items.text(uid)` has no vault flag. TextPanel calls `get('/api/items/<uid>/text?vault=1')` directly for
   vault items. The backend's `_guard` only checks the unlock window, so this works either way; adding
   `vault` to `api.items.text` would keep the call in one place.

## Behaviour the other areas should know

- Vault: every /files URL rendered for a vault item carries `vault=1` (`lib/preview.ts` `withVault`,
  `vaultify`). `api/files.py` ignores the param today (it checks the unlock window), which is fine.
  Vault query keys always contain `vault: true` (`['items', {vault:true,...}]`, `['search', {vault:true,...}]`,
  `['item', uid, true]`, `['item-text', uid, true]`); VaultPage removes them on lock, on expiry, on any
  `vault_locked` error and on leaving the page. Nothing vault-related is requested while locked.
- Query keys used: `items`, `item`, `item-text`, `search`, `people`, `person`, `person-faces`, `person-items`,
  `faces-unassigned`, `suggestions`, `documents`, `document-types`, `review`, `stats`, `config`.
  `lib/hooks.ts` `useRefreshLibrary()` invalidates all of them plus `trash`.
- Toasts: uses `useToast()` from `components/Toast.tsx` (`success`, `error`). The Viewer keeps its own toast
  line inside the overlay as well.
- Auth: uses `useAuth()` (`state`, `refresh`, `vaultUnlocked`, `setState`) and `startPasskeyAssertion()`,
  `passkeysSupported()` from `lib/auth.ts`.
- `MediaPage` (`/m/:uid`) accepts `?vault=1` (vault item), `?t=<sec>` (video start) and `?doc=1`
  (document card first).
- Search snippets are rendered as text: `<b>`/`</b>` (and `<mark>`, `\x02`/`\x03`) become highlights and
  the basic HTML entities are decoded; no innerHTML anywhere.
- `lib/format.ts` keeps every earlier export; `formatMediaCount` accepts `{item_count}` as well as the hub shape.
- `useItemList(key, fetchPage, {enabled, pollMs, keyOf})` is generic: the shell's TrashPage can use it with
  `(c) => api.trash.list(c)`.

## Shared components (not mine to change)

- `components/admin/fields.tsx` `SelectionBar` draws its Clear and Select all buttons at `h-8` (32 px), under
  the 40 px tap-target rule. My bulk bars use it; raising those two buttons to `h-10` there fixes every page.
- `components/Select.tsx` menus sit at `z-[80]`, above the viewer (`z-40`), so the category and document type
  pickers inside the viewer work as they are.
