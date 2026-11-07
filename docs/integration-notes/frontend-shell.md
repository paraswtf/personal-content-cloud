# Integration notes: frontend-shell

## For the frontend-library pages

- The shell renders `<main id="main">` around the routed page. Pages must not render their own `<main>`
  (the placeholders still do) and bring their own width and gutters: `mx-auto w-full max-w-7xl px-4 py-6`.
  On phones the shell already pads the bottom for the tab bar.
- Toasts moved to `components/Toast.tsx` (`useToast`, `ToastProvider`). `components/admin/Toast.tsx` is
  deleted. `ToastProvider` wraps the whole router in `App.tsx`, so `useToast` works in every page and portal.
- Auth: `useAuth()` from `lib/auth.ts` gives `{state, loading, refresh, vaultUnlocked, logout, setState}`.
  Vault unlock: `api.auth.vaultUnlockPassword(pw)` or `const {challenge_id, credential} = await
  startPasskeyAssertion(); api.auth.vaultUnlockPasskey(challenge_id, credential)`, then `setState(result)`.
  `authErrorMessage(err)` turns WebAuthn and API errors into plain copy. `passkeysSupported()` tells whether
  to offer the passkey button.
- Query keys used by the shell: `['auth','state']`, `['auth','passkeys']`, `['auth','sessions']`,
  `['system']`, `['config']` (api.config), `['stats']` (api.stats, read by the nav for the Review badge),
  `['trash']`, `['uploads-interrupted']`. Upload settles invalidate `['items']`, `['stats']`, `['documents']`.
  Trash restore, purge and empty use `useRefreshLibrary()` (people, person, review and search too).
- `VaultBadge` "lock now" and the countdown reaching zero remove cached queries whose key starts with
  `'vault'`, contains `{"vault": true}`, or is `['item', uid, true]` / `['item-text', uid, true]`
  (same rule as VaultPage); keep vault data under such keys.
- A page title helper for shell pages is `components/shell/title.ts` (`usePageTitle`).

## Tokens (index.css)

New tokens: `bg`, `surface`, `surface-2`, `ink`, `ink-soft`, `ink-faint`, `line`, `accent`, `accent-strong`,
`accent-soft`, `ok`, `warn`, `danger`, `--radius-card`, `--shadow-card`, `--tabbar-h`. The hub names `brand`,
`brand-dark`, `brand-tint`, `cream`, `cream-soft`, `paper` are aliases of `accent`, `accent-strong`,
`accent-soft`, `bg`, `surface-2`, `surface`, so the shared components keep working in both themes.

Shared-file changes the integrator may want (I did not edit them, they are shared contracts):
- `components/ui.tsx` `Modal`: the backdrop `bg-ink/50` turns light grey in dark mode because `ink` is light
  there. Suggest `bg-black/50`.
- `components/ui.tsx` `buttonClasses` `danger` variant uses the accent (blue) colours; suggest
  `bg-paper text-danger border border-danger/30 hover:bg-danger/10` so destructive buttons read as such.

## Backend expectations

- `GET /api/auth/state` should report `passkeys` (count) and `site_title` while signed out too: the login
  page puts the passkey button first when passkeys exist.
- `POST /api/auth/login` wrong credentials: 401 (or 400); rate limit: 429. Both are shown with their own copy.
- Passkey options may be bare py_webauthn JSON or wrapped as `{publicKey: {...}}`; `lib/auth.ts` unwraps both.
  The credential is sent back exactly as `@simplewebauthn/browser` returns it (RegistrationResponseJSON /
  AuthenticationResponseJSON), which `webauthn.helpers.parse_*_credential_json` accepts.
- `POST /api/auth/logout-all` may end the current session too; the Account tab handles both answers.
- `main.py` must serve `/sw.js`, `/manifest.webmanifest` and `/icons/*` from `dist/` as files (not the SPA
  fallback), with `application/javascript` and `application/manifest+json`. `sw.js` should be sent with
  `Cache-Control: no-cache` so updates reach browsers.
- `POST /share-target` normally never reaches the server (the service worker answers it). Before the worker is
  active (first share right after install) it would; suggest a backend route that answers 303 to
  `/upload?shared=1` so the user lands on the upload page instead of a 404/405.
- If `main.py` adds a Content-Security-Policy: the app needs `img-src 'self' blob: data:` (local previews of
  queued files), `media-src 'self' blob:`, `worker-src 'self'`, `connect-src 'self'`, and no inline scripts are
  used. If it adds a Permissions-Policy, allow `camera=(self), microphone=(self)` for the in-app camera.
- `UploadOut` has no item kind; the queue derives the icon from the file name and MIME type.
- The Wallet tab lists `SystemOut.wallet.missing` entries verbatim and explains env vars
  `WALLET_PASS_TYPE_ID`, `WALLET_TEAM_ID`, `WALLET_ORG_NAME` and files `cert.pem`, `key.pem`, `wwdr.pem` in
  `SystemOut.wallet.dir`. Missing entries named exactly like those get an explanation next to them.

## Package

- `frontend/package.json`: added `@simplewebauthn/browser` `^13.3.0` (installed; `package-lock.json` updated).
