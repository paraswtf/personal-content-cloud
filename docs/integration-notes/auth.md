# Auth area: integration notes

1. Mounting. `app.api.auth.router` is a plain `APIRouter()` without a prefix (same convention as the
   hub's routers). main.py must mount it as `app.include_router(auth.router, prefix="/api/auth")`.
   Do not add a second prefix.
2. The hub main.py still in the tree imports `from app import auth` and calls `auth.set_access_cookie` /
   `auth.ACCESS_COOKIE` / `auth.access_token`. Those are gone (the hub reference file
   `app/auth/hub_reference.py` was deleted as planned). The new main.py should import
   `from app.api import auth as auth_api` and has no visitor access key.
3. `app/auth/deps.py` listed in ARCHITECTURE section 2 does not exist: the dependencies live in the shared
   `app/api/deps.py` (require_user, optional_user, require_vault, check_csrf). Suggest updating section 2.
4. Error details used by the auth routes (frontend can key on them):
   - 401 `invalid_credentials`: wrong password, wrong passkey, wrong vault password, wrong current password;
   - 429 `too_many_attempts`: rate limit (10 failures / 600 s per IP, shared by password login, passkey
     login, vault unlock and password change);
   - 409 `already_set_up`: setup after the account exists;
   - 409 `setup_required`: login or passkey login before setup;
   - 400 `challenge_invalid` / `registration_failed`, 409 `credential_exists`: passkey registration;
   - 400 `password_or_passkey_required`: vault unlock with neither factor.
   Suggest adding these to section 13's code list.
5. Behaviour choices worth knowing:
   - `POST /password` revokes every other session (this one stays).
   - `POST /logout-all` revokes every session including the current one and clears the cookie.
   - `DELETE /sessions/{id}` on the current session also clears the cookie.
   - `AuthState.vault_until` is only reported while the window is open; `passkeys` and `username` are only
     reported to a logged-in session.
   - WebAuthn user handle is the constant `b"pcc-account-1"`; `RP_ID`/`ORIGIN` env override the derived
     values, `ORIGIN` may list several origins separated by commas.
   - Expired and used challenges are deleted whenever a new challenge is stored;
     `sessions.purge_expired` also clears expired ones.
6. Tests mount the auth router on their own FastAPI app (tests/test_auth_api.py `auth_app` fixture), so they
   do not depend on main.py. Once main.py is integrated the shared `client` / `vault_client` fixtures in
   conftest.py should work unchanged (setup creates the session and sets the cookie; vault unlock with
   `{password}` returns 200).
7. The login rate limiter is process-global and every TestClient uses the IP `testclient`. Suggest calling
   `app.auth.passwords.reset_all()` in the conftest `data_dir` fixture so failed logins in one test file
   cannot answer 429 to a later `vault_client`.
