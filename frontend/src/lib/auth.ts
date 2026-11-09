// Session state and passkey helpers (docs/ARCHITECTURE.md section 4).
//
// useAuth() reads GET /api/auth/state through react-query (key AUTH_KEY) and derives whether the NSFW
// vault is unlocked from `vault_until`, re-evaluated every 30 s so the UI locks itself when the window
// ends. The passkey helpers run the WebAuthn ceremonies with @simplewebauthn/browser against the
// options the server sends, passing the server's `challenge_id` back with the credential.
// usePasskeyPlace() says whether passkeys can work on this address: they belong to the server's own
// host (the RP ID), and on a custom domain the browser can use them only through WebAuthn related
// origins, which Firefox does not support. Where they cannot, the screens link to the server's own address.
import { useCallback, useEffect, useState } from 'react'
import { useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import {
  browserSupportsWebAuthn,
  startAuthentication,
  startRegistration,
  WebAuthnError,
  type PublicKeyCredentialCreationOptionsJSON,
  type PublicKeyCredentialRequestOptionsJSON,
} from '@simplewebauthn/browser'
import { ApiError, api } from './api'
import type { AuthState, PasskeyOut } from './types'

export const AUTH_KEY = ['auth', 'state'] as const
export const PASSKEYS_KEY = ['auth', 'passkeys'] as const
export const SESSIONS_KEY = ['auth', 'sessions'] as const
/** Public: where passkeys work (usePasskeyPlace). Kept across logouts like the auth state. */
export const PASSKEY_PLACE_KEY = ['auth', 'passkey-place'] as const

const VAULT_TICK_MS = 30_000

/** The state the app assumes after a logout or a 401: an account exists, nobody is signed in. */
export function loggedOutState(prev?: AuthState | null): AuthState {
  return {
    setup_required: false,
    authenticated: false,
    username: null,
    passkeys: prev?.passkeys ?? 0,
    vault_until: null,
    session_id: null,
    site_title: prev?.site_title ?? 'Personal Content Cloud',
  }
}

export function isVaultOpen(state: AuthState | null | undefined, now = Date.now()): boolean {
  if (!state?.authenticated || !state.vault_until) return false
  const t = Date.parse(state.vault_until)
  return Number.isFinite(t) && t > now
}

const PUBLIC_KEYS: readonly (readonly string[])[] = [AUTH_KEY, PASSKEY_PLACE_KEY]

/** Drop every cached query except the public auth ones (after logout or when the session ends). */
export function clearPrivateCache(qc: QueryClient): void {
  qc.removeQueries({
    predicate: (q) => !PUBLIC_KEYS.some((k) => q.queryKey[0] === k[0] && q.queryKey[1] === k[1]),
  })
}

export interface UseAuth {
  state: AuthState | undefined
  loading: boolean
  error: unknown
  refresh: () => Promise<AuthState | undefined>
  /** True while the session's vault window is open (re-checked every 30 s). */
  vaultUnlocked: boolean
  /** Sign out of this session; the cache is cleared and the state becomes logged out. */
  logout: () => Promise<void>
  /** Replace the cached state with what an auth endpoint returned. */
  setState: (s: AuthState) => void
}

export function useAuth(): UseAuth {
  const qc = useQueryClient()
  const q = useQuery({ queryKey: AUTH_KEY, queryFn: api.auth.state, staleTime: 30_000, retry: 1 })
  const [now, setNow] = useState(() => Date.now())
  const vaultUntil = q.data?.vault_until ?? null

  useEffect(() => {
    if (!vaultUntil) return
    setNow(Date.now())
    const t = window.setInterval(() => setNow(Date.now()), VAULT_TICK_MS)
    return () => window.clearInterval(t)
  }, [vaultUntil])

  const refetch = q.refetch
  const refresh = useCallback(async () => {
    const r = await refetch()
    return r.data
  }, [refetch])

  const setState = useCallback((s: AuthState) => qc.setQueryData(AUTH_KEY, s), [qc])

  const logout = useCallback(async () => {
    const prev = qc.getQueryData<AuthState>(AUTH_KEY)
    try {
      await api.auth.logout()
    } catch (e) {
      // a session that is already gone is a successful logout
      if (!(e instanceof ApiError && e.status === 401)) throw e
    }
    clearPrivateCache(qc)
    qc.setQueryData(AUTH_KEY, loggedOutState(prev))
  }, [qc])

  return {
    state: q.data,
    loading: q.isPending,
    error: q.error,
    refresh,
    vaultUnlocked: isVaultOpen(q.data, now),
    logout,
    setState,
  }
}

// ---------------------------------------------------------------------------
// passkeys

/** WebAuthn needs a browser that supports it and a secure context (https or localhost). */
export function passkeysSupported(): boolean {
  try {
    return typeof window !== 'undefined' && window.isSecureContext && browserSupportsWebAuthn()
  } catch {
    return false
  }
}

/** The server's own address, where passkeys always work. */
export interface HomeAddress {
  /** scheme://host[:port] */
  origin: string
  /** host[:port], for display */
  host: string
}

export type PasskeyPlace =
  /** No WebAuthn here at all (an old browser, or plain http). */
  | { kind: 'unsupported' }
  /** Still asking the server and the browser; keep the passkey button disabled meanwhile. */
  | { kind: 'checking' }
  | { kind: 'here' }
  /** This address cannot use the passkeys (a custom domain in a browser without related origins). */
  | { kind: 'elsewhere'; home: HomeAddress }

/** True when a page on `hostname` may use passkeys made for `rpId` directly: the RP ID itself or a subdomain of it. */
export function onRelyingParty(hostname: string, rpId: string): boolean {
  const h = hostname.toLowerCase().replace(/\.$/, '')
  const r = rpId.toLowerCase().replace(/\.$/, '')
  return h === r || h.endsWith(`.${r}`)
}

type ClientCapabilities = Record<string, boolean | undefined>
const CAPABILITIES_WAIT_MS = 3000

/**
 * Whether the browser reports WebAuthn related origins in PublicKeyCredential.getClientCapabilities(). A browser
 * without the method, one that throws, or one that does not answer within a few seconds does not.
 */
export async function relatedOriginsSupported(): Promise<boolean> {
  try {
    const pkc = window.PublicKeyCredential as unknown as
      | { getClientCapabilities?: () => Promise<ClientCapabilities> }
      | undefined
    if (typeof pkc?.getClientCapabilities !== 'function') return false
    const caps = await Promise.race([
      pkc.getClientCapabilities(),
      new Promise<null>((resolve) => window.setTimeout(() => resolve(null), CAPABILITIES_WAIT_MS)),
    ])
    return caps?.relatedOrigins === true
  } catch {
    return false
  }
}

function homeAddress(url: string): HomeAddress | null {
  try {
    const u = new URL(url)
    return { origin: u.origin, host: u.host }
  } catch {
    return null
  }
}

/** The server's own address when passkeys cannot work on this one, else null. */
async function findPasskeyHome(): Promise<HomeAddress | null> {
  const rp = await api.auth.rp()
  const here = window.location
  if (onRelyingParty(here.hostname, rp.rp_id)) return null
  const home = homeAddress(rp.home_url)
  // nowhere better to send the visitor: leave the passkey button as it was
  if (!home || home.host === here.host) return null
  return (await relatedOriginsSupported()) ? null : home
}

/**
 * Where passkeys work, for the screens that offer one (sign in, setup, vault unlock, adding a passkey). When the
 * server cannot be asked (an older server), the answer is 'here' and the screens behave as before.
 */
export function usePasskeyPlace(): PasskeyPlace {
  const supported = passkeysSupported()
  const q = useQuery({
    queryKey: PASSKEY_PLACE_KEY,
    queryFn: findPasskeyHome,
    enabled: supported,
    staleTime: Infinity,
    gcTime: Infinity,
    retry: 1,
  })
  if (!supported) return { kind: 'unsupported' }
  if (q.isPending) return { kind: 'checking' }
  return q.data ? { kind: 'elsewhere', home: q.data } : { kind: 'here' }
}

/**
 * The server may wrap the options (`{publicKey: {...}}`, as navigator.credentials expects) or send
 * them bare (py_webauthn's options_to_json); simplewebauthn wants the bare JSON.
 */
function unwrap<T>(options: Record<string, unknown>): T {
  const inner = options.publicKey
  return (inner && typeof inner === 'object' ? inner : options) as T
}

/** Register a new passkey for the signed-in account under `name`. */
export async function registerPasskey(name: string): Promise<PasskeyOut> {
  const { challenge_id, options } = await api.auth.passkeyRegisterOptions()
  const credential = await startRegistration({ optionsJSON: unwrap<PublicKeyCredentialCreationOptionsJSON>(options) })
  return api.auth.passkeyRegisterVerify(challenge_id, name.trim(), credential)
}

/** Sign in with a discoverable passkey. Returns the new auth state. */
export async function loginWithPasskey(): Promise<AuthState> {
  const { challenge_id, options } = await api.auth.passkeyLoginOptions()
  const credential = await startAuthentication({ optionsJSON: unwrap<PublicKeyCredentialRequestOptionsJSON>(options) })
  return api.auth.passkeyLoginVerify(challenge_id, credential)
}

/** Ask the authenticator to confirm the signed-in user again (vault unlock). */
export async function startPasskeyAssertion(): Promise<{ challenge_id: string; credential: unknown }> {
  const { challenge_id, options } = await api.auth.passkeyAssertOptions()
  const credential = await startAuthentication({ optionsJSON: unwrap<PublicKeyCredentialRequestOptionsJSON>(options) })
  return { challenge_id, credential }
}

/** True for the 403 `vault_locked` answer: the request needs the vault open (password or passkey again). */
export function isVaultLocked(e: unknown): boolean {
  return e instanceof ApiError && e.code === 'vault_locked'
}

/** Plain-language message for a failed passkey ceremony or auth request. */
export function authErrorMessage(e: unknown, fallback = 'Something went wrong. Try again.'): string {
  if (e instanceof ApiError) {
    if (e.status === 429) return 'Too many attempts. Wait a few minutes, then try again.'
    if (e.status === 0) return 'The server could not be reached. Check the connection and try again.'
    if (e.code === 'login_required') return 'Your session has ended. Sign in again.'
    if (e.code === 'csrf') return 'The request was refused. Reload the page and try again.'
    if (e.code === 'vault_locked') return 'Unlock the vault first: enter your password or use a passkey.'
    if (e.status === 401) return 'That did not work. Check the details and try again.'
    return e.message || fallback
  }
  const name = e instanceof Error ? e.name : ''
  const code = e instanceof WebAuthnError ? e.code : ''
  if (name === 'NotAllowedError' || code === 'ERROR_CEREMONY_ABORTED')
    return 'The passkey request was cancelled or timed out.'
  if (name === 'InvalidStateError' || code === 'ERROR_AUTHENTICATOR_PREVIOUSLY_REGISTERED')
    return 'This device already holds a passkey for this account.'
  if (name === 'SecurityError' || code === 'ERROR_INVALID_DOMAIN' || code === 'ERROR_INVALID_RP_ID')
    return 'Passkeys do not work on this address. Open the site through its https address.'
  if (name === 'NotSupportedError') return 'This browser or device cannot use passkeys.'
  if (name === 'AbortError') return 'The passkey request was cancelled.'
  return e instanceof Error && e.message ? e.message : fallback
}
