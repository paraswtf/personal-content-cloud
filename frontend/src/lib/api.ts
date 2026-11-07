// Typed client for the backend API (backend/app/api/*). All calls are same-origin and carry the
// CSRF header. A 401 anywhere means the session is gone: `onUnauthorized` (set by the app shell)
// sends the user to /login.
import type {
  AssignBodiesIn,
  AssignFacesIn,
  AssignOut,
  AuthState,
  BulkIn,
  BulkOut,
  Category,
  ChallengeOut,
  ChunkOut,
  ConfigOut,
  DocumentOut,
  DocumentPage,
  DocumentTypeOut,
  DocumentUpdate,
  DupGroupPage,
  FacePage,
  GuessIdsIn,
  GuessPage,
  ItemCard,
  ItemDetail,
  ItemKind,
  ItemPage,
  ItemTextOut,
  ItemUpdate,
  MergeIn,
  OkOut,
  PageOut,
  PasskeyOut,
  PersonCreate,
  PersonOut,
  PersonPage,
  PersonRef,
  PersonUpdate,
  SearchMode,
  SearchPage,
  SessionOut,
  Stats,
  SuggestionList,
  SystemOut,
  TagIn,
  UploadCreate,
  UploadList,
  UploadOut,
} from './types'

export const CSRF_HEADER = 'X-Requested-With'
export const CSRF_VALUE = 'pcc'

/** Called on every 401 (except on the auth routes themselves). The app shell sets it. */
export let onUnauthorized: (() => void) | null = null
export function setUnauthorizedHandler(fn: (() => void) | null): void {
  onUnauthorized = fn
}

export class ApiError extends Error {
  status: number
  detail: unknown
  body: unknown
  constructor(status: number, message: string, detail: unknown, body: unknown) {
    super(message)
    this.status = status
    this.detail = detail
    this.body = body
  }
  /** The machine code the backend sent in `detail`, when it is a string. */
  get code(): string | null {
    return typeof this.detail === 'string' ? this.detail : null
  }
}

type Query = Record<string, string | number | boolean | null | undefined | Array<string | number>>

/** Build "?a=1&b=2&person=1&person=2"; skips null/undefined/'' values; arrays repeat the key. */
export function qs(params: Query = {}): string {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === '') continue
    if (Array.isArray(v)) v.forEach((x) => sp.append(k, String(x)))
    else if (typeof v === 'boolean') sp.append(k, v ? '1' : '0')
    else sp.append(k, String(v))
  }
  const s = sp.toString()
  return s ? `?${s}` : ''
}

function messageFrom(status: number, body: unknown): { message: string; detail: unknown } {
  const detail = body && typeof body === 'object' && 'detail' in body ? (body as { detail: unknown }).detail : body
  let message = `Request failed (${status})`
  if (typeof detail === 'string') message = detail
  else if (Array.isArray(detail) && detail[0] && typeof detail[0] === 'object' && 'msg' in detail[0])
    message = String((detail[0] as { msg: unknown }).msg)
  return { message, detail }
}

const AUTH_PREFIX = '/api/auth/'

async function request<T>(method: string, url: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const headers: Record<string, string> = { [CSRF_HEADER]: CSRF_VALUE, Accept: 'application/json' }
  let payload: BodyInit | undefined
  if (body instanceof FormData) {
    payload = body
  } else if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }
  const res = await fetch(url, { method, headers, body: payload, credentials: 'same-origin', signal })
  const text = await res.text()
  let data: unknown = undefined
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = text
    }
  }
  if (!res.ok) {
    const { message, detail } = messageFrom(res.status, data)
    if (res.status === 401 && !url.startsWith(AUTH_PREFIX)) onUnauthorized?.()
    throw new ApiError(res.status, message, detail, data)
  }
  return data as T
}

export const get = <T>(url: string, signal?: AbortSignal) => request<T>('GET', url, undefined, signal)
export const post = <T>(url: string, body?: unknown) => request<T>('POST', url, body ?? {})
export const patch = <T>(url: string, body: unknown) => request<T>('PATCH', url, body)
export const del = <T>(url: string) => request<T>('DELETE', url)

const enc = encodeURIComponent

/** Seconds of a video frame as sent to the server: floored to the millisecond. */
export function frameSeconds(t: number): string {
  const ms = Number.isFinite(t) && t > 0 ? Math.floor(t * 1000) : 0
  return (ms / 1000).toFixed(3)
}

/** URL of one video frame as a JPEG (`size` preview is at most 1280 px). */
export function frameUrl(
  uid: string,
  version: number,
  t: number,
  size: 'preview' | 'full' = 'full',
  download = false,
  nsfw = false,
): string {
  // Vault items carry an 'n' in their cache key (see the server's url_version), so a flagged video
  // never reuses a frame the browser cached before it was flagged.
  const v = nsfw ? `${version}n` : String(version)
  return `/files/frame/${enc(uid)}${qs({ t: frameSeconds(t), size, v, download: download ? 1 : undefined })}`
}

export interface ItemsQuery {
  category?: Category | ''
  kind?: ItemKind | ''
  person?: number[]
  from?: string
  to?: string
  favorite?: boolean
  misclicks?: 'exclude' | 'include' | 'only'
  dups?: 'collapse' | 'all'
  vault?: boolean
  sort?: 'taken' | 'added'
  cursor?: string | null
  limit?: number
}

export interface SearchQuery {
  q: string
  mode?: SearchMode
  category?: Category | ''
  kind?: ItemKind | ''
  person?: number[]
  from?: string
  to?: string
  vault?: boolean
  limit?: number
  offset?: number
}

export interface ChunkProgress {
  onProgress?: (sentInThisChunk: number) => void
  signal?: AbortSignal
}

/** Thrown by uploadChunk when the server's offset differs (resync with err.receivedBytes). */
export class OffsetMismatchError extends Error {
  receivedBytes: number
  constructor(receivedBytes: number) {
    super('offset_mismatch')
    this.receivedBytes = receivedBytes
  }
}

/** A chunk request that sees no progress for this long is aborted as stalled (then retried). */
const CHUNK_STALL_MS = 60_000

/** PUT one chunk with XHR so upload progress events are available. */
export function uploadChunk(id: string, offset: number, data: Blob, opts: ChunkProgress = {}): Promise<ChunkOut> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', `/api/uploads/${enc(id)}/chunk?offset=${offset}`)
    xhr.setRequestHeader('Content-Type', 'application/octet-stream')
    xhr.setRequestHeader(CSRF_HEADER, CSRF_VALUE)
    xhr.responseType = 'text'
    let last = Date.now()
    let stalled = false
    let tick = Date.now()
    const watchdog = setInterval(() => {
      const now = Date.now()
      if (now - tick > 15_000) last = now
      tick = now
      if (now - last > CHUNK_STALL_MS) {
        stalled = true
        xhr.abort()
      }
    }, 5000)
    xhr.onloadend = () => clearInterval(watchdog)
    xhr.upload.onprogress = (e) => {
      last = Date.now()
      opts.onProgress?.(e.loaded)
    }
    xhr.onprogress = () => {
      last = Date.now()
    }
    xhr.onload = () => {
      let body: unknown = undefined
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : undefined
      } catch {
        body = xhr.responseText
      }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body as ChunkOut)
      if (xhr.status === 401) onUnauthorized?.()
      if (xhr.status === 409 && body && typeof body === 'object' && 'received_bytes' in body)
        return reject(new OffsetMismatchError(Number((body as { received_bytes: number }).received_bytes)))
      const { message, detail } = messageFrom(xhr.status, body)
      reject(new ApiError(xhr.status, message, detail, body))
    }
    xhr.onerror = () => reject(new ApiError(0, 'Network error', null, null))
    xhr.ontimeout = () => reject(new ApiError(0, 'Upload timed out', null, null))
    xhr.onabort = () =>
      reject(stalled ? new ApiError(0, 'Upload stalled', null, null) : new DOMException('Aborted', 'AbortError'))
    if (opts.signal) {
      if (opts.signal.aborted) {
        clearInterval(watchdog)
        return reject(new DOMException('Aborted', 'AbortError'))
      }
      opts.signal.addEventListener('abort', () => xhr.abort(), { once: true })
    }
    xhr.send(data)
  })
}

export const api = {
  config: () => get<ConfigOut>('/api/config'),
  stats: () => get<Stats>('/api/stats'),
  /** GET /api/system; `api.system.retryFailed()` queues failed uploads again (POST /api/system/retry-failed). */
  system: Object.assign(() => get<SystemOut>('/api/system'), {
    retryFailed: () => post<{ requeued: number }>('/api/system/retry-failed'),
  }),
  downloadModels: () => post<OkOut>('/api/system/models/download'),
  housekeeping: () => post<OkOut>('/api/system/housekeeping'),

  auth: {
    state: () => get<AuthState>('/api/auth/state'),
    setup: (username: string, password: string) => post<AuthState>('/api/auth/setup', { username, password }),
    login: (username: string, password: string) => post<AuthState>('/api/auth/login', { username, password }),
    logout: () => post<AuthState>('/api/auth/logout'),
    logoutAll: () => post<AuthState>('/api/auth/logout-all'),
    sessions: () => get<SessionOut[]>('/api/auth/sessions'),
    revokeSession: (id: number) => del<OkOut>(`/api/auth/sessions/${id}`),
    changePassword: (current_password: string, new_password: string) =>
      post<OkOut>('/api/auth/password', { current_password, new_password }),
    passkeys: () => get<PasskeyOut[]>('/api/auth/passkeys'),
    passkeyRegisterOptions: () => post<ChallengeOut>('/api/auth/passkeys/register/options'),
    passkeyRegisterVerify: (challenge_id: string, name: string, credential: unknown) =>
      post<PasskeyOut>('/api/auth/passkeys/register/verify', { challenge_id, name, credential }),
    renamePasskey: (id: number, name: string) => patch<PasskeyOut>(`/api/auth/passkeys/${id}`, { name }),
    deletePasskey: (id: number) => del<OkOut>(`/api/auth/passkeys/${id}`),
    passkeyLoginOptions: () => post<ChallengeOut>('/api/auth/passkeys/login/options'),
    passkeyLoginVerify: (challenge_id: string, credential: unknown) =>
      post<AuthState>('/api/auth/passkeys/login/verify', { challenge_id, credential }),
    passkeyAssertOptions: () => post<ChallengeOut>('/api/auth/passkeys/assert/options'),
    vaultUnlockPassword: (password: string) => post<AuthState>('/api/auth/vault/unlock', { password }),
    vaultUnlockPasskey: (challenge_id: string, credential: unknown) =>
      post<AuthState>('/api/auth/vault/unlock', { challenge_id, credential }),
    vaultLock: () => post<AuthState>('/api/auth/vault/lock'),
  },

  items: {
    list: (q: ItemsQuery = {}, signal?: AbortSignal) => get<ItemPage>(`/api/items${qs({ ...q })}`, signal),
    detail: (uid: string, vault = false) => get<ItemDetail>(`/api/items/${enc(uid)}${qs({ vault })}`),
    update: (uid: string, body: ItemUpdate) => patch<ItemDetail>(`/api/items/${enc(uid)}`, body),
    remove: (uid: string) => del<OkOut>(`/api/items/${enc(uid)}`),
    restore: (uid: string) => post<ItemDetail>(`/api/items/${enc(uid)}/restore`),
    bulk: (body: BulkIn) => post<BulkOut>('/api/items/bulk', body),
    reprocess: (uid: string, stages?: Array<'analyze' | 'text' | 'preview'>) =>
      post<OkOut>(`/api/items/${enc(uid)}/reprocess`, stages ? { stages } : {}),
    pages: (uid: string) => get<PageOut[]>(`/api/items/${enc(uid)}/pages`),
    text: (uid: string, vault = false) => get<ItemTextOut>(`/api/items/${enc(uid)}/text${qs({ vault })}`),
    addTag: (uid: string, body: TagIn) => post<PersonRef[]>(`/api/items/${enc(uid)}/tags`, body),
    removeTag: (uid: string, personId: number) => del<PersonRef[]>(`/api/items/${enc(uid)}/tags/${personId}`),
  },

  trash: {
    list: (cursor?: string | null, limit = 60) => get<ItemPage>(`/api/trash${qs({ cursor, limit })}`),
    empty: () => post<BulkOut>('/api/trash/empty'),
    purge: (uid: string) => del<OkOut>(`/api/trash/${enc(uid)}`),
  },

  review: {
    misclicks: (cursor?: string | null, limit = 60) => get<ItemPage>(`/api/review/misclicks${qs({ cursor, limit })}`),
    guesses: (cursor?: string | null, limit = 60, person?: number) =>
      get<GuessPage>(`/api/review/guesses${qs({ cursor, limit, person })}`),
    duplicates: (cursor?: string | null, limit = 30) => get<DupGroupPage>(`/api/duplicates${qs({ cursor, limit })}`),
    setBest: (uid: string) => post<OkOut>(`/api/duplicates/${enc(uid)}/best`),
  },

  people: {
    list: (q: { q?: string; filter?: 'all' | 'named' | 'unnamed'; sort?: 'items' | 'name' | 'recent'; cursor?: string | null; limit?: number } = {}) =>
      get<PersonPage>(`/api/people${qs(q)}`),
    create: (body: PersonCreate) => post<PersonOut>('/api/people', body),
    get: (id: number) => get<PersonOut>(`/api/people/${id}`),
    update: (id: number, body: PersonUpdate) => patch<PersonOut>(`/api/people/${id}`, body),
    remove: (id: number) => del<OkOut>(`/api/people/${id}`),
    merge: (body: MergeIn) => post<PersonOut>('/api/people/merge', body),
    faces: (id: number, cursor?: string | null, limit = 200) => get<FacePage>(`/api/people/${id}/faces${qs({ cursor, limit })}`),
    items: (id: number, cursor?: string | null, limit = 60, vault = false) =>
      get<ItemPage>(`/api/people/${id}/items${qs({ cursor, limit, vault })}`),
    unassignedFaces: (cursor?: string | null, limit = 200) => get<FacePage>(`/api/faces/unassigned${qs({ cursor, limit })}`),
    assignFaces: (body: AssignFacesIn) => post<AssignOut>('/api/faces/assign', body),
    deleteFace: (id: number) => del<OkOut>(`/api/faces/${id}`),
    assignBodies: (body: AssignBodiesIn) => post<AssignOut>('/api/bodies/assign', body),
    deleteBody: (id: number) => del<OkOut>(`/api/bodies/${id}`),
    confirmGuesses: (body: GuessIdsIn) => post<BulkOut>('/api/guesses/confirm', body),
    rejectGuesses: (body: GuessIdsIn) => post<BulkOut>('/api/guesses/reject', body),
    suggestions: (limit = 30) => get<SuggestionList>(`/api/suggestions${qs({ limit })}`),
    dismissSuggestion: (a: number, b: number) => post<OkOut>('/api/suggestions/dismiss', { a, b }),
    recluster: () => post<OkOut>('/api/recluster'),
  },

  documents: {
    list: (q: { type?: string; group?: string; q?: string; cursor?: string | null; limit?: number } = {}) =>
      get<DocumentPage>(`/api/documents${qs(q)}`),
    types: () => get<DocumentTypeOut[]>('/api/documents/types'),
    get: (uid: string) => get<DocumentOut>(`/api/documents/${enc(uid)}`),
    update: (uid: string, body: DocumentUpdate) => patch<DocumentOut>(`/api/documents/${enc(uid)}`, body),
    reparse: (uid: string) => post<OkOut>(`/api/documents/${enc(uid)}/reparse`),
    walletUrl: (uid: string) => `/api/documents/${enc(uid)}/wallet.pkpass`,
  },

  search: (q: SearchQuery, signal?: AbortSignal) => get<SearchPage>(`/api/search${qs({ ...q })}`, signal),

  uploads: {
    create: (body: UploadCreate) => post<UploadOut>('/api/uploads', body),
    chunk: uploadChunk,
    complete: (id: string) => post<UploadOut>(`/api/uploads/${enc(id)}/complete`),
    get: (id: string) => get<UploadOut>(`/api/uploads/${enc(id)}`),
    many: (ids: string[]) => get<UploadList>(`/api/uploads${qs({ ids: ids.join(',') })}`),
    mine: (clientId: string) => get<UploadList>(`/api/uploads/mine${qs({ client_id: clientId })}`),
    cancel: (id: string) => del<UploadOut>(`/api/uploads/${enc(id)}`),
    simple: (file: File | Blob, filename: string, source: 'web' | 'share' | 'camera' = 'web') => {
      const fd = new FormData()
      fd.append('file', file, filename)
      fd.append('source', source)
      return post<UploadOut>('/api/uploads/simple', fd)
    },
  },
}

export type { ItemCard }
