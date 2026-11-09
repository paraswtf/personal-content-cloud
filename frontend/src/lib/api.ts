// Typed client for the backend API (backend/app/api/*). All calls are same-origin and carry the
// CSRF header. A 401 anywhere means the session is gone: `onUnauthorized` (set by the app shell)
// sends the user to /login. Every write carries X-Action-Id (lib/undo.ts: one id per user action), and
// an answer with X-Action-Id is recorded in the undo history.
import type {
  AnimalPage,
  AnimalSpeciesIn,
  Species,
  AssignAnimalsIn,
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
  DeviceCreated,
  DocumentOut,
  DocumentPage,
  DocumentTypeOut,
  DocumentUpdate,
  DupGroupPage,
  FacePage,
  GuessIdsIn,
  GuessPage,
  LearningApplyOut,
  LearningStatus,
  ItemCard,
  ItemDetail,
  ItemKind,
  ItemPage,
  ItemTextOut,
  ItemUpdate,
  MergeIn,
  OkOut,
  FaceDeleteOut,
  UndoOut,
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
  TagOut,
  ItemTagOut,
  UserTagIn,
  UploadCreate,
  UploadList,
  UploadOut,
  VaultSensitivity,
  AddAnimalIn,
  AddPersonBoxIn,
  PersonBoxOut,
  AnimalOut,
  BodyOut,
  BoxIn,
  FaceOut,
  ThroughputOut,
  ThroughputRange,
  ItemsPeopleOut,
  PeopleBulkOut,
  PeopleMoveIn,
  PeopleRemoveIn,
  PeopleTagIn,
} from './types'
import { actionIdForRequest, recordAction } from './undo'

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

const WRITE_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE'])

async function request<T>(method: string, url: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const headers: Record<string, string> = { [CSRF_HEADER]: CSRF_VALUE, Accept: 'application/json' }
  const write = WRITE_METHODS.has(method)
  if (write) headers['X-Action-Id'] = actionIdForRequest()
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
  if (write) {
    const action = res.headers.get('X-Action-Id')
    if (action) recordAction(action, method, url)
  }
  return data as T
}

export const get = <T>(url: string, signal?: AbortSignal) => request<T>('GET', url, undefined, signal)
export const post = <T>(url: string, body?: unknown) => request<T>('POST', url, body ?? {})
export const patch = <T>(url: string, body: unknown) => request<T>('PATCH', url, body)
export const put = <T>(url: string, body: unknown) => request<T>('PUT', url, body)
export const del = <T>(url: string) => request<T>('DELETE', url)

/** A DELETE that answers 404 found the thing gone already (an earlier step took it): that is done. */
function goneIsDone(e: unknown): OkOut {
  if (e instanceof ApiError && e.status === 404) return { ok: true }
  throw e
}

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

/**
 * Download link of a Live Photo's clip (section 20): the hidden companion's original for an iPhone
 * pair, or the clip extracted from a motion photo. Both use the still's version key `v` (the
 * server's url_version, which carries an 'n' for vault items). Null when the item has no clip.
 * The caller adds vault=1 for vault items (withVault).
 */
export function liveClipDownloadUrl(detail: Pick<ItemDetail, 'uid' | 'version' | 'nsfw' | 'live_video' | 'urls'>): string | null {
  const live = detail.live_video
  if (!live) return null
  let v = detail.nsfw ? `${detail.version}n` : String(detail.version)
  const still = detail.urls.download ?? detail.urls.original
  if (still) {
    const fromStill = new URLSearchParams(still.split('?')[1] ?? '').get('v')
    if (fromStill) v = fromStill
  }
  const path = live.uid ? `/files/original/${enc(live.uid)}` : `/files/live/${enc(detail.uid)}`
  return `${path}${qs({ v, download: 1 })}`
}

export interface ItemsQuery {
  category?: Category | ''
  kind?: ItemKind | ''
  person?: number[]
  from?: string
  to?: string
  favorite?: boolean
  /** Live Photos and motion photos only (live=1). Pass true or leave it out: false sends live=0. */
  live?: boolean
  misclicks?: 'exclude' | 'include' | 'only'
  /** Forwards (section 29) are left out unless asked for. */
  forwards?: 'exclude' | 'include' | 'only'
  /** Items that came through a messaging app: an app name, or 'any'. */
  received?: string
  dups?: 'collapse' | 'all'
  vault?: boolean
  sort?: 'taken' | 'added'
  /** Tag names, repeated as tag=; every one is required. */
  tag?: string[]
  /** Members of this album only (section 24). */
  album?: number
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
  /** Live Photos and motion photos only (live=1). Pass true or leave it out. */
  live?: boolean
  /** Tag names, repeated as tag=; every one is required. */
  tag?: string[]
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
  /** GET /api/system; `api.system.retryFailed()` queues failed uploads again (POST /api/system/retry-failed).
   *  The learning calls are /api/system/learning (section 18 of docs/ARCHITECTURE.md). */
  system: Object.assign(() => get<SystemOut>('/api/system'), {
    retryFailed: () => post<{ requeued: number }>('/api/system/retry-failed'),
    /** Takes a backup snapshot now (section 31); started is false when one is running or backups are off. */
    backupNow: () => post<{ started: boolean; running: boolean }>('/api/system/backup'),
    /** Finished jobs per bucket by type (section 32): the last hour by minute, the day by hour, the week by six hours. */
    throughput: (range: ThroughputRange) => get<ThroughputOut>(`/api/system/throughput?range=${range}`),
    learning: () => get<LearningStatus>('/api/system/learning'),
    /** The Settings switch (kv flag learning_enabled). The reply body is not used: refetch the status. */
    setLearningEnabled: (enabled: boolean) => patch<OkOut>('/api/system/learning', { enabled }),
    /** The vault sensitivity (same route as the switch). The reply is the status with `vault_moved`
     *  set: the server re-sorted the automatically decided items straight away. */
    setVaultSensitivity: (level: VaultSensitivity) =>
      patch<LearningStatus>('/api/system/learning', { vault_sensitivity: level }),
    /** Trains every learner now and returns the new status. */
    trainLearning: () => post<LearningStatus>('/api/system/learning/train'),
    /** Queues a fresh analysis of automatically decided items and documents; returns the queued counts. */
    applyLearning: () => post<LearningApplyOut>('/api/system/learning/apply'),
    /** Deletes the learned models (the choices themselves stay). The reply body is not used. */
    resetLearning: () => del<OkOut>('/api/system/learning'),
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
    /** Device tokens for the iOS app (same shape as sessions, with device_name set). */
    devices: () => get<SessionOut[]>('/api/auth/devices'),
    /** Answers 403 vault_locked while the vault is locked. The token in the reply is shown once. */
    createDevice: (name: string) => post<DeviceCreated>('/api/auth/devices', { name }),
    revokeDevice: (id: number) => del<OkOut>(`/api/auth/devices/${id}`),
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
    /** A pet box the owner drew (section 22): the server cuts the crop, takes or picks the species, and matches it. */
    addAnimal: (uid: string, body: AddAnimalIn) => post<AnimalOut>(`/api/items/${enc(uid)}/animals`, body),
    /** A person box the owner drew (section 10): a person box with who it is, plus their face when the detector
     *  finds one inside it. Removed like any person box (people.deleteBody). */
    addPersonBox: (uid: string, body: AddPersonBoxIn) => post<PersonBoxOut>(`/api/items/${enc(uid)}/people-boxes`, body),
    /** The detector looks once more at a lower threshold; 409 models_unavailable when it is not loaded. */
    detectAnimals: (uid: string, threshold?: number) =>
      post<AnimalOut[]>(`/api/items/${enc(uid)}/animals/detect`, threshold === undefined ? {} : { threshold }),
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
    /** Free-form tag (section 19): by id, or by name (created when new). An auto row becomes the owner's. */
    addUserTag: (uid: string, body: UserTagIn) => post<ItemTagOut[]>(`/api/items/${enc(uid)}/user-tags`, body),
    /** Removing a suggestion also tells the learner it was wrong. */
    removeUserTag: (uid: string, tagId: number) => del<ItemTagOut[]>(`/api/items/${enc(uid)}/user-tags/${tagId}`),
  },

  /** Free-form tags (section 19 of docs/ARCHITECTURE.md). */
  tags: {
    list: (q?: string) => get<TagOut[]>(`/api/tags${qs({ q })}`),
    create: (name: string) => post<TagOut>('/api/tags', { name }),
    rename: (id: number, name: string) => patch<TagOut>(`/api/tags/${id}`, { name }),
    remove: (id: number) => del<OkOut>(`/api/tags/${id}`),
  },

  trash: {
    list: (cursor?: string | null, limit = 60) => get<ItemPage>(`/api/trash${qs({ cursor, limit })}`),
    empty: () => post<BulkOut>('/api/trash/empty'),
    purge: (uid: string) => del<OkOut>(`/api/trash/${enc(uid)}`),
  },

  review: {
    misclicks: (cursor?: string | null, limit = 60) => get<ItemPage>(`/api/review/misclicks${qs({ cursor, limit })}`),
    /** The flagged forwards, newest first (section 29). */
    forwards: (cursor?: string | null, limit = 60) => get<ItemPage>(`/api/review/forwards${qs({ cursor, limit })}`),
    guesses: (cursor?: string | null, limit = 60, person?: number) =>
      get<GuessPage>(`/api/review/guesses${qs({ cursor, limit, person })}`),
    duplicates: (cursor?: string | null, limit = 30) => get<DupGroupPage>(`/api/duplicates${qs({ cursor, limit })}`),
    setBest: (uid: string) => post<OkOut>(`/api/duplicates/${enc(uid)}/best`),
  },

  people: {
    list: (q: { q?: string; kind?: 'person' | 'pet' | 'all'; filter?: 'all' | 'named' | 'unnamed'; sort?: 'items' | 'name' | 'recent'; cursor?: string | null; limit?: number } = {}) =>
      get<PersonPage>(`/api/people${qs(q)}`),
    create: (body: PersonCreate) => post<PersonOut>('/api/people', body),
    get: (id: number) => get<PersonOut>(`/api/people/${id}`),
    update: (id: number, body: PersonUpdate) => patch<PersonOut>(`/api/people/${id}`, body),
    remove: (id: number) => del<OkOut>(`/api/people/${id}`),
    merge: (body: MergeIn) => post<PersonOut>('/api/people/merge', body),
    /** Every crop of a pet takes this species, and so does the pet (section 22). */
    /** A pet's species everywhere (section 22); the reply names the pet the change ends on (a pet of the same name
     *  and species takes the crops) and how many look-alike crops followed. */
    setSpecies: (id: number, species: Species) => post<AssignOut>(`/api/people/${id}/species`, { species }),
    faces: (id: number, cursor?: string | null, limit = 200) => get<FacePage>(`/api/people/${id}/faces${qs({ cursor, limit })}`),
    animals: (id: number, cursor?: string | null, limit = 120) => get<AnimalPage>(`/api/people/${id}/animals${qs({ cursor, limit })}`),
    /** `relation`: in (the default), related (items about them without them in it, section 10) or all. */
    items: (id: number, cursor?: string | null, limit = 60, vault = false, relation?: 'in' | 'related' | 'all') =>
      get<ItemPage>(`/api/people/${id}/items${qs({ cursor, limit, vault, relation })}`),
    unassignedFaces: (cursor?: string | null, limit = 200) => get<FacePage>(`/api/faces/unassigned${qs({ cursor, limit })}`),
    assignFaces: (body: AssignFacesIn) => post<AssignOut>('/api/faces/assign', body),
    /** "Not a face" (section 10): the server keeps it for learning and removes unassigned look-alikes with it
     *  (followed). A face already gone, for example as an earlier one's look-alike, counts as done. */
    deleteFace: (id: number) =>
      del<FaceDeleteOut>(`/api/faces/${id}`).catch((e: unknown): FaceDeleteOut => {
        if (e instanceof ApiError && e.status === 404) return { ok: true, followed: 0 }
        throw e
      }),
    assignBodies: (body: AssignBodiesIn) => post<AssignOut>('/api/bodies/assign', body),
    /** "Not a person". A box already gone counts as done, like deleteFace. */
    deleteBody: (id: number) => del<OkOut>(`/api/bodies/${id}`).catch(goneIsDone),
    /** Section 10: the owner moves or resizes a face box (422 `no_face` when it went where no face is). */
    moveFace: (id: number, box: BoxIn) => put<FaceOut>(`/api/faces/${id}/box`, box),
    /** Section 10: the owner moves or resizes a person box. */
    moveBody: (id: number, box: BoxIn) => put<BodyOut>(`/api/bodies/${id}/box`, box),
    confirmGuesses: (body: GuessIdsIn) => post<BulkOut>('/api/guesses/confirm', body),
    rejectGuesses: (body: GuessIdsIn) => post<BulkOut>('/api/guesses/reject', body),
    suggestions: (limit = 30) => get<SuggestionList>(`/api/suggestions${qs({ limit })}`),
    dismissSuggestion: (a: number, b: number) => post<OkOut>('/api/suggestions/dismiss', { a, b }),
    recluster: () => post<OkOut>('/api/recluster'),
    /** Section 10, people on a selection: who the selected items show (faces, bodies, pet boxes, tags). */
    onItems: (uids: string[]) => post<ItemsPeopleOut>('/api/items/people', { uids }),
    /** Everything of `from_person_id` on those items goes to the chosen person or pet (or a new name). */
    moveOnItems: (body: PeopleMoveIn) => post<PeopleBulkOut>('/api/items/people/move', body),
    /** The person's faces, bodies and boxes on those items are unassigned and their tags there removed. */
    removeFromItems: (body: PeopleRemoveIn) => post<PeopleBulkOut>('/api/items/people/remove', body),
    /** Tags a person or pet on every item, as on one item (relation in or related). */
    tagItems: (body: PeopleTagIn) => post<PeopleBulkOut>('/api/items/people/tag', body),
  },

  animals: {
    assign: (body: AssignAnimalsIn) => post<AssignOut>('/api/animals/assign', body),
    /** The owner's species for these boxes; look-alike crops follow (section 22). */
    setSpecies: (body: AnimalSpeciesIn) => post<AssignOut>('/api/animals/species', body),
    /** "Not a pet". A box already gone counts as done, like deleteFace. */
    remove: (id: number) => del<OkOut>(`/api/animals/${id}`).catch(goneIsDone),
    /** Section 10: the owner moves or resizes a pet box. */
    move: (id: number, box: BoxIn) => put<AnimalOut>(`/api/animals/${id}/box`, box),
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
  /** Undo (section 33): put back what an action changed. Undoing the answer's undo_id is a redo. */
  actions: {
    undo: (id: string) => post<UndoOut>(`/api/actions/${enc(id)}/undo`),
  },
}

export type { ItemCard }
