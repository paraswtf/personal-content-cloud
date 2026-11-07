// Upload engine for the /upload page (docs/ARCHITECTURE.md section 6: the hub's chunked, resumable
// protocol).
//
// UploadEngine is framework-agnostic: it owns the queue, hashes files up to 1 GiB in 4 MiB slices
// with hash-wasm (a known sha256 answers 'duplicate' before any bytes move), sends chunks with
// retries and resume, polls processing status in batches and keeps a resume map in
// localStorage['pcc.pending']. Any file type is accepted; the server's probe decides what it is.
// Every queue item carries its UploadSource (web, share, camera). useUploader() exposes the engine
// to React through useSyncExternalStore. The engine is a module singleton, so uploads keep running
// while the owner browses other pages of the app.
import { useSyncExternalStore } from 'react'
import { createSHA256 } from 'hash-wasm'
import { ApiError, OffsetMismatchError, api, uploadChunk } from './api'
import { TERMINAL_UPLOAD_STATUSES, type UploadOut, type UploadSource, type UploadStatus } from './types'

export const HASH_STEP = 4 * 1024 * 1024
export const HASH_LIMIT = 1024 * 1024 * 1024
export const MAX_IN_FLIGHT = 3
export const MAX_ATTEMPTS = 8
export const BACKOFF_S = [1, 2, 4, 8, 16, 30]
export const POLL_MS = 2000
const POLL_BATCH = 200
const MAX_RESYNCS = 50
const MIN_SLICE = 256 * 1024 // smallest chunk we fall back to when a proxy rejects the request size (413)
const PENDING_KEY = 'pcc.pending'
const CLIENT_KEY = 'pcc.clientId'
const PENDING_MAX_AGE_MS = 14 * 24 * 3600 * 1000
const EMIT_THROTTLE_MS = 100

/** Where a queue item is in its life. */
export type Phase =
  | 'waiting' // in the queue, not started yet
  | 'checking' // hashing locally / resuming / creating the upload
  | 'uploading' // sending chunks
  | 'finishing' // POST complete
  | 'processing' // server status queued or processing, polled
  | 'done' // server reached a terminal status (see item.status)
  | 'failed' // client-side failure (item.message says why)
  | 'cancelled'

export type FileKind = 'image' | 'video' | 'pdf' | 'document' | 'other'

export interface QueueItem {
  /** Local id, unique per queue entry. */
  key: string
  file: File
  name: string
  size: number
  kind: FileKind
  /** name|size|lastModified, used for the resume map and to skip double adds. */
  fileKey: string
  /** How the file came in: picked or dropped (web), shared to the app (share), or the in-app camera. */
  source: UploadSource
  phase: Phase
  /** 0..1 for checking (hash) and uploading (bytes sent); null when indeterminate. */
  progress: number | null
  uploadId: string | null
  status: UploadStatus | null
  message: string | null
  /** uid of the library item once the server stored it (or the existing item for a duplicate). */
  itemUid: string | null
  thumbUrl: string | null
  /** Rejected before any request (empty or too large): retry is pointless. */
  localReject: boolean
  /** True when this item continued an interrupted upload. */
  resumed: boolean
  /** Epoch ms of the next retry while backing off after a network error, else null. */
  retryAt: number | null
  /** Waiting for the browser to come back online. */
  offline: boolean
}

export interface UploaderConfig {
  maxUploadBytes: number
  chunkSize: number
  accept: string
}

export interface UploadSummary {
  total: number
  added: number
  duplicates: number
  failed: number
  /** Items still waiting, checking, uploading or finishing (bytes still to send). */
  sending: number
  processing: number
  cancelled: number
}

export interface UploaderSnapshot {
  items: QueueItem[]
  /** True while any file still has bytes to send: leaving the page would interrupt it. */
  active: boolean
  summary: UploadSummary
}

export interface AddResult {
  added: number
  rejected: number
  skipped: number
}

export interface PendingEntry {
  id: string
  name: string
  size: number
  t: number
}

// ---------------------------------------------------------------------------
// storage helpers (every access guarded: storage can be blocked or full)

function lsGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function lsSet(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key)
    else window.localStorage.setItem(key, value)
  } catch {
    // storage unavailable: resume is a convenience only
  }
}

function randomId(): string {
  try {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  } catch {
    // randomUUID needs a secure context; fall through
  }
  const bytes = new Uint8Array(16)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) crypto.getRandomValues(bytes)
  else for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

/** Random id for this browser, created once and kept in localStorage['pcc.clientId']. */
export function getClientId(): string {
  let id = lsGet(CLIENT_KEY)
  if (!id) {
    id = randomId()
    lsSet(CLIENT_KEY, id)
  }
  return id
}

export function fileKeyOf(file: File): string {
  return `${file.name}|${file.size}|${file.lastModified}`
}

function readPending(): Record<string, PendingEntry> {
  const raw = lsGet(PENDING_KEY)
  if (!raw) return {}
  try {
    const parsed: unknown = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object') return {}
    const out: Record<string, PendingEntry> = {}
    const now = Date.now()
    for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
      if (!v || typeof v !== 'object') continue
      const e = v as Partial<PendingEntry>
      if (typeof e.id !== 'string') continue
      const t = typeof e.t === 'number' ? e.t : now
      if (now - t > PENDING_MAX_AGE_MS) continue
      out[k] = { id: e.id, name: typeof e.name === 'string' ? e.name : k.split('|')[0], size: Number(e.size) || 0, t }
    }
    return out
  } catch {
    return {}
  }
}

function writePending(map: Record<string, PendingEntry>): void {
  lsSet(PENDING_KEY, Object.keys(map).length ? JSON.stringify(map) : null)
}

function setPending(fileKey: string, entry: PendingEntry | null): void {
  const map = readPending()
  if (entry) map[fileKey] = entry
  else delete map[fileKey]
  writePending(map)
}

/**
 * Interrupted uploads from earlier visits that the server still holds as 'uploading'.
 * Entries the server no longer knows, or that moved on, are pruned from the map.
 */
export async function checkInterrupted(): Promise<PendingEntry[]> {
  const map = readPending()
  const keys = Object.keys(map)
  if (!keys.length) return []
  const ids = keys.map((k) => map[k].id)
  const found = new Map<string, UploadOut>()
  for (let i = 0; i < ids.length; i += POLL_BATCH) {
    const res = await api.uploads.many(ids.slice(i, i + POLL_BATCH))
    for (const u of res.uploads) found.set(u.id, u)
  }
  const keep: Record<string, PendingEntry> = {}
  const latest = readPending()
  for (const k of Object.keys(latest)) {
    const u = found.get(latest[k].id)
    // keep entries we did not ask about (added meanwhile) and ones still uploading
    if (!ids.includes(latest[k].id) || (u && u.status === 'uploading')) keep[k] = latest[k]
  }
  writePending(keep)
  return Object.values(keep).filter((e) => ids.includes(e.id))
}

// ---------------------------------------------------------------------------
// file checks

// The browser often reports no MIME type (HEIC/MOV on Windows, camera RAW, some Android pickers), so
// the extension decides the local icon. This is display only: every file is uploaded and the
// server's probe makes the final decision.
const IMAGE_EXTENSIONS = new Set([
  '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tif', '.tiff', '.heic', '.heif', '.avif', '.jxl',
  '.dng', '.cr2', '.cr3', '.nef', '.arw', '.orf', '.rw2', '.raf', '.srw', '.pef',
])
const VIDEO_EXTENSIONS = new Set(['.mp4', '.mov', '.m4v', '.webm', '.mkv', '.avi', '.3gp', '.mts', '.m2ts', '.wmv', '.mpg', '.mpeg'])
const DOCUMENT_EXTENSIONS = new Set(['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.txt', '.md', '.csv', '.rtf', '.odt', '.ods', '.odp'])
const DOCUMENT_MIME_PREFIXES = [
  'text/',
  'application/msword',
  'application/vnd.openxmlformats-officedocument',
  'application/vnd.ms-',
  'application/vnd.oasis.opendocument',
  'application/rtf',
]

export function extensionOf(name: string): string {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i).toLowerCase() : ''
}

/** Kind of a file name plus MIME type, for icons only (pdf and office files count as documents). */
export function kindOfName(name: string, mime?: string | null): FileKind {
  const type = (mime || '').toLowerCase()
  const ext = extensionOf(name)
  if (type === 'application/pdf' || ext === '.pdf') return 'pdf'
  if (type.startsWith('image/')) return 'image'
  if (type.startsWith('video/')) return 'video'
  if (VIDEO_EXTENSIONS.has(ext)) return 'video'
  if (IMAGE_EXTENSIONS.has(ext)) return 'image'
  if (DOCUMENT_EXTENSIONS.has(ext) || DOCUMENT_MIME_PREFIXES.some((p) => type.startsWith(p))) return 'document'
  return 'other'
}

export function kindOf(file: File): FileKind {
  return kindOfName(file.name, file.type)
}

/** Does the file match an <input accept> string ("image/*,video/*,.heic,...")? The any-type token matches every file. */
export function matchesAccept(file: File, accept: string): boolean {
  const type = (file.type || '').toLowerCase()
  const ext = extensionOf(file.name)
  const tokens = accept
    .split(',')
    .map((t) => t.trim().toLowerCase())
    .filter(Boolean)
  for (const t of tokens) {
    if (t === '*/*' || t === '*') return true
    if (t.startsWith('.')) {
      if (ext === t) return true
    } else if (t.endsWith('/*')) {
      if (type.startsWith(t.slice(0, -1))) return true
    } else if (type === t) {
      return true
    }
  }
  return !type && (IMAGE_EXTENSIONS.has(ext) || VIDEO_EXTENSIONS.has(ext))
}

/** "16 GB" style label for a size limit. */
export function formatLimit(bytes: number): string {
  const gb = bytes / 1024 ** 3
  if (gb >= 1) return `${Number.isInteger(gb) ? gb : gb.toFixed(1)} GB`
  return `${Math.round(bytes / 1024 ** 2)} MB`
}

// ---------------------------------------------------------------------------
// errors and waiting

class CancelledError extends Error {
  constructor() {
    super('cancelled')
  }
}

function isAbort(err: unknown): boolean {
  return (
    err instanceof CancelledError ||
    (err instanceof DOMException && err.name === 'AbortError') ||
    (err instanceof Error && err.name === 'AbortError')
  )
}

/**
 * Network failures (fetch TypeError, XHR status 0) and server-side 5xx/429 are worth retrying.
 * 507 (the server is out of storage space) does not clear in seconds, so it fails at once.
 */
function isRetryable(err: unknown): boolean {
  if (err instanceof OffsetMismatchError || isAbort(err)) return false
  if (err instanceof ApiError) return err.status === 0 || err.status === 429 || (err.status >= 500 && err.status !== 507)
  return err instanceof TypeError
}

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 0) return 'Connection lost. Check your internet and try again.'
    if (err.status === 401) return 'Your session has ended. Sign in again, then press Retry to continue.'
    if (err.status === 413) return 'The server refused a file this large.'
    if (err.status === 507) return 'The server is out of storage space. Free some space, then press Retry.'
    return err.message
  }
  if (err instanceof DOMException && err.name === 'NotReadableError')
    return 'This file could not be read. It may have been moved or deleted.'
  if (err instanceof TypeError) return 'Connection lost. Check your internet and try again.'
  return err instanceof Error && err.message ? err.message : 'Something went wrong.'
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new CancelledError())
    const t = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(t)
      reject(new CancelledError())
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

function waitOnline(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (typeof navigator === 'undefined' || navigator.onLine !== false) return resolve()
    if (signal.aborted) return reject(new CancelledError())
    const done = () => {
      window.removeEventListener('online', done)
      signal.removeEventListener('abort', onAbort)
      resolve()
    }
    const onAbort = () => {
      window.removeEventListener('online', done)
      reject(new CancelledError())
    }
    window.addEventListener('online', done)
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

interface WakeLockLike {
  release: () => Promise<void>
}

// ---------------------------------------------------------------------------
// engine

export class UploadEngine {
  private items = new Map<string, QueueItem>()
  private order: string[] = []
  private controllers = new Map<string, AbortController>()
  private running = new Set<string>()
  private listeners = new Set<() => void>()
  private settledListeners = new Set<(item: QueueItem) => void>()
  private snapshot: UploaderSnapshot
  private emitTimer: ReturnType<typeof setTimeout> | null = null
  private pollTimer: ReturnType<typeof setInterval> | null = null
  private polling = false
  private config: UploaderConfig = { maxUploadBytes: 16 * 1024 ** 3, chunkSize: 8 * 1024 ** 2, accept: '*/*' }
  private sliceLimit = Number.POSITIVE_INFINITY // learned from 413s; caps chunk size for later chunks and files
  private wasActive = false
  private wakeLock: WakeLockLike | null = null
  private seq = 0

  constructor() {
    this.snapshot = this.buildSnapshot()
  }

  // public API

  configure(config: Partial<UploaderConfig>): void {
    this.config = { ...this.config, ...config }
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  getSnapshot = (): UploaderSnapshot => this.snapshot

  /** Called once for each item that reaches a final state (done, failed or cancelled). */
  onSettled(listener: (item: QueueItem) => void): () => void {
    this.settledListeners.add(listener)
    return () => this.settledListeners.delete(listener)
  }

  /** Queue files. Files already in the queue (same name, size, date) and not failed are skipped. */
  add(files: Iterable<File>, source: UploadSource = 'web'): AddResult {
    const result: AddResult = { added: 0, rejected: 0, skipped: 0 }
    const live = new Set(
      [...this.items.values()].filter((i) => i.phase !== 'failed' && i.phase !== 'cancelled').map((i) => i.fileKey),
    )
    for (const file of files) {
      const fileKey = fileKeyOf(file)
      if (live.has(fileKey)) {
        result.skipped++
        continue
      }
      live.add(fileKey)
      const item: QueueItem = {
        key: `q${++this.seq}`,
        file,
        name: file.name || 'Untitled',
        size: file.size,
        kind: kindOf(file),
        fileKey,
        source,
        phase: 'waiting',
        progress: null,
        uploadId: null,
        status: null,
        message: null,
        itemUid: null,
        thumbUrl: null,
        localReject: false,
        resumed: false,
        retryAt: null,
        offline: false,
      }
      const reject = this.localRejection(file)
      if (reject) {
        item.phase = 'failed'
        item.message = reject
        item.localReject = true
        result.rejected++
      } else {
        result.added++
      }
      this.items.set(item.key, item)
      this.order.push(item.key)
    }
    this.emit(true)
    this.pump()
    return result
  }

  cancel(key: string): void {
    const item = this.items.get(key)
    if (!item || !this.canCancel(item)) return
    this.controllers.get(key)?.abort()
    if (item.uploadId) {
      const id = item.uploadId
      api.uploads.cancel(id).catch(() => undefined)
    }
    setPending(item.fileKey, null)
    this.update(key, { phase: 'cancelled', progress: null, retryAt: null, offline: false, message: 'Cancelled.' }, true)
    this.settle(key)
  }

  retry(key: string): void {
    const item = this.items.get(key)
    if (!item || !this.canRetry(item)) return
    this.update(
      key,
      {
        phase: 'waiting',
        progress: null,
        status: null,
        message: null,
        itemUid: null,
        thumbUrl: null,
        retryAt: null,
        offline: false,
        // a server-side processing error needs a fresh upload, and so does a cancelled one (its
        // DELETE may still be in flight, so a GET could briefly report it as resumable);
        // otherwise resume if possible
        uploadId: item.status === 'error' || item.phase === 'cancelled' ? null : item.uploadId,
      },
      true,
    )
    this.pump()
  }

  remove(key: string): void {
    const item = this.items.get(key)
    if (!item) return
    if (this.canCancel(item)) this.cancel(key)
    this.items.delete(key)
    this.order = this.order.filter((k) => k !== key)
    this.emit(true)
  }

  /** Drop every item that needs nothing more (done, failed, cancelled). */
  clearFinished(): void {
    for (const key of [...this.order]) {
      const item = this.items.get(key)
      if (item && this.isFinished(item)) this.items.delete(key)
    }
    this.order = this.order.filter((k) => this.items.has(k))
    this.emit(true)
  }

  canCancel(item: QueueItem): boolean {
    return item.phase === 'waiting' || item.phase === 'checking' || item.phase === 'uploading'
  }

  canRetry(item: QueueItem): boolean {
    if (item.localReject) return false
    if (item.phase === 'failed' || item.phase === 'cancelled') return true
    return item.phase === 'done' && item.status === 'error'
  }

  isFinished(item: QueueItem): boolean {
    return item.phase === 'done' || item.phase === 'failed' || item.phase === 'cancelled'
  }

  // internals: state

  private localRejection(file: File): string | null {
    if (file.size > this.config.maxUploadBytes)
      return `This file is larger than the ${formatLimit(this.config.maxUploadBytes)} limit.`
    if (file.size === 0) return 'This file is empty.'
    return null
  }

  private update(key: string, patch: Partial<QueueItem>, immediate = false): QueueItem | null {
    const cur = this.items.get(key)
    if (!cur) return null
    // late callbacks from a cancelled run must not revive the item
    if (cur.phase === 'cancelled' && !('phase' in patch)) return cur
    const next = { ...cur, ...patch }
    this.items.set(key, next)
    this.emit(immediate)
    return next
  }

  private applyServer(key: string, u: UploadOut): void {
    const item = this.items.get(key)
    if (!item || item.phase === 'cancelled') return
    const terminal = TERMINAL_UPLOAD_STATUSES.has(u.status)
    const processing = u.status === 'queued' || u.status === 'processing'
    const patch: Partial<QueueItem> = {
      uploadId: u.id,
      status: u.status,
      message: u.message,
      itemUid: u.item_uid,
      thumbUrl: u.item_thumb_url,
      retryAt: null,
      offline: false,
    }
    if (terminal) {
      patch.phase = u.status === 'cancelled' ? 'cancelled' : 'done'
      patch.progress = null
      setPending(item.fileKey, null)
    } else if (processing) {
      patch.phase = 'processing'
      patch.progress = null
    }
    this.update(key, patch, true)
    if (terminal) this.settle(key)
    if (processing) this.ensurePolling()
  }

  private fail(key: string, message: string): void {
    const item = this.items.get(key)
    if (!item || item.phase === 'cancelled') return
    this.update(key, { phase: 'failed', message, progress: null, retryAt: null, offline: false }, true)
    this.settle(key)
  }

  private settle(key: string): void {
    const item = this.items.get(key)
    if (!item) return
    for (const l of this.settledListeners) {
      try {
        l(item)
      } catch {
        // listeners must not break the engine
      }
    }
  }

  private buildSnapshot(): UploaderSnapshot {
    const items = this.order.map((k) => this.items.get(k)).filter((i): i is QueueItem => !!i)
    const summary: UploadSummary = {
      total: items.length,
      added: 0,
      duplicates: 0,
      failed: 0,
      sending: 0,
      processing: 0,
      cancelled: 0,
    }
    for (const i of items) {
      if (i.phase === 'waiting' || i.phase === 'checking' || i.phase === 'uploading' || i.phase === 'finishing') summary.sending++
      else if (i.phase === 'processing') summary.processing++
      else if (i.phase === 'failed') summary.failed++
      else if (i.phase === 'cancelled') summary.cancelled++
      else if (i.phase === 'done') {
        if (i.status === 'stored') summary.added++
        else if (i.status === 'duplicate') summary.duplicates++
        else if (i.status === 'rejected' || i.status === 'error') summary.failed++
      }
    }
    return { items, active: summary.sending > 0, summary }
  }

  /** Publish a new snapshot: now for phase changes, throttled for progress ticks. */
  private emit(immediate = false): void {
    if (immediate) {
      if (this.emitTimer) {
        clearTimeout(this.emitTimer)
        this.emitTimer = null
      }
      this.flush()
      return
    }
    if (this.emitTimer) return
    this.emitTimer = setTimeout(() => {
      this.emitTimer = null
      this.flush()
    }, EMIT_THROTTLE_MS)
  }

  private flush(): void {
    this.snapshot = this.buildSnapshot()
    this.syncActiveEffects(this.snapshot.active)
    for (const l of this.listeners) l()
  }

  // internals: page lifecycle (unload warning, wake lock)

  private onBeforeUnload = (e: BeforeUnloadEvent) => {
    e.preventDefault()
    e.returnValue = ''
  }

  private onVisibility = () => {
    if (document.visibilityState === 'visible' && this.wasActive && !this.wakeLock) void this.acquireWakeLock()
  }

  private syncActiveEffects(active: boolean): void {
    if (typeof window === 'undefined' || active === this.wasActive) return
    this.wasActive = active
    if (active) {
      window.addEventListener('beforeunload', this.onBeforeUnload)
      document.addEventListener('visibilitychange', this.onVisibility)
      void this.acquireWakeLock()
    } else {
      window.removeEventListener('beforeunload', this.onBeforeUnload)
      document.removeEventListener('visibilitychange', this.onVisibility)
      const lock = this.wakeLock
      this.wakeLock = null
      lock?.release().catch(() => undefined)
    }
  }

  private async acquireWakeLock(): Promise<void> {
    try {
      const wl = (navigator as Navigator & { wakeLock?: { request: (t: 'screen') => Promise<WakeLockLike & EventTarget> } })
        .wakeLock
      if (!wl || document.visibilityState !== 'visible') return
      const lock = await wl.request('screen')
      if (!this.wasActive) {
        lock.release().catch(() => undefined)
        return
      }
      this.wakeLock = lock
      lock.addEventListener('release', () => {
        if (this.wakeLock === lock) this.wakeLock = null
      })
    } catch {
      // not supported, not a secure context, or refused (battery saver): uploads work anyway
    }
  }

  // internals: scheduling

  private pump(): void {
    for (const key of this.order) {
      if (this.running.size >= MAX_IN_FLIGHT) break
      const item = this.items.get(key)
      if (!item || item.phase !== 'waiting' || this.running.has(key)) continue
      this.running.add(key)
      const ctrl = new AbortController()
      this.controllers.set(key, ctrl)
      void this.run(key, ctrl.signal).finally(() => {
        this.running.delete(key)
        if (this.controllers.get(key) === ctrl) this.controllers.delete(key)
        this.pump()
      })
    }
  }

  /** Call fn, retrying network errors and 5xx with backoff (MAX_ATTEMPTS attempts in total). */
  private async withRetry<T>(key: string, signal: AbortSignal, fn: () => Promise<T>): Promise<T> {
    for (let attempt = 1; ; attempt++) {
      if (signal.aborted) throw new CancelledError()
      try {
        const res = await fn()
        if (signal.aborted) throw new CancelledError()
        const cur = this.items.get(key)
        if (cur && (cur.retryAt || cur.offline)) this.update(key, { retryAt: null, offline: false }, true)
        return res
      } catch (err) {
        if (signal.aborted) throw new CancelledError()
        if (!isRetryable(err) || attempt >= MAX_ATTEMPTS) throw err
        if (typeof navigator !== 'undefined' && navigator.onLine === false) {
          this.update(key, { offline: true, retryAt: null }, true)
          await waitOnline(signal)
          this.update(key, { offline: false }, true)
          attempt-- // time spent offline does not use up attempts
          continue
        }
        const delay = BACKOFF_S[Math.min(attempt - 1, BACKOFF_S.length - 1)] * 1000
        this.update(key, { retryAt: Date.now() + delay }, true)
        await sleep(delay, signal)
      }
    }
  }

  private async run(key: string, signal: AbortSignal): Promise<void> {
    try {
      await this.process(key, signal)
    } catch (err) {
      if (isAbort(err) || signal.aborted) return
      const item = this.items.get(key)
      // definite 4xx answers end this upload id; network failures and lost access (401/403, fixed by
      // signing in again) keep it resumable
      if (
        item &&
        err instanceof ApiError &&
        err.status >= 400 &&
        err.status < 500 &&
        err.status !== 429 &&
        err.status !== 401 &&
        err.status !== 403
      ) {
        setPending(item.fileKey, null)
        this.update(key, { uploadId: null })
      }
      this.fail(key, errorMessage(err))
    }
  }

  private async process(key: string, signal: AbortSignal): Promise<void> {
    let item = this.items.get(key)
    if (!item) return
    this.update(key, { phase: 'checking', progress: null, message: null }, true)

    // 1. resume an earlier upload of the same file when the server still has it
    let upload: UploadOut | null = await this.tryResume(key, signal)
    if (signal.aborted) throw new CancelledError()

    // 2. new upload: hash (small enough files only), then create
    if (!upload) {
      item = this.items.get(key)
      if (!item) return
      let sha256: string | null = null
      if (item.size <= HASH_LIMIT) sha256 = await this.hash(key, item.file, signal)
      this.update(key, { progress: null }, true)
      const body = {
        filename: item.name,
        size: item.size,
        mime: item.file.type || null,
        client_id: getClientId(),
        sha256,
        source: item.source,
      }
      upload = await this.withRetry(key, signal, async () => {
        const u = await api.uploads.create(body)
        // cancelled while the request was in the air: release the server-side upload
        if (signal.aborted && u.status === 'uploading') api.uploads.cancel(u.id).catch(() => undefined)
        return u
      })
      if (upload.status === 'uploading')
        setPending(item.fileKey, { id: upload.id, name: item.name, size: item.size, t: Date.now() })
      this.update(key, { uploadId: upload.id, status: upload.status })
    }
    if (upload.status !== 'uploading') {
      this.applyServer(key, upload)
      return
    }

    // 3. chunks, then complete (a 409 on complete means the server has fewer bytes: resend)
    let offset = upload.received_bytes
    const chunkSize = upload.chunk_size > 0 ? upload.chunk_size : this.config.chunkSize
    for (let round = 0; ; round++) {
      offset = await this.sendChunks(key, upload.id, offset, chunkSize, signal)
      this.update(key, { phase: 'finishing', progress: 1, retryAt: null }, true)
      try {
        const id = upload.id
        const done = await this.withRetry(key, signal, () => api.uploads.complete(id))
        this.applyServer(key, done)
        return
      } catch (err) {
        if (!(err instanceof ApiError && err.status === 409) || round >= 3) throw err
        const id = upload.id
        const cur = await this.withRetry(key, signal, () => api.uploads.get(id))
        if (cur.status !== 'uploading') {
          this.applyServer(key, cur)
          return
        }
        offset = cur.received_bytes
      }
    }
  }

  /** Returns the server's upload when this file has a resumable or finished earlier upload. */
  private async tryResume(key: string, signal: AbortSignal): Promise<UploadOut | null> {
    const item = this.items.get(key)
    if (!item) return null
    const pending = readPending()[item.fileKey]
    const id = pending?.id ?? item.uploadId
    if (!id) return null
    try {
      const u = await this.withRetry(key, signal, () => api.uploads.get(id))
      if (u.status === 'uploading' || u.status === 'queued' || u.status === 'processing') {
        this.update(key, { resumed: u.received_bytes > 0 || u.status !== 'uploading', uploadId: u.id }, true)
        return u
      }
      // cancelled, failed or finished earlier: start again so this selection is handled
      setPending(item.fileKey, null)
      return null
    } catch (err) {
      if (isAbort(err)) throw err
      // lost access (401/403) is fixed by signing in again: keep the upload resumable
      if (err instanceof ApiError && err.status >= 400 && err.status < 500 && err.status !== 401 && err.status !== 403) {
        setPending(item.fileKey, null)
        return null
      }
      throw err
    }
  }

  /** SHA-256 over 4 MiB slices: memory stays flat whatever the file size. */
  private async hash(key: string, file: File, signal: AbortSignal): Promise<string> {
    const hasher = await createSHA256()
    hasher.init()
    const size = file.size
    let last = 0
    for (let pos = 0; pos < size; pos += HASH_STEP) {
      if (signal.aborted) throw new CancelledError()
      const buf = new Uint8Array(await file.slice(pos, Math.min(pos + HASH_STEP, size)).arrayBuffer())
      hasher.update(buf)
      const now = Date.now()
      if (now - last > 80) {
        last = now
        this.update(key, { progress: Math.min(1, (pos + buf.length) / size) })
      }
    }
    return hasher.digest('hex')
  }

  /** Send chunks from offset to the end. Returns the final offset (the file size). */
  private async sendChunks(key: string, id: string, start: number, chunkSize: number, signal: AbortSignal): Promise<number> {
    const item = this.items.get(key)
    if (!item) throw new CancelledError()
    const { file, size } = item
    let offset = start
    let resyncs = 0
    let slice = Math.min(chunkSize, this.sliceLimit)
    this.update(key, { phase: 'uploading', progress: size ? offset / size : 0, retryAt: null }, true)
    while (offset < size) {
      const end = Math.min(offset + slice, size)
      const from = offset
      try {
        const res = await this.withRetry(key, signal, () =>
          uploadChunk(id, from, file.slice(from, end), {
            signal,
            onProgress: (sent) => this.update(key, { progress: Math.min(1, (from + sent) / size) }),
          }),
        )
        offset = Number.isFinite(res?.received_bytes) ? res.received_bytes : end
      } catch (err) {
        if (err instanceof ApiError && err.status === 413 && end - from > MIN_SLICE) {
          // A proxy in front of the server refused the request size: resend this offset in smaller
          // pieces, and remember the size for the rest of this file and the next ones.
          slice = Math.max(MIN_SLICE, Math.floor((end - from) / 2))
          this.sliceLimit = Math.min(this.sliceLimit, slice)
          continue
        }
        if (!(err instanceof OffsetMismatchError)) throw err
        // A forward resync (the server stored a chunk whose answer was lost) is progress; only
        // resyncs that do not move the offset forward count toward the cap.
        if (err.receivedBytes > offset) resyncs = 0
        else if (++resyncs > MAX_RESYNCS) throw new Error('The upload got out of step with the server. Try again.')
        offset = err.receivedBytes
      }
      this.update(key, { progress: size ? Math.min(1, offset / size) : 1 })
    }
    return offset
  }

  // internals: status polling

  private ensurePolling(): void {
    if (this.pollTimer) return
    this.pollTimer = setInterval(() => void this.poll(), POLL_MS)
  }

  private async poll(): Promise<void> {
    if (this.polling) return
    const waiting = [...this.items.values()].filter((i) => i.phase === 'processing' && i.uploadId)
    if (!waiting.length) {
      if (this.pollTimer) clearInterval(this.pollTimer)
      this.pollTimer = null
      return
    }
    this.polling = true
    try {
      const byId = new Map<string, string[]>()
      for (const i of waiting) byId.set(i.uploadId!, [...(byId.get(i.uploadId!) ?? []), i.key])
      const ids = [...byId.keys()]
      for (let b = 0; b < ids.length; b += POLL_BATCH) {
        const res = await api.uploads.many(ids.slice(b, b + POLL_BATCH))
        for (const u of res.uploads) {
          for (const key of byId.get(u.id) ?? []) {
            const cur = this.items.get(key)
            if (cur?.phase === 'processing' && (cur.status !== u.status || TERMINAL_UPLOAD_STATUSES.has(u.status)))
              this.applyServer(key, u)
          }
        }
      }
    } catch {
      // transient: the next tick tries again
    } finally {
      this.polling = false
    }
  }
}

let engine: UploadEngine | null = null

/** The app-wide upload engine (created on first use). */
export function getUploadEngine(): UploadEngine {
  if (!engine) engine = new UploadEngine()
  return engine
}

/** React binding: the current queue snapshot plus the engine's actions. */
export function useUploader(): UploaderSnapshot & { engine: UploadEngine } {
  const e = getUploadEngine()
  const snap = useSyncExternalStore(e.subscribe, e.getSnapshot, e.getSnapshot)
  return { ...snap, engine: e }
}
