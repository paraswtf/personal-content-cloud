// Undo history (docs/ARCHITECTURE.md sections 14 and 33). Every write the app sends carries an action id
// (X-Action-Id); one user action in the app (a React Query mutation, MutationCache in main.tsx) shares one
// id across all its requests, so a bulk loop undoes as one. When the server answers with X-Action-Id, the
// change can be undone and is recorded here. The history lives in this browser only (localStorage, newest
// first, at most MAX entries), labelled by the success message the app showed for it (claimForToast) or,
// failing that, by a plain description of the request (describe). An entry's activeId is the server
// action whose effect is live (the original, or the redo after an undo); undoId is the undo to undo for a
// redo. This module holds no network code: lib/undoRun.ts sends the undo and redo requests.
import { useSyncExternalStore } from 'react'

export type UndoState = 'done' | 'undone'

export interface UndoEntry {
  /** The action id the change was made with: the entry's key. */
  id: string
  label: string | null
  /** When it was recorded, ms since the epoch. */
  at: number
  method: string
  path: string
  state: UndoState
  /** The server action whose effect is live now: undo this one. */
  activeId: string
  /** After an undo: the undo's own action id, undone for a redo. */
  undoId?: string
  undoneAt?: number
  /** The server no longer keeps it (older than 30 days, or unknown there). */
  expired?: boolean
  /** The last undo or redo left some of it as it had become. */
  partial?: boolean
}

const KEY = 'pcc.undo.v1'
export const MAX_ENTRIES = 200
const ID_RE = /^[A-Za-z0-9_-]{8,64}$/
/** A success message shown this soon after an action finished still names that action. */
const CLAIM_MS = 2500

let entries: UndoEntry[] = load()
const listeners = new Set<() => void>()

function valid(e: unknown): e is UndoEntry {
  if (!e || typeof e !== 'object') return false
  const x = e as Record<string, unknown>
  return typeof x.id === 'string' && ID_RE.test(x.id) && typeof x.at === 'number' && typeof x.method === 'string'
    && typeof x.path === 'string' && (x.state === 'done' || x.state === 'undone') && typeof x.activeId === 'string'
}

function load(): UndoEntry[] {
  try {
    const raw = window.localStorage.getItem(KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.filter(valid).slice(0, MAX_ENTRIES) : []
  } catch {
    return []
  }
}

function save(): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(entries))
  } catch {
    // private mode or a full quota: the history still works for this page
  }
}

function set(next: UndoEntry[]): void {
  entries = next.slice(0, MAX_ENTRIES)
  save()
  for (const fn of listeners) fn()
}

function update(id: string, change: Partial<UndoEntry>): void {
  if (!entries.some((e) => e.id === id)) return
  set(entries.map((e) => (e.id === id ? { ...e, ...change } : e)))
}

if (typeof window !== 'undefined') {
  // another tab changed the history
  window.addEventListener('storage', (ev) => {
    if (ev.key !== KEY) return
    entries = load()
    for (const fn of listeners) fn()
  })
}

export function subscribe(fn: () => void): () => void {
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
  }
}

export function getEntries(): UndoEntry[] {
  return entries
}

export function getEntry(id: string): UndoEntry | undefined {
  return entries.find((e) => e.id === id)
}

/** The history, newest first, re-rendering on every change. */
export function useUndoHistory(): UndoEntry[] {
  return useSyncExternalStore(subscribe, getEntries, getEntries)
}

// ------------------------------------------------------------------ action ids
/** A new action id: 32 hex digits (crypto.getRandomValues works outside secure contexts too). */
export function newActionId(): string {
  const bytes = new Uint8Array(16)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

let current: string | null = null
let lastEnded: { id: string; at: number } | null = null

/** A user action starts (MutationCache onMutate): its requests share this id until it ends. */
export function beginAction(): string {
  current = newActionId()
  return current
}

/** The user action ends (MutationCache onSettled). */
export function endAction(id: string): void {
  if (current === id) {
    current = null
    lastEnded = { id, at: Date.now() }
  }
}

/** The id a write request is sent with: the running action's, else a fresh one for this request. */
export function actionIdForRequest(): string {
  return current ?? newActionId()
}

/** The server said this action can be undone (its answer carried X-Action-Id). */
export function recordAction(id: string, method: string, url: string): void {
  if (!ID_RE.test(id)) return
  const path = url.split('?')[0]
  if (path.startsWith('/api/actions/')) return
  if (entries.some((e) => e.id === id)) return // another request of the same action
  set([{ id, label: null, at: Date.now(), method, path, state: 'done', activeId: id }, ...entries])
  if (!current) lastEnded = { id, at: Date.now() } // a request outside a mutation is an action of its own
}

/** A success message is about to show: when it describes an action that was just recorded and has no label
 *  yet (the running one, or one that ended a moment ago), that action takes the message as its label and
 *  the entry id comes back, so the message can offer Undo. */
export function claimForToast(text: string): string | null {
  const unlabelled = (id: string | null | undefined) => {
    const e = id ? getEntry(id) : undefined
    return e && e.label === null && e.state === 'done' ? e : undefined
  }
  let e = unlabelled(current)
  if (!e && lastEnded && Date.now() - lastEnded.at <= CLAIM_MS) e = unlabelled(lastEnded.id)
  if (!e) return null
  update(e.id, { label: text.trim().slice(0, 200) || null })
  return e.id
}

export function markUndone(id: string, undoId: string, partial: boolean): void {
  update(id, { state: 'undone', undoId, undoneAt: Date.now(), partial })
}

export function markRedone(id: string, redoId: string, partial: boolean): void {
  update(id, { state: 'done', activeId: redoId, undoId: undefined, undoneAt: undefined, partial })
}

export function markExpired(id: string): void {
  update(id, { expired: true })
}

/** The server says it was undone already (another tab or device): no redo from here. */
export function markAlreadyUndone(id: string): void {
  update(id, { state: 'undone', undoId: undefined, undoneAt: Date.now() })
}

export function clearHistory(): void {
  set([])
}

/** The newest change that can be undone, for Ctrl+Z. */
export function newestUndoable(): UndoEntry | undefined {
  return entries.find((e) => e.state === 'done' && !e.expired)
}

/** The change undone last that can be redone, for Ctrl+Shift+Z. */
export function newestRedoable(): UndoEntry | undefined {
  let best: UndoEntry | undefined
  for (const e of entries) {
    if (e.state === 'undone' && e.undoId && !e.expired && (!best || (e.undoneAt ?? 0) > (best.undoneAt ?? 0))) best = e
  }
  return best
}

/** The history entries whose live action is one of these server action ids (newest first). */
export function entriesForActions(actionIds: string[]): UndoEntry[] {
  const want = new Set(actionIds)
  return entries.filter((e) => e.state === 'done' && want.has(e.activeId))
}

// ------------------------------------------------------------------ labels
const RULES: Array<[string, RegExp, string]> = [
  ['PATCH', /^\/api\/items\/[^/]+$/, 'Changed the details of an item'],
  ['DELETE', /^\/api\/items\/[^/]+$/, 'Moved an item to the trash'],
  ['POST', /^\/api\/items\/[^/]+\/restore$/, 'Restored an item from the trash'],
  ['POST', /^\/api\/items\/bulk$/, 'Changed several items'],
  ['POST', /^\/api\/items\/[^/]+\/tags$/, 'Tagged someone'],
  ['DELETE', /^\/api\/items\/[^/]+\/tags\/[^/]+$/, 'Removed someone from an item'],
  ['POST', /^\/api\/items\/[^/]+\/people-boxes$/, 'Added a person box'],
  ['POST', /^\/api\/items\/[^/]+\/faces$/, 'Added a face box'],
  ['PUT', /^\/api\/faces\/\d+\/box$/, 'Moved a face box'],
  ['PUT', /^\/api\/bodies\/\d+\/box$/, 'Moved a person box'],
  ['PUT', /^\/api\/animals\/\d+\/box$/, 'Moved a pet box'],
  ['POST', /^\/api\/items\/[^/]+\/animals$/, 'Added a pet box'],
  ['POST', /^\/api\/items\/[^/]+\/animals\/detect$/, 'Looked for pets again'],
  ['POST', /^\/api\/items\/[^/]+\/user-tags$/, 'Added a tag'],
  ['DELETE', /^\/api\/items\/[^/]+\/user-tags\/[^/]+$/, 'Removed a tag'],
  ['POST', /^\/api\/items\/people\/move$/, 'Changed who is in the selected items'],
  ['POST', /^\/api\/items\/people\/remove$/, 'Took someone off the selected items'],
  ['POST', /^\/api\/items\/people\/tag$/, 'Tagged someone in the selected items'],
  ['POST', /^\/api\/faces\/assign$/, 'Changed who a face is'],
  ['POST', /^\/api\/bodies\/assign$/, 'Changed who a person box is'],
  ['POST', /^\/api\/animals\/assign$/, 'Changed which pet a box is'],
  ['POST', /^\/api\/animals\/species$/, 'Changed a species'],
  ['DELETE', /^\/api\/faces\/[^/]+$/, 'Removed a face as not a face'],
  ['DELETE', /^\/api\/bodies\/[^/]+$/, 'Removed a person box'],
  ['DELETE', /^\/api\/animals\/[^/]+$/, 'Removed a pet box'],
  ['POST', /^\/api\/guesses\/confirm$/, 'Confirmed a guess'],
  ['POST', /^\/api\/guesses\/reject$/, 'Turned down a guess'],
  ['POST', /^\/api\/suggestions\/dismiss$/, 'Dismissed a suggestion'],
  ['POST', /^\/api\/people\/merge$/, 'Merged people'],
  ['POST', /^\/api\/people$/, 'Added a person'],
  ['PATCH', /^\/api\/people\/[^/]+$/, 'Changed a person'],
  ['DELETE', /^\/api\/people\/[^/]+$/, 'Deleted a person'],
  ['POST', /^\/api\/people\/[^/]+\/species$/, 'Changed a pet'],
  ['POST', /^\/api\/albums$/, 'Made an album'],
  ['PATCH', /^\/api\/albums\/[^/]+$/, 'Changed an album'],
  ['DELETE', /^\/api\/albums\/[^/]+$/, 'Deleted an album'],
  ['POST', /^\/api\/albums\/[^/]+\/items$/, 'Added to an album'],
  ['DELETE', /^\/api\/albums\/[^/]+\/items$/, 'Took items out of an album'],
  ['POST', /^\/api\/shares$/, 'Shared items'],
  ['PATCH', /^\/api\/shares\/[^/]+$/, 'Changed a share'],
  ['DELETE', /^\/api\/shares\/[^/]+$/, 'Stopped a share'],
  ['POST', /^\/api\/tags$/, 'Made a tag'],
  ['PATCH', /^\/api\/tags\/[^/]+$/, 'Changed a tag'],
  ['DELETE', /^\/api\/tags\/[^/]+$/, 'Deleted a tag'],
  ['PATCH', /^\/api\/documents\/[^/]+$/, 'Edited a document'],
  ['POST', /^\/api\/duplicates\/[^/]+\/best$/, 'Picked the best of duplicates'],
]

/** A plain description of a change from its request, when the app showed no message for it. */
export function describe(method: string, path: string): string {
  for (const [m, re, text] of RULES) if (m === method && re.test(path)) return text
  return 'Made a change'
}

/** What the change was, in a few words: the first sentence of the message the app showed for it, unless
 *  that is too short to say anything ("Removed."), else the description of its request. */
export function labelOf(e: UndoEntry): string {
  const first = (e.label ?? '').split(/(?<=[.!?])\s+/)[0]?.trim().replace(/[.\s]+$/, '') ?? ''
  return first.length >= 12 ? first : describe(e.method, e.path)
}

// ------------------------------------------------------------------ history dialog
let historyOpen = false
const openListeners = new Set<() => void>()

export function openUndoHistory(): void {
  historyOpen = true
  for (const fn of openListeners) fn()
}

export function closeUndoHistory(): void {
  historyOpen = false
  for (const fn of openListeners) fn()
}

function subscribeOpen(fn: () => void): () => void {
  openListeners.add(fn)
  return () => {
    openListeners.delete(fn)
  }
}

export function useUndoHistoryOpen(): boolean {
  return useSyncExternalStore(subscribeOpen, () => historyOpen, () => false)
}
