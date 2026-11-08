// Undo and redo of history entries (lib/undo.ts) against the server (POST /api/actions/{id}/undo,
// docs/ARCHITECTURE.md section 33). Each run refreshes every query and reports through the caller's
// messages: "Undone: ..." with Redo, "Redone: ..." with Undo, and a plain sentence for each refusal.
import type { QueryClient } from '@tanstack/react-query'
import { api, ApiError } from './api'
import {
  entriesForActions,
  getEntry,
  labelOf,
  markAlreadyUndone,
  markExpired,
  markRedone,
  markUndone,
  type UndoEntry,
} from './undo'

export interface MessageAction {
  label: string
  onClick: () => void
}

export interface UndoUi {
  ok: (text: string, action?: MessageAction) => void
  error: (text: string, action?: MessageAction) => void
}

const busy = new Set<string>()

function sentence(text: string): string {
  const t = text.trim().replace(/[.\s]+$/, '')
  return t ? `${t}.` : t
}

function blockingIds(err: ApiError): string[] {
  const body = err.body as { blocking?: unknown } | null
  return Array.isArray(body?.blocking) ? body.blocking.filter((x): x is string => typeof x === 'string') : []
}

/** Undo one entry without messages; throws what the server answered. */
async function undoOnce(e: UndoEntry): Promise<boolean> {
  const out = await api.actions.undo(e.activeId)
  markUndone(e.id, out.undo_id, out.partial)
  return out.partial
}

function explain(err: unknown, entryId: string, redo: boolean, ui: UndoUi, qc: QueryClient): void {
  if (err instanceof ApiError) {
    if (err.status === 409 && err.code === 'later_changes') {
      const ids = blockingIds(err)
      const later = entriesForActions(ids)
      if (!redo && later.length > 0 && later.length === ids.length) {
        ui.error(
          later.length === 1
            ? 'A later change touches the same things. Undo that one first.'
            : `${later.length} later changes touch the same things. Undo those first.`,
          { label: later.length === 1 ? 'Undo both' : `Undo all ${later.length + 1}`, onClick: () => void undoChain([...later.map((l) => l.id), entryId], ui, qc) },
        )
      } else {
        ui.error('Changes made since touch the same things, some of them elsewhere. Undo those first.')
      }
      return
    }
    if (err.status === 409 && err.code === 'already_undone') {
      if (redo) markExpired(entryId)
      else markAlreadyUndone(entryId)
      ui.error(redo ? 'That was redone already, maybe in another tab.' : 'That was undone already, maybe in another tab.')
      return
    }
    if (err.status === 410 || err.status === 404) {
      markExpired(entryId)
      ui.error('That change is too old to undo. Changes can be undone for 30 days.')
      return
    }
    if (err.status === 403) {
      ui.error('Unlock the vault to undo this.')
      return
    }
    if (err.status === 401) {
      ui.error('Sign in again to undo this.')
      return
    }
    if (err.status === 409) {
      ui.error('It cannot be undone now: something it would bring back is in the way, for example a name used since.')
      return
    }
    if (err.status !== 0) {
      ui.error(sentence(err.message) || 'Could not undo it. Try again.')
      return
    }
  }
  ui.error(redo ? 'Could not redo it. Check the connection and try again.' : 'Could not undo it. Check the connection and try again.')
}

export async function runUndo(entryId: string, ui: UndoUi, qc: QueryClient): Promise<boolean> {
  const e = getEntry(entryId)
  if (!e || e.state !== 'done' || e.expired || busy.has(entryId)) return false
  busy.add(entryId)
  try {
    const partial = await undoOnce(e)
    void qc.invalidateQueries()
    ui.ok(`Undone: ${sentence(labelOf(e))}${partial ? ' Some of it had changed again since and stays as it is.' : ''}`, {
      label: 'Redo',
      onClick: () => void runRedo(entryId, ui, qc),
    })
    return true
  } catch (err) {
    explain(err, entryId, false, ui, qc)
    return false
  } finally {
    busy.delete(entryId)
  }
}

export async function runRedo(entryId: string, ui: UndoUi, qc: QueryClient): Promise<boolean> {
  const e = getEntry(entryId)
  if (!e || e.state !== 'undone' || !e.undoId || e.expired || busy.has(entryId)) return false
  busy.add(entryId)
  try {
    const out = await api.actions.undo(e.undoId)
    markRedone(entryId, out.undo_id, out.partial)
    void qc.invalidateQueries()
    ui.ok(`Redone: ${sentence(labelOf(e))}${out.partial ? ' Some of it had changed since and stays as it is.' : ''}`, {
      label: 'Undo',
      onClick: () => void runUndo(entryId, ui, qc),
    })
    return true
  } catch (err) {
    explain(err, entryId, true, ui, qc)
    return false
  } finally {
    busy.delete(entryId)
  }
}

/** Undo several entries in order (newest first), then say once how it went. */
export async function undoChain(entryIds: string[], ui: UndoUi, qc: QueryClient): Promise<void> {
  let done = 0
  for (const id of entryIds) {
    const e = getEntry(id)
    if (!e || e.state !== 'done' || e.expired) continue
    try {
      await undoOnce(e)
      done += 1
    } catch (err) {
      void qc.invalidateQueries()
      explain(err, id, false, ui, qc)
      return
    }
  }
  void qc.invalidateQueries()
  ui.ok(done === 1 ? 'Undone 1 change.' : `Undone ${done} changes.`)
}
