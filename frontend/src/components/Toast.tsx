// App-wide toasts (moved from the hub admin). ToastProvider wraps the app shell; useToast() gives
// success, error and info. At most four toasts show at once; they last 5 s (errors 9 s, 8 s and 12 s with a
// button). Undo (docs/ARCHITECTURE.md sections 14 and 33): a success message that describes an action just
// recorded in the undo history (lib/undo.ts claimForToast) labels that action and offers Undo; the result
// offers Redo. undo(entryId) and redo(entryId) run the same for the viewer's own messages and the history
// dialog. Keyboard: Ctrl+Z or Cmd+Z undoes the newest change, Ctrl+Shift+Z, Cmd+Shift+Z or Ctrl+Y redoes the
// change undone last; never while typing, nor while a dialog other than the viewer has the focus or is open.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { CheckCircle2, Info, TriangleAlert, X } from 'lucide-react'
import { claimForToast, newestRedoable, newestUndoable } from '../lib/undo'
import { runRedo, runUndo, type MessageAction, type UndoUi } from '../lib/undoRun'

type Tone = 'ok' | 'error' | 'info'
interface ToastItem {
  id: number
  tone: Tone
  text: string
  action?: MessageAction
}
interface ToastApi {
  success: (text: string) => void
  error: (text: string) => void
  info: (text: string) => void
  /** Undo or redo a history entry (lib/undo.ts), with its own messages; true when it went through. */
  undo: (entryId: string) => Promise<boolean>
  redo: (entryId: string) => Promise<boolean>
}

const ToastContext = createContext<ToastApi | null>(null)

/** Ctrl+Z stays with the page while someone types, or while a dialog other than the viewer is in use. */
function keysBlocked(target: EventTarget | null): boolean {
  const el = target instanceof HTMLElement ? target : null
  if (el && (el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName))) return true
  const dialog = el?.closest('[role="dialog"]')
  if (dialog && !dialog.hasAttribute('data-undo-keys')) return true
  for (const d of Array.from(document.querySelectorAll('[role="dialog"][aria-modal="true"]'))) {
    if (!d.hasAttribute('data-undo-keys')) return true
  }
  return false
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient()
  const [items, setItems] = useState<ToastItem[]>([])
  const nextId = useRef(1)

  const dismiss = useCallback((id: number) => setItems((cur) => cur.filter((t) => t.id !== id)), [])
  const push = useCallback(
    (tone: Tone, text: string, action?: MessageAction) => {
      const id = nextId.current++
      setItems((cur) => [...cur.slice(-3), { id, tone, text, action }])
      const ms = tone === 'error' ? (action ? 12000 : 9000) : action ? 8000 : 5000
      window.setTimeout(() => dismiss(id), ms)
    },
    [dismiss],
  )
  const ui = useMemo<UndoUi>(
    () => ({ ok: (text, action) => push('ok', text, action), error: (text, action) => push('error', text, action) }),
    [push],
  )
  const undo = useCallback((entryId: string) => runUndo(entryId, ui, qc), [ui, qc])
  const redo = useCallback((entryId: string) => runRedo(entryId, ui, qc), [ui, qc])
  const api = useMemo<ToastApi>(
    () => ({
      success: (t) => {
        const entry = claimForToast(t)
        push('ok', t, entry ? { label: 'Undo', onClick: () => void undo(entry) } : undefined)
      },
      error: (t) => push('error', t),
      info: (t) => push('info', t),
      undo,
      redo,
    }),
    [push, undo, redo],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.repeat || e.altKey || !(e.ctrlKey || e.metaKey)) return
      const key = e.key.toLowerCase()
      const isUndo = key === 'z' && !e.shiftKey
      const isRedo = (key === 'z' && e.shiftKey) || (key === 'y' && e.ctrlKey && !e.metaKey && !e.shiftKey)
      if (!isUndo && !isRedo) return
      if (keysBlocked(e.target) || keysBlocked(document.activeElement)) return
      e.preventDefault()
      const entry = isUndo ? newestUndoable() : newestRedoable()
      if (!entry) push('info', isUndo ? 'Nothing to undo.' : 'Nothing to redo.')
      else if (isUndo) void undo(entry.id)
      else void redo(entry.id)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [push, undo, redo])

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        aria-live="polite"
        className="pointer-events-none fixed inset-x-4 top-4 z-[60] flex flex-col items-stretch gap-2 sm:left-auto sm:right-4 sm:w-96"
      >
        {items.map((t) => (
          <div
            key={t.id}
            role={t.tone === 'error' ? 'alert' : 'status'}
            className={clsx(
              'pointer-events-auto flex items-start gap-3 rounded-2xl border bg-surface px-4 py-3 text-sm shadow-lg',
              t.tone === 'error' ? 'border-danger/40' : 'border-line',
            )}
          >
            {t.tone === 'ok' && <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-ok" aria-hidden />}
            {t.tone === 'error' && <TriangleAlert className="mt-0.5 size-4 shrink-0 text-danger" aria-hidden />}
            {t.tone === 'info' && <Info className="mt-0.5 size-4 shrink-0 text-ink-soft" aria-hidden />}
            <p className="min-w-0 flex-1 break-words">{t.text}</p>
            {t.action && (
              <button
                type="button"
                onClick={() => {
                  dismiss(t.id)
                  t.action?.onClick()
                }}
                className="-my-1 shrink-0 rounded-full px-2.5 py-1 font-semibold text-accent hover:bg-accent/10 focus-visible:outline-2 focus-visible:outline-accent"
              >
                {t.action.label}
              </button>
            )}
            <button
              onClick={() => dismiss(t.id)}
              className="-m-1 rounded-full p-1 text-ink-faint hover:bg-ink/5 hover:text-ink"
              aria-label="Dismiss message"
            >
              <X className="size-4" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used inside ToastProvider')
  return ctx
}
