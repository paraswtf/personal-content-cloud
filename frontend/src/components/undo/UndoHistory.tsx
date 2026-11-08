// The undo history dialog (docs/ARCHITECTURE.md sections 14 and 33): every change this browser recorded,
// newest first (lib/undo.ts), with Undo or Redo on each, and "Clear history" (this browser only). Opened
// from the account menu (desktop) and the More sheet (phones) through openUndoHistory(); UndoHistoryHost
// renders it once in the app shell.
import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { Redo2, Undo2 } from 'lucide-react'
import { formatRelative } from '../../lib/format'
import { clearHistory, closeUndoHistory, labelOf, useUndoHistory, useUndoHistoryOpen, type UndoEntry } from '../../lib/undo'
import { useToast } from '../Toast'
import { Button, Modal } from '../ui'

function status(e: UndoEntry, now: number): string {
  const parts = [formatRelative(new Date(e.at).toISOString(), now)]
  if (e.expired) parts.push('too old to undo')
  else if (e.state === 'undone') parts.push(e.undoId ? 'undone' : 'undone elsewhere')
  if (e.partial) parts.push('partly, as some of it had changed since')
  return parts.join(', ')
}

export function UndoHistory({ open, onClose }: { open: boolean; onClose: () => void }) {
  const entries = useUndoHistory()
  const toast = useToast()
  const [busy, setBusy] = useState<string | null>(null)
  const [clearing, setClearing] = useState(false)
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!open) {
      setClearing(false)
      return
    }
    setNow(Date.now())
    const t = window.setInterval(() => setNow(Date.now()), 30_000)
    return () => window.clearInterval(t)
  }, [open])

  const run = async (e: UndoEntry, redo: boolean) => {
    if (busy) return
    setBusy(e.id)
    try {
      await (redo ? toast.redo(e.id) : toast.undo(e.id))
    } finally {
      setBusy(null)
    }
  }

  const footer = entries.length ? (
    clearing ? (
      <>
        <span className="mr-auto self-center text-sm text-ink-soft">Forget all {entries.length} in this browser?</span>
        <Button variant="ghost" size="sm" onClick={() => setClearing(false)}>
          Keep
        </Button>
        <Button
          variant="danger"
          size="sm"
          onClick={() => {
            clearHistory()
            setClearing(false)
          }}
        >
          Clear
        </Button>
      </>
    ) : (
      <Button variant="ghost" size="sm" onClick={() => setClearing(true)}>
        Clear history
      </Button>
    )
  ) : undefined

  return (
    <Modal open={open} onClose={onClose} title="Undo history" footer={footer}>
      {entries.length === 0 ? (
        <p className="text-sm text-ink-soft">
          Nothing to undo yet. The changes you make in this browser show up here, newest first, and can be undone for 30 days.
        </p>
      ) : (
        <>
          <p className="mb-2 text-sm text-ink-soft">
            Changes made in this browser, newest first.
            <span className="hidden sm:inline"> Ctrl+Z (Cmd+Z on a Mac) undoes the newest, Ctrl+Shift+Z redoes.</span>
          </p>
          <ul className="flex flex-col divide-y divide-line" aria-label="Changes">
            {entries.map((e) => {
              const label = labelOf(e)
              const canUndo = e.state === 'done' && !e.expired
              const canRedo = e.state === 'undone' && !!e.undoId && !e.expired
              return (
                <li key={e.id} className="flex items-center gap-3 py-2.5">
                  <div className="min-w-0 flex-1">
                    <p className={clsx('text-sm font-medium break-words', e.state === 'undone' ? 'text-ink-soft line-through' : 'text-ink')}>
                      {label}
                    </p>
                    <p className="text-xs text-ink-faint">{status(e, now)}</p>
                  </div>
                  {(canUndo || canRedo) && (
                    <Button
                      variant="secondary"
                      size="sm"
                      className="shrink-0"
                      loading={busy === e.id}
                      disabled={busy !== null && busy !== e.id}
                      aria-label={`${canUndo ? 'Undo' : 'Redo'}: ${label}`}
                      onClick={() => void run(e, canRedo)}
                    >
                      {busy !== e.id && (canUndo ? <Undo2 className="size-4" aria-hidden /> : <Redo2 className="size-4" aria-hidden />)}
                      {canUndo ? 'Undo' : 'Redo'}
                    </Button>
                  )}
                </li>
              )
            })}
          </ul>
        </>
      )}
    </Modal>
  )
}

export default function UndoHistoryHost() {
  const open = useUndoHistoryOpen()
  return <UndoHistory open={open} onClose={closeUndoHistory} />
}
