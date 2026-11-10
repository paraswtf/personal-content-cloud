// A larger look at one item in the trash (Space on a tile): the preview image, a video with its controls, or a
// document's first page, with Restore and Delete forever. Keys while it is open: Left and Right for the previous and
// next item, Enter (on the dialog itself; a focused button keeps its own click) or R to restore, Delete or Backspace
// to delete forever (asks first), Space or Esc to close. Nothing fires while the delete dialog is open over it, in a
// video's controls, or with Ctrl, Cmd or Alt held. Private items show the same locked tile as the grid.
import { useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, EyeOff, File as FileIcon, FileText, Film, RotateCcw, Trash2 } from 'lucide-react'
import type { ItemCard } from '../../lib/types'
import { formatDuration } from '../../lib/format'
import { Button, Kbd, Modal } from '../ui'

export default function TrashPreview({
  item,
  index,
  total,
  busy,
  keysOff,
  onPrev,
  onNext,
  onRestore,
  onPurge,
  onClose,
  prefetch,
}: {
  item: ItemCard
  index: number
  total: number
  /** Being restored or deleted, or already gone and waiting to leave the list. */
  busy: boolean
  /** Another dialog (the delete confirm) is over the preview. */
  keysOff: boolean
  onPrev?: () => void
  onNext?: () => void
  onRestore: () => void
  /** Ask before deleting for good; `byKey` focuses the dialog's confirm button. */
  onPurge: (byKey: boolean) => void
  /** bySpace: closed with Space, whose keyup must not click what takes focus next. */
  onClose: (bySpace?: boolean) => void
  /** The next item's preview URL, loaded ahead. */
  prefetch?: string | null
}) {
  const [broken, setBroken] = useState<string | null>(null)
  const isDoc = item.kind === 'pdf' || item.kind === 'document'
  const Icon = isDoc ? FileText : item.kind === 'video' ? Film : FileIcon

  useEffect(() => {
    if (keysOff) return
    const space = { held: false }
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return
      const t = e.target
      if (t instanceof HTMLVideoElement || t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement) return
      const onDialog = t instanceof HTMLElement && t.getAttribute('role') === 'dialog'
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault()
        if (e.key === 'ArrowLeft') onPrev?.()
        else onNext?.()
      } else if (e.key === ' ') {
        e.preventDefault() // not a focused button's click: Space closes the preview
        space.held = true
        if (!e.repeat) onClose(true)
      } else if ((e.key === 'Enter' && onDialog) || e.key === 'r' || e.key === 'R') {
        e.preventDefault()
        if (!e.repeat && !busy) onRestore()
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault()
        if (!e.repeat && !busy) onPurge(true)
      }
    }
    // Firefox turns the keyup of Space on a focused button into a click
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === ' ' && space.held) {
        space.held = false
        e.preventDefault()
      }
    }
    document.addEventListener('keydown', onKey)
    document.addEventListener('keyup', onKeyUp)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.removeEventListener('keyup', onKeyUp)
    }
  }, [keysOff, busy, onPrev, onNext, onRestore, onPurge, onClose])

  // the next item's picture is fetched ahead, so moving on shows it at once
  useEffect(() => {
    if (prefetch) new Image().src = prefetch
  }, [prefetch])

  // A focused button that turns disabled (Restore while it works) drops focus to the page: the dialog takes it back,
  // so its keys keep working and Enter restores only when it is on the dialog itself.
  const content = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (keysOff) return
    const a = document.activeElement
    if (!a || a === document.body || (a instanceof HTMLButtonElement && a.disabled))
      content.current?.closest<HTMLElement>('[role="dialog"]')?.focus()
  }, [keysOff, busy, item.uid])

  const shown = broken !== item.uid
  return (
    <Modal open onClose={() => onClose()} title={item.filename} wide>
      <div ref={content} className="flex flex-col gap-3">
        <div className="relative flex h-[60dvh] items-center justify-center overflow-hidden rounded-xl bg-surface-2">
          {item.nsfw ? (
            <div className="flex flex-col items-center gap-1 text-ink-faint">
              <EyeOff className="size-8" aria-hidden />
              <span className="text-sm font-medium">Private</span>
            </div>
          ) : !shown ? (
            <Icon className="size-10 text-ink-faint" aria-hidden />
          ) : item.kind === 'video' && item.urls.video ? (
            <video
              key={item.uid}
              src={item.urls.video}
              poster={item.urls.preview}
              controls
              playsInline
              className="max-h-full max-w-full"
              onError={() => setBroken(item.uid)}
            />
          ) : (
            <img
              key={item.uid}
              src={item.urls.preview}
              alt={item.filename}
              className={isDoc ? 'max-h-full max-w-full bg-white object-contain' : 'max-h-full max-w-full object-contain'}
              onError={() => setBroken(item.uid)}
            />
          )}
        </div>
        <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-ink-soft">
          <span className="tabular-nums">
            {index + 1} of {total.toLocaleString()}
          </span>
          {item.kind === 'video' && item.duration_s ? <span>Video, {formatDuration(item.duration_s)}</span> : null}
        </p>
        <p className="hidden text-xs text-ink-soft pointer-fine:block">
          Keys: <Kbd>←</Kbd> <Kbd>→</Kbd> previous and next, <Kbd>Enter</Kbd> or <Kbd>R</Kbd> restores, <Kbd>Delete</Kbd> deletes
          forever (asks first), <Kbd>Space</Kbd> or <Kbd>Esc</Kbd> closes.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          {/* aria-disabled, not disabled: a focused button that turned disabled would drop focus, and the next Enter
              meant for it would restore */}
          <Button variant="secondary" size="md" onClick={() => onPrev?.()} aria-disabled={!onPrev} className={onPrev ? '' : 'opacity-40'} aria-label="Previous item">
            <ChevronLeft className="size-4" aria-hidden />
          </Button>
          <Button variant="secondary" size="md" onClick={() => onNext?.()} aria-disabled={!onNext} className={onNext ? '' : 'opacity-40'} aria-label="Next item">
            <ChevronRight className="size-4" aria-hidden />
          </Button>
          <div className="ml-auto flex gap-2">
            <Button variant="danger" size="md" onClick={() => onPurge(false)} disabled={busy} aria-keyshortcuts="Delete">
              <Trash2 className="size-4" aria-hidden />
              Delete forever
            </Button>
            <Button size="md" onClick={onRestore} loading={busy} aria-keyshortcuts="R">
              {!busy && <RotateCcw className="size-4" aria-hidden />}
              Restore
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  )
}
