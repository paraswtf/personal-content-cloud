// Large drop target plus the file picker button for the upload page. Any file type is accepted:
// photos, videos, camera RAW, PDFs, office documents and anything else; the server sorts them out.
//
// iOS hands picked photos over only after the picker closes and, for originals kept in iCloud, after
// they are downloaded; that can take a while, and WebKit sometimes delivers the files late or with no
// change event at all. So after the picker opens the input is polled for files (PICK_WAIT_MS at most)
// and a notice says what is happening, instead of the page looking as if nothing was chosen.
import { useCallback, useEffect, useRef, useState, type DragEvent, type ReactNode } from 'react'
import clsx from 'clsx'
import { CloudUpload, FolderOpen } from 'lucide-react'
import { Button, Spinner } from '../ui'

const PICK_WAIT_MS = 90_000
const PICK_POLL_MS = 500
const PICK_NOTICE_AFTER_MS = 1_500

export default function DropZone({
  accept,
  maxLabel,
  onFiles,
  extra,
}: {
  /** accept attribute for the picker; '*' style values let every file through. */
  accept?: string
  maxLabel: string
  onFiles: (files: File[]) => void
  /** More actions under the main button (the camera button). */
  extra?: ReactNode
}) {
  const input = useRef<HTMLInputElement>(null)
  const depth = useRef(0)
  const [over, setOver] = useState(false)
  const [waiting, setWaiting] = useState(false)
  const poll = useRef<number | null>(null)
  const noticeTimer = useRef<number | null>(null)

  const stopWaiting = useCallback(() => {
    if (poll.current !== null) window.clearInterval(poll.current)
    if (noticeTimer.current !== null) window.clearTimeout(noticeTimer.current)
    poll.current = null
    noticeTimer.current = null
    setWaiting(false)
  }, [])
  useEffect(() => stopWaiting, [stopWaiting])

  // 'cancel' (the picker was dismissed) is not in React's input props: keep polling quietly for a
  // while, since WebKit can still hand files over after a dismissal that looked like a cancel, but
  // drop the notice
  useEffect(() => {
    const el = input.current
    if (!el) return
    const onCancel = () => {
      if (noticeTimer.current !== null) window.clearTimeout(noticeTimer.current)
      noticeTimer.current = null
      setWaiting(false)
    }
    el.addEventListener('cancel', onCancel)
    return () => el.removeEventListener('cancel', onCancel)
  }, [])

  /** Take whatever the input holds; the value is reset afterwards so the same files can be chosen again. */
  const takeFiles = useCallback(
    (el: HTMLInputElement) => {
      const files = Array.from(el.files ?? [])
      if (!files.length) return false
      stopWaiting()
      el.value = ''
      onFiles(files)
      return true
    },
    [onFiles, stopWaiting],
  )

  const openPicker = () => {
    const el = input.current
    if (!el) return
    stopWaiting()
    el.click()
    const started = Date.now()
    poll.current = window.setInterval(() => {
      const now = input.current
      if (!now || takeFiles(now) || Date.now() - started > PICK_WAIT_MS) stopWaiting()
    }, PICK_POLL_MS)
    noticeTimer.current = window.setTimeout(() => setWaiting(true), PICK_NOTICE_AFTER_MS)
  }

  const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes('Files')

  const onDragEnter = (e: DragEvent) => {
    if (!hasFiles(e)) return
    e.preventDefault()
    depth.current++
    setOver(true)
  }
  const onDragOver = (e: DragEvent) => {
    if (!hasFiles(e)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }
  const onDragLeave = (e: DragEvent) => {
    if (!hasFiles(e)) return
    depth.current = Math.max(0, depth.current - 1)
    if (depth.current === 0) setOver(false)
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    depth.current = 0
    setOver(false)
    stopWaiting()
    const files = Array.from(e.dataTransfer?.files ?? [])
    if (files.length) onFiles(files)
  }

  // A bare "*/*" accept makes some mobile pickers hide documents; no accept at all shows everything.
  const acceptAttr = accept && !/^\s*\*\/\*\s*$/.test(accept) ? accept : undefined

  return (
    <div
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className={clsx(
        'relative flex flex-col items-center gap-4 rounded-[var(--radius-card)] border-2 border-dashed px-4 py-10 text-center transition-colors sm:py-14',
        over ? 'border-accent bg-accent-soft/60' : 'border-line bg-surface',
      )}
    >
      <span
        className={clsx(
          'flex size-16 items-center justify-center rounded-full transition-colors',
          over ? 'bg-accent text-white' : 'bg-accent-soft text-accent',
        )}
      >
        <CloudUpload className="size-8" aria-hidden />
      </span>
      <div className="flex flex-col gap-1">
        <p className="text-lg font-semibold">{over ? 'Drop to upload' : 'Add photos, videos and documents'}</p>
        <p className="hidden text-sm text-ink-soft sm:block">Drag files here, or choose them from this device.</p>
        <p className="text-sm text-ink-soft sm:hidden">Pick as many files as you like.</p>
      </div>
      <div className="flex flex-col items-stretch gap-2 sm:flex-row sm:items-center">
        <Button size="lg" type="button" onClick={openPicker}>
          <FolderOpen className="size-5" aria-hidden />
          Choose files
        </Button>
        {extra}
      </div>
      {waiting && (
        <p role="status" aria-live="polite" className="flex max-w-md items-start gap-2 text-left text-sm text-ink-soft">
          <Spinner className="mt-0.5 size-4 shrink-0" />
          <span>
            Waiting for the files. On iPhone and iPad, photos kept in iCloud are downloaded before they are handed over,
            which can take a while; keep this screen open until they appear in the list.
          </span>
        </p>
      )}
      <p className="max-w-md text-xs text-ink-faint">
        Up to {maxLabel} per file. Photos (including HEIC and RAW), videos, PDFs, Word, Excel, PowerPoint and text files are
        indexed. Originals are kept exactly as uploaded.
      </p>
      <input
        ref={input}
        type="file"
        accept={acceptAttr}
        multiple
        tabIndex={-1}
        aria-hidden
        className="sr-only"
        onChange={(e) => takeFiles(e.currentTarget)}
      />
    </div>
  )
}
