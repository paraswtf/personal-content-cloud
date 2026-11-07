// Large drop target plus the file picker button for the upload page. Any file type is accepted:
// photos, videos, camera RAW, PDFs, office documents and anything else; the server sorts them out.
import { useRef, useState, type DragEvent, type ReactNode } from 'react'
import clsx from 'clsx'
import { CloudUpload, FolderOpen } from 'lucide-react'
import { Button } from '../ui'

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
        <Button size="lg" type="button" onClick={() => input.current?.click()}>
          <FolderOpen className="size-5" aria-hidden />
          Choose files
        </Button>
        {extra}
      </div>
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
        onChange={(e) => {
          const files = Array.from(e.target.files ?? [])
          // reset so choosing the same files again still fires a change event
          e.target.value = ''
          if (files.length) onFiles(files)
        }}
      />
    </div>
  )
}
