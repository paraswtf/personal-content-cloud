// One file in the upload queue: thumbnail (or a kind icon: photo, video, PDF or document), name, size,
// progress, outcome and actions. Finished files link to the viewer at /m/<uid>.
import { memo, useEffect, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router'
import clsx from 'clsx'
import { Camera, ExternalLink, File as FileIcon, FileText, Film, ImageIcon, RotateCcw, Share2, Trash2, WifiOff, X } from 'lucide-react'
import type { FileKind, QueueItem } from '../../lib/uploader'
import { formatBytes } from '../../lib/format'
import { CancelledBadge, FailedBadge, StatusBadge } from './status'

// Larger local images are not previewed: decoding many big originals at once costs too much memory.
const PREVIEW_MAX_BYTES = 25 * 1024 * 1024

const KIND_ICON: Record<FileKind, typeof FileIcon> = {
  image: ImageIcon,
  video: Film,
  pdf: FileText,
  document: FileText,
  other: FileIcon,
}

/** Thumbnail: the server thumb once known, else a local object URL made only while on screen. */
function Thumb({ item }: { item: QueueItem }) {
  const box = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  const [localUrl, setLocalUrl] = useState<string | null>(null)
  // the local preview fails for formats this browser cannot decode (HEIC outside Safari);
  // that must not hide the server thumbnail that arrives later
  const [localFailed, setLocalFailed] = useState(false)
  const [brokenSrc, setBrokenSrc] = useState<string | null>(null)
  const wantLocal = item.kind === 'image' && !item.thumbUrl && item.size <= PREVIEW_MAX_BYTES && !localFailed

  useEffect(() => {
    const el = box.current
    if (!el || visible) return
    if (typeof IntersectionObserver === 'undefined') {
      setVisible(true)
      return
    }
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) setVisible(true)
    }, { rootMargin: '200px 0px' })
    io.observe(el)
    return () => io.disconnect()
  }, [visible])

  // created and revoked in the same effect so StrictMode remounts and removal never leak or break it
  useEffect(() => {
    if (!wantLocal || !visible) return
    const url = URL.createObjectURL(item.file)
    setLocalUrl(url)
    return () => {
      URL.revokeObjectURL(url)
      setLocalUrl(null)
    }
  }, [wantLocal, visible, item.file])

  const candidate = item.thumbUrl ?? (wantLocal ? localUrl : null)
  const src = candidate && candidate !== brokenSrc ? candidate : null
  const Icon = KIND_ICON[item.kind]
  const isDoc = item.kind === 'pdf' || item.kind === 'document'
  return (
    <div
      ref={box}
      className="relative flex size-14 shrink-0 items-center justify-center overflow-hidden rounded-xl bg-surface-2 text-ink-faint ring-1 ring-line sm:size-16"
    >
      {src ? (
        <img
          src={src}
          alt=""
          decoding="async"
          className={clsx('size-full', isDoc ? 'bg-white object-contain' : 'object-cover')}
          onError={() => {
            setBrokenSrc(src)
            if (src !== item.thumbUrl) setLocalFailed(true)
          }}
        />
      ) : (
        <span className="flex flex-col items-center gap-0.5">
          <Icon className="size-6" aria-hidden />
          {isDoc && <span className="text-[9px] font-bold uppercase tracking-wide">{extLabel(item.name)}</span>}
        </span>
      )}
      {(item.kind === 'video' || isDoc) && src && (
        <span className="absolute bottom-1 right-1 rounded bg-black/65 p-0.5 text-white">
          {item.kind === 'video' ? <Film className="size-3" aria-hidden /> : <FileText className="size-3" aria-hidden />}
        </span>
      )}
    </div>
  )
}

function extLabel(name: string): string {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i + 1, i + 5) : 'file'
}

function useSecondsUntil(at: number | null): number | null {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!at) return
    setNow(Date.now())
    const t = setInterval(() => setNow(Date.now()), 500)
    return () => clearInterval(t)
  }, [at])
  return at ? Math.max(0, Math.ceil((at - now) / 1000)) : null
}

function ProgressBar({ value, label, tone = 'accent' }: { value: number | null; label: string; tone?: 'accent' | 'soft' }) {
  const pct = value === null ? null : Math.round(value * 100)
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct ?? undefined}
      className="relative h-1.5 w-full overflow-hidden rounded-full bg-ink/8"
    >
      {pct === null ? (
        <div className="h-full w-full animate-pulse rounded-full bg-ink-faint/50" />
      ) : (
        <div
          className={clsx('h-full rounded-full transition-[width] duration-200', tone === 'accent' ? 'bg-accent' : 'bg-ink-faint')}
          style={{ width: `${pct}%` }}
        />
      )}
    </div>
  )
}

function IconButton({
  label,
  onClick,
  children,
}: {
  label: string
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className="inline-flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft transition-colors hover:bg-ink/5 hover:text-ink"
    >
      {children}
    </button>
  )
}

function QueueRow({
  item,
  canCancel,
  canRetry,
  onCancel,
  onRetry,
  onRemove,
}: {
  item: QueueItem
  canCancel: boolean
  canRetry: boolean
  onCancel: (key: string) => void
  onRetry: (key: string) => void
  onRemove: (key: string) => void
}) {
  const retryIn = useSecondsUntil(item.retryAt)
  const pct = item.progress === null ? null : Math.round(item.progress * 100)

  let phaseText: string | null = null
  let bar: { value: number | null; tone: 'accent' | 'soft' } | null = null
  switch (item.phase) {
    case 'waiting':
      phaseText = 'Waiting'
      break
    case 'checking':
      phaseText = item.progress !== null ? `Checking ${pct}%` : item.resumed ? 'Resuming' : 'Checking'
      bar = { value: item.progress, tone: 'soft' }
      break
    case 'uploading':
      phaseText = `${item.resumed ? 'Resuming' : 'Uploading'} ${pct ?? 0}%`
      bar = { value: item.progress ?? 0, tone: 'accent' }
      break
    case 'finishing':
    case 'processing':
      phaseText = item.phase === 'finishing' ? 'Finishing' : 'Processing'
      bar = { value: null, tone: 'soft' }
      break
  }

  const waitingNetwork = item.offline || retryIn !== null
  const finished = item.phase === 'done' || item.phase === 'failed' || item.phase === 'cancelled'

  return (
    <li className="flex gap-3 px-3 py-3 sm:px-4">
      <Thumb item={item} />
      <div className="flex min-w-0 flex-1 flex-col gap-1.5">
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1 pt-0.5">
            <p className="truncate text-sm font-medium" title={item.name}>
              {item.name}
            </p>
            <p className="flex flex-wrap items-center gap-x-1 text-xs text-ink-soft">
              {item.source === 'camera' && (
                <span className="inline-flex items-center gap-1">
                  <Camera className="size-3" aria-hidden />
                  Camera<span aria-hidden> · </span>
                </span>
              )}
              {item.source === 'share' && (
                <span className="inline-flex items-center gap-1">
                  <Share2 className="size-3" aria-hidden />
                  Shared<span aria-hidden> · </span>
                </span>
              )}
              {formatBytes(item.size)}
              {phaseText && <span aria-hidden> · </span>}
              {phaseText && <span>{phaseText}</span>}
            </p>
          </div>
          <div className="-mr-2 -mt-1.5 flex shrink-0 items-center">
            {item.phase === 'done' && item.itemUid && (
              <Link
                to={`/m/${item.itemUid}`}
                className="inline-flex h-10 items-center gap-1 rounded-full px-3 text-sm font-semibold text-accent hover:bg-accent-soft"
              >
                View
                <ExternalLink className="size-3.5" aria-hidden />
                <span className="sr-only"> {item.name}</span>
              </Link>
            )}
            {canRetry && (
              <IconButton label={`Try ${item.name} again`} onClick={() => onRetry(item.key)}>
                <RotateCcw className="size-4.5" aria-hidden />
              </IconButton>
            )}
            {canCancel && (
              <IconButton label={`Cancel ${item.name}`} onClick={() => onCancel(item.key)}>
                <X className="size-5" aria-hidden />
              </IconButton>
            )}
            {finished && (
              <IconButton label={`Remove ${item.name} from the list`} onClick={() => onRemove(item.key)}>
                <Trash2 className="size-4.5" aria-hidden />
              </IconButton>
            )}
          </div>
        </div>

        {bar && <ProgressBar value={bar.value} tone={bar.tone} label={`${item.name}: ${phaseText ?? ''}`} />}

        {waitingNetwork && !finished && (
          <p className="flex items-center gap-1.5 text-xs text-warn">
            <WifiOff className="size-3.5 shrink-0" aria-hidden />
            {item.offline
              ? 'You are offline. The upload continues when the connection is back.'
              : `Connection problem. Trying again in ${retryIn} s.`}
          </p>
        )}

        {finished && (
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            {item.phase === 'done' && item.status && <StatusBadge status={item.status} />}
            {item.phase === 'failed' && <FailedBadge label={item.localReject ? 'Skipped' : 'Not uploaded'} />}
            {item.phase === 'cancelled' && <CancelledBadge />}
            {item.message && item.phase !== 'cancelled' && (
              <span className="min-w-0 text-xs text-ink-soft">{item.message}</span>
            )}
          </div>
        )}
      </div>
    </li>
  )
}

export default memo(QueueRow)
