// /s/:token (docs/ARCHITECTURE.md section 26): a shared link, outside the shell and without a
// session. The title and count, "Shared from <site title>", a password form when the link has one, a
// responsive grid, and a lightbox with prev and next (arrow keys, Escape), a video element and a
// Download button when the link allows it. Expired and missing links get a plain message.
import { useCallback, useEffect, useId, useState, type FormEvent, useRef } from 'react'
import { useParams } from 'react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ChevronLeft, ChevronRight, Cloud, Download, Lock, Play, X } from 'lucide-react'
import { ApiError } from '../lib/api'
import { useAuth } from '../lib/auth'
import { formatDate, formatDuration } from '../lib/format'
import { SHARE_PUBLIC_KEY, shareErrorText, sharePublic, shareUnlock, type ShareItem, type SharePublic } from '../lib/shares'
import { inputClass } from '../components/admin/fields'
import { usePageTitle } from '../components/shell/title'
import { Button, PageSpinner, lockScroll, tabbables } from '../components/ui'
import { iconButton } from '../components/viewer/shared'

export default function SharePage() {
  const { token = '' } = useParams()
  const qc = useQueryClient()
  const auth = useAuth()
  const siteTitle = auth.state?.site_title || 'Personal Content Cloud'
  const key = [SHARE_PUBLIC_KEY, token]
  const q = useQuery({
    queryKey: key,
    queryFn: () => sharePublic(token),
    enabled: token !== '',
    retry: false,
    staleTime: 60_000,
  })
  const [open, setOpen] = useState<number | null>(null)
  usePageTitle(q.data?.title || 'Shared photos')

  const code = q.error instanceof ApiError ? q.error.code : null

  if (!token) return <Message title="This link is not complete." siteTitle={siteTitle} />
  if (q.isPending) return <PageSpinner />
  if (q.isError && code === 'share_password_required')
    return <PasswordGate token={token} siteTitle={siteTitle} onUnlocked={(data) => qc.setQueryData(key, data)} />
  if (q.isError) {
    const title =
      code === 'share_expired'
        ? 'This link has expired.'
        : code === 'share_not_found'
          ? 'This link does not exist or was taken down.'
          : shareErrorText(q.error)
    return <Message title={title} siteTitle={siteTitle} onRetry={code ? undefined : () => void q.refetch()} />
  }
  const data = q.data
  const items = data.items
  return (
    <main className="min-h-dvh bg-bg text-ink">
      <header className="mx-auto flex w-full max-w-7xl flex-col gap-1 px-4 pb-4 pt-6 sm:pt-8">
        <p className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-ink-faint">
          <Cloud className="size-4" aria-hidden />
          Shared from {siteTitle}
        </p>
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">{data.title || 'Shared photos'}</h1>
        <p className="text-sm text-ink-soft">
          {data.count} {data.count === 1 ? 'item' : 'items'}
          {data.expires_at ? `. Available until ${formatDate(data.expires_at)}` : ''}
          {data.allow_download ? '. Originals can be downloaded.' : '.'}
        </p>
      </header>
      {items.length === 0 ? (
        <p className="mx-auto w-full max-w-7xl px-4 py-12 text-center text-sm text-ink-soft">Nothing to show here yet.</p>
      ) : (
        <ul className="mx-auto grid w-full max-w-7xl grid-cols-3 gap-0.5 px-0 sm:grid-cols-4 sm:gap-1 sm:px-4 md:grid-cols-5 lg:grid-cols-6" role="list">
          {items.map((it, i) => (
            <li key={it.uid} className="min-w-0">
              <button
                type="button"
                onClick={() => setOpen(i)}
                className="group relative block aspect-square w-full overflow-hidden bg-surface-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                aria-label={`Open ${it.filename}`}
              >
                <img src={it.urls.thumb} alt="" loading="lazy" decoding="async" className="size-full object-cover" />
                {it.kind === 'video' && (
                  <span className="absolute bottom-1.5 right-1.5 flex items-center gap-1 rounded-full bg-black/60 px-1.5 py-0.5 text-[11px] font-medium text-white">
                    <Play className="size-3 fill-current" aria-hidden />
                    {formatDuration(it.duration_s)}
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
      <footer className="mx-auto w-full max-w-7xl px-4 py-8 text-center text-xs text-ink-faint">Shared from {siteTitle}.</footer>
      {open !== null && items[open] && (
        <Lightbox
          items={items}
          index={open}
          allowDownload={data.allow_download}
          onClose={() => setOpen(null)}
          onMove={(i) => setOpen(i)}
        />
      )}
    </main>
  )
}

function Message({ title, siteTitle, onRetry }: { title: string; siteTitle: string; onRetry?: () => void }) {
  return (
    <main className="flex min-h-dvh flex-col items-center justify-center gap-3 bg-bg px-4 text-center text-ink">
      <span className="flex size-12 items-center justify-center rounded-xl bg-ink text-bg">
        <Cloud className="size-6" aria-hidden />
      </span>
      <h1 className="text-lg font-semibold">{title}</h1>
      <p className="text-sm text-ink-soft">Shared from {siteTitle}.</p>
      {onRetry && (
        <Button variant="secondary" onClick={onRetry}>
          Try again
        </Button>
      )}
    </main>
  )
}

function PasswordGate({ token, siteTitle, onUnlocked }: { token: string; siteTitle: string; onUnlocked: (data: SharePublic) => void }) {
  const id = useId()
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const unlock = useMutation({
    mutationFn: () => shareUnlock(token, password),
    onSuccess: (data) => onUnlocked(data),
    onError: (e) => setError(shareErrorText(e, 'Could not open the link.')),
  })
  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!password || unlock.isPending) return
    setError('')
    unlock.mutate()
  }
  return (
    <main className="flex min-h-dvh flex-col items-center justify-center gap-4 bg-bg px-4 text-ink">
      <span className="flex size-12 items-center justify-center rounded-xl bg-ink text-bg">
        <Lock className="size-6" aria-hidden />
      </span>
      <h1 className="text-lg font-semibold">This link needs a password</h1>
      <form onSubmit={submit} className="flex w-full max-w-sm flex-col gap-3">
        <label htmlFor={id} className="text-sm font-medium">
          Password
        </label>
        <input
          id={id}
          className={inputClass}
          type="password"
          autoComplete="current-password"
          autoFocus
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          aria-invalid={!!error}
        />
        {error && (
          <p role="alert" className="text-sm text-danger">
            {error}
          </p>
        )}
        <Button type="submit" loading={unlock.isPending} disabled={!password}>
          Open
        </Button>
      </form>
      <p className="text-sm text-ink-soft">Shared from {siteTitle}.</p>
    </main>
  )
}

/** Keep Tab and Shift+Tab inside the lightbox, the way the app's Modal does. */
function trapTab(e: KeyboardEvent, root: HTMLElement | null): void {
  if (!root) return
  const nodes = tabbables(root)
  if (nodes.length === 0) {
    e.preventDefault()
    root.focus()
    return
  }
  const first = nodes[0]
  const last = nodes[nodes.length - 1]
  const active = document.activeElement
  const inside = active instanceof Node && root.contains(active)
  if (e.shiftKey ? active === first || !inside : active === last || !inside) {
    e.preventDefault()
    ;(e.shiftKey ? last : first).focus()
  }
}

function Lightbox({
  items,
  index,
  allowDownload,
  onClose,
  onMove,
}: {
  items: ShareItem[]
  index: number
  allowDownload: boolean
  onClose: () => void
  onMove: (i: number) => void
}) {
  const item = items[index]
  const prev = index > 0 ? () => onMove(index - 1) : null
  const next = index < items.length - 1 ? () => onMove(index + 1) : null
  const panel = useRef<HTMLDivElement>(null)
  const closeButton = useRef<HTMLButtonElement>(null)
  const onKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowLeft' && prev) prev()
      else if (e.key === 'ArrowRight' && next) next()
      else if (e.key === 'Tab') trapTab(e, panel.current)
    },
    [onClose, prev, next],
  )
  useEffect(() => {
    document.addEventListener('keydown', onKey)
    const unlock = lockScroll()
    return () => {
      document.removeEventListener('keydown', onKey)
      unlock()
    }
  }, [onKey])
  // focus moves into the dialog while it is open and back to the tile that opened it afterwards
  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null
    closeButton.current?.focus()
    return () => opener?.focus()
  }, [])
  const nav =
    'absolute top-1/2 z-10 flex size-11 -translate-y-1/2 items-center justify-center rounded-full bg-black/45 text-white backdrop-blur-sm transition-colors hover:bg-black/70 sm:size-12'
  return (
    <div
      ref={panel}
      tabIndex={-1}
      role="dialog"
      aria-modal="true"
      aria-label="Photo viewer"
      className="fixed inset-0 z-40 flex flex-col bg-[#111] text-white outline-none"
    >
      <div className="flex h-14 shrink-0 items-center gap-1 px-1.5 sm:px-3">
        <button ref={closeButton} type="button" onClick={onClose} className={iconButton} aria-label="Close viewer" title="Close (Esc)">
          <X className="size-5" aria-hidden />
        </button>
        <span className="ml-1 text-sm tabular-nums text-white/75">
          {index + 1} / {items.length}
        </span>
        <span className="ml-2 hidden min-w-0 truncate text-sm text-white/60 md:inline" title={item.filename}>
          {item.filename}
        </span>
        <div className="ml-auto flex items-center gap-1">
          {allowDownload && item.urls.download && (
            <a href={item.urls.download} download={item.filename} className={iconButton} aria-label="Download the original" title="Download">
              <Download className="size-5" aria-hidden />
            </a>
          )}
        </div>
      </div>
      <div className="relative flex min-h-0 flex-1 items-center justify-center p-2 sm:p-4">
        {item.kind === 'video' && item.urls.video ? (
          <video key={item.uid} controls playsInline preload="metadata" poster={item.urls.preview} src={item.urls.video} className="max-h-full max-w-full" />
        ) : (
          <img key={item.uid} src={item.urls.preview} alt={item.filename} className="max-h-full max-w-full object-contain" />
        )}
        {prev && (
          <button type="button" onClick={prev} className={`${nav} left-2 sm:left-4`} aria-label="Previous">
            <ChevronLeft className="size-6" aria-hidden />
          </button>
        )}
        {next && (
          <button type="button" onClick={next} className={`${nav} right-2 sm:right-4`} aria-label="Next">
            <ChevronRight className="size-6" aria-hidden />
          </button>
        )}
      </div>
    </div>
  )
}
