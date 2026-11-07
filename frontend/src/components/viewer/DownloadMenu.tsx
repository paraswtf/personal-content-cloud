// Viewer download control: a small menu with the original (always byte for byte) and, for videos,
// the frame on screen as a JPEG; documents also get the current page image. Every download is a
// plain link, so the tap itself is the download gesture (iOS Safari ignores downloads started later).
import { useEffect, useRef, useState, type KeyboardEvent, type MouseEvent } from 'react'
import { Download, FileImage, Image as ImageIcon } from 'lucide-react'
import { frameUrl } from '../../lib/api'
import type { ItemDetail } from '../../lib/types'
import { formatBytes } from '../../lib/format'
import { withVault } from '../../lib/preview'
import { iconButton } from './shared'

const itemClass =
  'flex min-h-11 w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm text-white/90 hover:bg-white/10 focus-visible:bg-white/10'

interface Props {
  detail: ItemDetail
  vault: boolean
  /** Videos: the time of the frame on screen now, in seconds. */
  frameTime?: () => number
  /** Documents: the URL of the page on screen. */
  pageUrl?: string | null
}

export default function DownloadMenu({ detail, vault, frameTime, pageUrl }: Props) {
  const [open, setOpen] = useState(false)
  const [openedAt, setOpenedAt] = useState(0)
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const menu = useRef<HTMLDivElement>(null)

  useEffect(() => setOpen(false), [detail.uid])
  useEffect(() => {
    if (!open) return
    const onDown = (e: Event) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    menu.current?.querySelector<HTMLElement>('[role="menuitem"]')?.focus()
    return () => document.removeEventListener('pointerdown', onDown)
  }, [open])

  const isVideo = detail.kind === 'video'
  const now = () => frameTime?.() ?? 0
  const frameHref = (t: number) => withVault(frameUrl(detail.uid, detail.version, t, 'full', true, detail.nsfw), vault)
  const onFrameClick = (e: MouseEvent<HTMLAnchorElement>) => {
    e.currentTarget.href = frameHref(now())
    setOpen(false)
  }

  const items = () => Array.from(menu.current?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [])
  const onKeyDown = (e: KeyboardEvent) => {
    if (!open) return
    const list = items()
    const at = list.indexOf(document.activeElement as HTMLElement)
    if (e.key === 'Escape') {
      e.stopPropagation()
      e.preventDefault()
      setOpen(false)
      trigger.current?.focus()
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      list[(at + 1) % list.length]?.focus()
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      list[(at - 1 + list.length) % list.length]?.focus()
    } else if (e.key === 'Tab') {
      setOpen(false)
    }
  }

  if (!detail.urls.download) return null
  return (
    <div ref={root} className="relative" onKeyDown={onKeyDown}>
      <button
        ref={trigger}
        type="button"
        className={iconButton}
        aria-label="Download"
        title="Download"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-pressed={open}
        onClick={() => {
          if (!open && isVideo) setOpenedAt(now())
          setOpen((v) => !v)
        }}
      >
        <Download className="size-5" aria-hidden />
      </button>
      {open && (
        <div
          ref={menu}
          role="menu"
          aria-label="Download options"
          className="absolute right-0 top-full z-30 mt-1 w-64 max-w-[calc(100vw-1rem)] rounded-xl border border-white/10 bg-[#222] p-1.5 shadow-2xl"
        >
          <a role="menuitem" href={detail.urls.download} download className={itemClass} onClick={() => setOpen(false)}>
            <Download className="size-4 shrink-0" aria-hidden />
            <span className="min-w-0">
              Download original
              <span className="block text-xs text-white/50">{formatBytes(detail.size_bytes)}, full quality</span>
            </span>
          </a>
          {isVideo && (
            <a role="menuitem" href={frameHref(openedAt)} download className={itemClass} onClick={onFrameClick}>
              <ImageIcon className="size-4 shrink-0" aria-hidden />
              Download this frame
            </a>
          )}
          {pageUrl && (
            <a role="menuitem" href={pageUrl} download className={itemClass} onClick={() => setOpen(false)}>
              <FileImage className="size-4 shrink-0" aria-hidden />
              Download this page as an image
            </a>
          )}
        </div>
      )}
    </div>
  )
}
