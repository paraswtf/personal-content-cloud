// Document mode of the viewer: the current page large, previous and next page buttons, and a strip
// of all pages below. Without page renders yet it shows the first-page preview; files that cannot be
// previewed show a card with the download.
import { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import { ChevronLeft, ChevronRight, Download, ExternalLink, FileText, Loader2 } from 'lucide-react'
import type { ItemDetail } from '../../lib/types'
import { formatBytes } from '../../lib/format'
import { docTypeLabel, extLabel } from '../documents/docMeta'
import { darkButton } from './shared'

interface Props {
  detail: ItemDetail
  page: number
  setPage: (n: number) => void
}

function PageImage({ src, label }: { src: string; label: string }) {
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    setLoaded(false)
    setFailed(false)
  }, [src])
  return (
    <div className="relative flex size-full items-center justify-center">
      {failed ? (
        <p className="rounded-xl bg-black/50 p-4 text-center text-sm text-white/80">This page could not be loaded.</p>
      ) : (
        <img
          src={src}
          alt={label}
          draggable={false}
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
          className={clsx(
            'max-h-full max-w-full select-none bg-white object-contain shadow-2xl transition-opacity duration-200',
            loaded ? 'opacity-100' : 'opacity-0',
          )}
        />
      )}
      {!loaded && !failed && (
        <span className="absolute inset-0 flex items-center justify-center" role="status">
          <Loader2 className="size-7 animate-spin text-white/70" aria-hidden />
          <span className="sr-only">Loading page</span>
        </span>
      )}
    </div>
  )
}

export function FileCard({ detail }: { detail: ItemDetail }) {
  return (
    <div className="flex max-w-sm flex-col items-center gap-4 rounded-2xl bg-white/5 p-6 text-center text-white">
      <span className="flex size-16 items-center justify-center rounded-2xl bg-white/10">
        <FileText className="size-8" aria-hidden />
      </span>
      <div className="min-w-0">
        <p className="break-all text-base font-semibold">{detail.filename}</p>
        <p className="mt-1 text-sm text-white/60">
          {extLabel(detail.filename)} file, {formatBytes(detail.size_bytes)}
        </p>
      </div>
      <p className="text-sm text-white/70">There is no preview for this kind of file. The original is kept exactly as uploaded.</p>
      {detail.urls.download && (
        <a href={detail.urls.download} download className={darkButton('primary')}>
          <Download className="size-4" aria-hidden />
          Download original
        </a>
      )}
    </div>
  )
}

export default function DocumentStage({ detail, page, setPage }: Props) {
  const pages = detail.urls.pages
  const count = pages.length
  const strip = useRef<HTMLUListElement>(null)
  const current = Math.min(Math.max(page, 0), Math.max(count - 1, 0))
  const label = detail.document ? docTypeLabel(detail.document.doc_type) : detail.filename

  useEffect(() => {
    strip.current?.querySelector<HTMLElement>(`[data-page="${current}"]`)?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }, [current])

  if (!count) {
    return (
      <div className="flex size-full flex-col items-center justify-center gap-3">
        <div className="flex min-h-0 w-full flex-1 items-center justify-center">
          <PageImage src={detail.urls.preview} label={`${label}, first page`} />
        </div>
        <p className="flex items-center gap-2 text-sm text-white/65" role="status">
          {detail.text_status === 'pending' && <Loader2 className="size-4 animate-spin" aria-hidden />}
          {detail.text_status === 'pending' ? 'Pages are still being read.' : 'Only the first page has a preview.'}
        </p>
      </div>
    )
  }

  const navBtn =
    'flex size-10 shrink-0 items-center justify-center rounded-full bg-black/45 text-white hover:bg-black/70 disabled:opacity-30 disabled:hover:bg-black/45'

  return (
    <div className="flex size-full min-h-0 flex-col gap-2">
      <div className="relative flex min-h-0 flex-1 items-center justify-center">
        <PageImage src={pages[current]} label={`${label}, page ${current + 1} of ${count}`} />
      </div>
      <div className="flex shrink-0 items-center gap-2" data-no-swipe>
        <button type="button" className={navBtn} onClick={() => setPage(current - 1)} disabled={current <= 0} aria-label="Previous page">
          <ChevronLeft className="size-5" aria-hidden />
        </button>
        <ul ref={strip} className="flex min-w-0 flex-1 gap-1.5 overflow-x-auto overscroll-x-contain py-1" aria-label="Pages">
          {pages.map((src, i) => (
            <li key={src} data-page={i} className="shrink-0">
              <button
                type="button"
                onClick={() => setPage(i)}
                aria-current={i === current ? 'page' : undefined}
                aria-label={`Page ${i + 1}`}
                className={clsx(
                  'relative block h-16 w-12 overflow-hidden rounded bg-white ring-2 transition',
                  i === current ? 'ring-accent' : 'ring-transparent opacity-70 hover:opacity-100',
                )}
              >
                <img src={src} alt="" loading="lazy" className="size-full object-cover object-top" />
                <span className="absolute inset-x-0 bottom-0 bg-black/55 text-center text-[10px] font-semibold text-white">{i + 1}</span>
              </button>
            </li>
          ))}
        </ul>
        <span className="hidden shrink-0 text-sm tabular-nums text-white/70 sm:inline">
          {current + 1} / {count}
        </span>
        <button type="button" className={navBtn} onClick={() => setPage(current + 1)} disabled={current >= count - 1} aria-label="Next page">
          <ChevronRight className="size-5" aria-hidden />
        </button>
        {detail.kind === 'pdf' && detail.urls.original && (
          <a
            href={detail.urls.original}
            target="_blank"
            rel="noopener"
            className="hidden size-10 shrink-0 items-center justify-center rounded-full text-white/80 hover:bg-white/10 sm:flex"
            aria-label="Open the PDF in a new tab"
            title="Open the PDF in a new tab"
          >
            <ExternalLink className="size-5" aria-hidden />
          </a>
        )}
      </div>
    </div>
  )
}
