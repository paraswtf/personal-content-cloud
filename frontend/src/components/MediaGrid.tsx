// Tight grid of square tiles with lazy thumbnails, badges, optional selection and infinite scroll.
// Documents show their first page inside a paper frame with the type badge; videos show their length;
// Live Photos and motion photos show a LIVE badge;
// search hits can show the matched moment and a text snippet.
import { memo, useState, type ReactNode } from 'react'
import clsx from 'clsx'
import { Check, Copy, FileText, ImageOff, Play, Star, TriangleAlert } from 'lucide-react'
import type { ItemCard } from '../lib/types'
import { formatDate, formatDuration } from '../lib/format'
import { useInfiniteSentinel } from '../lib/hooks'
import { withVault } from '../lib/preview'
import { Button, EmptyState, Spinner } from './ui'
import { docTypeLabel, extLabel } from './documents/docMeta'
import { reasonKeys, reasonLabel } from './library/misclick'
import Snippet from './library/Snippet'
import { LiveIcon } from './viewer/LiveButton'

/** A grid item; search hits also carry the matched moment of a video and a text snippet. */
export type GridItem = ItemCard & { match_time_s?: number | null; snippet?: string | null; source?: string }

interface Props {
  items: GridItem[]
  onOpen: (uid: string, index: number) => void
  hasMore: boolean
  loadMore: () => void
  loading: boolean
  /** Selection mode: a tap toggles selection instead of opening the item. */
  selectable?: boolean
  selected?: Set<string>
  onToggleSelect?: (uid: string) => void
  /** Show a select checkbox on every tile even outside selection mode (it starts selecting). */
  selectOnHover?: boolean
  /** Show mis-click reasons as chips on the tiles (review bin). */
  showReasons?: boolean
  /** Show search snippets on the tiles. */
  showSnippets?: boolean
  /** Vault items: file URLs carry vault=1. */
  vault?: boolean
  empty?: ReactNode
}

export const GRID = 'grid grid-cols-3 gap-0.5 sm:grid-cols-4 sm:gap-1 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-7'

function matchTime(item: GridItem): number | null {
  return item.kind === 'video' && typeof item.match_time_s === 'number' ? item.match_time_s : null
}

const KIND_NAMES: Record<string, string> = { image: 'Photo', video: 'Video', pdf: 'PDF', document: 'Document', other: 'File' }

function describe(item: GridItem): string {
  const what = item.doc_type
    ? docTypeLabel(item.doc_type)
    : item.live === true && item.kind === 'image'
      ? 'Live Photo'
      : (KIND_NAMES[item.kind] ?? 'Item')
  const len = item.kind === 'video' && item.duration_s ? `, ${formatDuration(item.duration_s)}` : ''
  const at = matchTime(item)
  const match = at === null ? '' : `, match at ${formatDuration(at)}`
  const when = formatDate(item.taken_at ?? item.created_at)
  const flags = [item.favorite && 'favourite', item.misclick && 'possible mis-click', item.dup_count > 0 && `${item.dup_count} similar`]
    .filter(Boolean)
    .join(', ')
  return `${what}${len}${match}${when ? `, ${when}` : ''}${flags ? `, ${flags}` : ''}`
}

function isDocument(item: GridItem): boolean {
  return item.kind === 'pdf' || item.kind === 'document' || (item.kind === 'image' && !!item.doc_type && item.category === 'documents')
}

const Tile = memo(function Tile({
  item,
  index,
  selectable,
  selectOnHover,
  isSelected,
  showReasons,
  showSnippets,
  vault,
  onOpen,
  onToggleSelect,
}: {
  item: GridItem
  index: number
  selectable: boolean
  selectOnHover: boolean
  isSelected: boolean
  showReasons: boolean
  showSnippets: boolean
  vault: boolean
  onOpen: (uid: string, index: number) => void
  onToggleSelect?: (uid: string) => void
}) {
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)
  const label = describe(item)
  const doc = isDocument(item)
  const thumb = withVault(item.urls.thumb, vault)
  const reasons = showReasons ? reasonKeys(item.misclick_reasons) : []
  const at = matchTime(item)

  const img = failed ? (
    <span className="absolute inset-0 flex items-center justify-center text-ink-faint">
      {doc ? <FileText className="size-7" aria-hidden /> : <ImageOff className="size-6" aria-hidden />}
    </span>
  ) : (
    <img
      src={thumb}
      alt=""
      loading="lazy"
      decoding="async"
      onLoad={() => setLoaded(true)}
      onError={() => setFailed(true)}
      className={clsx(
        'transition-[opacity,transform] duration-300',
        loaded ? 'opacity-100' : 'opacity-0',
        doc ? 'size-full object-contain object-top' : 'absolute inset-0 size-full object-cover',
        !doc && (selectable && isSelected ? 'scale-[0.86] rounded-md' : 'group-hover:scale-[1.03]'),
      )}
    />
  )

  return (
    <div className="group relative aspect-square w-full overflow-hidden bg-surface-2">
      <button
        type="button"
        onClick={() => (selectable ? onToggleSelect?.(item.uid) : onOpen(item.uid, index))}
        aria-label={selectable ? `Select ${label}` : `Open ${label}`}
        aria-pressed={selectable ? isSelected : undefined}
        className="absolute inset-0 block size-full outline-none focus-visible:z-10 focus-visible:ring-3 focus-visible:ring-inset focus-visible:ring-accent"
      >
        {doc ? (
          <span
            className={clsx(
              'absolute inset-[10%] block overflow-hidden rounded-[3px] bg-white shadow-[0_1px_3px_rgb(0_0_0/0.25)] ring-1 ring-black/5 transition-transform',
              selectable && isSelected ? 'scale-[0.9]' : 'group-hover:-translate-y-0.5',
            )}
          >
            {img}
          </span>
        ) : (
          img
        )}
      </button>

      {doc && (
        <span className="pointer-events-none absolute bottom-1 left-1 max-w-[calc(100%-0.5rem)] truncate rounded-full bg-ink/80 px-1.5 py-0.5 text-[11px] font-semibold text-white">
          {item.doc_type ? docTypeLabel(item.doc_type) : extLabel(item.filename)}
          {item.page_count && item.page_count > 1 ? ` (${item.page_count} p)` : ''}
        </span>
      )}

      {item.kind === 'video' && (
        <span className="pointer-events-none absolute bottom-1 right-1 inline-flex items-center gap-1 rounded-full bg-black/60 px-1.5 py-0.5 text-[11px] font-semibold tabular-nums text-white">
          <Play className="size-3 fill-current" aria-hidden />
          {item.duration_s ? formatDuration(item.duration_s) : 'Video'}
        </span>
      )}

      {at !== null && (
        <span className="pointer-events-none absolute bottom-1 left-1 inline-flex items-center rounded-full bg-accent px-1.5 py-0.5 text-[11px] font-semibold tabular-nums text-white">
          at {formatDuration(at)}
        </span>
      )}

      <span className="pointer-events-none absolute left-1 top-1 flex flex-wrap gap-1">
        {item.favorite && (
          <span className="inline-flex size-5 items-center justify-center rounded-full bg-black/55 text-amber-300" title="Favourite">
            <Star className="size-3 fill-current" aria-hidden />
          </span>
        )}
        {item.live === true && (
          <span className="inline-flex items-center gap-1 rounded-full bg-black/55 px-1.5 py-0.5 text-[11px] font-bold tracking-wide text-white" title="Live Photo">
            <LiveIcon className="size-3" />
            LIVE
          </span>
        )}
        {item.misclick && !showReasons && (
          <span className="inline-flex items-center gap-1 rounded-full bg-warn px-1.5 py-0.5 text-[11px] font-semibold text-white" title="Possible mis-click">
            <TriangleAlert className="size-3" aria-hidden />
            <span className="hidden sm:inline">Mis-click</span>
          </span>
        )}
        {item.dup_count > 0 && (
          <span className="inline-flex items-center gap-0.5 rounded-full bg-black/60 px-1.5 py-0.5 text-[11px] font-semibold text-white" title="Similar copies">
            <Copy className="size-3" aria-hidden />+{item.dup_count}
          </span>
        )}
      </span>

      {reasons.length > 0 && (
        <span className="pointer-events-none absolute inset-x-1 bottom-1 flex flex-wrap gap-1">
          {reasons.map((r) => (
            <span key={r} className="rounded-full bg-warn px-1.5 py-0.5 text-[11px] font-semibold text-white">
              {reasonLabel(r)}
            </span>
          ))}
        </span>
      )}

      {showSnippets && item.snippet && (
        <span className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 via-black/60 to-transparent px-1.5 pb-1 pt-5 text-[11px] leading-snug text-white">
          <Snippet text={item.snippet} className="line-clamp-3" />
        </span>
      )}

      {(selectable || selectOnHover) && onToggleSelect && (
        <button
          type="button"
          onClick={() => onToggleSelect(item.uid)}
          aria-label={`${isSelected ? 'Deselect' : 'Select'} ${label}`}
          aria-pressed={isSelected}
          className={clsx(
            'absolute right-0 top-0 z-10 flex size-10 items-center justify-center',
            !selectable && 'opacity-0 focus-visible:opacity-100 group-hover:opacity-100',
          )}
        >
          <span
            aria-hidden
            className={clsx(
              'flex size-6 items-center justify-center rounded-full border-2 transition-colors',
              isSelected ? 'border-white bg-accent text-white' : 'border-white/90 bg-black/30 text-transparent',
            )}
          >
            <Check className="size-3.5" strokeWidth={3} />
          </span>
        </button>
      )}
      {selectable && isSelected && <span aria-hidden className="pointer-events-none absolute inset-0 ring-3 ring-inset ring-accent" />}
    </div>
  )
})

export default function MediaGrid({
  items,
  onOpen,
  hasMore,
  loadMore,
  loading,
  selectable = false,
  selected,
  onToggleSelect,
  selectOnHover = false,
  showReasons = false,
  showSnippets = false,
  vault = false,
  empty,
}: Props) {
  // The sentinel div is always rendered so the observer can attach once enabled.
  const sentinel = useInfiniteSentinel(loadMore, hasMore && !loading)
  const firstLoad = loading && items.length === 0

  return (
    <div>
      {firstLoad ? (
        <div className={GRID} aria-busy="true" aria-label="Loading">
          {Array.from({ length: 18 }, (_, i) => (
            <div key={i} className="aspect-square animate-pulse bg-surface-2" />
          ))}
        </div>
      ) : items.length === 0 && !hasMore ? (
        (empty ?? <EmptyState icon={<ImageOff className="size-8" aria-hidden />} title="Nothing to show here yet" />)
      ) : (
        <ul className={GRID} role="list">
          {items.map((item, i) => (
            <li key={item.uid} className="min-w-0">
              <Tile
                item={item}
                index={i}
                selectable={selectable}
                selectOnHover={selectOnHover}
                isSelected={!!selected?.has(item.uid)}
                showReasons={showReasons}
                showSnippets={showSnippets}
                vault={vault}
                onOpen={onOpen}
                onToggleSelect={onToggleSelect}
              />
            </li>
          ))}
        </ul>
      )}

      <div ref={sentinel} aria-hidden className="h-px" />
      {loading && items.length > 0 && (
        <div className="flex justify-center py-6">
          <Spinner label="Loading more" />
        </div>
      )}
      {!loading && hasMore && (
        <div className="flex justify-center py-6">
          <Button variant="secondary" onClick={loadMore}>
            Load more
          </Button>
        </div>
      )}
    </div>
  )
}
