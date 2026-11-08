// The library: filters and one search box in the URL, the item grid with multi-select and bulk
// actions, and the viewer (?m=<uid>). With `vault` it lists only private items and every request
// carries vault=1; the vault page renders it only while the vault is unlocked.
import { useCallback, useMemo, useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router'
import { ImageOff, SearchX, Upload } from 'lucide-react'
import { formatCount } from '../../lib/format'
import { useStableCallback } from '../../lib/hooks'
import { useLibraryItems } from '../../lib/useItemList'
import { useSearchList } from '../../lib/useSearchList'
import type { SearchMode } from '../../lib/types'
import MediaGrid from '../MediaGrid'
import Viewer from '../Viewer'
import { useSelection } from '../admin/adminHooks'
import Segmented from '../public/Segmented'
import { withParam } from '../public/params'
import { useViewerParam } from '../public/useViewerParam'
import { EmptyState, ErrorState, buttonClasses } from '../ui'
import BulkBar from './BulkBar'
import FilterBar from './FilterBar'
import SearchBox from './SearchBox'
import { itemsQuery, readFilters, searchQuery } from './filters'

interface Props {
  vault?: boolean
  title?: string
  /** Extra controls in the header (the vault's countdown and lock button). */
  headerExtra?: ReactNode
  /** Show one album's members only (section 24). */
  album?: number
}

export default function Library({ vault = false, title, headerExtra, album }: Props) {
  const [params, setParams] = useSearchParams()
  const filters = useMemo(() => readFilters(params, vault), [params, vault])
  const searching = filters.q !== ''

  const iq = useMemo(() => itemsQuery(filters, vault, album), [filters, vault, album])
  const sq = useMemo(() => searchQuery(filters, vault), [filters, vault])
  const browse = useLibraryItems(iq, !searching)
  const search = useSearchList(sq, searching)
  const list = searching ? search : browse
  const viewer = useViewerParam(list.items, list.hasMore, list.loadMore, list.total)
  const openItem = useStableCallback((uid: string) => viewer.open(uid))

  const selection = useSelection<string>()
  const [selectMode, setSelectMode] = useState(false)
  const selecting = selectMode || selection.count > 0

  const set = useCallback(
    (key: string, value: string | null) => {
      selection.clear()
      setParams(withParam(params, key, value), { replace: true })
    },
    [params, setParams, selection],
  )
  const clearAll = () => {
    const p = new URLSearchParams()
    if (searching) p.set('q', filters.q)
    if (filters.mode !== 'auto') p.set('mode', filters.mode)
    setParams(p, { replace: true })
  }
  const clearSearch = () => set('q', null)
  const filtered = !!(filters.category || filters.kind || filters.people.length || filters.tags.length || filters.live || filters.from || filters.to || filters.favorite)

  const heading = title ?? (vault ? 'Vault' : 'Library')
  const countLine =
    list.loading && !list.items.length
      ? searching
        ? 'Searching'
        : 'Loading'
      : searching
        ? search.unavailable
          ? ''
          : `${formatCount(list.total, 'result')} for "${filters.q}"`
        : formatCount(list.total, 'item')

  const empty = searching ? (
    <EmptyState
      icon={<ImageOff className="size-8" aria-hidden />}
      title="Nothing matches. Try other words."
      action={
        <button type="button" onClick={clearSearch} className={buttonClasses('secondary', 'md')}>
          Clear search
        </button>
      }
    >
      {search.indexing
        ? 'Some items are still being indexed, so try again in a few minutes.'
        : 'Describe what is in the picture, like "beach at sunset", or type words from a document, like a name or a number.'}
    </EmptyState>
  ) : filtered ? (
    <EmptyState
      icon={<ImageOff className="size-8" aria-hidden />}
      title="Nothing matches these filters"
      action={
        <button type="button" onClick={clearAll} className={buttonClasses('secondary', 'md')}>
          Clear filters
        </button>
      }
    />
  ) : vault ? (
    <EmptyState icon={<ImageOff className="size-8" aria-hidden />} title="The vault is empty">
      Items marked private, and photos found to be explicit, are kept here.
    </EmptyState>
  ) : (
    <EmptyState
      icon={<ImageOff className="size-8" aria-hidden />}
      title="Nothing here yet"
      action={
        <Link to="/upload" className={buttonClasses('primary', 'md')}>
          <Upload className="size-4" aria-hidden />
          Upload
        </Link>
      }
    >
      Photos, videos, PDFs and other documents you upload appear here, sorted and searchable.
    </EmptyState>
  )

  return (
    <div className="mx-auto w-full max-w-7xl px-4">
      <div className="flex flex-wrap items-end justify-between gap-3 pb-4 pt-6">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">{searching ? 'Search results' : heading}</h1>
          <p className="mt-1 text-sm tabular-nums text-ink-soft" aria-live="polite">
            {countLine}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {headerExtra}
          <button
            type="button"
            onClick={() => {
              if (selecting) {
                selection.clear()
                setSelectMode(false)
              } else setSelectMode(true)
            }}
            aria-pressed={selecting}
            className={buttonClasses('secondary', 'md')}
          >
            {selecting ? 'Done' : 'Select'}
          </button>
          {!vault && (
            <Link to="/upload" className={buttonClasses('primary', 'md')}>
              <Upload className="size-4" aria-hidden />
              Upload
            </Link>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-2 pb-4">
        <SearchBox
          value={filters.q}
          onCommit={(next) => set('q', next || null)}
          placeholder={vault ? 'Search the vault' : 'Search photos, videos and documents'}
        />
        {searching && (
          <div className="flex flex-wrap items-center gap-3">
            <Segmented<SearchMode>
              label="Search by"
              value={filters.mode}
              onChange={(v) => set('mode', v === 'auto' ? null : v)}
              options={[
                { value: 'auto', label: 'Best match' },
                { value: 'semantic', label: 'Looks like' },
                { value: 'text', label: 'Text' },
              ]}
            />
            {search.indexing && !search.unavailable && (
              <p className="text-sm text-ink-soft" role="status">
                Still indexing: {search.indexed.toLocaleString()} items can be found so far.
              </p>
            )}
          </div>
        )}
      </div>

      <FilterBar filters={filters} set={set} clearAll={clearAll} vault={vault} searching={searching} />

      <div className="-mx-4 pt-3 sm:mx-0">
        {searching && search.unavailable ? (
          <EmptyState
            icon={<SearchX className="size-8" aria-hidden />}
            title="Search is not available right now"
            action={
              <button type="button" onClick={clearSearch} className={buttonClasses('secondary', 'md')}>
                Browse instead
              </button>
            }
          >
            The search model is not installed or still loading. Settings shows its status.
          </EmptyState>
        ) : list.error && !list.items.length ? (
          <ErrorState error={list.error} onRetry={list.refetch} />
        ) : (
          <MediaGrid
            items={list.items}
            onOpen={openItem}
            hasMore={list.hasMore}
            loadMore={list.loadMore}
            loading={list.loading}
            selectable={selecting}
            selected={selection.selected}
            onToggleSelect={selection.toggle}
            selectOnHover
            showSnippets={searching}
            vault={vault}
            empty={empty}
          />
        )}
        {!!list.error && list.items.length > 0 && !list.loading && (
          <div className="flex flex-col items-center gap-2 py-6 text-center" role="alert">
            <p className="text-sm text-ink-soft">Could not load more.</p>
            <button type="button" onClick={list.loadMore} className={buttonClasses('secondary', 'md')}>
              Try again
            </button>
          </div>
        )}
      </div>

      {selection.count > 0 && (
        <div className="px-0 pb-4">
          <BulkBar
            items={list.items}
            selected={selection.selected}
            onClear={() => {
              selection.clear()
              setSelectMode(false)
            }}
            onSelectAll={() => selection.setAll(list.items.map((i) => i.uid))}
            total={list.items.length}
            vault={vault}
            album={album}
          />
        </div>
      )}

      {viewer.uid && (
        <Viewer
          uid={viewer.uid}
          onClose={viewer.close}
          onPrev={viewer.onPrev}
          onNext={viewer.onNext}
          position={viewer.position}
          startAt={viewer.t}
          vault={vault}
        />
      )}
    </div>
  )
}
