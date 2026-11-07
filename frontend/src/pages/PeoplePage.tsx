// People (/people): everyone found in the library, named or not, with search, filters, merge
// suggestions, a multi-select merge, and the drawer of faces that belong to nobody yet.
import { useCallback, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router'
import { useMutation } from '@tanstack/react-query'
import { GitMerge, PawPrint, ScanFace, Users } from 'lucide-react'
import { api } from '../lib/api'
import type { PersonOut } from '../lib/types'
import { formatCount, personName } from '../lib/format'
import { useDebounced, useInfiniteSentinel, usePageTitle, useRefreshLibrary } from '../lib/hooks'
import { useItemList } from '../lib/useItemList'
import MergeModal from '../components/people/MergeModal'
import PersonCard from '../components/people/PersonCard'
import SuggestionsPanel from '../components/people/SuggestionsPanel'
import UnassignedFaces from '../components/people/UnassignedFaces'
import Select from '../components/Select'
import { useToast } from '../components/Toast'
import { errorMessage, useSelection } from '../components/admin/adminHooks'
import { LoadMore, SearchInput, Segmented, SelectionBar } from '../components/admin/fields'
import { Button, EmptyState, ErrorState, buttonClasses } from '../components/ui'

type Filter = 'all' | 'named' | 'unnamed'
type Sort = 'items' | 'name' | 'recent'

/** Pets found in the library, below the people. Search and filters above apply to people only. */
function PetsSection() {
  const fetchPage = useCallback((cursor: string | null) => api.people.list({ kind: 'pet', sort: 'items', cursor, limit: 60 }), [])
  const list = useItemList<PersonOut>(['people', 'pets'], fetchPage, { keyOf: (p) => p.id })
  const sentinel = useInfiniteSentinel(list.loadMore, list.hasMore && !list.loading)
  return (
    <section className="mt-10 pb-8" aria-labelledby="pets-heading">
      <div className="mb-3">
        <h2 id="pets-heading" className="text-xl font-bold tracking-tight">
          Pets
        </h2>
        <p className="mt-1 text-sm text-ink-soft" aria-live="polite">
          {list.loading && !list.items.length ? 'Loading' : formatCount(list.total, 'pet')}
        </p>
      </div>
      {list.error && !list.items.length ? (
        <ErrorState error={list.error} onRetry={list.refetch} />
      ) : list.loading && !list.items.length ? (
        <ul className="grid grid-cols-3 gap-2 sm:grid-cols-4 sm:gap-3 md:grid-cols-5 lg:grid-cols-6" aria-busy="true">
          {Array.from({ length: 6 }, (_, i) => (
            <li key={i} className="aspect-[4/5] animate-pulse rounded-2xl bg-surface-2" />
          ))}
        </ul>
      ) : list.items.length === 0 ? (
        <EmptyState icon={<PawPrint className="size-8" aria-hidden />} title="No pets found yet">
          Pets are found in new photos as they are analysed.
        </EmptyState>
      ) : (
        <ul className="grid grid-cols-3 gap-2 sm:grid-cols-4 sm:gap-3 md:grid-cols-5 lg:grid-cols-6">
          {list.items.map((p) => (
            <li key={p.id} className="min-w-0">
              <PersonCard person={p} />
            </li>
          ))}
        </ul>
      )}
      <LoadMore hasMore={list.hasMore} loading={list.loading} onClick={list.loadMore} sentinelRef={sentinel} />
    </section>
  )
}

export default function PeoplePage() {
  usePageTitle('People')
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const [params, setParams] = useSearchParams()
  const filter: Filter = params.get('filter') === 'named' || params.get('filter') === 'unnamed' ? (params.get('filter') as Filter) : 'all'
  const sort: Sort = params.get('sort') === 'name' || params.get('sort') === 'recent' ? (params.get('sort') as Sort) : 'items'
  const [text, setText] = useState(params.get('q') ?? '')
  const q = useDebounced(text.trim(), 300)
  const [drawer, setDrawer] = useState(false)
  const [selecting, setSelecting] = useState(false)
  const sel = useSelection<number>()
  const [merging, setMerging] = useState(false)

  const query = useMemo(() => ({ q: q || undefined, kind: 'person' as const, filter, sort }), [q, filter, sort])
  const fetchPage = useCallback((cursor: string | null) => api.people.list({ ...query, cursor, limit: 60 }), [query])
  const list = useItemList<PersonOut>(['people', 'list', query], fetchPage, { keyOf: (p) => p.id })
  const sentinel = useInfiniteSentinel(list.loadMore, list.hasMore && !list.loading)
  const chosen = list.items.filter((p) => sel.has(p.id))

  const setParam = (key: string, value: string | null) => {
    const p = new URLSearchParams(params)
    if (value) p.set(key, value)
    else p.delete(key)
    setParams(p, { replace: true })
  }

  const merge = useMutation({
    mutationFn: (v: { target: number; sources: number[] }) => api.people.merge({ target_id: v.target, source_ids: v.sources }),
    onSuccess: (p) => {
      toast.success(`Merged into ${personName(p)}.`)
      setMerging(false)
      sel.clear()
      setSelecting(false)
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not merge.')),
  })

  return (
    <div className="mx-auto w-full max-w-7xl px-4">
      <div className="flex flex-wrap items-end justify-between gap-3 pb-4 pt-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">People</h1>
          <p className="mt-1 text-sm text-ink-soft" aria-live="polite">
            {list.loading && !list.items.length ? 'Loading' : formatCount(list.total, 'person', 'people')}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" className={buttonClasses('secondary', 'md')} onClick={() => setDrawer(true)}>
            <ScanFace className="size-4" aria-hidden />
            Unassigned faces
          </button>
          <button
            type="button"
            className={buttonClasses('secondary', 'md')}
            aria-pressed={selecting}
            onClick={() => {
              sel.clear()
              setSelecting((v) => !v)
            }}
          >
            <GitMerge className="size-4" aria-hidden />
            {selecting ? 'Done' : 'Merge'}
          </button>
        </div>
      </div>

      <SuggestionsPanel />

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <SearchInput value={text} onChange={(v) => {
          setText(v)
          setParam('q', v.trim() || null)
        }} label="Search people by name" className="w-full sm:w-72" />
        <Segmented<Filter>
          label="Show"
          value={filter}
          onChange={(v) => setParam('filter', v === 'all' ? null : v)}
          options={[
            { value: 'all', label: 'All' },
            { value: 'named', label: 'Named' },
            { value: 'unnamed', label: 'Unnamed' },
          ]}
        />
        <Select
          label="Sort"
          value={sort}
          onChange={(v) => setParam('sort', v === 'items' ? null : v)}
          options={[
            { value: 'items', label: 'Most items' },
            { value: 'name', label: 'Name' },
            { value: 'recent', label: 'Recently seen' },
          ]}
        />
      </div>

      {selecting && <p className="mb-3 text-sm text-ink-soft">Pick two or more people who are the same person.</p>}

      {list.error && !list.items.length ? (
        <ErrorState error={list.error} onRetry={list.refetch} />
      ) : list.loading && !list.items.length ? (
        <ul className="grid grid-cols-3 gap-2 sm:grid-cols-4 sm:gap-3 md:grid-cols-5 lg:grid-cols-6" aria-busy="true">
          {Array.from({ length: 12 }, (_, i) => (
            <li key={i} className="aspect-[4/5] animate-pulse rounded-2xl bg-surface-2" />
          ))}
        </ul>
      ) : list.items.length === 0 ? (
        <EmptyState icon={<Users className="size-8" aria-hidden />} title={q ? 'No one matches that name' : 'No people yet'}>
          {q ? 'Try another spelling.' : 'Faces found in your photos and videos are grouped into people here. Name them to find them later.'}
        </EmptyState>
      ) : (
        <ul className="grid grid-cols-3 gap-2 sm:grid-cols-4 sm:gap-3 md:grid-cols-5 lg:grid-cols-6">
          {list.items.map((p) => (
            <li key={p.id} className="min-w-0">
              {selecting ? <PersonCard person={p} selected={sel.has(p.id)} onSelect={() => sel.toggle(p.id)} /> : <PersonCard person={p} />}
            </li>
          ))}
        </ul>
      )}
      <LoadMore hasMore={list.hasMore} loading={list.loading} onClick={list.loadMore} sentinelRef={sentinel} />

      {selecting && sel.count > 0 && (
        <SelectionBar count={sel.count} noun={sel.count === 1 ? 'person' : 'people'} onClear={sel.clear}>
          <Button disabled={sel.count < 2} onClick={() => setMerging(true)}>
            <GitMerge className="size-4" aria-hidden />
            Merge
          </Button>
        </SelectionBar>
      )}

      <MergeModal
        open={merging}
        people={chosen}
        busy={merge.isPending}
        onConfirm={(target, sources) => merge.mutate({ target, sources })}
        onClose={() => setMerging(false)}
      />
      <UnassignedFaces open={drawer} onClose={() => setDrawer(false)} />

      <PetsSection />
    </div>
  )
}
