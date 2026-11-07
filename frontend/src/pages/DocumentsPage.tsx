// Documents (/documents): parsed documents as virtual cards, filtered by type (grouped) and searched
// by their text and fields. A card opens the viewer in document mode.
import { useCallback, useMemo } from 'react'
import { useSearchParams } from 'react-router'
import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { FileSearch, FileText } from 'lucide-react'
import { api } from '../lib/api'
import type { DocumentListItem, DocumentTypeOut } from '../lib/types'
import { formatCount } from '../lib/format'
import { usePageTitle, useInfiniteSentinel } from '../lib/hooks'
import { useItemList } from '../lib/useItemList'
import DocumentCard from '../components/documents/DocumentCard'
import { GROUP_ORDER, groupMeta, docTypeLabel } from '../components/documents/docMeta'
import SearchBox from '../components/library/SearchBox'
import { withParam } from '../components/public/params'
import { useViewerParam } from '../components/public/useViewerParam'
import Viewer from '../components/Viewer'
import { EmptyState, ErrorState } from '../components/ui'
import { LoadMore } from '../components/admin/fields'

const chip = (on: boolean) =>
  clsx(
    'inline-flex h-10 shrink-0 items-center gap-1.5 rounded-full border px-3.5 text-sm font-semibold transition-colors',
    on ? 'border-ink bg-ink text-surface' : 'border-line bg-surface text-ink-soft hover:border-ink-faint hover:text-ink',
  )

function TypeChips({
  types,
  type,
  group,
  set,
}: {
  types: DocumentTypeOut[]
  type: string
  group: string
  set: (key: 'type' | 'group', value: string | null) => void
}) {
  const groups = useMemo(() => {
    const by = new Map<string, DocumentTypeOut[]>()
    for (const t of types) {
      if (!t.count) continue
      const list = by.get(t.group) ?? []
      list.push(t)
      by.set(t.group, list)
    }
    const order = [...GROUP_ORDER, ...[...by.keys()].filter((g) => !GROUP_ORDER.includes(g as never))]
    return order.filter((g) => by.has(g)).map((g) => ({ group: g, types: by.get(g)! }))
  }, [types])
  const total = types.reduce((n, t) => n + (t.count || 0), 0)

  return (
    <div className="flex flex-col gap-3" role="group" aria-label="Document types">
      <div className="flex flex-wrap gap-2">
        <button type="button" className={chip(!type && !group)} aria-pressed={!type && !group} onClick={() => set('group', null)}>
          All <span className="font-normal tabular-nums opacity-70">{total}</span>
        </button>
      </div>
      {groups.map(({ group: g, types: list }) => {
        const meta = groupMeta(g)
        const Icon = meta.icon
        const n = list.reduce((s, t) => s + t.count, 0)
        return (
          <div key={g} className="flex min-w-0 flex-wrap items-center gap-2">
            <button
              type="button"
              className={chip(group === g && !type)}
              aria-pressed={group === g && !type}
              onClick={() => set('group', group === g && !type ? null : g)}
            >
              <Icon className="size-4" aria-hidden style={{ color: group === g && !type ? undefined : meta.from }} />
              {meta.label} <span className="font-normal tabular-nums opacity-70">{n}</span>
            </button>
            {list.map((t) => (
              <button key={t.type} type="button" className={chip(type === t.type)} aria-pressed={type === t.type} onClick={() => set('type', type === t.type ? null : t.type)}>
                {t.label || docTypeLabel(t.type)} <span className="font-normal tabular-nums opacity-70">{t.count}</span>
              </button>
            ))}
          </div>
        )
      })}
    </div>
  )
}

export default function DocumentsPage() {
  usePageTitle('Documents')
  const [params, setParams] = useSearchParams()
  const type = params.get('type') ?? ''
  const group = params.get('group') ?? ''
  const q = (params.get('q') ?? '').trim()

  const types = useQuery({ queryKey: ['document-types'], queryFn: api.documents.types, staleTime: 60_000 })
  const query = useMemo(() => ({ type: type || undefined, group: type ? undefined : group || undefined, q: q || undefined }), [type, group, q])
  const fetchPage = useCallback((cursor: string | null) => api.documents.list({ ...query, cursor, limit: 40 }), [query])
  const list = useItemList<DocumentListItem>(['documents', query], fetchPage, { keyOf: (d) => d.item.uid })
  const items = useMemo(() => list.items.map((d) => d.item), [list.items])
  const viewer = useViewerParam(items, list.hasMore, list.loadMore, list.total)
  const sentinel = useInfiniteSentinel(list.loadMore, list.hasMore && !list.loading)

  const set = useCallback(
    (key: string, value: string | null) => {
      let p = withParam(params, key, value)
      if (key === 'group') p = withParam(p, 'type', null)
      setParams(p, { replace: true })
    },
    [params, setParams],
  )

  return (
    <div className="mx-auto w-full max-w-7xl px-4">
      <div className="pb-4 pt-6">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Documents</h1>
        <p className="mt-1 text-sm text-ink-soft" aria-live="polite">
          {list.loading && !list.items.length ? 'Loading' : formatCount(list.total, 'document')}
          {q ? ` matching "${q}"` : ''}
        </p>
      </div>
      <div className="flex flex-col gap-4 pb-4">
        <SearchBox value={q} onCommit={(next) => set('q', next || null)} placeholder="Search names, numbers and words in your documents" />
        {types.data && <TypeChips types={types.data} type={type} group={group} set={set} />}
      </div>

      {list.error && !list.items.length ? (
        <ErrorState error={list.error} onRetry={list.refetch} />
      ) : list.loading && !list.items.length ? (
        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3" aria-busy="true" aria-label="Loading documents">
          {Array.from({ length: 6 }, (_, i) => (
            <li key={i} className="h-40 animate-pulse rounded-2xl bg-surface-2" />
          ))}
        </ul>
      ) : list.items.length === 0 ? (
        q || type || group ? (
          <EmptyState icon={<FileSearch className="size-8" aria-hidden />} title="No documents match">
            Try other words, or a different type.
          </EmptyState>
        ) : (
          <EmptyState icon={<FileText className="size-8" aria-hidden />} title="No documents yet">
            Upload PDFs, office files or photos of IDs, bills and tickets. Their text is read and the details are filled in for you.
          </EmptyState>
        )
      ) : (
        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {list.items.map((d) => (
            <li key={d.item.uid} className="min-w-0">
              <DocumentCard item={d.item} doc={d.document} onOpen={() => viewer.open(d.item.uid)} />
            </li>
          ))}
        </ul>
      )}
      <LoadMore hasMore={list.hasMore} loading={list.loading} onClick={list.loadMore} sentinelRef={sentinel} />

      {viewer.uid && (
        <Viewer
          uid={viewer.uid}
          onClose={viewer.close}
          onPrev={viewer.onPrev}
          onNext={viewer.onNext}
          position={viewer.position}
          mode="document"
        />
      )}
    </div>
  )
}
