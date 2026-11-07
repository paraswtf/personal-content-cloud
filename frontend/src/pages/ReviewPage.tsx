// Review (/review): the bins that need a decision. Mis-clicks (keep or delete), duplicates (pick
// the best copy, delete the others) and person guesses (confirm or reject).
import { useCallback, useMemo, useState, type ReactNode } from 'react'
import { useSearchParams } from 'react-router'
import { useMutation, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { CheckCheck, Copy, Crown, Sparkles, Trash2, TriangleAlert, UsersRound } from 'lucide-react'
import { api } from '../lib/api'
import type { DupGroup, GuessItem, ItemCard } from '../lib/types'
import { formatCount, formatDate, formatResolution } from '../lib/format'
import { useInfiniteSentinel, usePageTitle, useRefreshLibrary, useStableCallback } from '../lib/hooks'
import { useItemList } from '../lib/useItemList'
import MediaGrid from '../components/MediaGrid'
import Viewer from '../components/Viewer'
import GuessCard from '../components/people/GuessCard'
import { guessKey, useGuessReview } from '../components/people/useGuessReview'
import { useToast } from '../components/Toast'
import { Confirm } from '../components/admin/AdminModal'
import { errorMessage, useSelection } from '../components/admin/adminHooks'
import { LoadMore, SelectionBar } from '../components/admin/fields'
import { useViewerParam } from '../components/public/useViewerParam'
import { Button, EmptyState, ErrorState } from '../components/ui'

type Tab = 'misclicks' | 'duplicates' | 'guesses'
const TABS: { value: Tab; label: string; icon: typeof Copy }[] = [
  { value: 'misclicks', label: 'Mis-clicks', icon: TriangleAlert },
  { value: 'duplicates', label: 'Duplicates', icon: Copy },
  { value: 'guesses', label: 'Guesses', icon: UsersRound },
]

function MisclicksTab() {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const fetchPage = useCallback((c: string | null) => api.review.misclicks(c, 60), [])
  const list = useItemList(['review', 'misclicks'], fetchPage)
  const sel = useSelection<string>()
  const viewer = useViewerParam(list.items, list.hasMore, list.loadMore, list.total)
  const openItem = useStableCallback((uid: string) => viewer.open(uid))
  const [confirmDelete, setConfirmDelete] = useState(false)

  const bulk = useMutation({
    mutationFn: (action: 'keep' | 'delete') => api.items.bulk({ uids: [...sel.selected], action }),
    onSuccess: (r, action) => {
      toast.success(`${formatCount(r.updated, 'item')} ${action === 'keep' ? 'kept' : 'moved to the trash'}.`)
      setConfirmDelete(false)
      sel.clear()
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not update the selection.')),
  })

  return (
    <div>
      <p className="mb-3 text-sm text-ink-soft">
        Blurry, dark, blown-out, pocket and burst shots. Nothing here is deleted until you say so. Keep returns an item to the library.
      </p>
      {list.error && !list.items.length ? (
        <ErrorState error={list.error} onRetry={list.refetch} />
      ) : (
        <>
          {list.items.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-2">
              <Button variant="secondary" onClick={() => sel.setAll(list.items.map((i) => i.uid))}>
                Select all {list.items.length < list.total ? 'loaded' : ''}
              </Button>
            </div>
          )}
          <div className="-mx-4 sm:mx-0">
            <MediaGrid
              items={list.items}
              onOpen={openItem}
              hasMore={list.hasMore}
              loadMore={list.loadMore}
              loading={list.loading}
              selectable={sel.count > 0}
              selected={sel.selected}
              onToggleSelect={sel.toggle}
              selectOnHover
              showReasons
              empty={
                <EmptyState icon={<CheckCheck className="size-8" aria-hidden />} title="No mis-clicks to review">
                  Shots that look accidental show up here after upload.
                </EmptyState>
              }
            />
          </div>
        </>
      )}
      {sel.count > 0 && (
        <SelectionBar count={sel.count} noun={sel.count === 1 ? 'item' : 'items'} onClear={sel.clear} onSelectAll={() => sel.setAll(list.items.map((i) => i.uid))} allSelected={sel.count >= list.items.length} total={list.items.length}>
          <Button loading={bulk.isPending && bulk.variables === 'keep'} disabled={bulk.isPending} onClick={() => bulk.mutate('keep')}>
            Keep
          </Button>
          <Button variant="danger" disabled={bulk.isPending} onClick={() => setConfirmDelete(true)}>
            <Trash2 className="size-4" aria-hidden />
            Delete
          </Button>
        </SelectionBar>
      )}
      <Confirm
        open={confirmDelete}
        title={`Move ${formatCount(sel.count, 'item')} to the trash?`}
        body="They can be restored from Trash until it is emptied."
        confirmLabel="Move to trash"
        danger
        loading={bulk.isPending}
        onConfirm={() => bulk.mutate('delete')}
        onClose={() => !bulk.isPending && setConfirmDelete(false)}
      />
      {viewer.uid && (
        <Viewer uid={viewer.uid} onClose={viewer.close} onPrev={viewer.onPrev} onNext={viewer.onNext} position={viewer.position} />
      )}
    </div>
  )
}

function DupTile({
  item,
  best,
  onOpen,
  children,
}: {
  item: ItemCard
  best: boolean
  onOpen: () => void
  children?: ReactNode
}) {
  return (
    <li className="flex w-36 shrink-0 flex-col gap-1.5 sm:w-44">
      <button
        type="button"
        onClick={onOpen}
        className={clsx(
          'relative block aspect-square w-full overflow-hidden rounded-xl bg-surface-2 ring-1',
          best ? 'ring-2 ring-ok' : 'ring-line hover:ring-ink-faint',
        )}
        aria-label={`Open ${best ? 'the best copy' : 'this copy'}`}
      >
        <img src={item.urls.thumb} alt="" loading="lazy" className="size-full object-cover" />
        {best && (
          <span className="absolute left-1.5 top-1.5 inline-flex items-center gap-1 rounded-full bg-ok px-2 py-0.5 text-[11px] font-semibold text-white">
            <Crown className="size-3" aria-hidden /> Best
          </span>
        )}
      </button>
      <p className="truncate text-xs text-ink-soft" title={item.filename}>
        {formatResolution(item.width, item.height) || item.filename}
      </p>
      <p className="text-xs text-ink-faint">{formatDate(item.taken_at ?? item.created_at)}</p>
      {children}
    </li>
  )
}

function DuplicatesTab() {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const fetchPage = useCallback(
    (c: string | null) => api.review.duplicates(c, 30).then((p) => ({ items: p.groups, next_cursor: p.next_cursor, total: p.total })),
    [],
  )
  const list = useItemList<DupGroup>(['review', 'duplicates'], fetchPage, { keyOf: (g) => g.best.uid })
  const flat = useMemo(() => list.items.flatMap((g) => [g.best, ...g.copies]), [list.items])
  const viewer = useViewerParam(flat, list.hasMore, list.loadMore)
  const sentinel = useInfiniteSentinel(list.loadMore, list.hasMore && !list.loading)
  const [deleting, setDeleting] = useState<ItemCard | null>(null)

  const setBest = useMutation({
    mutationFn: (uid: string) => api.review.setBest(uid),
    onSuccess: () => {
      toast.success('Best copy changed.')
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not change the best copy.')),
  })
  const remove = useMutation({
    mutationFn: (uid: string) => api.items.remove(uid),
    onSuccess: () => {
      toast.success('Copy moved to the trash.')
      setDeleting(null)
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not delete the copy.')),
  })

  return (
    <div>
      <p className="mb-3 text-sm text-ink-soft">
        Near-identical pictures are grouped. The library shows only the best copy; the others stay here, kept at full quality, until you delete them.
      </p>
      {list.error && !list.items.length ? (
        <ErrorState error={list.error} onRetry={list.refetch} />
      ) : list.loading && !list.items.length ? (
        <div className="h-48 animate-pulse rounded-2xl bg-surface-2" aria-busy="true" />
      ) : list.items.length === 0 ? (
        <EmptyState icon={<Copy className="size-8" aria-hidden />} title="No duplicates">
          Similar copies of the same picture show up here.
        </EmptyState>
      ) : (
        <ul className="flex flex-col gap-3">
          {list.items.map((g) => (
            <li key={g.best.uid} className="rounded-2xl border border-line bg-surface p-3">
              <p className="mb-2 text-sm font-semibold">{formatCount(g.copies.length + 1, 'copy', 'copies')}</p>
              <ul className="flex gap-3 overflow-x-auto pb-1">
                <DupTile item={g.best} best onOpen={() => viewer.open(g.best.uid)} />
                {g.copies.map((c) => (
                  <DupTile key={c.uid} item={c} best={false} onOpen={() => viewer.open(c.uid)}>
                    <div className="flex flex-col gap-1">
                      <Button variant="secondary" disabled={setBest.isPending} loading={setBest.isPending && setBest.variables === c.uid} onClick={() => setBest.mutate(c.uid)}>
                        Make best
                      </Button>
                      <Button variant="danger" onClick={() => setDeleting(c)}>
                        Delete copy
                      </Button>
                    </div>
                  </DupTile>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      )}
      <LoadMore hasMore={list.hasMore} loading={list.loading} onClick={list.loadMore} sentinelRef={sentinel} />
      <Confirm
        open={deleting !== null}
        title="Delete this copy?"
        body="It moves to the trash. The best copy stays in the library."
        confirmLabel="Move to trash"
        danger
        loading={remove.isPending}
        onConfirm={() => deleting && remove.mutate(deleting.uid)}
        onClose={() => !remove.isPending && setDeleting(null)}
      />
      {viewer.uid && (
        <Viewer uid={viewer.uid} onClose={viewer.close} onPrev={viewer.onPrev} onNext={viewer.onNext} position={viewer.position} />
      )}
    </div>
  )
}

function GuessesTab() {
  const fetchPage = useCallback((c: string | null) => api.review.guesses(c, 60), [])
  const list = useItemList<GuessItem>(['review', 'guesses'], fetchPage, { keyOf: guessKey })
  const review = useGuessReview()
  const items = useMemo(() => {
    const seen = new Set<string>()
    return list.items.map((g) => g.item).filter((i) => (seen.has(i.uid) ? false : (seen.add(i.uid), true)))
  }, [list.items])
  const sentinel = useInfiniteSentinel(list.loadMore, list.hasMore && !list.loading)
  const [confirmAll, setConfirmAll] = useState(false)
  const viewer = useViewerParam(items, list.hasMore, list.loadMore)

  return (
    <div>
      <p className="mb-3 text-sm text-ink-soft">
        People guessed from someone tagged on the same day, in the same scene and wearing the same clothes. Confirming tags them; rejecting stops that guess.
      </p>
      {list.items.length > 1 && (
        <div className="mb-3 flex flex-wrap gap-2">
          <Button variant="secondary" disabled={review.pending} onClick={() => setConfirmAll(true)}>
            <CheckCheck className="size-4" aria-hidden />
            Confirm all {list.items.length} shown
          </Button>
        </div>
      )}
      {list.error && !list.items.length ? (
        <ErrorState error={list.error} onRetry={list.refetch} />
      ) : list.loading && !list.items.length ? (
        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3" aria-busy="true">
          {Array.from({ length: 6 }, (_, i) => (
            <li key={i} className="h-44 animate-pulse rounded-2xl bg-surface-2" />
          ))}
        </ul>
      ) : list.items.length === 0 ? (
        <EmptyState icon={<Sparkles className="size-8" aria-hidden />} title="No guesses to review">
          Tag someone in a photo and other photos from that day with the same clothes get a guess.
        </EmptyState>
      ) : (
        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {list.items.map((g) => (
            <li key={guessKey(g)}>
              <GuessCard
                g={g}
                busy={review.busyFor(g)}
                onConfirm={() => review.confirm([g])}
                onReject={() => review.reject([g])}
                onOpenItem={(uid) => viewer.open(uid)}
              />
            </li>
          ))}
        </ul>
      )}
      <LoadMore hasMore={list.hasMore} loading={list.loading} onClick={list.loadMore} sentinelRef={sentinel} />
      <Confirm
        open={confirmAll}
        title={`Confirm ${formatCount(list.items.length, 'guess', 'guesses')}?`}
        body="Every guess shown is tagged as the guessed person. You can still change a tag later from the photo."
        confirmLabel="Confirm all"
        loading={review.pending}
        onConfirm={() => {
          review.confirm(list.items)
          setConfirmAll(false)
        }}
        onClose={() => setConfirmAll(false)}
      />
      {viewer.uid && (
        <Viewer uid={viewer.uid} onClose={viewer.close} onPrev={viewer.onPrev} onNext={viewer.onNext} position={viewer.position} />
      )}
    </div>
  )
}

export default function ReviewPage() {
  usePageTitle('Review')
  const [params, setParams] = useSearchParams()
  const raw = params.get('tab')
  const tab: Tab = raw === 'duplicates' || raw === 'guesses' ? raw : 'misclicks'
  const stats = useQuery({ queryKey: ['stats'], queryFn: api.stats, staleTime: 15_000 })
  const counts: Record<Tab, number | undefined> = {
    misclicks: stats.data?.misclicks,
    duplicates: stats.data?.duplicates,
    guesses: stats.data?.guesses,
  }

  const setTab = (t: Tab) => {
    const p = new URLSearchParams()
    if (t !== 'misclicks') p.set('tab', t)
    setParams(p, { replace: true })
  }

  return (
    <div className="mx-auto w-full max-w-7xl px-4">
      <div className="pb-4 pt-6">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Review</h1>
        <p className="mt-1 text-sm text-ink-soft">Decisions the app left to you.</p>
      </div>
      <div role="tablist" aria-label="Review bins" className="-mx-4 mb-4 flex gap-1 overflow-x-auto border-b border-line px-4 sm:mx-0 sm:px-0">
        {TABS.map((t) => {
          const Icon = t.icon
          const n = counts[t.value]
          return (
            <button
              key={t.value}
              type="button"
              role="tab"
              aria-selected={tab === t.value}
              onClick={() => setTab(t.value)}
              className={clsx(
                '-mb-px inline-flex h-11 shrink-0 items-center gap-2 border-b-2 px-3 text-sm font-semibold transition-colors',
                tab === t.value ? 'border-accent text-ink' : 'border-transparent text-ink-soft hover:text-ink',
              )}
            >
              <Icon className="size-4" aria-hidden />
              {t.label}
              {n !== undefined && n > 0 && <span className="rounded-full bg-surface-2 px-2 py-0.5 text-xs tabular-nums">{n.toLocaleString()}</span>}
            </button>
          )
        })}
      </div>
      <div role="tabpanel" className="pb-8">
        {tab === 'misclicks' && <MisclicksTab />}
        {tab === 'duplicates' && <DuplicatesTab />}
        {tab === 'guesses' && <GuessesTab />}
      </div>
    </div>
  )
}
