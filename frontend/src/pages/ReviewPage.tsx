// Review (/review): the bins that need a decision. Mis-clicks (keep or delete), forwards (keep or
// delete, section 29), duplicates (pick the best copy, delete the others) and person guesses (confirm
// or reject), all of them or only those from clothes, or from tattoos, nails and mehndi (section 34).
// Guesses take keys on a computer (section 14): the first card's Yes is focused, Enter confirms, Space opens the
// "Someone else" name box, X rejects, and focus moves to the next card after each decision; the viewer opened from
// a card answers the same keys for that card's guess.
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useSearchParams } from 'react-router'
import { useMutation, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { CheckCheck, Copy, Crown, Forward, Sparkles, Trash2, TriangleAlert, UsersRound } from 'lucide-react'
import { api } from '../lib/api'
import type { DupGroup, GuessItem, ItemCard } from '../lib/types'
import { formatCount, formatDate, formatResolution } from '../lib/format'
import { useInfiniteSentinel, usePageTitle, useRefreshLibrary, useStableCallback } from '../lib/hooks'
import { useItemList } from '../lib/useItemList'
import MediaGrid from '../components/MediaGrid'
import Viewer from '../components/Viewer'
import { forwardReasons } from '../components/library/forwards'
import GuessCard from '../components/people/GuessCard'
import { guessKey, useGuessReview, type DecideOpts } from '../components/people/useGuessReview'
import { useToast } from '../components/Toast'
import { Confirm } from '../components/admin/AdminModal'
import { errorMessage, useSelection } from '../components/admin/adminHooks'
import { LoadMore, SelectionBar } from '../components/admin/fields'
import { useViewerParam } from '../components/public/useViewerParam'
import { Button, EmptyState, ErrorState, Kbd } from '../components/ui'

type Tab = 'misclicks' | 'forwards' | 'duplicates' | 'guesses'
const TABS: { value: Tab; label: string; icon: typeof Copy }[] = [
  { value: 'misclicks', label: 'Mis-clicks', icon: TriangleAlert },
  { value: 'forwards', label: 'Forwards', icon: Forward },
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

function ForwardsTab() {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const fetchPage = useCallback((c: string | null) => api.review.forwards(c, 60), [])
  const list = useItemList(['review', 'forwards'], fetchPage)
  const sel = useSelection<string>()
  const viewer = useViewerParam(list.items, list.hasMore, list.loadMore, list.total)
  const openItem = useStableCallback((uid: string) => viewer.open(uid))
  const [confirmDelete, setConfirmDelete] = useState(false)
  const reasonsOf = useCallback((i: ItemCard) => forwardReasons(i.forward_reasons), [])

  const bulk = useMutation({
    mutationFn: (action: 'unforward' | 'delete') => api.items.bulk({ uids: [...sel.selected], action }),
    onSuccess: (r, action) => {
      toast.success(`${formatCount(r.updated, 'item')} ${action === 'unforward' ? 'kept in the library' : 'moved to the trash'}.`)
      setConfirmDelete(false)
      sel.clear()
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not update the selection.')),
  })

  return (
    <div>
      <p className="mb-3 text-sm text-ink-soft">
        Memes, greetings, posters and text images that came through WhatsApp or another messaging app. A real photo received the
        same way stays in the library. Keep returns an item to the library and teaches the app; nothing is deleted until you say so.
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
              reasonsOf={reasonsOf}
              empty={
                <EmptyState icon={<CheckCheck className="size-8" aria-hidden />} title="No forwards to review">
                  Media received through a messaging app that looks made for sharing shows up here after upload.
                </EmptyState>
              }
            />
          </div>
        </>
      )}
      {sel.count > 0 && (
        <SelectionBar count={sel.count} noun={sel.count === 1 ? 'item' : 'items'} onClear={sel.clear} onSelectAll={() => sel.setAll(list.items.map((i) => i.uid))} allSelected={sel.count >= list.items.length} total={list.items.length}>
          <Button loading={bulk.isPending && bulk.variables === 'unforward'} disabled={bulk.isPending} onClick={() => bulk.mutate('unforward')}>
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

type Basis = 'all' | 'clothes' | 'trait' | 'face' | 'place'
const BASES: { value: Basis; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'clothes', label: 'Clothes' },
  { value: 'trait', label: 'Tattoos and nails' },
  { value: 'face', label: 'Trained face' },
  { value: 'place', label: 'Position' },
]

function GuessesTab() {
  const [basis, setBasis] = useState<Basis>('all')
  const fetchPage = useCallback((c: string | null) => api.review.guesses(c, 60, undefined, basis), [basis])
  const list = useItemList<GuessItem>(basis === 'all' ? ['review', 'guesses'] : ['review', 'guesses', 'basis', basis], fetchPage, {
    keyOf: guessKey,
  })
  const review = useGuessReview()
  const items = useMemo(() => {
    const seen = new Set<string>()
    return list.items.map((g) => g.item).filter((i) => (seen.has(i.uid) ? false : (seen.add(i.uid), true)))
  }, [list.items])
  const sentinel = useInfiniteSentinel(list.loadMore, list.hasMore && !list.loading)
  const [confirmAll, setConfirmAll] = useState(false)
  const viewer = useViewerParam(items, list.hasMore, list.loadMore)

  // Keys (a computer only): the first card's Yes takes focus once per filter (not after a Tab while the list loads:
  // the person is moving focus themselves), and after a decision the card now in the decided one's place does, once
  // the decided card has left the list (its buttons are disabled meanwhile, so focus sits on the page and a second
  // Enter cannot answer twice). A viewer closed after its card was decided there hands focus to that card's place.
  const listRef = useRef<HTMLUListElement>(null)
  const after = useRef<{ key: string; index: number } | null>(null)
  const initFor = useRef<Basis | null>(null)
  const tabbed = useRef(false)
  const wasOpen = useRef(false)
  const openedAt = useRef(0)
  const [openedFor, setOpenedFor] = useState<string | null>(null)
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Tab') tabbed.current = true
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [])
  const focusYesAt = (i: number) => {
    const ys = listRef.current?.querySelectorAll<HTMLButtonElement>('[data-guess-yes]')
    const y = ys?.[Math.min(i, ys.length - 1)]
    y?.focus({ preventScroll: true })
    y?.closest('li')?.scrollIntoView({ block: 'nearest' })
  }
  useEffect(() => {
    const closed = wasOpen.current && !viewer.uid
    wasOpen.current = !!viewer.uid
    if (!list.items.length || viewer.uid || !finePointer()) return
    const decided = after.current
    if (decided && !list.items.some((g) => guessKey(g) === decided.key)) {
      after.current = null
      if (focusFree(listRef.current)) focusYesAt(decided.index)
      return
    }
    if (initFor.current !== basis) {
      initFor.current = basis
      if (!tabbed.current && !focusTaken()) focusYesAt(0)
      return
    }
    if (closed && document.activeElement === document.body) focusYesAt(openedAt.current)
  }, [list.items, basis, viewer.uid])
  // A failed decision keeps its card: its Yes takes focus back once its buttons are enabled again, which is the render
  // after the error (focus() on a disabled button does nothing).
  const refocusAt = useRef<number | null>(null)
  useEffect(() => {
    if (review.pending || refocusAt.current === null) return
    const i = refocusAt.current
    refocusAt.current = null
    if (finePointer() && focusFree(listRef.current)) focusYesAt(i)
  }, [review.pending])
  const decide = (g: GuessItem, i: number): DecideOpts => {
    after.current = { key: guessKey(g), index: i }
    return {
      onError: () => {
        after.current = null
        refocusAt.current = i
      },
    }
  }

  return (
    <div>
      <p className="mb-3 text-sm text-ink-soft">
        People guessed from someone tagged within an hour in the same scene and clothes, from standing in the same spot in
        a photo taken moments apart, from the same tattoo in a photo taken at any time, or by recognition trained on one
        person. Confirming tags them; rejecting stops that guess.
      </p>
      <p className="mb-3 hidden text-sm text-ink-soft pointer-fine:block">
        Keys: <Kbd>Enter</Kbd> says yes to the selected guess, <Kbd>Space</Kbd> opens the name box for someone else,{' '}
        <Kbd>X</Kbd> says no, <Kbd>Esc</Kbd> closes the name box.
      </p>
      <div role="group" aria-label="Show guesses made from" className="mb-3 flex flex-wrap gap-1.5">
        {BASES.map((b) => (
          <button
            key={b.value}
            type="button"
            aria-pressed={basis === b.value}
            onClick={() => {
              tabbed.current = false
              setBasis(b.value)
            }}
            className={clsx(
              'h-10 rounded-full border px-4 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
              basis === b.value ? 'border-ink bg-ink text-bg' : 'border-line text-ink-soft hover:bg-ink/5',
            )}
          >
            {b.label}
          </button>
        ))}
      </div>
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
        <EmptyState icon={<Sparkles className="size-8" aria-hidden />} title={basis === 'all' ? 'No guesses to review' : 'No guesses of this kind'}>
          Tag someone in a photo: other photos within an hour with the same clothes, photos taken moments apart where someone
          stands in the same spot, and photos taken at any time with the same tattoo, get a guess. Recognition trained on one
          person (on their page) adds the faces it finds.
        </EmptyState>
      ) : (
        <ul ref={listRef} className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {list.items.map((g, i) => (
            <li key={guessKey(g)}>
              <GuessCard
                keys
                g={g}
                busy={review.busyFor(g)}
                onConfirm={() => review.confirm([g], decide(g, i))}
                onReject={() => review.reject([g], decide(g, i))}
                onCorrect={(choice) => review.correct([g], choice, decide(g, i))}
                onOpenItem={(uid, t) => {
                  setOpenedFor(guessKey(g))
                  openedAt.current = i
                  viewer.open(uid, t)
                }}
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
        <Viewer
          uid={viewer.uid}
          onClose={viewer.close}
          onPrev={viewer.onPrev}
          onNext={viewer.onNext}
          position={viewer.position}
          guessKeys
          openedFor={openedFor}
        />
      )}
    </div>
  )
}

/** A mouse or trackpad: keys and the focus ring; touch screens keep their taps only. */
function finePointer(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(pointer: fine)').matches
}

/** Focus is on the page itself or already in the guess list: moving it to a card takes nothing from the person. */
function focusFree(list: HTMLElement | null): boolean {
  const a = document.activeElement
  return !a || a === document.body || !!list?.contains(a)
}

/** Focus is somewhere the person put it (a field, a dialog, a card): the first-card focus leaves it there. */
function focusTaken(): boolean {
  const a = document.activeElement
  return (
    a instanceof HTMLElement &&
    (a.isContentEditable ||
      ['INPUT', 'TEXTAREA', 'SELECT'].includes(a.tagName) ||
      !!a.closest('[role="dialog"]') ||
      !!a.closest('article'))
  )
}

export default function ReviewPage() {
  usePageTitle('Review')
  const [params, setParams] = useSearchParams()
  const raw = params.get('tab')
  const tab: Tab = raw === 'forwards' || raw === 'duplicates' || raw === 'guesses' ? raw : 'misclicks'
  const stats = useQuery({ queryKey: ['stats'], queryFn: api.stats, staleTime: 15_000 })
  const counts: Record<Tab, number | undefined> = {
    misclicks: stats.data?.misclicks,
    forwards: stats.data?.forwards,
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
        {tab === 'forwards' && <ForwardsTab />}
        {tab === 'duplicates' && <DuplicatesTab />}
        {tab === 'guesses' && <GuessesTab />}
      </div>
    </div>
  )
}
