// /trash: items deleted in the last TRASH_DAYS days (api.trash.list), as a plain grid of thumbnails.
// Each item can be restored or deleted for good; "Empty trash" deletes everything here for good.
// Private (NSFW) items show a locked tile: their thumbnails are only served inside the vault.
// Keys on a computer (a fine pointer): the first tile is selected (its Restore focused, a ring on the tile) once the
// list is in; arrows, Home and End move between tiles; Enter or R restores; Delete or Backspace asks to delete for
// good with the dialog's button focused, so Enter confirms; Space opens a larger preview (TrashPreview) with the same
// keys. After an answer the tile now in its place is selected, or the same one again when the request failed.
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { useInfiniteQuery, useMutation, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { EyeOff, File as FileIcon, FileText, Film, RotateCcw, Trash2 } from 'lucide-react'
import { api } from '../lib/api'
import { formatDuration } from '../lib/format'
import { useRefreshLibrary } from '../lib/hooks'
import type { ItemCard } from '../lib/types'
import { Button, ConfirmModal, EmptyState, ErrorState, Kbd, PageSpinner } from '../components/ui'
import { useToast } from '../components/Toast'
import { usePageTitle } from '../components/shell/title'
import TrashPreview from '../components/trash/TrashPreview'

const PAGE = 60

function errorText(e: unknown, fallback: string): string {
  return e instanceof Error && e.message ? e.message : fallback
}

function Tile({
  item,
  busy,
  onRestore,
  onPurge,
}: {
  item: ItemCard
  /** Being answered, or answered and waiting to leave: its buttons do nothing (aria-disabled, not disabled, so focus
   *  and the keys stay on the tile and the arrows still move on). */
  busy: boolean
  onRestore: () => void
  onPurge: () => void
}) {
  const [broken, setBroken] = useState(false)
  const isDoc = item.kind === 'pdf' || item.kind === 'document'
  const Icon = isDoc ? FileText : item.kind === 'video' ? Film : FileIcon
  return (
    <li
      className={clsx(
        'flex flex-col overflow-hidden rounded-xl bg-surface shadow-[var(--shadow-card)]',
        // the selected tile (keys): a computer only, touch screens have no keys
        'pointer-fine:focus-within:ring-2 pointer-fine:focus-within:ring-accent pointer-fine:focus-within:ring-offset-2 pointer-fine:focus-within:ring-offset-bg',
      )}
    >
      <div className="relative aspect-square bg-surface-2">
        {item.nsfw ? (
          <div className="flex size-full flex-col items-center justify-center gap-1 text-ink-faint">
            <EyeOff className="size-6" aria-hidden />
            <span className="text-xs font-medium">Private</span>
          </div>
        ) : broken ? (
          <div className="flex size-full items-center justify-center text-ink-faint">
            <Icon className="size-7" aria-hidden />
          </div>
        ) : (
          <img
            src={item.urls.thumb}
            alt=""
            loading="lazy"
            decoding="async"
            onError={() => setBroken(true)}
            className={clsx('size-full', isDoc ? 'bg-white object-contain p-1' : 'object-cover')}
          />
        )}
        {item.kind === 'video' && item.duration_s ? (
          <span className="absolute bottom-1.5 right-1.5 rounded-md bg-black/65 px-1.5 py-0.5 text-[11px] font-semibold tabular-nums text-white">
            {formatDuration(item.duration_s)}
          </span>
        ) : null}
      </div>
      <p className="truncate px-2 pt-2 text-xs font-medium" title={item.filename}>
        {item.filename}
      </p>
      <div className="flex items-center gap-1 p-1">
        <button
          type="button"
          onClick={onRestore}
          aria-disabled={busy}
          data-trash-restore=""
          aria-keyshortcuts="Enter R"
          className="inline-flex h-10 flex-1 items-center justify-center gap-1.5 rounded-lg text-sm font-semibold text-ink hover:bg-ink/5 aria-disabled:opacity-50 pointer-fine:focus:outline-hidden"
        >
          <RotateCcw className="size-4" aria-hidden />
          Restore
          <span className="sr-only"> {item.filename}</span>
        </button>
        <button
          type="button"
          onClick={onPurge}
          aria-disabled={busy}
          aria-label={`Delete ${item.filename} forever`}
          aria-keyshortcuts="Delete"
          title="Delete forever"
          className="inline-flex size-10 shrink-0 items-center justify-center rounded-lg text-danger hover:bg-danger/10 aria-disabled:opacity-50"
        >
          <Trash2 className="size-4" aria-hidden />
        </button>
      </div>
    </li>
  )
}

/** An answered tile: its place, and the tiles before it then (some may leave too, which moves its place up). */
type Pos = { uid: string; index: number; before?: string[] }

const addTo = (s: ReadonlySet<string>, uid: string) => new Set(s).add(uid)
const dropFrom = (s: ReadonlySet<string>, uid: string) => {
  const next = new Set(s)
  next.delete(uid)
  return next
}

export default function TrashPage() {
  usePageTitle('Trash')
  const refreshLibrary = useRefreshLibrary()
  const toast = useToast()
  const config = useQuery({ queryKey: ['config'], queryFn: api.config, staleTime: 60_000 })
  const list = useInfiniteQuery({
    queryKey: ['trash'],
    queryFn: ({ pageParam }) => api.trash.list(pageParam, PAGE),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
  })
  const [purging, setPurging] = useState<ItemCard | null>(null)
  const purgingRef = useRef<ItemCard | null>(null)
  purgingRef.current = purging
  const [purgeByKey, setPurgeByKey] = useState(false)
  const [emptying, setEmptying] = useState(false)
  // Being restored or deleted, and answered but still listed until the list is fetched again (uid -> when): both keep
  // the tile's buttons disabled, so a quick second key or click cannot answer it twice.
  const [working, setWorking] = useState<ReadonlySet<string>>(() => new Set())
  const [gone, setGone] = useState<ReadonlyMap<string, number>>(() => new Map())

  const items = useMemo(() => {
    const seen = new Set<string>()
    const out: ItemCard[] = []
    for (const p of list.data?.pages ?? [])
      for (const it of p.items) {
        if (seen.has(it.uid)) continue
        seen.add(it.uid)
        out.push(it)
      }
    return out
  }, [list.data])
  const total = list.data?.pages[0]?.total ?? items.length
  const busy = (uid: string) => working.has(uid) || gone.has(uid)

  // An answered tile is enabled again once a list fetched after the answer is in: by then it has left, or it is back
  // (its restore was undone before the list came) and can be answered again.
  const fetchedAt = list.dataUpdatedAt
  useEffect(() => {
    setGone((g) => {
      const keep = [...g].filter(([uid, at]) => fetchedAt < at && items.some((i) => i.uid === uid))
      return keep.length === g.size ? g : new Map(keep)
    })
  }, [items, fetchedAt])

  // ---- keys: focus, selection and where it goes after an answer
  const gridRef = useRef<HTMLUListElement>(null)
  const after = useRef<Pos | null>(null)
  const refocus = useRef<Pos | null>(null)
  const initDone = useRef(false)
  const tabbed = useRef(false)
  const spaceHeld = useRef(false)
  const [preview, setPreview] = useState<Pos | null>(null)

  const { hasNextPage, isFetchingNextPage, fetchNextPage } = list
  const nearEnd = useCallback(
    (i: number) => {
      // cancelRefetch false: a next page must not cancel the list fetch after an answer (the old pages would stay)
      if (i >= items.length - 6 && hasNextPage && !isFetchingNextPage) void fetchNextPage({ cancelRefetch: false })
    },
    [items.length, hasNextPage, isFetchingNextPage, fetchNextPage],
  )
  const focusTile = useCallback(
    (i: number) => {
      const lis = Array.from(gridRef.current?.children ?? []) as HTMLElement[]
      if (!lis.length) return
      const at = Math.max(0, Math.min(i, lis.length - 1))
      lis[at].querySelector<HTMLButtonElement>('[data-trash-restore]')?.focus({ preventScroll: true })
      lis[at].scrollIntoView({ block: 'nearest' })
      nearEnd(at)
    },
    [nearEnd],
  )
  const tileIndex = (t: EventTarget | null): number => {
    const li = t instanceof Element ? t.closest('li') : null
    return li && li.parentElement === gridRef.current ? Array.prototype.indexOf.call(gridRef.current?.children, li) : -1
  }
  /** The tile one row down (1) or up (-1) from tile i, nearest in x; -1 when there is no such row. */
  const rowStep = (i: number, dir: 1 | -1): number => {
    const lis = Array.from(gridRef.current?.children ?? []) as HTMLElement[]
    const cur = lis[i]?.getBoundingClientRect()
    if (!cur) return -1
    const cx = cur.left + cur.width / 2
    let best = -1
    let bestTop = dir > 0 ? Infinity : -Infinity
    let bestDx = Infinity
    lis.forEach((li, j) => {
      const r = li.getBoundingClientRect()
      if (dir > 0 ? r.top <= cur.top + 4 : r.top >= cur.top - 4) return
      const dx = Math.abs(r.left + r.width / 2 - cx)
      const nearerRow = dir > 0 ? r.top < bestTop - 4 : r.top > bestTop + 4
      if (nearerRow || (Math.abs(r.top - bestTop) <= 4 && dx < bestDx)) {
        best = j
        bestTop = r.top
        bestDx = dx
      }
    })
    return best
  }

  useEffect(() => {
    // a Tab while the list loads: the person is moving focus themselves, so the first tile leaves it be
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Tab') tabbed.current = true
    }
    // Firefox turns the keyup of Space on a focused button into a click: the Space that opened the preview must not
    const onKeyUp = (e: globalThis.KeyboardEvent) => {
      if (e.key === ' ' && spaceHeld.current) {
        spaceHeld.current = false
        e.preventDefault()
      }
    }
    document.addEventListener('keydown', onKey, true)
    document.addEventListener('keyup', onKeyUp)
    return () => {
      document.removeEventListener('keydown', onKey, true)
      document.removeEventListener('keyup', onKeyUp)
    }
  }, [])

  useEffect(() => {
    if (!items.length || !finePointer()) return
    const done = after.current
    if (done && !items.some((i) => i.uid === done.uid)) {
      // The answered tile has left the list with the focus on it: the one now in its place is selected (not when the
      // person has moved on to another tile meanwhile). Tiles before it that left too move that place up, and a tile
      // still being answered there is passed over.
      after.current = null
      if (preview || !focusLost()) return
      const listed = new Set(items.map((i) => i.uid))
      let at = done.index - (done.before ?? []).filter((uid) => !listed.has(uid)).length
      while (at < items.length - 1 && busy(items[at].uid)) at++
      focusTile(at)
      return
    }
    if (!initDone.current) {
      initDone.current = true
      if (!tabbed.current && !focusTaken()) focusTile(0)
    }
  }, [items, preview, focusTile])

  useEffect(() => {
    // a failed answer: the same tile again if the focus went astray meanwhile (it normally stays on the tile)
    const r = refocus.current
    if (!r || working.has(r.uid)) return
    refocus.current = null
    const at = items.findIndex((i) => i.uid === r.uid)
    if (finePointer() && !preview && focusLost() && at >= 0) focusTile(at)
  }, [working, preview, focusTile])

  // restoring or purging changes people counts, person lists and review bins as well as the lists
  const refresh = refreshLibrary

  const restore = useMutation({
    mutationFn: (uid: string) => api.items.restore(uid),
    onMutate: (uid) => setWorking((w) => addTo(w, uid)),
    onSuccess: (_d, uid) => {
      setGone((g) => new Map(g).set(uid, Date.now()))
      const it = items.find((i) => i.uid === uid)
      toast.success(it ? `${it.filename} is back in the library.` : 'Restored.')
      void refresh()
    },
    onError: (e, uid) => {
      toast.error(errorText(e, 'Could not restore. Try again.'))
      if (after.current?.uid === uid) {
        refocus.current = after.current
        after.current = null
      }
    },
    onSettled: (_d, _e, uid) => setWorking((w) => dropFrom(w, uid)),
  })

  const purge = useMutation({
    mutationFn: (uid: string) => api.trash.purge(uid),
    onMutate: (uid) => setWorking((w) => addTo(w, uid)),
    onSuccess: (_d, uid) => {
      setGone((g) => new Map(g).set(uid, Date.now()))
      // the tile is still listed: the list effect selects the one now in its place once it leaves. A dialog opened
      // since for another item stays open.
      if (purgingRef.current?.uid === uid) closePurge(-1)
      toast.success('Deleted for good.')
      void refresh()
    },
    onError: (e, uid) => {
      // the dialog stays open for another try or Esc
      toast.error(errorText(e, 'Could not delete. Try again.'))
      if (after.current?.uid === uid) after.current = null
    },
    onSettled: (_d, _e, uid) => setWorking((w) => dropFrom(w, uid)),
  })

  const empty = useMutation({
    mutationFn: api.trash.empty,
    onSuccess: (r) => {
      setEmptying(false)
      toast.success(r.updated === 1 ? '1 item deleted for good.' : `${r.updated.toLocaleString()} items deleted for good.`)
      void refresh()
    },
    onError: (e) => toast.error(errorText(e, 'Could not empty the trash. Try again.')),
  })

  const doRestore = (item: ItemCard, index: number) => {
    if (busy(item.uid)) return
    after.current = { uid: item.uid, index, before: items.slice(0, index).map((i) => i.uid) }
    restore.mutate(item.uid)
  }
  const askPurge = (item: ItemCard, byKey: boolean) => {
    if (busy(item.uid)) return
    setPurgeByKey(byKey)
    setPurging(item)
  }
  // The delete dialog opened by a key focuses its own button first, so it cannot hand focus back by itself: the
  // preview under it takes focus again, else the tile (or, once a deleted one leaves, the one now in its place).
  const purgeReturn = useRef<number | null>(null)
  const closePurge = (at: number) => {
    purgeReturn.current = purgeByKey && finePointer() ? at : null
    setPurging(null)
  }
  useEffect(() => {
    // after the commit that removed the dialog (and ran its own focus hand-back)
    const at = purgeReturn.current
    if (purging || at === null) return
    purgeReturn.current = null
    const dialog = document.querySelector<HTMLElement>('[role="dialog"][aria-modal="true"]')
    if (dialog) dialog.focus()
    else if (at >= 0 && focusFree(gridRef.current)) focusTile(at)
  }, [purging, focusTile])

  // ---- the preview: the item by its uid, else the one now at its place (an answered item hands over to the next)
  const previewIndex = (() => {
    if (!preview || !items.length) return -1
    const i = items.findIndex((x) => x.uid === preview.uid)
    return i >= 0 ? i : Math.min(preview.index, items.length - 1)
  })()
  const shown = previewIndex >= 0 ? items[previewIndex] : null
  useEffect(() => {
    if (!preview) return
    if (!shown) setPreview(null)
    else if (shown.uid !== preview.uid || previewIndex !== preview.index) setPreview({ uid: shown.uid, index: previewIndex })
  }, [preview, shown, previewIndex])
  const openPreview = (i: number) => {
    const it = items[i]
    if (it) setPreview({ uid: it.uid, index: i })
  }
  const movePreview = (to: number) => {
    const it = items[to]
    if (!it) return
    setPreview({ uid: it.uid, index: to })
    nearEnd(to)
  }
  const previewReturn = useRef<number | null>(null)
  const closePreview = (bySpace = false) => {
    if (purging) return // Esc closes the delete dialog over it first
    // Firefox turns the keyup of Space on a focused button into a click: the tile's Restore takes focus before it comes
    if (bySpace) spaceHeld.current = true
    previewReturn.current = finePointer() && previewIndex >= 0 ? previewIndex : null
    setPreview(null)
  }
  useEffect(() => {
    // after the dialog has put focus back where it was opened: the tile of the item last shown
    const at = previewReturn.current
    if (preview || at === null) return
    previewReturn.current = null
    if (focusFree(gridRef.current)) focusTile(at)
  }, [preview, focusTile])

  const onGridKey = (e: KeyboardEvent<HTMLUListElement>) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return
    const i = tileIndex(e.target)
    const item = items[i]
    if (!item) return
    const k = e.key
    if (k === 'ArrowLeft' || k === 'ArrowRight' || k === 'ArrowUp' || k === 'ArrowDown' || k === 'Home' || k === 'End') {
      const to =
        k === 'Home'
          ? 0
          : k === 'End'
            ? items.length - 1
            : k === 'ArrowLeft'
              ? i - 1
              : k === 'ArrowRight'
                ? i + 1
                : rowStep(i, k === 'ArrowDown' ? 1 : -1)
      if (to < 0 || to >= items.length) {
        if (k === 'ArrowDown' || k === 'ArrowRight') nearEnd(items.length - 1)
        return
      }
      e.preventDefault()
      focusTile(to)
    } else if (k === ' ') {
      e.preventDefault() // not the focused button's click: Space opens the preview
      spaceHeld.current = true
      if (!e.repeat) openPreview(i)
    } else if (k === 'Enter' && e.repeat) {
      e.preventDefault() // a held Enter answers once
    } else if (k === 'r' || k === 'R') {
      e.preventDefault()
      if (!e.repeat) doRestore(item, i)
    } else if (k === 'Delete' || k === 'Backspace') {
      e.preventDefault()
      if (!e.repeat) askPurge(item, true)
    }
  }

  const days = config.data?.trash_days ?? 30
  const purgingIndex = purging ? items.findIndex((i) => i.uid === purging.uid) : -1

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-6">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-x-4 gap-y-3">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold tracking-tight">Trash</h1>
          <p className="mt-1 max-w-2xl text-sm text-ink-soft">
            Deleted items stay here for {days} days, then they and their files are removed for good. Restore puts an item
            back exactly as it was.
          </p>
        </div>
        {items.length > 0 && (
          <Button variant="danger" onClick={() => setEmptying(true)}>
            <Trash2 className="size-4" aria-hidden />
            Empty trash
          </Button>
        )}
      </div>

      {list.isPending ? (
        <PageSpinner />
      ) : list.isError ? (
        <ErrorState error={list.error} onRetry={() => void list.refetch()} />
      ) : items.length === 0 ? (
        <EmptyState icon={<Trash2 className="size-10" aria-hidden />} title="The trash is empty">
          Items you delete appear here for {days} days before they are removed for good.
        </EmptyState>
      ) : (
        <>
          <p className="mb-1 text-sm text-ink-soft">
            {total.toLocaleString()} {total === 1 ? 'item' : 'items'}
          </p>
          <p className="mb-3 hidden text-sm text-ink-soft pointer-fine:block">
            Keys: arrows move, <Kbd>Enter</Kbd> or <Kbd>R</Kbd> restores, <Kbd>Space</Kbd> shows it larger,{' '}
            <Kbd>Delete</Kbd> deletes forever (asks first), <Kbd>Ctrl</Kbd>+<Kbd>Z</Kbd> undoes a restore.
          </p>
          <ul
            ref={gridRef}
            onKeyDown={onGridKey}
            className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6"
          >
            {items.map((item, i) => (
              <Tile
                key={item.uid}
                item={item}
                busy={busy(item.uid)}
                onRestore={() => doRestore(item, i)}
                onPurge={() => askPurge(item, false)}
              />
            ))}
          </ul>
          {list.hasNextPage && (
            <div className="mt-6 flex justify-center">
              <Button variant="secondary" loading={list.isFetchingNextPage} onClick={() => void list.fetchNextPage({ cancelRefetch: false })}>
                Load more
              </Button>
            </div>
          )}
        </>
      )}

      {shown && (
        <TrashPreview
          item={shown}
          index={previewIndex}
          total={Math.max(total, items.length)}
          busy={busy(shown.uid)}
          keysOff={!!purging}
          onPrev={previewIndex > 0 ? () => movePreview(previewIndex - 1) : undefined}
          onNext={previewIndex < items.length - 1 ? () => movePreview(previewIndex + 1) : undefined}
          onRestore={() => doRestore(shown, previewIndex)}
          onPurge={(byKey) => askPurge(shown, byKey)}
          onClose={closePreview}
          prefetch={items[previewIndex + 1] && !items[previewIndex + 1].nsfw ? items[previewIndex + 1].urls.preview : null}
        />
      )}

      <ConfirmModal
        open={!!purging}
        title="Delete forever"
        body={
          <>
            <span className="font-semibold text-ink">{purging?.filename}</span> and all its files will be deleted for good.
            This cannot be undone.
          </>
        }
        confirmLabel="Delete forever"
        danger
        loading={!!purging && working.has(purging.uid)}
        focusConfirm={purgeByKey}
        onConfirm={() => {
          if (!purging || busy(purging.uid)) return
          after.current = { uid: purging.uid, index: purgingIndex, before: items.slice(0, Math.max(0, purgingIndex)).map((i) => i.uid) }
          purge.mutate(purging.uid)
        }}
        onClose={() => closePurge(purgingIndex)}
      />
      <ConfirmModal
        open={emptying}
        title="Empty the trash"
        body={`All ${total.toLocaleString()} ${total === 1 ? 'item' : 'items'} in the trash and their files will be deleted for good. This cannot be undone.`}
        confirmLabel="Empty trash"
        danger
        loading={empty.isPending}
        onConfirm={() => empty.mutate()}
        onClose={() => setEmptying(false)}
      />
    </div>
  )
}

/** A mouse or trackpad: keys and the selection ring; touch screens keep their taps only. */
function finePointer(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(pointer: fine)').matches
}

/** Focus is on the page itself or already in the grid: moving it to a tile takes nothing from the person. */
function focusFree(grid: HTMLElement | null): boolean {
  const a = document.activeElement
  return !a || a === document.body || !!grid?.contains(a)
}

/** The focused element went away (its tile left the list): focus sits on the page. */
function focusLost(): boolean {
  const a = document.activeElement
  return !a || a === document.body
}

/** Focus is somewhere the person put it (a field or a dialog): the first-tile focus leaves it there. */
function focusTaken(): boolean {
  const a = document.activeElement
  return (
    a instanceof HTMLElement &&
    (a.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(a.tagName) || !!a.closest('[role="dialog"]'))
  )
}
