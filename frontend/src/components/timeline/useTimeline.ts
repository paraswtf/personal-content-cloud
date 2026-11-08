// The timeline's data (docs/ARCHITECTURE.md section 23): the month skeleton (['timeline', 'buckets',
// query]) and the cards of the months that were asked for, loaded page by page (['timeline',
// 'bucket', query, key, page], pages of BUCKET_PAGE cards in library order, each page's cursor
// coming from the page before it). The grid asks for the months on screen and how many of their
// cards it needs; the viewer's neighbours are kept loaded so prev and next walk every month; months
// far from the screen stop being observed once more than MAX_MONTHS are held, and their pages leave
// the cache after GC_MS. Both key prefixes are in LIBRARY_KEYS, so bulk actions and viewer edits
// refetch them.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { keepPreviousData, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../../lib/api'
import type { ItemCard, ItemPage } from '../../lib/types'
import { BUCKET_PAGE, bucketKeyOf, fetchBucketPage, fetchBuckets, type TimelineBucket, type TimelineQuery } from '../../lib/timeline'

export interface MonthRequest {
  key: string
  /** How many cards from the start of the month are wanted (the grid's last visible index plus one). */
  need: number
}

export interface TimelineFocus {
  key: string
  uid: string
}

export interface TimelineState {
  buckets: TimelineBucket[]
  total: number
  /** The skeleton is loading and nothing is known yet. */
  loading: boolean
  error: unknown
  refetch: () => void
  /** The loaded cards of a month, from its start; undefined before the first page arrives. */
  cards: (key: string) => ItemCard[] | undefined
  /** Every card of the month is loaded. */
  complete: (key: string) => boolean
  /** True while a page of the month is loading. */
  pending: (key: string) => boolean
  /** Ask for months and how much of them; the grid calls this for the months on screen. */
  request: (reqs: MonthRequest[]) => void
  /** Loaded cards in timeline order, for the viewer and the bulk bar. */
  items: ItemCard[]
  /** Some month is not fully loaded yet. */
  hasMore: boolean
  /** Load the next page after the open item's month (or the first unloaded month). */
  loadMore: () => void
  /** A loaded card's index in the whole library (0 = the newest card). */
  positionOf: (uid: string) => number | null
  /** The open item's month, resolved from the server when it was not loaded: scroll there, then clear. */
  focus: TimelineFocus | null
  clearFocus: () => void
}

const NO_BUCKETS: TimelineBucket[] = []
/** Months kept observed at most (the screen's plus the viewer's neighbourhood always stay). */
const MAX_MONTHS = 24
const GC_MS = 10 * 60_000
/** Prev and next prefetch when the open card is this close to the loaded edge of its month. */
const EDGE = 3

interface Wanted {
  key: string
  pages: number
}

function pagesFor(need: number): number {
  return Math.max(1, Math.ceil(need / BUCKET_PAGE))
}

export function useTimeline(query: TimelineQuery, enabled: boolean, openUid: string | null): TimelineState {
  const qc = useQueryClient()
  const querySig = JSON.stringify(query)
  const bucketsQ = useQuery({
    queryKey: ['timeline', 'buckets', query],
    queryFn: ({ signal }) => fetchBuckets(query, signal),
    enabled,
    placeholderData: keepPreviousData,
  })
  const buckets = bucketsQ.data?.buckets ?? NO_BUCKETS
  const total = bucketsQ.data?.total ?? 0

  // the first card index of every month, and the month order
  const { firstIndex, order } = useMemo(() => {
    const first = new Map<string, number>()
    const idx = new Map<string, number>()
    let n = 0
    buckets.forEach((b, i) => {
      first.set(b.key, n)
      idx.set(b.key, i)
      n += b.count
    })
    return { firstIndex: first, order: idx }
  }, [buckets])

  // months wanted, oldest request first (re-requesting moves a month to the end); a new query starts over
  const [wanted, setWanted] = useState<Wanted[]>([])
  const lastSig = useRef(querySig)
  if (lastSig.current !== querySig) {
    lastSig.current = querySig
    setWanted([])
  }
  // months that must stay observed: the open card's month and its neighbours (set below each render)
  const pinned = useRef<Set<string>>(new Set())
  const request = useCallback((reqs: MonthRequest[]) => {
    if (reqs.length === 0) return
    setWanted((prev) => {
      const byKey = new Map(prev.map((w) => [w.key, w.pages]))
      let changed = false
      const touched = new Set<string>()
      for (const r of reqs) {
        touched.add(r.key)
        const pages = pagesFor(r.need)
        const have = byKey.get(r.key)
        if (have === undefined || have < pages) {
          byKey.set(r.key, Math.max(have ?? 0, pages))
          changed = true
        }
      }
      // keep the touched months most recent
      const kept = prev.filter((w) => !touched.has(w.key))
      if (kept.length !== prev.length) changed = true
      let next: Wanted[] = [...kept, ...reqs.map((r) => ({ key: r.key, pages: byKey.get(r.key) ?? 1 }))]
      // forget months far from the screen and the viewer when too many are held
      if (next.length > MAX_MONTHS) {
        const drop = next.length - MAX_MONTHS
        let dropped = 0
        next = next.filter((w) => {
          if (dropped >= drop || touched.has(w.key) || pinned.current.has(w.key)) return true
          dropped++
          return false
        })
        if (dropped) changed = true
      }
      return changed ? next : prev
    })
  }, [])

  // one query per (month, page); a page beyond the first waits for the page before it. Built on
  // every render (not memoised): a page arriving re-renders through useQueries, and the cache read
  // then marks the next page ready.
  const pageQueries: { key: string; page: number; cursor: string | null; ready: boolean }[] = []
  for (const w of wanted) {
    let cursor: string | null = null
    for (let page = 0; page < w.pages; page++) {
      if (page > 0) {
        const prev = qc.getQueryData<ItemPage>(['timeline', 'bucket', query, w.key, page - 1])
        if (!prev) {
          pageQueries.push({ key: w.key, page, cursor: null, ready: false })
          break
        }
        if (!prev.next_cursor) break // the month ended on the page before
        cursor = prev.next_cursor
      }
      pageQueries.push({ key: w.key, page, cursor, ready: true })
    }
  }

  const results = useQueries({
    queries: pageQueries.map((p) => ({
      queryKey: ['timeline', 'bucket', query, p.key, p.page],
      queryFn: ({ signal }: { signal: AbortSignal }) => fetchBucketPage(query, p.key, p.cursor, signal),
      enabled: enabled && p.ready,
      staleTime: 60_000,
      gcTime: GC_MS,
    })),
  })
  // react-query hands out a new results array on every render: rebuild the maps only when data moved
  const sig = pageQueries.map((p) => `${p.key}/${p.page}`).join(',') + '|' + results.map((r) => `${r.dataUpdatedAt}:${r.isPending ? 1 : 0}`).join(',')
  const resultsRef = useRef(results)
  resultsRef.current = results
  const months = useMemo(() => {
    const out = new Map<string, { cards: ItemCard[]; complete: boolean; pending: boolean; started: boolean }>()
    pageQueries.forEach((p, i) => {
      const r = resultsRef.current[i]
      let m = out.get(p.key)
      if (!m) {
        m = { cards: [], complete: false, pending: false, started: false }
        out.set(p.key, m)
      }
      if (!r) return
      if (r.data && m.cards.length === p.page * BUCKET_PAGE) {
        // pages are contiguous from the start; a later page without its predecessor is not counted
        m.started = true
        m.cards = m.cards.concat(r.data.items)
        if (!r.data.next_cursor) m.complete = true
      } else if (r.isPending || !p.ready) m.pending = true
    })
    return out
    // sig covers the page list and every result's data stamp
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig])

  const items = useMemo(() => {
    const out: ItemCard[] = []
    for (const b of buckets) {
      const m = months.get(b.key)
      if (m && m.started) out.push(...m.cards)
    }
    return out
  }, [buckets, months])

  const countOf = useCallback((key: string) => buckets[order.get(key) ?? -1]?.count ?? 0, [buckets, order])
  const isComplete = useCallback(
    (key: string) => {
      const m = months.get(key)
      return !!m && (m.complete || (m.started && m.cards.length >= countOf(key)))
    },
    [months, countOf],
  )
  const hasMore = buckets.some((b) => !isComplete(b.key))

  const locate = useCallback(
    (uid: string): { key: string; index: number } | null => {
      for (const [key, m] of months) {
        const i = m.cards.findIndex((c) => c.uid === uid)
        if (i >= 0) return { key, index: i }
      }
      return null
    },
    [months],
  )
  const positionOf = useCallback(
    (uid: string): number | null => {
      const at = locate(uid)
      if (!at) return null
      const first = firstIndex.get(at.key)
      return first === undefined ? null : first + at.index
    },
    [locate, firstIndex],
  )

  // the open card's month and neighbours stay observed, and the pages around the card are fetched
  const openAt = openUid ? locate(openUid) : null
  const openOrder = openAt ? (order.get(openAt.key) ?? -1) : -1
  const pins = new Set<string>()
  if (openAt) {
    pins.add(openAt.key)
    const prev = buckets[openOrder - 1]
    const next = buckets[openOrder + 1]
    if (prev) pins.add(prev.key)
    if (next) pins.add(next.key)
  }
  pinned.current = pins
  useEffect(() => {
    if (!openAt) return
    const m = months.get(openAt.key)
    if (!m) return
    const reqs: MonthRequest[] = []
    if (openAt.index >= m.cards.length - EDGE) {
      if (!isComplete(openAt.key)) reqs.push({ key: openAt.key, need: m.cards.length + BUCKET_PAGE })
      else {
        const next = buckets[openOrder + 1]
        if (next && !months.get(next.key)?.started) reqs.push({ key: next.key, need: BUCKET_PAGE })
      }
    }
    if (openAt.index <= EDGE - 1) {
      // prev lands on the newer month's last card, so that month is needed in full
      const prev = buckets[openOrder - 1]
      if (prev && !isComplete(prev.key)) reqs.push({ key: prev.key, need: prev.count })
    }
    if (reqs.length) request(reqs)
  }, [openAt?.key, openAt?.index, openOrder, buckets, months, isComplete, request]) // eslint-disable-line react-hooks/exhaustive-deps

  const loadMore = useCallback(() => {
    const from = openOrder >= 0 ? openOrder : 0
    const here = buckets[from]
    if (here && !isComplete(here.key)) {
      request([{ key: here.key, need: (months.get(here.key)?.cards.length ?? 0) + BUCKET_PAGE }])
      return
    }
    const next = buckets.slice(from + 1).find((b) => !isComplete(b.key)) ?? buckets.find((b) => !isComplete(b.key))
    if (next) request([{ key: next.key, need: (months.get(next.key)?.cards.length ?? 0) + BUCKET_PAGE }])
  }, [openOrder, buckets, isComplete, months, request])

  // ?m=<uid> for a card in a month that is not loaded: find its month through the item detail and
  // load that month in full (the card can sit anywhere in it)
  const [focus, setFocus] = useState<TimelineFocus | null>(null)
  const resolving = useRef<string | null>(null)
  const inItems = !!openAt
  useEffect(() => {
    if (!openUid || !enabled || buckets.length === 0 || inItems || resolving.current === openUid) return
    resolving.current = openUid
    let cancelled = false
    api.items
      .detail(openUid, !!query.vault)
      .then((d) => {
        if (cancelled) return
        const key = bucketKeyOf(buckets, d, query.sort)
        if (key) {
          request([{ key, need: countOf(key) }])
          setFocus({ key, uid: openUid })
        }
      })
      .catch(() => undefined)
      .finally(() => {
        if (resolving.current === openUid) resolving.current = null
      })
    return () => {
      cancelled = true
    }
  }, [openUid, enabled, buckets, inItems, query.vault, query.sort, request, countOf])

  const cards = useCallback(
    (key: string) => {
      const m = months.get(key)
      return m && m.started ? m.cards : undefined
    },
    [months],
  )
  const pending = useCallback((key: string) => !!months.get(key)?.pending, [months])
  const clearFocus = useCallback(() => setFocus(null), [])
  const refetch = useCallback(() => void bucketsQ.refetch(), [bucketsQ])

  return {
    buckets,
    total,
    loading: bucketsQ.isPending,
    error: bucketsQ.error,
    refetch,
    cards,
    complete: isComplete,
    pending,
    request,
    items,
    hasMore,
    loadMore,
    positionOf,
    focus,
    clearFocus,
  }
}
