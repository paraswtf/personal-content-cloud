// Shared hooks and helpers for the admin pages.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useInfiniteQuery, useQueryClient, type QueryKey } from '@tanstack/react-query'
import { useNavigate, useSearchParams } from 'react-router'
import { ApiError } from '../../lib/api'

export function errorMessage(e: unknown, fallback = 'Something went wrong.'): string {
  if (e instanceof ApiError || e instanceof Error) return e.message || fallback
  return fallback
}

export function isUnauthorized(e: unknown): boolean {
  return e instanceof ApiError && e.status === 401
}

export function useDebounced<T>(value: T, ms = 300): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms)
    return () => window.clearTimeout(t)
  }, [value, ms])
  return v
}

export interface Selection<K> {
  selected: Set<K>
  count: number
  has: (k: K) => boolean
  toggle: (k: K) => void
  setAll: (ks: K[]) => void
  clear: () => void
}

export function useSelection<K extends string | number>(): Selection<K> {
  const [selected, setSelected] = useState<Set<K>>(() => new Set())
  const toggle = useCallback((k: K) => {
    setSelected((cur) => {
      const next = new Set(cur)
      if (next.has(k)) next.delete(k)
      else next.add(k)
      return next
    })
  }, [])
  const setAll = useCallback((ks: K[]) => setSelected(new Set(ks)), [])
  const clear = useCallback(() => setSelected(new Set()), [])
  return { selected, count: selected.size, has: (k) => selected.has(k), toggle, setAll, clear }
}

/** Invalidate everything an admin change can affect. */
export function useRefreshAll() {
  const qc = useQueryClient()
  return useCallback(
    () =>
      Promise.all(
        ['admin', 'people', 'person', 'media', 'media-detail', 'uploaders', 'config'].map((k) =>
          qc.invalidateQueries({ queryKey: [k] }),
        ),
      ),
    [qc],
  )
}

/** Run an action for each item one after another (SQLite writes), collecting failures. */
export async function runBulk<T>(
  items: T[],
  fn: (item: T) => Promise<unknown>,
): Promise<{ done: number; failed: number; firstError: unknown }> {
  let done = 0
  let failed = 0
  let firstError: unknown = null
  for (const item of items) {
    try {
      await fn(item)
      done++
    } catch (e) {
      failed++
      if (firstError === null) firstError = e
      if (isUnauthorized(e)) break
    }
  }
  // Let the layout see an expired session (it watches mutation errors for 401).
  if (isUnauthorized(firstError)) throw firstError
  return { done, failed, firstError }
}

interface CursorPage<T> {
  items: T[]
  next_cursor: string | null
  total: number
}

/** Cursor paginated list with load-more helpers. */
export function useCursorList<T, P extends CursorPage<T> = CursorPage<T>>(
  key: QueryKey,
  fetchPage: (cursor: string | null) => Promise<P>,
  opts: { poll?: (items: T[]) => boolean; enabled?: boolean } = {},
) {
  const poll = opts.poll
  const query = useInfiniteQuery({
    queryKey: key,
    queryFn: ({ pageParam }) => fetchPage(pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor ?? undefined,
    enabled: opts.enabled ?? true,
    refetchInterval: poll
      ? (q) => (poll((q.state.data?.pages ?? []).flatMap((p) => p.items)) ? 8000 : false)
      : false,
  })
  // Offset cursors shift when rows change between page loads, so a row can arrive on two pages.
  // Keep the first copy so keys stay unique.
  const items = useMemo(() => {
    const seen = new Set<unknown>()
    return (query.data?.pages ?? [])
      .flatMap((p) => p.items)
      .filter((it) => {
        const row = it as { id?: unknown; uid?: unknown }
        const k = row.id ?? row.uid
        if (k == null) return true
        if (seen.has(k)) return false
        seen.add(k)
        return true
      })
  }, [query.data])
  const pages = query.data?.pages
  const total = pages && pages.length ? pages[pages.length - 1].total : 0
  const { hasNextPage, isFetchingNextPage, fetchNextPage } = query
  const loadMore = useCallback(() => {
    if (hasNextPage && !isFetchingNextPage) void fetchNextPage()
  }, [hasNextPage, isFetchingNextPage, fetchNextPage])
  return {
    items,
    total,
    hasMore: !!hasNextPage,
    loadMore,
    loadingMore: isFetchingNextPage,
    query,
  }
}

/**
 * Viewer controlled by the "m" search param so the back button closes it.
 * Prev/next walk the loaded list; at the end they load the next page.
 */
export function useViewerParam(items: { uid: string }[], hasMore: boolean, loadMore: () => void, total?: number) {
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const uid = params.get('m')
  const index = uid ? items.findIndex((i) => i.uid === uid) : -1
  const pushed = useRef(false)
  const seen = useRef<string | null>(null)

  const withParam = useCallback(
    (next: string | null) => {
      const p = new URLSearchParams(params)
      if (next) p.set('m', next)
      else p.delete('m')
      return p
    },
    [params],
  )

  const open = useCallback(
    (target: string) => {
      pushed.current = true
      setParams(withParam(target))
    },
    [setParams, withParam],
  )
  const close = useCallback(() => {
    if (pushed.current) {
      pushed.current = false
      navigate(-1)
    } else {
      setParams(withParam(null), { replace: true })
    }
  }, [navigate, setParams, withParam])

  useEffect(() => {
    if (index >= 0) seen.current = uid
    else if (uid && seen.current === uid) {
      // The open item vanished (deleted): close the viewer.
      seen.current = null
      setParams(withParam(null), { replace: true })
    }
  }, [index, uid, setParams, withParam])

  const go = (delta: number) => {
    const target = items[index + delta]
    if (target) setParams(withParam(target.uid), { replace: true })
    else if (delta > 0 && hasMore) loadMore()
  }
  const onPrev = index > 0 ? () => go(-1) : undefined
  const onNext = index >= 0 && (index < items.length - 1 || hasMore) ? () => go(1) : undefined
  const position = index >= 0 ? { index, total: Math.max(total ?? items.length, items.length) } : undefined
  return { uid, open, close, onPrev, onNext, position }
}

/** A callback with a stable identity that always calls the latest function. */
export function useStable<A extends unknown[], R>(fn: (...args: A) => R): (...args: A) => R {
  const ref = useRef(fn)
  ref.current = fn
  return useCallback((...args: A) => ref.current(...args), [])
}
