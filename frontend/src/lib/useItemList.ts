// Infinite keyset-paginated lists of items (library, trash, review bins, a person's items).
// `fetchPage(cursor, signal)` returns one page; items are de-duplicated by `keyOf` (default: uid,
// then id), and `total` comes from the first page.
import { useCallback, useMemo } from 'react'
import { useInfiniteQuery, type QueryKey } from '@tanstack/react-query'
import { api, type ItemsQuery } from './api'
import type { ItemCard } from './types'

export interface CursorPage<T> {
  items: T[]
  next_cursor: string | null
  total: number
}

export interface ItemListResult<T> {
  items: T[]
  total: number
  hasMore: boolean
  /** No-op while a page is in flight. */
  loadMore: () => void
  /** True while the first page or a further page is loading. */
  loading: boolean
  error: unknown
  refetch: () => void
}

function keyOf(it: unknown): unknown {
  const row = it as { uid?: unknown; id?: unknown }
  return row.uid ?? row.id
}

export function useItemList<T = ItemCard>(
  key: QueryKey,
  fetchPage: (cursor: string | null, signal?: AbortSignal) => Promise<CursorPage<T>>,
  opts: { enabled?: boolean; pollMs?: number | false; keyOf?: (item: T) => unknown } = {},
): ItemListResult<T> {
  const enabled = opts.enabled ?? true
  const getKey = opts.keyOf ?? keyOf
  const q = useInfiniteQuery({
    queryKey: key,
    queryFn: ({ pageParam, signal }) => fetchPage(pageParam, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor ?? undefined,
    enabled,
    refetchInterval: opts.pollMs ?? false,
  })

  const items = useMemo(() => {
    const seen = new Set<unknown>()
    const out: T[] = []
    for (const page of q.data?.pages ?? []) {
      for (const it of page.items) {
        const k = getKey(it)
        if (k !== undefined && k !== null) {
          if (seen.has(k)) continue
          seen.add(k)
        }
        out.push(it)
      }
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q.data])

  const { hasNextPage, isFetchingNextPage, fetchNextPage, refetch } = q
  const loadMore = useCallback(() => {
    if (hasNextPage && !isFetchingNextPage) void fetchNextPage()
  }, [hasNextPage, isFetchingNextPage, fetchNextPage])
  const doRefetch = useCallback(() => void refetch(), [refetch])

  return {
    items,
    total: q.data?.pages[0]?.total ?? 0,
    // A failed page stops automatic loading (sentinel, viewer look-ahead); loadMore retries it.
    hasMore: !!hasNextPage && !q.isFetchNextPageError,
    loadMore,
    loading: enabled && (q.isPending || isFetchingNextPage),
    error: q.error,
    refetch: doRefetch,
  }
}

/** The library list: GET /api/items for one filter. Query key ['items', query]. */
export function useLibraryItems(query: Omit<ItemsQuery, 'cursor'>, enabled = true): ItemListResult<ItemCard> {
  const fetchPage = useCallback(
    (cursor: string | null, signal?: AbortSignal) => api.items.list({ ...query, cursor }, signal),
    [query],
  )
  return useItemList<ItemCard>(['items', query], fetchPage, { enabled })
}
