// Infinite, offset-paginated search (semantic and text). Query key ['search', query].
import { useCallback, useMemo } from 'react'
import { useInfiniteQuery } from '@tanstack/react-query'
import { api, ApiError, type SearchQuery } from './api'
import type { SearchHit, SearchMode } from './types'

export interface SearchListResult {
  items: SearchHit[]
  total: number
  hasMore: boolean
  loadMore: () => void
  loading: boolean
  error: unknown
  /** The server cannot search right now (503 search_unavailable). */
  unavailable: boolean
  indexed: number
  indexing: boolean
  /** The mode the server actually used (auto picks semantic or text order). */
  mode: SearchMode | null
  refetch: () => void
}

/** Pages of GET /api/search for one query. Pass `enabled: false` when there is no query text. */
export function useSearchList(query: Omit<SearchQuery, 'offset'>, enabled: boolean): SearchListResult {
  const q = useInfiniteQuery({
    queryKey: ['search', query],
    queryFn: ({ pageParam, signal }) => api.search({ ...query, offset: pageParam }, signal),
    initialPageParam: 0,
    getNextPageParam: (last) => last.next_offset ?? undefined,
    enabled,
    retry: (count, err) => !(err instanceof ApiError && err.status >= 400 && err.status < 600) && count < 2,
  })

  const items = useMemo(() => {
    const seen = new Set<string>()
    const out: SearchHit[] = []
    for (const page of q.data?.pages ?? []) {
      for (const it of page.items) {
        if (seen.has(it.uid)) continue
        seen.add(it.uid)
        out.push(it)
      }
    }
    return out
  }, [q.data])

  const { hasNextPage, isFetchingNextPage, fetchNextPage, refetch } = q
  const loadMore = useCallback(() => {
    if (hasNextPage && !isFetchingNextPage) void fetchNextPage()
  }, [hasNextPage, isFetchingNextPage, fetchNextPage])
  const doRefetch = useCallback(() => void refetch(), [refetch])

  const pages = q.data?.pages
  const last = pages?.[pages.length - 1]
  return {
    items,
    total: pages?.[0]?.total ?? 0,
    hasMore: !!hasNextPage && !q.isFetchNextPageError,
    loadMore,
    loading: enabled && (q.isPending || isFetchingNextPage),
    error: q.error,
    unavailable: q.error instanceof ApiError && q.error.status === 503,
    indexed: last?.indexed ?? 0,
    indexing: last?.indexing ?? false,
    mode: pages?.[0]?.mode ?? null,
    refetch: doRefetch,
  }
}
