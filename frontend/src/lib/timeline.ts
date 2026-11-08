// Timeline API (docs/ARCHITECTURE.md section 23): the month buckets of the library and one month's
// cards, with the same filters as GET /api/items. The types live here rather than in lib/types.ts.
import { get, qs, type ItemsQuery } from './api'
import type { ItemCard, ItemPage } from './types'

export interface TimelineBucket {
  /** "YYYY-MM" in the server's local timezone. */
  key: string
  count: number
  /** The sort keys (stored UTC time strings) at the month's ends. */
  newest: string
  oldest: string
}

export interface TimelineBuckets {
  buckets: TimelineBucket[]
  total: number
}

export type TimelineQuery = Omit<ItemsQuery, 'cursor' | 'limit'>

/** Cards per page of a month; the grid asks for pages as its rows come into view. */
export const BUCKET_PAGE = 500

export function fetchBuckets(q: TimelineQuery, signal?: AbortSignal): Promise<TimelineBuckets> {
  return get<TimelineBuckets>(`/api/timeline/buckets${qs({ ...q })}`, signal)
}

/** One page of a month's cards in library order; `cursor` is the previous page's next_cursor. */
export function fetchBucketPage(q: TimelineQuery, key: string, cursor: string | null, signal?: AbortSignal): Promise<ItemPage> {
  return get<ItemPage>(`/api/timeline/bucket${qs({ ...q, key, cursor, limit: BUCKET_PAGE })}`, signal)
}

/** The sort key the timeline orders a card by. */
export function sortKeyOf(card: Pick<ItemCard, 'taken_at' | 'created_at'>, sort: ItemsQuery['sort']): string {
  return sort === 'added' ? card.created_at : (card.taken_at ?? card.created_at)
}

/** The month a card belongs to, from the buckets' key ranges; null when no bucket holds its key. */
export function bucketKeyOf(buckets: TimelineBucket[], card: Pick<ItemCard, 'taken_at' | 'created_at'>, sort: ItemsQuery['sort']): string | null {
  const k = sortKeyOf(card, sort)
  const b = buckets.find((x) => k >= x.oldest && k <= x.newest)
  return b ? b.key : null
}

/** "March 2024" for a bucket key, in the browser's language. */
export function monthLabel(key: string): string {
  const [y, m] = key.split('-').map(Number)
  if (!y || !m) return key
  return new Date(Date.UTC(y, m - 1, 1)).toLocaleDateString(undefined, { month: 'long', year: 'numeric', timeZone: 'UTC' })
}
