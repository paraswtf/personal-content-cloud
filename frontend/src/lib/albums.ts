// Albums (docs/ARCHITECTURE.md section 24): types and calls for /api/albums. Members are listed through
// the library (GET /api/items?album=), added and removed with the album and unalbum bulk actions
// (api.items.bulk) or POST /api/albums/{id}/items. Query keys: ['albums', ...] for lists and pickers,
// ['album', id] for one album; both prefixes are in LIBRARY_KEYS, so library changes refetch them.
import { del, get, patch, post, qs } from './api'
import type { OkOut } from './types'

export interface AlbumOut {
  id: number
  name: string
  description: string
  /** Live members on the normal side. */
  count: number
  /** Live members in the vault (a number only; their cards show in an unlocked vault). */
  vault_count: number
  /** The chosen cover's thumb when it is a live normal-side member, else the newest such member's. */
  cover_url: string | null
  /** Sort keys (taken, else added) of the newest and oldest normal-side members. */
  newest: string | null
  oldest: string | null
  created_at: string
  updated_at: string
}

export interface AlbumPage {
  items: AlbumOut[]
  next_cursor: string | null
  total: number
}

export interface AlbumIn {
  name: string
  description?: string
  item_uids?: string[]
}

export interface AlbumUpdate {
  name?: string
  description?: string
  /** null clears the chosen cover; leave it out to keep it. */
  cover_item_uid?: string | null
}

export type AlbumSort = 'updated' | 'name' | 'created'

export const ALBUMS_KEY = ['albums'] as const
export const albumKey = (id: number) => ['album', id] as const

export const albumsApi = {
  list: (q: { q?: string; sort?: AlbumSort; cursor?: string | null; limit?: number } = {}, signal?: AbortSignal) =>
    get<AlbumPage>(`/api/albums${qs({ ...q })}`, signal),
  create: (body: AlbumIn) => post<AlbumOut>('/api/albums', body),
  get: (id: number) => get<AlbumOut>(`/api/albums/${id}`),
  update: (id: number, body: AlbumUpdate) => patch<AlbumOut>(`/api/albums/${id}`, body),
  remove: (id: number) => del<OkOut>(`/api/albums/${id}`),
  addItems: (id: number, uids: string[]) => post<{ added: number; skipped: number }>(`/api/albums/${id}/items`, { uids }),
}

/** "March 2024" for one month, "Jun 2023 to May 2024" for a range, '' without dates. */
export function albumDateRange(oldest: string | null, newest: string | null): string {
  if (!oldest || !newest) return ''
  const a = new Date(oldest)
  const b = new Date(newest)
  if (Number.isNaN(a.getTime()) || Number.isNaN(b.getTime())) return ''
  const month = (d: Date, long: boolean) => d.toLocaleDateString(undefined, { month: long ? 'long' : 'short', year: 'numeric' })
  if (a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth()) return month(a, true)
  return `${month(a, false)} to ${month(b, false)}`
}
