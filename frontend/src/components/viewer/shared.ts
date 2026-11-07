// Helpers shared by the Viewer pieces: cache invalidation, cache lookups, dark-theme styles.
import { useCallback } from 'react'
import { useQueryClient, type QueryClient } from '@tanstack/react-query'
import type { ItemCard } from '../../lib/types'
import { useRefreshLibrary } from '../../lib/hooks'

/** Dark-theme button classes for controls on the #111 viewer. */
export function darkButton(kind: 'default' | 'danger' | 'primary' = 'default', extra?: string): string {
  const base =
    'inline-flex h-10 items-center justify-center gap-2 rounded-full px-4 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50 whitespace-nowrap'
  const tone = {
    default: 'border border-white/15 text-white hover:bg-white/10',
    danger: 'border border-red-400/40 text-red-300 hover:bg-red-500/15',
    primary: 'bg-accent text-white hover:opacity-90',
  }[kind]
  return `${base} ${tone}${extra ? ` ${extra}` : ''}`
}

export const iconButton =
  'inline-flex size-10 shrink-0 items-center justify-center rounded-full text-white/85 transition-colors hover:bg-white/10 hover:text-white aria-pressed:bg-white/15 aria-pressed:text-white disabled:opacity-40'

/** Invalidate everything a change made in the viewer can affect (the item, lists, people, documents). */
export function useInvalidateItem(uid: string) {
  const refresh = useRefreshLibrary()
  const qc = useQueryClient()
  return useCallback(() => {
    void qc.invalidateQueries({ queryKey: ['item', uid] })
    void refresh()
  }, [qc, uid, refresh])
}

function asCard(v: unknown, uid: string): ItemCard | undefined {
  if (!v || typeof v !== 'object') return undefined
  const o = v as Record<string, unknown>
  if (o.uid === uid && o.urls && typeof o.urls === 'object') return o as unknown as ItemCard
  // Document list rows ({item, document}), guess rows ({item, ...}) and duplicate groups ({best, copies}).
  for (const k of ['item', 'best']) {
    const hit = asCard(o[k], uid)
    if (hit) return hit
  }
  if (Array.isArray(o.copies)) for (const c of o.copies) if ((c as { uid?: unknown })?.uid === uid) return c as ItemCard
  return undefined
}

/** An item already in a list cache, used as a placeholder while the detail loads. */
export function findCachedItem(qc: QueryClient, uid: string): ItemCard | undefined {
  const lists = qc.getQueriesData<unknown>({
    predicate: (q) => ['items', 'search', 'person-items', 'review', 'documents', 'trash'].includes(String(q.queryKey[0])),
  })
  for (const [, data] of lists) {
    if (!data || typeof data !== 'object') continue
    const pages: unknown[] = 'pages' in data && Array.isArray((data as { pages: unknown }).pages) ? (data as { pages: unknown[] }).pages : [data]
    for (const page of pages) {
      const p = page as { items?: unknown; groups?: unknown } | null
      const rows = Array.isArray(p?.items) ? p.items : Array.isArray(p?.groups) ? p.groups : null
      if (!rows) continue
      for (const row of rows) {
        const hit = asCard(row, uid)
        if (hit) return hit
      }
    }
  }
  return undefined
}
