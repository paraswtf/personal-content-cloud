// Small shared hooks: infinite-scroll sentinel, page title, debounced values, the library cache refresh.
import { useCallback, useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './api'

const DEFAULT_TITLE = 'Personal Content Cloud'

/**
 * Calls onVisible when the returned ref's element scrolls near the viewport (infinite scroll sentinel).
 * Pass enabled=false while a page is loading or when there is nothing more to load.
 */
export function useInfiniteSentinel(onVisible: () => void, enabled: boolean) {
  const ref = useRef<HTMLDivElement | null>(null)
  const cb = useRef(onVisible)
  cb.current = onVisible
  useEffect(() => {
    const el = ref.current
    if (!el || !enabled) return
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) cb.current()
      },
      { rootMargin: '800px 0px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [enabled])
  return ref
}

/** Site config (title, categories, vault window). Cached for the session. */
export function useConfig() {
  return useQuery({ queryKey: ['config'], queryFn: api.config, staleTime: 5 * 60_000 })
}

/** Set document.title as "<page> - <site title>" (or just the site title). */
export function usePageTitle(page?: string | null) {
  const site = useConfig().data?.site_title ?? DEFAULT_TITLE
  useEffect(() => {
    document.title = page ? `${page} - ${site}` : site
  }, [page, site])
}

/** `value`, updated only after it has stayed the same for `ms`. */
export function useDebounced<T>(value: T, ms = 300): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms)
    return () => window.clearTimeout(t)
  }, [value, ms])
  return v
}

/** Query key prefixes that any change to items, people or documents can affect. */
export const LIBRARY_KEYS = [
  'items',
  'item',
  'item-text',
  'item-pages',
  'search',
  'people',
  'person',
  'person-faces',
  'person-items',
  'faces-unassigned',
  'suggestions',
  'documents',
  'document-types',
  'review',
  'stats',
  'trash',
] as const

/** Invalidate every library query (lists, details, people, documents, review bins, stats). */
export function useRefreshLibrary() {
  const qc = useQueryClient()
  return useCallback(
    () => Promise.all(LIBRARY_KEYS.map((k) => qc.invalidateQueries({ queryKey: [k] }))).then(() => undefined),
    [qc],
  )
}
