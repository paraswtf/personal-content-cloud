// Library filters kept in URL search params, and their mapping to the items and search queries.
import type { ItemsQuery, SearchQuery } from '../../lib/api'
import { CATEGORIES, type Category, type ItemKind, type SearchMode } from '../../lib/types'
import { parseIds } from '../public/params'
import { MAX_QUERY } from './SearchBox'

export interface LibraryFilters {
  q: string
  mode: SearchMode
  category: Category | ''
  kind: ItemKind | ''
  people: number[]
  /** Free-form tag names (URL tag=, repeated); every one is required. */
  tags: string[]
  from: string
  to: string
  favorite: boolean
  /** Live Photos and motion photos only (URL live=1). */
  live: boolean
  misclicks: 'exclude' | 'include' | 'only'
  dups: 'collapse' | 'all'
  sort: 'taken' | 'added'
}

const KINDS: ItemKind[] = ['image', 'video', 'pdf', 'document', 'other']
const DATE = /^\d{4}-\d{2}-\d{2}$/
/** Tag names are 1 to 40 characters (section 19). */
const MAX_TAG = 40
const MAX_TAGS = 20

/** The tag= values: trimmed, non-empty, without repeats (case-insensitive), at most 20. */
export function readTags(params: URLSearchParams): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const raw of params.getAll('tag')) {
    const t = raw.trim()
    const key = t.toLowerCase()
    if (!t || t.length > MAX_TAG || seen.has(key)) continue
    seen.add(key)
    out.push(t)
    if (out.length >= MAX_TAGS) break
  }
  return out
}

export function readFilters(params: URLSearchParams, vault: boolean): LibraryFilters {
  const cat = params.get('cat') as Category | null
  const kind = params.get('kind') as ItemKind | null
  const mode = params.get('mode')
  const mis = params.get('mis')
  const from = params.get('from') ?? ''
  const to = params.get('to') ?? ''
  return {
    q: (params.get('q') ?? '').trim().slice(0, MAX_QUERY),
    mode: mode === 'semantic' || mode === 'text' ? mode : 'auto',
    category: !vault && cat && CATEGORIES.includes(cat) && cat !== 'nsfw' ? cat : '',
    kind: kind && KINDS.includes(kind) ? kind : '',
    people: parseIds(params.get('people')),
    tags: readTags(params),
    from: DATE.test(from) ? from : '',
    to: DATE.test(to) ? to : '',
    favorite: params.get('fav') === '1',
    live: params.get('live') === '1',
    misclicks: mis === 'include' || mis === 'only' ? mis : 'exclude',
    dups: params.get('dups') === 'all' ? 'all' : 'collapse',
    sort: params.get('sort') === 'added' ? 'added' : 'taken',
  }
}

export function itemsQuery(f: LibraryFilters, vault: boolean, album?: number): Omit<ItemsQuery, 'cursor'> {
  return {
    album: album ?? undefined,
    category: f.category || undefined,
    kind: f.kind || undefined,
    person: f.people.length ? f.people : undefined,
    tag: f.tags.length ? f.tags : undefined,
    from: f.from || undefined,
    to: f.to || undefined,
    favorite: f.favorite || undefined,
    live: f.live || undefined,
    misclicks: f.misclicks === 'exclude' ? undefined : f.misclicks,
    dups: f.dups === 'all' ? 'all' : undefined,
    sort: f.sort === 'added' ? 'added' : undefined,
    vault: vault || undefined,
    limit: 60,
  }
}

export function searchQuery(f: LibraryFilters, vault: boolean): Omit<SearchQuery, 'offset'> {
  return {
    q: f.q,
    mode: f.mode,
    category: f.category || undefined,
    kind: f.kind || undefined,
    person: f.people.length ? f.people : undefined,
    tag: f.tags.length ? f.tags : undefined,
    from: f.from || undefined,
    to: f.to || undefined,
    live: f.live || undefined,
    vault: vault || undefined,
    limit: 60,
  }
}

/** Any filter beyond the category chips is set (opens the filter panel on load). */
export function hasExtraFilters(f: LibraryFilters): boolean {
  return !!(f.kind || f.people.length || f.tags.length || f.from || f.to || f.favorite || f.live || f.misclicks !== 'exclude' || f.dups === 'all')
}
