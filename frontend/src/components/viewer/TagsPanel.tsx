// Viewer info panel, Tags section (section 19 of docs/ARCHITECTURE.md): the owner's free-form tags as
// solid chips, the tag learner's suggestions as dashed chips (accept adds them as the owner's, remove
// teaches the learner they were wrong), and an input that offers existing tags as you type. Enter adds
// the typed name, creating the tag when it is new. Person tags stay in the People section.
import { useId, useMemo, useState, type FormEvent, type KeyboardEvent } from 'react'
import { Link } from 'react-router'
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Check, Loader2, Plus, X } from 'lucide-react'
import { api } from '../../lib/api'
import type { ItemDetail, ItemTagOut, UserTagIn } from '../../lib/types'
import { formatPercent } from '../../lib/format'
import { useDebounced } from '../../lib/hooks'
import { errorMessage } from '../admin/adminHooks'
import { useToast } from '../Toast'
import { useInvalidateItem } from './shared'

/** Query key prefix of every tags list (the filter bar, the bulk bar and this panel). */
export const TAGS_KEY = ['tags'] as const
/** Tag names are 1 to 40 characters (section 19). */
export const MAX_TAG_LENGTH = 40

/** GET /api/tags?q=, sorted by items then name. */
export function useTagList(q = '', enabled = true) {
  return useQuery({
    queryKey: [...TAGS_KEY, 'list', q],
    queryFn: () => api.tags.list(q || undefined),
    enabled,
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  })
}

/** Invalidate the tags lists (LIBRARY_KEYS does not cover them). */
export function useInvalidateTags() {
  const qc = useQueryClient()
  return () => void qc.invalidateQueries({ queryKey: TAGS_KEY })
}

const sameName = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase()

const chipButton =
  'flex size-10 shrink-0 items-center justify-center rounded-full text-white/65 hover:bg-white/10 hover:text-white disabled:opacity-50'

interface Props {
  detail: ItemDetail
}

export default function TagsPanel({ detail }: Props) {
  const toast = useToast()
  const invalidateItem = useInvalidateItem(detail.uid)
  const invalidateTags = useInvalidateTags()
  const inputId = useId()
  const [text, setText] = useState('')
  const [status, setStatus] = useState('')
  // The list from the last change, shown until the refetched detail replaces it.
  const [local, setLocal] = useState<{ base: ItemTagOut[] | undefined; list: ItemTagOut[] } | null>(null)
  const [busyId, setBusyId] = useState<number | null>(null)

  const serverTags = detail.user_tags
  const tags = local && local.base === serverTags ? local.list : (serverTags ?? [])
  const mine = tags.filter((t) => t.source === 'user')
  const suggested = tags.filter((t) => t.source === 'auto')

  const typed = text.trim()
  const q = useDebounced(typed, 200)
  const list = useTagList(q, q !== '')
  const options = useMemo(() => {
    if (!typed) return []
    const onItem = new Set(mine.map((t) => t.id))
    return (list.data ?? []).filter((t) => !onItem.has(t.id)).slice(0, 8)
  }, [list.data, mine, typed])

  const done = (next: ItemTagOut[], message: string) => {
    setLocal({ base: serverTags, list: Array.isArray(next) ? next : tags })
    setStatus(message)
    invalidateItem()
    invalidateTags()
  }

  const add = useMutation({
    mutationFn: (v: { body: UserTagIn; label: string }) => api.items.addUserTag(detail.uid, v.body),
    onMutate: (v) => setBusyId(v.body.tag_id ?? null),
    onSettled: () => setBusyId(null),
    onSuccess: (next, v) => {
      setText('')
      done(next, `Tag ${v.label} added.`)
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not add the tag.')),
  })
  const remove = useMutation({
    mutationFn: (t: ItemTagOut) => api.items.removeUserTag(detail.uid, t.id),
    onMutate: (t) => setBusyId(t.id),
    onSettled: () => setBusyId(null),
    onSuccess: (next, t) => done(next, t.source === 'auto' ? `Suggestion ${t.name} removed.` : `Tag ${t.name} removed.`),
    onError: (e) => toast.error(errorMessage(e, 'Could not remove the tag.')),
  })
  const busy = add.isPending || remove.isPending

  const addTyped = () => {
    if (!typed || busy) return
    if (mine.some((t) => sameName(t.name, typed))) {
      setText('')
      return
    }
    const known = (list.data ?? []).find((t) => sameName(t.name, typed)) ?? suggested.find((t) => sameName(t.name, typed))
    add.mutate(known ? { body: { tag_id: known.id }, label: known.name } : { body: { name: typed }, label: typed })
  }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    addTyped()
  }
  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Escape' && text) {
      // keep the viewer open: it ignores an Escape that was already handled
      e.preventDefault()
      setText('')
    }
  }

  const libraryPath = detail.nsfw ? '/vault' : '/'

  return (
    <div>
      {mine.length > 0 && (
        <ul className="flex flex-wrap gap-2" aria-label="Your tags">
          {mine.map((t) => (
            <li key={t.id} className="flex max-w-full items-center rounded-full bg-white/10 transition-colors hover:bg-white/15">
              <Link
                to={`${libraryPath}?tag=${encodeURIComponent(t.name)}`}
                className="flex h-10 min-w-0 items-center rounded-full pl-4 pr-2 text-sm font-medium text-white"
                title={`Show everything tagged ${t.name}`}
              >
                <span className="truncate">{t.name}</span>
              </Link>
              <button
                type="button"
                onClick={() => remove.mutate(t)}
                disabled={busy}
                aria-label={`Remove tag ${t.name}`}
                className={`-ml-1 ${chipButton}`}
              >
                {busyId === t.id && remove.isPending ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <X className="size-4" aria-hidden />}
              </button>
            </li>
          ))}
        </ul>
      )}

      {suggested.length > 0 && (
        <div className={mine.length > 0 ? 'mt-3' : undefined}>
          <p className="mb-1.5 text-xs text-white/55">Suggested from items you tagged that look alike:</p>
          <ul className="flex flex-wrap gap-2" aria-label="Suggested tags">
            {suggested.map((t) => (
              <li
                key={t.id}
                className="flex max-w-full items-center rounded-full border border-dashed border-white/35 pl-4"
                title={t.score != null ? `Similarity ${formatPercent(t.score)}` : undefined}
              >
                <span className="min-w-0 truncate text-sm font-medium text-white/90">{t.name}</span>
                <span className="ml-1.5 shrink-0 text-xs text-white/50">suggested</span>
                <button
                  type="button"
                  onClick={() => add.mutate({ body: { tag_id: t.id }, label: t.name })}
                  disabled={busy}
                  aria-label={`Accept suggested tag ${t.name}`}
                  className={`ml-0.5 ${chipButton}`}
                >
                  {busyId === t.id && add.isPending ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <Check className="size-4" aria-hidden />}
                </button>
                <button
                  type="button"
                  onClick={() => remove.mutate(t)}
                  disabled={busy}
                  aria-label={`Remove suggested tag ${t.name}`}
                  className={`-ml-1.5 ${chipButton}`}
                >
                  {busyId === t.id && remove.isPending ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <X className="size-4" aria-hidden />}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {tags.length === 0 && <p className="text-sm text-white/60">No tags yet.</p>}

      <form onSubmit={onSubmit} className="mt-3 flex items-center gap-2">
        <label htmlFor={inputId} className="sr-only">
          Add a tag
        </label>
        <input
          id={inputId}
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          maxLength={MAX_TAG_LENGTH}
          placeholder="Add a tag, like wedding"
          autoComplete="off"
          spellCheck={false}
          enterKeyHint="done"
          aria-busy={add.isPending}
          className="h-10 min-w-0 flex-1 rounded-full border border-white/15 bg-white/5 px-4 text-base text-white placeholder:text-white/40 focus:border-white/40 focus:outline-none focus:ring-2 focus:ring-white/15 sm:text-sm"
        />
        <button
          type="submit"
          disabled={!typed || busy}
          className="inline-flex h-10 shrink-0 items-center gap-1.5 rounded-full border border-white/15 px-4 text-sm font-semibold text-white transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {add.isPending && busyId === null ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <Plus className="size-4" aria-hidden />}
          Add
        </button>
      </form>

      {options.length > 0 && (
        <ul className="mt-2 flex flex-wrap gap-1.5" aria-label="Existing tags">
          {options.map((t) => (
            <li key={t.id}>
              <button
                type="button"
                onClick={() => add.mutate({ body: { tag_id: t.id }, label: t.name })}
                disabled={busy}
                className="inline-flex h-10 max-w-full items-center gap-1.5 rounded-full bg-white/10 px-3 text-sm text-white/90 hover:bg-white/15 disabled:opacity-50"
                aria-label={`Add tag ${t.name}`}
              >
                <Plus className="size-3.5 shrink-0" aria-hidden />
                <span className="truncate">{t.name}</span>
                <span className="shrink-0 text-xs tabular-nums text-white/50">{t.items}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {typed && !list.isFetching && options.length === 0 && !mine.some((t) => sameName(t.name, typed)) && (
        <p className="mt-2 text-xs text-white/55">New tag. Press Enter or Add to create {typed}.</p>
      )}

      <p className="sr-only" aria-live="polite">
        {status}
      </p>
    </div>
  )
}
