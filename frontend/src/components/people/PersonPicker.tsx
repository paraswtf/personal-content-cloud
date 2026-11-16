// Person search-and-pick combobox with an inline option list (works inside popovers and modals).
// Offers "Create <name>" for a typed name that matches nobody exactly. With kind 'all' it lists people
// and then a Pets section, and a new name can become a person ("Create") or a pet ("New pet").
// Naming by likeness (docs/ARCHITECTURE.md section 10): with `like` (what is being named: boxes, a person's boxes
// on some items, the whole person, or an item) the empty box lists the most alike named people first, under "Most
// alike", each with why ("Looks like them") and "Likely" when the matching would take them, then everyone else;
// typing still filters by name, and the names it finds come in order of likeness. The list waits for the likeness
// answer before it shows, so nothing moves under the pointer or the arrow keys once it is there. People without a
// name are never listed (pets without one still are).
import { Fragment, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { PawPrint, Search, UserPlus } from 'lucide-react'
import { api } from '../../lib/api'
import type { PersonKind, PersonOut, SimilarOption, SimilarPerson, SimilarSource, Species } from '../../lib/types'
import { formatCount, personName, speciesLabel } from '../../lib/format'
import { useDebounced } from '../../lib/hooks'
import { Avatar, Spinner } from '../ui'

/** A new name carries `kind: 'pet'` only when the owner picked "New pet" in a picker listing both kinds. */
export type PersonChoice = { personId: number } | { newName: string; kind?: PersonKind }

/** The person behind a pick: an alike person comes without counts, a listed one with them. */
export type PickedPerson = SimilarPerson | PersonOut

/** Request body fields for an assign or tag call from a picker choice. */
export function choiceBody(choice: PersonChoice | null): { person_id?: number | null; new_person_name?: string; kind?: PersonKind } {
  if (!choice) return { person_id: null }
  if ('personId' in choice) return { person_id: choice.personId }
  return choice.kind === 'pet' ? { new_person_name: choice.newName, kind: 'pet' } : { new_person_name: choice.newName }
}

/** How many of the most alike people the empty box lists first. */
export const ALIKE_LIMIT = 6

/** A stable key for a likeness source (callers build a new object on every render), or null when it names
 *  nothing to compare. */
export function likeKey(src: SimilarSource | null | undefined): string | null {
  if (!src) return null
  const ids = (xs?: number[]) => (xs?.length ? [...new Set(xs)].sort((a, b) => a - b).join(',') : '')
  // uids name that person's boxes on those items (no person, or no items: nothing); a person alone is the whole person
  const uids = src.uids?.length && src.person_id !== undefined ? [...new Set(src.uids)].sort().join(',') : ''
  const person = src.person_id !== undefined && (src.uids === undefined || uids) ? String(src.person_id) : ''
  const parts = [ids(src.face_ids), ids(src.body_ids), ids(src.animal_ids), uids, person, src.around_uid ?? '']
  return parts.every((p) => p === '') ? null : parts.join('|')
}

interface Props {
  /** The picked choice, and the person behind it when an existing one was picked. */
  onPick: (choice: PersonChoice, person?: PickedPerson) => void
  placeholder?: string
  allowCreate?: boolean
  excludeIds?: number[]
  autoFocus?: boolean
  /** Who to list: people (the default), pets, or 'all' (people, then a Pets section; a new name offers both). */
  kind?: PersonKind | 'all'
  /** Pets: this species is listed first. */
  species?: Species
  /** Pets: list only `species`. */
  speciesOnly?: boolean
  /** What is being named: the most alike named people are listed first (section 10, naming by likeness). */
  like?: SimilarSource | null
}

type Option =
  | { kind: 'person'; person: PickedPerson; alike: SimilarOption | null; section: 'alike' | 'rest' }
  | { kind: 'create'; name: string; as: PersonKind }

/** "12 faces, 30 items" (a pet: "Dog, 30 items"), or null for an alike person, who comes without counts. */
function countsOf(p: PickedPerson, pet: boolean): string | null {
  if (!('item_count' in p)) return null
  return pet ? `${speciesLabel(p.species)}, ${formatCount(p.item_count, 'item')}` : `${formatCount(p.face_count, 'face')}, ${formatCount(p.item_count, 'item')}`
}

export default function PersonPicker({
  onPick,
  placeholder,
  allowCreate = true,
  excludeIds,
  autoFocus = false,
  kind = 'person',
  species,
  speciesOnly = false,
  like,
}: Props) {
  const pets = kind === 'pet'
  const both = kind === 'all'
  const hint =
    placeholder ?? (both ? 'Search people and pets or type a new name' : pets ? 'Search pets or type a new name' : 'Search people or type a new name')
  const [text, setText] = useState('')
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLUListElement>(null)
  const baseId = useId()
  const q = useDebounced(text.trim(), 200)
  // A people-only picker asks for named people (an unnamed group is never a choice); pets are listed as before.
  const filter = kind === 'person' ? 'named' : 'all'

  const persons = useQuery({
    queryKey: ['people', 'picker', kind, filter, q],
    queryFn: () => api.people.list({ q: q || undefined, kind, sort: q ? 'name' : 'items', filter, limit: 30 }),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })

  // The likeness answer is read once per source and kept while the picker is open: not under the 'people' key,
  // so a refresh of the people lists elsewhere never reorders what the owner is moving through.
  const likeId = likeKey(like)
  const alike = useQuery({
    queryKey: ['likeness', likeId],
    queryFn: () => api.likeness.people(like as SimilarSource, ALIKE_LIMIT),
    enabled: likeId !== null,
    staleTime: Infinity,
    gcTime: 30_000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: false,
  })
  const waiting = likeId !== null && alike.isPending
  const ranking = likeId !== null && alike.isSuccess ? alike.data : null

  const options = useMemo<Option[]>(() => {
    if (waiting) return []
    const exclude = new Set(excludeIds ?? [])
    const all = persons.data?.items ?? []
    const kindOf = (p: PickedPerson) => p.kind ?? 'person'
    const fits = (p: PickedPerson) =>
      !exclude.has(p.id) &&
      (both || kindOf(p) === kind) &&
      (kindOf(p) === 'pet' || !!p.name?.trim()) &&
      !(speciesOnly && species && p.species !== species)
    // Both kinds: people first, then the pets (the Pets heading sits before the first of them).
    const byKind = both ? [...all.filter((p) => kindOf(p) !== 'pet'), ...all.filter((p) => kindOf(p) === 'pet')] : all
    // Pets of the box's species first, the others after them.
    const found = (species ? [...byKind].sort((a, b) => Number(b.species === species) - Number(a.species === species)) : byKind).filter(fits)
    const typed = text.trim()
    const list: Option[] = []
    if (ranking && !typed) {
      // an alike person who is in the list below too keeps the list's details (with the counts)
      const listed = new Map(found.map((p) => [p.id, p]))
      const top = ranking.items.filter((o) => fits(o.person))
      for (const o of top) list.push({ kind: 'person', person: listed.get(o.person.id) ?? o.person, alike: o, section: 'alike' })
      const shown = new Set(top.map((o) => o.person.id))
      for (const p of found) if (!shown.has(p.id)) list.push({ kind: 'person', person: p, alike: null, section: 'rest' })
    } else if (ranking) {
      // Typing: the names found keep their groups (people, then pets) and come in order of likeness within them.
      const alikeOf = new Map(ranking.items.map((o) => [o.person.id, o]))
      const score = (p: PersonOut) => ranking.scores[String(p.id)] ?? -1
      const group = (p: PersonOut) => (both && kindOf(p) === 'pet' ? 1 : 0)
      const order = found.map((p, i) => ({ p, i })).sort((a, b) => group(a.p) - group(b.p) || score(b.p) - score(a.p) || a.i - b.i)
      for (const { p } of order) list.push({ kind: 'person', person: p, alike: alikeOf.get(p.id) ?? null, section: 'rest' })
    } else {
      for (const p of found) list.push({ kind: 'person', person: p, alike: null, section: 'rest' })
    }
    if (allowCreate && typed) {
      const lower = typed.toLowerCase()
      const exact = all.some((p) => (p.name ?? '').toLowerCase() === lower)
      if (!exact) {
        if (both) list.push({ kind: 'create', name: typed, as: 'person' }, { kind: 'create', name: typed, as: 'pet' })
        else list.push({ kind: 'create', name: typed, as: pets ? 'pet' : 'person' })
      }
    }
    return list
  }, [waiting, ranking, persons.data, excludeIds, allowCreate, text, kind, species, speciesOnly, both, pets])

  useEffect(() => setActive(0), [q, options.length])
  useEffect(() => {
    if (!autoFocus) return
    inputRef.current?.focus({ preventScroll: true })
    inputRef.current?.select() // the name box opens highlighted, ready to type over
  }, [autoFocus])
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [active])

  // A pet-only picker's new name stays a plain name: its callers know the kind (and the species) already.
  const pick = (opt: Option) =>
    opt.kind === 'person'
      ? onPick({ personId: opt.person.id }, opt.person)
      : onPick(both && opt.as === 'pet' ? { newName: opt.name, kind: 'pet' } : { newName: opt.name })

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      e.stopPropagation()
      setActive((a) => Math.min(a + 1, options.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      e.stopPropagation()
      setActive((a) => Math.max(a - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      // While the typed text is still debouncing the list belongs to an earlier search.
      if (text.trim() !== q || persons.isPlaceholderData || persons.isPending || waiting) return
      const opt = options[active]
      if (opt) pick(opt)
    } else if (e.key === 'Escape' && text) {
      e.stopPropagation()
      e.preventDefault()
      setText('')
    }
  }

  const optionId = (i: number) => `${baseId}-opt-${i}`
  const searching = persons.isFetching && text.trim() !== q
  const loading = persons.isPending || waiting

  return (
    <div className="flex flex-col gap-2 text-ink">
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-ink-faint" aria-hidden />
        <input
          ref={inputRef}
          type="text"
          role="combobox"
          aria-expanded="true"
          aria-controls={`${baseId}-list`}
          aria-activedescendant={options[active] ? optionId(active) : undefined}
          aria-autocomplete="list"
          aria-label={hint}
          placeholder={hint}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          autoComplete="off"
          spellCheck={false}
          className="h-10 w-full rounded-full border border-line bg-surface pl-9 pr-9 text-base text-ink placeholder:text-ink-faint hover:border-ink-faint focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20 sm:text-sm"
        />
        {(searching || loading) && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2">
            <Spinner className="[&_svg]:size-4" label={waiting ? 'Finding who looks alike' : 'Searching'} />
          </span>
        )}
      </div>

      <ul
        ref={listRef}
        id={`${baseId}-list`}
        role="listbox"
        aria-label={both ? 'People and pets' : pets ? 'Pets' : 'People'}
        aria-busy={loading || undefined}
        className="max-h-64 overflow-y-auto overscroll-contain rounded-xl border border-line bg-surface p-1"
      >
        {persons.isError && (
          <li className="px-3 py-2 text-sm text-danger" role="alert">
            Could not load people.
          </li>
        )}
        {!loading && !persons.isError && options.length === 0 && (
          <li className="px-3 py-2 text-sm text-ink-soft">{text.trim()
              ? pets
                ? 'No pet matches that name.'
                : 'No one matches that name.'
              : both
                ? 'No people or pets yet. Type a name to add one.'
                : pets
                  ? 'No pets yet. Type a name to add one.'
                  : 'No people yet. Type a name to create one.'}</li>
        )}
        {options.map((opt, i) => {
          const prev = options[i - 1]
          const petRow = opt.kind === 'person' ? opt.person.kind === 'pet' : opt.as === 'pet'
          let heading: string | null = null
          if (opt.kind === 'person' && opt.section === 'alike') heading = i === 0 ? 'Most alike' : null
          else if (opt.kind === 'person' && prev?.kind === 'person' && prev.section === 'alike') heading = both && petRow ? 'Pets' : 'Everyone else'
          else if (both && opt.kind === 'person' && petRow && prev?.kind === 'person' && prev.person.kind !== 'pet') heading = 'Pets'
          const counts = opt.kind === 'person' ? countsOf(opt.person, petRow) : null
          const detail = opt.kind !== 'person' ? '' : opt.alike ? [opt.alike.reason || 'Alike', counts].filter(Boolean).join(', ') : (counts ?? '')
          return (
          <Fragment key={opt.kind === 'person' ? opt.person.id : `create-${opt.as}`}>
          {heading && (
            <li role="presentation" className="px-2 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-[0.1em] text-ink-faint">
              {heading}
            </li>
          )}
          <li
            id={optionId(i)}
            data-index={i}
            role="option"
            aria-selected={i === active}
            onMouseDown={(e) => e.preventDefault()}
            onMouseEnter={() => setActive(i)}
            onClick={() => pick(opt)}
            className={clsx(
              'flex min-h-10 cursor-pointer items-center gap-2.5 rounded-lg px-2 py-1.5 text-sm',
              i === active ? 'bg-accent-soft' : 'hover:bg-surface-2',
            )}
          >
            {opt.kind === 'person' ? (
              <>
                <Avatar src={opt.person.cover_url} name={opt.person.name ?? '?'} size={32} />
                <span className="min-w-0 flex-1">
                  <span className={clsx('block truncate font-medium', !opt.person.name && 'text-ink-soft')}>{personName(opt.person)}</span>
                  <span className="block text-xs text-ink-soft">{detail}</span>
                </span>
                {opt.alike?.likely && (
                  <span className="shrink-0 rounded-full border border-accent/40 px-2 py-0.5 text-[11px] font-semibold text-accent">Likely</span>
                )}
              </>
            ) : (
              <>
                <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-accent-soft text-accent">
                  {petRow ? <PawPrint className="size-4" aria-hidden /> : <UserPlus className="size-4" aria-hidden />}
                </span>
                <span className="min-w-0 flex-1 truncate">
                  {petRow ? 'New pet' : 'Create'} <span className="font-semibold">"{opt.name}"</span>
                </span>
              </>
            )}
          </li>
          </Fragment>
          )
        })}
      </ul>
    </div>
  )
}
