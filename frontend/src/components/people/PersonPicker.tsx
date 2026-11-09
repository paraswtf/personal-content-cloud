// Person search-and-pick combobox with an inline option list (works inside popovers and modals).
// Offers "Create <name>" for a typed name that matches nobody exactly. With kind 'all' it lists people
// and then a Pets section, and a new name can become a person ("Create") or a pet ("New pet").
import { Fragment, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { PawPrint, Search, UserPlus } from 'lucide-react'
import { api } from '../../lib/api'
import type { PersonKind, PersonOut, Species } from '../../lib/types'
import { formatCount, personName, speciesLabel } from '../../lib/format'
import { useDebounced } from '../../lib/hooks'
import { Avatar, Spinner } from '../ui'

/** A new name carries `kind: 'pet'` only when the owner picked "New pet" in a picker listing both kinds. */
export type PersonChoice = { personId: number } | { newName: string; kind?: PersonKind }

/** Request body fields for an assign or tag call from a picker choice. */
export function choiceBody(choice: PersonChoice | null): { person_id?: number | null; new_person_name?: string; kind?: PersonKind } {
  if (!choice) return { person_id: null }
  if ('personId' in choice) return { person_id: choice.personId }
  return choice.kind === 'pet' ? { new_person_name: choice.newName, kind: 'pet' } : { new_person_name: choice.newName }
}

interface Props {
  /** The picked choice, and the person behind it when an existing one was picked. */
  onPick: (choice: PersonChoice, person?: PersonOut) => void
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
}

type Option = { kind: 'person'; person: PersonOut } | { kind: 'create'; name: string; as: PersonKind }

export default function PersonPicker({
  onPick,
  placeholder,
  allowCreate = true,
  excludeIds,
  autoFocus = false,
  kind = 'person',
  species,
  speciesOnly = false,
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

  const persons = useQuery({
    queryKey: ['people', 'picker', kind, q],
    queryFn: () => api.people.list({ q: q || undefined, kind, sort: q ? 'name' : 'items', filter: 'all', limit: 30 }),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })

  const options = useMemo<Option[]>(() => {
    const exclude = new Set(excludeIds ?? [])
    const all = persons.data?.items ?? []
    // Both kinds: people first, then the pets (the Pets heading sits before the first of them).
    const found = both
      ? [...all.filter((p) => (p.kind ?? 'person') !== 'pet'), ...all.filter((p) => p.kind === 'pet')]
      : all.filter((p) => (p.kind ?? 'person') === kind)
    // Pets of the box's species first, the others after them.
    const ranked = species ? [...found].sort((a, b) => Number(b.species === species) - Number(a.species === species)) : found
    const list: Option[] = ranked
      .filter((p) => !exclude.has(p.id) && !(speciesOnly && species && p.species !== species))
      .map((p) => ({ kind: 'person', person: p }))
    const typed = text.trim()
    if (allowCreate && typed) {
      const lower = typed.toLowerCase()
      const exact = all.some((p) => (p.name ?? '').toLowerCase() === lower)
      if (!exact) {
        if (both) list.push({ kind: 'create', name: typed, as: 'person' }, { kind: 'create', name: typed, as: 'pet' })
        else list.push({ kind: 'create', name: typed, as: pets ? 'pet' : 'person' })
      }
    }
    return list
  }, [persons.data, excludeIds, allowCreate, text, kind, species, speciesOnly, both, pets])

  useEffect(() => setActive(0), [q, options.length])
  useEffect(() => {
    if (autoFocus) inputRef.current?.focus({ preventScroll: true })
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
      if (text.trim() !== q || persons.isPlaceholderData || persons.isPending) return
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
        {(searching || persons.isPending) && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2">
            <Spinner className="[&_svg]:size-4" label="Searching" />
          </span>
        )}
      </div>

      <ul
        ref={listRef}
        id={`${baseId}-list`}
        role="listbox"
        aria-label={both ? 'People and pets' : pets ? 'Pets' : 'People'}
        className="max-h-64 overflow-y-auto overscroll-contain rounded-xl border border-line bg-surface p-1"
      >
        {persons.isError && (
          <li className="px-3 py-2 text-sm text-danger" role="alert">
            Could not load people.
          </li>
        )}
        {!persons.isPending && !persons.isError && options.length === 0 && (
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
          const heading = both && opt.kind === 'person' && petRow && prev?.kind === 'person' && prev.person.kind !== 'pet' ? 'Pets' : null
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
                  <span className="block text-xs text-ink-soft">
                    {petRow
                      ? `${speciesLabel(opt.person.species)}, ${formatCount(opt.person.item_count, 'item')}`
                      : `${formatCount(opt.person.face_count, 'face')}, ${formatCount(opt.person.item_count, 'item')}`}
                  </span>
                </span>
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
