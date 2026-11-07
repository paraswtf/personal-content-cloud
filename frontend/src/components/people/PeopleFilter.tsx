// Multi-select of people and pets for the library filters: selected ones as removable chips plus an
// "Add a person" search popover (searches the server, 30 at a time).
import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { keepPreviousData, useQueries, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { PawPrint, Search, UserPlus, X } from 'lucide-react'
import { api, ApiError } from '../../lib/api'
import { formatCount, personName, speciesLabel } from '../../lib/format'
import { useDebounced } from '../../lib/hooks'
import { Avatar, Spinner } from '../ui'

interface Props {
  value: number[]
  onChange: (ids: number[]) => void
  exclude?: number[]
}

export default function PeopleFilter({ value, onChange, exclude }: Props) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ left: number; width: number }>({ left: 0, width: 320 })
  const [text, setText] = useState('')
  const [active, setActive] = useState(0)
  const rootRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const baseId = useId()
  const q = useDebounced(text.trim(), 200)

  const people = useQuery({
    queryKey: ['people', 'filter', q],
    queryFn: () => api.people.list({ q: q || undefined, kind: 'all', sort: q ? 'name' : 'items', limit: 30 }),
    enabled: open,
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  })
  const chosen = useQueries({
    queries: value.map((id) => ({ queryKey: ['person', id], queryFn: () => api.people.get(id), retry: false, staleTime: 60_000 })),
  })

  const options = useMemo(() => {
    const skip = new Set([...value, ...(exclude ?? [])])
    return (people.data?.items ?? []).filter((p) => !skip.has(p.id))
  }, [people.data, value, exclude])

  useEffect(() => setActive(0), [q, open])

  useLayoutEffect(() => {
    if (!open) return
    const place = () => {
      const rect = rootRef.current?.getBoundingClientRect()
      if (!rect) return
      const vw = document.documentElement.clientWidth
      const width = Math.min(320, vw - 32)
      const left = Math.max(16 - rect.left, Math.min(0, vw - 16 - width - rect.left))
      setPos({ left, width })
    }
    place()
    window.addEventListener('resize', place)
    return () => window.removeEventListener('resize', place)
  }, [open])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent | TouchEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('touchstart', onDown)
    const t = setTimeout(() => inputRef.current?.focus({ preventScroll: true }), 0)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('touchstart', onDown)
      clearTimeout(t)
    }
  }, [open])

  const close = (refocus: boolean) => {
    setOpen(false)
    setText('')
    if (refocus) buttonRef.current?.focus()
  }
  const add = (id: number) => {
    onChange([...value, id])
    close(true)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActive((a) => Math.min(a + 1, options.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((a) => Math.max(a - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      const p = options[active]
      if (p) add(p.id)
    } else if (e.key === 'Escape') {
      e.preventDefault()
      e.stopPropagation()
      close(true)
    }
  }

  const optionId = (i: number) => `${baseId}-opt-${i}`

  return (
    <div ref={rootRef} className="relative flex flex-wrap items-center gap-2">
      {value.map((id, i) => {
        const r = chosen[i]
        const gone = r?.error instanceof ApiError && r.error.status === 404
        const name = r?.data ? personName(r.data) : gone ? 'Person not available' : `Person ${id}`
        return (
          <span key={id} className="inline-flex h-10 max-w-full items-center gap-2 rounded-full border border-line bg-surface pl-1 pr-0.5 text-sm">
            <Avatar src={r?.data?.cover_url} name={r?.data?.name ?? '?'} size={30} />
            {r?.data?.kind === 'pet' && <PawPrint className="-mr-1 size-3.5 shrink-0 text-ink-soft" aria-label="Pet" />}
            <span className="min-w-0 truncate font-medium">{r?.isPending ? <Spinner className="[&_svg]:size-4" /> : name}</span>
            <button
              type="button"
              onClick={() => onChange(value.filter((v) => v !== id))}
              className="flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft hover:bg-surface-2 hover:text-ink"
              aria-label={`Remove ${name}`}
            >
              <X className="size-4" aria-hidden />
            </button>
          </span>
        )
      })}

      <button
        ref={buttonRef}
        type="button"
        onClick={() => (open ? close(false) : setOpen(true))}
        aria-expanded={open}
        aria-haspopup="listbox"
        className={clsx(
          'inline-flex h-10 items-center gap-2 rounded-full border border-dashed px-4 text-sm font-medium transition-colors',
          open ? 'border-accent bg-accent-soft text-accent' : 'border-ink-faint/60 text-ink-soft hover:border-ink-soft hover:text-ink',
        )}
      >
        <UserPlus className="size-4" aria-hidden />
        {value.length ? 'Add another person or pet' : 'Add a person or pet'}
      </button>

      {open && (
        <div
          style={{ left: pos.left, width: pos.width }}
          className="absolute top-full z-20 mt-2 rounded-2xl border border-line bg-surface p-2 shadow-xl"
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              e.stopPropagation()
              close(true)
            }
          }}
        >
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
              aria-label="Search people and pets by name"
              placeholder="Search by name"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={onKeyDown}
              autoComplete="off"
              spellCheck={false}
              className="h-10 w-full rounded-full border border-line bg-surface pl-9 pr-3 text-base text-ink placeholder:text-ink-faint hover:border-ink-faint focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20 sm:text-sm"
            />
          </div>
          <ul id={`${baseId}-list`} role="listbox" aria-label="People" className="mt-2 max-h-72 overflow-y-auto overscroll-contain">
            {people.isPending && (
              <li className="flex justify-center py-4">
                <Spinner />
              </li>
            )}
            {people.isError && (
              <li className="px-3 py-3 text-sm text-ink-soft">
                Could not load people.{' '}
                <button type="button" className="font-semibold text-accent underline" onClick={() => void people.refetch()}>
                  Try again
                </button>
              </li>
            )}
            {people.isSuccess && options.length === 0 && (
              <li className="px-3 py-3 text-sm text-ink-soft">{q ? 'No one matches that name.' : 'No more people to add.'}</li>
            )}
            {options.map((p, i) => (
              <li
                key={p.id}
                id={optionId(i)}
                role="option"
                aria-selected={i === active}
                onMouseDown={(e) => e.preventDefault()}
                onMouseEnter={() => setActive(i)}
                onClick={() => add(p.id)}
                className={clsx(
                  'flex min-h-11 cursor-pointer items-center gap-2.5 rounded-xl px-2 py-1.5 text-sm',
                  i === active ? 'bg-accent-soft' : 'hover:bg-surface-2',
                )}
              >
                <Avatar src={p.cover_url} name={p.name ?? '?'} size={32} />
                <span className="min-w-0 flex-1">
                  <span className={clsx('flex items-center gap-1 font-medium', !p.name && 'text-ink-soft')}>
                    {p.kind === 'pet' && <PawPrint className="size-3.5 shrink-0" aria-label="Pet" />}
                    <span className="truncate">{personName(p)}</span>
                  </span>
                  <span className="block text-xs text-ink-faint">
                    {p.kind === 'pet' ? `${speciesLabel(p.species)}, ${formatCount(p.item_count, 'item')}` : formatCount(p.item_count, 'item')}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
