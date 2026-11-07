// Library filter bar: category chips, then (behind "Filters" on small screens) kind, people, dates,
// favourites, mis-clicks and duplicates. Every change goes to the URL through `set`.
import { useState } from 'react'
import clsx from 'clsx'
import { Copy, SlidersHorizontal, Star, TriangleAlert, X } from 'lucide-react'
import { CATEGORIES, CATEGORY_LABELS, type ItemKind } from '../../lib/types'
import Select from '../Select'
import Segmented from '../public/Segmented'
import PeopleFilter from '../people/PeopleFilter'
import { hasExtraFilters, type LibraryFilters } from './filters'

interface Props {
  filters: LibraryFilters
  set: (key: string, value: string | null) => void
  clearAll: () => void
  vault: boolean
  searching: boolean
}

const chip = (on: boolean) =>
  clsx(
    'inline-flex h-10 shrink-0 items-center gap-1.5 rounded-full border px-4 text-sm font-semibold transition-colors',
    on ? 'border-ink bg-ink text-surface' : 'border-line bg-surface text-ink-soft hover:border-ink-faint hover:text-ink',
  )

const KIND_OPTIONS: { value: '' | ItemKind; label: string }[] = [
  { value: '', label: 'All types' },
  { value: 'image', label: 'Photos' },
  { value: 'video', label: 'Videos' },
  { value: 'pdf', label: 'PDFs' },
  { value: 'document', label: 'Office and text files' },
  { value: 'other', label: 'Other files' },
]

const dateInput =
  'h-10 rounded-full border border-line bg-surface px-3 text-sm text-ink hover:border-ink-faint focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20'

export default function FilterBar({ filters: f, set, clearAll, vault, searching }: Props) {
  const [open, setOpen] = useState(() => hasExtraFilters(f))
  const extra = hasExtraFilters(f)
  const any = extra || !!f.category

  return (
    <div className="flex flex-col gap-3 border-y border-line py-3" role="group" aria-label="Filters">
      <div className="flex items-center gap-2">
        {!vault && (
          <div className="-mx-4 flex min-w-0 flex-1 gap-2 overflow-x-auto px-4 pb-0.5 sm:mx-0 sm:px-0" role="group" aria-label="Category">
            <button type="button" className={chip(!f.category)} aria-pressed={!f.category} onClick={() => set('cat', null)}>
              All
            </button>
            {CATEGORIES.filter((c) => c !== 'nsfw').map((c) => (
              <button key={c} type="button" className={chip(f.category === c)} aria-pressed={f.category === c} onClick={() => set('cat', f.category === c ? null : c)}>
                {CATEGORY_LABELS[c]}
              </button>
            ))}
          </div>
        )}
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className={clsx(chip(open || extra), 'ml-auto')}
        >
          <SlidersHorizontal className="size-4" aria-hidden />
          Filters
          {extra && <span className="size-2 rounded-full bg-accent" aria-label="Some filters are on" />}
        </button>
      </div>

      {open && (
        <div className="flex flex-col gap-3">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <span className="text-sm font-semibold text-ink-soft">People</span>
            <PeopleFilter value={f.people} onChange={(ids) => set('people', ids.join(',') || null)} />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Select
              label="Type"
              value={f.kind}
              onChange={(v) => set('kind', v || null)}
              options={KIND_OPTIONS.map((o) => ({ value: o.value, label: o.label }))}
              className="max-w-[14rem]"
            />
            <label className="flex items-center gap-1.5 text-sm text-ink-soft">
              From
              <input type="date" value={f.from} max={f.to || undefined} onChange={(e) => set('from', e.target.value || null)} className={dateInput} />
            </label>
            <label className="flex items-center gap-1.5 text-sm text-ink-soft">
              To
              <input type="date" value={f.to} min={f.from || undefined} onChange={(e) => set('to', e.target.value || null)} className={dateInput} />
            </label>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" className={chip(f.favorite)} aria-pressed={f.favorite} onClick={() => set('fav', f.favorite ? null : '1')}>
              <Star className={clsx('size-4', f.favorite && 'fill-current')} aria-hidden />
              Favourites
            </button>
            {!searching && (
              <>
                <button
                  type="button"
                  className={chip(f.misclicks !== 'exclude')}
                  aria-pressed={f.misclicks !== 'exclude'}
                  onClick={() => set('mis', f.misclicks === 'exclude' ? 'include' : null)}
                >
                  <TriangleAlert className="size-4" aria-hidden />
                  Include mis-clicks
                </button>
                <button type="button" className={chip(f.dups === 'all')} aria-pressed={f.dups === 'all'} onClick={() => set('dups', f.dups === 'all' ? null : 'all')}>
                  <Copy className="size-4" aria-hidden />
                  Show every copy
                </button>
                <Segmented<'taken' | 'added'>
                  label="Sort"
                  value={f.sort}
                  onChange={(v) => set('sort', v === 'added' ? 'added' : null)}
                  options={[
                    { value: 'taken', label: 'Date taken' },
                    { value: 'added', label: 'Date added' },
                  ]}
                />
              </>
            )}
          </div>
        </div>
      )}

      {any && (
        <div>
          <button
            type="button"
            onClick={clearAll}
            className="inline-flex h-10 items-center gap-1.5 rounded-full px-3 text-sm font-semibold text-accent hover:bg-accent-soft"
          >
            <X className="size-4" aria-hidden />
            Clear filters
          </button>
        </div>
      )}
    </div>
  )
}
