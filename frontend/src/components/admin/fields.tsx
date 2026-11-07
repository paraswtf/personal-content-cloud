// Form controls and small layout pieces shared by the admin pages.
import type { InputHTMLAttributes, ReactNode } from 'react'
import clsx from 'clsx'
import { Search, X } from 'lucide-react'
import Select, { type SelectOption } from '../Select'

export const inputClass =
  'h-10 w-full rounded-full border border-line bg-paper px-4 text-sm text-ink placeholder:text-ink-faint hover:border-ink-faint focus-visible:border-brand focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/20 disabled:cursor-not-allowed disabled:opacity-60 aria-[invalid=true]:border-brand'

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string
  description?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-x-4 gap-y-3">
      <div className="min-w-0">
        <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-sm text-ink-soft">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

export function SearchInput({
  value,
  onChange,
  label,
  placeholder,
  className,
}: {
  value: string
  onChange: (v: string) => void
  label: string
  placeholder?: string
  className?: string
}) {
  return (
    <div className={clsx('relative', className)}>
      <Search className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-ink-faint" aria-hidden />
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-label={label}
        placeholder={placeholder ?? label}
        className={clsx(inputClass, 'pl-10 pr-10')}
      />
      {value && (
        <button
          type="button"
          onClick={() => onChange('')}
          aria-label="Clear search"
          className="absolute right-1 top-1/2 flex size-8 -translate-y-1/2 items-center justify-center rounded-full text-ink-faint hover:bg-ink/5 hover:text-ink"
        >
          <X className="size-4" />
        </button>
      )}
    </div>
  )
}

export function TextInput({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={clsx(inputClass, className)} {...rest} />
}

/** Labelled brand dropdown for the admin toolbars (see components/Select.tsx). */
export function SelectField({
  label,
  value,
  onChange,
  options,
  className,
  searchable,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: SelectOption[]
  className?: string
  searchable?: boolean
}) {
  return <Select showLabel label={label} value={value} onChange={onChange} options={options} className={className} searchable={searchable} />
}

export function Segmented<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: T
  onChange: (v: T) => void
  options: { value: T; label: string }[]
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex max-w-full overflow-x-auto rounded-full border border-line bg-paper p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
          className={clsx(
            'h-9 shrink-0 rounded-full px-3.5 text-sm font-medium transition-colors',
            value === o.value ? 'bg-ink text-paper' : 'text-ink-soft hover:bg-ink/5',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

/** Sticky action bar shown while items are selected. */
export function SelectionBar({
  count,
  noun,
  onClear,
  onSelectAll,
  allSelected,
  total,
  selectingAll,
  children,
}: {
  count: number
  noun: string
  onClear: () => void
  onSelectAll?: () => void
  allSelected?: boolean
  /** Size of the whole list: the select-all button then reads "Select all {total}". */
  total?: number
  /** True while select-all is still loading the rest of the list. */
  selectingAll?: boolean
  children: ReactNode
}) {
  return (
    <div
      role="region"
      aria-label="Selection actions"
      className="sticky bottom-3 z-30 mt-4 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-2xl border border-line bg-paper px-4 py-3 shadow-lg"
    >
      <p className="mr-auto text-sm font-semibold" aria-live="polite">
        {count} {noun} selected
      </p>
      <div className="flex flex-wrap items-center gap-2">
        {children}
        {onSelectAll && !allSelected && (
          <button
            type="button"
            onClick={onSelectAll}
            disabled={selectingAll}
            className="h-10 rounded-full px-3 text-sm font-medium text-ink-soft hover:bg-ink/5 disabled:cursor-wait disabled:opacity-60"
          >
            {selectingAll ? 'Loading...' : total !== undefined ? `Select all ${total.toLocaleString()}` : 'Select all loaded'}
          </button>
        )}
        <button type="button" onClick={onClear} className="h-10 rounded-full px-3 text-sm font-medium text-ink-soft hover:bg-ink/5">
          Clear
        </button>
      </div>
    </div>
  )
}

export function LoadMore({
  hasMore,
  loading,
  onClick,
  sentinelRef,
}: {
  hasMore: boolean
  loading: boolean
  onClick: () => void
  sentinelRef?: React.Ref<HTMLDivElement>
}) {
  if (!hasMore) return null
  return (
    <div ref={sentinelRef} className="flex justify-center py-6">
      <button
        type="button"
        onClick={onClick}
        disabled={loading}
        className="inline-flex h-10 items-center rounded-full border border-line bg-paper px-5 text-sm font-semibold hover:border-ink-faint disabled:opacity-60"
      >
        {loading ? 'Loading...' : 'Load more'}
      </button>
    </div>
  )
}

export function ListSkeleton({ count = 8, className }: { count?: number; className?: string }) {
  return (
    <div className={className} aria-hidden>
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="aspect-square animate-pulse rounded-2xl bg-ink/5" />
      ))}
    </div>
  )
}
