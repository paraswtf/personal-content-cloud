// Small segmented control (radio group) used for gallery filters.
import { useId } from 'react'
import clsx from 'clsx'

export interface SegmentOption<T extends string> {
  value: T
  label: string
}

export default function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: SegmentOption<T>[]
  onChange: (v: T) => void
}) {
  const name = useId()
  return (
    <fieldset className="min-w-0">
      <legend className="sr-only">{label}</legend>
      <div className="inline-flex h-10 items-center rounded-full border border-line bg-paper p-0.5">
        {options.map((o) => (
          <label
            key={o.value}
            className={clsx(
              'relative flex h-full cursor-pointer items-center rounded-full px-3.5 text-sm font-semibold transition-colors',
              'has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-1 has-[:focus-visible]:outline-brand',
              value === o.value ? 'bg-ink text-white' : 'text-ink-soft hover:text-ink',
            )}
          >
            <input
              type="radio"
              name={name}
              value={o.value}
              checked={value === o.value}
              onChange={() => onChange(o.value)}
              className="sr-only"
            />
            {o.label}
          </label>
        ))}
      </div>
    </fieldset>
  )
}
