// Brand-themed dropdown: a button that opens a listbox popover (replaces the OS-styled native <select>).
// The menu is rendered in a portal with fixed positioning so overflow containers never clip it, and it
// flips upward when there is no room below. Lists with more than 8 options get a filter input.
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import clsx from 'clsx'
import { Check, ChevronDown, Search } from 'lucide-react'

export interface SelectOption {
  value: string
  label: string
  /** Quiet text on the right of the row, for example a count. */
  hint?: string
}

interface Props {
  value: string
  onChange: (value: string) => void
  options: SelectOption[]
  /** Accessible name. Shown inline before the trigger when `showLabel` is set. */
  label: string
  showLabel?: boolean
  /** Shown in the trigger when `value` matches no option. */
  placeholder?: string
  size?: 'sm' | 'md'
  disabled?: boolean
  className?: string
  /** Filter input at the top of the menu. Defaults to on when there are more than 8 options. */
  searchable?: boolean
}

const SEARCH_AFTER = 8
const MENU_MAX = 320
const ROW_H = 40
const GAP = 6
const EDGE = 8

interface Pos {
  left: number
  width: number
  maxHeight: number
  top?: number
  bottom?: number
  up: boolean
}

export default function Select({
  value,
  onChange,
  options,
  label,
  showLabel = false,
  placeholder = 'Select',
  size = 'md',
  disabled = false,
  className,
  searchable,
}: Props) {
  const uid = useId()
  const triggerId = `${uid}-trigger`
  const labelId = `${uid}-label`
  const valueId = `${uid}-value`
  const listId = `${uid}-list`
  const optId = (i: number) => `${uid}-opt-${i}`

  const canSearch = searchable ?? options.length > SEARCH_AFTER
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const [pos, setPos] = useState<Pos | null>(null)

  const triggerRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const typed = useRef({ text: '', at: 0 })

  const selected = options.find((o) => o.value === value)

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!canSearch || !q) return options
    return options.filter((o) => o.label.toLowerCase().includes(q) || (o.hint ?? '').toLowerCase().includes(q))
  }, [options, query, canSearch])

  const place = useCallback(() => {
    const el = triggerRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const vw = document.documentElement.clientWidth
    const vh = window.innerHeight
    const width = Math.min(Math.max(r.width, 200), vw - EDGE * 2)
    const left = Math.min(Math.max(r.left, EDGE), vw - width - EDGE)
    const below = vh - r.bottom - GAP - EDGE
    const above = r.top - GAP - EDGE
    const want = Math.min(options.length * ROW_H + 16 + (canSearch ? 56 : 0), MENU_MAX)
    const up = below < want && above > below
    const room = Math.max(up ? above : below, 120)
    setPos({
      left,
      width,
      maxHeight: Math.min(MENU_MAX, room),
      up,
      ...(up ? { bottom: vh - r.top + GAP } : { top: r.bottom + GAP }),
    })
  }, [options.length, canSearch])

  useLayoutEffect(() => {
    if (!open) {
      setPos(null)
      return
    }
    place()
    const onScroll = (e: Event) => {
      if (panelRef.current?.contains(e.target as Node)) return
      place()
    }
    window.addEventListener('resize', place)
    window.addEventListener('scroll', onScroll, true)
    return () => {
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', onScroll, true)
    }
  }, [open, place])

  // Click or tap outside closes without stealing focus.
  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node
      if (triggerRef.current?.contains(t) || panelRef.current?.contains(t)) return
      setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    return () => document.removeEventListener('pointerdown', onDown)
  }, [open])

  // Searchable menus take focus into the filter input once they are on screen.
  useEffect(() => {
    if (open && canSearch && pos) inputRef.current?.focus({ preventScroll: true })
    // Only when the menu first appears.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, canSearch, pos === null])

  useEffect(() => {
    if (!open) return
    document.getElementById(optId(active))?.scrollIntoView({ block: 'nearest' })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, active, shown, pos === null])

  // Keep the active row valid while the filter changes.
  useEffect(() => {
    setActive((a) => Math.min(a, Math.max(shown.length - 1, 0)))
  }, [shown.length])

  const openMenu = (at?: number) => {
    if (disabled) return
    setQuery('')
    const sel = options.findIndex((o) => o.value === value)
    setActive(at ?? Math.max(sel, 0))
    setOpen(true)
  }

  const close = (refocus: boolean) => {
    setOpen(false)
    setQuery('')
    if (refocus) triggerRef.current?.focus()
  }

  const choose = (o: SelectOption | undefined) => {
    if (!o) return
    if (o.value !== value) onChange(o.value)
    close(true)
  }

  const typeAhead = (ch: string) => {
    const now = Date.now()
    const t = typed.current
    t.text = now - t.at > 700 ? ch : t.text + ch
    t.at = now
    const needle = t.text.toLowerCase()
    const list = open ? shown : options
    const n = list.length
    if (!n) return
    const start = open ? active + (needle.length === 1 ? 1 : 0) : Math.max(options.findIndex((o) => o.value === value), -1) + 1
    for (let k = 0; k < n; k++) {
      const i = (start + k) % n
      if (list[i].label.toLowerCase().startsWith(needle)) {
        if (open) setActive(i)
        else openMenu(i)
        return
      }
    }
  }

  const onKeyDown = (e: KeyboardEvent<HTMLElement>, fromInput: boolean) => {
    const n = shown.length
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        if (!open) openMenu()
        else setActive((a) => Math.min(a + 1, n - 1))
        return
      case 'ArrowUp':
        e.preventDefault()
        if (!open) openMenu()
        else setActive((a) => Math.max(a - 1, 0))
        return
      case 'Home':
        if (open && !fromInput) {
          e.preventDefault()
          setActive(0)
        }
        return
      case 'End':
        if (open && !fromInput) {
          e.preventDefault()
          setActive(Math.max(n - 1, 0))
        }
        return
      case 'PageDown':
        if (open) {
          e.preventDefault()
          setActive((a) => Math.min(a + 8, n - 1))
        }
        return
      case 'PageUp':
        if (open) {
          e.preventDefault()
          setActive((a) => Math.max(a - 8, 0))
        }
        return
      case 'Enter':
      case ' ':
        if (e.key === ' ' && fromInput) return
        e.preventDefault()
        if (!open) openMenu()
        else choose(shown[active])
        return
      case 'Escape':
        if (open) {
          e.preventDefault()
          e.stopPropagation()
          close(true)
        }
        return
      case 'Tab':
        if (open) close(true)
        return
      default:
        if (!fromInput && e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey && e.key !== ' ') typeAhead(e.key)
    }
  }

  const heightCls = size === 'sm' ? 'h-8 text-sm' : 'h-10 text-sm'
  const activeId = open && shown[active] ? optId(active) : undefined

  return (
    <div className={clsx('inline-flex max-w-full items-center gap-2', className)}>
      <span id={labelId} className={showLabel ? 'shrink-0 text-sm text-ink-soft' : 'sr-only'}>
        {label}
      </span>
      <button
        ref={triggerRef}
        id={triggerId}
        type="button"
        disabled={disabled}
        // Without a filter input the trigger itself is a select-only combobox, so its
        // aria-activedescendant is announced. With one, it is a button that opens the menu.
        role={canSearch ? undefined : 'combobox'}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        aria-labelledby={canSearch ? `${labelId} ${valueId}` : labelId}
        aria-activedescendant={!canSearch ? activeId : undefined}
        onClick={() => (open ? close(false) : openMenu())}
        onKeyDown={(e) => onKeyDown(e, false)}
        // Some browsers turn the keyup of Space into a click even after keydown was handled,
        // which would close the menu again (or reopen it after a choice).
        onKeyUp={(e) => {
          if (e.key === ' ') e.preventDefault()
        }}
        className={clsx(
          'flex flex-1 items-center gap-2 rounded-full border bg-paper pl-4 pr-3 text-left font-medium text-ink transition-colors',
          'focus-visible:border-brand focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/20',
          'disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:border-line',
          open ? 'border-brand ring-2 ring-brand/20' : 'border-line hover:border-ink-faint',
          showLabel ? 'min-w-36' : 'min-w-0',
          heightCls,
        )}
      >
        <span id={valueId} className={clsx('min-w-0 flex-1 truncate', !selected && 'font-normal text-ink-faint')}>
          {selected ? selected.label : placeholder}
        </span>
        <ChevronDown
          className={clsx('size-4 shrink-0 text-ink-faint transition-transform duration-150', open && 'rotate-180 text-brand')}
          aria-hidden
        />
      </button>

      {open &&
        pos &&
        createPortal(
          <div
            ref={panelRef}
            style={{ left: pos.left, width: pos.width, maxHeight: pos.maxHeight, top: pos.top, bottom: pos.bottom }}
            className={clsx(
              'select-pop fixed z-[80] flex flex-col overflow-hidden rounded-2xl border border-line bg-paper p-1.5 shadow-xl',
              pos.up ? 'origin-bottom' : 'origin-top',
            )}
          >
            {canSearch && (
              <div className="relative mb-1 shrink-0">
                <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-ink-faint" aria-hidden />
                <input
                  ref={inputRef}
                  type="search"
                  role="combobox"
                  aria-expanded="true"
                  aria-controls={listId}
                  aria-activedescendant={activeId}
                  aria-autocomplete="list"
                  aria-label={`Search ${label.toLowerCase()}`}
                  placeholder="Search"
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value)
                    setActive(0)
                  }}
                  onKeyDown={(e) => onKeyDown(e, true)}
                  autoComplete="off"
                  spellCheck={false}
                  className="h-9 w-full rounded-full border border-line bg-paper pl-9 pr-3 text-base text-ink sm:text-sm focus:border-brand focus:outline-none focus:ring-2 focus:ring-brand/20"
                />
              </div>
            )}
            <ul
              id={listId}
              role="listbox"
              aria-label={label}
              className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
            >
              {shown.length === 0 && <li className="px-3 py-3 text-sm text-ink-soft">No matches.</li>}
              {shown.map((o, i) => {
                const isSel = o.value === value
                return (
                  <li
                    key={o.value}
                    id={optId(i)}
                    role="option"
                    aria-selected={isSel}
                    onMouseDown={(e) => e.preventDefault()}
                    onMouseMove={() => active !== i && setActive(i)}
                    onClick={() => choose(o)}
                    className={clsx(
                      'flex min-h-10 cursor-pointer items-center gap-2 rounded-xl px-3 py-2 text-sm',
                      i === active ? 'bg-cream-soft' : '',
                      isSel ? 'font-semibold text-ink' : 'text-ink',
                    )}
                  >
                    <span className="min-w-0 flex-1 truncate">{o.label}</span>
                    {o.hint && <span className="shrink-0 text-xs font-normal tabular-nums text-ink-faint">{o.hint}</span>}
                    <span className="flex size-4 shrink-0 items-center justify-center text-brand">
                      {isSel && <Check className="size-4" aria-hidden />}
                    </span>
                  </li>
                )
              })}
            </ul>
          </div>,
          document.body,
        )}
    </div>
  )
}
