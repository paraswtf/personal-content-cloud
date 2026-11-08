// Drop-down menus for the header (docs/ARCHITECTURE.md section 14): a trigger button, a menu of links,
// actions, radio choices, headings and separators, and flyout submenus. Keyboard: Enter, Space or
// ArrowDown on the trigger opens the menu on its first item; ArrowUp and ArrowDown move, Home and End
// jump, ArrowRight (or Enter) opens a submenu on its first item, ArrowLeft or Escape closes it again,
// Escape closes the menu and returns focus to the trigger, Tab closes it. Mouse: a submenu opens on hover
// and on click; touch opens it on tap. A click outside, choosing an item or a route change closes it.
// A submenu flies out to the right when there is room, else to the left.
import { useCallback, useEffect, useId, useRef, useState, type ComponentType, type KeyboardEvent, type ReactNode } from 'react'
import { Link, useLocation } from 'react-router'
import clsx from 'clsx'
import { Check, ChevronDown, ChevronRight } from 'lucide-react'

type Icon = ComponentType<{ className?: string; 'aria-hidden'?: boolean }>

export type MenuEntry =
  | { kind: 'link'; to: string; label: string; icon?: Icon; badge?: number; hint?: string; active?: boolean }
  | { kind: 'action'; label: string; icon?: Icon; onSelect: () => void; danger?: boolean; disabled?: boolean }
  | { kind: 'radio'; label: string; icon?: Icon; checked: boolean; onSelect: () => void }
  | { kind: 'submenu'; label: string; icon?: Icon; hint?: string; active?: boolean; entries: MenuEntry[] }
  | { kind: 'heading'; label: string }
  | { kind: 'separator' }

const ITEM = 'flex min-h-10 w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm text-ink outline-none transition-colors hover:bg-ink/5 focus-visible:bg-ink/[0.07]'
const PANEL = 'z-40 w-60 rounded-xl border border-line bg-surface p-1.5 shadow-[0_8px_30px_rgb(0_0_0/0.12)]'
const SUB_W = 240

function Count({ n }: { n?: number }) {
  if (!n) return null
  return (
    <span className="ml-auto min-w-5 rounded-full bg-accent px-1.5 text-center text-[11px] font-semibold leading-5 text-white tabular-nums">
      {n > 999 ? '999+' : n}
      <span className="sr-only"> waiting</span>
    </span>
  )
}

function items(root: HTMLElement | null, level: number): HTMLElement[] {
  return Array.from(root?.querySelectorAll<HTMLElement>(`[data-menu-level="${level}"]`) ?? []).filter(
    (el) => !el.hasAttribute('aria-disabled'),
  )
}

interface ListProps {
  entries: MenuEntry[]
  level: number
  label: string
  onClose: (focusTrigger: boolean) => void
  onBack?: () => void
  className?: string
}

function MenuList({ entries, level, label, onClose, onBack, className }: ListProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [openSub, setOpenSub] = useState<number | null>(null)
  const [subSide, setSubSide] = useState<'left' | 'right'>('right')
  const [focusSub, setFocusSub] = useState(false)
  const leaveTimer = useRef<number | null>(null)
  const baseId = useId()

  const openSubmenu = useCallback((i: number, focusFirst: boolean, anchor: HTMLElement | null) => {
    if (leaveTimer.current) window.clearTimeout(leaveTimer.current)
    const rect = anchor?.getBoundingClientRect()
    setSubSide(rect && rect.right + SUB_W + 8 > window.innerWidth ? 'left' : 'right')
    setFocusSub(focusFirst)
    setOpenSub(i)
  }, [])

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const list = items(ref.current, level)
    const at = list.indexOf(document.activeElement as HTMLElement)
    if (at < 0) return // the key belongs to a submenu
    const move = (to: number) => {
      e.preventDefault()
      setOpenSub(null)
      list[(to + list.length) % list.length]?.focus()
    }
    if (e.key === 'ArrowDown') move(at + 1)
    else if (e.key === 'ArrowUp') move(at - 1)
    else if (e.key === 'Home') move(0)
    else if (e.key === 'End') move(list.length - 1)
    else if (e.key === 'ArrowRight' || ((e.key === 'Enter' || e.key === ' ') && list[at].dataset.submenu)) {
      const idx = list[at].dataset.submenu
      if (idx !== undefined) {
        e.preventDefault()
        openSubmenu(Number(idx), true, list[at])
      }
    } else if ((e.key === 'ArrowLeft' || e.key === 'Escape') && onBack) {
      e.preventDefault()
      e.stopPropagation()
      onBack()
    } else if (e.key === 'Escape') {
      e.preventDefault()
      e.stopPropagation()
      onClose(true)
    } else if (e.key === 'Tab') {
      onClose(false)
    }
  }

  return (
    <div ref={ref} role="menu" aria-label={label} className={clsx(PANEL, className)} onKeyDown={onKeyDown}>
      {entries.map((entry, i) => {
        if (entry.kind === 'separator') return <div key={i} role="separator" className="my-1 h-px bg-line" />
        if (entry.kind === 'heading')
          return (
            <div key={i} role="presentation" className="px-3 pt-2 pb-1 text-[11px] font-semibold tracking-wide text-ink-faint uppercase">
              {entry.label}
            </div>
          )
        const Icon = entry.icon
        const icon = Icon ? <Icon className="size-4 shrink-0 text-ink-soft" aria-hidden /> : null
        if (entry.kind === 'link')
          return (
            <Link
              key={i}
              to={entry.to}
              role="menuitem"
              tabIndex={-1}
              data-menu-level={level}
              aria-current={entry.active ? 'page' : undefined}
              onClick={() => onClose(false)}
              onPointerEnter={() => setOpenSub(null)}
              className={clsx(ITEM, entry.active && 'bg-ink/[0.06] font-semibold')}
            >
              {icon}
              <span className="min-w-0 flex-1">
                {entry.label}
                {entry.hint && <span className="block text-xs font-normal text-ink-soft">{entry.hint}</span>}
              </span>
              <Count n={entry.badge} />
            </Link>
          )
        if (entry.kind === 'action')
          return (
            <button
              key={i}
              type="button"
              role="menuitem"
              tabIndex={-1}
              data-menu-level={level}
              aria-disabled={entry.disabled || undefined}
              onClick={() => {
                if (entry.disabled) return
                onClose(false)
                entry.onSelect()
              }}
              onPointerEnter={() => setOpenSub(null)}
              className={clsx(ITEM, entry.danger && 'text-danger hover:bg-danger/10', entry.disabled && 'opacity-50')}
            >
              {Icon ? <Icon className={clsx('size-4 shrink-0', entry.danger ? 'text-danger' : 'text-ink-soft')} aria-hidden /> : null}
              <span className="flex-1">{entry.label}</span>
            </button>
          )
        if (entry.kind === 'radio')
          return (
            <button
              key={i}
              type="button"
              role="menuitemradio"
              aria-checked={entry.checked}
              tabIndex={-1}
              data-menu-level={level}
              onClick={() => {
                entry.onSelect()
                onClose(false)
              }}
              className={clsx(ITEM, entry.checked && 'font-semibold')}
            >
              {icon}
              <span className="flex-1">{entry.label}</span>
              {entry.checked && <Check className="size-4 shrink-0 text-accent" aria-hidden />}
            </button>
          )
        // submenu
        const open = openSub === i
        const subId = `${baseId}-sub-${i}`
        return (
          <div
            key={i}
            className="relative"
            onPointerEnter={(e) => {
              if (e.pointerType === 'mouse') openSubmenu(i, false, e.currentTarget)
            }}
            onPointerLeave={(e) => {
              if (e.pointerType !== 'mouse') return
              leaveTimer.current = window.setTimeout(() => setOpenSub((s) => (s === i ? null : s)), 180)
            }}
          >
            <button
              type="button"
              role="menuitem"
              tabIndex={-1}
              data-menu-level={level}
              data-submenu={i}
              aria-haspopup="menu"
              aria-expanded={open}
              aria-controls={open ? subId : undefined}
              onClick={(e) => (open ? setOpenSub(null) : openSubmenu(i, true, e.currentTarget))}
              className={clsx(ITEM, (open || entry.active) && 'bg-ink/[0.06]', entry.active && 'font-semibold')}
            >
              {icon}
              <span className="min-w-0 flex-1">
                {entry.label}
                {entry.hint && <span className="block text-xs font-normal text-ink-soft">{entry.hint}</span>}
              </span>
              <ChevronRight className="size-4 shrink-0 text-ink-faint" aria-hidden />
            </button>
            {open && (
              <div id={subId} className={clsx('absolute -top-1.5', subSide === 'right' ? 'left-full pl-1' : 'right-full pr-1')}>
                <SubList
                  entries={entry.entries}
                  level={level + 1}
                  label={entry.label}
                  focusFirst={focusSub}
                  onClose={onClose}
                  onBack={() => {
                    setOpenSub(null)
                    ref.current?.querySelector<HTMLElement>(`[data-menu-level="${level}"][data-submenu="${i}"]`)?.focus()
                  }}
                />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

/** A submenu panel: focuses its first item when opened from the keyboard or by a tap. */
function SubList(props: Omit<ListProps, 'className'> & { focusFirst: boolean }) {
  const { focusFirst, ...rest } = props
  const wrap = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (focusFirst) wrap.current?.querySelector<HTMLElement>(`[data-menu-level="${rest.level}"]`)?.focus()
  }, [focusFirst, rest.level])
  return (
    <div ref={wrap}>
      <MenuList {...rest} />
    </div>
  )
}

export interface HeaderMenuProps {
  /** The trigger's text (also the menu's accessible name). */
  label: string
  icon?: Icon
  entries: MenuEntry[]
  align?: 'left' | 'right'
  /** A route inside this menu is open: the trigger looks current. */
  active?: boolean
  /** A small dot on the trigger (something waits inside). */
  dot?: boolean
  /** Replace the trigger's look (the account avatar). */
  trigger?: ReactNode
  /** Accessible name of the trigger when it differs from the label. */
  ariaLabel?: string
  /** Show the label from this breakpoint up (icon only below). */
  labelFrom?: 'always' | 'lg' | 'xl'
}

export default function HeaderMenu({ label, icon: Icon, entries, align = 'left', active, dot, trigger, ariaLabel, labelFrom = 'lg' }: HeaderMenuProps) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const button = useRef<HTMLButtonElement>(null)
  const { pathname, search } = useLocation()
  const menuId = useId()

  useEffect(() => setOpen(false), [pathname, search])
  useEffect(() => {
    if (!open) return
    const onDown = (e: Event) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    root.current?.querySelector<HTMLElement>('[data-menu-level="0"]')?.focus()
    return () => document.removeEventListener('pointerdown', onDown)
  }, [open])

  const close = useCallback((focusTrigger: boolean) => {
    setOpen(false)
    if (focusTrigger) button.current?.focus()
  }, [])

  return (
    <div ref={root} className="relative">
      <button
        ref={button}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-label={ariaLabel}
        title={ariaLabel ?? label}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown' && !open) {
            e.preventDefault()
            setOpen(true)
          }
        }}
        className={
          trigger
            ? 'relative inline-flex size-10 items-center justify-center rounded-full outline-none focus-visible:ring-2 focus-visible:ring-accent'
            : clsx(
                'relative inline-flex h-10 min-w-10 items-center justify-center gap-1.5 rounded-full px-2.5 text-sm font-semibold transition-colors outline-none focus-visible:ring-2 focus-visible:ring-accent lg:px-3',
                active || open ? 'bg-ink/[0.08] text-ink' : 'text-ink-soft hover:bg-ink/5 hover:text-ink',
              )
        }
      >
        {trigger ?? (
          <>
            {Icon && <Icon className="size-[18px]" aria-hidden />}
            <span className={clsx(labelFrom === 'lg' && 'sr-only lg:not-sr-only', labelFrom === 'xl' && 'sr-only xl:not-sr-only')}>{label}</span>
            <ChevronDown className={clsx('size-3.5 transition-transform', open && 'rotate-180')} aria-hidden />
          </>
        )}
        {dot && <span className="absolute top-1.5 right-1.5 size-2 rounded-full bg-accent ring-2 ring-bg" aria-hidden />}
      </button>
      {open && (
        <div id={menuId} className={clsx('absolute top-full mt-2', align === 'right' ? 'right-0' : 'left-0')}>
          <MenuList entries={entries} level={0} label={label} onClose={close} />
        </div>
      )}
    </div>
  )
}
