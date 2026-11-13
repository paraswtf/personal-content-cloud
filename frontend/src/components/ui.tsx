// Small shared UI primitives. Pages build on these so the look stays consistent.
import { useEffect, useId, useRef, type ButtonHTMLAttributes, type CSSProperties, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import clsx from 'clsx'
import { Loader2, X } from 'lucide-react'
import { initials } from '../lib/format'

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
type ButtonSize = 'sm' | 'md' | 'lg'

export function buttonClasses(variant: ButtonVariant = 'primary', size: ButtonSize = 'md', extra?: string) {
  return clsx(
    'inline-flex items-center justify-center gap-2 rounded-full font-semibold transition-colors',
    'disabled:cursor-not-allowed disabled:opacity-50 select-none whitespace-nowrap [&_svg]:shrink-0',
    {
      sm: 'h-8 px-3 text-sm',
      md: 'h-10 px-4 text-sm',
      lg: 'h-12 px-6 text-base',
    }[size],
    {
      primary: 'bg-brand text-white hover:bg-brand-dark shadow-sm',
      secondary: 'bg-paper text-ink border border-line hover:border-ink-faint',
      ghost: 'text-ink hover:bg-ink/5',
      danger: 'bg-paper text-danger border border-danger/30 hover:bg-danger/10',
    }[variant],
    extra,
  )
}

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  className,
  children,
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: ButtonSize; loading?: boolean }) {
  return (
    <button className={buttonClasses(variant, size, className)} disabled={disabled || loading} {...rest}>
      {loading && <Loader2 className="size-4 animate-spin" aria-hidden />}
      {children}
    </button>
  )
}

export function Spinner({ className, label = 'Loading' }: { className?: string; label?: string }) {
  return (
    <span role="status" className={clsx('inline-flex items-center gap-2 text-ink-soft', className)}>
      <Loader2 className="size-5 animate-spin" aria-hidden />
      <span className="sr-only">{label}</span>
    </span>
  )
}

export function PageSpinner() {
  return (
    <div className="flex min-h-[40vh] items-center justify-center">
      <Spinner />
    </div>
  )
}

export function EmptyState({
  icon,
  title,
  children,
  action,
}: {
  icon?: ReactNode
  title: string
  children?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="mx-auto flex max-w-md flex-col items-center gap-3 px-4 py-16 text-center">
      {icon && <div className="text-ink-faint">{icon}</div>}
      <h3 className="text-lg font-semibold">{title}</h3>
      {children && <div className="text-sm text-ink-soft">{children}</div>}
      {action}
    </div>
  )
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const msg = error instanceof Error ? error.message : 'Something went wrong.'
  return (
    <div className="mx-auto flex max-w-md flex-col items-center gap-3 px-4 py-12 text-center">
      <p className="text-sm text-ink-soft">{msg}</p>
      {onRetry && (
        <Button variant="secondary" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  )
}

export function Badge({
  tone = 'neutral',
  children,
  className,
}: {
  tone?: 'neutral' | 'brand' | 'ok' | 'warn'
  children: ReactNode
  className?: string
}) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
        {
          neutral: 'bg-ink/5 text-ink-soft',
          brand: 'bg-brand-tint text-brand-dark',
          ok: 'bg-ok/10 text-ok',
          warn: 'bg-warn/10 text-warn',
        }[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

/** Round face avatar: face crop if available, initials otherwise. */
export function Avatar({
  src,
  name,
  size = 96,
  className,
}: {
  src: string | null | undefined
  name: string | null | undefined
  size?: number
  className?: string
}) {
  return (
    <span
      className={clsx(
        'inline-flex shrink-0 items-center justify-center overflow-hidden rounded-full bg-cream-soft text-ink-soft ring-1 ring-line',
        className,
      )}
      style={{ width: size, height: size }}
    >
      {src ? (
        <img src={src} alt="" loading="lazy" decoding="async" className="size-full object-cover" />
      ) : (
        <span className="font-semibold" style={{ fontSize: size / 3 }}>
          {initials(name)}
        </span>
      )}
    </span>
  )
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={clsx('rounded-[var(--radius-card)] bg-paper shadow-[var(--shadow-card)]', className)}>{children}</div>
}

// Page scroll lock shared by every overlay. It is reference counted because nested overlays (a
// confirm inside the viewer) unmount parent-first, so saving and restoring the style per overlay
// would leave the page locked.
let scrollLocks = 0
let savedOverflow = ''
export function lockScroll(): () => void {
  if (scrollLocks++ === 0) {
    savedOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
  }
  let done = false
  return () => {
    if (done) return
    done = true
    if (--scrollLocks === 0) document.body.style.overflow = savedOverflow
  }
}

const TABBABLE =
  'a[href], area[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), iframe, audio[controls], video[controls], [contenteditable]:not([contenteditable="false"]), [tabindex]:not([tabindex="-1"])'

/** The elements Tab can reach inside `root`, in document order (hidden ones left out). */
export function tabbables(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(TABBABLE)).filter(
    (n) => n.tabIndex >= 0 && (n.offsetParent !== null || n.getClientRects().length > 0),
  )
}

/**
 * Keep Tab and Shift+Tab inside `panel`: past the last tabbable element focus goes to the first and
 * back. With nothing tabbable, focus stays on the panel. Only the topmost modal dialog traps, so a
 * dialog opened over another one is the one that keeps the focus.
 */
function trapTab(e: KeyboardEvent, panel: HTMLElement | null): void {
  if (!panel) return
  const dialogs = document.querySelectorAll<HTMLElement>('[role="dialog"][aria-modal="true"]')
  if (dialogs[dialogs.length - 1] !== panel) return
  const nodes = tabbables(panel)
  const active = document.activeElement
  if (!nodes.length) {
    e.preventDefault()
    panel.focus()
    return
  }
  const first = nodes[0]
  const last = nodes[nodes.length - 1]
  if (!panel.contains(active)) {
    e.preventDefault()
    ;(e.shiftKey ? last : first).focus()
  } else if (e.shiftKey && (active === first || active === panel)) {
    e.preventDefault()
    last.focus()
  } else if (!e.shiftKey && active === last) {
    e.preventDefault()
    first.focus()
  }
}

/** Accessible modal dialog rendered in a portal. Closes on Escape and backdrop click; Tab stays inside it. */
export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  wide = false,
}: {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  footer?: ReactNode
  wide?: boolean
}) {
  const panel = useRef<HTMLDivElement>(null)
  // The latest onClose, so a parent passing a new closure each render does not re-run the effect
  // (which would move focus back to the panel and away from an input).
  const closeRef = useRef(onClose)
  useEffect(() => {
    closeRef.current = onClose
  })
  useEffect(() => {
    if (!open) return
    const prev = document.activeElement as HTMLElement | null
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeRef.current()
      else if (e.key === 'Tab') trapTab(e, panel.current)
    }
    document.addEventListener('keydown', onKey)
    // An autoFocus field inside the panel keeps its focus; otherwise the panel itself takes it.
    const t = setTimeout(() => {
      const el = panel.current
      if (el && !el.contains(document.activeElement)) el.focus()
    }, 0)
    const unlock = lockScroll()
    return () => {
      document.removeEventListener('keydown', onKey)
      clearTimeout(t)
      unlock()
      prev?.focus?.()
    }
  }, [open])
  if (!open) return null
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 p-0 sm:items-center sm:p-4" onMouseDown={onClose}>
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        onMouseDown={(e) => e.stopPropagation()}
        className={clsx(
          'flex max-h-[90dvh] w-full flex-col rounded-t-2xl bg-paper shadow-xl outline-none sm:rounded-2xl',
          wide ? 'sm:max-w-3xl' : 'sm:max-w-md',
        )}
      >
        <div className="flex items-center justify-between gap-4 border-b border-line px-5 py-4">
          <h2 className="text-base font-semibold">{title}</h2>
          <button onClick={onClose} className="rounded-full p-1 text-ink-soft hover:bg-ink/5" aria-label="Close">
            <X className="size-5" />
          </button>
        </div>
        <div className="overflow-y-auto px-5 py-4">{children}</div>
        {footer && <div className="flex justify-end gap-2 border-t border-line px-5 py-3">{footer}</div>}
      </div>
    </div>,
    document.body,
  )
}

/** Confirm dialog helper built on Modal. */
export function ConfirmModal({
  open,
  title,
  body,
  confirmLabel = 'Confirm',
  danger = false,
  loading = false,
  onConfirm,
  onClose,
  focusConfirm = false,
}: {
  open: boolean
  title: string
  body: ReactNode
  confirmLabel?: string
  danger?: boolean
  loading?: boolean
  onConfirm: () => void
  onClose: () => void
  /** The confirm button takes focus, so Enter confirms (the dialog was opened by a key meant for this). */
  focusConfirm?: boolean
}) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title={title}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant={danger ? 'danger' : 'primary'} loading={loading} onClick={onConfirm} autoFocus={focusConfirm}>
            {confirmLabel}
          </Button>
        </>
      }
    >
      <div className="text-sm text-ink-soft">{body}</div>
    </Modal>
  )
}

/** A key name in running text ("Keys: Enter yes ..."). */
export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-line bg-surface-2 px-1 font-sans text-xs text-ink">{children}</kbd>
}

/** An on/off switch. `label` names it for screen readers; put the visible label beside it. */
export function Switch({
  checked,
  onChange,
  disabled,
  label,
}: {
  checked: boolean
  onChange: (next: boolean) => void
  disabled?: boolean
  label: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="inline-flex h-10 shrink-0 items-center gap-3 rounded-full pl-1 pr-1 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-60"
    >
      <span className="hidden sm:inline">{checked ? 'On' : 'Off'}</span>
      <span
        aria-hidden
        className={clsx(
          'relative inline-flex h-7 w-12 items-center rounded-full transition-colors',
          checked ? 'bg-brand' : 'bg-ink/20',
        )}
      >
        <span
          className={clsx(
            'inline-block size-5 rounded-full bg-white shadow transition-transform',
            checked ? 'translate-x-6' : 'translate-x-1',
          )}
        />
      </span>
    </button>
  )
}

/** Width of the slider thumb in px: index.css draws it, the tick marks here line up with its centre. */
const SLIDER_THUMB = 20

/**
 * A labelled range slider: the native input (keyboard, touch and screen readers for free), styled once in
 * index.css with a 44 px tall touch target. `valueText` is the value in words, shown beside the label and read
 * out; `ticks` are values marked under the track; `fill` colours the track up to the thumb (turn it off for a
 * scale whose zero is in the middle, or whose stops are choices and not amounts).
 */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  valueText,
  ticks,
  fill = true,
  disabled,
  className,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (next: number) => void
  valueText?: string
  ticks?: number[]
  fill?: boolean
  disabled?: boolean
  className?: string
}) {
  const id = useId()
  const span = max - min
  const share = (v: number) => (span > 0 ? Math.min(1, Math.max(0, (v - min) / span)) : 0)
  return (
    <div className={clsx('min-w-0', className)}>
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <label htmlFor={id} className={clsx('shrink-0 font-medium', disabled ? 'text-ink-soft' : 'text-ink')}>
          {label}
        </label>
        {valueText !== undefined && <span className="min-w-0 truncate text-right text-ink-soft tabular-nums">{valueText}</span>}
      </div>
      <div className="relative">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          aria-valuetext={valueText}
          onChange={(e) => onChange(Number(e.target.value))}
          style={{ '--fill': fill ? `${share(value) * 100}%` : '0%' } as CSSProperties}
        />
        {ticks && ticks.length > 0 && (
          <div aria-hidden className="pointer-events-none absolute inset-x-0 top-1/2 h-0">
            {ticks.map((t) => {
              const f = share(t)
              return (
                <span
                  key={t}
                  className="absolute top-3 h-1.5 w-0.5 -translate-x-1/2 rounded-full bg-ink-faint"
                  style={{ left: `calc(${f * 100}% + ${(0.5 - f) * SLIDER_THUMB}px)` }}
                />
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
