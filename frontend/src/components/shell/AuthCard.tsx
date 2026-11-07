// Centered card used by /setup and /login (outside the app shell), plus the password field with a
// show/hide toggle that both pages and Settings use.
import { useId, useState, type InputHTMLAttributes, type ReactNode } from 'react'
import clsx from 'clsx'
import { Cloud, Eye, EyeOff } from 'lucide-react'
import { inputClass } from '../admin/fields'
import ThemeToggle from './ThemeToggle'

export function AuthCard({ title, subtitle, children }: { title: string; subtitle?: ReactNode; children: ReactNode }) {
  return (
    <div className="flex min-h-dvh flex-col bg-bg">
      <div className="flex justify-end px-4 pt-[max(0.75rem,env(safe-area-inset-top))]">
        <ThemeToggle />
      </div>
      <main id="main" className="flex flex-1 flex-col items-center justify-center px-4 pb-16 pt-4">
        <div className="w-full max-w-sm">
          <div className="mb-6 flex flex-col items-center gap-3 text-center">
            <span className="flex size-12 items-center justify-center rounded-xl bg-ink text-bg">
              <Cloud className="size-6" aria-hidden />
            </span>
            <div>
              <h1 className="text-xl font-bold tracking-tight">{title}</h1>
              {subtitle && <p className="mt-1 text-sm text-ink-soft">{subtitle}</p>}
            </div>
          </div>
          <div className="rounded-[var(--radius-card)] bg-surface p-5 shadow-[var(--shadow-card)] sm:p-6">{children}</div>
        </div>
      </main>
    </div>
  )
}

export function Field({
  label,
  hint,
  error,
  children,
  id,
}: {
  label: string
  hint?: ReactNode
  error?: string | null
  children: ReactNode
  id: string
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-sm font-medium">
        {label}
      </label>
      {children}
      {hint && !error && <p className="text-xs text-ink-soft">{hint}</p>}
      {error && (
        <p className="text-xs font-medium text-danger" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}

export function PasswordInput({
  className,
  ...rest
}: Omit<InputHTMLAttributes<HTMLInputElement>, 'type'>) {
  const [show, setShow] = useState(false)
  return (
    <div className="relative">
      <input type={show ? 'text' : 'password'} className={clsx(inputClass, 'pr-12', className)} {...rest} />
      <button
        type="button"
        onClick={() => setShow((v) => !v)}
        aria-label={show ? 'Hide password' : 'Show password'}
        aria-pressed={show}
        className="absolute right-0 top-1/2 flex size-10 -translate-y-1/2 items-center justify-center rounded-full text-ink-faint hover:text-ink"
      >
        {show ? <EyeOff className="size-4" aria-hidden /> : <Eye className="size-4" aria-hidden />}
      </button>
    </div>
  )
}

export function FormError({ children }: { children: ReactNode }) {
  if (!children) return null
  return (
    <p role="alert" className="rounded-xl bg-danger/10 px-3 py-2 text-sm text-danger">
      {children}
    </p>
  )
}

/** Rough strength of a new password, for the hint under the field (the server only checks length). */
export function passwordStrength(pw: string): { score: 0 | 1 | 2 | 3; label: string } {
  if (!pw) return { score: 0, label: '' }
  let classes = 0
  if (/[a-z]/.test(pw)) classes++
  if (/[A-Z]/.test(pw)) classes++
  if (/[0-9]/.test(pw)) classes++
  if (/[^A-Za-z0-9]/.test(pw)) classes++
  const unique = new Set(pw).size
  if (pw.length < 10) return { score: 0, label: 'Too short: use at least 10 characters.' }
  if (unique < 5 || (pw.length < 12 && classes < 2)) return { score: 1, label: 'Weak: add length or mix in other kinds of characters.' }
  if (pw.length >= 16 || (pw.length >= 12 && classes >= 3)) return { score: 3, label: 'Strong.' }
  return { score: 2, label: 'Fair: a few more characters make it stronger.' }
}

export function StrengthMeter({ password }: { password: string }) {
  const id = useId()
  const { score, label } = passwordStrength(password)
  if (!password) return null
  const tone = score >= 3 ? 'bg-ok' : score === 2 ? 'bg-warn' : 'bg-danger'
  return (
    <div className="flex flex-col gap-1">
      <div className="grid grid-cols-3 gap-1" aria-hidden>
        {[1, 2, 3].map((n) => (
          <span key={n} className={clsx('h-1 rounded-full', score >= n ? tone : 'bg-ink/10')} />
        ))}
      </div>
      <p id={id} className="text-xs text-ink-soft" aria-live="polite">
        {label}
      </p>
    </div>
  )
}
