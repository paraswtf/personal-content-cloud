// Small pieces shared by the Settings tabs.
import type { ReactNode } from 'react'
import clsx from 'clsx'
import { ApiError } from '../../../lib/api'

export function Section({
  title,
  description,
  children,
  aside,
  className,
}: {
  title: string
  description?: ReactNode
  children?: ReactNode
  aside?: ReactNode
  className?: string
}) {
  return (
    <section className={clsx('rounded-[var(--radius-card)] bg-surface p-4 shadow-[var(--shadow-card)] sm:p-5', className)}>
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div className="min-w-0">
          <h2 className="text-base font-semibold">{title}</h2>
          {description && <div className="mt-1 max-w-2xl text-sm text-ink-soft">{description}</div>}
        </div>
        {aside}
      </div>
      {children && <div className="mt-4">{children}</div>}
    </section>
  )
}

export function errorText(e: unknown, fallback = 'Something went wrong. Try again.'): string {
  if (e instanceof ApiError && e.status === 0) return 'The server could not be reached.'
  return e instanceof Error && e.message ? e.message : fallback
}

/** "Chrome on Android", "Safari on iPhone", from a user agent string (best effort). */
export function describeAgent(ua: string | null | undefined): string {
  if (!ua) return 'Unknown device'
  const browser = /Edg\//.test(ua)
    ? 'Edge'
    : /OPR\/|Opera/.test(ua)
      ? 'Opera'
      : /SamsungBrowser/.test(ua)
        ? 'Samsung Internet'
        : /Firefox\/|FxiOS/.test(ua)
          ? 'Firefox'
          : /CriOS|Chrome\//.test(ua)
            ? 'Chrome'
            : /Safari\//.test(ua)
              ? 'Safari'
              : /curl|python|httpx|okhttp/i.test(ua)
                ? 'Script'
                : 'Browser'
  const os = /iPhone/.test(ua)
    ? 'iPhone'
    : /iPad/.test(ua)
      ? 'iPad'
      : /Android/.test(ua)
        ? 'Android'
        : /CrOS/.test(ua)
          ? 'ChromeOS'
          : /Mac OS X|Macintosh/.test(ua)
            ? 'Mac'
            : /Windows/.test(ua)
              ? 'Windows'
              : /Linux/.test(ua)
                ? 'Linux'
                : ''
  return os ? `${browser} on ${os}` : browser
}

export function Dot({ tone }: { tone: 'ok' | 'warn' | 'danger' | 'idle' }) {
  return (
    <span
      aria-hidden
      className={clsx(
        'inline-block size-2 shrink-0 rounded-full',
        tone === 'ok' ? 'bg-ok' : tone === 'warn' ? 'bg-warn' : tone === 'danger' ? 'bg-danger' : 'bg-ink-faint',
      )}
    />
  )
}
