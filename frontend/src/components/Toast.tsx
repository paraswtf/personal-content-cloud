// App-wide toasts (moved from the hub admin). ToastProvider wraps the app shell; useToast() gives
// success, error and info. At most four toasts show at once; they last 5 s (errors 9 s).
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react'
import clsx from 'clsx'
import { CheckCircle2, Info, TriangleAlert, X } from 'lucide-react'

type Tone = 'ok' | 'error' | 'info'
interface ToastItem {
  id: number
  tone: Tone
  text: string
}
interface ToastApi {
  success: (text: string) => void
  error: (text: string) => void
  info: (text: string) => void
}

const ToastContext = createContext<ToastApi | null>(null)

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([])
  const nextId = useRef(1)

  const dismiss = useCallback((id: number) => setItems((cur) => cur.filter((t) => t.id !== id)), [])
  const push = useCallback(
    (tone: Tone, text: string) => {
      const id = nextId.current++
      setItems((cur) => [...cur.slice(-3), { id, tone, text }])
      window.setTimeout(() => dismiss(id), tone === 'error' ? 9000 : 5000)
    },
    [dismiss],
  )
  const api = useMemo<ToastApi>(
    () => ({
      success: (t) => push('ok', t),
      error: (t) => push('error', t),
      info: (t) => push('info', t),
    }),
    [push],
  )

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        aria-live="polite"
        className="pointer-events-none fixed inset-x-4 top-4 z-[60] flex flex-col items-stretch gap-2 sm:left-auto sm:right-4 sm:w-96"
      >
        {items.map((t) => (
          <div
            key={t.id}
            role={t.tone === 'error' ? 'alert' : 'status'}
            className={clsx(
              'pointer-events-auto flex items-start gap-3 rounded-2xl border bg-surface px-4 py-3 text-sm shadow-lg',
              t.tone === 'error' ? 'border-danger/40' : 'border-line',
            )}
          >
            {t.tone === 'ok' && <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-ok" aria-hidden />}
            {t.tone === 'error' && <TriangleAlert className="mt-0.5 size-4 shrink-0 text-danger" aria-hidden />}
            {t.tone === 'info' && <Info className="mt-0.5 size-4 shrink-0 text-ink-soft" aria-hidden />}
            <p className="min-w-0 flex-1 break-words">{t.text}</p>
            <button
              onClick={() => dismiss(t.id)}
              className="-m-1 rounded-full p-1 text-ink-faint hover:bg-ink/5 hover:text-ink"
              aria-label="Dismiss message"
            >
              <X className="size-4" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used inside ToastProvider')
  return ctx
}
