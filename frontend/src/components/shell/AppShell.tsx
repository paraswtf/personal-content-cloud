// Session guard and layout for every page except /setup and /login (docs/ARCHITECTURE.md section 14).
//
// While the auth state loads it shows a spinner; `setup_required` sends the browser to /setup and a
// missing session to /login?next=<path>. A 401 from any API call (api.ts onUnauthorized) marks the
// state as logged out, which triggers the same redirect. Inside: the header (desktop navigation), the routed page in <main id="main"> and the bottom tab bar on phones. Pages bring
// their own width and gutters (mx-auto max-w-7xl px-4). ToastProvider wraps the whole router in
// App.tsx, so toasts work here, on the auth pages and in portals alike.
import { Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { Navigate, Outlet, useLocation, useNavigate } from 'react-router'
import { useQueryClient } from '@tanstack/react-query'
import { Cloud } from 'lucide-react'
import { setUnauthorizedHandler } from '../../lib/api'
import { AUTH_KEY, clearPrivateCache, loggedOutState, useAuth } from '../../lib/auth'
import type { AuthState } from '../../lib/types'
import { ErrorState, PageSpinner } from '../ui'
import { useToast } from '../Toast'
import Header from './Header'
import { BottomTabBar } from './Nav'

/** Where /login should send the user back to; only same-app paths, never the auth pages. */
export function safeNext(raw: string | null | undefined): string {
  if (!raw || !raw.startsWith('/') || raw.startsWith('//') || raw.startsWith('/\\')) return '/'
  if (raw.startsWith('/login') || raw.startsWith('/setup')) return '/'
  return raw
}

export default function AppShell() {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const location = useLocation()
  const toast = useToast()
  const { state, loading, error, refresh, logout } = useAuth()
  const leaving = useRef(false)
  const [loggingOut, setLoggingOut] = useState(false)

  // any 401 from the API: the session is gone
  useEffect(() => {
    setUnauthorizedHandler(() => {
      const prev = qc.getQueryData<AuthState>(AUTH_KEY)
      if (prev && !prev.authenticated) return
      qc.setQueryData(AUTH_KEY, loggedOutState(prev))
    })
    return () => setUnauthorizedHandler(null)
  }, [qc])

  const onLogout = useCallback(async () => {
    if (loggingOut) return
    setLoggingOut(true)
    leaving.current = true
    try {
      await logout()
      navigate('/login', { replace: true })
    } catch {
      leaving.current = false
      toast.error('Could not sign out. Try again.')
    } finally {
      setLoggingOut(false)
    }
  }, [loggingOut, logout, navigate, toast])

  // when the session ends while the page is open, drop cached private data after leaving the shell
  const authenticated = state?.authenticated === true
  useEffect(() => {
    if (state && !authenticated) {
      const t = window.setTimeout(() => clearPrivateCache(qc), 0)
      return () => window.clearTimeout(t)
    }
  }, [state, authenticated, qc])

  if (loading) {
    return (
      <div className="min-h-dvh bg-bg">
        <PageSpinner />
      </div>
    )
  }
  if (!state) {
    return (
      <main className="flex min-h-dvh flex-col items-center justify-center bg-bg px-4">
        <span className="mb-3 flex size-12 items-center justify-center rounded-xl bg-ink text-bg">
          <Cloud className="size-6" aria-hidden />
        </span>
        <h1 className="text-lg font-semibold">The server could not be reached</h1>
        <ErrorState error={error} onRetry={() => void refresh()} />
      </main>
    )
  }
  if (state.setup_required) return <Navigate to="/setup" replace />
  if (!state.authenticated) {
    if (leaving.current) return <Navigate to="/login" replace />
    const next = location.pathname + location.search
    return <Navigate to={next === '/' ? '/login' : `/login?next=${encodeURIComponent(next)}`} replace />
  }

  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only z-50 rounded-full bg-surface px-4 py-2 text-sm font-semibold shadow focus:not-sr-only focus:fixed focus:left-4 focus:top-3"
      >
        Skip to content
      </a>
      <Header onLogout={() => void onLogout()} loggingOut={loggingOut} />
      <main id="main" className="flex-1 pb-[calc(var(--tabbar-h)+env(safe-area-inset-bottom))] md:pb-0">
        {/* pages are lazy chunks: suspend inside main so the header stays while one loads */}
        <Suspense fallback={<PageSpinner />}>
          <Outlet />
        </Suspense>
      </main>
      <BottomTabBar onLogout={() => void onLogout()} />
    </div>
  )
}
