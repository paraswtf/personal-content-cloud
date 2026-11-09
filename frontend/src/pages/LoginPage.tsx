// /login: passkey first (when the browser can use one), password below. `next` brings the user back
// to the page that sent them here. 401 shows a plain "wrong details" message, 429 the rate limit.
// On a custom domain where this browser cannot use the passkeys, a note links to the server's own address.
import { useEffect, useId, useState, type FormEvent } from 'react'
import { Navigate, useNavigate, useSearchParams } from 'react-router'
import { useQueryClient } from '@tanstack/react-query'
import { Fingerprint, LogIn } from 'lucide-react'
import { ApiError, api } from '../lib/api'
import { authErrorMessage, clearPrivateCache, loginWithPasskey, usePasskeyPlace, useAuth } from '../lib/auth'
import type { AuthState } from '../lib/types'
import { Button, PageSpinner } from '../components/ui'
import { inputClass } from '../components/admin/fields'
import { AuthCard, Field, FormError, PasswordInput } from '../components/shell/AuthCard'
import PasskeyElsewhere from '../components/shell/PasskeyElsewhere'
import { safeNext } from '../components/shell/AppShell'
import { usePageTitle } from '../components/shell/title'

function loginError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 401 || err.status === 400) return 'Wrong username or password.'
    if (err.status === 429) return 'Too many failed attempts. Wait ten minutes, then try again.'
  }
  return authErrorMessage(err, 'Could not sign in. Try again.')
}

export default function LoginPage() {
  usePageTitle('Sign in')
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [params] = useSearchParams()
  const next = safeNext(params.get('next'))
  const { state, loading, setState } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState<'password' | 'passkey' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const ids = { user: useId(), pw: useId() }
  const place = usePasskeyPlace()
  const canPasskey = place.kind !== 'unsupported'
  // a passkey button that can work (or may, while the check runs)
  const passkeyHere = place.kind === 'here' || place.kind === 'checking'

  // data cached by an earlier session must not leak into the next one
  useEffect(() => {
    clearPrivateCache(qc)
  }, [qc])

  if (loading) return <PageSpinner />
  if (state?.setup_required) return <Navigate to="/setup" replace />
  if (state?.authenticated) return <Navigate to={next} replace />

  const done = (s: AuthState) => {
    if (!s.authenticated) {
      setError('Could not sign in. Try again.')
      setBusy(null)
      return
    }
    setState(s)
    navigate(next, { replace: true })
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (busy || !username.trim() || !password) return
    setBusy('password')
    setError(null)
    try {
      done(await api.auth.login(username.trim(), password))
    } catch (err) {
      if (err instanceof ApiError && err.code === 'setup_required') {
        navigate('/setup', { replace: true })
        return
      }
      setError(loginError(err))
      setBusy(null)
    }
  }

  const passkey = async () => {
    if (busy) return
    setBusy('passkey')
    setError(null)
    try {
      done(await loginWithPasskey())
    } catch (err) {
      setError(
        err instanceof ApiError && (err.status === 401 || err.status === 400)
          ? 'That passkey is not registered here. Sign in with your password, then add it in Settings.'
          : authErrorMessage(err, 'Could not sign in with a passkey. Try again.'),
      )
      setBusy(null)
    }
  }

  const hasPasskeys = (state?.passkeys ?? 0) > 0

  return (
    <AuthCard title="Sign in" subtitle={state?.site_title || 'Personal Content Cloud'}>
      <div className="flex flex-col gap-5">
        {place.kind === 'elsewhere' && (
          <PasskeyElsewhere home={place.home} lead="Sign in with a passkey at" />
        )}
        {passkeyHere && (
          <div className="flex flex-col gap-2">
            <Button
              type="button"
              size="lg"
              variant={hasPasskeys ? 'primary' : 'secondary'}
              loading={busy === 'passkey'}
              disabled={busy === 'password' || place.kind === 'checking'}
              onClick={() => void passkey()}
              className="w-full"
            >
              {busy !== 'passkey' && <Fingerprint className="size-5" aria-hidden />}
              Sign in with a passkey
            </Button>
            {!hasPasskeys && (
              <p className="text-xs text-ink-soft">No passkey is set up yet. Sign in with your password, then add one in Settings.</p>
            )}
          </div>
        )}
        {canPasskey && (
          <div className="flex items-center gap-3 text-xs text-ink-faint" aria-hidden>
            <span className="h-px flex-1 bg-line" />
            or with your password
            <span className="h-px flex-1 bg-line" />
          </div>
        )}
        <form onSubmit={submit} className="flex flex-col gap-4">
          <Field id={ids.user} label="Username">
            <input
              id={ids.user}
              name="username"
              autoComplete="username webauthn"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              autoFocus={!hasPasskeys}
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className={inputClass}
            />
          </Field>
          <Field id={ids.pw} label="Password">
            <PasswordInput
              id={ids.pw}
              name="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </Field>
          <FormError>{error}</FormError>
          <Button
            type="submit"
            size="lg"
            variant={passkeyHere && hasPasskeys ? 'secondary' : 'primary'}
            loading={busy === 'password'}
            disabled={!username.trim() || !password || busy === 'passkey'}
            className="w-full"
          >
            {busy !== 'password' && <LogIn className="size-4" aria-hidden />}
            Sign in
          </Button>
        </form>
      </div>
    </AuthCard>
  )
}
