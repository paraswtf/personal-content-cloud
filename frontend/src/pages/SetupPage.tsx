// /setup: first run. Creates the single account (username 1..64, password 10..200 characters) and
// goes to the library. Once an account exists the page sends the visitor to /login (or / when signed in).
// On a custom domain where this browser cannot use passkeys, the passkey line becomes a link to the
// server's own address.
import { useId, useState, type FormEvent } from 'react'
import { Navigate, useNavigate } from 'react-router'
import { ApiError, api } from '../lib/api'
import { authErrorMessage, usePasskeyPlace, useAuth } from '../lib/auth'
import { Button, PageSpinner } from '../components/ui'
import { inputClass } from '../components/admin/fields'
import { AuthCard, Field, FormError, PasswordInput, StrengthMeter } from '../components/shell/AuthCard'
import PasskeyElsewhere from '../components/shell/PasskeyElsewhere'
import { usePageTitle } from '../components/shell/title'

export const USERNAME_MAX = 64
export const PASSWORD_MIN = 10
export const PASSWORD_MAX = 200

export default function SetupPage() {
  usePageTitle('Set up')
  const navigate = useNavigate()
  const { state, loading, setState } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [touched, setTouched] = useState(false)
  const ids = { user: useId(), pw: useId(), pw2: useId() }
  const place = usePasskeyPlace()

  if (loading) return <PageSpinner />
  if (state && !state.setup_required) return <Navigate to={state.authenticated ? '/' : '/login'} replace />

  const name = username.trim()
  const userError = touched && (!name ? 'Choose a username.' : name.length > USERNAME_MAX ? `Keep it under ${USERNAME_MAX} characters.` : null)
  const pwError =
    touched && password.length < PASSWORD_MIN
      ? `Use at least ${PASSWORD_MIN} characters.`
      : touched && password.length > PASSWORD_MAX
        ? `Keep it under ${PASSWORD_MAX} characters.`
        : null
  const matchError = touched && confirm !== password ? 'The two passwords do not match.' : null

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setTouched(true)
    if (busy) return
    if (!name || name.length > USERNAME_MAX || password.length < PASSWORD_MIN || password.length > PASSWORD_MAX || confirm !== password)
      return
    setBusy(true)
    setError(null)
    try {
      let s = await api.auth.setup(name, password)
      if (!s.authenticated) s = await api.auth.login(name, password)
      setState(s)
      navigate('/', { replace: true })
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        // someone finished setup meanwhile: sign in instead
        navigate('/login', { replace: true })
        return
      }
      setError(authErrorMessage(err, 'Could not create the account. Try again.'))
      setBusy(false)
    }
  }

  return (
    <AuthCard title="Set up your cloud" subtitle="Create the one account that owns everything stored here.">
      <form onSubmit={submit} noValidate className="flex flex-col gap-4">
        <Field id={ids.user} label="Username" error={userError || null}>
          <input
            id={ids.user}
            name="username"
            autoComplete="username"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            autoFocus
            maxLength={USERNAME_MAX}
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            aria-invalid={!!userError}
            className={inputClass}
          />
        </Field>
        <Field id={ids.pw} label="Password" error={pwError}>
          <PasswordInput
            id={ids.pw}
            name="new-password"
            autoComplete="new-password"
            maxLength={PASSWORD_MAX}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            aria-invalid={!!pwError}
          />
          <StrengthMeter password={password} />
        </Field>
        <Field id={ids.pw2} label="Password again" error={matchError}>
          <PasswordInput
            id={ids.pw2}
            name="confirm-password"
            autoComplete="new-password"
            maxLength={PASSWORD_MAX}
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            aria-invalid={!!matchError}
          />
        </Field>
        <FormError>{error}</FormError>
        <Button type="submit" size="lg" loading={busy} className="w-full">
          Create account
        </Button>
        {place.kind === 'elsewhere' ? (
          <PasskeyElsewhere home={place.home} lead="To add one after setup, sign in at" />
        ) : (
          <p className="text-xs text-ink-soft">
            After this you can add a passkey in Settings, so this device signs in with a fingerprint, face or screen lock.
          </p>
        )}
      </form>
    </AuthCard>
  )
}
