// Settings > Account: the username, change password and sign out everywhere.
import { useId, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router'
import { useQueryClient } from '@tanstack/react-query'
import { LogOut } from 'lucide-react'
import { ApiError, api } from '../../../lib/api'
import { clearPrivateCache, loggedOutState, useAuth } from '../../../lib/auth'
import { Button, ConfirmModal } from '../../ui'
import { useToast } from '../../Toast'
import { Field, FormError, PasswordInput, StrengthMeter } from '../AuthCard'
import { Section, errorText } from './common'

const PASSWORD_MIN = 10
const PASSWORD_MAX = 200

export default function AccountTab() {
  const { state, setState } = useAuth()
  const qc = useQueryClient()
  const navigate = useNavigate()
  const toast = useToast()
  const ids = { cur: useId(), pw: useId(), pw2: useId() }
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [confirm, setConfirm] = useState('')
  const [touched, setTouched] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [askAll, setAskAll] = useState(false)
  const [allBusy, setAllBusy] = useState(false)

  const nextError =
    touched && next.length < PASSWORD_MIN
      ? `Use at least ${PASSWORD_MIN} characters.`
      : touched && next.length > PASSWORD_MAX
        ? `Keep it under ${PASSWORD_MAX} characters.`
        : touched && next === current && next
          ? 'The new password is the same as the current one.'
          : null
  const matchError = touched && confirm !== next ? 'The two passwords do not match.' : null

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setTouched(true)
    if (busy || !current || next.length < PASSWORD_MIN || next.length > PASSWORD_MAX || next !== confirm || next === current) return
    setBusy(true)
    setError(null)
    try {
      await api.auth.changePassword(current, next)
      setCurrent('')
      setNext('')
      setConfirm('')
      setTouched(false)
      toast.success('Password changed.')
    } catch (err) {
      if (err instanceof ApiError && err.code === 'login_required') {
        setState(loggedOutState(state))
        return
      }
      setError(
        err instanceof ApiError && (err.status === 400 || err.status === 401 || err.status === 403) && err.code !== 'csrf'
          ? 'The current password is not right.'
          : err instanceof ApiError && err.status === 429
            ? 'Too many attempts. Wait a few minutes, then try again.'
            : errorText(err, 'Could not change the password. Try again.'),
      )
    } finally {
      setBusy(false)
    }
  }

  const logoutAll = async () => {
    setAllBusy(true)
    try {
      const s = await api.auth.logoutAll().catch((err) => {
        if (err instanceof ApiError && err.status === 401) return loggedOutState(state)
        throw err
      })
      clearPrivateCache(qc)
      setState(s.authenticated ? s : loggedOutState(s))
      setAskAll(false)
      if (!s.authenticated) navigate('/login', { replace: true })
      else toast.success('Every other session was signed out.')
    } catch (err) {
      toast.error(errorText(err, 'Could not sign out everywhere. Try again.'))
    } finally {
      setAllBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <Section title="Account" description="This cloud has one account. It owns everything stored here.">
        <dl className="text-sm">
          <dt className="text-ink-soft">Username</dt>
          <dd className="mt-0.5 font-semibold">{state?.username ?? '-'}</dd>
        </dl>
      </Section>

      <Section title="Change password" description="Other signed-in devices stay signed in. Use Sign out everywhere below to end them.">
        <form onSubmit={submit} noValidate className="flex max-w-sm flex-col gap-4">
          {/* lets password managers attach the change to the right account */}
          <input type="text" name="username" autoComplete="username" value={state?.username ?? ''} readOnly hidden />
          <Field id={ids.cur} label="Current password">
            <PasswordInput
              id={ids.cur}
              autoComplete="current-password"
              value={current}
              onChange={(e) => setCurrent(e.target.value)}
              aria-invalid={touched && !current}
            />
          </Field>
          <Field id={ids.pw} label="New password" error={nextError}>
            <PasswordInput
              id={ids.pw}
              autoComplete="new-password"
              maxLength={PASSWORD_MAX}
              value={next}
              onChange={(e) => setNext(e.target.value)}
              aria-invalid={!!nextError}
            />
            <StrengthMeter password={next} />
          </Field>
          <Field id={ids.pw2} label="New password again" error={matchError}>
            <PasswordInput
              id={ids.pw2}
              autoComplete="new-password"
              maxLength={PASSWORD_MAX}
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              aria-invalid={!!matchError}
            />
          </Field>
          <FormError>{error}</FormError>
          <div>
            <Button type="submit" loading={busy}>
              Change password
            </Button>
          </div>
        </form>
      </Section>

      <Section
        title="Sign out everywhere"
        description="Ends every session, including this one. Use it if a device was lost or someone else may have signed in."
        aside={
          <Button variant="danger" onClick={() => setAskAll(true)}>
            <LogOut className="size-4" aria-hidden />
            Sign out everywhere
          </Button>
        }
      />

      <ConfirmModal
        open={askAll}
        title="Sign out everywhere"
        body="Every device, this one included, will need the password or a passkey to sign in again."
        confirmLabel="Sign out everywhere"
        danger
        loading={allBusy}
        onConfirm={() => void logoutAll()}
        onClose={() => setAskAll(false)}
      />
    </div>
  )
}
