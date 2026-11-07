// Vault unlock form: the password, or a passkey when the account has one. The vault page shows it full
// size while locked; Settings shows it compact inside a dialog when a passkey change needs the vault
// open. `onUnlocked` gets the AuthState the unlock endpoint returned.
import { useId, useState, type FormEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Fingerprint, Lock, LockKeyhole } from 'lucide-react'
import { api, ApiError } from '../../lib/api'
import type { AuthState } from '../../lib/types'
import { startPasskeyAssertion } from '../../lib/auth'
import { errorMessage } from '../admin/adminHooks'
import { Button } from '../ui'

export default function UnlockForm({
  onUnlocked,
  hasPasskeys,
  compact = false,
  description,
}: {
  onUnlocked: (s: AuthState) => void
  hasPasskeys: boolean
  /** Inside a dialog: no large icon or heading, just the optional description and the form. */
  compact?: boolean
  description?: string
}) {
  const passwordId = useId()
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const withPassword = useMutation({
    mutationFn: () => api.auth.vaultUnlockPassword(password),
    onSuccess: (s) => {
      setPassword('')
      onUnlocked(s)
    },
    onError: (e) => setError(e instanceof ApiError && e.status === 401 ? 'That password is not right.' : errorMessage(e, 'Could not unlock.')),
  })
  const withPasskey = useMutation({
    mutationFn: async () => {
      const { challenge_id, credential } = await startPasskeyAssertion()
      return api.auth.vaultUnlockPasskey(challenge_id, credential)
    },
    onSuccess: (s) => onUnlocked(s),
    onError: (e) =>
      setError(
        e instanceof DOMException && (e.name === 'NotAllowedError' || e.name === 'AbortError')
          ? 'The passkey request was cancelled.'
          : errorMessage(e, 'Could not unlock with the passkey.'),
      ),
  })
  const busy = withPassword.isPending || withPasskey.isPending

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!password || busy) return
    setError(null)
    withPassword.mutate()
  }

  const form = (
    <form onSubmit={submit} className={compact ? 'flex w-full flex-col gap-3 text-left' : 'mt-6 flex w-full flex-col gap-3 text-left'}>
      <label htmlFor={passwordId} className="text-sm font-medium">
        Password
      </label>
      <input
        id={passwordId}
        type="password"
        autoComplete="current-password"
        autoFocus={compact}
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        aria-invalid={!!error}
        className="h-11 w-full rounded-full border border-line bg-surface px-4 text-base text-ink hover:border-ink-faint focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
      />
      {error && (
        <p className="text-sm text-danger" role="alert">
          {error}
        </p>
      )}
      <Button type="submit" size="lg" loading={withPassword.isPending} disabled={!password || busy}>
        <Lock className="size-4" aria-hidden />
        Unlock
      </Button>
      {hasPasskeys && (
        <Button
          type="button"
          size="lg"
          variant="secondary"
          loading={withPasskey.isPending}
          disabled={busy}
          onClick={() => {
            setError(null)
            withPasskey.mutate()
          }}
        >
          <Fingerprint className="size-4" aria-hidden />
          Use passkey
        </Button>
      )}
    </form>
  )

  if (compact)
    return (
      <div className="flex flex-col gap-3">
        {description && <p className="text-sm text-ink-soft">{description}</p>}
        {form}
      </div>
    )

  return (
    <div className="mx-auto flex w-full max-w-sm flex-col items-center px-4 py-16 text-center">
      <span className="flex size-16 items-center justify-center rounded-2xl bg-surface-2 text-ink">
        <LockKeyhole className="size-8" aria-hidden />
      </span>
      <h1 className="mt-4 text-2xl font-bold tracking-tight">The vault is locked</h1>
      <p className="mt-2 text-sm text-ink-soft">
        {description ?? 'Private items are kept out of the library, search and people. Confirm it is you to see them for a while.'}
      </p>
      {form}
    </div>
  )
}
