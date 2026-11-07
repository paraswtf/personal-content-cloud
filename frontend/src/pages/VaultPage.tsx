// Vault (/vault): private items. Locked, it asks for the password or a passkey again and requests
// nothing else (no lists, no thumbnails). Unlocked, it shows the library of private items with the
// minutes left and a Lock now button; when the window ends every vault query is dropped.
import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { useMutation, useQueryClient, type Query } from '@tanstack/react-query'
import { Fingerprint, Lock, LockKeyhole, ShieldCheck } from 'lucide-react'
import { api, ApiError } from '../lib/api'
import type { AuthState } from '../lib/types'
import { passkeysSupported, startPasskeyAssertion, useAuth } from '../lib/auth'
import { formatMinutes } from '../lib/format'
import { usePageTitle } from '../lib/hooks'
import Library from '../components/library/Library'
import { useToast } from '../components/Toast'
import { errorMessage } from '../components/admin/adminHooks'
import { Button } from '../components/ui'

/** Query keys that hold vault data: lists and searches with vault set, details fetched with vault. */
function isVaultQuery(q: Query): boolean {
  const k = q.queryKey
  if ((k[0] === 'item' || k[0] === 'item-text') && k[2] === true) return true
  return k.some((part) => !!part && typeof part === 'object' && (part as { vault?: unknown }).vault === true)
}

function useMinutesLeft(until: string | null | undefined): number | null {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!until) return
    const t = window.setInterval(() => setNow(Date.now()), 15_000)
    return () => window.clearInterval(t)
  }, [until])
  if (!until) return null
  const end = new Date(until).getTime()
  if (Number.isNaN(end)) return null
  return (end - now) / 60_000
}

function UnlockForm({ onUnlocked, hasPasskeys }: { onUnlocked: (s: AuthState) => void; hasPasskeys: boolean }) {
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

  return (
    <div className="mx-auto flex w-full max-w-sm flex-col items-center px-4 py-16 text-center">
      <span className="flex size-16 items-center justify-center rounded-2xl bg-surface-2 text-ink">
        <LockKeyhole className="size-8" aria-hidden />
      </span>
      <h1 className="mt-4 text-2xl font-bold tracking-tight">The vault is locked</h1>
      <p className="mt-2 text-sm text-ink-soft">
        Private items are kept out of the library, search and people. Confirm it is you to see them for a while.
      </p>
      <form onSubmit={submit} className="mt-6 flex w-full flex-col gap-3 text-left">
        <label htmlFor="vault-password" className="text-sm font-medium">
          Password
        </label>
        <input
          id="vault-password"
          type="password"
          autoComplete="current-password"
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
    </div>
  )
}

export default function VaultPage() {
  usePageTitle('Vault')
  const { state, refresh, vaultUnlocked, setState } = useAuth()
  const qc = useQueryClient()
  const toast = useToast()
  const minutes = useMinutesLeft(vaultUnlocked ? state?.vault_until : null)
  const open = vaultUnlocked && (minutes === null || minutes > 0)

  const dropVaultData = useCallback(() => {
    void qc.cancelQueries({ predicate: isVaultQuery })
    qc.removeQueries({ predicate: isVaultQuery })
  }, [qc])

  // The window ran out while the page was open: forget everything and ask again.
  useEffect(() => {
    if (vaultUnlocked && minutes !== null && minutes <= 0) {
      dropVaultData()
      void refresh()
    }
  }, [vaultUnlocked, minutes, dropVaultData, refresh])
  // Locked from elsewhere (another tab, a 403): nothing private stays in memory.
  useEffect(() => {
    if (!vaultUnlocked) dropVaultData()
  }, [vaultUnlocked, dropVaultData])
  // Leaving the page drops the vault data too.
  useEffect(() => dropVaultData, [dropVaultData])

  // A vault request answered 403 vault_locked: the server window ended first.
  useEffect(() => {
    if (!open) return
    return qc.getQueryCache().subscribe((ev) => {
      const err = ev.query.state.error
      if (err instanceof ApiError && err.code === 'vault_locked') {
        dropVaultData()
        void refresh()
      }
    })
  }, [open, qc, dropVaultData, refresh])

  const lock = useMutation({
    mutationFn: () => api.auth.vaultLock(),
    onSuccess: (s) => {
      dropVaultData()
      setState(s)
      toast.success('Vault locked')
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not lock the vault.')),
  })

  if (!open) return <UnlockForm onUnlocked={setState} hasPasskeys={(state?.passkeys ?? 0) > 0 && passkeysSupported()} />

  return (
    <Library
      vault
      title="Vault"
      headerExtra={
        <>
          <span className="inline-flex h-10 items-center gap-1.5 rounded-full bg-surface-2 px-3 text-sm text-ink-soft" aria-live="polite">
            <ShieldCheck className="size-4 text-ok" aria-hidden />
            {minutes !== null ? `Unlocked for ${formatMinutes(minutes)}` : 'Unlocked'}
          </span>
          <Button variant="secondary" loading={lock.isPending} onClick={() => lock.mutate()}>
            <Lock className="size-4" aria-hidden />
            Lock now
          </Button>
        </>
      }
    />
  )
}
