// Vault (/vault): private items. Locked, it asks for the password or a passkey again and requests
// nothing else (no lists, no thumbnails). Unlocked, it shows the library of private items with the
// minutes left and a Lock now button; when the window ends every vault query is dropped.
import { useCallback, useEffect, useState } from 'react'
import { useMutation, useQueryClient, type Query } from '@tanstack/react-query'
import { Lock, ShieldCheck } from 'lucide-react'
import { api, ApiError } from '../lib/api'
import { passkeysSupported, useAuth } from '../lib/auth'
import { formatMinutes } from '../lib/format'
import { usePageTitle } from '../lib/hooks'
import Library from '../components/library/Library'
import { useToast } from '../components/Toast'
import { errorMessage } from '../components/admin/adminHooks'
import { Button } from '../components/ui'
import UnlockForm from '../components/vault/UnlockForm'

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
