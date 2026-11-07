// Shown in the header while the vault is unlocked: a link to /vault with the time left, and a button
// that locks it again at once. The countdown ticks every second; at zero the auth state is refetched
// so every page sees the vault as locked.
import { useEffect, useState } from 'react'
import { Link } from 'react-router'
import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Lock, LockOpen } from 'lucide-react'
import { api } from '../../lib/api'
import { AUTH_KEY, useAuth } from '../../lib/auth'
import { useToast } from '../Toast'

function remaining(until: string | null, now: number): number {
  if (!until) return 0
  const t = Date.parse(until)
  return Number.isFinite(t) ? Math.max(0, Math.floor((t - now) / 1000)) : 0
}

function mmss(s: number): string {
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

export default function VaultBadge({ className }: { className?: string }) {
  const { state, vaultUnlocked, setState } = useAuth()
  const qc = useQueryClient()
  const toast = useToast()
  const until = state?.vault_until ?? null
  const [now, setNow] = useState(() => Date.now())
  const [locking, setLocking] = useState(false)

  useEffect(() => {
    if (!vaultUnlocked) return
    setNow(Date.now())
    const t = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(t)
  }, [vaultUnlocked, until])

  const left = remaining(until, now)
  useEffect(() => {
    if (vaultUnlocked && until && left === 0) void qc.invalidateQueries({ queryKey: AUTH_KEY })
  }, [vaultUnlocked, until, left, qc])

  if (!vaultUnlocked || left === 0) return null

  const lock = async () => {
    setLocking(true)
    try {
      setState(await api.auth.vaultLock())
      // drop private data fetched while unlocked
      qc.removeQueries({ predicate: (q) => JSON.stringify(q.queryKey).includes('"vault":true') || q.queryKey[0] === 'vault' })
      toast.info('The vault is locked.')
    } catch {
      toast.error('Could not lock the vault. Try again.')
    } finally {
      setLocking(false)
    }
  }

  return (
    <div
      className={clsx(
        'inline-flex h-10 shrink-0 items-center rounded-full bg-warn/10 text-warn ring-1 ring-warn/25',
        className,
      )}
    >
      <Link
        to="/vault"
        className="inline-flex h-10 items-center gap-1.5 rounded-l-full pl-3 pr-2 text-sm font-semibold"
        title="The vault is unlocked"
      >
        <LockOpen className="size-4" aria-hidden />
        <span className="sr-only">Vault unlocked, </span>
        <span className="tabular-nums">
          {mmss(left)}
        </span>
      </Link>
      <button
        type="button"
        onClick={() => void lock()}
        disabled={locking}
        aria-label="Lock the vault now"
        title="Lock the vault now"
        className="inline-flex size-10 items-center justify-center rounded-r-full hover:bg-warn/10 disabled:opacity-50"
      >
        <Lock className="size-4" aria-hidden />
      </button>
    </div>
  )
}
