// Settings > Sessions: every signed-in browser, this one marked. Others can be revoked one by one;
// revoking this one is the same as signing out.
import { useState } from 'react'
import { useNavigate } from 'react-router'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Laptop, Smartphone } from 'lucide-react'
import { api } from '../../../lib/api'
import { SESSIONS_KEY, useAuth } from '../../../lib/auth'
import { formatDate } from '../../../lib/format'
import type { SessionOut } from '../../../lib/types'
import { Badge, Button, ConfirmModal, ErrorState, Spinner } from '../../ui'
import { useToast } from '../../Toast'
import { Section, describeAgent, errorText } from './common'

function isPhone(ua: string | null): boolean {
  return !!ua && /iPhone|Android|Mobile/.test(ua)
}

export default function SessionsTab() {
  const qc = useQueryClient()
  const toast = useToast()
  const navigate = useNavigate()
  const { logout } = useAuth()
  const q = useQuery({ queryKey: SESSIONS_KEY, queryFn: api.auth.sessions })
  const [revoking, setRevoking] = useState<SessionOut | null>(null)
  const [busy, setBusy] = useState(false)

  const confirm = async () => {
    if (!revoking) return
    setBusy(true)
    try {
      if (revoking.current) {
        await logout()
        navigate('/login', { replace: true })
        return
      }
      await api.auth.revokeSession(revoking.id)
      toast.success(`${describeAgent(revoking.user_agent)} was signed out.`)
      setRevoking(null)
      void qc.invalidateQueries({ queryKey: SESSIONS_KEY })
    } catch (err) {
      toast.error(errorText(err, 'Could not end that session.'))
    } finally {
      setBusy(false)
    }
  }

  const rows = [...(q.data ?? [])].sort((a, b) => Number(b.current) - Number(a.current) || b.last_seen_at.localeCompare(a.last_seen_at))

  return (
    <Section title="Signed-in sessions" description="Sessions last 30 days from their last use. End any you do not recognise.">
      {q.isPending ? (
        <div className="flex justify-center py-6">
          <Spinner label="Loading sessions" />
        </div>
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : (
        <ul className="divide-y divide-line">
          {rows.map((s) => {
            const Icon = isPhone(s.user_agent) ? Smartphone : Laptop
            return (
              <li key={s.id} className="flex items-center gap-3 py-3">
                <span className="flex size-10 shrink-0 items-center justify-center rounded-full bg-surface-2 text-ink-soft">
                  <Icon className="size-5" aria-hidden />
                </span>
                <div className="min-w-0 flex-1">
                  <p className="flex flex-wrap items-center gap-2 text-sm font-semibold">
                    {describeAgent(s.user_agent)}
                    {s.current && <Badge tone="ok">This device</Badge>}
                  </p>
                  <p className="text-xs text-ink-soft">
                    {s.ip ? `${s.ip}, ` : ''}signed in {formatDate(s.created_at)}, last active {formatDate(s.last_seen_at, true)}
                  </p>
                </div>
                <Button variant={s.current ? 'secondary' : 'danger'} size="sm" className="h-10" onClick={() => setRevoking(s)}>
                  {s.current ? 'Sign out' : 'Revoke'}
                </Button>
              </li>
            )
          })}
        </ul>
      )}
      <ConfirmModal
        open={!!revoking}
        title={revoking?.current ? 'Sign out of this device' : 'Revoke session'}
        body={
          revoking?.current
            ? 'You will need the password or a passkey to sign in again.'
            : `${describeAgent(revoking?.user_agent)} will be signed out at once.`
        }
        confirmLabel={revoking?.current ? 'Sign out' : 'Revoke'}
        danger={!revoking?.current}
        loading={busy}
        onConfirm={() => void confirm()}
        onClose={() => setRevoking(null)}
      />
    </Section>
  )
}
