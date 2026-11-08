// Settings > Sharing (docs/ARCHITECTURE.md section 26): every share link with its status, views and
// expiry, and Copy, Edit (title, password, expiry, downloads) and Revoke.
import { useEffect, useId, useState, type FormEvent } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Copy, Link2, Lock } from 'lucide-react'
import { formatDate, formatRelative } from '../../../lib/format'
import {
  EXPIRY_CHOICES,
  SHARES_KEY,
  copyText,
  expiryText,
  shareErrorText,
  shareStatus,
  shareUrl,
  shares,
  type ShareOut,
  type ShareUpdate,
} from '../../../lib/shares'
import { inputClass } from '../../admin/fields'
import { useToast } from '../../Toast'
import { Badge, Button, ConfirmModal, ErrorState, Modal, Spinner } from '../../ui'
import { Section } from './common'

function describe(s: ShareOut): string {
  const what = s.kind === 'album' ? `Album${s.album ? `: ${s.album.name}` : ''}` : `${s.count} ${s.count === 1 ? 'item' : 'items'}`
  return s.kind === 'album' ? `${what} (${s.count} ${s.count === 1 ? 'item' : 'items'})` : what
}

export default function SharingTab() {
  const qc = useQueryClient()
  const toast = useToast()
  const q = useQuery({ queryKey: [...SHARES_KEY], queryFn: shares.list })
  const [editing, setEditing] = useState<ShareOut | null>(null)
  const [revoking, setRevoking] = useState<ShareOut | null>(null)

  const revoke = useMutation({
    mutationFn: (s: ShareOut) => shares.revoke(s.id),
    onSuccess: () => {
      toast.success('The link was taken down.')
      setRevoking(null)
      void qc.invalidateQueries({ queryKey: [...SHARES_KEY] })
    },
    onError: (e) => toast.error(shareErrorText(e, 'Could not revoke the link.')),
  })

  const copy = async (s: ShareOut) => {
    if (await copyText(shareUrl(s.path))) toast.success('Link copied.')
    else toast.error('Could not copy. Select the link and copy it by hand.')
  }

  const rows = q.data ?? []
  const live = rows.filter((s) => shareStatus(s) === 'active')
  const rest = rows.filter((s) => shareStatus(s) !== 'active')

  return (
    <div className="flex flex-col gap-4">
      <Section
        title="Share links"
        description="Links you created to show photos to people without an account. Private items are never part of a link."
      >
        {q.isPending ? (
          <Spinner />
        ) : q.isError ? (
          <ErrorState error={q.error} onRetry={() => void q.refetch()} />
        ) : rows.length === 0 ? (
          <p className="text-sm text-ink-soft">
            No links yet. Select photos in the library and choose Share, or share an album from its page.
          </p>
        ) : (
          <ul className="divide-y divide-line">
            {[...live, ...rest].map((s) => {
              const status = shareStatus(s)
              return (
                <li key={s.id} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-3">
                  <div className="min-w-0 flex-1">
                    <p className="flex flex-wrap items-center gap-2 font-medium">
                      <Link2 className="size-4 shrink-0 text-ink-faint" aria-hidden />
                      <span className="truncate">{s.title || 'Untitled link'}</span>
                      {status === 'active' ? (
                        <Badge tone="ok">Active</Badge>
                      ) : status === 'expired' ? (
                        <Badge tone="warn">Expired</Badge>
                      ) : (
                        <Badge>Revoked</Badge>
                      )}
                      {s.has_password && (
                        <Badge tone="brand">
                          <Lock className="size-3" aria-hidden />
                          Password
                        </Badge>
                      )}
                    </p>
                    <p className="text-sm text-ink-soft">
                      {describe(s)}. Created {formatDate(s.created_at)}. {s.view_count} {s.view_count === 1 ? 'view' : 'views'}
                      {s.last_viewed_at ? `, last ${formatRelative(s.last_viewed_at)}` : ''}.{' '}
                      {status === 'revoked' ? `Taken down ${formatDate(s.revoked_at)}.` : expiryText(s.expires_at) + '.'}
                      {!s.allow_download && ' Downloads off.'}
                    </p>
                  </div>
                  {status !== 'revoked' && (
                    <div className="flex flex-wrap gap-2">
                      <Button size="sm" variant="secondary" onClick={() => void copy(s)}>
                        <Copy className="size-4" aria-hidden />
                        Copy
                      </Button>
                      <Button size="sm" variant="secondary" onClick={() => setEditing(s)}>
                        Edit
                      </Button>
                      <Button size="sm" variant="danger" onClick={() => setRevoking(s)}>
                        Revoke
                      </Button>
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </Section>
      <EditShareModal share={editing} onClose={() => setEditing(null)} />
      <ConfirmModal
        open={!!revoking}
        title="Take this link down?"
        body="Anyone who has it will see that it no longer exists. The photos themselves are not touched."
        confirmLabel="Revoke"
        danger
        loading={revoke.isPending}
        onConfirm={() => revoking && revoke.mutate(revoking)}
        onClose={() => !revoke.isPending && setRevoking(null)}
      />
    </div>
  )
}

function EditShareModal({ share, onClose }: { share: ShareOut | null; onClose: () => void }) {
  const qc = useQueryClient()
  const toast = useToast()
  const ids = useId()
  const [title, setTitle] = useState('')
  const [passwordMode, setPasswordMode] = useState<'keep' | 'set' | 'clear'>('keep')
  const [password, setPassword] = useState('')
  const [expiryMode, setExpiryMode] = useState<'keep' | string>('keep')
  const [allowDownload, setAllowDownload] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!share) return
    setTitle(share.title)
    setPasswordMode('keep')
    setPassword('')
    setExpiryMode('keep')
    setAllowDownload(share.allow_download)
    setError('')
  }, [share])

  const save = useMutation({
    mutationFn: (body: ShareUpdate) => shares.update(share!.id, body),
    onSuccess: () => {
      toast.success('Link updated.')
      void qc.invalidateQueries({ queryKey: [...SHARES_KEY] })
      onClose()
    },
    onError: (e) => setError(shareErrorText(e, 'Could not save the changes.')),
  })

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!share || save.isPending) return
    const body: ShareUpdate = { title: title.trim(), allow_download: allowDownload }
    if (passwordMode === 'set') {
      if (!password.trim()) {
        setError('Enter the new password, or choose to keep or clear the current one.')
        return
      }
      body.password = password.trim()
    } else if (passwordMode === 'clear') body.password = ''
    if (expiryMode !== 'keep') body.expires_days = expiryMode ? Number(expiryMode) : null
    save.mutate(body)
  }

  const formId = `${ids}-edit`
  return (
    <Modal
      open={!!share}
      onClose={() => !save.isPending && onClose()}
      title="Edit link"
      footer={
        <>
          <Button variant="secondary" onClick={onClose} disabled={save.isPending}>
            Cancel
          </Button>
          <Button type="submit" form={formId} loading={save.isPending}>
            Save
          </Button>
        </>
      }
    >
      {share && (
        <form id={formId} onSubmit={submit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <label htmlFor={`${ids}-title`} className="text-sm font-medium">
              Title
            </label>
            <input id={`${ids}-title`} className={inputClass} value={title} maxLength={120} onChange={(e) => setTitle(e.target.value)} />
          </div>
          <fieldset className="flex flex-col gap-2">
            <legend className="text-sm font-medium">Password</legend>
            <label className="flex items-center gap-2 text-sm">
              <input type="radio" name={`${ids}-pw`} checked={passwordMode === 'keep'} onChange={() => setPasswordMode('keep')} />
              {share.has_password ? 'Keep the current password' : 'No password'}
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="radio" name={`${ids}-pw`} checked={passwordMode === 'set'} onChange={() => setPasswordMode('set')} />
              {share.has_password ? 'Change the password' : 'Set a password'}
            </label>
            {passwordMode === 'set' && (
              <input
                className={inputClass}
                type="password"
                autoComplete="new-password"
                aria-label="New password"
                value={password}
                maxLength={200}
                onChange={(e) => setPassword(e.target.value)}
              />
            )}
            {share.has_password && (
              <label className="flex items-center gap-2 text-sm">
                <input type="radio" name={`${ids}-pw`} checked={passwordMode === 'clear'} onChange={() => setPasswordMode('clear')} />
                Remove the password
              </label>
            )}
          </fieldset>
          <div className="flex flex-col gap-1">
            <label htmlFor={`${ids}-expiry`} className="text-sm font-medium">
              Expires
            </label>
            <select id={`${ids}-expiry`} className={inputClass} value={expiryMode} onChange={(e) => setExpiryMode(e.target.value)}>
              <option value="keep">Keep ({expiryText(share.expires_at).toLowerCase()})</option>
              {EXPIRY_CHOICES.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.value ? `${c.label} from now` : c.label}
                </option>
              ))}
            </select>
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={allowDownload} onChange={(e) => setAllowDownload(e.target.checked)} />
            Allow downloading the original files
          </label>
          {error && (
            <p role="alert" className="text-sm text-danger">
              {error}
            </p>
          )}
        </form>
      )}
    </Modal>
  )
}
