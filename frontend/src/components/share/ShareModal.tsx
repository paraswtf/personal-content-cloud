// Create a share link (docs/ARCHITECTURE.md section 26) for a hand-picked set or an album: title,
// password, expiry and the download switch, then the link with a Copy button. Opened by the bulk bar's
// Share, the album page's Share album and the viewer's Share.
import { useEffect, useId, useState, type FormEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Check, Copy } from 'lucide-react'
import { ApiError } from '../../lib/api'
import { EXPIRY_CHOICES, SHARES_KEY, copyText, shareErrorText, shareUrl, shares, type ShareOut } from '../../lib/shares'
import { inputClass } from '../admin/fields'
import { useToast } from '../Toast'
import { Button, Modal } from '../ui'

export type ShareTarget = { kind: 'items'; uids: string[] } | { kind: 'album'; albumId: number }

interface Props {
  open: boolean
  onClose: () => void
  target: ShareTarget
  /** The title field's starting value (an album's name, for example). */
  defaultTitle?: string
}

export default function ShareModal({ open, onClose, target, defaultTitle = '' }: Props) {
  const qc = useQueryClient()
  const toast = useToast()
  const ids = useId()
  const [title, setTitle] = useState(defaultTitle)
  const [password, setPassword] = useState('')
  const [expiry, setExpiry] = useState('')
  const [allowDownload, setAllowDownload] = useState(true)
  const [created, setCreated] = useState<ShareOut | null>(null)
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState('')

  // a fresh form every time the dialog opens
  useEffect(() => {
    if (!open) return
    setTitle(defaultTitle)
    setPassword('')
    setExpiry('')
    setAllowDownload(true)
    setCreated(null)
    setCopied(false)
    setError('')
  }, [open, defaultTitle])

  const create = useMutation({
    mutationFn: () =>
      shares.create({
        kind: target.kind,
        album_id: target.kind === 'album' ? target.albumId : undefined,
        item_uids: target.kind === 'items' ? target.uids : undefined,
        title: title.trim(),
        password: password.trim() || null,
        expires_days: expiry ? Number(expiry) : null,
        allow_download: allowDownload,
      }),
    onSuccess: (share) => {
      setCreated(share)
      setError('')
      void qc.invalidateQueries({ queryKey: [...SHARES_KEY] })
      toast.success('Link created.')
    },
    onError: (e) => setError(shareErrorText(e, 'The link could not be created.')),
  })

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!create.isPending) create.mutate()
  }

  const copy = async () => {
    if (!created) return
    const ok = await copyText(shareUrl(created.path))
    if (ok) {
      setCopied(true)
      toast.success('Link copied.')
      window.setTimeout(() => setCopied(false), 2000)
    } else toast.error('Could not copy. Select the link and copy it by hand.')
  }

  const count = target.kind === 'items' ? target.uids.length : null
  const what = target.kind === 'album' ? 'the album' : count === 1 ? 'this item' : `these ${count} items`
  const busy = create.isPending
  const formId = `${ids}-form`

  return (
    <Modal
      open={open}
      onClose={() => !busy && onClose()}
      title={created ? 'Share link' : 'Share'}
      footer={
        created ? (
          <Button onClick={onClose}>Done</Button>
        ) : (
          <>
            <Button variant="secondary" onClick={onClose} disabled={busy}>
              Cancel
            </Button>
            <Button type="submit" form={formId} loading={busy}>
              Create link
            </Button>
          </>
        )
      }
    >
      {created ? (
        <div className="flex flex-col gap-3">
          <p className="text-sm text-ink-soft">
            Anyone with this link can see {what}
            {created.has_password ? ' after entering the password' : ''}. Your private items are never part of a link.
          </p>
          <div className="flex gap-2">
            <input
              className={inputClass}
              readOnly
              value={shareUrl(created.path)}
              aria-label="Share link"
              onFocus={(e) => e.currentTarget.select()}
            />
            <Button variant="secondary" onClick={() => void copy()} aria-live="polite">
              {copied ? <Check className="size-4" aria-hidden /> : <Copy className="size-4" aria-hidden />}
              {copied ? 'Copied' : 'Copy'}
            </Button>
          </div>
          {target.kind === 'album' && created.count === 0 && (
            <p className="text-sm text-warn">This album has no photos the link can show yet.</p>
          )}
          <p className="text-xs text-ink-faint">
            {created.count} {created.count === 1 ? 'item' : 'items'}. You can change the password, the expiry or take the link
            down under Settings, Sharing.
          </p>
        </div>
      ) : (
        <form id={formId} onSubmit={submit} className="flex flex-col gap-4">
          <p className="text-sm text-ink-soft">
            Make a link that shows {what} to someone without an account. Private items are left out.
          </p>
          <div className="flex flex-col gap-1">
            <label htmlFor={`${ids}-title`} className="text-sm font-medium">
              Title
            </label>
            <input
              id={`${ids}-title`}
              className={inputClass}
              value={title}
              maxLength={120}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Shared photos"
              autoFocus
            />
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor={`${ids}-password`} className="text-sm font-medium">
              Password <span className="font-normal text-ink-faint">(optional)</span>
            </label>
            <input
              id={`${ids}-password`}
              className={inputClass}
              type="password"
              autoComplete="new-password"
              value={password}
              maxLength={200}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor={`${ids}-expiry`} className="text-sm font-medium">
              Expires
            </label>
            <select id={`${ids}-expiry`} className={inputClass} value={expiry} onChange={(e) => setExpiry(e.target.value)}>
              {EXPIRY_CHOICES.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.label}
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

/** Whether an error from the share API says the selection had a private item. */
export function isVaultRefusal(e: unknown): boolean {
  return e instanceof ApiError && e.code === 'vault_item'
}
