// Settings > Passkeys: list, add (named), rename and remove. A passkey signs in with the device's
// fingerprint, face or screen lock; the password keeps working as a fallback.
import { useId, useState, type FormEvent } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Fingerprint, KeyRound, Pencil, Plus, Trash2 } from 'lucide-react'
import { api } from '../../../lib/api'
import { AUTH_KEY, PASSKEYS_KEY, authErrorMessage, passkeysSupported, registerPasskey } from '../../../lib/auth'
import { formatDate } from '../../../lib/format'
import type { PasskeyOut } from '../../../lib/types'
import { Badge, Button, ConfirmModal, ErrorState, Modal, Spinner } from '../../ui'
import { inputClass } from '../../admin/fields'
import { useToast } from '../../Toast'
import { FormError } from '../AuthCard'
import { Section, describeAgent, errorText } from './common'

const NAME_MAX = 60

function defaultName(): string {
  return typeof navigator === 'undefined' ? 'This device' : describeAgent(navigator.userAgent)
}

export default function PasskeysTab() {
  const qc = useQueryClient()
  const toast = useToast()
  const supported = passkeysSupported()
  const q = useQuery({ queryKey: PASSKEYS_KEY, queryFn: api.auth.passkeys })
  const nameId = useId()
  const [name, setName] = useState(defaultName)
  const [adding, setAdding] = useState(false)
  const [addError, setAddError] = useState<string | null>(null)
  const [renaming, setRenaming] = useState<PasskeyOut | null>(null)
  const [newName, setNewName] = useState('')
  const [renameBusy, setRenameBusy] = useState(false)
  const [removing, setRemoving] = useState<PasskeyOut | null>(null)
  const [removeBusy, setRemoveBusy] = useState(false)

  const refresh = () => {
    void qc.invalidateQueries({ queryKey: PASSKEYS_KEY })
    void qc.invalidateQueries({ queryKey: AUTH_KEY })
  }

  const add = async (e: FormEvent) => {
    e.preventDefault()
    const clean = name.replace(/\s+/g, ' ').trim()
    if (!clean || adding) return
    setAdding(true)
    setAddError(null)
    try {
      await registerPasskey(clean.slice(0, NAME_MAX))
      toast.success(`Passkey "${clean}" added. You can now sign in with it.`)
      refresh()
    } catch (err) {
      setAddError(authErrorMessage(err, 'Could not add the passkey. Try again.'))
    } finally {
      setAdding(false)
    }
  }

  const rename = async (e: FormEvent) => {
    e.preventDefault()
    const clean = newName.replace(/\s+/g, ' ').trim()
    if (!renaming || !clean) return
    setRenameBusy(true)
    try {
      await api.auth.renamePasskey(renaming.id, clean.slice(0, NAME_MAX))
      setRenaming(null)
      refresh()
    } catch (err) {
      toast.error(errorText(err, 'Could not rename the passkey.'))
    } finally {
      setRenameBusy(false)
    }
  }

  const remove = async () => {
    if (!removing) return
    setRemoveBusy(true)
    try {
      await api.auth.deletePasskey(removing.id)
      toast.success(`Passkey "${removing.name}" removed.`)
      setRemoving(null)
      refresh()
    } catch (err) {
      toast.error(errorText(err, 'Could not remove the passkey.'))
    } finally {
      setRemoveBusy(false)
    }
  }

  const keys = q.data ?? []

  return (
    <div className="flex flex-col gap-4">
      <Section
        title="Add a passkey"
        description="A passkey lets this device sign you in with its fingerprint, face or screen lock, without typing the password. It also unlocks the vault."
      >
        {supported ? (
          <form onSubmit={add} className="flex max-w-lg flex-col gap-3">
            <label htmlFor={nameId} className="text-sm font-medium">
              Name
            </label>
            <div className="flex flex-col gap-2 sm:flex-row">
              <input
                id={nameId}
                value={name}
                maxLength={NAME_MAX}
                onChange={(e) => setName(e.target.value)}
                placeholder="For example, My phone"
                className={inputClass}
              />
              <Button type="submit" loading={adding} disabled={!name.trim()} className="shrink-0">
                {!adding && <Plus className="size-4" aria-hidden />}
                Add passkey
              </Button>
            </div>
            <FormError>{addError}</FormError>
          </form>
        ) : (
          <p className="rounded-xl bg-warn/10 px-3 py-2 text-sm text-warn">
            This browser cannot create passkeys here. Passkeys need a current browser and the site opened through https (or
            localhost).
          </p>
        )}
      </Section>

      <Section title="Your passkeys" description="The password always works too, so removing every passkey never locks you out.">
        {q.isPending ? (
          <div className="flex justify-center py-6">
            <Spinner label="Loading passkeys" />
          </div>
        ) : q.isError ? (
          <ErrorState error={q.error} onRetry={() => void q.refetch()} />
        ) : keys.length === 0 ? (
          <p className="flex items-center gap-2 text-sm text-ink-soft">
            <KeyRound className="size-4 text-ink-faint" aria-hidden />
            No passkeys yet.
          </p>
        ) : (
          <ul className="divide-y divide-line">
            {keys.map((k) => (
              <li key={k.id} className="flex items-center gap-3 py-3">
                <span className="flex size-10 shrink-0 items-center justify-center rounded-full bg-surface-2 text-ink-soft">
                  <Fingerprint className="size-5" aria-hidden />
                </span>
                <div className="min-w-0 flex-1">
                  <p className="flex flex-wrap items-center gap-2 text-sm font-semibold">
                    <span className="truncate">{k.name}</span>
                    {k.backup_eligible && <Badge>Synced</Badge>}
                  </p>
                  <p className="text-xs text-ink-soft">
                    Added {formatDate(k.created_at)}
                    {k.last_used_at ? `, last used ${formatDate(k.last_used_at, true)}` : ', not used yet'}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setRenaming(k)
                    setNewName(k.name)
                  }}
                  aria-label={`Rename ${k.name}`}
                  title="Rename"
                  className="inline-flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft hover:bg-ink/5 hover:text-ink"
                >
                  <Pencil className="size-4" aria-hidden />
                </button>
                <button
                  type="button"
                  onClick={() => setRemoving(k)}
                  aria-label={`Remove ${k.name}`}
                  title="Remove"
                  className="inline-flex size-10 shrink-0 items-center justify-center rounded-full text-danger hover:bg-danger/10"
                >
                  <Trash2 className="size-4" aria-hidden />
                </button>
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Modal
        open={!!renaming}
        onClose={() => setRenaming(null)}
        title="Rename passkey"
        footer={
          <>
            <Button variant="ghost" onClick={() => setRenaming(null)}>
              Cancel
            </Button>
            <Button type="submit" form="rename-passkey" loading={renameBusy} disabled={!newName.trim()}>
              Save
            </Button>
          </>
        }
      >
        <form id="rename-passkey" onSubmit={rename} className="flex flex-col gap-1.5">
          <label htmlFor="rename-passkey-name" className="text-sm font-medium">
            Name
          </label>
          <input
            id="rename-passkey-name"
            value={newName}
            maxLength={NAME_MAX}
            autoFocus
            onChange={(e) => setNewName(e.target.value)}
            className={inputClass}
          />
        </form>
      </Modal>

      <ConfirmModal
        open={!!removing}
        title="Remove passkey"
        body={
          <>
            <span className="font-semibold text-ink">{removing?.name}</span> will no longer sign in here. Also remove it from the
            device or password manager that holds it, so it is not offered again.
          </>
        }
        confirmLabel="Remove"
        danger
        loading={removeBusy}
        onConfirm={() => void remove()}
        onClose={() => setRemoving(null)}
      />
    </div>
  )
}
