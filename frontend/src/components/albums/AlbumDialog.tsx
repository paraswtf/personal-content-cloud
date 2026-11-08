// New album and Rename dialogs (docs/ARCHITECTURE.md section 24): a name and a description.
import { useEffect, useState, type FormEvent } from 'react'
import { Button, Modal } from '../ui'
import { inputClass } from '../admin/fields'

export default function AlbumDialog({
  open,
  onClose,
  onSubmit,
  busy = false,
  title,
  submitLabel,
  initial,
  error,
}: {
  open: boolean
  onClose: () => void
  onSubmit: (values: { name: string; description: string }) => void
  busy?: boolean
  title: string
  submitLabel: string
  initial?: { name: string; description: string }
  error?: string | null
}) {
  const [name, setName] = useState(initial?.name ?? '')
  const [description, setDescription] = useState(initial?.description ?? '')
  useEffect(() => {
    if (open) {
      setName(initial?.name ?? '')
      setDescription(initial?.description ?? '')
    }
  }, [open, initial?.name, initial?.description])
  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!name.trim() || busy) return
    onSubmit({ name: name.trim(), description: description.trim() })
  }
  return (
    <Modal
      open={open}
      onClose={() => !busy && onClose()}
      title={title}
      footer={
        <>
          <Button type="button" variant="secondary" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button type="submit" form="album-dialog-form" loading={busy} disabled={!name.trim()}>
            {submitLabel}
          </Button>
        </>
      }
    >
      <form id="album-dialog-form" onSubmit={submit} className="flex flex-col gap-4">
        <label className="flex flex-col gap-1.5 text-sm font-semibold">
          Name
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            className={inputClass}
            maxLength={120}
            required
            autoFocus
            placeholder="Goa trip"
          />
        </label>
        <label className="flex flex-col gap-1.5 text-sm font-semibold">
          Description
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={3}
            maxLength={2000}
            placeholder="Optional"
            className="w-full rounded-2xl border border-line bg-paper px-4 py-2.5 text-sm font-normal text-ink placeholder:text-ink-faint focus-visible:border-brand focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/20"
          />
        </label>
        {error && (
          <p role="alert" className="text-sm text-danger">
            {error}
          </p>
        )}
      </form>
    </Modal>
  )
}
