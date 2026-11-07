// Modal that asks "which person?" with PersonPicker (an existing person or a new name).
import { Spinner } from '../ui'
import { AdminModal } from '../admin/AdminModal'
import PersonPicker, { type PersonChoice } from './PersonPicker'

export type { PersonChoice }

export default function PersonChoiceModal({
  open,
  title,
  description,
  busy = false,
  allowCreate = true,
  excludeIds,
  onPick,
  onClose,
}: {
  open: boolean
  title: string
  description?: string
  busy?: boolean
  allowCreate?: boolean
  excludeIds?: number[]
  onPick: (choice: PersonChoice) => void
  onClose: () => void
}) {
  return (
    <AdminModal open={open} onClose={busy ? () => undefined : onClose} title={title}>
      <div className="min-h-64 space-y-3">
        {description && <p className="text-sm text-ink-soft">{description}</p>}
        {busy ? (
          <div className="flex justify-center py-10">
            <Spinner label="Working" />
          </div>
        ) : (
          <PersonPicker autoFocus allowCreate={allowCreate} excludeIds={excludeIds} placeholder="Search people by name" onPick={onPick} />
        )}
      </div>
    </AdminModal>
  )
}
