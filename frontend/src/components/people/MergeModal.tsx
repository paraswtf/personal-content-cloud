// Choose which person to keep when merging two or more people.
import { useState } from 'react'
import clsx from 'clsx'
import { Avatar, Button } from '../ui'
import { AdminModal } from '../admin/AdminModal'
import type { PersonOut } from '../../lib/types'
import { formatCount, personName } from '../../lib/format'

function defaultTarget(people: PersonOut[]): number {
  const ranked = [...people].sort((a, b) => Number(!!b.name) - Number(!!a.name) || b.item_count - a.item_count || a.id - b.id)
  return ranked[0].id
}

function MergeForm({
  people,
  busy,
  onConfirm,
  onClose,
}: {
  people: PersonOut[]
  busy: boolean
  onConfirm: (targetId: number, sourceIds: number[]) => void
  onClose: () => void
}) {
  const [target, setTarget] = useState(() => defaultTarget(people))
  const keep = people.find((p) => p.id === target) ?? people[0]
  const others = people.filter((p) => p.id !== keep.id)
  return (
    <AdminModal
      open
      onClose={busy ? () => undefined : onClose}
      title={`Merge ${people.length} people`}
      footer={
        <>
          <Button variant="ghost" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button loading={busy} onClick={() => onConfirm(keep.id, others.map((p) => p.id))}>
            Merge into {personName(keep)}
          </Button>
        </>
      }
    >
      <fieldset className="space-y-2">
        <legend className="mb-2 text-sm text-ink-soft">
          Choose the person to keep. All faces, bodies and tags move to them. If they have no name, they take the name of one of the others.
        </legend>
        {people.map((p) => (
          <label
            key={p.id}
            className={clsx(
              'flex cursor-pointer items-center gap-3 rounded-2xl border p-3 transition-colors',
              target === p.id ? 'border-accent bg-accent-soft' : 'border-line hover:border-ink-faint',
            )}
          >
            <input type="radio" name="merge-target" value={p.id} checked={target === p.id} onChange={() => setTarget(p.id)} />
            <Avatar src={p.cover_url} name={p.name} size={44} />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-semibold">{personName(p)}</span>
              <span className="block text-xs text-ink-soft">
                {formatCount(p.item_count, 'item')}, {formatCount(p.face_count, 'face')}
              </span>
            </span>
            {target === p.id && <span className="text-xs font-semibold text-accent">Keep</span>}
          </label>
        ))}
      </fieldset>
    </AdminModal>
  )
}

export default function MergeModal(props: {
  open: boolean
  people: PersonOut[]
  busy?: boolean
  onConfirm: (targetId: number, sourceIds: number[]) => void
  onClose: () => void
}) {
  if (!props.open || props.people.length < 2) return null
  return (
    <MergeForm
      key={props.people.map((p) => p.id).join(',')}
      people={props.people}
      busy={props.busy ?? false}
      onConfirm={props.onConfirm}
      onClose={props.onClose}
    />
  )
}
