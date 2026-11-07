// Popover for one animal box: name the pet (pick one of that species, or a new name), take the
// name off, or remove the box when it is not an animal. Same shape as the face editor.
import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Ban, PawPrint, UserX, X } from 'lucide-react'
import { api } from '../../lib/api'
import { personName, speciesLabel } from '../../lib/format'
import PersonPicker, { choiceBody, type PersonChoice } from '../people/PersonPicker'
import { Button } from '../ui'
import type { AnimalBox } from './boxes'
import { placement } from './FaceEditor'
import { useInvalidateItem } from './shared'

interface Props {
  uid: string
  box: AnimalBox
  anchor: HTMLElement | null
  onClose: () => void
  onDone: (message: string) => void
}

export default function AnimalEditor({ uid, box, anchor, onClose, onDone }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [style, setStyle] = useState<CSSProperties>(() => placement(anchor))
  const invalidate = useInvalidateItem(uid)
  const sheet = style.bottom === 0 && style.left === 0
  const species = speciesLabel(box.species)

  useLayoutEffect(() => setStyle(placement(anchor)), [anchor, box.id])

  useEffect(() => {
    const onResize = () => setStyle(placement(anchor))
    window.addEventListener('resize', onResize)
    const onDown = (e: MouseEvent | TouchEvent) => {
      const t = e.target as Node
      if (ref.current?.contains(t) || anchor?.contains(t)) return
      onClose()
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('touchstart', onDown)
    return () => {
      window.removeEventListener('resize', onResize)
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('touchstart', onDown)
    }
  }, [anchor, onClose])

  const assign = useMutation({
    mutationFn: (choice: PersonChoice | null) => api.animals.assign({ animal_ids: [box.id], ...choiceBody(choice) }),
    onSuccess: (res, choice) => {
      invalidate()
      if (!choice) onDone(box.person ? `Removed from ${personName(box.person)}` : 'Unassigned')
      else onDone(res.person ? `Tagged as ${personName(res.person)}` : 'Saved')
    },
  })
  const remove = useMutation({
    mutationFn: () => api.animals.remove(box.id),
    onSuccess: () => {
      invalidate()
      onDone('Removed. It will not be shown as a pet again.')
    },
  })
  const busy = assign.isPending || remove.isPending
  const error = assign.error ?? remove.error

  return (
    <div
      ref={ref}
      role="dialog"
      aria-label="Edit pet"
      style={style}
      className={
        'fixed z-50 flex max-h-[85dvh] flex-col overflow-y-auto bg-surface p-3 text-ink shadow-2xl ' +
        (sheet ? 'rounded-t-2xl pb-[max(0.75rem,env(safe-area-inset-bottom))]' : 'rounded-2xl')
      }
    >
      <div className="flex items-center gap-3">
        <img src={box.url} alt="" className="size-12 shrink-0 rounded-full bg-surface-2 object-cover ring-1 ring-line" />
        <div className="min-w-0 flex-1">
          <p className="inline-flex items-center gap-1 text-xs text-ink-faint">
            <PawPrint className="size-3" aria-hidden />
            {box.person ? `${species}, currently tagged as` : `${species}, not named yet`}
          </p>
          <p className="truncate text-sm font-semibold">{box.person ? personName({ ...box.person, species: box.species }) : 'Which pet is this?'}</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft hover:bg-surface-2"
          aria-label="Close"
        >
          <X className="size-5" aria-hidden />
        </button>
      </div>

      <div className="mt-3">
        <PersonPicker
          kind="pet"
          species={box.species}
          autoFocus={!sheet}
          placeholder={box.person ? 'Move to another pet' : 'Search pets or type a new name'}
          excludeIds={box.person ? [box.person.id] : undefined}
          onPick={(choice) => {
            if (!busy) assign.mutate(choice)
          }}
        />
        <p className="mt-2 text-xs text-ink-soft">Pets of the same species come first. A box can only go to a pet of its own species.</p>
      </div>

      {error && (
        <p className="mt-2 text-sm text-danger" role="alert">
          {error instanceof Error ? error.message : 'That did not work. Try again.'}
        </p>
      )}

      <div className="mt-3 flex flex-wrap gap-2">
        {box.person && (
          <Button variant="secondary" disabled={busy} loading={assign.isPending && assign.variables === null} onClick={() => assign.mutate(null)}>
            <UserX className="size-4" aria-hidden />
            Not this pet
          </Button>
        )}
        <Button variant="danger" disabled={busy} loading={remove.isPending} onClick={() => remove.mutate()}>
          <Ban className="size-4" aria-hidden />
          Remove box
        </Button>
      </div>
    </div>
  )
}
