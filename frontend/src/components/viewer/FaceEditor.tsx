// Popover for one face or body box: confirm or reject a guess, assign to a person (or a new one),
// unassign, or delete it as "not a face" / "not a person". A face in an unnamed group names the
// whole group (or merges it into the picked person) unless "Only change this face" is ticked.
import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Ban, Check, UserX, X } from 'lucide-react'
import { api } from '../../lib/api'
import { formatCount, formatPercent, personName } from '../../lib/format'
import PersonPicker, { choiceBody, type PersonChoice } from '../people/PersonPicker'
import { evidenceLine } from '../people/evidence'
import { Button } from '../ui'
import type { BoxTarget } from './boxes'
import { useInvalidateItem } from './shared'

interface Props {
  uid: string
  target: BoxTarget
  anchor: HTMLElement | null
  onClose: () => void
  onDone: (message: string) => void
}

const WIDTH = 320

function placement(anchor: HTMLElement | null): CSSProperties {
  const vw = window.innerWidth
  const vh = window.innerHeight
  if (!anchor || vw < 640) return { left: 0, right: 0, bottom: 0 }
  const r = anchor.getBoundingClientRect()
  const left = Math.min(Math.max(8, r.left + r.width / 2 - WIDTH / 2), vw - WIDTH - 8)
  const need = 460
  if (vh - r.bottom >= need) return { left, top: r.bottom + 8, width: WIDTH }
  if (r.top >= need) return { left, bottom: vh - r.top + 8, width: WIDTH }
  const side = r.right + 8 + WIDTH <= vw - 8 ? r.right + 8 : Math.max(8, r.left - WIDTH - 8)
  return { left: side, top: Math.max(8, (vh - need) / 2), width: WIDTH }
}

export default function FaceEditor({ uid, target, anchor, onClose, onDone }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [style, setStyle] = useState<CSSProperties>(() => placement(anchor))
  const invalidate = useInvalidateItem(uid)
  const sheet = style.bottom === 0 && style.left === 0
  const { kind, box } = target
  const isFace = kind === 'face'
  const noun = isFace ? 'face' : 'person'

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

  const group = isFace && box.person && !box.person.name ? box.person : null
  const groupInfo = useQuery({ queryKey: ['person', group?.id], queryFn: () => api.people.get(group!.id), enabled: !!group })
  const groupFaces = groupInfo.data?.face_count ?? 0
  const [onlyThis, setOnlyThis] = useState(false)
  const groupScope = !!group && !onlyThis

  const assign = useMutation({
    mutationFn: (choice: PersonChoice | null) =>
      isFace ? api.people.assignFaces({ face_ids: [box.id], ...choiceBody(choice) }) : api.people.assignBodies({ body_ids: [box.id], ...choiceBody(choice) }),
    onSuccess: (res, choice) => {
      invalidate()
      if (!choice) onDone(box.person ? `Removed from ${personName(box.person)}` : 'Unassigned')
      else onDone(res.person ? `Tagged as ${personName(res.person)}` : 'Saved')
    },
  })
  const nameGroup = useMutation({
    mutationFn: async (choice: PersonChoice) =>
      'newName' in choice
        ? { merged: false, person: await api.people.update(group!.id, { name: choice.newName }) }
        : { merged: true, person: await api.people.merge({ target_id: choice.personId, source_ids: [group!.id] }) },
    onSuccess: ({ merged, person }) => {
      invalidate()
      onDone(merged ? `Merged into ${personName(person)}` : `Named ${personName(person)}${groupFaces > 1 ? ` (${formatCount(groupFaces, 'face')})` : ''}`)
    },
  })
  const guessAct = useMutation({
    mutationFn: (action: 'confirm' | 'reject') => {
      const body = isFace ? { faces: [box.id] } : { bodies: [box.id] }
      return action === 'confirm' ? api.people.confirmGuesses(body) : api.people.rejectGuesses(body)
    },
    onSuccess: (_r, action) => {
      invalidate()
      const n = box.guess ? personName(box.guess.person) : ''
      onDone(action === 'confirm' ? `Tagged as ${n}` : `Not ${n}`)
    },
  })
  const remove = useMutation({
    mutationFn: () => (isFace ? api.people.deleteFace(box.id) : api.people.deleteBody(box.id)),
    onSuccess: () => {
      invalidate()
      onDone(`Removed. It will not be shown as a ${noun} again.`)
    },
  })
  const busy = assign.isPending || remove.isPending || nameGroup.isPending || guessAct.isPending
  const error = assign.error ?? remove.error ?? nameGroup.error ?? guessAct.error
  const guess = !box.person ? box.guess : null

  return (
    <div
      ref={ref}
      role="dialog"
      aria-label={isFace ? 'Edit face' : 'Edit person'}
      style={style}
      className={
        'fixed z-50 flex max-h-[85dvh] flex-col overflow-y-auto bg-surface p-3 text-ink shadow-2xl ' +
        (sheet ? 'rounded-t-2xl pb-[max(0.75rem,env(safe-area-inset-bottom))]' : 'rounded-2xl')
      }
    >
      <div className="flex items-center gap-3">
        <img src={box.url} alt="" className="size-12 shrink-0 rounded-full bg-surface-2 object-cover ring-1 ring-line" />
        <div className="min-w-0 flex-1">
          <p className="text-xs text-ink-faint">
            {group ? 'Not named yet' : box.person ? 'Currently tagged as' : guess ? 'Our guess' : 'Not tagged yet'}
          </p>
          <p className="truncate text-sm font-semibold">
            {box.person ? personName(box.person) : guess ? `${personName(guess.person)}?` : 'Who is this?'}
            {group && groupFaces > 0 && <span className="font-normal text-ink-soft">, {formatCount(groupFaces, 'face')}</span>}
          </p>
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

      {guess && (
        <div className="mt-3 rounded-xl bg-warn/10 p-3">
          <p className="text-sm">
            Looks like <span className="font-semibold">{personName(guess.person)}</span>, {formatPercent(guess.score)} sure.
          </p>
          {evidenceLine(guess) && <p className="mt-0.5 text-xs text-ink-soft">{evidenceLine(guess)}</p>}
          <div className="mt-2 flex flex-wrap gap-2">
            <Button size="md" disabled={busy} loading={guessAct.isPending && guessAct.variables === 'confirm'} onClick={() => guessAct.mutate('confirm')}>
              <Check className="size-4" aria-hidden />
              Yes, it is
            </Button>
            <Button
              size="md"
              variant="secondary"
              disabled={busy}
              loading={guessAct.isPending && guessAct.variables === 'reject'}
              onClick={() => guessAct.mutate('reject')}
            >
              <X className="size-4" aria-hidden />
              No
            </Button>
          </div>
        </div>
      )}

      <div className="mt-3">
        <PersonPicker
          autoFocus={!sheet && !guess}
          placeholder={groupScope ? 'Type their name, or pick someone' : box.person ? 'Move to someone else' : 'Search or type a new name'}
          excludeIds={box.person ? [box.person.id] : undefined}
          onPick={(choice) => {
            if (busy) return
            if (groupScope) nameGroup.mutate(choice)
            else assign.mutate(choice)
          }}
        />
        {groupScope && (
          <p className="mt-2 text-xs text-ink-soft">
            {groupFaces > 1
              ? `The name goes on all ${groupFaces} faces in this group. Picking someone already named merges the group into them.`
              : 'Picking someone already named merges this face into them.'}
          </p>
        )}
        {group && groupFaces > 1 && (
          <label className="mt-2 flex min-h-10 cursor-pointer items-center gap-2 text-sm text-ink-soft">
            <input type="checkbox" checked={onlyThis} onChange={(e) => setOnlyThis(e.target.checked)} />
            Only change this face
          </label>
        )}
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
            Not this person
          </Button>
        )}
        <Button variant="danger" disabled={busy} loading={remove.isPending} onClick={() => remove.mutate()}>
          <Ban className="size-4" aria-hidden />
          {isFace ? 'Not a face' : 'Not a person'}
        </Button>
      </div>
    </div>
  )
}
