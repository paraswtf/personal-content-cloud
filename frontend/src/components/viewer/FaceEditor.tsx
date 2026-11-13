// Popover for one face or body box: confirm or reject a guess, assign to a person (or a new one),
// unassign, or delete it as "not a face" / "not a person". A face in an unnamed group names the
// whole group (or merges it into the picked person) unless "Only change this face" is ticked.
// Body traits (section 34): the guess panel shows why, with this photo's tattoo, nails or mehndi beside the
// guessed person's; "Seen on this person" lists the trait regions of the box (or of the person box holding the
// face), each with "Not a tattoo" (nails, mehndi), "Not this person's", and "Show again" once hidden (this photo
// only). Those keep the editor open and offer Undo in the app's toast.
import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { Ban, Check, Eye, EyeOff, Move, UserX, X } from 'lucide-react'
import { api } from '../../lib/api'
import type { AssignOut, BodyOut, ItemDetail, TraitOut } from '../../lib/types'
import { formatCount, personName } from '../../lib/format'
import { boxOrigin, mergeOrigins } from '../../lib/origin'
import OriginIcon from '../people/OriginIcon'
import PersonPicker, { choiceBody, type PersonChoice } from '../people/PersonPicker'
import TraitCompare from '../people/TraitCompare'
import { clothesLine, guessSure, placeReason, safeHex, traitReasons } from '../people/evidence'
import { notTraitLabel, traitActionKey, useTraitActions } from '../people/useTraitActions'
import { useToast } from '../Toast'
import { Button } from '../ui'
import type { PersonBoxTarget } from './boxes'
import { useInvalidateItem } from './shared'

interface Props {
  uid: string
  target: PersonBoxTarget
  /** Whole-video editing (section 10): every track of this person in the video, the target among them. */
  tracks?: PersonBoxTarget[]
  anchor: HTMLElement | null
  onClose: () => void
  onDone: (message: string) => void
  /** Move or resize the box on the picture (section 10). */
  onAdjust?: () => void
  /** Opened by "Someone else" on a guess: the name field takes the focus. */
  pickFirst?: boolean
}

export const WIDTH = 320

export function placement(anchor: HTMLElement | null): CSSProperties {
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

/** The person box a face belongs to, from the item the viewer holds (its traits are on the body). */
function bodyOfFace(qc: QueryClient, uid: string, faceId: number): BodyOut | null {
  for (const [, data] of qc.getQueriesData<ItemDetail>({ queryKey: ['item', uid] })) {
    if (!data || !Array.isArray(data.bodies)) continue
    const hit = data.bodies.find((b) => b.face_id === faceId)
    if (hit) return hit
  }
  return null
}

export default function FaceEditor({ uid, target, anchor, onClose, onDone, tracks, onAdjust, pickFirst = false }: Props) {
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
      // a control that opens this editor itself ("Someone else" on a guess): it re-anchors the editor, not close it
      if (t instanceof Element && t.closest('[data-editor-opener]')) return
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

  // body traits: read on the person box, so a face shows the ones of the body holding it
  const qc = useQueryClient()
  const traitBody = target.kind === 'body' ? target.box : bodyOfFace(qc, uid, target.box.id)
  const traits: TraitOut[] = traitBody?.traits ?? []
  const [traitOpen, setTraitOpen] = useState<number | null>(null)
  const appToast = useToast()
  const traitActs = useTraitActions({
    onDone: (msg) => {
      setTraitOpen(null)
      appToast.success(msg)
    },
    onError: (msg) => appToast.error(msg),
  })

  const group = isFace && box.person && !box.person.name ? box.person : null
  const groupInfo = useQuery({ queryKey: ['person', group?.id], queryFn: () => api.people.get(group!.id), enabled: !!group })
  const groupFaces = groupInfo.data?.face_count ?? 0
  const [onlyThis, setOnlyThis] = useState(false)
  const groupScope = !!group && !onlyThis

  // whole-video editing: the group's tracks unless the owner narrows it to this one
  const trackList = tracks && tracks.length > 1 ? tracks : null
  const [onlyTrack, setOnlyTrack] = useState(false)
  const scopeTargets: PersonBoxTarget[] = trackList && !onlyTrack ? trackList : [target]
  const faceIds = scopeTargets.filter((t) => t.kind === 'face').map((t) => t.box.id)
  const bodyIds = scopeTargets.filter((t) => t.kind === 'body').map((t) => t.box.id)
  const trackTail = scopeTargets.length > 1 ? ` (${scopeTargets.length} tracks)` : ''

  const assign = useMutation({
    mutationFn: async (choice: PersonChoice | null) => {
      const body = choiceBody(choice)
      let res: AssignOut | null = null
      if (faceIds.length) res = await api.people.assignFaces({ face_ids: faceIds, ...body })
      if (bodyIds.length) {
        const r2 = await api.people.assignBodies({ body_ids: bodyIds, ...body })
        res = res ? { ...res, updated: res.updated + r2.updated, followed: (res.followed ?? 0) + (r2.followed ?? 0), person: res.person ?? r2.person } : r2
      }
      return res as AssignOut
    },
    onSuccess: (res, choice) => {
      invalidate()
      const tail = res.followed ? `, with ${formatCount(res.followed, 'look-alike face')}` : ''
      if (!choice) onDone((box.person ? `Removed from ${personName(box.person)}${tail}` : 'Unassigned') + trackTail)
      else onDone((res.person ? `Tagged as ${personName(res.person)}${tail}` : 'Saved') + trackTail)
    },
  })
  const nameGroup = useMutation({
    mutationFn: async (choice: PersonChoice) => {
      if ('newName' in choice) {
        // A name another person already has merges the group into them (the server returns that person).
        const person = await api.people.update(group!.id, { name: choice.newName })
        return { merged: person.id !== group!.id, person }
      }
      return { merged: true, person: await api.people.merge({ target_id: choice.personId, source_ids: [group!.id] }) }
    },
    onSuccess: ({ merged, person }) => {
      invalidate()
      onDone(merged ? `Merged into ${personName(person)}` : `Named ${personName(person)}${groupFaces > 1 ? ` (${formatCount(groupFaces, 'face')})` : ''}`)
    },
  })
  // "Someone else" on a guess (section 10): learned as wrong and given to the person picked, in one step
  const correct = useMutation({
    mutationFn: (choice: PersonChoice) => api.people.correctGuesses({ faces: faceIds, bodies: bodyIds, ...choiceBody(choice) }),
    onSuccess: (res) => {
      invalidate()
      onDone((res.person ? `Tagged as ${personName(res.person)}` : 'Saved') + trackTail)
    },
  })
  const guessAct = useMutation({
    mutationFn: (action: 'confirm' | 'reject') => {
      const body = { faces: faceIds, bodies: bodyIds }
      return action === 'confirm' ? api.people.confirmGuesses(body) : api.people.rejectGuesses(body)
    },
    onSuccess: (_r, action) => {
      invalidate()
      const n = box.guess ? personName(box.guess.person) : ''
      onDone((action === 'confirm' ? `Tagged as ${n}` : `Not ${n}`) + trackTail)
    },
  })
  const remove = useMutation({
    mutationFn: async () => {
      let followed = 0
      for (const t of scopeTargets) {
        if (t.kind === 'face') followed += (await api.people.deleteFace(t.box.id)).followed ?? 0
        else await api.people.deleteBody(t.box.id)
      }
      return followed
    },
    onSuccess: (followed) => {
      invalidate()
      const more = followed > 0 ? ` ${formatCount(followed, 'look-alike')} went with it.` : ''
      onDone(`Removed. It will not be shown as a ${noun} again.${more}${trackTail}`)
    },
  })
  const busy = assign.isPending || remove.isPending || nameGroup.isPending || guessAct.isPending || correct.isPending
  const error = assign.error ?? remove.error ?? nameGroup.error ?? guessAct.error ?? correct.error
  const guess = !box.person ? box.guess : null
  const guessReasons = guess ? traitReasons(guess) : []
  const guessPlace = guess ? placeReason(guess) : null
  // any open guess among the boxes edited (a whole-video group may hold guessed and plain tracks): a pick corrects
  // those guesses (learned as wrong) instead of overwriting them
  const anyGuess = scopeTargets.some((t) => !t.box.person && !!t.box.guess)
  // who put them there (section 10): for the whole group of tracks, the owner when any track is theirs
  const origin = scopeTargets.length > 1 ? mergeOrigins(scopeTargets.map((t) => boxOrigin(t.box))) : boxOrigin(box)

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
          <p className="flex min-w-0 items-center gap-1 text-sm font-semibold">
            <span className="truncate">
              {box.person ? personName(box.person) : guess ? `${personName(guess.person)}?` : 'Who is this?'}
              {group && groupFaces > 0 && <span className="font-normal text-ink-soft">, {formatCount(groupFaces, 'face')}</span>}
            </span>
            {origin && <OriginIcon origin={origin} className="text-ink-soft" />}
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
            Looks like <span className="font-semibold">{personName(guess.person)}</span>, {guessSure(guess)}.
          </p>
          {clothesLine(guess) && <p className="mt-0.5 text-xs text-ink-soft">{clothesLine(guess)}</p>}
          {guessPlace && <p className="mt-0.5 text-xs text-ink-soft">{guessPlace.text}</p>}
          {guessReasons.map((r, i) => (
            <p key={i} className="mt-0.5 text-xs text-ink-soft">
              {r.text}
            </p>
          ))}
          {guessReasons[0] && <TraitCompare className="mt-2" reason={guessReasons[0]} name={personName(guess.person)} />}
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

      {traits.length > 0 && (
        <section aria-labelledby={`seen-${box.id}`} className="mt-3">
          <p id={`seen-${box.id}`} className="text-xs text-ink-faint">
            Seen on this person
          </p>
          <ul className="mt-1.5 flex flex-wrap gap-1.5">
            {traits.map((tr) => {
              const hex = tr.kind === 'nails' ? safeHex(tr.hex) : undefined
              return (
                <li key={tr.id}>
                  <button
                    type="button"
                    aria-expanded={traitOpen === tr.id}
                    aria-label={`${tr.label}${tr.hidden ? ', hidden' : ''}: options`}
                    onClick={() => setTraitOpen((cur) => (cur === tr.id ? null : tr.id))}
                    className={
                      'flex min-h-10 max-w-full items-center gap-1.5 rounded-full bg-surface-2 py-1 pl-1 pr-3 text-xs hover:bg-line focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ' +
                      (traitOpen === tr.id ? 'ring-2 ring-violet-400' : '')
                    }
                  >
                    <img src={tr.crop_url} alt="" className={'size-8 shrink-0 rounded-full object-cover ' + (tr.hidden ? 'opacity-50' : '')} />
                    {hex && <span aria-hidden className="size-3 shrink-0 rounded-full ring-1 ring-line" style={{ backgroundColor: hex }} />}
                    <span className={'min-w-0 truncate ' + (tr.hidden ? 'text-ink-faint line-through' : 'text-ink')}>{tr.label}</span>
                  </button>
                </li>
              )
            })}
          </ul>
          {traits
            .filter((tr) => tr.id === traitOpen)
            .map((tr) => (
              <div key={tr.id} role="group" aria-label={`What to do with ${tr.label}`} className="mt-2 flex flex-wrap gap-2 rounded-xl bg-surface-2 p-2">
                {tr.hidden ? (
                  <>
                    <Button
                      size="md"
                      variant="secondary"
                      disabled={traitActs.pending}
                      loading={traitActs.pendingKey === traitActionKey(tr.id, null)}
                      onClick={() => traitActs.show(tr)}
                    >
                      <Eye className="size-4" aria-hidden />
                      Show again
                    </Button>
                    <p className="w-full text-xs text-ink-soft">Only on this photo. Hidden look-alikes elsewhere stay hidden.</p>
                  </>
                ) : (
                  <>
                    <Button
                      size="md"
                      variant="secondary"
                      disabled={traitActs.pending}
                      loading={traitActs.pendingKey === traitActionKey(tr.id, 'not_trait')}
                      onClick={() => traitActs.hide(tr, 'not_trait')}
                    >
                      <EyeOff className="size-4" aria-hidden />
                      {notTraitLabel(tr.kind)}
                    </Button>
                    <Button
                      size="md"
                      variant="secondary"
                      disabled={traitActs.pending}
                      loading={traitActs.pendingKey === traitActionKey(tr.id, 'not_theirs')}
                      onClick={() => traitActs.hide(tr, 'not_theirs', box.person ? personName(box.person) : null)}
                    >
                      <UserX className="size-4" aria-hidden />
                      Not this person&apos;s
                    </Button>
                  </>
                )}
              </div>
            ))}
        </section>
      )}

      <div className="mt-3">
        <PersonPicker
          key={pickFirst ? 'pick' : 'plain'}
          autoFocus={pickFirst || (!sheet && !guess)}
          placeholder={
            groupScope
              ? 'Type their name, or pick someone'
              : box.person
                ? 'Move to someone else'
                : guess
                  ? `Not ${personName(guess.person)}? Type who it is`
                  : 'Search or type a new name'
          }
          excludeIds={box.person ? [box.person.id] : guess ? [guess.person.id] : undefined}
          onPick={(choice) => {
            if (busy) return
            if (groupScope) nameGroup.mutate(choice)
            else if (anyGuess) correct.mutate(choice) // the guesses are learned as wrong, not silently overwritten
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
        {trackList && (
          <div className="mt-2 rounded-xl bg-surface-2 p-2 text-xs text-ink-soft" role="status">
            <p>{onlyTrack ? 'Applies to this track only.' : `Applies to all ${trackList.length} tracks of this person in the video.`}</p>
            <label className="mt-1 flex min-h-10 cursor-pointer items-center gap-2 text-sm">
              <input type="checkbox" checked={onlyTrack} onChange={(e) => setOnlyTrack(e.target.checked)} />
              Only this track
            </label>
          </div>
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
        {onAdjust && (
          <Button variant="secondary" disabled={busy} onClick={onAdjust}>
            <Move className="size-4" aria-hidden />
            Move or resize
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
