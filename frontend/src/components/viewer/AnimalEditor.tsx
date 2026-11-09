// Popover for one animal box: name the pet (pick one of that species, or a new name), take the
// name off, say what species it really is (section 22: look-alike crops follow), move it to a pet
// of another species by making that pet this species too, or remove the box when it is not an
// animal. Same shape as the face editor.
import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Ban, Move, PawPrint, Shuffle, UserX, X } from 'lucide-react'
import { api } from '../../lib/api'
import { aSpecies, formatCount, personName, speciesLabel } from '../../lib/format'
import { boxOrigin, mergeOrigins } from '../../lib/origin'
import { SPECIES, type PersonOut, type Species } from '../../lib/types'
import OriginIcon from '../people/OriginIcon'
import PersonPicker, { choiceBody, type PersonChoice } from '../people/PersonPicker'
import { Button } from '../ui'
import type { AnimalBox } from './boxes'
import { placement } from './FaceEditor'
import { useInvalidateItem } from './shared'

interface Props {
  uid: string
  box: AnimalBox
  /** Whole-video editing (section 10): every track of this pet in the video, the box among them. */
  tracks?: AnimalBox[]
  anchor: HTMLElement | null
  onClose: () => void
  onDone: (message: string) => void
  /** Move or resize the box on the picture (section 10). */
  onAdjust?: () => void
}

/** " 2 look-alike crops followed." after a species change, or nothing. */
export function followedCrops(followed: number | undefined): string {
  return followed ? ` ${formatCount(followed, 'look-alike crop')} followed.` : ''
}

export default function AnimalEditor({ uid, box, anchor, onClose, onDone, tracks, onAdjust }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [style, setStyle] = useState<CSSProperties>(() => placement(anchor))
  const [picking, setPicking] = useState(false)
  // a pet of another species was picked: the move needs that pet to become this species first
  const [mismatch, setMismatch] = useState<PersonOut | null>(null)
  const invalidate = useInvalidateItem(uid)
  const sheet = style.bottom === 0 && style.left === 0
  const species = speciesLabel(box.species)
  const trackList = tracks && tracks.length > 1 ? tracks : null
  const [onlyTrack, setOnlyTrack] = useState(false)
  const ids = trackList && !onlyTrack ? trackList.map((b) => b.id) : [box.id]
  const trackTail = ids.length > 1 ? ` (${ids.length} tracks)` : ''
  // who put the pet there (section 10): for the whole group of tracks, the owner when any track is theirs
  const origin = trackList && !onlyTrack ? mergeOrigins(trackList.map((b) => boxOrigin(b))) : boxOrigin(box)

  useLayoutEffect(() => setStyle(placement(anchor)), [anchor, box.id])
  useEffect(() => {
    setPicking(false)
    setMismatch(null)
  }, [box.id])

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
    mutationFn: (v: { choice: PersonChoice | null; fix?: boolean }) =>
      api.animals.assign({ animal_ids: ids, ...choiceBody(v.choice), ...(v.fix ? { fix_species: true } : {}) }),
    onSuccess: (res, v) => {
      invalidate()
      setMismatch(null)
      if (!v.choice) onDone((box.person ? `Removed from ${personName(box.person)}` : 'Unassigned') + trackTail)
      else if (v.fix && res.person) onDone(`Tagged as ${personName(res.person)}, now ${aSpecies(box.species)}${trackTail}`)
      else onDone((res.person ? `Tagged as ${personName(res.person)}` : 'Saved') + trackTail)
    },
  })
  const setSpecies = useMutation({
    mutationFn: (next: Species) => api.animals.setSpecies({ animal_ids: ids, species: next }),
    onSuccess: (res, next) => {
      invalidate()
      setPicking(false)
      onDone(`Changed to ${speciesLabel(next).toLowerCase()}.${followedCrops(res.followed)}`)
    },
  })
  const remove = useMutation({
    mutationFn: async () => {
      for (const id of ids) await api.animals.remove(id)
    },
    onSuccess: () => {
      invalidate()
      onDone(`Removed. It will not be shown as a pet again.${trackTail}`)
    },
  })
  const busy = assign.isPending || remove.isPending || setSpecies.isPending
  const error = assign.error ?? remove.error ?? setSpecies.error

  const pick = (choice: PersonChoice, person?: PersonOut) => {
    if (busy) return
    if (person && person.species && person.species !== box.species) {
      setMismatch(person)
      return
    }
    setMismatch(null)
    assign.mutate({ choice })
  }

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
            {box.species_source === 'user' && ' (your choice)'}
          </p>
          <p className="flex min-w-0 items-center gap-1 text-sm font-semibold">
            <span className="truncate">{box.person ? personName({ ...box.person, species: box.species }) : 'Which pet is this?'}</span>
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

      <div className="mt-3">
        <PersonPicker
          kind="pet"
          species={box.species}
          autoFocus={!sheet}
          placeholder={box.person ? 'Move to another pet' : 'Search pets or type a new name'}
          excludeIds={box.person ? [box.person.id] : undefined}
          onPick={pick}
        />
        <p className="mt-2 text-xs text-ink-soft">Pets of the same species come first. A pet of another species can take this box once it becomes {aSpecies(box.species)} too.</p>
        {trackList && (
          <div className="mt-2 rounded-xl bg-surface-2 p-2 text-xs text-ink-soft" role="status">
            <p>{onlyTrack ? 'Applies to this track only.' : `Applies to all ${trackList.length} tracks of this pet in the video.`}</p>
            <label className="mt-1 flex min-h-10 cursor-pointer items-center gap-2 text-sm">
              <input type="checkbox" checked={onlyTrack} onChange={(e) => setOnlyTrack(e.target.checked)} />
              Only this track
            </label>
          </div>
        )}
      </div>

      {mismatch && (
        <div className="mt-2 rounded-xl bg-surface-2 p-2 text-sm" role="status">
          <p className="text-ink-soft">
            {personName(mismatch)} is {aSpecies(mismatch.species)}; this box is {aSpecies(box.species)}.
          </p>
          <Button
            className="mt-2"
            variant="secondary"
            disabled={busy}
            loading={assign.isPending && !!assign.variables?.fix}
            onClick={() => assign.mutate({ choice: { personId: mismatch.id }, fix: true })}
          >
            <Shuffle className="size-4" aria-hidden />
            Move and make {personName(mismatch)} {aSpecies(box.species)}
          </Button>
        </div>
      )}

      {picking ? (
        <div className="mt-3" role="group" aria-label="Species">
          <p className="mb-1.5 text-xs text-ink-soft">This is a</p>
          <div className="flex flex-wrap gap-1.5">
            {SPECIES.map((s) => (
              <button
                key={s}
                type="button"
                disabled={busy || s === box.species}
                aria-pressed={s === box.species}
                onClick={() => setSpecies.mutate(s)}
                className={
                  'h-9 rounded-full border px-3 text-sm font-medium transition-colors disabled:cursor-default ' +
                  (s === box.species
                    ? 'border-ink bg-ink text-surface'
                    : 'border-line bg-surface text-ink-soft hover:border-ink-faint hover:text-ink')
                }
              >
                {speciesLabel(s)}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {error && (
        <p className="mt-2 text-sm text-danger" role="alert">
          {error instanceof Error ? error.message : 'That did not work. Try again.'}
        </p>
      )}

      <div className="mt-3 flex flex-wrap gap-2">
        <Button variant="secondary" disabled={busy} aria-expanded={picking} onClick={() => setPicking((v) => !v)}>
          <PawPrint className="size-4" aria-hidden />
          Not {aSpecies(box.species)}
        </Button>
        {box.person && (
          <Button
            variant="secondary"
            disabled={busy}
            loading={assign.isPending && assign.variables?.choice === null}
            onClick={() => assign.mutate({ choice: null })}
          >
            <UserX className="size-4" aria-hidden />
            Not this pet
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
          Remove box
        </Button>
      </div>
    </div>
  )
}
