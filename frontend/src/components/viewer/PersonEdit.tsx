// Editing a person from the viewer's info panel (docs/ARCHITECTURE.md section 14): the changes a face box makes,
// for everything of theirs on this item at once. "Change to..." moves their faces, bodies, pet boxes and tags here
// to someone else (POST /api/items/people/move, learned like a move from a face box), "Rename" names them
// everywhere (PATCH /api/people/{id}; a name someone else has merges them), and "Not NAME" takes them off this
// item: their boxes stay, unnamed and remembered as not them, and their tags here go (POST
// /api/items/people/remove). Like the box editors, the menu also takes their boxes here away: "Not a face" (their
// faces, kept for the not-a-face learner, look-alikes with them), "Not a person" (their person boxes without a face)
// and "Not a pet" (a pet's boxes), each one undoable action; and "Species..." says what kind of animal a pet is,
// everywhere like a rename (POST /api/people/{id}/species, its boxes follow). The chip menu in InfoPanel opens the card below for "Change to...",
// "Rename" and the species.
import { useEffect, useId, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { X } from 'lucide-react'
import { api } from '../../lib/api'
import { SPECIES, type PersonRef, type Species } from '../../lib/types'
import { formatCount, personName, speciesLabel } from '../../lib/format'
import { errorMessage } from '../admin/adminHooks'
import { displayName, followedTail } from '../people/PeopleBulkAction'
import PersonPicker, { choiceBody, type PersonChoice } from '../people/PersonPicker'
import { Button } from '../ui'
import { useInvalidateItem } from './shared'

export type PersonEditMode = 'change' | 'rename' | 'species'
/** The kind of box "Not a face", "Not a person" and "Not a pet" take away. */
export type BoxRemoval = 'face' | 'body' | 'animal'

export interface PersonEditTarget {
  mode: PersonEditMode
  who: PersonRef
  /** A pet's species (from its boxes here), so the picker lists that species. */
  species: Species | null
}

/** The three changes, with the id of the person one of them is working on (for the chip's spinner). */
export function usePersonEdits(uid: string, onToast: (message: string) => void, onDone: () => void) {
  const invalidate = useInvalidateItem(uid)
  const kindOf = (who: PersonRef) => who.kind ?? 'person'
  const change = useMutation({
    mutationFn: (v: { who: PersonRef; choice: PersonChoice }) => {
      const body = choiceBody(v.choice)
      return api.people.moveOnItems({
        uids: [uid],
        from_person_id: v.who.id,
        person_id: body.person_id ?? undefined,
        new_person_name: body.new_person_name,
      })
    },
    onSuccess: (r, v) => {
      invalidate()
      onDone()
      onToast(
        r.updated
          ? `Changed to ${r.person ? personName(r.person) : 'someone else'}.${followedTail(r, kindOf(v.who))}`
          : `${displayName({ name: v.who.name, kind: kindOf(v.who) })} is not on this item: nothing changed.`,
      )
    },
    onError: (e) => onToast(errorMessage(e, 'Could not change them.')),
  })
  const rename = useMutation({
    mutationFn: (v: { who: PersonRef; name: string | null }) => api.people.update(v.who.id, { name: v.name }),
    onSuccess: (p, v) => {
      invalidate()
      onDone()
      // A name someone else already has merges this person into them (the server returns that person).
      onToast(p.id !== v.who.id ? `Merged into ${personName(p)}.` : p.name ? (v.who.name ? `Renamed to ${p.name}.` : `Named ${p.name}.`) : 'Name removed.')
    },
    onError: (e) => onToast(errorMessage(e, 'Could not save the name.')),
  })
  const notThem = useMutation({
    mutationFn: (who: PersonRef) => api.people.removeFromItems({ uids: [uid], person_id: who.id }),
    onSuccess: (r, who) => {
      invalidate()
      onDone()
      const name = displayName({ name: who.name, kind: kindOf(who) })
      onToast(r.updated ? `Not ${name}: removed from this item.${followedTail(r, kindOf(who))}` : `${name} is not on this item: nothing changed.`)
    },
    onError: (e) => onToast(errorMessage(e, 'Could not remove them.')),
  })
  // "Not a face" / "Not a person" / "Not a pet": every box of that kind of theirs here, as the box editors remove one.
  // One mutation, so one undo puts all of them back.
  const removeBoxes = useMutation({
    /** stays: a face taken away sat in a person box of theirs, which keeps them on the item. */
    mutationFn: async (v: { who: PersonRef; kind: BoxRemoval; ids: number[]; stays?: boolean }) => {
      let followed = 0
      for (const id of v.ids) {
        if (v.kind === 'face') followed += (await api.people.deleteFace(id)).followed ?? 0
        else if (v.kind === 'body') await api.people.deleteBody(id)
        else await api.animals.remove(id)
      }
      return followed
    },
    // Some boxes may be gone after a failure half way: the item is read again either way.
    onSettled: () => invalidate(),
    onSuccess: (followed, v) => {
      onDone()
      const name = displayName({ name: v.who.name, kind: kindOf(v.who) })
      const what = v.kind === 'face' ? 'face' : v.kind === 'body' ? 'person box' : 'pet box'
      const boxes = v.ids.length === 1 ? `${name}'s ${what}` : `${name}'s ${formatCount(v.ids.length, what)}`
      const tail = followed ? ` ${formatCount(followed, 'look-alike face')} went with ${v.ids.length === 1 ? 'it' : 'them'}.` : ''
      const stays = v.stays ? ` ${name} stays here through their person box.` : ''
      onToast(
        v.kind === 'face'
          ? `Removed ${boxes} as not a face.${tail}${stays}`
          : v.kind === 'body'
            ? `Removed ${boxes}: not a person.`
            : `Removed ${boxes}: not a pet.`,
      )
    },
    onError: (e) => onToast(errorMessage(e, 'Could not remove the boxes.')),
  })
  // A pet's species, everywhere (its boxes follow), like a rename: the pet stays on this item.
  const species = useMutation({
    mutationFn: (v: { who: PersonRef; species: Species }) => api.people.setSpecies(v.who.id, v.species),
    onSuccess: (r, v) => {
      invalidate()
      onDone()
      const pet = r.person ?? v.who
      const crops = r.followed ? ` ${formatCount(r.followed, 'look-alike crop')} followed.` : ''
      const joined = r.person && r.person.id !== v.who.id ? ` It joined ${displayName({ name: r.person.name, kind: 'pet' })}.` : ''
      onToast(`${displayName({ name: pet.name ?? v.who.name, kind: 'pet' })} is a ${speciesLabel(v.species).toLowerCase()} now.${joined}${crops}`)
    },
    onError: (e) => onToast(errorMessage(e, 'Could not change the species.')),
  })
  const busyId = change.isPending
    ? change.variables?.who.id
    : rename.isPending
      ? rename.variables?.who.id
      : notThem.isPending
        ? notThem.variables?.id
        : removeBoxes.isPending
          ? removeBoxes.variables?.who.id
          : species.isPending
            ? species.variables?.who.id
            : undefined
  const busy = change.isPending || rename.isPending || notThem.isPending || removeBoxes.isPending || species.isPending
  return { change, rename, notThem, removeBoxes, species, busyId: busyId ?? null, busy }
}

interface CardProps {
  edit: PersonEditTarget
  busy: boolean
  /** The item is a video (the wording says so). */
  video: boolean
  onPick: (choice: PersonChoice) => void
  onRename: (name: string | null) => void
  /** A pet's boxes here are another species. */
  onSpecies: (species: Species) => void
  onCancel: () => void
}

/** The light card under the chips for "Change to..." (a person or pet picker) and "Rename" (a name field). */
export function PersonEditCard({ edit, busy, video, onPick, onRename, onSpecies, onCancel }: CardProps) {
  const fieldId = useId()
  const root = useRef<HTMLDivElement>(null)
  // The card opens under the chips, possibly below the visible part of the panel (the phone sheet), and grows when
  // the picker's list arrives: keep all of it in view. (Block bodies: an effect must return nothing but a cleanup.)
  useEffect(() => {
    const el = root.current
    if (!el) return
    el.scrollIntoView({ block: 'nearest' })
    if (typeof ResizeObserver === 'undefined') return
    const watch = new ResizeObserver(() => {
      el.scrollIntoView({ block: 'nearest' })
    })
    watch.observe(el)
    return () => watch.disconnect()
  }, [])
  const pet = edit.who.kind === 'pet'
  const name = displayName({ name: edit.who.name, kind: edit.who.kind ?? 'person' })
  const [text, setText] = useState(edit.who.name ?? '')
  const title =
    edit.mode === 'change'
      ? `Change ${name} to...`
      : edit.mode === 'species'
        ? `What is ${name}?`
        : edit.who.name
          ? `Rename ${name}`
          : `Name this ${pet ? 'pet' : 'person'}`

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (busy) return
    const next = text.trim() || null
    if (next === (edit.who.name ?? null)) {
      onCancel()
      return
    }
    onRename(next)
  }
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key !== 'Escape' || busy) return
    // The viewer closes itself on Escape unless something before it took the key.
    e.preventDefault()
    e.stopPropagation()
    onCancel()
  }

  return (
    <div ref={root} role="group" aria-label={title} className="mt-3 rounded-2xl bg-surface p-2.5" onKeyDown={onKeyDown}>
      <div className="mb-2 flex items-center justify-between gap-2 px-1">
        <p className="min-w-0 truncate text-sm font-semibold text-ink">{title}</p>
        <button
          type="button"
          onClick={onCancel}
          disabled={busy}
          className="flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft hover:bg-surface-2 disabled:opacity-50"
          aria-label="Cancel"
        >
          <X className="size-4" aria-hidden />
        </button>
      </div>
      {edit.mode === 'species' ? (
        <div className="px-1">
          <p className="mb-2 text-xs text-ink-soft">
            {`The species changes everywhere, on all its boxes, not only in this ${video ? 'video' : 'picture'}.`}
          </p>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Species">
            {SPECIES.map((s) => (
              <button
                key={s}
                type="button"
                autoFocus={s === SPECIES.find((x) => x !== edit.species)}
                disabled={busy || s === edit.species}
                aria-pressed={s === edit.species}
                onClick={() => onSpecies(s)}
                className={
                  'h-10 rounded-full border px-3 text-sm font-medium transition-colors disabled:cursor-default ' +
                  (s === edit.species
                    ? 'border-ink bg-ink text-surface'
                    : 'border-line bg-surface text-ink-soft hover:border-ink-faint hover:text-ink')
                }
              >
                {speciesLabel(s)}
              </button>
            ))}
          </div>
          {busy && <p className="mt-2 text-xs text-ink-soft" role="status">Changing</p>}
        </div>
      ) : edit.mode === 'change' ? (
        <>
          <p className="mb-2 px-1 text-xs text-ink-soft">
            {pet
              ? `Its boxes and tags in this ${video ? 'video' : 'picture'} go to the pet you pick.`
              : `Their faces and tags in this ${video ? 'video' : 'picture'} go to the one you pick, and look-alike faces may follow.`}
          </p>
          <PersonPicker
            autoFocus
            kind={pet ? 'pet' : 'person'}
            species={edit.species ?? undefined}
            speciesOnly={pet && !!edit.species}
            excludeIds={[edit.who.id]}
            placeholder={pet ? 'Search pets or type a new name' : 'Search people or type a new name'}
            onPick={(choice) => !busy && onPick(choice)}
          />
          {busy && <p className="mt-2 px-1 text-xs text-ink-soft" role="status">Changing</p>}
        </>
      ) : (
        <form onSubmit={submit} className="px-1">
          <label htmlFor={fieldId} className="mb-2 block text-xs text-ink-soft">
            {edit.who.name ? 'The new name shows everywhere, not only here.' : 'The name shows everywhere they appear.'} A name someone
            else already has merges them.
          </label>
          <div className="flex items-center gap-2">
            <input
              id={fieldId}
              autoFocus
              value={text}
              onChange={(e) => setText(e.target.value)}
              maxLength={120}
              placeholder={pet ? 'Its name' : 'Their name'}
              className="h-10 min-w-0 flex-1 rounded-full border border-line bg-surface px-4 text-base text-ink placeholder:text-ink-faint focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20 sm:text-sm"
            />
            <Button type="submit" loading={busy}>
              Save
            </Button>
          </div>
        </form>
      )}
    </div>
  )
}
