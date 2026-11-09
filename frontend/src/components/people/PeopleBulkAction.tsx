// "People" for selection bars (docs/ARCHITECTURE.md section 10, people on a selection). It opens a sheet
// "People in N items": on a person's or pet's page (`from`) a first block "Not NAME?" moves their
// appearances on the selected items to someone else, or says they are not NAME there ("Not NAME": their
// boxes stay, unassigned and remembered as not them, as "Not this person" and "Not this pet" do in the box
// editors); then everyone the items show (POST /api/items/people), each with Change to..., Rename and Not
// this person (Not this pet); and Tag someone in these. One sheet with steps (no modal over a modal), so it
// fits a phone. A move, a "Not NAME" or a tag closes the sheet and clears the selection; a rename keeps it
// open and reloads the list.
import { useMemo, useState, type FormEvent, type ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router'
import clsx from 'clsx'
import { ArrowLeft, Link2, Pencil, UserPlus, Users, UserRoundPen, UserX } from 'lucide-react'
import { api } from '../../lib/api'
import { SPECIES, type ItemPersonOut, type PeopleBulkOut, type PersonKind, type Species, type TagRelation } from '../../lib/types'
import { formatCount, personName, speciesLabel } from '../../lib/format'
import { useRefreshLibrary } from '../../lib/hooks'
import { Avatar, Button, Spinner } from '../ui'
import { useToast } from '../Toast'
import { AdminModal } from '../admin/AdminModal'
import { errorMessage } from '../admin/adminHooks'
import OriginIcon from './OriginIcon'
import PersonPicker, { choiceBody, type PersonChoice } from './PersonPicker'

/** The person or pet the selection was made on (their page). */
export interface BulkFrom {
  id: number
  name: string | null
  kind: PersonKind
  species: Species | null
}

interface Props {
  uids: string[]
  /** The selection is in the vault; the server checks the vault itself (403 while it is locked). */
  vault: boolean
  /** Another bulk action is running. */
  busy?: boolean
  /** Called after a move, a removal or a tag: the caller clears its selection. */
  onDone: () => void
  from?: BulkFrom
}

type Who = { id: number; name: string | null; kind: PersonKind; species: Species | null }

type Step =
  | { kind: 'list' }
  | { kind: 'move'; who: Who }
  | { kind: 'rename'; who: Who }
  | { kind: 'not'; who: Who }
  | { kind: 'tag' }
  | { kind: 'tag-species'; name: string }

const ITEMS_PEOPLE_KEY = 'items-people'

/** "Asha", or "Unnamed person" / "Unnamed pet" (the sheet's wording for someone without a name). */
export function displayName(p: { name: string | null; kind: PersonKind }): string {
  if (p.name?.trim()) return p.name
  return p.kind === 'pet' ? 'Unnamed pet' : 'Unnamed person'
}

/** The "Not ..." choice: "Not Asha", or "Not this person" / "Not this pet" for someone without a name. */
export function notLabel(p: { name: string | null; kind: PersonKind }): string {
  if (p.name?.trim()) return `Not ${p.name}`
  return p.kind === 'pet' ? 'Not this pet' : 'Not this person'
}

/** Who a "Not ..." is about, inside a sentence: "Asha", or "this person" / "this pet". */
function notWho(p: { name: string | null; kind: PersonKind }): string {
  if (p.name?.trim()) return p.name
  return p.kind === 'pet' ? 'this pet' : 'this person'
}

function itemsTail(n: number): string {
  return n === 1 ? 'this item' : `these ${n.toLocaleString()} items`
}

/** " 3 look-alike faces followed." and " 2 look-alike groups joined them." after a move or a removal. */
export function followedTail(r: PeopleBulkOut, kind: PersonKind): string {
  const followed = r.followed ? ` ${formatCount(r.followed, kind === 'pet' ? 'look-alike crop' : 'look-alike face')} followed.` : ''
  const pulled = r.pulled ? ` ${formatCount(r.pulled, 'look-alike group')} joined them.` : ''
  return followed + pulled
}

export default function PeopleBulkAction({ uids, busy = false, onDone, from }: Props) {
  const toast = useToast()
  const qc = useQueryClient()
  const navigate = useNavigate()
  const refresh = useRefreshLibrary()
  const [open, setOpen] = useState(false)
  const [step, setStep] = useState<Step>({ kind: 'list' })
  const [related, setRelated] = useState(false)
  const [nameText, setNameText] = useState('')
  const key = useMemo(() => [...uids].sort(), [uids])
  const count = uids.length

  const people = useQuery({
    queryKey: [ITEMS_PEOPLE_KEY, key],
    queryFn: () => api.people.onItems(uids),
    enabled: open && count > 0,
    staleTime: 0,
    gcTime: 0,
  })
  const total = people.data?.items ?? count

  const close = () => {
    setOpen(false)
    setStep({ kind: 'list' })
    setRelated(false)
    qc.removeQueries({ queryKey: [ITEMS_PEOPLE_KEY] })
  }
  const finished = (message: string, changed: boolean) => {
    if (changed) toast.success(message)
    else toast.info(message)
    close()
    onDone()
    void refresh()
  }

  const move = useMutation({
    mutationFn: (v: { who: Who; choice: PersonChoice }) => {
      const body = choiceBody(v.choice)
      return api.people.moveOnItems({
        uids,
        from_person_id: v.who.id,
        person_id: body.person_id ?? undefined,
        new_person_name: body.new_person_name,
      })
    },
    onSuccess: (r, v) =>
      finished(
        r.updated
          ? `${formatCount(r.updated, 'item')} moved to ${r.person ? personName(r.person) : 'someone else'}.${followedTail(r, v.who.kind)}`
          : `${displayName(v.who)} is not on ${itemsTail(count)}: nothing moved.`,
        r.updated > 0,
      ),
    onError: (e) => toast.error(errorMessage(e, 'Could not move them.')),
  })
  // "Not NAME" on these items (POST /api/items/people/remove): their boxes stay, unassigned and remembered as
  // not them, look-alikes go along, and their tags here go.
  const notThem = useMutation({
    mutationFn: (who: Who) => api.people.removeFromItems({ uids, person_id: who.id }),
    onSuccess: (r, who) =>
      finished(
        r.updated
          ? `${formatCount(r.updated, 'item')} marked as not ${notWho(who)}.${followedTail(r, who.kind)}`
          : `${displayName(who)} is not on ${itemsTail(count)}: nothing changed.`,
        r.updated > 0,
      ),
    onError: (e) => toast.error(errorMessage(e, 'Could not save that.')),
  })
  const tag = useMutation({
    mutationFn: (v: { choice?: PersonChoice; newPet?: { name: string; species: Species }; relation: TagRelation }) =>
      api.people.tagItems(
        v.newPet
          ? { uids, new_person_name: v.newPet.name, kind: 'pet', species: v.newPet.species, relation: v.relation }
          : { uids, ...choiceBody(v.choice ?? null), relation: v.relation },
      ),
    onSuccess: (r, v) =>
      finished(
        r.updated
          ? `${formatCount(r.updated, 'item')} tagged ${r.person ? personName(r.person) : ''}${v.relation === 'related' ? ' as related' : ''}.`
          : `${r.person ? personName(r.person) : 'They'} already ${r.person?.kind === 'pet' ? 'is' : 'are'} on ${itemsTail(count)}.`,
        r.updated > 0,
      ),
    onError: (e) => toast.error(errorMessage(e, 'Could not tag them.')),
  })
  const rename = useMutation({
    mutationFn: (v: { who: Who; name: string | null }) => api.people.update(v.who.id, { name: v.name }),
    onSuccess: (p, v) => {
      // A name someone else already has merges this person into them (the server returns that person).
      const merged = p.id !== v.who.id
      toast.success(merged ? `Merged into ${personName(p)}.` : p.name ? `Renamed to ${p.name}.` : 'Name removed.')
      void refresh()
      if (merged && from && v.who.id === from.id) {
        // The page's own person went away: follow them to the one they joined.
        close()
        onDone()
        navigate(`/people/${p.id}`, { replace: true })
        return
      }
      setStep({ kind: 'list' })
      void qc.invalidateQueries({ queryKey: [ITEMS_PEOPLE_KEY] })
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not save the name.')),
  })
  const working = move.isPending || notThem.isPending || tag.isPending || rename.isPending

  const startRename = (who: Who) => {
    setNameText(who.name ?? '')
    setStep({ kind: 'rename', who })
  }
  const pickTag = (choice: PersonChoice) => {
    if (working) return
    if ('newName' in choice && choice.kind === 'pet') {
      setStep({ kind: 'tag-species', name: choice.newName })
      return
    }
    tag.mutate({ choice, relation: related ? 'related' : 'in' })
  }
  const submitRename = (e: FormEvent) => {
    e.preventDefault()
    if (step.kind !== 'rename' || working) return
    const next = nameText.trim() || null
    if (next === (step.who.name ?? null)) {
      setStep({ kind: 'list' })
      return
    }
    rename.mutate({ who: step.who, name: next })
  }

  const back = (
    <button
      type="button"
      onClick={() => setStep({ kind: 'list' })}
      disabled={working}
      className="mb-3 inline-flex h-10 items-center gap-1.5 rounded-full px-2 text-sm font-medium text-ink-soft hover:bg-ink/5 hover:text-ink disabled:opacity-50"
    >
      <ArrowLeft className="size-4" aria-hidden />
      Back
    </button>
  )

  const row = (p: ItemPersonOut) => {
    const who: Who = { id: p.id, name: p.name, kind: p.kind, species: p.species }
    const name = displayName(p)
    const detail = [`in ${p.items.toLocaleString()} of ${total.toLocaleString()}`]
    // section 10: on how many the owner put them (a tag, a box they tagged or drew); older servers do not say
    const mine = p.by_you === undefined ? null : Math.min(p.by_you, p.items)
    const auto = mine === null ? 0 : p.items - mine
    if (mine && auto) detail.push(`${mine.toLocaleString()} by you`)
    if (p.related) detail.push(`related on ${p.related.toLocaleString()}`)
    if (p.kind === 'pet' && p.species) detail.push(speciesLabel(p.species).toLowerCase())
    return (
      <li key={p.id} aria-label={name} className="py-3 first:pt-0 last:pb-0">
        <div className="flex min-w-0 items-center gap-3">
          <Avatar src={p.cover_url} name={p.name ?? '?'} size={36} />
          <div className="min-w-0 flex-1">
            <p className="flex min-w-0 items-center gap-1.5 text-sm font-semibold">
              <span className={clsx('truncate', !p.name && 'text-ink-soft')}>{name}</span>
              {!!mine && <OriginIcon origin="you" label={`Tagged or drawn by you on ${formatCount(mine, 'item')}`} className="text-ink-soft" />}
              {auto > 0 && <OriginIcon origin="auto" label={`Found automatically on ${formatCount(auto, 'item')}`} className="text-ink-soft" />}
            </p>
            <p className="text-xs text-ink-soft">{detail.join(', ')}</p>
          </div>
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5 pl-12">
          <Button size="sm" variant="secondary" disabled={working} onClick={() => setStep({ kind: 'move', who })}>
            <UserRoundPen className="size-4" aria-hidden />
            Change to...
          </Button>
          <Button size="sm" variant="secondary" disabled={working} onClick={() => startRename(who)}>
            <Pencil className="size-4" aria-hidden />
            {p.name ? 'Rename' : 'Name'}
          </Button>
          <Button size="sm" variant="secondary" disabled={working} onClick={() => setStep({ kind: 'not', who })}>
            <UserX className="size-4" aria-hidden />
            {p.kind === 'pet' ? 'Not this pet' : 'Not this person'}
          </Button>
        </div>
      </li>
    )
  }

  let body: ReactNode
  if (step.kind === 'move') {
    const who = step.who
    const pet = who.kind === 'pet'
    body = (
      <div>
        {back}
        <p className="mb-3 text-sm text-ink-soft">
          Move {displayName(who)} on {itemsTail(total)} to {pet ? 'another pet' : 'someone else'}. Their faces, boxes and tags
          there go to the one you pick{pet ? '' : '; look-alike faces may follow'}.
        </p>
        {move.isPending ? (
          <div className="flex justify-center py-8">
            <Spinner label="Moving" />
          </div>
        ) : (
          <PersonPicker
            autoFocus
            kind={pet ? 'pet' : 'person'}
            species={who.species ?? undefined}
            speciesOnly={pet && !!who.species}
            excludeIds={[who.id]}
            placeholder={pet ? 'Search pets or type a new name' : 'Search people or type a new name'}
            onPick={(choice) => !working && move.mutate({ who, choice })}
          />
        )}
      </div>
    )
  } else if (step.kind === 'rename') {
    const who = step.who
    body = (
      <form onSubmit={submitRename}>
        {back}
        <label className="mb-2 block text-sm text-ink-soft" htmlFor="people-bulk-name">
          {who.name ? `Rename ${who.name} everywhere, not only on these items.` : `Name this ${who.kind === 'pet' ? 'pet' : 'person'} everywhere.`} A
          name someone else already has merges them.
        </label>
        <div className="flex items-center gap-2">
          <input
            id="people-bulk-name"
            autoFocus
            value={nameText}
            onChange={(e) => setNameText(e.target.value)}
            maxLength={120}
            placeholder={who.kind === 'pet' ? 'Its name' : 'Their name'}
            aria-label="New name"
            className="h-10 min-w-0 flex-1 rounded-full border border-line bg-surface px-4 text-base text-ink placeholder:text-ink-faint focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20 sm:text-sm"
          />
          <Button type="submit" loading={rename.isPending}>
            Save
          </Button>
        </div>
      </form>
    )
  } else if (step.kind === 'not') {
    const who = step.who
    const pet = who.kind === 'pet'
    body = (
      <div>
        {back}
        <p className="text-sm font-semibold text-ink">
          Not {notWho(who)} on {itemsTail(total)}?
        </p>
        <p className="mt-1 text-sm text-ink-soft">
          {pet
            ? `Its boxes here stay, unassigned and remembered as not ${notWho(who)}, so they are not matched to it again. Look-alike crops go along, and its tags here are removed.`
            : `Their faces and bodies here stay, unassigned and remembered as not ${notWho(who)}, so they are not matched to them again. Look-alike faces go along, and their tags here are removed.`}
        </p>
        <div className="mt-4 flex flex-wrap justify-end gap-2">
          <Button variant="ghost" disabled={notThem.isPending} onClick={() => setStep({ kind: 'list' })}>
            Cancel
          </Button>
          <Button loading={notThem.isPending} onClick={() => notThem.mutate(who)}>
            <UserX className="size-4" aria-hidden />
            {notLabel(who)}
          </Button>
        </div>
      </div>
    )
  } else if (step.kind === 'tag') {
    body = (
      <div>
        {back}
        <button
          type="button"
          role="switch"
          aria-checked={related}
          onClick={() => setRelated((v) => !v)}
          className="mb-2 flex min-h-10 w-full items-center gap-3 rounded-xl px-1 text-left text-sm text-ink hover:bg-surface-2"
        >
          <span aria-hidden className={clsx('relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors', related ? 'bg-accent' : 'bg-ink/20')}>
            <span className={clsx('inline-block size-4 rounded-full bg-white shadow transition-transform', related ? 'translate-x-6' : 'translate-x-1')} />
          </span>
          <span className="min-w-0 flex-1">
            Not in the picture, related to them
            <span className="block text-xs text-ink-soft">
              {related ? 'Shown under Related on their page and found by search.' : 'Off: they are in the pictures.'}
            </span>
          </span>
        </button>
        {tag.isPending ? (
          <div className="flex justify-center py-8">
            <Spinner label="Tagging" />
          </div>
        ) : (
          <PersonPicker autoFocus kind="all" onPick={pickTag} />
        )}
      </div>
    )
  } else if (step.kind === 'tag-species') {
    const name = step.name
    body = (
      <div>
        <button
          type="button"
          onClick={() => setStep({ kind: 'tag' })}
          disabled={working}
          className="mb-3 inline-flex h-10 items-center gap-1.5 rounded-full px-2 text-sm font-medium text-ink-soft hover:bg-ink/5 hover:text-ink disabled:opacity-50"
        >
          <ArrowLeft className="size-4" aria-hidden />
          Back
        </button>
        <p className="mb-2 text-sm font-semibold text-ink">What is {name}?</p>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Species">
          {SPECIES.map((sp: Species) => (
            <button
              key={sp}
              type="button"
              disabled={working}
              onClick={() => tag.mutate({ newPet: { name, species: sp }, relation: related ? 'related' : 'in' })}
              className="rounded-full border border-line px-3 py-1.5 text-sm font-semibold text-ink hover:bg-surface-2 disabled:opacity-50"
            >
              {speciesLabel(sp)}
            </button>
          ))}
        </div>
        {tag.isPending && <p className="mt-2 text-xs text-ink-soft">Tagging</p>}
      </div>
    )
  } else {
    const list = people.data?.people ?? []
    body = (
      <div className="flex flex-col gap-5">
        {from && (
          <section aria-label={`Not ${displayName(from)}?`} className="rounded-2xl bg-surface-2 p-3">
            <p className="text-sm font-semibold text-ink">Not {displayName(from)}?</p>
            <p className="mt-0.5 text-xs text-ink-soft">
              Fix who {count === 1 ? 'this item shows' : `these ${count.toLocaleString()} items show`}.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <Button size="sm" disabled={working} onClick={() => setStep({ kind: 'move', who: from })}>
                <UserRoundPen className="size-4" aria-hidden />
                Move these to...
              </Button>
              <Button size="sm" variant="secondary" disabled={working} onClick={() => setStep({ kind: 'not', who: from })}>
                <UserX className="size-4" aria-hidden />
                {notLabel(from)}
              </Button>
            </div>
          </section>
        )}
        <section aria-label="In these items">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-ink-faint">In these items</h3>
          {people.isPending ? (
            <div className="flex justify-center py-6">
              <Spinner label="Loading people" />
            </div>
          ) : people.isError ? (
            <div className="rounded-xl bg-danger/10 px-3 py-2 text-sm text-danger" role="alert">
              {errorMessage(people.error, 'Could not load who is on these items.')}{' '}
              <button type="button" className="font-semibold underline" onClick={() => void people.refetch()}>
                Try again
              </button>
            </div>
          ) : list.length === 0 ? (
            <p className="text-sm text-ink-soft">Nobody is named on {itemsTail(total)} yet.</p>
          ) : (
            <ul className="divide-y divide-line">{list.map(row)}</ul>
          )}
          {(people.data?.unnamed_faces ?? 0) > 0 && (
            <p className="mt-3 text-xs text-ink-soft">
              {formatCount(people.data?.unnamed_faces ?? 0, 'face')} here {people.data?.unnamed_faces === 1 ? 'is' : 'are'} not named
              yet. Name {people.data?.unnamed_faces === 1 ? 'it' : 'them'} from the photo or the People page.
            </p>
          )}
        </section>
        <section aria-label="Tag someone in these">
          <Button variant="secondary" disabled={working} onClick={() => setStep({ kind: 'tag' })}>
            <UserPlus className="size-4" aria-hidden />
            Tag someone in these
          </Button>
          <p className="mt-1.5 flex items-center gap-1 text-xs text-ink-soft">
            <Link2 className="size-3.5" aria-hidden />
            Can also mark them as related, not in the pictures.
          </p>
        </section>
      </div>
    )
  }

  return (
    <>
      <Button variant="secondary" disabled={busy || count === 0} aria-haspopup="dialog" onClick={() => setOpen(true)}>
        <Users className="size-4" aria-hidden />
        People
      </Button>
      <AdminModal open={open} onClose={() => !working && close()} title={`People in ${formatCount(count, 'item')}`}>
        {body}
      </AdminModal>
    </>
  )
}

