// One person (/people/:id): rename in place, cover, their items (with the viewer), their faces
// (move, remove, use as cover), guesses waiting for them, merge into someone else, delete.
// A pet has no cover picker, faces or guesses: it shows its animal crops instead.
import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router'
import { useMutation, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { ArrowLeft, Check, CheckCheck, GitMerge, Images, ListChecks, PawPrint, Pencil, ScanFace, Sparkles, Star, Trash2, UserMinus, X } from 'lucide-react'
import { ApiError, api } from '../lib/api'
import type { AnimalListItem, FaceListItem, GuessItem, PersonKind, Species } from '../lib/types'
import { followedTail, formatCount, personName, speciesLabel } from '../lib/format'
import { useInfiniteSentinel, usePageTitle, useRefreshLibrary, useStableCallback } from '../lib/hooks'
import { useItemList } from '../lib/useItemList'
import MediaGrid from '../components/MediaGrid'
import Viewer from '../components/Viewer'
import AnimalGrid from '../components/people/AnimalGrid'
import FaceGrid from '../components/people/FaceGrid'
import GuessCard from '../components/people/GuessCard'
import PersonChoiceModal from '../components/people/PersonChoiceModal'
import { choiceBody, type PersonChoice } from '../components/people/PersonPicker'
import { guessKey, useGuessReview } from '../components/people/useGuessReview'
import { useToast } from '../components/Toast'
import { Confirm } from '../components/admin/AdminModal'
import { errorMessage, useSelection } from '../components/admin/adminHooks'
import { LoadMore, SelectionBar } from '../components/admin/fields'
import { useViewerParam } from '../components/public/useViewerParam'
import { Avatar, Button, EmptyState, ErrorState, PageSpinner, buttonClasses } from '../components/ui'

type Tab = 'items' | 'faces' | 'guesses' | 'animals'

function NameEditor({ id, name, kind, species }: { id: number; name: string | null; kind: PersonKind; species: Species | null }) {
  const toast = useToast()
  const navigate = useNavigate()
  const refresh = useRefreshLibrary()
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState(name ?? '')
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (!editing) setText(name ?? '')
  }, [name, editing])
  useEffect(() => {
    if (editing) input.current?.select()
  }, [editing])
  const save = useMutation({
    mutationFn: (next: string | null) => api.people.update(id, { name: next }),
    onSuccess: (p) => {
      setEditing(false)
      // A name someone else already has merges this person into them; the server returns that person.
      if (p.id !== id) {
        toast.success(`Merged into ${personName(p)}.${followedTail(p.followed)}`)
        void refresh()
        navigate(`/people/${p.id}`, { replace: true })
        return
      }
      toast.success(p.name ? `Named ${p.name}.` : 'Name removed.')
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not save the name.')),
  })
  const submit = (e: FormEvent) => {
    e.preventDefault()
    const next = text.trim() || null
    if (next === (name ?? null)) return setEditing(false)
    save.mutate(next)
  }
  if (!editing)
    return (
      <div className="flex min-w-0 items-center gap-1">
        <h1 className={clsx('min-w-0 break-words text-2xl font-bold tracking-tight sm:text-3xl', !name && 'text-ink-soft')}>{personName({ name, kind, species })}</h1>
        <button
          type="button"
          onClick={() => setEditing(true)}
          className="flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft hover:bg-surface-2 hover:text-ink"
          aria-label={name ? 'Rename' : 'Add a name'}
        >
          <Pencil className="size-4" aria-hidden />
        </button>
      </div>
    )
  return (
    <form onSubmit={submit} className="flex min-w-0 items-center gap-1">
      <input
        ref={input}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === 'Escape' && setEditing(false)}
        maxLength={120}
        placeholder={kind === 'pet' ? 'Its name' : 'Their name'}
        aria-label="Name"
        className="h-11 min-w-0 flex-1 rounded-full border border-accent bg-surface px-4 text-lg font-semibold text-ink focus:outline-none focus:ring-2 focus:ring-accent/20"
      />
      <Button type="submit" size="md" loading={save.isPending} aria-label="Save name">
        {!save.isPending && <Check className="size-4" aria-hidden />}
      </Button>
      <Button type="button" variant="ghost" size="md" onClick={() => setEditing(false)} aria-label="Cancel">
        <X className="size-4" aria-hidden />
      </Button>
    </form>
  )
}

function ItemsTab({ id }: { id: number }) {
  const fetchPage = useCallback((c: string | null) => api.people.items(id, c, 60), [id])
  const list = useItemList(['person-items', id], fetchPage)
  const viewer = useViewerParam(list.items, list.hasMore, list.loadMore, list.total)
  const openItem = useStableCallback((uid: string) => viewer.open(uid))
  return (
    <div className="-mx-4 sm:mx-0">
      {list.error && !list.items.length ? (
        <ErrorState error={list.error} onRetry={list.refetch} />
      ) : (
        <MediaGrid
          items={list.items}
          onOpen={openItem}
          hasMore={list.hasMore}
          loadMore={list.loadMore}
          loading={list.loading}
          empty={<EmptyState icon={<Images className="size-8" aria-hidden />} title="No items yet" />}
        />
      )}
      {viewer.uid && <Viewer uid={viewer.uid} onClose={viewer.close} onPrev={viewer.onPrev} onNext={viewer.onNext} position={viewer.position} />}
    </div>
  )
}

function FacesTab({ id, name, coverId }: { id: number; name: string; coverId: number | null }) {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const fetchPage = useCallback((c: string | null) => api.people.faces(id, c, 120), [id])
  const list = useItemList<FaceListItem>(['person-faces', id], fetchPage, { keyOf: (f) => f.id })
  const sel = useSelection<number>()
  const [selecting, setSelecting] = useState(false)
  const [moving, setMoving] = useState(false)
  const itemRefs = useMemo(() => {
    const seen = new Set<string>()
    return list.items.filter((f) => (seen.has(f.item_uid) ? false : (seen.add(f.item_uid), true))).map((f) => ({ uid: f.item_uid }))
  }, [list.items])
  const viewer = useViewerParam(itemRefs, list.hasMore, list.loadMore)

  const assign = useMutation({
    mutationFn: (choice: PersonChoice | null) => api.people.assignFaces({ face_ids: [...sel.selected], ...choiceBody(choice) }),
    onSuccess: (r, choice) => {
      const tail = r.followed ? ` ${formatCount(r.followed, 'look-alike face')} went the same way.` : ''
      toast.success((choice ? `${formatCount(r.updated, 'face')} moved${r.person ? ` to ${personName(r.person)}` : ''}.` : `${formatCount(r.updated, 'face')} removed from ${name}.`) + tail)
      setMoving(false)
      sel.clear()
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not move the faces.')),
  })
  const cover = useMutation({
    mutationFn: (faceId: number) => api.people.update(id, { cover_face_id: faceId }),
    onSuccess: () => {
      toast.success('Cover changed.')
      sel.clear()
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not change the cover.')),
  })
  const busy = assign.isPending || cover.isPending

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Button
          variant="secondary"
         
          aria-pressed={selecting}
          onClick={() => {
            sel.clear()
            setSelecting((v) => !v)
          }}
        >
          <ListChecks className="size-4" aria-hidden />
          {selecting ? 'Done' : 'Select faces'}
        </Button>
        <p className="text-sm text-ink-soft">{selecting ? 'Pick faces to move, remove, or use as the cover.' : 'Tap a face to open where it was found.'}</p>
      </div>
      {list.error && !list.items.length ? (
        <ErrorState error={list.error} onRetry={list.refetch} />
      ) : list.loading && !list.items.length ? (
        <PageSpinner />
      ) : list.items.length === 0 ? (
        <EmptyState icon={<ScanFace className="size-8" aria-hidden />} title="No faces">
          This person was tagged by hand or by clothes only.
        </EmptyState>
      ) : (
        <FaceGrid
          faces={list.items}
          selecting={selecting}
          selected={sel.selected}
          onToggle={sel.toggle}
          onOpen={(f) => viewer.open(f.item_uid, f.frame_time_s)}
          hasMore={list.hasMore}
          loadMore={list.loadMore}
          loadingMore={list.loading}
          coverId={coverId}
        />
      )}
      {selecting && sel.count > 0 && (
        <SelectionBar count={sel.count} noun={sel.count === 1 ? 'face' : 'faces'} onClear={sel.clear} onSelectAll={() => sel.setAll(list.items.map((f) => f.id))} allSelected={sel.count >= list.items.length} total={list.items.length}>
          {sel.count === 1 && (
            <Button variant="secondary" disabled={busy} loading={cover.isPending} onClick={() => cover.mutate([...sel.selected][0])}>
              <Star className="size-4" aria-hidden />
              Use as cover
            </Button>
          )}
          <Button variant="secondary" disabled={busy} onClick={() => setMoving(true)}>
            <GitMerge className="size-4" aria-hidden />
            Move to
          </Button>
          <Button variant="danger" disabled={busy} loading={assign.isPending && assign.variables === null} onClick={() => assign.mutate(null)}>
            <UserMinus className="size-4" aria-hidden />
            Not {name}
          </Button>
        </SelectionBar>
      )}
      <PersonChoiceModal
        open={moving}
        title={`Move ${formatCount(sel.count, 'face')} to`}
        busy={assign.isPending}
        excludeIds={[id]}
        onPick={(c) => assign.mutate(c)}
        onClose={() => setMoving(false)}
      />
      {viewer.uid && <Viewer uid={viewer.uid} onClose={viewer.close} onPrev={viewer.onPrev} onNext={viewer.onNext} position={viewer.position} startAt={viewer.t} />}
    </div>
  )
}

function AnimalsTab({ id, name, species }: { id: number; name: string; species: Species | null }) {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const fetchPage = useCallback((c: string | null) => api.people.animals(id, c, 120), [id])
  const list = useItemList<AnimalListItem>(['person-animals', id], fetchPage, { keyOf: (a) => a.id })
  const sel = useSelection<number>()
  const [selecting, setSelecting] = useState(false)
  const [moving, setMoving] = useState(false)
  const [removing, setRemoving] = useState(false)
  const itemRefs = useMemo(() => {
    const seen = new Set<string>()
    return list.items.filter((a) => (seen.has(a.item_uid) ? false : (seen.add(a.item_uid), true))).map((a) => ({ uid: a.item_uid }))
  }, [list.items])
  const viewer = useViewerParam(itemRefs, list.hasMore, list.loadMore)

  const assign = useMutation({
    mutationFn: (choice: PersonChoice | null) => api.animals.assign({ animal_ids: [...sel.selected], ...choiceBody(choice) }),
    onSuccess: (r, choice) => {
      toast.success(choice ? `${formatCount(r.updated, 'crop')} moved${r.person ? ` to ${personName(r.person)}` : ''}.` : `${formatCount(r.updated, 'crop')} removed from ${name}.`)
      setMoving(false)
      sel.clear()
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not move the crops.')),
  })
  const remove = useMutation({
    mutationFn: (ids: number[]) => Promise.all(ids.map((animalId) => api.animals.remove(animalId))),
    onSuccess: (done) => {
      toast.success(`${formatCount(done.length, 'box', 'boxes')} removed.`)
      setRemoving(false)
      sel.clear()
      void refresh()
    },
    onError: (e) => {
      setRemoving(false)
      toast.error(errorMessage(e, 'Could not remove the boxes.'))
    },
  })
  const busy = assign.isPending || remove.isPending

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Button
          variant="secondary"
          aria-pressed={selecting}
          onClick={() => {
            sel.clear()
            setSelecting((v) => !v)
          }}
        >
          <ListChecks className="size-4" aria-hidden />
          {selecting ? 'Done' : 'Select crops'}
        </Button>
        <p className="text-sm text-ink-soft">{selecting ? 'Pick crops to move or remove.' : 'Tap a crop to open where it was found.'}</p>
      </div>
      {list.error && !list.items.length ? (
        <ErrorState error={list.error} onRetry={list.refetch} />
      ) : list.loading && !list.items.length ? (
        <PageSpinner />
      ) : list.items.length === 0 ? (
        <EmptyState icon={<PawPrint className="size-8" aria-hidden />} title="No crops yet" />
      ) : (
        <AnimalGrid
          animals={list.items}
          selecting={selecting}
          selected={sel.selected}
          onToggle={sel.toggle}
          onOpen={(a) => viewer.open(a.item_uid, a.frame_time_s)}
          hasMore={list.hasMore}
          loadMore={list.loadMore}
          loadingMore={list.loading}
        />
      )}
      {selecting && sel.count > 0 && (
        <SelectionBar count={sel.count} noun={sel.count === 1 ? 'crop' : 'crops'} onClear={sel.clear} onSelectAll={() => sel.setAll(list.items.map((a) => a.id))} allSelected={sel.count >= list.items.length} total={list.items.length}>
          <Button variant="secondary" disabled={busy} onClick={() => setMoving(true)}>
            <GitMerge className="size-4" aria-hidden />
            Move to
          </Button>
          <Button variant="secondary" disabled={busy} loading={assign.isPending && assign.variables === null} onClick={() => assign.mutate(null)}>
            <UserMinus className="size-4" aria-hidden />
            Not {name}
          </Button>
          <Button variant="danger" disabled={busy} onClick={() => setRemoving(true)}>
            <Trash2 className="size-4" aria-hidden />
            Remove boxes
          </Button>
        </SelectionBar>
      )}
      <PersonChoiceModal
        open={moving}
        title={`Move ${formatCount(sel.count, 'crop')} to`}
        busy={assign.isPending}
        excludeIds={[id]}
        kind="pet"
        species={species ?? undefined}
        onPick={(c) => assign.mutate(c)}
        onClose={() => setMoving(false)}
      />
      <Confirm
        open={removing}
        title={`Remove ${formatCount(sel.count, 'box', 'boxes')}?`}
        body="These are not pets. The boxes are removed from the photos and will not come back."
        confirmLabel="Remove"
        danger
        loading={remove.isPending}
        onConfirm={() => remove.mutate([...sel.selected])}
        onClose={() => !remove.isPending && setRemoving(false)}
      />
      {viewer.uid && <Viewer uid={viewer.uid} onClose={viewer.close} onPrev={viewer.onPrev} onNext={viewer.onNext} position={viewer.position} startAt={viewer.t} />}
    </div>
  )
}

function GuessesTab({ id }: { id: number }) {
  const fetchPage = useCallback((c: string | null) => api.review.guesses(c, 60, id), [id])
  const list = useItemList<GuessItem>(['review', 'guesses', id], fetchPage, { keyOf: guessKey })
  const mine = useMemo(() => list.items.filter((g) => g.guess.person.id === id), [list.items, id])
  const review = useGuessReview()
  const items = useMemo(() => {
    const seen = new Set<string>()
    return mine.map((g) => g.item).filter((i) => (seen.has(i.uid) ? false : (seen.add(i.uid), true)))
  }, [mine])
  const viewer = useViewerParam(items, false, () => undefined)
  const sentinel = useInfiniteSentinel(list.loadMore, list.hasMore && !list.loading)

  return (
    <div>
      {mine.length > 1 && (
        <div className="mb-3">
          <Button variant="secondary" disabled={review.pending} onClick={() => review.confirm(mine)}>
            <CheckCheck className="size-4" aria-hidden />
            Confirm all {mine.length}
          </Button>
        </div>
      )}
      {list.error && !list.items.length ? (
        <ErrorState error={list.error} onRetry={list.refetch} />
      ) : list.loading && !list.items.length ? (
        <PageSpinner />
      ) : mine.length === 0 && !list.hasMore ? (
        <EmptyState icon={<Sparkles className="size-8" aria-hidden />} title="No guesses for this person">
          When this person is tagged, other photos within an hour with the same clothes get a guess here.
        </EmptyState>
      ) : (
        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {mine.map((g) => (
            <li key={guessKey(g)}>
              <GuessCard g={g} busy={review.busyFor(g)} onConfirm={() => review.confirm([g])} onReject={() => review.reject([g])} onOpenItem={(uid) => viewer.open(uid)} />
            </li>
          ))}
        </ul>
      )}
      <LoadMore hasMore={list.hasMore} loading={list.loading} onClick={list.loadMore} sentinelRef={sentinel} />
      {viewer.uid && <Viewer uid={viewer.uid} onClose={viewer.close} onPrev={viewer.onPrev} onNext={viewer.onNext} position={viewer.position} />}
    </div>
  )
}

function PersonView({ id }: { id: number }) {
  const navigate = useNavigate()
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const [search, setSearch] = useSearchParams()
  const raw = search.get('tab')
  const person = useQuery({ queryKey: ['person', id], queryFn: () => api.people.get(id) })
  const p = person.data
  usePageTitle(p ? personName(p) : 'Person')

  const [mergeOpen, setMergeOpen] = useState(false)
  const [mergeTarget, setMergeTarget] = useState<number | null>(null)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const targetQ = useQuery({ queryKey: ['person', mergeTarget], queryFn: () => api.people.get(mergeTarget as number), enabled: mergeTarget !== null })

  const merge = useMutation({
    mutationFn: (target: number) => api.people.merge({ target_id: target, source_ids: [id] }),
    onSuccess: (t) => {
      toast.success(`Merged into ${personName(t)}.${followedTail(t.followed)}`)
      setMergeTarget(null)
      void refresh()
      navigate(`/people/${t.id}`, { replace: true })
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not merge.')),
  })
  const remove = useMutation({
    mutationFn: () => api.people.remove(id),
    onSuccess: () => {
      toast.success(person.data?.kind === 'pet' ? 'Pet deleted. Its crops are unassigned again.' : 'Person deleted. Their faces are unassigned again.')
      void refresh()
      navigate('/people', { replace: true })
    },
    onError: (e) => {
      setDeleteOpen(false)
      toast.error(errorMessage(e, 'Could not delete.'))
    },
  })

  const setTab = (t: Tab) => {
    const next = new URLSearchParams()
    if (t !== 'items') next.set('tab', t)
    setSearch(next, { replace: true })
  }

  if (person.isPending) return <PageSpinner />
  if (person.isError) {
    if (person.error instanceof ApiError && person.error.status === 404)
      return (
        <EmptyState
          title="This person is not here any more"
          action={
            <Link to="/people" className={buttonClasses('secondary', 'md')}>
              All people
            </Link>
          }
        >
          They may have been merged into someone else or deleted.
        </EmptyState>
      )
    return <ErrorState error={person.error} onRetry={() => void person.refetch()} />
  }
  if (!p) return <PageSpinner />
  const name = personName(p)
  const pet = p.kind === 'pet'
  const tab: Tab = pet ? (raw === 'animals' ? raw : 'items') : raw === 'faces' || raw === 'guesses' ? raw : 'items'
  const tabs: { value: Tab; label: string; count?: number }[] = pet
    ? [
        { value: 'items', label: 'Items', count: p.item_count },
        { value: 'animals', label: 'Crops', count: p.animal_count },
      ]
    : [
        { value: 'items', label: 'Items', count: p.item_count },
        { value: 'faces', label: 'Faces', count: p.face_count },
        { value: 'guesses', label: 'Guesses', count: p.guess_count },
      ]

  return (
    <div className="mx-auto w-full max-w-7xl px-4 pb-8">
      <Link to="/people" className="mt-4 inline-flex h-10 items-center gap-1.5 text-sm font-medium text-ink-soft hover:text-ink">
        <ArrowLeft className="size-4" aria-hidden /> {pet ? 'People and pets' : 'All people'}
      </Link>
      <div className="mt-2 flex flex-col gap-4 sm:flex-row sm:items-center">
        <Avatar src={p.cover_url} name={p.name} size={96} />
        <div className="min-w-0 flex-1">
          <NameEditor key={p.id} id={p.id} name={p.name} kind={p.kind} species={p.species} />
          {pet ? (
            <p className="mt-1 flex flex-wrap items-center gap-x-1.5 text-sm text-ink-soft">
              <PawPrint className="size-4 shrink-0" aria-hidden />
              {speciesLabel(p.species)}, {formatCount(p.item_count, 'item')}, {formatCount(p.animal_count, 'crop')}
            </p>
          ) : (
            <p className="mt-1 text-sm text-ink-soft">
              {formatCount(p.item_count, 'item')}, {formatCount(p.face_count, 'face')}
              {(p.confirmed_faces ?? 0) > 0 && `, ${p.confirmed_faces} confirmed by you`}
              {p.guess_count > 0 && `, ${formatCount(p.guess_count, 'guess', 'guesses')} to review`}
            </p>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={() => setMergeOpen(true)}>
            <GitMerge className="size-4" aria-hidden />
            Merge into
          </Button>
          <Button variant="danger" onClick={() => setDeleteOpen(true)}>
            <Trash2 className="size-4" aria-hidden />
            Delete
          </Button>
        </div>
      </div>

      <div role="tablist" aria-label="Person sections" className="-mx-4 mb-4 mt-6 flex gap-1 overflow-x-auto border-b border-line px-4 sm:mx-0 sm:px-0">
        {tabs.map((t) => (
          <button
            key={t.value}
            type="button"
            role="tab"
            aria-selected={tab === t.value}
            onClick={() => setTab(t.value)}
            className={clsx(
              '-mb-px inline-flex h-11 shrink-0 items-center gap-2 border-b-2 px-3 text-sm font-semibold transition-colors',
              tab === t.value ? 'border-accent text-ink' : 'border-transparent text-ink-soft hover:text-ink',
            )}
          >
            {t.label}
            {!!t.count && <span className="rounded-full bg-surface-2 px-2 py-0.5 text-xs tabular-nums">{t.count.toLocaleString()}</span>}
          </button>
        ))}
      </div>

      <div role="tabpanel">
        {tab === 'items' && <ItemsTab id={id} />}
        {tab === 'faces' && <FacesTab id={id} name={name} coverId={p.cover_face_id} />}
        {tab === 'animals' && <AnimalsTab id={id} name={name} species={p.species} />}
        {tab === 'guesses' && <GuessesTab id={id} />}
      </div>

      <PersonChoiceModal
        open={mergeOpen}
        title={`Merge ${name} into`}
        description={
          pet
            ? `Pick the pet this really is. Only pets of the same species (${speciesLabel(p.species).toLowerCase()}) are listed. Every crop and tag moves to them and this entry goes away.`
            : 'Pick the person this really is. Every face, body and tag moves to them and this entry goes away.'
        }
        allowCreate={false}
        excludeIds={[id]}
        kind={pet ? 'pet' : undefined}
        species={p.species ?? undefined}
        speciesOnly={pet}
        onPick={(c) => {
          if ('personId' in c) {
            setMergeOpen(false)
            setMergeTarget(c.personId)
          }
        }}
        onClose={() => setMergeOpen(false)}
      />
      <Confirm
        open={mergeTarget !== null}
        title={`Merge into ${targetQ.data ? personName(targetQ.data) : 'this person'}?`}
        body={`${name} and everything tagged as ${pet ? 'it' : 'them'} becomes ${targetQ.data ? personName(targetQ.data) : pet ? 'the other pet' : 'the other person'}.`}
        confirmLabel="Merge"
        loading={merge.isPending}
        onConfirm={() => mergeTarget !== null && merge.mutate(mergeTarget)}
        onClose={() => !merge.isPending && setMergeTarget(null)}
      />
      <Confirm
        open={deleteOpen}
        title={`Delete ${name}?`}
        body={
          pet
            ? 'The pet goes away. Its boxes stay in the photos, unassigned, and its tags are removed.'
            : 'The person goes away. Their faces and bodies stay in the photos, unassigned, and their tags are removed.'
        }
        confirmLabel="Delete"
        danger
        loading={remove.isPending}
        onConfirm={() => remove.mutate()}
        onClose={() => !remove.isPending && setDeleteOpen(false)}
      />
    </div>
  )
}

export default function PersonPage() {
  const { id } = useParams()
  const n = Number(id)
  if (!Number.isInteger(n) || n <= 0)
    return (
      <EmptyState title="No such person" action={<Link to="/people" className={buttonClasses('secondary', 'md')}>All people</Link>} />
    )
  return <PersonView key={n} id={n} />
}
