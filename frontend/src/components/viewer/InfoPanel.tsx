// Viewer info panel: document card, people and guesses, free-form tags (TagsPanel), category with override, mis-click reasons,
// labels, read text, file details, and the item actions (favourite, vault, reprocess, download, trash).
// Tags (section 10): "Tag someone" lists people and pets (a new pet asks for its species) and can mark the item as
// related to someone who is not in it; related tags show under "Related to". Every person's chip has a menu (section 14):
// Change to..., Rename, Not NAME (or Remove from this item for a tag alone) and, for the owner's tags, the relation switch.
import AlbumsSection from '../albums/AlbumsSection'
import LocationRow from '../map/LocationRow'
import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { Link } from 'react-router'
import { useMutation } from '@tanstack/react-query'
import clsx from 'clsx'
import {
  AlertCircle,
  ArrowUpRight,
  Download,
  EyeOff,
  Forward,
  Link2,
  Loader2,
  Lock,
  MoreHorizontal,
  PawPrint,
  Pencil,
  RefreshCw,
  ScanFace,
  SquareUser,
  Star,
  Tag,
  Trash2,
  TriangleAlert,
  Unlock,
  UserPlus,
  UserRoundPen,
  UserX,
  X,
} from 'lucide-react'
import { ApiError, api } from '../../lib/api'
import {
  CATEGORIES,
  CATEGORY_LABELS,
  SPECIES,
  type Category,
  type ItemDetail,
  type PersonRef,
  type Species,
  type TagIn,
  type TagRelation,
} from '../../lib/types'
import {
  formatBytes,
  formatCount,
  formatDate,
  formatDuration,
  formatPercent,
  formatResolution,
  personName,
  speciesLabel,
} from '../../lib/format'
import Select from '../Select'
import { errorMessage } from '../admin/adminHooks'
import VirtualDocument from '../documents/VirtualDocument'
import { forwardReasons, receivedLabel } from '../library/forwards'
import { burstBestUid, reasonKeys, reasonLabel } from '../library/misclick'
import PersonPicker, { choiceBody, type PersonChoice } from '../people/PersonPicker'
import { boxKey, stillBoxes, videoBoxes, type BoxTarget } from './boxes'
import GuessChip from './GuessChip'
import { PersonEditCard, usePersonEdits, type PersonEditMode, type PersonEditTarget } from './PersonEdit'
import TagsPanel from './TagsPanel'
import TextPanel from './TextPanel'
import { darkButton, useInvalidateItem } from './shared'

interface Props {
  detail: ItemDetail
  vault: boolean
  /** Video: ids of the faces on screen at the player's current time; null for stills or without a player. */
  onScreen: Set<number> | null
  onClose: () => void
  onToast: (message: string) => void
  onRequestDelete: () => void
  /** The item left this list (moved into or out of the vault): advance or close. */
  onLeave: () => void
  tagging: boolean
  setTagging: (v: boolean) => void
  guesses: {
    confirm: (t: BoxTarget) => void
    reject: (t: BoxTarget) => void
    pendingKey: string | null
  }
  onEditBox: (target: BoxTarget, anchor: HTMLElement) => void
  /** Draw mode for a pet the detector missed (section 22): the photo or video stage takes a box from the owner. */
  onAddPetBox?: () => void
  /** Draw mode for a person box (section 10): someone in a helmet, from behind or far away, or a face the detector missed. */
  onAddPersonBox?: () => void
  /** Opened from the documents page: the document card comes first. */
  documentFirst?: boolean
  /** A tap on a person's badge (section 14): on a video, jump to them as their chip under the video does (onScreen:
   *  to the appearance in the frame now); on a photo, show their box. Badges of people with no box here only link. */
  onShowPerson?: (personId: number, onScreen?: boolean) => void
  /** The person the stage is showing now (the video's picked track, the photo's highlighted box). */
  activePersonId?: number | null
}

function Section({ title, children, id }: { title: string; children: ReactNode; id?: string }) {
  return (
    <section id={id} className="border-t border-white/10 px-4 py-4 first:border-t-0">
      <h3 className="mb-2.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-white/50">{title}</h3>
      {children}
    </section>
  )
}

export function Row({ label, value }: { label: string; value: ReactNode }) {
  if (value === '' || value === null || value === undefined) return null
  return (
    <div className="flex gap-3 py-1 text-sm">
      <dt className="w-24 shrink-0 text-white/50">{label}</dt>
      <dd className="min-w-0 flex-1 break-words text-white/90">{value}</dd>
    </div>
  )
}

const menuItemClass =
  'flex min-h-11 w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm text-white/90 hover:bg-white/10 focus-visible:bg-white/10 disabled:opacity-50'

/** What the owner can do with someone on this item (section 14): change them to someone else, rename them, take them
 * off the item, and for a tag of the owner's, switch its relation (the same POST with the other one). */
interface ChipMenu {
  busy: boolean
  /** The owner tagged them here: the tag's relation, which the menu switches. */
  relation?: TagRelation
  onRelation: (relation: TagRelation) => void
  onChange: () => void
  onRename: () => void
  /** "Not NAME" when boxes of theirs are here (they stay, remembered as not them), "Remove from this item" for a tag alone. */
  onRemove: () => void
  tagOnly: boolean
}

function PersonChip({
  person,
  thumb,
  menu,
  onShow,
  active = false,
  where,
}: {
  person: PersonRef
  thumb: string | null | undefined
  menu?: ChipMenu
  /** Show them on the stage (section 14); without it the name is only a label. */
  onShow?: () => void
  active?: boolean
  where: 'video' | 'photo'
}) {
  const name = personName(person)
  const pet = person.kind === 'pet'
  const related = person.relation === 'related'
  const [open, setOpen] = useState(false)
  const [alignRight, setAlignRight] = useState(false)
  const root = useRef<HTMLLIElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const list = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: Event) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onDown)
    list.current?.querySelector<HTMLElement>('[role="menuitem"]')?.focus({ preventScroll: true })
    // In the phone sheet the menu may hang below its visible edge: bring all of it into view.
    list.current?.scrollIntoView({ block: 'nearest' })
    return () => document.removeEventListener('pointerdown', onDown)
  }, [open])

  const toggle = () => {
    if (!open) {
      // The menu hangs from the chip's left edge unless that would run past the panel.
      const r = root.current?.getBoundingClientRect()
      const bound = root.current?.closest('aside')?.getBoundingClientRect()
      setAlignRight(!!r && r.left + 232 > (bound?.right ?? window.innerWidth) - 8)
    }
    setOpen((v) => !v)
  }
  const onKeyDown = (e: KeyboardEvent) => {
    if (!open) return
    const items = Array.from(list.current?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [])
    const at = items.indexOf(document.activeElement as HTMLElement)
    if (e.key === 'Escape') {
      e.stopPropagation()
      e.preventDefault()
      setOpen(false)
      trigger.current?.focus()
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      items[(at + 1) % items.length]?.focus()
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      items[(at - 1 + items.length) % items.length]?.focus()
    } else if (e.key === 'Tab') {
      setOpen(false)
    }
  }
  const act = (fn: () => void) => () => {
    setOpen(false)
    fn()
  }

  const mark = related ? (
    <Link2 className="size-3.5 text-white/70" aria-hidden />
  ) : pet ? (
    <PawPrint className="size-3.5 text-white/70" aria-hidden />
  ) : (
    <Tag className="size-3.5 text-white/70" aria-hidden />
  )
  const face = (
    <>
      {thumb ? (
        <img src={thumb} alt="" className="size-8 shrink-0 rounded-full object-cover" />
      ) : (
        <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-white/10">{mark}</span>
      )}
      {thumb && related && <Link2 className="-mr-0.5 size-3.5 shrink-0 text-white/70" aria-label="Related" />}
      {thumb && !related && pet && <PawPrint className="-mr-0.5 size-3.5 shrink-0 text-white/70" aria-label="Pet" />}
      <span className={person.name ? 'truncate' : 'truncate text-white/70'}>{name}</span>
    </>
  )
  const iconButton =
    'flex size-10 shrink-0 items-center justify-center rounded-full text-white/60 hover:bg-white/10 hover:text-white focus-visible:text-white disabled:opacity-50'
  return (
    <li
      ref={root}
      className={clsx(
        'relative flex max-w-full items-center rounded-full transition-colors',
        active ? 'bg-white/25 ring-2 ring-white/80' : 'bg-white/10 hover:bg-white/15',
      )}
      onKeyDown={onKeyDown}
    >
      {onShow ? (
        <button
          type="button"
          onClick={onShow}
          aria-current={active ? 'true' : undefined}
          aria-label={`Show ${name} in the ${where}`}
          title={`Show ${name} in the ${where}`}
          className="flex h-10 min-w-0 items-center gap-2 rounded-full pl-1 pr-2 text-left text-sm font-medium text-white hover:bg-white/10"
        >
          {face}
        </button>
      ) : (
        <span className="flex h-10 min-w-0 items-center gap-2 pl-1 pr-2 text-sm font-medium text-white">{face}</span>
      )}
      <Link
        to={`/people/${person.id}`}
        aria-label={`Open ${name}'s page`}
        title={`Open ${name}'s page`}
        className={clsx(iconButton, '-ml-1')}
      >
        <ArrowUpRight className="size-4" aria-hidden />
      </Link>
      {menu && (
        <button
          ref={trigger}
          type="button"
          onClick={toggle}
          disabled={menu.busy}
          aria-label={`Edit ${name}`}
          title={`Edit ${name}`}
          aria-haspopup="menu"
          aria-expanded={open}
          data-person-menu={person.id}
          className={clsx(iconButton, '-ml-1')}
        >
          {menu.busy ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <Pencil className="size-4" aria-hidden />}
        </button>
      )}
      {menu && open && (
        <div
          ref={list}
          role="menu"
          aria-label={`Edit ${name}`}
          className={clsx(
            'absolute top-full z-30 mt-1 w-56 max-w-[calc(100vw-1rem)] rounded-xl border border-white/10 bg-[#222] p-1.5 shadow-2xl',
            alignRight ? 'right-0' : 'left-0',
          )}
        >
          <button type="button" role="menuitem" className={menuItemClass} onClick={act(menu.onChange)}>
            <UserRoundPen className="size-4 shrink-0" aria-hidden />
            <span className="min-w-0">
              Change to...
              <span className="block text-xs text-white/50">{pet ? 'It is another pet' : 'It is someone else'}</span>
            </span>
          </button>
          <button type="button" role="menuitem" className={menuItemClass} onClick={act(menu.onRename)}>
            <Pencil className="size-4 shrink-0" aria-hidden />
            <span className="min-w-0">
              {person.name ? 'Rename' : 'Name'}
              <span className="block text-xs text-white/50">Everywhere, not only here</span>
            </span>
          </button>
          {menu.relation === 'related' ? (
            <button type="button" role="menuitem" className={menuItemClass} onClick={act(() => menu.onRelation('in'))}>
              <ScanFace className="size-4 shrink-0" aria-hidden />
              <span className="min-w-0">
                Make this 'in'
                <span className="block text-xs text-white/50">{pet ? 'The pet is in the picture' : 'They are in the picture'}</span>
              </span>
            </button>
          ) : menu.relation === 'in' ? (
            <button type="button" role="menuitem" className={menuItemClass} onClick={act(() => menu.onRelation('related'))}>
              <Link2 className="size-4 shrink-0" aria-hidden />
              <span className="min-w-0">
                Make this 'related'
                <span className="block text-xs text-white/50">Not in the picture, it is about them</span>
              </span>
            </button>
          ) : null}
          {menu.tagOnly ? (
            <button type="button" role="menuitem" className={menuItemClass} onClick={act(menu.onRemove)}>
              <X className="size-4 shrink-0" aria-hidden />
              <span className="min-w-0">
                Remove from this item
                <span className="block text-xs text-white/50">Removes the tag</span>
              </span>
            </button>
          ) : (
            <button type="button" role="menuitem" className={menuItemClass} onClick={act(menu.onRemove)}>
              <UserX className="size-4 shrink-0" aria-hidden />
              <span className="min-w-0">
                Not {name}
                <span className="block text-xs text-white/50">
                  {pet ? 'Its boxes here stay, unnamed and remembered as not it' : 'Their faces here stay, unnamed and remembered as not them'}
                </span>
              </span>
            </button>
          )}
        </div>
      )}
    </li>
  )
}

const QUALITY_ROWS: { key: string; label: string; fmt: (v: number) => string }[] = [
  { key: 'blur_var', label: 'Sharpness', fmt: (v) => `${Math.round(v)} (blurry under 40)` },
  { key: 'motion_blur_p', label: 'Motion blur', fmt: (v) => formatPercent(v) },
  { key: 'mean_lum', label: 'Brightness', fmt: (v) => `${Math.round(v)} of 255` },
  { key: 'p95_lum', label: 'Highlights', fmt: (v) => `${Math.round(v)} of 255` },
  { key: 'clipped_frac', label: 'Blown out', fmt: (v) => formatPercent(v) },
  { key: 'accident_p', label: 'Accidental shot', fmt: (v) => formatPercent(v) },
]

function ScoreBars({ scores }: { scores: Record<string, number> }) {
  const rows = Object.entries(scores)
    .filter(([, v]) => typeof v === 'number' && Number.isFinite(v))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
  if (!rows.length) return null
  return (
    <ul className="mt-3 space-y-1.5" aria-label="Category scores">
      {rows.map(([k, v]) => (
        <li key={k} className="flex items-center gap-2 text-xs">
          <span className="w-20 shrink-0 truncate text-white/60">{CATEGORY_LABELS[k as Category] ?? k}</span>
          <span className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-white/10">
            <span className="block h-full rounded-full bg-white/70" style={{ width: `${Math.max(2, Math.min(100, v * 100))}%` }} />
          </span>
          <span className="w-10 shrink-0 text-right tabular-nums text-white/60">{formatPercent(v)}</span>
        </li>
      ))}
    </ul>
  )
}

export default function InfoPanel({
  detail,
  vault,
  onScreen,
  onClose,
  onToast,
  onRequestDelete,
  onLeave,
  tagging,
  setTagging,
  guesses,
  onEditBox,
  onAddPetBox,
  onAddPersonBox,
  documentFirst = false,
  onShowPerson,
  activePersonId = null,
}: Props) {
  const invalidate = useInvalidateItem(detail.uid)
  const [busyId, setBusyId] = useState<number | null>(null)
  const isVideo = detail.kind === 'video'
  const isDoc = detail.kind === 'pdf' || detail.kind === 'document'
  // section 22: a pet box the owner draws, and a second look by the detector (photos and videos);
  // section 10: a person box the owner draws
  const boxable = detail.kind === 'image' || isVideo
  const canAddBoxes = boxable && !!onAddPetBox
  const canAddPersonBox = boxable && !!onAddPersonBox
  // section 10: the tag form's switch, and the new pet whose species is still to pick
  const [related, setRelated] = useState(false)
  const [newPet, setNewPet] = useState<string | null>(null)
  // section 14: the person whose "Change to..." or "Rename" card is open under the chips
  const [personEdit, setPersonEdit] = useState<PersonEditTarget | null>(null)
  useEffect(() => {
    if (!tagging) {
      setRelated(false)
      setNewPet(null)
    } else setPersonEdit(null)
  }, [tagging])
  useEffect(() => {
    setPersonEdit(null)
  }, [detail.uid])
  const edits = usePersonEdits(detail.uid, onToast, () => setPersonEdit(null))
  const [moreBoxes, setMoreBoxes] = useState(false)
  const findPets = useMutation({
    mutationFn: () => api.items.detectAnimals(detail.uid),
    onSuccess: (r) => {
      invalidate()
      onToast(r.length ? `Added ${formatCount(r.length, 'pet box', 'pet boxes')}` : 'No pet found')
    },
    onError: (e) =>
      onToast(e instanceof ApiError && e.status === 409 ? 'The models are not loaded' : errorMessage(e, 'Could not look for pets')),
  })

  const thumbs = new Map<number, string>()
  for (const f of detail.faces) if (f.person && !thumbs.has(f.person.id)) thumbs.set(f.person.id, f.url)
  for (const b of detail.bodies) if (b.person && !thumbs.has(b.person.id)) thumbs.set(b.person.id, b.url)
  for (const a of detail.animals) if (a.person && !thumbs.has(a.person.id)) thumbs.set(a.person.id, a.url)
  const unnamedAnimals = detail.animals.filter((a) => !a.person).length
  const tagIds = new Set(detail.tags.map((t) => t.id))
  const inPeople = detail.people.filter((p) => p.relation !== 'related')
  // The people with a box here (a face, a person box or a pet box): their badge can show them (section 14), on a
  // video once it can play.
  const canShow = !!onShowPerson && (!isVideo || !!detail.urls.video)
  const shown = new Set<number>(
    (isVideo ? videoBoxes(detail) : boxable ? stillBoxes(detail) : [])
      .map((t) => t.box.person?.id)
      .filter((id): id is number => typeof id === 'number'),
  )
  const relatedPeople = detail.people.filter((p) => p.relation === 'related')
  const unmatched = detail.faces.filter((f) => !f.person && !f.guess).length
  const guessTargets: BoxTarget[] = [
    ...detail.faces.filter((f) => !f.person && f.guess).map((box) => ({ kind: 'face' as const, box })),
    ...detail.bodies.filter((b) => !b.person && b.guess).map((box) => ({ kind: 'body' as const, box })),
  ]

  const inFrame: PersonRef[] = []
  if (isVideo && onScreen) {
    const seen = new Set<number>()
    for (const f of detail.faces) {
      if (!f.person || !onScreen.has(f.id) || seen.has(f.person.id)) continue
      seen.add(f.person.id)
      inFrame.push(f.person)
    }
  }

  const fail = (fallback: string) => (e: unknown) => onToast(errorMessage(e, fallback))

  const addTag = useMutation({
    mutationFn: (body: TagIn) => api.items.addTag(detail.uid, body),
    onSuccess: (_d, body) => {
      invalidate()
      setTagging(false)
      onToast(body.relation === 'related' ? 'Tagged as related' : 'Tag added')
    },
  })
  /** The picker's choice: a new pet first asks for its species, anyone else is tagged right away. */
  const pickTag = (choice: PersonChoice) => {
    if (addTag.isPending) return
    if ('newName' in choice && choice.kind === 'pet') {
      setNewPet(choice.newName)
      return
    }
    addTag.mutate({ ...choiceBody(choice), relation: related ? 'related' : 'in' })
  }
  const setRelation = useMutation({
    mutationFn: (v: { personId: number; relation: TagRelation }) => api.items.addTag(detail.uid, { person_id: v.personId, relation: v.relation }),
    onMutate: (v) => setBusyId(v.personId),
    onSettled: () => setBusyId(null),
    onSuccess: (_d, v) => {
      invalidate()
      onToast(v.relation === 'related' ? 'Marked as related: not in the picture' : 'Marked as in the picture')
    },
    onError: fail('Could not change the tag'),
  })
  const removeTag = useMutation({
    mutationFn: (personId: number) => api.items.removeTag(detail.uid, personId),
    onMutate: (personId) => setBusyId(personId),
    onSettled: () => setBusyId(null),
    onSuccess: () => {
      invalidate()
      onToast('Tag removed')
    },
    onError: fail('Could not remove the tag'),
  })
  // Who has a face, body or pet box here (the rest are here through a tag alone).
  const boxed = new Set<number>()
  for (const f of detail.faces) if (f.person) boxed.add(f.person.id)
  for (const b of detail.bodies) if (b.person) boxed.add(b.person.id)
  for (const a of detail.animals) if (a.person) boxed.add(a.person.id)
  const startEdit = (mode: PersonEditMode, p: PersonRef) => {
    setTagging(false)
    setPersonEdit({ mode, who: p, species: detail.animals.find((a) => a.person?.id === p.id)?.species ?? null })
  }
  const closeEdit = () => {
    const id = personEdit?.who.id
    setPersonEdit(null)
    if (id !== undefined)
      requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-person-menu="${id}"]`)?.focus())
  }
  const personMenu = (p: PersonRef): ChipMenu => {
    const tagged = tagIds.has(p.id) || p.relation === 'related'
    const tagOnly = tagged && !boxed.has(p.id)
    return {
      busy: busyId === p.id || edits.busyId === p.id,
      relation: tagged ? (p.relation ?? 'in') : undefined,
      onRelation: (relation) => setRelation.mutate({ personId: p.id, relation }),
      onChange: () => startEdit('change', p),
      onRename: () => startEdit('rename', p),
      onRemove: () => {
        setPersonEdit(null)
        if (tagOnly) removeTag.mutate(p.id)
        else edits.notThem.mutate(p)
      },
      tagOnly,
    }
  }
  const update = useMutation({
    mutationFn: (body: Parameters<typeof api.items.update>[1]) => api.items.update(detail.uid, body),
    onSuccess: (_d, body) => {
      invalidate()
      if (body.nsfw !== undefined) {
        onToast(body.nsfw ? 'Moved to the vault' : 'Moved out of the vault')
        onLeave()
      } else if (body.misclick === false) onToast('Kept. It is no longer flagged.')
      else if (body.forward === false) onToast('Kept. It stays in the library.')
      else if (body.forward === true) onToast('Set apart as a forward.')
      else if (body.category) onToast(`Category set to ${CATEGORY_LABELS[body.category]}`)
      else if (body.favorite !== undefined) onToast(body.favorite ? 'Added to favourites' : 'Removed from favourites')
    },
    onError: fail('Could not save the change'),
  })
  const reprocess = useMutation({
    mutationFn: (stages: Array<'analyze' | 'text' | 'preview'>) => api.items.reprocess(detail.uid, stages),
    onSuccess: () => {
      invalidate()
      onToast('Processing again. This can take a minute.')
    },
    onError: fail('Could not start processing'),
  })

  const notes: { text: string; busy: boolean }[] = []
  if (detail.analysis_status === 'pending') notes.push({ text: 'Looking for faces, people and categories.', busy: true })
  if (detail.analysis_status === 'error') notes.push({ text: 'Analysis failed for this item. Try "Analyse again".', busy: false })
  if (detail.text_status === 'pending') notes.push({ text: 'Reading the text.', busy: true })
  if (detail.text_status === 'error') notes.push({ text: 'The text could not be read. Try "Read text again".', busy: false })
  if (detail.preview_status === 'pending' && isVideo) notes.push({ text: 'A lighter version of this video is being prepared.', busy: true })
  if (detail.preview_status === 'error' && isVideo)
    notes.push({ text: 'The web version of this video could not be made. The original is still available.', busy: false })

  const reasons = reasonKeys(detail.misclick_reasons)
  const bestUid = burstBestUid(detail.misclick_reasons)
  const forwardLabels = forwardReasons(detail.forward_reasons)
  const receivedTitle = receivedLabel(detail.received_via)
  const quality = detail.quality_scores ?? {}
  const qualityRows = QUALITY_ROWS.filter((r) => typeof quality[r.key] === 'number')
  const categoryOptions = CATEGORIES.filter((c) => c !== 'nsfw').map((c) => ({ value: c, label: CATEGORY_LABELS[c] }))
  const kindTitle = isVideo ? 'Video details' : isDoc ? 'Document details' : detail.kind === 'image' ? 'Photo details' : 'File details'

  const documentSection = detail.document && (
    <Section title="Document" id="viewer-document">
      <VirtualDocument
        uid={detail.uid}
        doc={detail.document}
        showPhoto
        notify={{ success: onToast, error: onToast }}
      />
    </Section>
  )

  return (
    <div className="text-white">
      <div className="sticky top-0 z-10 flex items-center justify-between gap-2 border-b border-white/10 bg-[#1b1b1b] px-4 py-2 lg:bg-[#171717]">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold">{kindTitle}</h2>
          <p className="truncate text-xs text-white/55">
            {detail.taken_at ? `Taken ${formatDate(detail.taken_at, true)}` : `Added ${formatDate(detail.created_at)}`}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="flex size-10 shrink-0 items-center justify-center rounded-full text-white/70 hover:bg-white/10 hover:text-white"
          aria-label="Close details"
        >
          <X className="size-5" aria-hidden />
        </button>
      </div>

      {detail.nsfw && (
        <div className="mx-4 mt-4 flex items-center gap-2 rounded-xl bg-white/10 px-3 py-2 text-sm text-white/85">
          <EyeOff className="size-4 shrink-0" aria-hidden />
          Private. Only shown in the vault.
        </div>
      )}

      {detail.misclick && (
        <Section title="Possible mis-click">
          <ul className="flex flex-wrap gap-1.5" aria-label="Reasons">
            {reasons.map((r) => (
              <li key={r} className="inline-flex items-center gap-1 rounded-full bg-amber-400/15 px-2.5 py-1 text-xs font-semibold text-amber-200">
                <TriangleAlert className="size-3" aria-hidden />
                {reasonLabel(r)}
              </li>
            ))}
          </ul>
          {bestUid && bestUid !== detail.uid && (
            <p className="mt-2 text-sm text-white/65">
              A sharper shot of this burst is{' '}
              <Link to={`/m/${bestUid}${vault || detail.nsfw ? '?vault=1' : ''}`} className="font-semibold text-white underline underline-offset-2">
                here
              </Link>
              .
            </p>
          )}
          <div className="mt-3 flex flex-wrap gap-2">
            <button type="button" className={darkButton('primary')} disabled={update.isPending} onClick={() => update.mutate({ misclick: false })}>
              Keep
            </button>
            <button type="button" className={darkButton('danger')} onClick={onRequestDelete}>
              <Trash2 className="size-4" aria-hidden />
              Move to trash
            </button>
          </div>
        </Section>
      )}

      {(receivedTitle || detail.forward) && (
        <Section title={receivedTitle ?? 'Forward'}>
          {detail.forward ? (
            <>
              <p className="text-sm text-white/75">Set apart as a forward. It is not in the library unless forwards are included.</p>
              {forwardLabels.length > 0 && (
                <ul className="mt-2 flex flex-wrap gap-1.5" aria-label="Reasons">
                  {forwardLabels.map((r) => (
                    <li key={r} className="inline-flex items-center gap-1 rounded-full bg-sky-400/15 px-2.5 py-1 text-xs font-semibold text-sky-200">
                      <Forward className="size-3" aria-hidden />
                      {r}
                    </li>
                  ))}
                </ul>
              )}
              <div className="mt-3 flex flex-wrap gap-2">
                <button type="button" className={darkButton('primary')} disabled={update.isPending} onClick={() => update.mutate({ forward: false })}>
                  Not a forward
                </button>
                <button type="button" className={darkButton('danger')} onClick={onRequestDelete}>
                  <Trash2 className="size-4" aria-hidden />
                  Move to trash
                </button>
              </div>
            </>
          ) : (
            <>
              <p className="text-sm text-white/75">
                Kept in the library as a photo{detail.forward_source === 'user' ? ', as you decided' : ''}.
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                <button type="button" className={darkButton()} disabled={update.isPending} onClick={() => update.mutate({ forward: true })}>
                  <Forward className="size-4" aria-hidden />
                  Mark as forward
                </button>
              </div>
            </>
          )}
        </Section>
      )}

      {documentFirst && documentSection}

      {notes.length > 0 && (
        <Section title="Processing">
          <ul className="flex flex-col gap-2">
            {notes.map((n) => (
              <li key={n.text} className="flex gap-2 text-sm text-white/75">
                {n.busy ? (
                  <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin text-white/50" aria-hidden />
                ) : (
                  <AlertCircle className="mt-0.5 size-4 shrink-0 text-amber-300" aria-hidden />
                )}
                {n.text}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {isVideo && onScreen && (
        <Section title="In the frame">
          {inFrame.length > 0 ? (
            <ul className="flex flex-wrap gap-2" aria-label="People in the frame">
              {inFrame.map((p) => (
                <PersonChip
                  key={p.id}
                  person={p}
                  thumb={thumbs.get(p.id)}
                  where="video"
                  active={activePersonId === p.id}
                  onShow={canShow ? () => onShowPerson?.(p.id, true) : undefined}
                />
              ))}
            </ul>
          ) : (
            <p className="flex min-h-10 items-center text-sm text-white/60">No one known in this frame.</p>
          )}
        </Section>
      )}

      {(!isDoc || detail.people.length > 0 || detail.faces.length > 0) && (
        <Section title={isVideo ? 'In this video' : detail.people.some((p) => p.kind === 'pet') || detail.animals.length > 0 ? 'People and pets' : 'People'}>
          {inPeople.length > 0 ? (
            <ul className="flex flex-wrap gap-2" aria-label="People in this item">
              {inPeople.map((p) => (
                <PersonChip
                  key={p.id}
                  person={p}
                  thumb={thumbs.get(p.id) ?? p.cover_url}
                  menu={personMenu(p)}
                  where={isVideo ? 'video' : 'photo'}
                  active={activePersonId === p.id}
                  onShow={canShow && shown.has(p.id) ? () => onShowPerson?.(p.id) : undefined}
                />
              ))}
            </ul>
          ) : (
            <p className="text-sm text-white/60">
              {detail.analysis_status === 'pending'
                ? 'Still looking for people.'
                : relatedPeople.length > 0
                  ? isVideo
                    ? 'No one is in this video.'
                    : 'No one is in this picture.'
                  : 'No one is tagged here yet.'}
            </p>
          )}

          {relatedPeople.length > 0 && (
            <div className="mt-3">
              <p className="mb-1.5 flex items-center gap-1.5 text-xs text-white/55">
                <Link2 className="size-3.5" aria-hidden />
                Related to
              </p>
              <ul className="flex flex-wrap gap-2" aria-label="Related to">
                {relatedPeople.map((p) => (
                  <PersonChip key={p.id} person={p} thumb={p.cover_url} menu={personMenu(p)} where={isVideo ? 'video' : 'photo'} />
                ))}
              </ul>
            </div>
          )}

          {personEdit && (
            <PersonEditCard
              key={`${personEdit.mode}-${personEdit.who.id}`}
              edit={personEdit}
              busy={edits.busy}
              video={isVideo}
              onPick={(choice) => edits.change.mutate({ who: personEdit.who, choice })}
              onRename={(name) => edits.rename.mutate({ who: personEdit.who, name })}
              onCancel={closeEdit}
            />
          )}

          {guessTargets.length > 0 && (
            <div className="mt-3">
              <p className="mb-1.5 text-xs text-white/55">Guessed from the same hour, scene and clothes:</p>
              <ul className="flex flex-col gap-2">
                {guessTargets.map((t) => (
                  <li key={boxKey(t)} className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={(e) => onEditBox(t, e.currentTarget)}
                      className="size-10 shrink-0 overflow-hidden rounded-full ring-1 ring-white/20"
                      aria-label={`Edit this ${t.kind === 'face' ? 'face' : 'person'}`}
                    >
                      <img src={t.box.url} alt="" className="size-full object-cover" />
                    </button>
                    <GuessChip
                      guess={t.box.guess!}
                      busy={guesses.pendingKey === boxKey(t)}
                      onConfirm={() => guesses.confirm(t)}
                      onReject={() => guesses.reject(t)}
                    />
                  </li>
                ))}
              </ul>
            </div>
          )}

          {unmatched > 0 && (
            <p className="mt-2.5 text-xs text-white/55">
              {unmatched === 1 ? '1 face is not matched to anyone.' : `${unmatched} faces are not matched to anyone.`} Tap a face{' '}
              {isVideo ? 'chip' : 'box'} to tag it.
            </p>
          )}

          {unnamedAnimals > 0 && (
            <p className="mt-2.5 text-xs text-white/55">
              {unnamedAnimals === 1 ? '1 animal is not matched to a pet.' : `${unnamedAnimals} animals are not matched to a pet.`} Tap its{' '}
              {isVideo ? 'chip' : 'box'} to name it.
            </p>
          )}

          {tagging ? (
            <div className="mt-3 rounded-2xl bg-surface p-2.5">
              <div className="mb-2 flex items-center justify-between gap-2 px-1">
                <p className="text-sm font-semibold text-ink">{newPet ? `What is ${newPet}?` : 'Tag a person or pet'}</p>
                <button
                  type="button"
                  onClick={() => setTagging(false)}
                  className="flex size-10 items-center justify-center rounded-full text-ink-soft hover:bg-surface-2"
                  aria-label="Cancel tagging"
                >
                  <X className="size-4" aria-hidden />
                </button>
              </div>
              {newPet ? (
                <div className="px-1">
                  <div className="flex flex-wrap gap-1.5" role="group" aria-label="Species">
                    {SPECIES.map((sp: Species) => (
                      <button
                        key={sp}
                        type="button"
                        disabled={addTag.isPending}
                        onClick={() => addTag.mutate({ new_person_name: newPet, kind: 'pet', species: sp, relation: related ? 'related' : 'in' })}
                        className="rounded-full border border-line px-3 py-1.5 text-sm font-semibold text-ink hover:bg-surface-2 disabled:opacity-50"
                      >
                        {speciesLabel(sp)}
                      </button>
                    ))}
                  </div>
                  <button type="button" disabled={addTag.isPending} onClick={() => setNewPet(null)} className="mt-2 h-10 rounded-full px-3 text-sm font-medium text-ink-soft hover:bg-surface-2">
                    Back
                  </button>
                </div>
              ) : (
                <>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={related}
                    onClick={() => setRelated((v) => !v)}
                    className="mb-2 flex min-h-10 w-full items-center gap-3 rounded-xl px-1 text-left text-sm text-ink hover:bg-surface-2"
                  >
                    <span
                      aria-hidden
                      className={clsx('relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors', related ? 'bg-accent' : 'bg-ink/20')}
                    >
                      <span className={clsx('inline-block size-4 rounded-full bg-white shadow transition-transform', related ? 'translate-x-6' : 'translate-x-1')} />
                    </span>
                    <span className="min-w-0 flex-1">
                      Not in the picture, related to them
                      <span className="block text-xs text-ink-soft">
                        {related ? 'Shown under Related on their page and found by search.' : 'Off: they are in the picture.'}
                      </span>
                    </span>
                  </button>
                  <PersonPicker autoFocus kind="all" excludeIds={detail.people.map((p) => p.id)} onPick={pickTag} />
                </>
              )}
              {addTag.isPending && <p className="mt-2 px-1 text-xs text-ink-soft">Adding tag</p>}
              {addTag.error && (
                <p className="mt-2 px-1 text-sm text-danger" role="alert">
                  {errorMessage(addTag.error, 'Could not add the tag')}
                </p>
              )}
            </div>
          ) : (
            <button type="button" onClick={() => setTagging(true)} className={darkButton('default', 'mt-3')}>
              <UserPlus className="size-4" aria-hidden />
              Tag someone
            </button>
          )}

          {(canAddBoxes || canAddPersonBox) && (
            <div className="mt-3">
              <div className="flex flex-wrap gap-2">
                {canAddPersonBox && (
                  <button type="button" onClick={onAddPersonBox} className={darkButton('default')}>
                    <SquareUser className="size-4" aria-hidden />
                    Add a person box
                  </button>
                )}
                {canAddBoxes && detail.animals.length > 0 && (
                  <button type="button" aria-expanded={moreBoxes} onClick={() => setMoreBoxes((v) => !v)} className={darkButton('default')}>
                    <MoreHorizontal className="size-4" aria-hidden />
                    More
                  </button>
                )}
                {canAddBoxes && (detail.animals.length === 0 || moreBoxes) && (
                  <>
                    <button type="button" onClick={onAddPetBox} className={darkButton('default')}>
                      <PawPrint className="size-4" aria-hidden />
                      Add a pet box
                    </button>
                    <button type="button" onClick={() => findPets.mutate()} disabled={findPets.isPending} className={darkButton('default')}>
                      <RefreshCw className={clsx('size-4', findPets.isPending && 'animate-spin')} aria-hidden />
                      Find pets again
                    </button>
                  </>
                )}
              </div>
              {canAddBoxes && detail.animals.length === 0 && (
                <p className="mt-2 text-xs text-white/55">
                  {isVideo
                    ? 'No pet box in this video. Pause on a frame with the pet and draw a box over it, or let the detector look again more closely.'
                    : 'No pet box here. Draw one over the pet, or let the detector look again more closely.'}
                </p>
              )}
            </div>
          )}
        </Section>
      )}

      <Section title="Tags" id="viewer-tags">
        <TagsPanel key={detail.uid} detail={detail} />
      </Section>

      {!documentFirst && documentSection}

      {!detail.nsfw && (
        <Section title="Category">
          <div className="flex flex-wrap items-center gap-2">
            <Select
              label="Category"
              value={detail.category}
              options={categoryOptions}
              onChange={(v) => v !== detail.category && update.mutate({ category: v as Category })}
             
              disabled={update.isPending}
            />
            <span className="text-xs text-white/55">{detail.category_source === 'user' ? 'Set by you' : 'Chosen automatically'}</span>
          </div>
          {detail.category_scores && <ScoreBars scores={detail.category_scores} />}
        </Section>
      )}

      {detail.labels.length > 0 && (
        <Section title="Labels">
          <ul className="flex flex-wrap gap-1.5">
            {detail.labels.map((l) => (
              <li key={l.label} className="rounded-full bg-white/10 px-2.5 py-1 text-xs text-white/85" title={formatPercent(l.score)}>
                {l.label}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {qualityRows.length > 0 && (
        <Section title="Quality">
          <dl>
            {qualityRows.map((r) => (
              <Row key={r.key} label={r.label} value={r.fmt(quality[r.key] as number)} />
            ))}
          </dl>
        </Section>
      )}
      {quality.skipped !== undefined && quality.skipped !== false && qualityRows.length === 0 && (
        <Section title="Quality">
          <p className="text-sm text-white/60">Quality checks were skipped for this item.</p>
        </Section>
      )}

      {detail.text_status === 'done' && (
        <Section title="Text">
          <TextPanel uid={detail.uid} vault={vault} />
        </Section>
      )}

      <AlbumsSection detail={detail} onToast={onToast} />

      <Section title="Details">
        <dl>
          <Row label="Date taken" value={formatDate(detail.taken_at, true)} />
          <Row label="Added" value={formatDate(detail.created_at, true)} />
          <Row label="File name" value={detail.filename} />
          <Row label="Type" value={detail.mime} />
          <Row label="Dimensions" value={formatResolution(detail.width, detail.height)} />
          <Row label="Size" value={formatBytes(detail.size_bytes)} />
          <Row label="Pages" value={detail.page_count ? String(detail.page_count) : ''} />
          <Row label="Camera" value={detail.camera} />
          <LocationRow detail={detail} />
          {isVideo && <Row label="Length" value={formatDuration(detail.duration_s)} />}
          {isVideo && (
            <Row
              label="Video"
              value={[
                detail.video_codec?.toUpperCase(),
                detail.fps ? `${Math.round(detail.fps * 100) / 100} fps` : '',
                detail.bitrate ? `${(detail.bitrate / 1e6).toFixed(1)} Mbit/s` : '',
                detail.has_audio === false ? 'no sound' : '',
              ]
                .filter(Boolean)
                .join(', ')}
            />
          )}
        </dl>
      </Section>

      <Section title="Actions">
        <div className="flex flex-wrap gap-2">
          <button type="button" className={darkButton()} disabled={update.isPending} onClick={() => update.mutate({ favorite: !detail.favorite })}>
            <Star className={clsx('size-4', detail.favorite && 'fill-amber-300 text-amber-300')} aria-hidden />
            {detail.favorite ? 'Unfavourite' : 'Favourite'}
          </button>
          {detail.urls.download && (
            <a href={detail.urls.download} download className={darkButton()}>
              <Download className="size-4" aria-hidden />
              Download original
            </a>
          )}
          <button type="button" className={darkButton()} disabled={update.isPending} onClick={() => update.mutate({ nsfw: !detail.nsfw })}>
            {detail.nsfw ? <Unlock className="size-4" aria-hidden /> : <Lock className="size-4" aria-hidden />}
            {detail.nsfw ? 'Not private' : 'Mark private'}
          </button>
          <button
            type="button"
            className={darkButton()}
            disabled={reprocess.isPending || detail.analysis_status === 'pending'}
            onClick={() => reprocess.mutate(['analyze'])}
          >
            <RefreshCw className={clsx('size-4', reprocess.isPending && reprocess.variables?.[0] === 'analyze' && 'animate-spin')} aria-hidden />
            Analyse again
          </button>
          {(isDoc || detail.text_status !== 'none') && (
            <button
              type="button"
              className={darkButton()}
              disabled={reprocess.isPending || detail.text_status === 'pending'}
              onClick={() => reprocess.mutate(['text'])}
            >
              <RefreshCw className={clsx('size-4', reprocess.isPending && reprocess.variables?.[0] === 'text' && 'animate-spin')} aria-hidden />
              Read text again
            </button>
          )}
          {isVideo && (
            <button
              type="button"
              className={darkButton()}
              disabled={reprocess.isPending || detail.preview_status === 'pending'}
              onClick={() => reprocess.mutate(['preview'])}
            >
              <RefreshCw className={clsx('size-4', reprocess.isPending && reprocess.variables?.[0] === 'preview' && 'animate-spin')} aria-hidden />
              Remake video copy
            </button>
          )}
          <button type="button" className={darkButton('danger')} onClick={onRequestDelete}>
            <Trash2 className="size-4" aria-hidden />
            Move to trash
          </button>
        </div>
      </Section>
    </div>
  )
}
