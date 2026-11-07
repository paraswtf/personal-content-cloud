// Viewer info panel: document card, people and guesses, free-form tags (TagsPanel), category with override, mis-click reasons,
// labels, read text, file details, and the item actions (favourite, vault, reprocess, download, trash).
import { useState, type ReactNode } from 'react'
import { Link } from 'react-router'
import { useMutation } from '@tanstack/react-query'
import clsx from 'clsx'
import {
  AlertCircle,
  Download,
  EyeOff,
  Loader2,
  Lock,
  RefreshCw,
  Star,
  Tag,
  Trash2,
  TriangleAlert,
  Unlock,
  UserPlus,
  X,
} from 'lucide-react'
import { api } from '../../lib/api'
import { CATEGORIES, CATEGORY_LABELS, type Category, type ItemDetail, type PersonRef } from '../../lib/types'
import { formatBytes, formatDate, formatDuration, formatPercent, formatResolution, personName } from '../../lib/format'
import Select from '../Select'
import { errorMessage } from '../admin/adminHooks'
import VirtualDocument from '../documents/VirtualDocument'
import { burstBestUid, reasonKeys, reasonLabel } from '../library/misclick'
import PersonPicker, { choiceBody, type PersonChoice } from '../people/PersonPicker'
import { boxKey, type BoxTarget } from './boxes'
import GuessChip from './GuessChip'
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
  /** Opened from the documents page: the document card comes first. */
  documentFirst?: boolean
}

function Section({ title, children, id }: { title: string; children: ReactNode; id?: string }) {
  return (
    <section id={id} className="border-t border-white/10 px-4 py-4 first:border-t-0">
      <h3 className="mb-2.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-white/50">{title}</h3>
      {children}
    </section>
  )
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  if (value === '' || value === null || value === undefined) return null
  return (
    <div className="flex gap-3 py-1 text-sm">
      <dt className="w-24 shrink-0 text-white/50">{label}</dt>
      <dd className="min-w-0 flex-1 break-words text-white/90">{value}</dd>
    </div>
  )
}

function PersonChip({
  person,
  thumb,
  removable,
  removing,
  onRemove,
}: {
  person: PersonRef
  thumb: string | null | undefined
  removable: boolean
  removing: boolean
  onRemove: () => void
}) {
  const name = personName(person)
  return (
    <li className="flex max-w-full items-center rounded-full bg-white/10 transition-colors hover:bg-white/15">
      <Link to={`/people/${person.id}`} className="flex h-10 min-w-0 items-center gap-2 rounded-full pl-1 pr-3 text-sm font-medium text-white">
        {thumb ? (
          <img src={thumb} alt="" className="size-8 shrink-0 rounded-full object-cover" />
        ) : (
          <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-white/10">
            <Tag className="size-3.5 text-white/70" aria-hidden />
          </span>
        )}
        <span className={person.name ? 'truncate' : 'truncate text-white/70'}>{name}</span>
      </Link>
      {removable && (
        <button
          type="button"
          onClick={onRemove}
          disabled={removing}
          aria-label={`Remove tag ${name}`}
          className="-ml-2 flex size-10 shrink-0 items-center justify-center rounded-full text-white/60 hover:bg-white/10 hover:text-white disabled:opacity-50"
        >
          {removing ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <X className="size-4" aria-hidden />}
        </button>
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
  documentFirst = false,
}: Props) {
  const invalidate = useInvalidateItem(detail.uid)
  const [removingId, setRemovingId] = useState<number | null>(null)
  const isVideo = detail.kind === 'video'
  const isDoc = detail.kind === 'pdf' || detail.kind === 'document'

  const thumbs = new Map<number, string>()
  for (const f of detail.faces) if (f.person && !thumbs.has(f.person.id)) thumbs.set(f.person.id, f.url)
  for (const b of detail.bodies) if (b.person && !thumbs.has(b.person.id)) thumbs.set(b.person.id, b.url)
  const tagIds = new Set(detail.tags.map((t) => t.id))
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
    mutationFn: (choice: PersonChoice) => api.items.addTag(detail.uid, choiceBody(choice)),
    onSuccess: () => {
      invalidate()
      setTagging(false)
      onToast('Tag added')
    },
  })
  const removeTag = useMutation({
    mutationFn: (personId: number) => api.items.removeTag(detail.uid, personId),
    onMutate: (personId) => setRemovingId(personId),
    onSettled: () => setRemovingId(null),
    onSuccess: () => {
      invalidate()
      onToast('Tag removed')
    },
    onError: fail('Could not remove the tag'),
  })
  const update = useMutation({
    mutationFn: (body: Parameters<typeof api.items.update>[1]) => api.items.update(detail.uid, body),
    onSuccess: (_d, body) => {
      invalidate()
      if (body.nsfw !== undefined) {
        onToast(body.nsfw ? 'Moved to the vault' : 'Moved out of the vault')
        onLeave()
      } else if (body.misclick === false) onToast('Kept. It is no longer flagged.')
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
                <PersonChip key={p.id} person={p} thumb={thumbs.get(p.id)} removable={false} removing={false} onRemove={() => {}} />
              ))}
            </ul>
          ) : (
            <p className="flex min-h-10 items-center text-sm text-white/60">No one known in this frame.</p>
          )}
        </Section>
      )}

      {(!isDoc || detail.people.length > 0 || detail.faces.length > 0) && (
        <Section title={isVideo ? 'In this video' : 'People'}>
          {detail.people.length > 0 ? (
            <ul className="flex flex-wrap gap-2" aria-label="People in this item">
              {detail.people.map((p) => (
                <PersonChip
                  key={p.id}
                  person={p}
                  thumb={thumbs.get(p.id) ?? p.cover_url}
                  removable={tagIds.has(p.id)}
                  removing={removingId === p.id}
                  onRemove={() => removeTag.mutate(p.id)}
                />
              ))}
            </ul>
          ) : (
            <p className="text-sm text-white/60">
              {detail.analysis_status === 'pending' ? 'Still looking for people.' : 'No one is tagged here yet.'}
            </p>
          )}

          {guessTargets.length > 0 && (
            <div className="mt-3">
              <p className="mb-1.5 text-xs text-white/55">Guessed from the same day, scene and clothes:</p>
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

          {tagging ? (
            <div className="mt-3 rounded-2xl bg-surface p-2.5">
              <div className="mb-2 flex items-center justify-between gap-2 px-1">
                <p className="text-sm font-semibold text-ink">Tag someone not detected</p>
                <button
                  type="button"
                  onClick={() => setTagging(false)}
                  className="flex size-10 items-center justify-center rounded-full text-ink-soft hover:bg-surface-2"
                  aria-label="Cancel tagging"
                >
                  <X className="size-4" aria-hidden />
                </button>
              </div>
              <PersonPicker autoFocus excludeIds={detail.people.map((p) => p.id)} onPick={(c) => !addTag.isPending && addTag.mutate(c)} />
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
