// One person guess for review: the face or body crop, the item it is in (for a box in a video, the frame at
// the box's time with the box outlined, not the poster), the guessed person with the score and the
// evidence ("25 min apart, same scene 0.86, same clothes 0.91"), confirm, reject, and "Someone else" (a name
// field: the guess is learned as wrong and the box goes to the person picked, in one step).
import { useState } from 'react'
import { Link } from 'react-router'
import { Check, Shirt, UserRoundPen, X } from 'lucide-react'
import type { GuessItem } from '../../lib/types'
import { frameUrl } from '../../lib/api'
import { formatDuration, formatPercent, personName } from '../../lib/format'
import { Avatar, Button } from '../ui'
import OriginIcon from './OriginIcon'
import PersonPicker, { type PersonChoice } from './PersonPicker'
import { evidenceLine } from './evidence'

export default function GuessCard({
  g,
  busy,
  onConfirm,
  onReject,
  onCorrect,
  onOpenItem,
}: {
  g: GuessItem
  busy: 'confirm' | 'reject' | 'correct' | null
  onConfirm: () => void
  onReject: () => void
  /** "Someone else": who it really is. */
  onCorrect?: (choice: PersonChoice) => void
  onOpenItem: (uid: string, t?: number | null) => void
}) {
  const [picking, setPicking] = useState(false)
  const name = personName(g.guess.person)
  const why = evidenceLine(g.guess)
  const t = g.box.frame_time_s
  const inVideo = g.item.kind === 'video' && t !== null && t !== undefined
  const scene = inVideo ? frameUrl(g.item.uid, g.item.version, t, 'preview', false, g.item.nsfw) : g.item.urls.thumb
  return (
    <article className="flex flex-col gap-3 rounded-2xl border border-line bg-surface p-3">
      <div className="flex items-center gap-3">
        <img src={g.box.url} alt={g.kind === 'face' ? 'Face' : 'Person'} className="size-20 shrink-0 rounded-xl bg-surface-2 object-cover ring-1 ring-line" />
        <button
          type="button"
          onClick={() => onOpenItem(g.item.uid, inVideo ? t : null)}
          className="relative h-20 shrink-0 overflow-hidden rounded-xl bg-surface-2 ring-1 ring-line hover:ring-ink-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          aria-label={inVideo ? `Open the video at ${formatDuration(t)}` : `Open the ${g.item.kind === 'video' ? 'video' : 'photo'}`}
        >
          {inVideo ? (
            <span className="relative block h-full">
              <img src={scene} alt="" loading="lazy" className="h-full w-auto max-w-36 object-contain" />
              <span
                aria-hidden
                className="pointer-events-none absolute rounded-sm ring-2 ring-accent"
                style={{ left: `${g.box.x * 100}%`, top: `${g.box.y * 100}%`, width: `${g.box.w * 100}%`, height: `${g.box.h * 100}%` }}
              />
            </span>
          ) : (
            <img src={scene} alt="" loading="lazy" className="size-20 object-cover" />
          )}
        </button>
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1 text-xs text-ink-soft">
            {g.kind === 'body' && <Shirt className="size-3.5" aria-hidden />}
            {g.kind === 'face' ? 'Face' : 'By clothes'}, {formatPercent(g.guess.score)} sure{inVideo ? `, at ${formatDuration(t)}` : ''}
          </p>
          <Link to={`/people/${g.guess.person.id}`} className="mt-1 flex min-w-0 items-center gap-2 hover:underline">
            <Avatar src={g.guess.person.cover_url} name={g.guess.person.name} size={28} />
            <span className="truncate text-sm font-semibold">{name}?</span>
            <OriginIcon origin="guess" className="-ml-1 text-ink-soft" />
          </Link>
        </div>
      </div>
      {why && <p className="text-xs text-ink-soft">{why}</p>}
      <div className="flex flex-wrap gap-2">
        <Button size="md" className="flex-1" loading={busy === 'confirm'} disabled={!!busy} onClick={onConfirm}>
          {busy !== 'confirm' && <Check className="size-4" aria-hidden />}
          Yes
        </Button>
        <Button size="md" variant="secondary" className="flex-1" loading={busy === 'reject'} disabled={!!busy} onClick={onReject}>
          {busy !== 'reject' && <X className="size-4" aria-hidden />}
          No
        </Button>
        {onCorrect && (
          <Button
            size="md"
            variant="secondary"
            className="flex-1 whitespace-nowrap"
            loading={busy === 'correct'}
            disabled={!!busy}
            aria-expanded={picking}
            onClick={() => setPicking((v) => !v)}
          >
            {busy !== 'correct' && <UserRoundPen className="size-4" aria-hidden />}
            Someone else
          </Button>
        )}
      </div>
      {onCorrect && picking && (
        <PersonPicker
          autoFocus
          kind="person"
          excludeIds={[g.guess.person.id]}
          placeholder={`Not ${name}? Type who it is`}
          onPick={(choice) => {
            if (busy) return
            setPicking(false)
            onCorrect(choice)
          }}
        />
      )}
    </article>
  )
}
