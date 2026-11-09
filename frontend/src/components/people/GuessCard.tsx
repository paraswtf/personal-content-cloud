// One person guess for review: the face or body crop, the item it is in (for a box in a video, the frame at
// the box's time with the box outlined, not the poster), the guessed person with what the guess rests on and the
// evidence, confirm, reject, and "Someone else" (a name field: the guess is learned as wrong and the box goes to the
// person picked, in one step). A clothes guess shows its score ("By clothes, 85% sure") and its line ("25 min apart,
// same scene 0.86, same clothes 0.91"). Tattoos, nails and mehndi (section 34) show their strongest reason with the
// two crops side by side, and the other signs folded under "N more signs"; a guess from traits alone says "strong
// match", never a percentage.
import { useState } from 'react'
import { Link } from 'react-router'
import { Check, Hand, PenTool, Shirt, UserRoundPen, X } from 'lucide-react'
import type { GuessItem } from '../../lib/types'
import { frameUrl } from '../../lib/api'
import { formatDuration, personName } from '../../lib/format'
import { Avatar, Button } from '../ui'
import OriginIcon from './OriginIcon'
import PersonPicker, { type PersonChoice } from './PersonPicker'
import TraitCompare from './TraitCompare'
import { againstReasons, clothesLine, guessIcons, guessLabel, guessSure, traitReasons } from './evidence'

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
  const reasons = traitReasons(g.guess)
  const lead = reasons[0]
  const clothes = clothesLine(g.guess)
  const more = [...reasons.slice(1).map((r) => r.text), ...againstReasons(g.guess)]
  const icons = guessIcons(g.kind, g.guess)
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
            {icons.includes('clothes') && <Shirt className="size-3.5 shrink-0" aria-hidden />}
            {icons.includes('tattoo') && <PenTool className="size-3.5 shrink-0" aria-hidden />}
            {icons.includes('hand') && <Hand className="size-3.5 shrink-0" aria-hidden />}
            {guessLabel(g.kind, g.guess)}, {guessSure(g.guess)}
            {inVideo ? `, at ${formatDuration(t)}` : ''}
          </p>
          <Link to={`/people/${g.guess.person.id}`} className="mt-1 flex min-w-0 items-center gap-2 hover:underline">
            <Avatar src={g.guess.person.cover_url} name={g.guess.person.name} size={28} />
            <span className="truncate text-sm font-semibold">{name}?</span>
            <OriginIcon origin="guess" className="-ml-1 text-ink-soft" />
          </Link>
        </div>
      </div>
      {clothes && <p className="text-xs text-ink-soft">{clothes}</p>}
      {lead && (
        <div className="flex flex-col gap-2">
          <p className="text-xs text-ink-soft">{lead.text}</p>
          <TraitCompare reason={lead} name={name} onOpenRef={(uid) => onOpenItem(uid)} />
        </div>
      )}
      {more.length > 0 && (
        <details className="text-xs text-ink-soft">
          <summary className="flex min-h-10 cursor-pointer items-center rounded-lg font-medium text-ink hover:text-accent-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent">
            {more.length === 1 ? '1 more sign' : `${more.length} more signs`}
          </summary>
          <ul className="flex list-disc flex-col gap-1 pl-5">
            {more.map((text, i) => (
              <li key={i}>{text}</li>
            ))}
          </ul>
        </details>
      )}
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
