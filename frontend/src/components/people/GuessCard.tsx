// One person guess for review: the face or body crop, the item it is in, the guessed person with
// the score and the evidence ("same day, same scene 0.86, same clothes 0.91"), confirm and reject.
import { Link } from 'react-router'
import { Check, Shirt, X } from 'lucide-react'
import type { GuessItem } from '../../lib/types'
import { formatPercent, personName } from '../../lib/format'
import { Avatar, Button } from '../ui'
import { evidenceLine } from './evidence'

export default function GuessCard({
  g,
  busy,
  onConfirm,
  onReject,
  onOpenItem,
}: {
  g: GuessItem
  busy: 'confirm' | 'reject' | null
  onConfirm: () => void
  onReject: () => void
  onOpenItem: (uid: string) => void
}) {
  const name = personName(g.guess.person)
  const why = evidenceLine(g.guess)
  return (
    <article className="flex flex-col gap-3 rounded-2xl border border-line bg-surface p-3">
      <div className="flex items-center gap-3">
        <img src={g.box.url} alt={g.kind === 'face' ? 'Face' : 'Person'} className="size-20 shrink-0 rounded-xl bg-surface-2 object-cover ring-1 ring-line" />
        <button
          type="button"
          onClick={() => onOpenItem(g.item.uid)}
          className="size-20 shrink-0 overflow-hidden rounded-xl bg-surface-2 ring-1 ring-line hover:ring-ink-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          aria-label={`Open the ${g.item.kind === 'video' ? 'video' : 'photo'}`}
        >
          <img src={g.item.urls.thumb} alt="" loading="lazy" className="size-full object-cover" />
        </button>
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1 text-xs text-ink-faint">
            {g.kind === 'body' && <Shirt className="size-3.5" aria-hidden />}
            {g.kind === 'face' ? 'Face' : 'By clothes'}, {formatPercent(g.guess.score)} sure
          </p>
          <Link to={`/people/${g.guess.person.id}`} className="mt-1 flex min-w-0 items-center gap-2 hover:underline">
            <Avatar src={g.guess.person.cover_url} name={g.guess.person.name} size={28} />
            <span className="truncate text-sm font-semibold">{name}?</span>
          </Link>
        </div>
      </div>
      {why && <p className="text-xs text-ink-soft">{why}</p>}
      <div className="flex gap-2">
        <Button size="md" className="flex-1" loading={busy === 'confirm'} disabled={!!busy} onClick={onConfirm}>
          {busy !== 'confirm' && <Check className="size-4" aria-hidden />}
          Yes
        </Button>
        <Button size="md" variant="secondary" className="flex-1" loading={busy === 'reject'} disabled={!!busy} onClick={onReject}>
          {busy !== 'reject' && <X className="size-4" aria-hidden />}
          No
        </Button>
      </div>
    </article>
  )
}
