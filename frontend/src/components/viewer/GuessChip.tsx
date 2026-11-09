// "Alice?" with confirm and reject buttons: a person guess waiting for an answer, with the bot of a
// suggestion made by the system (section 10). In the details the name is a button that shows the box (section 14).
import clsx from 'clsx'
import { Check, Loader2, X } from 'lucide-react'
import type { GuessOut } from '../../lib/types'
import { formatPercent, personName } from '../../lib/format'
import OriginIcon from '../people/OriginIcon'
import { evidenceLine } from '../people/evidence'

export default function GuessChip({
  guess,
  busy,
  onConfirm,
  onReject,
  className,
  compact = false,
  onShow,
  showLabel,
}: {
  guess: GuessOut
  busy: boolean
  onConfirm: () => void
  onReject: () => void
  className?: string
  /** Inside a picture: smaller, buttons keep a 40 px hit area through an invisible margin. */
  compact?: boolean
  /** Show the guessed box on the stage: the name becomes a button labelled `showLabel`. */
  onShow?: () => void
  showLabel?: string
}) {
  const name = personName(guess.person)
  const why = evidenceLine(guess)
  const btn = clsx(
    'relative flex shrink-0 items-center justify-center rounded-full transition-colors disabled:opacity-50',
    compact ? 'size-7 after:absolute after:-inset-1.5 after:content-[""]' : 'size-10',
  )
  return (
    <span
      className={clsx(
        'pointer-events-auto inline-flex max-w-full items-center gap-0.5 rounded-full bg-amber-300 pl-2.5 text-black shadow',
        compact ? 'h-8 text-xs' : 'h-10 text-sm',
        className,
      )}
      title={`Maybe ${name}, ${formatPercent(guess.score)} sure${why ? `: ${why}` : ''}`}
    >
      {onShow ? (
        <button
          type="button"
          onClick={onShow}
          aria-label={showLabel ?? `Show ${name}`}
          className="-ml-1 flex h-full min-w-0 items-center gap-1 rounded-full pl-1 pr-1.5 hover:bg-black/10"
        >
          <OriginIcon origin="guess" size={compact ? 'xs' : 'sm'} decorative className="mr-0.5" />
          <span className="min-w-0 truncate font-semibold">{name}?</span>
        </button>
      ) : (
        <>
          <OriginIcon origin="guess" size={compact ? 'xs' : 'sm'} className="mr-0.5" />
          <span className="min-w-0 truncate font-semibold">{name}?</span>
        </>
      )}
      {busy ? (
        <span className={btn}>
          <Loader2 className="size-4 animate-spin" aria-hidden />
        </span>
      ) : (
        <>
          <button type="button" onClick={onConfirm} className={clsx(btn, 'hover:bg-black/10')} aria-label={`Yes, this is ${name}`}>
            <Check className="size-4" aria-hidden />
          </button>
          <button type="button" onClick={onReject} className={clsx(btn, 'hover:bg-black/10')} aria-label={`No, this is not ${name}`}>
            <X className="size-4" aria-hidden />
          </button>
        </>
      )}
    </span>
  )
}
