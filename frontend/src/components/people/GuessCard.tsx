// One person guess for review: the face or body crop, the item it is in (for a box in a video, the frame at
// the box's time with the box outlined, not the poster), the guessed person with what the guess rests on and the
// evidence, confirm, reject, and "Someone else" (a name field: the guess is learned as wrong and the box goes to the
// person picked, in one step). A clothes guess shows its score ("By clothes, 85% sure") and its line ("25 min apart,
// same scene 0.86, same clothes 0.91"). Tattoos, nails and mehndi (section 34) show their strongest reason with the
// two crops side by side, and the other signs folded under "N more signs"; a guess from traits alone says "strong
// match", never a percentage. With `keys` (the Review page): Enter on the focused Yes confirms (the button's own
// click), Space on any of the three buttons opens the "Someone else" name box with its field focused, X rejects,
// and Escape in an empty name box closes it and puts focus back on Yes. Nothing fires while typing in the name box.
import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
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
  keys = false,
}: {
  g: GuessItem
  busy: 'confirm' | 'reject' | 'correct' | null
  onConfirm: () => void
  onReject: () => void
  /** "Someone else": who it really is. */
  onCorrect?: (choice: PersonChoice) => void
  onOpenItem: (uid: string, t?: number | null) => void
  /** Review page keyboard: Enter on the focused Yes confirms, Space opens "Someone else", X rejects. */
  keys?: boolean
}) {
  const [picking, setPicking] = useState(false)
  const articleRef = useRef<HTMLElement>(null)
  const pickerRef = useRef<HTMLDivElement>(null)
  const spaceDown = useRef(false) // the Space that opened the name box is still held
  const nameId = useId()
  const focusYes = () => articleRef.current?.querySelector<HTMLButtonElement>('[data-guess-yes]')?.focus({ preventScroll: true })
  useEffect(() => {
    // The name box opens under the buttons and its field takes focus without scrolling: keep the box in view, also
    // as its list of people loads and changes with the typing.
    const el = pickerRef.current
    if (!picking || !el) return
    const show = new ResizeObserver(() => el.scrollIntoView({ block: 'nearest' }))
    show.observe(el)
    return () => show.disconnect()
  }, [picking])
  const onKey = (e: KeyboardEvent<HTMLElement>) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return
    if (e.key === ' ' && e.repeat && spaceDown.current) {
      e.preventDefault() // a held Space does not type into the name box it opened
      return
    }
    if (e.key === 'Escape' && picking) {
      // an empty name box: PersonPicker clears text first and stops the event, so only an empty field gets here
      e.preventDefault()
      setPicking(false)
      focusYes()
      return
    }
    if (typing(e.target)) return
    if (e.key === 'Enter' && e.repeat && onAction(e.target)) {
      e.preventDefault() // a held Enter answers once
      return
    }
    if (e.key === ' ' && onCorrect && onAction(e.target)) {
      e.preventDefault() // not the button's own click: Space opens the name box
      if (e.repeat || busy) return
      spaceDown.current = true
      if (picking) {
        const input = articleRef.current?.querySelector('input')
        input?.focus()
        input?.select()
      } else setPicking(true)
      return
    }
    if ((e.key === 'x' || e.key === 'X') && !e.repeat && !busy) {
      e.preventDefault()
      onReject()
    }
  }
  // Firefox turns the keyup of Space on a focused button into a click: it would confirm
  const onKeyUp = (e: KeyboardEvent<HTMLElement>) => {
    if (e.key !== ' ') return
    spaceDown.current = false
    if (onAction(e.target)) e.preventDefault()
  }
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
    <article
      ref={articleRef}
      onKeyDown={keys ? onKey : undefined}
      onKeyUp={keys ? onKeyUp : undefined}
      className="flex flex-col gap-3 rounded-2xl border border-line bg-surface p-3"
    >
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
            <span id={nameId} className="truncate text-sm font-semibold">{name}?</span>
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
        <Button
          size="md"
          className={
            'flex-1' +
            (keys
              ? ' pointer-fine:focus:outline-none pointer-fine:focus:ring-2 pointer-fine:focus:ring-accent pointer-fine:focus:ring-offset-2 pointer-fine:focus:ring-offset-surface'
              : '')
          }
          loading={busy === 'confirm'}
          disabled={!!busy}
          onClick={onConfirm}
          data-guess-yes=""
          data-guess-action=""
          aria-keyshortcuts={keys ? 'Enter' : undefined}
          aria-describedby={keys ? nameId : undefined}
        >
          {busy !== 'confirm' && <Check className="size-4" aria-hidden />}
          Yes
        </Button>
        <Button
          size="md"
          variant="secondary"
          className="flex-1"
          loading={busy === 'reject'}
          disabled={!!busy}
          onClick={onReject}
          data-guess-action=""
          aria-keyshortcuts={keys ? 'X' : undefined}
        >
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
            data-guess-action=""
            aria-keyshortcuts={keys ? 'Space' : undefined}
          >
            {busy !== 'correct' && <UserRoundPen className="size-4" aria-hidden />}
            Someone else
          </Button>
        )}
      </div>
      {onCorrect && picking && (
        <div ref={pickerRef}>
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
        </div>
      )}
    </article>
  )
}


function typing(t: EventTarget): boolean {
  return t instanceof HTMLElement && (t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName))
}

function onAction(t: EventTarget): boolean {
  return t instanceof Element && !!t.closest('[data-guess-action]')
}
