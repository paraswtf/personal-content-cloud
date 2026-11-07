// A person or pet tile: round face (or animal crop), name ("Unnamed" when none), item count and a badge for open guesses.
// Links to the person page, or acts as a selectable button when onSelect is given.
import { Link } from 'react-router'
import clsx from 'clsx'
import { Check, PawPrint } from 'lucide-react'
import type { PersonOut } from '../../lib/types'
import { formatCount, personName, speciesLabel } from '../../lib/format'
import { Avatar } from '../ui'

interface Props {
  person: PersonOut
  size?: 'md' | 'sm'
  to?: string
  selected?: boolean
  onSelect?: () => void
}

export default function PersonCard({ person, size = 'md', to, selected = false, onSelect }: Props) {
  const md = size === 'md'
  const pet = person.kind === 'pet'
  const body = (
    <>
      <span className="relative">
        <Avatar
          src={person.cover_url}
          name={person.name ?? '?'}
          size={md ? 76 : 64}
          className={clsx('transition-shadow', selected ? 'ring-3 ring-accent' : 'group-hover:ring-2 group-hover:ring-accent/40')}
        />
        {onSelect && (
          <span
            aria-hidden
            className={clsx(
              'absolute -right-0.5 -top-0.5 flex size-6 items-center justify-center rounded-full border-2 border-surface',
              selected ? 'bg-accent text-white' : 'bg-surface text-transparent ring-1 ring-line',
            )}
          >
            <Check className="size-3.5" strokeWidth={3} />
          </span>
        )}
        {person.guess_count > 0 && (
          <span
            className="absolute -bottom-1 left-1/2 inline-flex -translate-x-1/2 items-center whitespace-nowrap rounded-full bg-warn px-2 py-0.5 text-[10px] font-semibold text-white"
            title={`${formatCount(person.guess_count, 'guess', 'guesses')} to review`}
          >
            {person.guess_count} {person.guess_count === 1 ? 'guess' : 'guesses'}
          </span>
        )}
      </span>
      <span className={clsx('flex min-w-0 max-w-full flex-col items-center', md ? 'mt-3 gap-0.5' : 'mt-2')}>
        <span className={clsx('max-w-full truncate font-semibold', md ? 'text-sm' : 'text-xs', person.name ? 'text-ink' : 'text-ink-soft')}>
          {personName(person)}
        </span>
        <span className={clsx('inline-flex items-center gap-1 text-ink-faint', md ? 'text-xs' : 'text-[11px]')}>
          {pet && <PawPrint className="size-3 shrink-0" aria-hidden />}
          {pet ? `${speciesLabel(person.species)}, ${formatCount(person.item_count, 'item')}` : formatCount(person.item_count, 'item')}
        </span>
      </span>
    </>
  )

  const cls = clsx(
    'group flex min-w-0 flex-col items-center rounded-2xl text-center outline-none transition-colors',
    'focus-visible:ring-2 focus-visible:ring-accent',
    md ? 'border border-line bg-surface p-2 hover:bg-surface-2 sm:p-4' : 'p-2 hover:bg-surface-2',
    selected && 'bg-accent-soft hover:bg-accent-soft',
  )

  if (onSelect) {
    return (
      <button type="button" onClick={onSelect} aria-pressed={selected} className={cls}>
        {body}
      </button>
    )
  }
  return (
    <Link to={to ?? `/people/${person.id}`} className={cls}>
      {body}
    </Link>
  )
}
