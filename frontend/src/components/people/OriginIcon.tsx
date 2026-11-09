// Beside a person's or pet's name (section 10): a person icon when the owner tagged them or drew the box, a bot
// when the system did. The words show as a tooltip and, unless the control around it already says them, to
// screen readers.
import clsx from 'clsx'
import { Bot, UserRound } from 'lucide-react'
import { ORIGIN_LABEL, isMine, type Origin } from '../../lib/origin'

const SIZE = { xs: 'size-3', sm: 'size-3.5', md: 'size-4' } as const

export default function OriginIcon({
  origin,
  label,
  size = 'sm',
  decorative = false,
  className,
}: {
  origin: Origin
  /** Other words than the standard ones, such as "Tagged by you on 2 of 3 items". */
  label?: string
  size?: keyof typeof SIZE
  /** Inside a control whose own label already says it: hidden from screen readers, the tooltip stays. */
  decorative?: boolean
  className?: string
}) {
  const mine = isMine(origin)
  const Icon = mine ? UserRound : Bot
  const text = label ?? ORIGIN_LABEL[origin]
  return (
    <span
      title={text}
      {...(decorative ? { 'aria-hidden': true } : { role: 'img', 'aria-label': text })}
      data-origin={mine ? 'you' : 'auto'}
      className={clsx('inline-flex shrink-0 items-center', className)}
    >
      <Icon className={SIZE[size]} strokeWidth={2.25} aria-hidden />
    </span>
  )
}
