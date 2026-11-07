// Live Photos and motion photos (section 20 of docs/ARCHITECTURE.md): the LIVE mark on grid tiles and
// the viewer's LIVE button. A tap or click toggles the clip; on desktop a press-and-hold plays it for as
// long as the mouse button is held. The clip never starts on its own.
import { useEffect, useRef, type PointerEvent } from 'react'
import clsx from 'clsx'

/** Concentric rings, the usual mark of a Live Photo. */
export function LiveIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className={className} aria-hidden>
      <circle cx="12" cy="12" r="3" fill="currentColor" stroke="none" />
      <circle cx="12" cy="12" r="6.5" />
      <circle cx="12" cy="12" r="10" strokeDasharray="1.6 2.6" strokeLinecap="round" />
    </svg>
  )
}

/** How long a mouse press lasts before it counts as press-and-hold. */
const HOLD_MS = 250

interface Props {
  playing: boolean
  onChange: (playing: boolean) => void
  className?: string
}

export default function LiveButton({ playing, onChange, className }: Props) {
  const timer = useRef<number | null>(null)
  const held = useRef(false)
  const skipClick = useRef(false)

  const clearTimer = () => {
    if (timer.current !== null) window.clearTimeout(timer.current)
    timer.current = null
  }
  useEffect(() => clearTimer, [])

  const onPointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    skipClick.current = false
    held.current = false
    // Touch and pen use the tap; a long press there opens the system menu instead.
    if (e.pointerType !== 'mouse' || e.button !== 0 || playing) return
    e.currentTarget.setPointerCapture?.(e.pointerId)
    clearTimer()
    timer.current = window.setTimeout(() => {
      timer.current = null
      held.current = true
      onChange(true)
    }, HOLD_MS)
  }
  const endHold = () => {
    clearTimer()
    if (!held.current) return
    held.current = false
    skipClick.current = true
    onChange(false)
  }

  return (
    <button
      type="button"
      data-no-swipe
      onPointerDown={onPointerDown}
      onPointerUp={endHold}
      onPointerCancel={endHold}
      onLostPointerCapture={endHold}
      onClick={() => {
        if (skipClick.current) {
          skipClick.current = false
          return
        }
        onChange(!playing)
      }}
      onContextMenu={(e) => e.preventDefault()}
      aria-label={playing ? 'Stop the clip' : 'Play the Live Photo clip'}
      title={playing ? 'Stop the clip' : 'Play the clip (hold to play while pressed)'}
      className={clsx(
        'inline-flex h-10 select-none items-center gap-1.5 rounded-full px-3 text-xs font-bold tracking-wider backdrop-blur-sm transition-colors [-webkit-touch-callout:none]',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white',
        playing ? 'bg-white text-[#111]' : 'bg-black/55 text-white hover:bg-black/75',
        className,
      )}
    >
      <LiveIcon className={clsx('size-4', playing && 'motion-safe:animate-pulse')} />
      LIVE
    </button>
  )
}
