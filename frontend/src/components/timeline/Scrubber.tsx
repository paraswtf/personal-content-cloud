// The timeline's scrubber (docs/ARCHITECTURE.md section 23): a thin rail on the right edge with the
// years at their share of the grid's height. Tapping or dragging scrolls to the month under the
// pointer and a floating label names it; the years are buttons too, for the keyboard. Always on
// from md up; on phones it appears while scrolling and for two seconds after.
import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import clsx from 'clsx'
import { monthLabel } from '../../lib/timeline'

export interface ScrubberBlock {
  key: string
  top: number
  height: number
}

interface Props {
  blocks: ScrubberBlock[]
  /** The grid's full height in px. */
  height: number
  /** The grid's top edge in document coordinates. */
  documentTop: () => number
  /** Where the top of a month should land (under the sticky header), in px from the viewport top. */
  stickyTop: number
}

const SHOW_MS = 2000
const MIN_LABEL_GAP = 18

export default function Scrubber({ blocks, height, documentTop, stickyTop }: Props) {
  const railRef = useRef<HTMLDivElement>(null)
  const [railHeight, setRailHeight] = useState(0)
  const [desktop, setDesktop] = useState(() => typeof window !== 'undefined' && window.matchMedia('(min-width: 768px)').matches)
  const [scrolling, setScrolling] = useState(false)
  const [fraction, setFraction] = useState(0)
  const [hover, setHover] = useState<{ y: number; label: string } | null>(null)
  const dragging = useRef(false)

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 768px)')
    const on = () => setDesktop(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])

  useEffect(() => {
    const el = railRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => setRailHeight(entries[0]?.contentRect.height ?? 0))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // the current position, and the phone's show-while-scrolling state
  useEffect(() => {
    let raf = 0
    let hide: ReturnType<typeof setTimeout> | null = null
    const update = () => {
      raf = 0
      const f = height > 0 ? (window.scrollY - documentTop() + stickyTop) / height : 0
      setFraction(Math.min(1, Math.max(0, f)))
    }
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update)
      setScrolling(true)
      if (hide) clearTimeout(hide)
      hide = setTimeout(() => setScrolling(false), SHOW_MS)
    }
    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      if (raf) cancelAnimationFrame(raf)
      if (hide) clearTimeout(hide)
    }
  }, [height, documentTop, stickyTop])

  const blockAt = useCallback(
    (f: number) => {
      const y = f * height
      let found = blocks[0]
      for (const b of blocks) {
        if (b.top <= y) found = b
        else break
      }
      return found
    },
    [blocks, height],
  )

  const fractionOf = (clientY: number) => {
    const rect = railRef.current?.getBoundingClientRect()
    if (!rect || rect.height === 0) return 0
    return Math.min(1, Math.max(0, (clientY - rect.top) / rect.height))
  }

  const jumpTo = useCallback(
    (top: number) => {
      window.scrollTo({ top: Math.max(0, documentTop() + top - stickyTop), behavior: 'auto' })
    },
    [documentTop, stickyTop],
  )

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest('button')) return
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    dragging.current = true
    const f = fractionOf(e.clientY)
    jumpTo(f * height)
    const b = blockAt(f)
    if (b) setHover({ y: e.clientY, label: monthLabel(b.key) })
  }
  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const f = fractionOf(e.clientY)
    const b = blockAt(f)
    if (b) setHover({ y: e.clientY, label: monthLabel(b.key) })
    if (dragging.current) jumpTo(f * height)
  }
  const endDrag = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (dragging.current && e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId)
    dragging.current = false
    if (e.type === 'pointerleave' || e.pointerType !== 'mouse') setHover(null)
  }

  // one label per year, at the first month of that year, skipping labels that would overlap
  const marks: { year: string; top: number }[] = []
  let lastPx = -Infinity
  for (const b of blocks) {
    const year = b.key.slice(0, 4)
    if (marks.length && marks[marks.length - 1].year === year) continue
    const px = height > 0 ? (b.top / height) * railHeight : 0
    if (px - lastPx < MIN_LABEL_GAP) continue
    marks.push({ year, top: b.top })
    lastPx = px
  }

  const visible = desktop || scrolling || dragging.current || hover !== null
  return (
    <div
      ref={railRef}
      role="group"
      aria-label="Timeline scrubber"
      data-testid="timeline-scrubber"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onPointerLeave={endDrag}
      className={clsx(
        'fixed right-1 z-30 w-11 select-none touch-none rounded-full border border-line bg-surface/90 shadow-sm backdrop-blur transition-opacity',
        'top-[calc(3.5rem+env(safe-area-inset-top)+0.5rem)] bottom-[calc(var(--tabbar-h)+env(safe-area-inset-bottom)+0.5rem)] md:bottom-2',
        visible ? 'opacity-100' : 'pointer-events-none opacity-0',
      )}
    >
      <span
        aria-hidden
        className="pointer-events-none absolute left-1 right-1 h-0.5 rounded-full bg-accent"
        style={{ top: `calc(${(fraction * 100).toFixed(3)}% - 1px)` }}
      />
      {marks.map((m) => (
        <button
          key={m.year}
          type="button"
          aria-label={`Jump to ${m.year}`}
          onClick={() => jumpTo(m.top)}
          className="absolute left-0 right-0 -translate-y-1/2 px-0.5 text-center text-[11px] font-semibold tabular-nums text-ink-soft hover:text-ink focus-visible:text-accent focus-visible:outline-none"
          style={{ top: `${((m.top / Math.max(height, 1)) * 100).toFixed(3)}%` }}
        >
          {m.year}
        </button>
      ))}
      {hover && (
        <span
          role="status"
          className="pointer-events-none fixed right-14 -translate-y-1/2 rounded-full bg-ink px-3 py-1 text-xs font-semibold text-bg shadow"
          style={{ top: hover.y }}
        >
          {hover.label}
        </span>
      )}
    </div>
  )
}
