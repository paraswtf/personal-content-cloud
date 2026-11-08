// The library's timeline grid (docs/ARCHITECTURE.md section 23). The month skeleton gives every
// month's height from its count alone (a header row plus ceil(count / columns) rows of square tiles,
// the same tiles and column counts as MediaGrid), so the page has its full height at once and the
// window's scrollbar is honest. Only the months that intersect the viewport plus one viewport above
// and below are rendered, and within a month only the rows in that range; months load page by page
// as their rows come into view, and rows not loaded yet are placeholder squares of the final size.
// Month headers stick under the app header while their month is on screen, with a Select month
// toggle; the scrubber jumps to any month, and from md up the grid leaves it room on the right.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import clsx from 'clsx'
import { formatCount } from '../../lib/format'
import { monthLabel } from '../../lib/timeline'
import type { ItemCard } from '../../lib/types'
import { GRID, Tile } from '../MediaGrid'
import Scrubber from './Scrubber'
import type { MonthRequest, TimelineState } from './useTimeline'

interface Props {
  timeline: TimelineState
  vault: boolean
  selectable: boolean
  selected: Set<string>
  onToggleSelect: (uid: string) => void
  /** Select month: add or remove a month's loaded cards. */
  onSelectMany: (uids: string[], select: boolean) => void
  onOpen: (uid: string, index: number) => void
  selectOnHover?: boolean
  empty: ReactNode
}

/** The month header row (px) and the gap under each month. */
export const HEADER_H = 44
const BLOCK_PAD = 12
/** The app header is 3.5rem plus the safe-area inset (Header.tsx); months stick under it. */
const STICKY_TOP = 'calc(3.5rem + env(safe-area-inset-top))'
const STICKY_PX = 56
const RERENDER_STEP = 64
/** The scrubber rail's footprint from the viewport's right edge (right-1 plus w-11, Scrubber.tsx). */
const RAIL_SPACE = 48

interface Block {
  key: string
  count: number
  rows: number
  top: number
  height: number
  /** Index of the month's first card among all cards (for the tiles' index). */
  first: number
  label: string
}

/** MediaGrid's columns per breakpoint: 3 below 640 px, 4 from 640, 5 from 768, 6 from 1024, 7 from 1280. */
export function columnsFor(width: number): number {
  return width < 640 ? 3 : width < 768 ? 4 : width < 1024 ? 5 : width < 1280 ? 6 : 7
}

/** MediaGrid's gap: 2 px below 640 px, 4 px from there. */
export function gapFor(width: number): number {
  return width < 640 ? 2 : 4
}

function layoutFor(buckets: TimelineState['buckets'], cols: number, rowH: number): { blocks: Block[]; height: number } {
  const blocks: Block[] = []
  let top = 0
  let first = 0
  for (const b of buckets) {
    const rows = Math.ceil(b.count / cols)
    const height = HEADER_H + rows * rowH + BLOCK_PAD
    blocks.push({ key: b.key, count: b.count, rows, top, height, first, label: monthLabel(b.key) })
    top += height
    first += b.count
  }
  return { blocks, height: top }
}

export default function TimelineGrid({ timeline, vault, selectable, selected, onToggleSelect, onSelectMany, onOpen, selectOnHover = false, empty }: Props) {
  const rootRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  // how far the scrubber rail would reach into the grid (px); 0 when the grid ends before it
  const [reserve, setReserve] = useState(0)
  const [win, setWin] = useState({ top: 0, bottom: 0 })
  const [md, setMd] = useState(() => typeof window !== 'undefined' && window.matchMedia('(min-width: 768px)').matches)
  const { buckets, cards, pending, complete, request, focus, clearFocus, positionOf } = timeline
  // the grid root exists only once the skeleton is known and not empty; the measuring effects below
  // run again when it appears (the first render is the loading placeholder without the ref)
  const ready = !(timeline.loading && buckets.length === 0) && !(!timeline.loading && timeline.total === 0)

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 768px)')
    const on = () => setMd(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])

  const measure = useCallback(() => {
    const el = rootRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setWidth((prev) => (Math.abs(prev - r.width) < 1 ? prev : r.width))
    const overlap = Math.max(0, Math.ceil(r.right - (document.documentElement.clientWidth - RAIL_SPACE)))
    setReserve((prev) => (prev === overlap ? prev : overlap))
  }, [])

  useLayoutEffect(() => {
    const el = rootRef.current
    if (!el) return
    const ro = new ResizeObserver(() => measure())
    ro.observe(el)
    measure()
    window.addEventListener('resize', measure)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [measure, ready])

  // the visible range in the grid's own coordinates, refreshed on scroll and resize (a frame at most)
  useEffect(() => {
    let raf = 0
    const update = () => {
      raf = 0
      const el = rootRef.current
      if (!el) return
      const r = el.getBoundingClientRect()
      const top = -r.top
      const bottom = -r.top + window.innerHeight
      setWin((prev) => (Math.abs(prev.top - top) < RERENDER_STEP && Math.abs(prev.bottom - bottom) < RERENDER_STEP ? prev : { top, bottom }))
    }
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update)
    }
    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
      if (raf) cancelAnimationFrame(raf)
    }
  }, [ready])

  const viewport = Math.max(win.bottom - win.top, 1)
  // the rail shows once the grid is taller than two screens; from md up the grid then leaves it room
  const fullLayout = useMemo(() => {
    const c = columnsFor(width)
    const g = gapFor(width)
    const t = width > 0 ? (width - (c - 1) * g) / c : 0
    return layoutFor(buckets, c, t + g)
  }, [buckets, width])
  const railShown = fullLayout.height > viewport * 2
  const railReserve = railShown && md ? Math.min(reserve, Math.max(0, width - 200)) : 0
  const inner = Math.max(0, width - railReserve)

  const cols = columnsFor(inner)
  const gap = gapFor(inner)
  const tile = inner > 0 ? (inner - (cols - 1) * gap) / cols : 0
  const rowH = tile + gap
  const layout = useMemo(() => (railReserve > 0 ? layoutFor(buckets, cols, rowH) : fullLayout), [buckets, cols, rowH, railReserve, fullLayout])

  const rangeTop = win.top - viewport
  const rangeBottom = win.bottom + viewport
  const visible = useMemo(
    () => layout.blocks.filter((b) => b.top < rangeBottom && b.top + b.height > rangeTop),
    [layout, rangeTop, rangeBottom],
  )
  const rowsOf = useCallback(
    (b: Block) => {
      const rowStart = Math.max(0, Math.floor((rangeTop - (b.top + HEADER_H)) / rowH))
      const rowEnd = Math.min(b.rows - 1, Math.ceil((rangeBottom - (b.top + HEADER_H)) / rowH))
      return { from: rowStart * cols, to: Math.min(b.count, (rowEnd + 1) * cols), rowStart }
    },
    [rangeTop, rangeBottom, rowH, cols],
  )

  // ask for the months on screen (and around it), up to the last visible card of each
  const needSig = visible.map((b) => `${b.key}:${rowsOf(b).to}`).join(',')
  useEffect(() => {
    if (!needSig) return
    const reqs: MonthRequest[] = needSig.split(',').map((part) => {
      const [key, need] = part.split(':')
      return { key, need: Math.max(1, Number(need) || 1) }
    })
    request(reqs)
  }, [needSig, request])

  const documentTop = useCallback(() => {
    const el = rootRef.current
    return el ? el.getBoundingClientRect().top + window.scrollY : 0
  }, [])

  // the open item's month, resolved by the hook: scroll to the month at once, and to the card's row
  // when the card is known; give up on the row once the month is complete without it
  const focusScrolled = useRef<string | null>(null)
  useEffect(() => {
    if (!focus || width === 0) return
    const block = layout.blocks.find((b) => b.key === focus.key)
    if (!block) {
      clearFocus()
      return
    }
    const pos = positionOf(focus.uid)
    if (pos !== null) {
      // the card's row lands one header height under the sticky app header, so its month header shows
      const row = Math.floor((pos - block.first) / cols)
      window.scrollTo({ top: Math.max(0, documentTop() + block.top + row * rowH - STICKY_PX), behavior: 'auto' })
      focusScrolled.current = null
      clearFocus()
      return
    }
    if (focusScrolled.current !== focus.uid) {
      focusScrolled.current = focus.uid
      window.scrollTo({ top: Math.max(0, documentTop() + block.top - STICKY_PX), behavior: 'auto' })
    }
    if (complete(focus.key)) {
      focusScrolled.current = null
      clearFocus()
    }
  }, [focus, layout, width, cols, rowH, documentTop, clearFocus, positionOf, complete])

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget) return
    if (e.key === 'Home') {
      e.preventDefault()
      window.scrollTo({ top: Math.max(0, documentTop() - STICKY_PX) })
    } else if (e.key === 'End') {
      e.preventDefault()
      window.scrollTo({ top: documentTop() + layout.height })
    }
  }

  if (timeline.loading && buckets.length === 0) {
    return (
      <div className={GRID} aria-busy="true" aria-label="Loading">
        {Array.from({ length: 18 }, (_, i) => (
          <div key={i} className="aspect-square animate-pulse bg-surface-2" />
        ))}
      </div>
    )
  }
  if (!timeline.loading && timeline.total === 0) return <div ref={rootRef}>{empty}</div>

  return (
    <div
      ref={rootRef}
      role="region"
      aria-label="Timeline"
      tabIndex={0}
      onKeyDown={onKeyDown}
      className="relative outline-none focus-visible:ring-2 focus-visible:ring-accent"
      style={{ height: layout.height }}
    >
      {width > 0 &&
        visible.map((b) => {
          const list = cards(b.key)
          const loadedCount = list?.length ?? 0
          const { from, to, rowStart } = rowsOf(b)
          const busy = pending(b.key) && loadedCount < to
          const uids = list ? list.map((c) => c.uid) : []
          const allSelected = uids.length > 0 && uids.every((u) => selected.has(u))
          return (
            <section
              key={b.key}
              aria-label={b.label}
              className="group absolute left-0"
              style={{ top: b.top, height: b.height, right: railReserve }}
            >
              <div
                className="sticky z-20 flex h-11 items-center justify-between gap-2 bg-bg/95 px-4 backdrop-blur sm:px-0"
                style={{ top: STICKY_TOP }}
              >
                <h2 className="min-w-0 truncate text-sm font-semibold">
                  {b.label}
                  <span className="ml-2 font-normal tabular-nums text-ink-soft">{formatCount(b.count, 'item')}</span>
                  {busy && <span className="sr-only">, loading</span>}
                </h2>
                {list && list.length > 0 && (
                  <button
                    type="button"
                    aria-pressed={allSelected}
                    aria-label={`${allSelected ? 'Deselect' : 'Select'} month ${b.label}`}
                    onClick={() => onSelectMany(uids, !allSelected)}
                    className={clsx(
                      'shrink-0 rounded-full px-2.5 py-1 text-xs font-semibold text-accent hover:bg-accent-soft focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
                      !selectable && !allSelected && 'opacity-0 group-hover:opacity-100',
                    )}
                  >
                    {allSelected ? 'Deselect month' : 'Select month'}
                  </button>
                )}
              </div>
              {to > from && (
                <ul
                  role="list"
                  aria-busy={busy || undefined}
                  className="absolute left-0 right-0"
                  style={{
                    top: HEADER_H + rowStart * rowH,
                    display: 'grid',
                    gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
                    gridAutoRows: `${tile}px`,
                    gap,
                  }}
                >
                  {Array.from({ length: to - from }, (_, k) => {
                    const i = from + k
                    const card: ItemCard | undefined = list?.[i]
                    return (
                      <li key={i} className="min-w-0">
                        {card ? (
                          <Tile
                            item={card}
                            index={b.first + i}
                            selectable={selectable}
                            selectOnHover={selectOnHover}
                            isSelected={selected.has(card.uid)}
                            showReasons={false}
                            showSnippets={false}
                            vault={vault}
                            onOpen={onOpen}
                            onToggleSelect={onToggleSelect}
                          />
                        ) : (
                          <div aria-hidden className={clsx('size-full bg-surface-2', busy && 'animate-pulse')} />
                        )}
                      </li>
                    )
                  })}
                </ul>
              )}
            </section>
          )
        })}
      {railShown && <Scrubber blocks={layout.blocks} height={layout.height} documentTop={documentTop} stickyTop={STICKY_PX} />}
    </div>
  )
}
