// Move or resize one face, person or pet box (section 10), guesses included. Placed over the picture in the same
// box the stage draws boxes in: drag the box to move it, drag a corner to resize it; on the keyboard the arrows
// move it (Shift for small steps), + and - resize it around its middle, Enter saves and Escape cancels. The
// buttons sit under the box, or over it near the bottom, inside the picture (not fixed: the stage's size
// containment would pin a fixed bar to the stage).
import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from 'react'
import clsx from 'clsx'
import { Check, X } from 'lucide-react'

export interface NormBox {
  x: number
  y: number
  w: number
  h: number
}

type Corner = 'nw' | 'ne' | 'sw' | 'se'
type Mode = 'move' | Corner

const STEP = 0.01
const FINE = 0.0025
const GROW = 0.02
/** The smallest side the owner can make, unless the box already was smaller (the server says the same). */
const MIN_SIDE = 0.015

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi)

function moved(mode: Mode, b: NormBox, dx: number, dy: number, minW: number, minH: number): NormBox {
  if (mode === 'move') return { ...b, x: clamp(b.x + dx, 0, 1 - b.w), y: clamp(b.y + dy, 0, 1 - b.h) }
  let x1 = b.x
  let y1 = b.y
  let x2 = b.x + b.w
  let y2 = b.y + b.h
  if (mode === 'nw' || mode === 'sw') x1 = clamp(b.x + dx, 0, x2 - minW)
  else x2 = clamp(x2 + dx, x1 + minW, 1)
  if (mode === 'nw' || mode === 'ne') y1 = clamp(b.y + dy, 0, y2 - minH)
  else y2 = clamp(y2 + dy, y1 + minH, 1)
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 }
}

function grown(b: NormBox, by: number, minW: number, minH: number): NormBox {
  const w = clamp(b.w + by, minW, 1)
  const h = clamp(b.h + by * (b.h / Math.max(b.w, 1e-6)), minH, 1)
  const x = clamp(b.x + (b.w - w) / 2, 0, 1 - w)
  const y = clamp(b.y + (b.h - h) / 2, 0, 1 - h)
  return { x, y, w, h }
}

export default function BoxAdjuster({
  start,
  label,
  saving,
  error,
  onSave,
  onCancel,
}: {
  start: NormBox
  /** Whose box it is, for the label: "Asha's face", "this pet". */
  label: string
  saving: boolean
  error: string | null
  onSave: (box: NormBox) => void
  onCancel: () => void
}) {
  const [box, setBox] = useState<NormBox>(start)
  const rootRef = useRef<HTMLDivElement>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  const barRef = useRef<HTMLDivElement>(null)
  // The picture's and the buttons' sizes, so the buttons stay inside the picture (a box at a side on a phone).
  const [size, setSize] = useState<{ W: number; H: number; bw: number; bh: number } | null>(null)
  useLayoutEffect(() => {
    const measure = () => {
      const r = rootRef.current?.getBoundingClientRect()
      const b = barRef.current?.getBoundingClientRect()
      if (!r || !b) return
      setSize((cur) =>
        cur && cur.W === r.width && cur.H === r.height && cur.bw === b.width && cur.bh === b.height
          ? cur
          : { W: r.width, H: r.height, bw: b.width, bh: b.height },
      )
    }
    measure()
    const ro = new ResizeObserver(measure)
    if (rootRef.current) ro.observe(rootRef.current)
    if (barRef.current) ro.observe(barRef.current)
    return () => ro.disconnect()
  }, [])
  const drag = useRef<{ mode: Mode; px: number; py: number; from: NormBox; cw: number; ch: number } | null>(null)
  const minW = Math.min(MIN_SIDE, start.w)
  const minH = Math.min(MIN_SIDE, start.h)
  const changed = Math.abs(box.x - start.x) + Math.abs(box.y - start.y) + Math.abs(box.w - start.w) + Math.abs(box.h - start.h) > 1e-4

  useEffect(() => {
    boxRef.current?.focus({ preventScroll: true })
  }, [])

  const down = (mode: Mode) => (e: ReactPointerEvent<HTMLElement>) => {
    if (saving || (e.button !== 0 && e.pointerType === 'mouse')) return
    e.preventDefault()
    e.stopPropagation()
    const r = rootRef.current?.getBoundingClientRect()
    if (!r || !r.width || !r.height) return
    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = { mode, px: e.clientX, py: e.clientY, from: box, cw: r.width, ch: r.height }
    boxRef.current?.focus({ preventScroll: true })
  }
  const move = (e: ReactPointerEvent<HTMLElement>) => {
    const d = drag.current
    if (!d) return
    e.preventDefault()
    setBox(moved(d.mode, d.from, (e.clientX - d.px) / d.cw, (e.clientY - d.py) / d.ch, minW, minH))
  }
  const up = () => {
    drag.current = null
  }

  // Keys on the box itself; Escape anywhere inside (the buttons too).
  const onBoxKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget) return
    const step = e.shiftKey ? FINE : STEP
    const by: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }
    let handled = true
    if (by[e.key]) setBox((b) => moved('move', b, by[e.key][0], by[e.key][1], minW, minH))
    else if (e.key === '+' || e.key === '=') setBox((b) => grown(b, GROW, minW, minH))
    else if (e.key === '-' || e.key === '_') setBox((b) => grown(b, -GROW, minW, minH))
    else if (e.key === 'Enter') {
      if (!saving) onSave(box)
    } else handled = false
    if (handled) {
      e.preventDefault()
      e.stopPropagation()
    }
  }
  const onRootKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'Escape') return
    e.preventDefault()
    e.stopPropagation()
    if (!saving) onCancel()
  }

  // The buttons: under the box, else over it, else inside its bottom edge; centred on it but kept in the picture.
  const GAP = 14
  let bar: { left: number; top: number } | null = null
  if (size) {
    const left = clamp((box.x + box.w / 2) * size.W - size.bw / 2, 4, Math.max(4, size.W - size.bw - 4))
    const bottom = (box.y + box.h) * size.H
    const top = box.y * size.H
    const y =
      bottom + GAP + size.bh <= size.H - 4
        ? bottom + GAP
        : top - GAP - size.bh >= 4
          ? top - GAP - size.bh
          : clamp(bottom - size.bh - 8, 4, Math.max(4, size.H - size.bh - 4))
    bar = { left, top: y }
  }
  const pct = (v: number) => `${v * 100}%`
  const corners: Corner[] = ['nw', 'ne', 'sw', 'se']
  return (
    <div
      ref={rootRef}
      data-no-swipe
      className="absolute inset-0 z-30 touch-none select-none"
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
      onKeyDown={onRootKeyDown}
    >
      <div className="pointer-events-none absolute inset-0 bg-black/30" aria-hidden />
      <div
        ref={boxRef}
        role="group"
        tabIndex={0}
        aria-label={`Move or resize ${label}. Drag the box or its corners, or use the arrow keys (Shift for small steps) and plus or minus. Enter saves, Escape cancels.`}
        onKeyDown={onBoxKeyDown}
        onPointerDown={down('move')}
        className="absolute cursor-move rounded-md border-2 border-accent bg-accent/10 shadow-[0_0_0_1px_rgba(0,0,0,0.6)] outline-none focus-visible:ring-2 focus-visible:ring-white"
        style={{ left: pct(box.x), top: pct(box.y), width: pct(box.w), height: pct(box.h) }}
      >
        {corners.map((c) => (
          <span
            key={c}
            aria-hidden
            onPointerDown={down(c)}
            className={clsx(
              'absolute flex size-7 items-center justify-center',
              c === 'nw' && '-left-3.5 -top-3.5 cursor-nwse-resize',
              c === 'ne' && '-right-3.5 -top-3.5 cursor-nesw-resize',
              c === 'sw' && '-bottom-3.5 -left-3.5 cursor-nesw-resize',
              c === 'se' && '-bottom-3.5 -right-3.5 cursor-nwse-resize',
            )}
          >
            <span className="size-3.5 rounded-sm border-2 border-white bg-accent shadow" />
          </span>
        ))}
      </div>
      <div
        ref={barRef}
        className="absolute flex items-center gap-1.5 whitespace-nowrap"
        style={bar ? { left: bar.left, top: bar.top } : { left: 0, top: 0, visibility: 'hidden' }}
      >
          <button
            type="button"
            disabled={saving}
            onClick={() => onSave(box)}
            className="inline-flex h-10 items-center gap-1.5 rounded-full bg-accent px-4 text-sm font-semibold text-white shadow-lg hover:bg-accent/90 disabled:opacity-60"
          >
            <Check className="size-4" aria-hidden />
            {saving ? 'Saving' : changed ? 'Save' : 'Done'}
          </button>
          <button
            type="button"
            disabled={saving}
            onClick={onCancel}
            className="inline-flex h-10 items-center gap-1.5 rounded-full bg-black/75 px-4 text-sm font-semibold text-white shadow-lg hover:bg-black/90 disabled:opacity-60"
          >
            <X className="size-4" aria-hidden />
            Cancel
          </button>
      </div>
      {error && (
        <p
          role="alert"
          className="absolute inset-x-2 top-2 mx-auto w-fit max-w-[90%] rounded-xl bg-black/85 px-3 py-2 text-center text-sm text-white shadow-lg"
        >
          {error}
        </p>
      )}
    </div>
  )
}
