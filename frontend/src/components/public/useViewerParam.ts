// Drives the Viewer overlay from the `m` search param of a gallery page.
// Opening pushes a history entry (so Back closes the viewer); prev/next replace it.
// A search hit on a video also carries `t`, the second to open the player at.
// An item that drops out of the list while it is open (the owner took the person off it on their page, or changed
// what the list filters on) stays open at its old place: prev and next go on from there, and once the owner moves
// to another item it is gone from the list. Deleting it or moving it to the vault leaves it in the viewer itself.
import { useCallback, useEffect, useRef } from 'react'
import { useLocation, useNavigate, useSearchParams } from 'react-router'

interface ViewerState {
  hubViewer?: boolean
}

/** Seconds from the `t` param, or null when absent or not a number. */
function parseTime(raw: string | null): number | null {
  if (!raw) return null
  const n = Number(raw)
  return Number.isFinite(n) && n >= 0 ? n : null
}

/** Value for the `t` param: whole milliseconds, no trailing zeros. */
function formatTime(t: number | null | undefined): string | null {
  return t === null || t === undefined || !Number.isFinite(t) || t < 0 ? null : String(Math.round(t * 1000) / 1000)
}

export function useViewerParam(
  items: { uid: string; match_time_s?: number | null }[],
  hasMore: boolean,
  loadMore: () => void,
  total?: number,
  /** The open item's true index in the whole list when `items` holds only part of it (the timeline). */
  positionOf?: (uid: string) => number | null,
) {
  const [params, setParams] = useSearchParams()
  const location = useLocation()
  const navigate = useNavigate()
  const uid = params.get('m')
  const t = uid ? parseTime(params.get('t')) : null
  const index = uid ? items.findIndex((i) => i.uid === uid) : -1
  const pushedByUs = !!(location.state as ViewerState | null)?.hubViewer
  // Where the open item last stood in the list (and its position as shown); while it is out of the list, its old place.
  const shownNow = index >= 0 && uid ? (positionOf?.(uid) ?? index) : null
  const anchor = useRef<{ uid: string; index: number; shown: number } | null>(null)
  useEffect(() => {
    if (uid && index >= 0 && shownNow !== null) anchor.current = { uid, index, shown: shownNow }
  }, [uid, index, shownNow])
  const gone = index < 0 && uid && anchor.current?.uid === uid ? anchor.current : null
  const ghost = gone ? Math.min(gone.index, items.length) : null
  // The neighbours: around the item, or for one out of the list, the items now either side of its old place.
  const prevIndex = index >= 0 ? index - 1 : ghost !== null ? ghost - 1 : -1
  const nextIndex = index >= 0 ? index + 1 : ghost !== null ? ghost : -1

  const withParam = useCallback(
    (next: string | null, time?: number | null) => {
      const p = new URLSearchParams(params)
      if (next) p.set('m', next)
      else p.delete('m')
      const ts = next ? formatTime(time) : null
      if (ts) p.set('t', ts)
      else p.delete('t')
      return p
    },
    [params],
  )

  // Without an explicit time, a video search hit opens at its matched frame (as prev/next do).
  const open = useCallback(
    (target: string, time?: number | null) => {
      const at = time !== undefined ? time : items.find((i) => i.uid === target)?.match_time_s
      setParams(withParam(target, at), { state: { hubViewer: true } satisfies ViewerState })
    },
    [items, setParams, withParam],
  )

  const close = useCallback(() => {
    if (pushedByUs) navigate(-1)
    else setParams(withParam(null), { replace: true })
  }, [pushedByUs, navigate, setParams, withParam])

  const go = useCallback(
    (to: number) => {
      const target = items[to]
      if (target) setParams(withParam(target.uid, target.match_time_s), { replace: true, state: location.state })
    },
    [items, setParams, withParam, location.state],
  )

  // Keep a page loaded ahead of the viewer so "next" never dead-ends on a long list.
  const at = index >= 0 ? index : ghost
  useEffect(() => {
    if (at !== null && at >= 0 && hasMore && at >= items.length - 3) loadMore()
  }, [at, hasMore, items.length, loadMore])

  const onPrev = prevIndex >= 0 && prevIndex < items.length ? () => go(prevIndex) : undefined
  const onNext = nextIndex >= 0 && nextIndex < items.length ? () => go(nextIndex) : undefined
  let position: { index: number; total: number } | undefined
  if (shownNow !== null) position = { index: shownNow, total: Math.max(total ?? items.length, items.length, shownNow + 1) }
  else if (gone) {
    // the list no longer counts it: count it while it is open
    position = { index: gone.shown, total: Math.max((total ?? items.length) + 1, items.length + 1, gone.shown + 1) }
  }
  return { uid, t, open, close, onPrev, onNext, position }
}
