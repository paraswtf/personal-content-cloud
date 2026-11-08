// Drives the Viewer overlay from the `m` search param of a gallery page.
// Opening pushes a history entry (so Back closes the viewer); prev/next replace it.
// A search hit on a video also carries `t`, the second to open the player at.
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
    (delta: number) => {
      const target = items[index + delta]
      if (target) setParams(withParam(target.uid, target.match_time_s), { replace: true, state: location.state })
    },
    [items, index, setParams, withParam, location.state],
  )

  // Keep a page loaded ahead of the viewer so "next" never dead-ends on a long list.
  useEffect(() => {
    if (index >= 0 && hasMore && index >= items.length - 3) loadMore()
  }, [index, hasMore, items.length, loadMore])

  // The open item disappeared from the list (deleted or hidden): close the viewer.
  const seen = useRef<string | null>(null)
  useEffect(() => {
    if (index >= 0) seen.current = uid
    else if (uid && seen.current === uid) {
      seen.current = null
      setParams(withParam(null), { replace: true })
    }
  }, [index, uid, setParams, withParam])

  const onPrev = index > 0 ? () => go(-1) : undefined
  const onNext = index >= 0 && index < items.length - 1 ? () => go(1) : undefined
  const shown = index >= 0 && uid ? (positionOf?.(uid) ?? index) : index
  const position = index >= 0 ? { index: shown, total: Math.max(total ?? items.length, items.length, shown + 1) } : undefined
  return { uid, t, open, close, onPrev, onNext, position }
}
