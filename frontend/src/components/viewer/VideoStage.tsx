// The video player of the viewer, plus the box of the face picked from the chips below it. The
// box is drawn over the paused frame once the seek has landed, and goes away when the user
// plays, seeks elsewhere or picks another chip.
//
// It also keeps `frameTimeRef` at the presentation time of the frame on screen (from
// requestVideoFrameCallback), which frame downloads send to the server: currentTime can sit
// anywhere inside a frame, and the server returns the first frame at or after the time it gets.
//
// Draw mode (pets in videos, section 22; people, section 10): the player pauses, the owner drags a box over
// the pet or the person in the frame on screen (or takes the whole frame), picks the species or who it is
// (PersonBoxCard), and the box is posted with the frame's time; it then shows as a track chip under the video.
import {
  useEffect,
  useLayoutEffect,
  useState,
  type MutableRefObject,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from 'react'
import { useMutation } from '@tanstack/react-query'
import { api } from '../../lib/api'
import { SPECIES, type ItemDetail, type Species } from '../../lib/types'
import { personName, speciesLabel } from '../../lib/format'
import { posterTime } from '../../lib/preview'
import { errorMessage } from '../admin/adminHooks'
import { animalLabel, type BoxTarget, type DrawKind } from './boxes'
import PersonBoxCard from './PersonBoxCard'
import { darkButton, useInvalidateItem } from './shared'
import { boxInVideo, type Rect } from './videoSeek'

interface Props {
  detail: ItemDetail
  /** urls.video of the detail (the stage is only shown once it is there). */
  src: string
  videoRef: RefObject<HTMLVideoElement | null>
  /** The face or body whose frame is on screen now (set after 'seeked'), or null. */
  face: BoxTarget | null
  onDismissFace: () => void
  /** Set to the mediaTime of the frame last presented; null when unknown (see currentFrameTime). */
  frameTimeRef: MutableRefObject<number | null>
  /** Draw mode, on the paused frame: a box for a pet the detector missed (section 22) or for a person
   *  (section 10); null when off. */
  drawing?: DrawKind | null
  /** Draw mode ended (cancelled, or the box was added). */
  onDrawEnd?: () => void
  /** The time of the frame on screen (the viewer's currentFrameTime), sent as `t` with the box. */
  frameTime?: () => number
  onToast?: (message: string) => void
}

export { posterTime }

/** The smallest box the owner can draw, in screen pixels. */
const MIN_DRAW_PX = 24
/** "Whole frame": the picture minus this margin on every side. */
const WHOLE_MARGIN = 0.03
const FULL = { x: 0, y: 0, w: 1, h: 1 }

interface Drag {
  x0: number
  y0: number
  x1: number
  y1: number
}
interface NormBox {
  x: number
  y: number
  w: number
  h: number
}

// The frame on screen starts at or just before currentTime, less than one frame earlier. A tracked
// time further away belongs to an earlier position whose new frame has not been presented yet.
const FRAME_SLACK_S = 0.1

/**
 * The time of the frame on screen in `video`: the tracked presentation time when it belongs to the
 * current position, otherwise currentTime (browsers without requestVideoFrameCallback, or a seek
 * whose frame has not been presented yet). 0 without a player.
 */
export function currentFrameTime(video: HTMLVideoElement | null, frameTime: number | null, posterTime = 0): number {
  if (!video) return 0
  const t = Number.isFinite(video.currentTime) ? video.currentTime : 0
  // Before the first play or seek the browser shows the poster, which is the frame at posterTime.
  if (frameTime === null && t === 0 && video.played.length === 0) return posterTime
  if (frameTime !== null && Number.isFinite(frameTime) && t - frameTime > -0.002 && t - frameTime < FRAME_SLACK_S) {
    return frameTime
  }
  return t
}

/** Keep `ref` at the mediaTime of each presented frame of `video`, for as long as it is mounted. */
function useFrameClock(videoRef: RefObject<HTMLVideoElement | null>, ref: MutableRefObject<number | null>, src: string) {
  useEffect(() => {
    const video = videoRef.current
    ref.current = null
    if (!video || !('requestVideoFrameCallback' in video)) return
    let handle = 0
    const onFrame: VideoFrameRequestCallback = (_now, meta) => {
      ref.current = meta.mediaTime
      handle = video.requestVideoFrameCallback(onFrame)
    }
    handle = video.requestVideoFrameCallback(onFrame)
    // A seek keeps the last time: a seek that lands on the frame already shown presents nothing new
    // (no callback), and currentFrameTime ignores the time when the position moved past it.
    const stale = () => {
      ref.current = null
    }
    video.addEventListener('emptied', stale)
    return () => {
      video.cancelVideoFrameCallback(handle)
      video.removeEventListener('emptied', stale)
      ref.current = null
    }
  }, [videoRef, ref, src])
}

export default function VideoStage({ detail, src, videoRef, face, onDismissFace, frameTimeRef, drawing = null, onDrawEnd, frameTime, onToast }: Props) {
  const [rect, setRect] = useState<Rect | null>(null)
  useFrameClock(videoRef, frameTimeRef, src)

  // Draw mode: where the picture sits inside the element (letterboxing), the drag, and the box awaiting its species.
  const [picture, setPicture] = useState<Rect | null>(null)
  const [drag, setDrag] = useState<Drag | null>(null)
  const [pending, setPending] = useState<{ box: NormBox; t: number } | null>(null)
  // the drawn person box on screen: PersonBoxCard sits next to it
  const [pendingEl, setPendingEl] = useState<HTMLDivElement | null>(null)
  const invalidate = useInvalidateItem(detail.uid)
  const person = drawing === 'person'
  const what = person ? 'person' : 'pet'

  useEffect(() => {
    // switched between a pet and a person: start over
    setDrag(null)
    setPending(null)
  }, [drawing])

  useEffect(() => {
    if (!drawing) {
      setDrag(null)
      setPending(null)
      setPicture(null)
      return
    }
    const video = videoRef.current
    if (!video) return
    // The frame under the box must stay put: the player pauses and stays paused while drawing.
    const pause = () => video.pause()
    pause()
    const update = () => setPicture(boxInVideo(video, FULL))
    update()
    const ro = new ResizeObserver(update)
    ro.observe(video)
    const events = ['loadedmetadata', 'resize', 'seeked'] as const
    for (const e of events) video.addEventListener(e, update)
    video.addEventListener('play', pause)
    return () => {
      ro.disconnect()
      for (const e of events) video.removeEventListener(e, update)
      video.removeEventListener('play', pause)
    }
  }, [drawing, videoRef])

  const add = useMutation({
    mutationFn: (v: { box: NormBox; t: number; species: Species }) => api.items.addAnimal(detail.uid, { ...v.box, species: v.species, t: v.t }),
    onSuccess: () => {
      setPending(null)
      invalidate()
      onDrawEnd?.()
      onToast?.('Pet box added. Name it from its chip under the video.')
    },
    onError: (e) => onToast?.(errorMessage(e, 'Could not add the box')),
  })

  const now = () => {
    const t = frameTime?.()
    if (t !== undefined && Number.isFinite(t)) return t
    const cur = videoRef.current?.currentTime
    return cur !== undefined && Number.isFinite(cur) ? cur : 0
  }
  const point = (e: ReactPointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    return {
      x: Math.min(Math.max(e.clientX - r.left, 0), r.width),
      y: Math.min(Math.max(e.clientY - r.top, 0), r.height),
      cw: r.width,
      ch: r.height,
    }
  }
  const onDrawDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 && e.pointerType === 'mouse') return
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    const p = point(e)
    setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y })
  }
  const onDrawMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag) return
    const p = point(e)
    setDrag((d) => (d ? { ...d, x1: p.x, y1: p.y } : d))
  }
  const onDrawUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag) return
    const p = point(e)
    const x0 = Math.min(drag.x0, p.x)
    const y0 = Math.min(drag.y0, p.y)
    const wpx = Math.abs(p.x - drag.x0)
    const hpx = Math.abs(p.y - drag.y0)
    setDrag(null)
    if (wpx < MIN_DRAW_PX || hpx < MIN_DRAW_PX || !p.cw || !p.ch) return
    setPending({ box: { x: x0 / p.cw, y: y0 / p.ch, w: wpx / p.cw, h: hpx / p.ch }, t: now() })
  }
  const dragRect = drag
    ? { left: Math.min(drag.x0, drag.x1), top: Math.min(drag.y0, drag.y1), width: Math.abs(drag.x1 - drag.x0), height: Math.abs(drag.y1 - drag.y0) }
    : null
  const pendingRect =
    pending && picture
      ? {
          left: picture.left + pending.box.x * picture.width,
          top: picture.top + pending.box.y * picture.height,
          width: pending.box.w * picture.width,
          height: pending.box.h * picture.height,
        }
      : null

  useLayoutEffect(() => {
    const video = videoRef.current
    if (!face || !video) {
      setRect(null)
      return
    }
    const update = () => setRect(boxInVideo(video, face.box))
    update()
    const ro = new ResizeObserver(update)
    ro.observe(video)
    // Our own seek has finished by the time a face is set, so any of these is the user's doing.
    const dismiss = () => onDismissFace()
    const events = ['play', 'seeking', 'emptied'] as const
    for (const e of events) video.addEventListener(e, dismiss)
    video.addEventListener('resize', update)
    return () => {
      ro.disconnect()
      for (const e of events) video.removeEventListener(e, dismiss)
      video.removeEventListener('resize', update)
    }
  }, [face, videoRef, onDismissFace])

  const p = face?.box.person
  const name = face?.kind === 'animal' ? animalLabel(face.box) : p ? personName(p) : face?.box.guess ? `${personName(face.box.guess.person)}?` : 'Not tagged'
  return (
    <div className="relative">
      <video
        ref={videoRef}
        controls
        playsInline
        preload="metadata"
        poster={detail.urls.preview_small}
        src={src}
        className="block bg-black"
        style={{ maxWidth: '100cqw', maxHeight: '100cqh' }}
      >
        Your browser cannot play this video. Use Download to save the original.
      </video>
      {face && rect && (
        <div
          className="pointer-events-none absolute rounded-sm border-2 border-accent shadow-[0_0_0_1px_rgba(0,0,0,0.55)]"
          style={{ left: rect.left, top: rect.top, width: rect.width, height: rect.height }}
          aria-hidden
        >
          <span
            className={
              'absolute left-[-2px] max-w-[12rem] truncate rounded-sm bg-accent px-1.5 py-0.5 text-xs font-semibold text-white ' +
              (rect.top >= 22 ? 'bottom-full mb-0.5' : 'top-full mt-0.5')
            }
          >
            {name}
          </span>
        </div>
      )}

      {drawing && picture && !pending && (
        <div
          data-no-swipe
          role="application"
          aria-label={`Draw a box over the ${what}`}
          className="absolute z-20 cursor-crosshair touch-none select-none"
          style={{ left: picture.left, top: picture.top, width: picture.width, height: picture.height }}
          onPointerDown={onDrawDown}
          onPointerMove={onDrawMove}
          onPointerUp={onDrawUp}
          onPointerCancel={() => setDrag(null)}
        >
          {dragRect && (
            <div
              className={
                'pointer-events-none absolute rounded-lg border-2 border-dashed ' +
                (person ? 'border-sky-300 bg-sky-300/10' : 'border-emerald-300 bg-emerald-300/10')
              }
              style={dragRect}
            />
          )}
        </div>
      )}

      {drawing && (
        <div data-no-swipe className="pointer-events-none absolute inset-x-0 top-0 z-30 flex flex-wrap items-center justify-center gap-2 p-2">
          {!pending && (
            <span role="status" className="rounded-full bg-black/70 px-3 py-1.5 text-sm font-semibold text-white">
              {picture ? `Paused. Drag over the ${what} in this frame` : 'Loading the video'}
            </span>
          )}
          {!pending && picture && (
            <button
              type="button"
              onClick={() => setPending({ box: { x: WHOLE_MARGIN, y: WHOLE_MARGIN, w: 1 - 2 * WHOLE_MARGIN, h: 1 - 2 * WHOLE_MARGIN }, t: now() })}
              className={darkButton('default', 'pointer-events-auto bg-black/60')}
            >
              Whole frame
            </button>
          )}
          <button type="button" onClick={() => onDrawEnd?.()} className={darkButton('default', 'pointer-events-auto bg-black/60')}>
            Cancel
          </button>
        </div>
      )}

      {person && pending && pendingRect && (
        <>
          <div
            ref={setPendingEl}
            className="pointer-events-none absolute z-20 rounded-lg border-2 border-sky-300 bg-sky-300/10"
            style={pendingRect}
          />
          {pendingEl && (
            <PersonBoxCard
              uid={detail.uid}
              box={pending.box}
              t={pending.t}
              anchor={pendingEl}
              onAdded={(message) => {
                setPending(null)
                onDrawEnd?.()
                onToast?.(message)
              }}
              onRedraw={() => setPending(null)}
              onCancel={() => onDrawEnd?.()}
            />
          )}
        </>
      )}

      {drawing === 'pet' && pending && (
        <>
          {pendingRect && (
            <div className="pointer-events-none absolute z-20 rounded-lg border-2 border-emerald-300 bg-emerald-300/10" style={pendingRect} />
          )}
          <div
            data-no-swipe
            role="dialog"
            aria-label="What is it?"
            className="absolute left-1/2 top-1/2 z-30 w-[min(22rem,92%)] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-[#1b1b1b] p-3 text-white shadow-2xl"
          >
            <p className="mb-2 text-sm font-semibold">What is it?</p>
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Species">
              {SPECIES.map((sp) => (
                <button
                  key={sp}
                  type="button"
                  disabled={add.isPending}
                  onClick={() => add.mutate({ ...pending, species: sp })}
                  className="rounded-full border border-white/15 px-3 py-1.5 text-sm font-semibold text-white hover:bg-white/10 disabled:opacity-50"
                >
                  {speciesLabel(sp)}
                </button>
              ))}
            </div>
            {add.isPending && <p className="mt-2 text-xs text-white/60">Adding the box</p>}
            <div className="mt-3 flex gap-2">
              <button type="button" disabled={add.isPending} onClick={() => setPending(null)} className={darkButton('default')}>
                Draw again
              </button>
              <button type="button" disabled={add.isPending} onClick={() => onDrawEnd?.()} className={darkButton('default')}>
                Cancel
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
