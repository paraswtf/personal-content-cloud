// The video player of the viewer, plus the box of the face picked from the chips below it. The
// box is drawn over the paused frame once the seek has landed, and goes away when the user
// plays, seeks elsewhere or picks another chip.
//
// It also keeps `frameTimeRef` at the presentation time of the frame on screen (from
// requestVideoFrameCallback), which frame downloads send to the server: currentTime can sit
// anywhere inside a frame, and the server returns the first frame at or after the time it gets.
import { useEffect, useLayoutEffect, useState, type MutableRefObject, type RefObject } from 'react'
import type { ItemDetail } from '../../lib/types'
import { personName } from '../../lib/format'
import { posterTime } from '../../lib/preview'
import { animalLabel, type BoxTarget } from './boxes'
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
}

export { posterTime }

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

export default function VideoStage({ detail, src, videoRef, face, onDismissFace, frameTimeRef }: Props) {
  const [rect, setRect] = useState<Rect | null>(null)
  useFrameClock(videoRef, frameTimeRef, src)

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
    </div>
  )
}
