// Frame-accurate seeking for the face chips under a video.
//
// A face's frame_time_s is the exact presentation time of the frame it was found in (0 = the
// first frame). A time that sits exactly on a frame boundary can show the previous frame, so we
// seek to the middle of the frame instead: frame_time + half a frame duration.

const FALLBACK_FPS = 30

/** The currentTime that shows the frame starting at `frameTime`, clamped to the video. */
export function frameSeekTime(frameTime: number, fps: number | null | undefined, duration: number): number {
  const rate = fps && Number.isFinite(fps) && fps > 0 ? fps : FALLBACK_FPS
  let t = Math.max(0, frameTime) + 0.5 / rate
  if (Number.isFinite(duration) && duration > 0) t = Math.min(t, duration)
  return t
}

function abortError(): DOMException {
  return new DOMException('Seek cancelled', 'AbortError')
}

/** Resolve on the first of `events` fired by `el`; reject on 'error' or when `signal` aborts. */
function once(el: HTMLMediaElement, events: string[], signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortError())
      return
    }
    const done = (fn: () => void) => {
      for (const e of events) el.removeEventListener(e, onEvent)
      el.removeEventListener('error', onError)
      signal.removeEventListener('abort', onAbort)
      fn()
    }
    const onEvent = () => done(resolve)
    const onError = () => done(() => reject(new Error('The video could not be loaded')))
    const onAbort = () => done(() => reject(abortError()))
    for (const e of events) el.addEventListener(e, onEvent)
    el.addEventListener('error', onError)
    signal.addEventListener('abort', onAbort)
  })
}

/**
 * Pause `video` and show the frame that starts at `frameTime`. Waits for the metadata first
 * (starting the load if the browser has not), then for the 'seeked' event. Rejects with an
 * AbortError when `signal` aborts (another chip was picked, the viewer moved on).
 */
export async function seekToFrame(
  video: HTMLVideoElement,
  frameTime: number,
  fps: number | null | undefined,
  signal: AbortSignal,
): Promise<void> {
  // A failed element never fires loadedmetadata, so waiting for it would hang until aborted.
  if (video.error) throw new Error('The video could not be loaded')
  if (video.readyState < HTMLMediaElement.HAVE_METADATA) {
    const waiting = once(video, ['loadedmetadata'], signal)
    const idle =
      video.networkState === HTMLMediaElement.NETWORK_EMPTY ||
      (video.networkState === HTMLMediaElement.NETWORK_IDLE && video.readyState === HTMLMediaElement.HAVE_NOTHING)
    if (idle) video.load()
    await waiting
  }
  if (signal.aborted) throw abortError()
  video.pause()
  const target = frameSeekTime(frameTime, fps, video.duration)
  const seeked = once(video, ['seeked'], signal)
  video.currentTime = target
  await seeked
}

export interface Rect {
  left: number
  top: number
  width: number
  height: number
}

/**
 * Where a box normalised to the video frame (x, y, w, h in 0..1) lies inside the video element,
 * in CSS pixels from the element's content box, taking the object-fit: contain letterboxing into
 * account. Null while the video's own size is not known yet.
 */
export function boxInVideo(video: HTMLVideoElement, box: { x: number; y: number; w: number; h: number }): Rect | null {
  const vw = video.videoWidth
  const vh = video.videoHeight
  const cw = video.clientWidth
  const ch = video.clientHeight
  if (!vw || !vh || !cw || !ch) return null
  const scale = Math.min(cw / vw, ch / vh)
  const dw = vw * scale
  const dh = vh * scale
  const ox = (cw - dw) / 2
  const oy = (ch - dh) / 2
  return {
    left: video.offsetLeft + video.clientLeft + ox + box.x * dw,
    top: video.offsetTop + video.clientTop + oy + box.y * dh,
    width: box.w * dw,
    height: box.h * dh,
  }
}
