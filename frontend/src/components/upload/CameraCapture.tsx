// In-app camera for the upload page (docs/ARCHITECTURE.md section 14). getUserMedia with the rear
// camera preferred; photos are drawn from the live video to a canvas and saved as JPEG, clips are
// recorded with MediaRecorder as MP4 when the browser can, else WebM. Every capture is handed to
// onCapture as a File right away (the page feeds it to the uploader with source 'camera'), so the
// owner can keep shooting. Without getUserMedia (or outside a secure context) the page uses a plain
// <input type="file" capture="environment"> instead: see cameraSupported() and CameraFallbackInput.
import { forwardRef, useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import clsx from 'clsx'
import { Camera, Circle, Loader2, RefreshCw, Square, SwitchCamera, Video, X } from 'lucide-react'
import { lockScroll } from '../ui'

type Mode = 'photo' | 'video'
type Facing = 'environment' | 'user'

const VIDEO_TYPES = [
  'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
  'video/mp4;codecs=avc1',
  'video/mp4',
  'video/webm;codecs=vp9,opus',
  'video/webm;codecs=vp8,opus',
  'video/webm',
]
const JPEG_QUALITY = 0.92

export function cameraSupported(): boolean {
  return (
    typeof window !== 'undefined' &&
    window.isSecureContext &&
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices &&
    typeof navigator.mediaDevices.getUserMedia === 'function'
  )
}

function recorderSupported(): boolean {
  return typeof window !== 'undefined' && typeof window.MediaRecorder !== 'undefined'
}

/** First recording format this browser supports ('' lets the browser choose). */
export function pickVideoType(): string {
  if (!recorderSupported()) return ''
  for (const t of VIDEO_TYPES) {
    try {
      if (MediaRecorder.isTypeSupported(t)) return t
    } catch {
      // isTypeSupported can throw on odd strings in old engines
    }
  }
  return ''
}

function stamp(d = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
}

function clock(s: number): string {
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

function cameraError(e: unknown): string {
  const name = e instanceof Error ? e.name : ''
  if (name === 'NotAllowedError' || name === 'SecurityError')
    return 'Camera access was refused. Allow the camera for this site in the browser settings, or use the system camera.'
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'No camera was found on this device.'
  if (name === 'NotReadableError' || name === 'AbortError') return 'The camera is in use by another app. Close it and try again.'
  return 'The camera could not be started.'
}

/** Hidden file input that opens the system camera; used when getUserMedia is not available. */
export const CameraFallbackInput = forwardRef<HTMLInputElement, { onFiles: (files: File[]) => void }>(
  function CameraFallbackInput({ onFiles }, ref) {
    return (
      <input
        ref={ref}
        type="file"
        accept="image/*,video/*"
        capture="environment"
        tabIndex={-1}
        aria-hidden
        className="sr-only"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? [])
          e.target.value = ''
          if (files.length) onFiles(files)
        }}
      />
    )
  },
)

export default function CameraCapture({
  open,
  onClose,
  onCapture,
  onUseSystemCamera,
}: {
  open: boolean
  onClose: () => void
  onCapture: (file: File) => void
  /** Open the system camera through a file input (shown when the live camera fails). */
  onUseSystemCamera?: () => void
}) {
  const videoEl = useRef<HTMLVideoElement>(null)
  const shutter = useRef<HTMLButtonElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const [mode, setMode] = useState<Mode>('photo')
  const [facing, setFacing] = useState<Facing>('environment')
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [silent, setSilent] = useState(false)
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [count, setCount] = useState(0)
  const [flash, setFlash] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const canRecord = recorderSupported()

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoEl.current) videoEl.current.srcObject = null
  }, [])

  // keep the latest onCapture for the recorder's onstop, which may fire after a re-render
  const captureRef = useRef(onCapture)
  useEffect(() => {
    captureRef.current = onCapture
  })

  const stopRecording = useCallback(() => {
    const rec = recorderRef.current
    if (rec && rec.state !== 'inactive') rec.stop()
  }, [])

  // start (or restart) the camera whenever it opens, or mode or facing changes
  useEffect(() => {
    if (!open) return
    let cancelled = false
    const start = async () => {
      setStarting(true)
      setError(null)
      setSilent(false)
      stopStream()
      const video: MediaTrackConstraints = {
        facingMode: { ideal: facing },
        width: { ideal: mode === 'photo' ? 3840 : 1920 },
        height: { ideal: mode === 'photo' ? 2160 : 1080 },
      }
      let stream: MediaStream | null = null
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video, audio: mode === 'video' })
      } catch (e) {
        if (mode === 'video') {
          // no microphone or no permission for it: record silent clips
          try {
            stream = await navigator.mediaDevices.getUserMedia({ video, audio: false })
            if (!cancelled) setSilent(true)
          } catch (e2) {
            if (!cancelled) setError(cameraError(e2))
          }
        } else if (!cancelled) {
          setError(cameraError(e))
        }
      }
      if (cancelled) {
        stream?.getTracks().forEach((t) => t.stop())
        return
      }
      setStarting(false)
      if (!stream) return
      streamRef.current = stream
      const el = videoEl.current
      if (el) {
        el.srcObject = stream
        el.play().catch(() => undefined)
      }
      shutter.current?.focus()
    }
    void start()
    return () => {
      cancelled = true
      stopRecording()
      stopStream()
    }
  }, [open, mode, facing, attempt, stopStream, stopRecording])

  // lock page scroll and close on Escape while open
  useEffect(() => {
    if (!open) return
    const unlock = lockScroll()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => {
      unlock()
      document.removeEventListener('keydown', onKey)
    }
  }, [open, onClose])

  // recording clock
  useEffect(() => {
    if (!recording) return
    const started = Date.now()
    setElapsed(0)
    const t = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 500)
    return () => window.clearInterval(t)
  }, [recording])

  useEffect(() => {
    if (!open) {
      setCount(0)
      setRecording(false)
    }
  }, [open])

  const takePhoto = () => {
    const el = videoEl.current
    if (!el || !el.videoWidth || !el.videoHeight) return
    const canvas = document.createElement('canvas')
    canvas.width = el.videoWidth
    canvas.height = el.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(el, 0, 0, canvas.width, canvas.height)
    setFlash(true)
    window.setTimeout(() => setFlash(false), 150)
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setError('The photo could not be saved. Try again.')
          return
        }
        const now = new Date()
        captureRef.current(new File([blob], `IMG_${stamp(now)}.jpg`, { type: 'image/jpeg', lastModified: now.getTime() }))
        setCount((n) => n + 1)
      },
      'image/jpeg',
      JPEG_QUALITY,
    )
  }

  const startRecording = () => {
    const stream = streamRef.current
    if (!stream || !canRecord) return
    const type = pickVideoType()
    let rec: MediaRecorder
    try {
      rec = type ? new MediaRecorder(stream, { mimeType: type }) : new MediaRecorder(stream)
    } catch {
      setError('This browser cannot record video here. Use the system camera instead.')
      return
    }
    chunksRef.current = []
    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data)
    }
    rec.onstop = () => {
      setRecording(false)
      recorderRef.current = null
      const mime = (rec.mimeType || type || 'video/webm').split(';')[0]
      const chunks = chunksRef.current
      chunksRef.current = []
      if (!chunks.length) return
      const blob = new Blob(chunks, { type: mime })
      const ext = mime.includes('mp4') ? 'mp4' : 'webm'
      const now = new Date()
      captureRef.current(new File([blob], `VID_${stamp(now)}.${ext}`, { type: mime, lastModified: now.getTime() }))
      setCount((n) => n + 1)
    }
    recorderRef.current = rec
    rec.start(1000)
    setRecording(true)
  }

  if (!open) return null

  const ready = !starting && !error && !!streamRef.current
  const switchDisabled = recording || starting

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Camera"
      className="fixed inset-0 z-50 flex flex-col bg-black text-white"
    >
      <div className="flex items-center justify-between gap-2 px-3 pt-[max(0.5rem,env(safe-area-inset-top))]">
        <button
          type="button"
          onClick={onClose}
          aria-label="Close the camera"
          className="inline-flex size-11 items-center justify-center rounded-full hover:bg-white/10"
        >
          <X className="size-6" aria-hidden />
        </button>
        <p className="text-sm text-white/80" aria-live="polite">
          {recording ? (
            <span className="inline-flex items-center gap-2 font-semibold text-white">
              <span className="size-2.5 animate-pulse rounded-full bg-red-500" aria-hidden />
              {clock(elapsed)}
            </span>
          ) : count > 0 ? (
            `${count} added to the uploads`
          ) : (
            ''
          )}
        </p>
        <button
          type="button"
          onClick={() => setFacing((f) => (f === 'environment' ? 'user' : 'environment'))}
          disabled={switchDisabled}
          aria-label={facing === 'environment' ? 'Use the front camera' : 'Use the rear camera'}
          className="inline-flex size-11 items-center justify-center rounded-full hover:bg-white/10 disabled:opacity-40"
        >
          <SwitchCamera className="size-6" aria-hidden />
        </button>
      </div>

      <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden">
        <video
          ref={videoEl}
          playsInline
          muted
          autoPlay
          className={clsx('max-h-full max-w-full', facing === 'user' && '-scale-x-100')}
        />
        {flash && <div className="absolute inset-0 bg-white/70" aria-hidden />}
        {starting && !error && (
          <Loader2 className="absolute size-8 animate-spin text-white/80" aria-label="Starting the camera" />
        )}
        {error && (
          <div className="absolute inset-x-4 flex flex-col items-center gap-3 text-center">
            <p className="max-w-sm text-sm" role="alert">
              {error}
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              <button
                type="button"
                onClick={() => setAttempt((n) => n + 1)}
                className="inline-flex h-10 items-center gap-2 rounded-full bg-white/15 px-4 text-sm font-semibold hover:bg-white/25"
              >
                <RefreshCw className="size-4" aria-hidden />
                Try again
              </button>
              {onUseSystemCamera && (
                <button
                  type="button"
                  onClick={onUseSystemCamera}
                  className="inline-flex h-10 items-center gap-2 rounded-full bg-white px-4 text-sm font-semibold text-black hover:bg-white/90"
                >
                  <Camera className="size-4" aria-hidden />
                  Use the system camera
                </button>
              )}
            </div>
          </div>
        )}
        {silent && mode === 'video' && !error && (
          <p className="absolute bottom-3 rounded-full bg-black/60 px-3 py-1 text-xs">No microphone: clips are recorded without sound.</p>
        )}
      </div>

      <div className="flex flex-col items-center gap-4 px-4 pb-[max(1.25rem,env(safe-area-inset-bottom))] pt-4">
        <div role="group" aria-label="Capture mode" className="inline-flex rounded-full bg-white/10 p-0.5">
          {(['photo', 'video'] as const).map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={mode === m}
              disabled={recording || (m === 'video' && !canRecord)}
              onClick={() => setMode(m)}
              className={clsx(
                'inline-flex h-10 items-center gap-1.5 rounded-full px-4 text-sm font-semibold disabled:opacity-40',
                mode === m ? 'bg-white text-black' : 'text-white/80 hover:text-white',
              )}
            >
              {m === 'photo' ? <Camera className="size-4" aria-hidden /> : <Video className="size-4" aria-hidden />}
              {m === 'photo' ? 'Photo' : 'Video'}
            </button>
          ))}
        </div>
        {mode === 'photo' ? (
          <button
            ref={shutter}
            type="button"
            onClick={takePhoto}
            disabled={!ready}
            aria-label="Take a photo"
            className="inline-flex size-[72px] items-center justify-center rounded-full border-4 border-white/90 disabled:opacity-40"
          >
            <span className="size-14 rounded-full bg-white" aria-hidden />
          </button>
        ) : recording ? (
          <button
            ref={shutter}
            type="button"
            onClick={stopRecording}
            aria-label="Stop recording"
            className="inline-flex size-[72px] items-center justify-center rounded-full border-4 border-white/90"
          >
            <Square className="size-7 fill-red-500 text-red-500" aria-hidden />
          </button>
        ) : (
          <button
            ref={shutter}
            type="button"
            onClick={startRecording}
            disabled={!ready}
            aria-label="Start recording"
            className="inline-flex size-[72px] items-center justify-center rounded-full border-4 border-white/90 disabled:opacity-40"
          >
            <Circle className="size-14 fill-red-500 text-red-500" aria-hidden />
          </button>
        )}
      </div>
    </div>,
    document.body,
  )
}
