// Full-screen viewer overlay for photos, videos and documents: face and body boxes with person
// guesses, prev/next (keys, buttons, swipe), the info panel, downloads of the original, and a
// document mode with the page viewer. `vault` shows vault items (every request carries vault=1).
// Live Photos and motion photos get a LIVE button over the stage that plays the clip in place of the
// still (section 20); leaving the item, Escape or a second tap stops it.
import { useCallback, useEffect, useMemo, useRef, useState, type TouchEvent } from 'react'
import { createPortal } from 'react-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { ChevronDown, ChevronLeft, ChevronRight, Info, Loader2, Pencil, PawPrint, ScanFace, Star, X } from 'lucide-react'
import { api, ApiError } from '../lib/api'
import type { BoxIn, ItemDetail } from '../lib/types'
import { formatDuration, personName } from '../lib/format'
import { boxOrigin, mergeOrigins, originPhrase, type Origin } from '../lib/origin'
import { useAuth } from '../lib/auth'
import { claimForToast } from '../lib/undo'
import { useConfig } from '../lib/hooks'
import { vaultify, withVault } from '../lib/preview'
import ErrorBoundary, { ErrorPanel } from './ErrorBoundary'
import OriginIcon from './people/OriginIcon'
import { useToast } from './Toast'
import { ConfirmModal, lockScroll } from './ui'
import { errorMessage } from './admin/adminHooks'
import DocumentStage, { FileCard } from './viewer/DocumentStage'
import DownloadMenu from './viewer/DownloadMenu'
import ShareButton from './share/ShareButton'
import AnimalEditor from './viewer/AnimalEditor'
import FaceEditor from './viewer/FaceEditor'
import InfoPanel from './viewer/InfoPanel'
import LiveButton from './viewer/LiveButton'
import PhotoStage from './viewer/PhotoStage'
import VideoPreparing from './viewer/VideoPreparing'
import VideoStage, { currentFrameTime, posterTime } from './viewer/VideoStage'
import { animalLabel, boxKey, findBox, groupVideoBoxes, stillBoxes, videoBoxes, type AnimalBox, type BoxKind, type BoxTarget, type DrawKind, type PersonBoxTarget } from './viewer/boxes'
import { useGuessActions } from './viewer/useGuessActions'
import { useOnScreenFaces } from './viewer/useOnScreenFaces'
import { seekToFrame } from './viewer/videoSeek'
import { darkButton, findCachedItem, iconButton, useInvalidateItem } from './viewer/shared'

export interface ViewerProps {
  uid: string
  onClose: () => void
  onPrev?: () => void
  onNext?: () => void
  /** `index` is zero-based (shown as index + 1); `total` is the size of the whole list. */
  position?: { index: number; total: number }
  /** Video only: open the player paused at this second (a search hit's best-matching moment). */
  startAt?: number | null
  /** The item lives in the vault: detail, files and text are requested with vault=1. */
  vault?: boolean
  /** 'document' opens the info panel with the document card first (from the documents page). */
  mode?: 'document' | null
}

// Remembered for the rest of the visit so prev/next and reopening keep the choice.
let rememberFaces = true
/** How long a photo's boxes stay lit after a tap on the person's badge in the details. */
const SPOTLIGHT_MS = 4000
let rememberInfo: boolean | null = null

function isTyping(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false
  return el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)
}

function VideoChips({
  targets,
  editingKey,
  activeKey,
  onSeek,
  onEdit,
}: {
  targets: BoxTarget[]
  editingKey: string | null
  activeKey: string | null
  onSeek: (t: BoxTarget) => void
  onEdit: (t: BoxTarget, anchor: HTMLElement, group?: boolean) => void
}) {
  // One chip per person or look-alike (section 10): editing it applies to every track of the group;
  // "Show tracks" opens the group so one track can be handled alone. Each name carries its origin: a person
  // icon when the owner tagged or drew any of the tracks, a bot when the system did.
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set())
  // The chip of the picked track comes into view, also when the pick came from the details.
  const listRef = useRef<HTMLUListElement>(null)
  useEffect(() => {
    if (!activeKey) return
    listRef.current?.querySelector<HTMLElement>('[aria-pressed="true"]')?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }, [activeKey])
  if (!targets.length) return null
  const groups = groupVideoBoxes(targets)
  const toggle = (key: string) =>
    setExpanded((cur) => {
      const next = new Set(cur)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  const chip = (t: BoxTarget, count: number, groupKey: string | null, origin: Origin | null, inner = false) => {
    const key = boxKey(t)
    const b = t.box
    const isAnimal = t.kind === 'animal'
    const name = isAnimal ? animalLabel(t.box) : b.person ? personName(b.person) : b.guess ? `${personName(b.guess.person)}?` : 'Not tagged'
    const whole = count > 1 && groupKey !== null
    const how = origin ? `, ${originPhrase(origin)}` : ''
    return (
      <li
        key={(whole ? 'group:' : '') + key}
        className={clsx(
          'flex shrink-0 items-center rounded-full',
          inner && 'ml-1 scale-95',
          activeKey === key ? 'bg-white/25' : 'bg-white/10',
          editingKey === key ? 'ring-2 ring-accent' : activeKey === key && 'ring-2 ring-white/80',
          !b.person && (b.guess ? 'outline outline-1 outline-amber-300' : 'outline outline-1 outline-dashed outline-amber-300/70'),
        )}
      >
        <button
          type="button"
          onClick={() => onSeek(t)}
          data-box-key={key}
          aria-pressed={activeKey === key}
          className="flex h-10 items-center gap-2 rounded-full pl-1 pr-3 text-sm text-white hover:bg-white/10"
          aria-label={(whole ? `${name}, ${count} tracks, show the first` : `${name}, ${formatDuration(b.frame_time_s)}, show this frame`) + how}
        >
          <img src={b.url} alt="" className="size-8 rounded-full object-cover" />
          {isAnimal && <PawPrint className="-mr-1 size-3.5 shrink-0 text-white/70" aria-hidden />}
          <span className={clsx('max-w-[9rem] truncate', b.guess && !b.person && 'text-amber-200')}>{name}</span>{' '}
          {origin && (
            <OriginIcon origin={origin} decorative className={clsx('-ml-1', origin === 'guess' ? 'text-amber-200' : 'text-white/70')} />
          )}
          <span className="text-xs tabular-nums text-white/55">{whole ? `${count} tracks` : formatDuration(b.frame_time_s)}</span>
        </button>
        <button
          type="button"
          onClick={(e) => onEdit(t, e.currentTarget, whole)}
          className="-ml-1 flex size-10 items-center justify-center rounded-full text-white/60 hover:bg-white/10 hover:text-white"
          aria-label={
            whole
              ? `Edit all ${count} tracks of ${name}`
              : b.person
                ? `Edit, tagged as ${name}`
                : isAnimal
                  ? 'Name this pet'
                  : 'Tag this person'
          }
        >
          <Pencil className="size-4" aria-hidden />
        </button>
        {whole && (
          <button
            type="button"
            onClick={() => toggle(groupKey)}
            aria-expanded={expanded.has(groupKey)}
            className="-ml-1 flex size-10 items-center justify-center rounded-full text-white/60 hover:bg-white/10 hover:text-white"
            aria-label={expanded.has(groupKey) ? `Hide the tracks of ${name}` : `Show the ${count} tracks of ${name}`}
          >
            <ChevronDown className={clsx('size-4 transition-transform', expanded.has(groupKey) && 'rotate-180')} aria-hidden />
          </button>
        )}
      </li>
    )
  }
  return (
    <div className="shrink-0 border-t border-white/10 px-2 py-2">
      <ul ref={listRef} className="flex gap-2 overflow-x-auto overscroll-x-contain px-1 pb-1" aria-label="People and pets in this video" data-no-swipe>
        {groups.flatMap((g) => {
          const whole = g.targets.length > 1
          const origin = whole ? mergeOrigins(g.targets.map((t) => boxOrigin(t.box))) : boxOrigin(g.first.box)
          const items = [chip(g.first, g.targets.length, whole ? g.key : null, origin)]
          if (whole && expanded.has(g.key)) items.push(...g.targets.map((t) => chip(t, 1, null, boxOrigin(t.box), true)))
          return items
        })}
      </ul>
    </div>
  )
}

/** The viewer inside an ErrorBoundary: a crash while drawing an item shows the error over the page
 *  with Close, instead of unmounting the whole app; opening another item clears it. */
export default function Viewer(props: ViewerProps) {
  return (
    <ErrorBoundary
      resetKey={props.uid}
      fallback={(error, reset) => (
        <div role="dialog" aria-modal="true" aria-label="Viewer error" className="fixed inset-0 z-40 flex items-center justify-center bg-bg/95 p-4">
          <div className="w-full max-w-md rounded-2xl bg-surface shadow-lg">
            <ErrorPanel error={error} title="The viewer hit an error" onRetry={reset} onClose={props.onClose} />
          </div>
        </div>
      )}
    >
      <ViewerInner {...props} />
    </ErrorBoundary>
  )
}

function ViewerInner({ uid, onClose, onPrev, onNext, position, startAt = null, vault = false, mode = null }: ViewerProps) {
  const qc = useQueryClient()
  const config = useConfig()
  const auth = useAuth()
  // While the auth state loads, vaultUnlocked reads false; do not treat that as a lock.
  const vaultClosed = !auth.loading && !auth.vaultUnlocked
  const rootRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const frameTimeRef = useRef<number | null>(null)
  const seekCtl = useRef<AbortController | null>(null)
  const touch = useRef<{ x: number; y: number; multi: boolean; ignore: boolean } | null>(null)

  const [showFaces, setShowFaces] = useState(rememberFaces)
  const [showInfo, setShowInfo] = useState<boolean>(
    () => mode === 'document' || (rememberInfo ?? (typeof window !== 'undefined' && window.matchMedia('(min-width: 1024px)').matches)),
  )
  const [editing, setEditing] = useState<{ kind: BoxKind; id: number; anchor: HTMLElement; group?: boolean; pick?: boolean } | null>(null)
  const [tagging, setTagging] = useState(false)
  const [drawing, setDrawing] = useState<DrawKind | null>(null) // a pet box (section 22) or a person box (section 10) the owner draws
  // section 10: the box the owner is moving or resizing (BoxAdjuster on the photo)
  const [adjusting, setAdjusting] = useState<{ kind: BoxKind; id: number } | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [sharing, setSharing] = useState(false) // the Share dialog (section 26): the keyboard steps aside like for the delete confirm
  const [exitAfter, setExitAfter] = useState<'next' | 'close' | null>(null)
  // The viewer's own message; `entry` is the undo history entry it describes, when it offers Undo.
  const [toast, setToast] = useState<{ text: string; entry: string | null } | null>(null)
  const appToast = useToast()
  const [seekFace, setSeekFace] = useState<{ key: string; shown: boolean } | null>(null)
  const [page, setPage] = useState(0)
  // A person whose boxes the photo highlights for a moment, after a tap on their badge in the details (section 14),
  // or one box (`key`), after a tap on a guess there; `at` tells two taps on the same badge apart, so a second tap
  // lights them for the full time again.
  const [spotlight, setSpotlight] = useState<{ person: number | null; key: string | null; at: number } | null>(null)
  const infoToggleRef = useRef<HTMLButtonElement>(null)
  const [livePlaying, setLivePlaying] = useState(false)

  const placeholder = useMemo(() => findCachedItem(qc, uid), [qc, uid])
  const detailQ = useQuery({
    queryKey: ['item', uid, vault],
    queryFn: () => api.items.detail(uid, vault),
    select: (d: ItemDetail) => vaultify(d, vault),
    refetchInterval: (q) => {
      const d = q.state.data as ItemDetail | undefined
      if (!d) return false
      if (d.analysis_status === 'pending' || d.text_status === 'pending') return 4000
      return d.kind === 'video' && d.preview_status === 'pending' ? 5000 : false
    },
  })
  // A private item stays hidden once the vault locks or its window ends, wherever it was opened
  // (a /m/<uid> link without vault=1 caches it under ['item', uid, false]). A refetch that fails
  // with vault_locked keeps the old data, so the gate is on the data itself.
  const hiddenPrivate = vaultClosed && !!detailQ.data?.nsfw
  const detail = hiddenPrivate ? undefined : detailQ.data
  const invalidate = useInvalidateItem(uid)
  // Drop the cached detail and text once per lock, so reopening does not bring them back. Once only:
  // if the server's window is still open for a moment (clock skew), a refetch must not loop.
  const purged = useRef<string | null>(null)
  useEffect(() => {
    if (!vaultClosed) purged.current = null
    if (!hiddenPrivate || purged.current === uid) return
    purged.current = uid
    qc.removeQueries({ queryKey: ['item', uid] })
    qc.removeQueries({ queryKey: ['item-text', uid] })
  }, [vaultClosed, hiddenPrivate, qc, uid])

  useEffect(() => {
    setEditing(null)
    setTagging(false)
    setDrawing(null)
    setAdjusting(null)
    setPage(0)
  }, [uid])

  const detailVersion = detail?.version
  // Leaving the item (or a new version of it) stops the clip: it only plays on request.
  useEffect(() => setLivePlaying(false), [uid, detailVersion])
  const liveUrl = detail?.kind === 'image' && detail.live_video ? (detail.urls.live ?? detail.live_video.url) : null
  const stopLive = useCallback(() => setLivePlaying(false), [])
  const onLiveError = useCallback(() => {
    setLivePlaying(false)
    setToast({ text: 'The clip could not be played', entry: null })
  }, [])
  useEffect(() => {
    return () => {
      seekCtl.current?.abort()
      seekCtl.current = null
      setSeekFace(null)
    }
  }, [uid, detailVersion])

  const showToast = useCallback((msg: string) => setToast({ text: msg, entry: claimForToast(msg) }), [])
  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), toast.entry ? 7000 : 2800)
    return () => clearTimeout(t)
  }, [toast])

  const editBox = useCallback(
    (t: BoxTarget, anchor: HTMLElement, group = false, pick = false) =>
      setEditing((cur) =>
        // a second tap on the same box closes its editor; "Someone else" (pick) always opens it at the name field
        !pick && cur && cur.kind === t.kind && cur.id === t.box.id ? null : { kind: t.kind, id: t.box.id, anchor, group, pick },
      ),
    [],
  )
  const guessActions = useGuessActions(uid, showToast)
  const guesses = useMemo(
    () => ({
      confirm: (t: BoxTarget) => t.kind !== 'animal' && t.box.guess && guessActions.confirm(t.kind, t.box.id, t.box.guess),
      reject: (t: BoxTarget) => t.kind !== 'animal' && t.box.guess && guessActions.reject(t.kind, t.box.id, t.box.guess),
      // "Someone else": the box's editor at its name field; the pick there corrects the guess in one step
      other: (t: BoxTarget, anchor: HTMLElement) => editBox(t, anchor, false, true),
      pendingKey: guessActions.pendingKey,
    }),
    [guessActions, editBox],
  )

  const fps = detail?.fps ?? null
  const seekTo = useCallback(
    (t: BoxTarget) => {
      const video = videoRef.current
      if (!video || t.box.frame_time_s === null) return
      seekCtl.current?.abort()
      const ctl = new AbortController()
      seekCtl.current = ctl
      const key = boxKey(t)
      setSeekFace({ key, shown: false })
      seekToFrame(video, t.box.frame_time_s, fps, ctl.signal).then(
        () => {
          if (ctl.signal.aborted) return
          setSeekFace(video.paused ? { key, shown: true } : null)
        },
        () => {
          if (!ctl.signal.aborted) setSeekFace(null)
        },
      )
    },
    [fps],
  )
  const videoUrl = detail?.kind === 'video' ? detail.urls.video : null
  const onScreen = useOnScreenFaces(videoRef, detail?.kind === 'video' ? detail.faces : null, videoUrl ? `${uid}:${detailVersion}:${videoUrl}` : null)
  useEffect(() => {
    if (spotlight === null) return
    const timer = window.setTimeout(() => setSpotlight(null), SPOTLIGHT_MS)
    return () => window.clearTimeout(timer)
  }, [spotlight])
  useEffect(() => setSpotlight(null), [uid])
  useEffect(() => {
    const video = videoRef.current
    if (startAt === null || !videoUrl || !video) return
    const ctl = new AbortController()
    seekToFrame(video, startAt, fps, ctl.signal).catch(() => {})
    return () => ctl.abort()
  }, [uid, detailVersion, videoUrl, startAt, fps])
  const dismissSeekFace = useCallback(() => {
    seekCtl.current?.abort()
    seekCtl.current = null
    setSeekFace(null)
  }, [])
  useEffect(() => {
    if (!showFaces) dismissSeekFace()
  }, [showFaces, dismissSeekFace])

  const editingTarget = editing && detail ? findBox(detail, editing.kind, editing.id) : null
  // whole-video editing (section 10): the tracks of the group the chip was opened from
  const editingTracks = useMemo(() => {
    if (!editing?.group || !detail || !editingTarget || detail.kind !== 'video') return undefined
    const key = boxKey(editingTarget)
    return groupVideoBoxes(videoBoxes(detail)).find((g) => g.targets.some((t) => boxKey(t) === key))?.targets
  }, [editing, detail, editingTarget])
  useEffect(() => {
    if (editing && detail && !editingTarget) setEditing(null)
  }, [editing, detail, editingTarget])
  const adjustTarget = adjusting && detail ? findBox(detail, adjusting.kind, adjusting.id) : null
  useEffect(() => {
    if (adjusting && detail && !adjustTarget) setAdjusting(null)
  }, [adjusting, detail, adjustTarget])
  // Adjust mode ends when its box cannot show any more: boxes hidden, a Live Photo playing, drawing started.
  useEffect(() => {
    if (adjusting && (!showFaces || livePlaying || drawing)) setAdjusting(null)
  }, [adjusting, showFaces, livePlaying, drawing])

  const poster = posterTime(detail?.duration_s)
  const frameTime = useCallback(() => currentFrameTime(videoRef.current, frameTimeRef.current, poster), [poster])

  const toggleFaces = useCallback(() => {
    setShowFaces((v) => {
      rememberFaces = !v
      return !v
    })
  }, [])
  const toggleInfo = useCallback(() => {
    setShowInfo((v) => {
      rememberInfo = !v
      return !v
    })
  }, [])
  /** Draw mode for a pet box (section 22) or a person box (section 10): the boxes show, the editor and the
   *  tagging form close, and on a phone the details sheet goes down so the photo is in reach. */
  const startDrawing = useCallback((kind: DrawKind) => {
    setEditing(null)
    setAdjusting(null)
    setTagging(false)
    setShowFaces(true)
    setDrawing(kind)
    if (typeof window !== 'undefined' && !window.matchMedia('(min-width: 1024px)').matches) setShowInfo(false)
  }, [])
  /** Adjust mode (section 10): the editor closes and the box gets handles; on a phone the details go down. */
  const moveBox = useMutation({
    mutationFn: async ({ t, box }: { t: BoxTarget; box: BoxIn }): Promise<unknown> =>
      t.kind === 'face' ? api.people.moveFace(t.box.id, box) : t.kind === 'body' ? api.people.moveBody(t.box.id, box) : api.animals.move(t.box.id, box),
    onSuccess: (_r, v) => {
      invalidate()
      setAdjusting(null)
      showToast(v.t.kind === 'face' ? 'Face box moved' : v.t.kind === 'body' ? 'Person box moved' : 'Pet box moved')
    },
  })
  const startAdjust = useCallback(
    (t: BoxTarget) => {
      moveBox.reset()
      setEditing(null)
      setTagging(false)
      setDrawing(null)
      setSpotlight(null)
      setLivePlaying(false)
      setShowFaces(true)
      setAdjusting({ kind: t.kind, id: t.box.id })
      // a video's box belongs to its frame: the player goes there and stays paused (VideoStage)
      if (detail?.kind === 'video') {
        videoRef.current?.pause()
        seekTo(t)
      }
      if (typeof window !== 'undefined' && !window.matchMedia('(min-width: 1024px)').matches) setShowInfo(false)
    },
    [moveBox, detail?.kind, seekTo],
  )
  // After adjusting a video's box, the focus goes back to its chip under the video (a photo's stage does its own).
  const adjustedKey = useRef<string | null>(null)
  useEffect(() => {
    if (adjusting) adjustedKey.current = `${adjusting.kind}:${adjusting.id}`
    else if (adjustedKey.current) {
      const key = adjustedKey.current
      adjustedKey.current = null
      if (detail?.kind === 'video') {
        requestAnimationFrame(() => rootRef.current?.querySelector<HTMLElement>(`[data-box-key="${key}"]`)?.focus({ preventScroll: true }))
      }
    }
  }, [adjusting, detail?.kind])
  const saveAdjust = (box: BoxIn) => {
    if (!adjustTarget || moveBox.isPending) return
    const b = adjustTarget.box
    if (Math.abs(box.x - b.x) + Math.abs(box.y - b.y) + Math.abs(box.w - b.w) + Math.abs(box.h - b.h) < 1e-4) {
      setAdjusting(null) // nothing moved
      return
    }
    moveBox.mutate({ t: adjustTarget, box })
  }
  const adjustError = moveBox.error
    ? moveBox.error instanceof ApiError && moveBox.error.code === 'no_face'
      ? 'No face there. For someone whose face does not show, use Add a person box instead.'
      : errorMessage(moveBox.error, 'Could not move the box')
    : null
  const addPetBox = useCallback(() => startDrawing('pet'), [startDrawing])
  const addFaceBox = useCallback(() => startDrawing('face'), [startDrawing])
  const addPersonBox = useCallback(() => startDrawing('person'), [startDrawing])
  /** A tap on a person's badge in the details (section 14): on a video, the same as their chip under the video
   *  (the first of their tracks, or with `now` the one on screen); on a photo, their boxes light up for a moment.
   *  The boxes show, and on a phone the details sheet goes down so the moment is in view. */
  /** On a phone the details sheet goes down so the stage is in view; the badge goes away with the sheet, so the
   *  focus waits on the details button, which brings it back. */
  const revealStage = useCallback(() => {
    if (typeof window !== 'undefined' && !window.matchMedia('(min-width: 1024px)').matches) {
      setShowInfo(false)
      requestAnimationFrame(() => infoToggleRef.current?.focus({ preventScroll: true }))
    }
  }, [])
  const showPerson = useCallback(
    (personId: number, now = false) => {
      if (!detail) return
      setEditing(null)
      setAdjusting(null)
      setShowFaces(true)
      if (detail.kind === 'video') {
        const mine = videoBoxes(detail).filter((t) => t.box.person?.id === personId)
        const pick = (now && onScreen ? mine.find((t) => t.kind === 'face' && onScreen.has(t.box.id)) : undefined) ?? mine[0]
        if (pick) seekTo(pick)
      } else {
        setLivePlaying(false) // boxes are not drawn over a playing Live Photo clip
        setSpotlight({ person: personId, key: null, at: Date.now() })
      }
      revealStage()
    },
    [detail, onScreen, seekTo, revealStage],
  )
  /** A tap on a guess in the details (section 14): on a video, the frame of that box, picked as its chip under the
   *  video picks it; on a photo, the box lights up for a moment. */
  const showBox = useCallback(
    (t: BoxTarget) => {
      if (!detail) return
      setEditing(null)
      setAdjusting(null)
      setShowFaces(true)
      if (detail.kind === 'video') seekTo(t)
      else {
        setLivePlaying(false)
        setSpotlight({ person: null, key: boxKey(t), at: Date.now() })
      }
      revealStage()
    },
    [detail, seekTo, revealStage],
  )
  // The person the stage shows now: the video's picked track, the photo's highlighted boxes.
  const activePersonId = useMemo(() => {
    if (!detail) return null
    if (detail.kind !== 'video') return spotlight?.person ?? null
    if (!seekFace) return null
    return videoBoxes(detail).find((t) => boxKey(t) === seekFace.key)?.box.person?.id ?? null
  }, [detail, seekFace, spotlight])
  const spotlightKeys = useMemo(() => {
    if (!detail || spotlight === null) return new Set<string>()
    if (spotlight.key !== null) return new Set([spotlight.key])
    return new Set(stillBoxes(detail).filter((t) => t.box.person?.id === spotlight.person).map(boxKey))
  }, [detail, spotlight])

  const closeEditor = useCallback(() => {
    setEditing((cur) => {
      if (cur?.anchor.isConnected) setTimeout(() => cur.anchor.focus({ preventScroll: true }), 0)
      return null
    })
  }, [])

  useEffect(() => {
    const prevFocus = document.activeElement as HTMLElement | null
    const unlock = lockScroll()
    rootRef.current?.focus({ preventScroll: true })
    return () => {
      unlock()
      if (prevFocus?.isConnected) prevFocus.focus({ preventScroll: true })
    }
  }, [])

  const favorite = useMutation({
    mutationFn: (value: boolean) => api.items.update(uid, { favorite: value }),
    onSuccess: (_d, value) => {
      invalidate()
      showToast(value ? 'Added to favourites' : 'Removed from favourites')
    },
    onError: (e) => showToast(errorMessage(e, 'Could not save the change')),
  })

  const isDocKind = detail?.kind === 'pdf' || detail?.kind === 'document'
  const pageCount = detail?.urls.pages.length ?? 0

  // Keyboard: Esc, arrows, i (info), f (faces), s (favourite), [ and ] for pages, and a focus trap.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (confirmDelete || sharing) {
        // The confirm and share dialogs (portaled after the viewer) have no Tab trap of their own: keep focus in them.
        if (e.key !== 'Tab') return
        const dialogs = Array.from(document.querySelectorAll<HTMLElement>('[role="dialog"][aria-modal="true"]')).filter(
          (d) => d !== rootRef.current,
        )
        const panel = dialogs[dialogs.length - 1]
        if (!panel) return
        const nodes = Array.from(
          panel.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), input:not([disabled]), textarea, select, [tabindex]:not([tabindex="-1"])'),
        ).filter((n) => n.offsetParent !== null)
        const active = document.activeElement
        if (!nodes.length) {
          e.preventDefault()
          panel.focus()
          return
        }
        const first = nodes[0]
        const last = nodes[nodes.length - 1]
        if (!panel.contains(active)) {
          e.preventDefault()
          ;(e.shiftKey ? last : first).focus()
        } else if (e.shiftKey && (active === first || active === panel)) {
          e.preventDefault()
          last.focus()
        } else if (!e.shiftKey && active === last) {
          e.preventDefault()
          first.focus()
        }
        return
      }
      if (drawing) {
        if (e.key === 'Escape') {
          e.preventDefault()
          setDrawing(null)
        }
        return
      }
      if (editing) {
        if (e.key === 'Escape') {
          e.preventDefault()
          closeEditor()
        }
        return
      }
      if (e.key === 'Tab') {
        const root = rootRef.current
        if (!root) return
        const nodes = Array.from(
          root.querySelectorAll<HTMLElement>('a[href], button:not([disabled]), input, video[controls], [tabindex]:not([tabindex="-1"])'),
        ).filter((n) => n.offsetParent !== null || n === document.activeElement)
        if (!nodes.length) return
        const first = nodes[0]
        const last = nodes[nodes.length - 1]
        const active = document.activeElement
        if (!root.contains(active)) {
          e.preventDefault()
          first.focus()
        } else if (e.shiftKey && (active === first || active === root)) {
          e.preventDefault()
          last.focus()
        } else if (!e.shiftKey && active === last) {
          e.preventDefault()
          first.focus()
        }
        return
      }
      // The box adjuster takes its own keys (arrows, Enter, Escape); nothing else while it is open.
      if (adjusting) return
      if (e.key === 'Escape') {
        if (e.defaultPrevented) return
        e.preventDefault()
        if (livePlaying) setLivePlaying(false)
        else if (tagging) setTagging(false)
        else onClose()
        return
      }
      if (isTyping(e.target) || e.altKey || e.ctrlKey || e.metaKey) return
      const onVideo = e.target instanceof HTMLVideoElement
      if (e.key === 'ArrowLeft' && onPrev && !onVideo) {
        e.preventDefault()
        onPrev()
      } else if (e.key === 'ArrowRight' && onNext && !onVideo) {
        e.preventDefault()
        onNext()
      } else if (e.key === 'i') {
        toggleInfo()
      } else if (e.key === 'f') {
        toggleFaces()
      } else if (e.key === 's' && detail && !favorite.isPending) {
        favorite.mutate(!detail.favorite)
      } else if (isDocKind && (e.key === '[' || e.key === 'PageUp')) {
        e.preventDefault()
        setPage((p) => Math.max(0, p - 1))
      } else if (isDocKind && (e.key === ']' || e.key === 'PageDown')) {
        e.preventDefault()
        setPage((p) => Math.min(Math.max(pageCount - 1, 0), p + 1))
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [confirmDelete, sharing, editing, drawing, adjusting, tagging, livePlaying, onClose, onPrev, onNext, closeEditor, toggleInfo, toggleFaces, detail, favorite, isDocKind, pageCount])

  const onTouchStart = (e: TouchEvent) => {
    const t = e.touches[0]
    const target = e.target as HTMLElement
    let ignore = (window.visualViewport?.scale ?? 1) > 1.01 || !!target.closest('[data-no-swipe]')
    const video = target.closest('video')
    if (video) {
      const r = video.getBoundingClientRect()
      if (t.clientY > r.top + r.height * 0.7) ignore = true
    }
    touch.current = { x: t.clientX, y: t.clientY, multi: e.touches.length > 1, ignore }
  }
  const onTouchMove = (e: TouchEvent) => {
    if (touch.current && e.touches.length > 1) touch.current.multi = true
  }
  const onTouchEnd = (e: TouchEvent) => {
    const s = touch.current
    touch.current = null
    if (!s || s.multi || s.ignore) return
    const t = e.changedTouches[0]
    const dx = t.clientX - s.x
    const dy = t.clientY - s.y
    if (Math.abs(dx) < 50 || Math.abs(dx) < Math.abs(dy) * 1.5) return
    if (dx < 0) onNext?.()
    else onPrev?.()
  }

  const del = useMutation({
    mutationFn: () => api.items.remove(uid),
    onSuccess: () => {
      setConfirmDelete(false)
      setExitAfter(onNext ? 'next' : 'close')
    },
  })
  const leave = useCallback(() => setExitAfter(onNext ? 'next' : 'close'), [onNext])
  // Leave only after the confirm dialog has unmounted, so its scroll lock is released first.
  useEffect(() => {
    if (confirmDelete || !exitAfter) return
    const how = exitAfter
    setExitAfter(null)
    if (how === 'next' && onNext) onNext()
    else onClose()
    qc.removeQueries({ queryKey: ['item', uid], exact: false })
    invalidate()
  }, [confirmDelete, exitAfter, onNext, onClose, qc, uid, invalidate])

  const kind = detail?.kind ?? placeholder?.kind
  const isVideo = kind === 'video'
  const boxCount = detail ? detail.faces.length + detail.bodies.filter((b) => b.face_id === null).length + detail.animals.length : 0
  const errCode = detailQ.error instanceof ApiError ? detailQ.error : null
  const notFound = errCode?.status === 404
  const locked = hiddenPrivate || errCode?.code === 'vault_locked'

  const stage = (() => {
    if (detail) {
      if (detail.kind === 'video') {
        if (!detail.urls.video) return <VideoPreparing detail={detail} />
        const shown =
          seekFace?.shown && showFaces && !adjusting ? (videoBoxes(detail).find((t) => boxKey(t) === seekFace.key) ?? null) : null
        return (
          <VideoStage
            key={detail.uid + ':' + detail.version}
            detail={detail}
            src={detail.urls.video}
            videoRef={videoRef}
            face={shown}
            onDismissFace={dismissSeekFace}
            frameTimeRef={frameTimeRef}
            drawing={drawing}
            onDrawEnd={() => setDrawing(null)}
            frameTime={frameTime}
            onToast={showToast}
            adjusting={adjustTarget}
            adjustSaving={moveBox.isPending}
            adjustError={adjustError}
            onAdjustSave={saveAdjust}
            onAdjustCancel={() => setAdjusting(null)}
          />
        )
      }
      if (detail.kind === 'pdf' || detail.kind === 'document') {
        return <DocumentStage key={detail.uid + ':' + detail.version} detail={detail} page={page} setPage={setPage} />
      }
      if (detail.kind === 'other') return <FileCard detail={detail} />
      return (
        <PhotoStage
          key={detail.uid + ':' + detail.version}
          detail={detail}
          showFaces={showFaces}
          editingKey={editing ? `${editing.kind}:${editing.id}` : null}
          spotlightKeys={spotlightKeys}
          onEditBox={editBox}
          guessPendingKey={guesses.pendingKey}
          onConfirmGuess={guesses.confirm}
          onRejectGuess={guesses.reject}
          onOtherGuess={guesses.other}
          liveSrc={livePlaying ? liveUrl : null}
          onLiveStop={stopLive}
          onLiveError={onLiveError}
          adjusting={adjustTarget}
          adjustSaving={moveBox.isPending}
          adjustError={adjustError}
          onAdjustSave={saveAdjust}
          onAdjustCancel={() => setAdjusting(null)}
          drawing={drawing}
          onDrawEnd={() => setDrawing(null)}
          onToast={showToast}
        />
      )
    }
    if (detailQ.isError || hiddenPrivate) {
      return (
        <div className="flex max-w-sm flex-col items-center gap-4 px-4 text-center text-white">
          <p className="text-base font-semibold">
            {locked ? 'The vault is locked' : notFound ? 'This item is no longer available' : 'Could not load this item'}
          </p>
          <p className="text-sm text-white/65">
            {locked
              ? 'Unlock the vault again to see private items.'
              : notFound
                ? 'It may have been moved to the trash or into the vault.'
                : errorMessage(detailQ.error)}
          </p>
          <div className="flex gap-2">
            {!notFound && !locked && (
              <button type="button" className={darkButton('primary')} onClick={() => void detailQ.refetch()}>
                Try again
              </button>
            )}
            <button type="button" className={darkButton()} onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      )
    }
    const pw = placeholder?.width || 4
    const ph = placeholder?.height || 3
    return (
      <div
        className="relative flex items-center justify-center bg-contain bg-center bg-no-repeat"
        style={
          placeholder
            ? {
                aspectRatio: `${pw} / ${ph}`,
                width: `min(100cqw, calc(100cqh * ${pw / ph}))`,
                backgroundImage: placeholder.nsfw && vaultClosed ? undefined : `url("${withVault(placeholder.urls.thumb, vault)}")`,
              }
            : undefined
        }
        role="status"
      >
        <Loader2 className="size-8 animate-spin text-white/80" aria-hidden />
        <span className="sr-only">Loading</span>
      </div>
    )
  })()

  const navButton =
    'absolute top-1/2 z-10 flex size-11 -translate-y-1/2 items-center justify-center rounded-full bg-black/45 text-white backdrop-blur-sm transition-colors hover:bg-black/70 sm:size-12'
  const trashDays = config.data?.trash_days ?? 30
  const noun = isVideo ? 'video' : kind === 'image' ? 'photo' : 'file'

  return createPortal(
    <div
      ref={rootRef}
      role="dialog"
      aria-modal="true"
      data-undo-keys=""
      aria-label={isVideo ? 'Video viewer' : isDocKind ? 'Document viewer' : 'Photo viewer'}
      tabIndex={-1}
      className="fixed inset-0 z-40 flex flex-col bg-[#111] text-white outline-none"
    >
      <div className="flex h-14 shrink-0 items-center gap-1 px-1.5 sm:px-3">
        <button type="button" onClick={onClose} className={iconButton} aria-label="Close viewer" title="Close (Esc)">
          <X className="size-5" aria-hidden />
        </button>
        {position && position.total > 0 && (
          <span className="ml-1 text-sm tabular-nums text-white/75" aria-label={`Item ${position.index + 1} of ${position.total}`}>
            {(position.index + 1).toLocaleString()} / {position.total.toLocaleString()}
          </span>
        )}
        {detail && (
          <span className="ml-2 hidden min-w-0 truncate text-sm text-white/60 md:inline" title={detail.filename}>
            {detail.filename}
          </span>
        )}
        <div className="ml-auto flex items-center gap-0.5 sm:gap-1">
          {detail && (
            <button
              type="button"
              onClick={() => favorite.mutate(!detail.favorite)}
              disabled={favorite.isPending}
              aria-pressed={detail.favorite}
              className={iconButton}
              aria-label={detail.favorite ? 'Remove from favourites' : 'Add to favourites'}
              title="Favourite (S)"
            >
              <Star className={clsx('size-5', detail.favorite && 'fill-amber-300 text-amber-300')} aria-hidden />
            </button>
          )}
          {boxCount > 0 && (
            <button
              type="button"
              onClick={toggleFaces}
              aria-pressed={showFaces}
              className={iconButton}
              aria-label={isVideo ? 'Show people and pets in this video' : 'Show face, person and pet boxes'}
              title="Faces (F)"
            >
              <ScanFace className="size-5" aria-hidden />
            </button>
          )}
          <button ref={infoToggleRef} type="button" onClick={toggleInfo} aria-pressed={showInfo} className={iconButton} aria-label="Show details" title="Details (I)">
            <Info className="size-5" aria-hidden />
          </button>
          {detail && !detail.nsfw && !vault && <ShareButton uid={detail.uid} onOpenChange={setSharing} />}
          {detail && (
            <DownloadMenu
              detail={detail}
              vault={vault}
              frameTime={detail.kind === 'video' ? frameTime : undefined}
              pageUrl={isDocKind && pageCount ? detail.urls.pages[Math.min(page, pageCount - 1)] : null}
            />
          )}
        </div>
      </div>

      <div className="relative flex min-h-0 flex-1">
        <div className="relative flex min-w-0 flex-1 flex-col">
          <div
            className="relative flex min-h-0 flex-1 items-center justify-center p-2 sm:p-4"
            style={{ containerType: 'size' }}
            onTouchStart={onTouchStart}
            onTouchMove={onTouchMove}
            onTouchEnd={onTouchEnd}
          >
            {stage}
            {liveUrl && <LiveButton playing={livePlaying} onChange={setLivePlaying} className="absolute left-3 top-3 z-10 sm:left-5 sm:top-5" />}
          </div>
          {onPrev && !adjusting && (
            <button type="button" onClick={onPrev} className={clsx(navButton, 'left-2 sm:left-4')} aria-label="Previous" title="Previous">
              <ChevronLeft className="size-6" aria-hidden />
            </button>
          )}
          {onNext && !adjusting && (
            <button type="button" onClick={onNext} className={clsx(navButton, 'right-2 sm:right-4')} aria-label="Next" title="Next">
              <ChevronRight className="size-6" aria-hidden />
            </button>
          )}
          {detail?.kind === 'video' && showFaces && !adjusting && (
            <VideoChips
              targets={videoBoxes(detail)}
              editingKey={editing ? `${editing.kind}:${editing.id}` : null}
              activeKey={seekFace?.key ?? null}
              onSeek={seekTo}
              onEdit={editBox}
            />
          )}
        </div>

        {showInfo && detail && (
          <>
            <button
              type="button"
              aria-label="Close details"
              tabIndex={-1}
              onClick={toggleInfo}
              className="absolute inset-0 z-10 bg-black/30 lg:hidden"
            />
            <aside
              data-no-swipe
              aria-label="Details"
              className={clsx(
                'absolute inset-x-0 bottom-0 z-20 max-h-[70dvh] overflow-y-auto overscroll-contain rounded-t-2xl bg-[#1b1b1b] shadow-2xl',
                'pb-[env(safe-area-inset-bottom)]',
                'lg:static lg:z-auto lg:max-h-none lg:w-[24rem] lg:shrink-0 lg:rounded-none lg:border-l lg:border-white/10 lg:bg-[#171717] lg:shadow-none',
              )}
            >
              <div className="mx-auto mt-2 h-1 w-10 rounded-full bg-white/20 lg:hidden" aria-hidden />
              <InfoPanel
                detail={detail}
                vault={vault}
                onScreen={onScreen}
                onClose={toggleInfo}
                onToast={showToast}
                onRequestDelete={() => setConfirmDelete(true)}
                onLeave={leave}
                tagging={tagging}
                setTagging={setTagging}
                guesses={guesses}
                onEditBox={editBox}
                onAddPetBox={addPetBox}
                onAddPersonBox={addPersonBox}
                onAddFaceBox={addFaceBox}
                onShowPerson={showPerson}
                onShowBox={showBox}
                activePersonId={activePersonId}
                documentFirst={mode === 'document'}
              />
            </aside>
          </>
        )}
      </div>

      {editing && editingTarget && (
        editingTarget.kind === 'animal' ? (
          <AnimalEditor
            uid={uid}
            box={editingTarget.box}
            tracks={editingTracks?.filter((t): t is { kind: 'animal'; box: AnimalBox } => t.kind === 'animal').map((t) => t.box)}
            anchor={editing.anchor}
            onClose={closeEditor}
            onDone={(msg) => {
              showToast(msg)
              closeEditor()
            }}
            onAdjust={detail?.kind === 'image' || detail?.kind === 'video' ? () => startAdjust(editingTarget) : undefined}
          />
        ) : (
          <FaceEditor
            uid={uid}
            target={editingTarget}
            tracks={editingTracks?.filter((t): t is PersonBoxTarget => t.kind !== 'animal')}
            anchor={editing.anchor}
            pickFirst={!!editing.pick}
            onClose={closeEditor}
            onDone={(msg) => {
              showToast(msg)
              closeEditor()
            }}
            onAdjust={detail?.kind === 'image' || detail?.kind === 'video' ? () => startAdjust(editingTarget) : undefined}
          />
        )
      )}

      <div aria-live="polite" className="pointer-events-none absolute inset-x-0 bottom-6 z-50 flex justify-center px-4">
        {toast && (
          <span
            className={clsx(
              'flex max-w-full items-center gap-1 rounded-full bg-white text-sm font-semibold text-[#111] shadow-lg',
              toast.entry ? 'py-1 pl-4 pr-1' : 'px-4 py-2',
            )}
          >
            <span className="min-w-0">{toast.text}</span>
            {toast.entry && (
              <button
                type="button"
                className="pointer-events-auto shrink-0 rounded-full px-3 py-1 text-[#0a58ca] hover:bg-black/5 focus-visible:outline-2 focus-visible:outline-[#0a58ca]"
                onClick={() => {
                  const entry = toast.entry
                  setToast(null)
                  if (entry) void appToast.undo(entry)
                }}
              >
                Undo
              </button>
            )}
          </span>
        )}
      </div>

      <ConfirmModal
        open={confirmDelete}
        title={`Move this ${noun} to the trash?`}
        body={
          <>
            <p>
              It leaves the library now and is deleted for good after {trashDays} days. Until then you can restore it from Trash with
              its tags and document details.
            </p>
            {del.error && (
              <p className="mt-3 text-danger" role="alert">
                {errorMessage(del.error, 'Could not move it to the trash. Try again.')}
              </p>
            )}
          </>
        }
        confirmLabel="Move to trash"
        danger
        loading={del.isPending}
        onConfirm={() => del.mutate()}
        onClose={() => {
          if (!del.isPending) setConfirmDelete(false)
        }}
      />
    </div>,
    document.body,
  )
}
