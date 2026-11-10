// The picture with face and body boxes drawn in percent of the rendered image box. A box with a
// person shows the name; a box with a guess shows "Name?" with confirm and reject; a tap on a box
// opens the editor. A Live Photo's clip plays over the still, in the same box, while `liveSrc` is set.
// Draw mode: the owner drags a box (or takes the whole photo) over a pet the detector missed (section 22),
// picks the species, and the new box's popover opens so the pet can be named; or over a person (section 10),
// picks who it is in PersonBoxCard, and the box is added with them.
// Each name carries its origin (section 10): a person icon when the owner tagged or drew it, a bot when the
// system did. Adjust mode (section 10): one box shows BoxAdjuster to move or resize it, the others step back.
// Body traits (section 34): while a box's editor is open, its tattoo, nails and mehndi regions show as dashed violet
// rings (the person box's, for a face), drawn only, never in the way of a tap.
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import clsx from 'clsx'
import { Loader2, PawPrint } from 'lucide-react'
import { api } from '../../lib/api'
import { SPECIES, type AnimalOut, type ItemDetail, type Species } from '../../lib/types'
import { personName, speciesLabel } from '../../lib/format'
import { previewSrcSet } from '../../lib/preview'
import { ORIGIN_LABEL, boxOrigin } from '../../lib/origin'
import { errorMessage } from '../admin/adminHooks'
import OriginIcon from '../people/OriginIcon'
import BoxAdjuster from './BoxAdjuster'
import { adjustLabel, animalLabel, boxKey, stillBoxes, type BoxTarget, type DrawKind } from './boxes'
import GuessChip from './GuessChip'
import PersonBoxCard from './PersonBoxCard'
import { darkButton, useInvalidateItem } from './shared'

/** The smallest box the owner can draw, in screen pixels. */
const MIN_DRAW_PX = 24
/** "Whole photo": the frame minus this margin on every side. */
const WHOLE_MARGIN = 0.03

interface Props {
  detail: ItemDetail
  showFaces: boolean
  editingKey: string | null
  /** Boxes lit for a moment: the person whose badge was tapped in the details (section 14). */
  spotlightKeys?: ReadonlySet<string>
  onEditBox: (target: BoxTarget, anchor: HTMLElement) => void
  guessPendingKey: string | null
  /** The guess the viewer's keys answer (Review page), by boxKey. */
  guessSelectedKey?: string | null
  onConfirmGuess: (target: BoxTarget) => void
  onRejectGuess: (target: BoxTarget) => void
  /** "Someone else" on a guess: the box's editor at its name field. */
  onOtherGuess?: (target: BoxTarget, anchor: HTMLElement) => void
  /** Use this image instead of the preview (a document page). */
  src?: string
  /** Live Photo clip to play over the still, muted and looping; null or undefined shows the still. */
  liveSrc?: string | null
  /** A tap on the playing clip stops it. */
  onLiveStop?: () => void
  /** The clip could not be loaded or played. */
  onLiveError?: () => void
  /** Draw mode: a box for a pet the detector missed (section 22) or for a person (section 10); null when off. */
  drawing?: DrawKind | null
  /** Draw mode ended (cancelled, or the box was added). */
  onDrawEnd?: () => void
  onToast?: (message: string) => void
  /** Adjust mode (section 10): the box being moved or resized, with the save in progress and its error. */
  adjusting?: BoxTarget | null
  adjustSaving?: boolean
  adjustError?: string | null
  onAdjustSave?: (box: NormBox) => void
  onAdjustCancel?: () => void
}

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

/** The clip over the still. Mounted only while playing, so it never starts on its own. */
function LiveClip({ src, onStop, onError }: { src: string; onStop?: () => void; onError?: () => void }) {
  const ref = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)
  const onErrorRef = useRef(onError)
  useEffect(() => {
    onErrorRef.current = onError
  }, [onError])
  useEffect(() => {
    const v = ref.current
    if (!v) return
    // Set before play(): iOS Safari only allows inline playback of a muted video this way.
    v.muted = true
    v.defaultMuted = true
    v.play().catch((e: unknown) => {
      if (!(e instanceof DOMException && e.name === 'AbortError')) onErrorRef.current?.()
    })
  }, [src])
  return (
    <video
      ref={ref}
      src={src}
      muted
      loop
      playsInline
      disablePictureInPicture
      preload="auto"
      aria-label="Live Photo clip, playing"
      onPlaying={() => setPlaying(true)}
      onError={() => onErrorRef.current?.()}
      onClick={onStop}
      className={clsx(
        'absolute inset-0 block size-full bg-black object-contain transition-opacity duration-200',
        playing ? 'opacity-100' : 'opacity-0',
      )}
    />
  )
}

export default function PhotoStage({
  detail,
  showFaces,
  editingKey,
  spotlightKeys,
  onEditBox,
  guessPendingKey,
  guessSelectedKey = null,
  onConfirmGuess,
  onRejectGuess,
  onOtherGuess,
  src,
  liveSrc,
  onLiveStop,
  onLiveError,
  drawing = null,
  onDrawEnd,
  onToast,
  adjusting = null,
  adjustSaving = false,
  adjustError = null,
  onAdjustSave,
  onAdjustCancel,
}: Props) {
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const boxRefs = useRef(new Map<string, HTMLButtonElement>())
  const [drag, setDrag] = useState<Drag | null>(null)
  const [pending, setPending] = useState<NormBox | null>(null)
  // the box just added: its popover opens once the fresh detail draws it
  const [pendingEdit, setPendingEdit] = useState<number | null>(null)
  // the drawn person box on screen: PersonBoxCard sits next to it
  const [pendingEl, setPendingEl] = useState<HTMLDivElement | null>(null)
  const invalidate = useInvalidateItem(detail.uid)
  const person = drawing === 'person'
  const faceBox = drawing === 'face'
  // a person or a face box is named in PersonBoxCard; a pet box gets its species
  const named = person || faceBox
  const what = faceBox ? 'face' : person ? 'person' : 'pet'

  useEffect(() => {
    // off, or switched between a pet and a person: start over
    setDrag(null)
    setPending(null)
  }, [drawing])

  const add = useMutation({
    mutationFn: (species: Species) => api.items.addAnimal(detail.uid, { ...(pending as NormBox), species }),
    onSuccess: (box: AnimalOut) => {
      setPending(null)
      setPendingEdit(box.id)
      invalidate()
      onDrawEnd?.()
    },
    onError: (e) => onToast?.(errorMessage(e, 'Could not add the box')),
  })

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
    setPending({ x: x0 / p.cw, y: y0 / p.ch, w: wpx / p.cw, h: hpx / p.ch })
  }

  const w = natural?.w || detail.width || 4
  const h = natural?.h || detail.height || 3
  const ratio = w / h
  const boxes = src ? [] : stillBoxes(detail)
  const names = detail.people.map((p) => personName(p)).join(', ')

  useEffect(() => {
    if (pendingEdit === null) return
    const t = boxes.find((b) => b.kind === 'animal' && b.box.id === pendingEdit)
    const el = t ? boxRefs.current.get(boxKey(t)) : undefined
    if (t && el) {
      setPendingEdit(null)
      onEditBox(t, el)
    }
  })

  const dragRect = drag
    ? { left: Math.min(drag.x0, drag.x1), top: Math.min(drag.y0, drag.y1), width: Math.abs(drag.x1 - drag.x0), height: Math.abs(drag.y1 - drag.y0) }
    : null

  // After adjusting, the focus goes back to the box's own button.
  const adjustKey = adjusting ? boxKey(adjusting) : null
  const lastAdjust = useRef<string | null>(null)
  useEffect(() => {
    if (adjustKey) lastAdjust.current = adjustKey
    else if (lastAdjust.current) {
      const key = lastAdjust.current
      lastAdjust.current = null
      requestAnimationFrame(() => boxRefs.current.get(key)?.focus({ preventScroll: true }))
    }
  }, [adjustKey])

  // The trait regions of the box being edited (a face shows its person box's), not while a box is moved.
  const traitRings = (() => {
    if (!editingKey || adjustKey || drawing) return []
    const [kind, raw] = editingKey.split(':')
    const id = Number(raw)
    const body = kind === 'body' ? detail.bodies.find((b) => b.id === id) : kind === 'face' ? detail.bodies.find((b) => b.face_id === id) : undefined
    return (body?.traits ?? []).filter((t) => !t.hidden)
  })()

  return (
    <div
      ref={rootRef}
      className="relative bg-center bg-no-repeat"
      style={{
        aspectRatio: `${w} / ${h}`,
        width: `min(100cqw, calc(100cqh * ${ratio}))`,
        backgroundImage: loaded ? undefined : `url("${detail.urls.thumb}")`,
        backgroundSize: 'cover',
      }}
    >
      {!failed ? (
        <img
          src={src ?? detail.urls.preview}
          srcSet={src ? undefined : previewSrcSet(detail.urls, detail.width, detail.height)}
          sizes="100vw"
          alt={detail.people.length ? `Photo with ${names}` : detail.filename}
          draggable={false}
          onLoad={(e) => {
            const img = e.currentTarget
            setLoaded(true)
            if (img.naturalWidth && img.naturalHeight) {
              const r = img.naturalWidth / img.naturalHeight
              if (Math.abs(r - (detail.width || 1) / (detail.height || 1)) > 0.01) setNatural({ w: img.naturalWidth, h: img.naturalHeight })
            }
          }}
          onError={() => setFailed(true)}
          className={clsx('block size-full select-none object-contain transition-opacity duration-300', loaded ? 'opacity-100' : 'opacity-0')}
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60 p-4 text-center text-sm text-white/80">
          The preview could not be loaded. You can still download the original.
        </div>
      )}
      {!loaded && !failed && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20" role="status">
          <Loader2 className="size-7 animate-spin text-white/80" aria-hidden />
          <span className="sr-only">Loading</span>
        </div>
      )}

      {liveSrc && <LiveClip key={liveSrc} src={liveSrc} onStop={onLiveStop} onError={onLiveError} />}

      {showFaces &&
        loaded &&
        !liveSrc &&
        boxes.map((t) => {
          const b = t.box
          const key = boxKey(t)
          if (key === adjustKey) return null
          const isBody = t.kind === 'body'
          const isAnimal = t.kind === 'animal'
          const noun = isAnimal ? 'pet' : isBody ? 'person' : 'face'
          const label = isAnimal ? animalLabel(t.box) : b.person ? personName(b.person) : isBody ? 'Person, not tagged' : 'Not tagged'
          // a guess shows its own chip; an untagged box the detector found has no origin
          const origin = boxOrigin(b)
          const shownOrigin = origin === 'guess' ? null : origin
          const how =
            origin === 'auto'
              ? ' automatically'
              : origin === 'you'
                ? ' by you'
                : origin === 'drawn'
                  ? ', drawn by you'
                  : origin === 'confirmed'
                    ? ', guessed automatically and confirmed by you'
                    : ''
          const below = b.y + b.h < 0.86
          const pos = { left: `${b.x * 100}%`, top: `${b.y * 100}%`, width: `${b.w * 100}%`, height: `${b.h * 100}%` }
          return (
            <div key={key} className={clsx('group absolute', adjustKey && 'pointer-events-none opacity-40')} style={pos}>
              <button
                type="button"
                ref={(el) => {
                  if (el) boxRefs.current.set(key, el)
                  else boxRefs.current.delete(key)
                }}
                className={clsx(
                  'absolute inset-0 rounded-md border-2 shadow-[0_0_0_1px_rgba(0,0,0,0.45)] transition-colors focus-visible:outline-none',
                  (isBody || isAnimal) && 'rounded-lg border-[1.5px]',
                  b.person
                    ? isAnimal
                      ? 'border-emerald-300/90'
                      : isBody
                        ? 'border-sky-300/80'
                        : 'border-white/90'
                    : isAnimal
                      ? 'border-dashed border-emerald-300/90'
                      : b.guess
                        ? 'border-amber-300'
                        : 'border-dashed border-amber-300/90',
                  editingKey === key
                    ? 'border-accent ring-2 ring-accent'
                    : spotlightKeys?.has(key)
                      ? 'border-accent ring-4 ring-accent/70 motion-safe:animate-pulse'
                      : 'hover:border-accent focus-visible:border-accent',
                )}
                aria-label={b.person ? `Edit ${noun} tagged as ${label}${how}` : `Tag this ${noun}${origin === 'drawn' ? ', drawn by you' : ''}`}
                title={shownOrigin ? `${label}: ${ORIGIN_LABEL[shownOrigin].toLowerCase()}` : undefined}
                onClick={(e) => onEditBox(t, e.currentTarget)}
              />
              <span
                className={clsx(
                  'absolute left-1/2 flex max-w-[14rem] -translate-x-1/2 justify-center whitespace-nowrap',
                  below ? 'top-full mt-1' : 'bottom-full mb-1',
                  b.guess ? 'z-10' : 'pointer-events-none',
                )}
              >
                {b.guess && !b.person ? (
                  <GuessChip
                    compact
                    guess={b.guess}
                    busy={guessPendingKey === key}
                    onConfirm={() => onConfirmGuess(t)}
                    onReject={() => onRejectGuess(t)}
                    onOther={onOtherGuess ? (el) => onOtherGuess(t, el) : undefined}
                    guessKey={key}
                    selected={guessSelectedKey === key}
                  />
                ) : (
                  <span
                    className={clsx(
                      'flex min-w-0 items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold',
                      b.person || isAnimal ? 'bg-black/70 text-white' : 'bg-amber-300 text-black',
                      isAnimal || (b.person && b.person.name) ? '' : 'opacity-0 group-hover:opacity-100 group-focus-within:opacity-100',
                    )}
                  >
                    {isAnimal && <PawPrint className="size-3 shrink-0" aria-hidden />}
                    <span className="truncate">{label}</span>
                    {shownOrigin && <OriginIcon origin={shownOrigin} size="xs" decorative className="opacity-80" />}
                  </span>
                )}
              </span>
            </div>
          )
        })}

      {showFaces &&
        loaded &&
        !liveSrc &&
        traitRings.map((tr) => (
          <span
            key={`trait:${tr.id}`}
            aria-hidden
            className="pointer-events-none absolute rounded-md border-2 border-dashed border-violet-400 shadow-[0_0_0_1px_rgba(0,0,0,0.45)]"
            style={{ left: `${tr.x * 100}%`, top: `${tr.y * 100}%`, width: `${tr.w * 100}%`, height: `${tr.h * 100}%` }}
          />
        ))}

      {adjusting && showFaces && loaded && !liveSrc && (
        <BoxAdjuster
          key={adjustKey}
          start={{ x: adjusting.box.x, y: adjusting.box.y, w: adjusting.box.w, h: adjusting.box.h }}
          label={adjustLabel(adjusting)}
          saving={adjustSaving}
          error={adjustError}
          onSave={(box) => onAdjustSave?.(box)}
          onCancel={() => onAdjustCancel?.()}
        />
      )}

      {drawing && loaded && !pending && (
        <div
          data-no-swipe
          role="application"
          aria-label={`Draw a box over the ${what}`}
          className="absolute inset-0 z-20 cursor-crosshair touch-none select-none"
          onPointerDown={onDrawDown}
          onPointerMove={onDrawMove}
          onPointerUp={onDrawUp}
          onPointerCancel={() => setDrag(null)}
        >
          {dragRect && (
            <div
              className={clsx(
                'pointer-events-none absolute rounded-lg border-2 border-dashed',
                faceBox ? 'border-white bg-white/10' : person ? 'border-sky-300 bg-sky-300/10' : 'border-emerald-300 bg-emerald-300/10',
              )}
              style={dragRect}
            />
          )}
        </div>
      )}

      {drawing && (
        <div data-no-swipe className="pointer-events-none absolute inset-x-0 top-0 z-30 flex flex-wrap items-center justify-center gap-2 p-2">
          {!pending && (
            <span role="status" className="rounded-full bg-black/70 px-3 py-1.5 text-sm font-semibold text-white">
              Drag over the {what}
            </span>
          )}
          {!pending && !faceBox && (
            <button
              type="button"
              onClick={() => setPending({ x: WHOLE_MARGIN, y: WHOLE_MARGIN, w: 1 - 2 * WHOLE_MARGIN, h: 1 - 2 * WHOLE_MARGIN })}
              className={darkButton('default', 'pointer-events-auto bg-black/60')}
            >
              Whole photo
            </button>
          )}
          <button type="button" onClick={() => onDrawEnd?.()} className={darkButton('default', 'pointer-events-auto bg-black/60')}>
            Cancel
          </button>
        </div>
      )}

      {named && pending && (
        <>
          <div
            ref={setPendingEl}
            className={clsx(
              'pointer-events-none absolute z-20 rounded-lg border-2',
              faceBox ? 'border-white bg-white/10' : 'border-sky-300 bg-sky-300/10',
            )}
            style={{ left: `${pending.x * 100}%`, top: `${pending.y * 100}%`, width: `${pending.w * 100}%`, height: `${pending.h * 100}%` }}
          />
          {pendingEl && (
            <PersonBoxCard
              kind={faceBox ? 'face' : 'person'}
              uid={detail.uid}
              box={pending}
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
          <div
            className="pointer-events-none absolute z-20 rounded-lg border-2 border-emerald-300 bg-emerald-300/10"
            style={{ left: `${pending.x * 100}%`, top: `${pending.y * 100}%`, width: `${pending.w * 100}%`, height: `${pending.h * 100}%` }}
          />
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
                  onClick={() => add.mutate(sp)}
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
