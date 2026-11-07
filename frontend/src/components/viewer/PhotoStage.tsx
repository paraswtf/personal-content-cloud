// The picture with face and body boxes drawn in percent of the rendered image box. A box with a
// person shows the name; a box with a guess shows "Name?" with confirm and reject; a tap on a box
// opens the editor.
import { useState } from 'react'
import clsx from 'clsx'
import { Loader2 } from 'lucide-react'
import type { ItemDetail } from '../../lib/types'
import { personName } from '../../lib/format'
import { previewSrcSet } from '../../lib/preview'
import { boxKey, stillBoxes, type BoxTarget } from './boxes'
import GuessChip from './GuessChip'

interface Props {
  detail: ItemDetail
  showFaces: boolean
  editingKey: string | null
  onEditBox: (target: BoxTarget, anchor: HTMLElement) => void
  guessPendingKey: string | null
  onConfirmGuess: (target: BoxTarget) => void
  onRejectGuess: (target: BoxTarget) => void
  /** Use this image instead of the preview (a document page). */
  src?: string
}

export default function PhotoStage({
  detail,
  showFaces,
  editingKey,
  onEditBox,
  guessPendingKey,
  onConfirmGuess,
  onRejectGuess,
  src,
}: Props) {
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)

  const w = natural?.w || detail.width || 4
  const h = natural?.h || detail.height || 3
  const ratio = w / h
  const boxes = src ? [] : stillBoxes(detail)
  const names = detail.people.map((p) => personName(p)).join(', ')

  return (
    <div
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

      {showFaces &&
        loaded &&
        boxes.map((t) => {
          const b = t.box
          const key = boxKey(t)
          const isBody = t.kind === 'body'
          const label = b.person ? personName(b.person) : isBody ? 'Person, not tagged' : 'Not tagged'
          const below = b.y + b.h < 0.86
          const pos = { left: `${b.x * 100}%`, top: `${b.y * 100}%`, width: `${b.w * 100}%`, height: `${b.h * 100}%` }
          return (
            <div key={key} className="group absolute" style={pos}>
              <button
                type="button"
                className={clsx(
                  'absolute inset-0 rounded-md border-2 shadow-[0_0_0_1px_rgba(0,0,0,0.45)] transition-colors focus-visible:outline-none',
                  isBody && 'rounded-lg border-[1.5px]',
                  b.person ? (isBody ? 'border-sky-300/80' : 'border-white/90') : b.guess ? 'border-amber-300' : 'border-dashed border-amber-300/90',
                  editingKey === key ? 'border-accent ring-2 ring-accent' : 'hover:border-accent focus-visible:border-accent',
                )}
                aria-label={b.person ? `Edit ${isBody ? 'person' : 'face'} tagged as ${label}` : `Tag this ${isBody ? 'person' : 'face'}`}
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
                  />
                ) : (
                  <span
                    className={clsx(
                      'truncate rounded-full px-2 py-0.5 text-[11px] font-semibold',
                      b.person ? 'bg-black/70 text-white' : 'bg-amber-300 text-black',
                      b.person && b.person.name ? '' : 'opacity-0 group-hover:opacity-100 group-focus-within:opacity-100',
                    )}
                  >
                    {label}
                  </span>
                )}
              </span>
            </div>
          )
        })}
    </div>
  )
}
