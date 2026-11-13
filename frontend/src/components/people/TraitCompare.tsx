// Two trait crops side by side (section 34): this photo's region and the one of the guessed person it matched,
// "This photo" and "Priya, 14 Mar 2026". The reference crop opens its photo when the caller can open one. A crop
// that no longer loads (the row was measured again, or the photo went to the vault or the trash) says "Photo
// changed". Nails show a swatch of the polish beside the crops; the colour is also in the reason text.
import { useState } from 'react'
import clsx from 'clsx'
import { formatDate } from '../../lib/format'
import { partWords, traitNoun, type TraitReason } from './evidence'

export function Crop({ src, alt, dark }: { src?: string; alt: string; dark: boolean }) {
  const [failed, setFailed] = useState(false)
  const box = clsx(
    'block size-14 shrink-0 overflow-hidden rounded-lg sm:size-[72px]',  // block: inside the reference button too
    dark ? 'bg-white/10 ring-1 ring-white/20' : 'bg-surface-2 ring-1 ring-line',
  )
  if (!src || failed)
    return (
      <span className={clsx(box, 'flex items-center justify-center p-1 text-center text-[10px] leading-tight', dark ? 'text-white/60' : 'text-ink-soft')}>
        Photo changed
      </span>
    )
  return (
    <span className={box}>
      <img src={src} alt={alt} loading="lazy" className="size-full object-cover" onError={() => setFailed(true)} />
    </span>
  )
}

export default function TraitCompare({
  reason,
  name,
  dark = false,
  onOpenRef,
  className,
}: {
  reason: TraitReason
  /** The guessed person's name, for the reference caption and the alt texts. */
  name: string
  /** Inside the viewer (always dark): white text and rings instead of the app tokens. */
  dark?: boolean
  /** Opens the photo the reference crop is from; without it the crop is a plain picture. */
  onOpenRef?: (uid: string) => void
  className?: string
}) {
  const what = `${traitNoun(reason.kind)} on the ${partWords(reason.part)}`
  const when = reason.refTakenAt ? formatDate(reason.refTakenAt) : ''
  const refCaption = when ? `${name}, ${when}` : name
  const caption = clsx('mt-1 max-w-[72px] truncate text-[11px]', dark ? 'text-white/60' : 'text-ink-soft')
  const refUid = reason.refItemUid
  return (
    <div className={clsx('flex items-start gap-2', className)}>
      <figure className="flex flex-col items-center">
        <Crop key={reason.cropUrl} src={reason.cropUrl} alt={`${what}, this photo`} dark={dark} />
        <figcaption className={caption}>This photo</figcaption>
      </figure>
      <figure className="flex flex-col items-center">
        {onOpenRef && refUid ? (
          <button
            type="button"
            onClick={() => onOpenRef(refUid)}
            className={clsx(
              'rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
              dark ? 'hover:opacity-85' : 'hover:opacity-90',
            )}
            aria-label={`Open ${name}'s photo${when ? ` from ${when}` : ''}`}
          >
            <Crop key={reason.refCropUrl} src={reason.refCropUrl} alt={`${what}, ${name}'s photo`} dark={dark} />
          </button>
        ) : (
          <Crop key={reason.refCropUrl} src={reason.refCropUrl} alt={`${what}, ${name}'s photo`} dark={dark} />
        )}
        <figcaption className={caption} title={refCaption}>
          {refCaption}
        </figcaption>
      </figure>
      {reason.kind === 'nails' && reason.hex && (
        <span
          aria-hidden
          className={clsx('mt-1 size-5 shrink-0 rounded-full', dark ? 'ring-1 ring-white/30' : 'ring-1 ring-line')}
          style={{ backgroundColor: reason.hex }}
        />
      )}
    </div>
  )
}
