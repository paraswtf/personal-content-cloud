// Two crops side by side for a position reason (section 35): this photo's box and the box the person was followed
// from, "This photo" and "Priya, 8 s earlier". The reference crop opens its photo when the caller can open one. A
// crop that no longer loads says "Photo changed" (TraitCompare's Crop).
import clsx from 'clsx'
import { Crop } from './TraitCompare'
import type { PlaceReason } from './evidence'

export default function PlaceCompare({
  reason,
  cropUrl,
  name,
  dark = false,
  onOpenRef,
  className,
}: {
  reason: PlaceReason
  /** The crop of the box the guess is for. */
  cropUrl?: string
  /** The guessed person's name, for the reference caption and the alt texts. */
  name: string
  dark?: boolean
  onOpenRef?: (uid: string) => void
  className?: string
}) {
  const refCaption = reason.when ? `${name}, ${reason.when}` : name
  const caption = clsx('mt-1 max-w-[72px] truncate text-[11px]', dark ? 'text-white/60' : 'text-ink-soft')
  const refUid = reason.refItemUid
  const ref = <Crop key={reason.refCropUrl} src={reason.refCropUrl} alt={`${name} in the other photo`} dark={dark} />
  return (
    <div className={clsx('flex items-start gap-2', className)}>
      <figure className="flex flex-col items-center">
        <Crop key={cropUrl} src={cropUrl} alt="The box in this photo" dark={dark} />
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
            aria-label={`Open the photo ${name} was followed from`}
          >
            {ref}
          </button>
        ) : (
          ref
        )}
        <figcaption className={caption} title={refCaption}>
          {refCaption}
        </figcaption>
      </figure>
    </div>
  )
}
