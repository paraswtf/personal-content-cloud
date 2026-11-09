// Grid of face crops with an optional select mode (a person's faces, unassigned faces). Each tile shows who put
// the face there (section 10): a person icon when the owner tagged it, a bot when the system grouped or guessed it.
import clsx from 'clsx'
import { Check, Film, Lock } from 'lucide-react'
import type { FaceListItem } from '../../lib/types'
import { formatDuration, personName } from '../../lib/format'
import { useInfiniteSentinel } from '../../lib/hooks'
import { boxOrigin, originPhrase } from '../../lib/origin'
import { LoadMore } from '../admin/fields'
import OriginIcon from './OriginIcon'

export default function FaceGrid({
  faces,
  selecting,
  selected,
  onToggle,
  onOpen,
  hasMore,
  loadMore,
  loadingMore,
  coverId,
}: {
  faces: FaceListItem[]
  selecting: boolean
  selected: Set<number>
  onToggle: (id: number) => void
  onOpen: (face: FaceListItem) => void
  hasMore: boolean
  loadMore: () => void
  loadingMore: boolean
  /** Face used as the person's cover, marked on its tile. */
  coverId?: number | null
}) {
  const sentinel = useInfiniteSentinel(loadMore, hasMore && !loadingMore)
  return (
    <>
      <ul className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
        {faces.map((f) => {
          const isSel = selected.has(f.id)
          const origin = boxOrigin({
            person: f.person_id !== null ? { id: f.person_id } : null,
            assigned_by: f.assigned_by,
            source: f.source,
            by_guess: f.by_guess,
            guess: f.guess,
          })
          const label = `Face ${f.id}, ${f.size_px} pixels${f.item_kind === 'video' ? ', from a video' : ''}${
            f.guess
              ? `, maybe ${personName(f.guess.person)}`
              : origin
                ? `, ${originPhrase(origin)}`
                : f.assigned_by === 'user'
                  ? ', kept unassigned by you'
                  : ''
          }`
          return (
            <li key={f.id}>
              <button
                type="button"
                onClick={() => (selecting ? onToggle(f.id) : onOpen(f))}
                aria-label={selecting ? `${isSel ? 'Deselect' : 'Select'} ${label}` : `Open the item of ${label}`}
                aria-pressed={selecting ? isSel : undefined}
                className={clsx(
                  'group relative block aspect-square w-full overflow-hidden rounded-2xl bg-surface-2 ring-1 ring-line transition',
                  isSel ? 'ring-4 ring-accent' : 'hover:ring-ink-faint',
                )}
              >
                <img src={f.url} alt="" loading="lazy" decoding="async" className={clsx('size-full object-cover transition', isSel && 'scale-95 rounded-xl')} />
                {selecting && (
                  <span
                    className={clsx(
                      'absolute left-1.5 top-1.5 flex size-6 items-center justify-center rounded-full border-2 border-white shadow',
                      isSel ? 'bg-accent text-white' : 'bg-black/40 text-transparent',
                    )}
                    aria-hidden
                  >
                    <Check className="size-3.5" strokeWidth={3} />
                  </span>
                )}
                {coverId === f.id && (
                  <span className="absolute right-1.5 top-1.5 rounded-full bg-black/60 px-1.5 py-0.5 text-[10px] font-semibold text-white">Cover</span>
                )}
                <span className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-1 bg-gradient-to-t from-black/70 to-transparent px-1.5 pb-1 pt-4 text-[11px] font-medium text-white">
                  <span className="inline-flex min-w-0 items-center gap-1">
                    {f.item_kind === 'video' && <Film className="size-3 shrink-0" aria-hidden />}
                    {f.guess ? (
                      <span className="truncate">{personName(f.guess.person)}?</span>
                    ) : f.item_kind === 'video' && f.frame_time_s !== null ? (
                      formatDuration(f.frame_time_s)
                    ) : (
                      `${f.size_px} px`
                    )}
                  </span>
                  {origin ? (
                    <OriginIcon origin={origin} size="xs" decorative />
                  ) : (
                    f.assigned_by === 'user' && <Lock className="size-3 shrink-0" aria-hidden />
                  )}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
      <LoadMore hasMore={hasMore} loading={loadingMore} onClick={loadMore} sentinelRef={sentinel} />
    </>
  )
}
