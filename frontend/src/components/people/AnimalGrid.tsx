// Grid of animal crops with an optional select mode (a pet's crops). Each tile shows who put the crop there
// (section 10): a person icon when the owner tagged it or drew the box, a bot when the system matched it.
import clsx from 'clsx'
import { Check, Film } from 'lucide-react'
import type { AnimalListItem } from '../../lib/types'
import { formatDuration, speciesLabel } from '../../lib/format'
import { useInfiniteSentinel } from '../../lib/hooks'
import { boxOrigin, originPhrase } from '../../lib/origin'
import { LoadMore } from '../admin/fields'
import OriginIcon from './OriginIcon'

export default function AnimalGrid({
  animals,
  selecting,
  selected,
  onToggle,
  onOpen,
  hasMore,
  loadMore,
  loadingMore,
}: {
  animals: AnimalListItem[]
  selecting: boolean
  selected: Set<number>
  onToggle: (id: number) => void
  onOpen: (animal: AnimalListItem) => void
  hasMore: boolean
  loadMore: () => void
  loadingMore: boolean
}) {
  const sentinel = useInfiniteSentinel(loadMore, hasMore && !loadingMore)
  return (
    <>
      <ul className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
        {animals.map((a) => {
          const isSel = selected.has(a.id)
          const origin = boxOrigin({ person: a.person_id !== null ? { id: a.person_id } : null, assigned_by: a.assigned_by, source: a.source })
          const label = `${speciesLabel(a.species)} ${a.id}${a.item_kind === 'video' ? ', from a video' : ''}${origin ? `, ${originPhrase(origin)}` : ''}`
          return (
            <li key={a.id}>
              <button
                type="button"
                onClick={() => (selecting ? onToggle(a.id) : onOpen(a))}
                aria-label={selecting ? `${isSel ? 'Deselect' : 'Select'} ${label}` : `Open the item of ${label}`}
                aria-pressed={selecting ? isSel : undefined}
                className={clsx(
                  'group relative block aspect-square w-full overflow-hidden rounded-2xl bg-surface-2 ring-1 ring-line transition',
                  isSel ? 'ring-4 ring-accent' : 'hover:ring-ink-faint',
                )}
              >
                <img src={a.url} alt="" loading="lazy" decoding="async" className={clsx('size-full object-cover transition', isSel && 'scale-95 rounded-xl')} />
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
                <span className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-1 bg-gradient-to-t from-black/70 to-transparent px-1.5 pb-1 pt-4 text-[11px] font-medium text-white">
                  <span className="inline-flex min-w-0 items-center gap-1">
                    {a.item_kind === 'video' && <Film className="size-3 shrink-0" aria-hidden />}
                    {a.item_kind === 'video' && a.frame_time_s !== null ? formatDuration(a.frame_time_s) : speciesLabel(a.species)}
                  </span>
                  {origin && <OriginIcon origin={origin} size="xs" decorative />}
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
