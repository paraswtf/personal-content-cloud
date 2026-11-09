// "Maybe related to" in the viewer's people section (section 10): the people this item may be related to, found from
// the items the owner marked as related to them (they look alike, or were taken at the same moment or place). Add
// marks it related; No is remembered for that person. Both are owner actions, so both can be undone.
import { useMutation, useQuery } from '@tanstack/react-query'
import { Check, Link2, X } from 'lucide-react'
import { api } from '../../lib/api'
import { personName } from '../../lib/format'
import type { ItemDetail, RelatedReason } from '../../lib/types'
import { errorMessage } from '../admin/adminHooks'
import { useInvalidateItem } from './shared'

const WHY: Record<RelatedReason, string> = {
  alike: 'looks like one of their related items',
  moment: 'same moment as one of their related items',
  place: 'same place as one of their related items',
}

const iconButton =
  'flex size-10 shrink-0 items-center justify-center rounded-full text-white/70 hover:bg-white/10 hover:text-white disabled:opacity-50'

export default function RelatedSuggest({ detail, onToast }: { detail: ItemDetail; onToast: (message: string) => void }) {
  const invalidate = useInvalidateItem(detail.uid)
  // under the item's key, so any change to the item asks again
  const q = useQuery({
    queryKey: ['item', detail.uid, 'related-suggestions'],
    queryFn: () => api.items.relatedSuggestions(detail.uid),
    staleTime: 30_000,
  })
  const add = useMutation({
    mutationFn: (personId: number) => api.items.addTag(detail.uid, { person_id: personId, relation: 'related' }),
    onSuccess: () => {
      invalidate()
      onToast('Tagged as related')
    },
    onError: (e) => onToast(errorMessage(e, 'Could not tag')),
  })
  const no = useMutation({
    mutationFn: (personId: number) => api.people.dismissRelated(personId, [detail.uid]),
    onSuccess: () => {
      invalidate()
      onToast('Not suggested for them again')
    },
    onError: (e) => onToast(errorMessage(e, 'Could not save that')),
  })
  const people = q.data?.people ?? []
  if (!people.length) return null
  const busy = add.isPending || no.isPending
  return (
    <div className="mt-3">
      <p className="mb-1.5 flex items-center gap-1.5 text-xs text-white/55">
        <Link2 className="size-3.5" aria-hidden />
        Maybe related to
      </p>
      <ul className="flex flex-col gap-1.5" aria-label="Maybe related to">
        {people.map(({ person, reason }) => {
          const name = personName(person)
          return (
            <li key={person.id} className="flex min-h-11 items-center gap-1 rounded-2xl bg-white/5 py-0.5 pl-3 pr-0.5">
              <span className="min-w-0 flex-1 text-sm text-white">
                <span className="font-semibold">{name}</span>
                <span className="block text-xs text-white/60">{WHY[reason]}</span>
              </span>
              <button type="button" className={iconButton} disabled={busy} onClick={() => add.mutate(person.id)} aria-label={`Add as related to ${name}`}>
                <Check className="size-5" aria-hidden />
              </button>
              <button type="button" className={iconButton} disabled={busy} onClick={() => no.mutate(person.id)} aria-label={`Not related to ${name}`}>
                <X className="size-5" aria-hidden />
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
