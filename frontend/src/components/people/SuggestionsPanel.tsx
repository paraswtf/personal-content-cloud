// Merge suggestions: pairs of people who look alike, with "Same person" and "Different people".
import { useState } from 'react'
import { Link } from 'react-router'
import { useMutation, useQuery } from '@tanstack/react-query'
import { GitMerge, UsersRound } from 'lucide-react'
import { api } from '../../lib/api'
import type { PersonOut, Suggestion } from '../../lib/types'
import { formatCount, personName } from '../../lib/format'
import { useRefreshLibrary } from '../../lib/hooks'
import { useToast } from '../Toast'
import { errorMessage } from '../admin/adminHooks'
import { Avatar, Badge, Button } from '../ui'
import MergeModal from './MergeModal'

function Side({ person, samples }: { person: PersonOut; samples: string[] }) {
  return (
    <div className="min-w-0 flex-1">
      <Link to={`/people/${person.id}`} className="flex min-w-0 items-center gap-2 hover:underline">
        <Avatar src={person.cover_url} name={person.name} size={44} />
        <span className="min-w-0">
          <span className="block truncate text-sm font-semibold">{personName(person)}</span>
          <span className="block text-xs text-ink-faint">{formatCount(person.item_count, 'item')}</span>
        </span>
      </Link>
      <ul className="mt-2 grid grid-cols-4 gap-1" aria-label={`Sample faces of ${personName(person)}`}>
        {samples.slice(0, 4).map((src, i) => (
          <li key={`${src}-${i}`}>
            <img src={src} alt="" loading="lazy" className="aspect-square w-full rounded-lg bg-surface-2 object-cover" />
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function SuggestionsPanel() {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const [choose, setChoose] = useState<Suggestion | null>(null)
  const q = useQuery({ queryKey: ['suggestions'], queryFn: () => api.people.suggestions(12) })

  const merge = useMutation({
    mutationFn: (v: { target: number; sources: number[] }) => api.people.merge({ target_id: v.target, source_ids: v.sources }),
    onSuccess: (p) => {
      toast.success(`Merged into ${personName(p)}.`)
      setChoose(null)
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not merge.')),
  })
  const dismiss = useMutation({
    mutationFn: (s: Suggestion) => api.people.dismissSuggestion(s.a.id, s.b.id),
    onSuccess: () => {
      toast.success('Marked as different people.')
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not save that.')),
  })

  const items = q.data?.items ?? []
  if (!items.length) return null

  const startMerge = (s: Suggestion) => {
    if (s.a.name && s.b.name) return setChoose(s)
    const keepA = s.a.name ? true : s.b.name ? false : s.a.item_count >= s.b.item_count
    const [t, src] = keepA ? [s.a, s.b] : [s.b, s.a]
    merge.mutate({ target: t.id, sources: [src.id] })
  }
  const pairBusy = (s: Suggestion) =>
    (merge.isPending && merge.variables && [s.a.id, s.b.id].includes(merge.variables.target)) ||
    (dismiss.isPending && dismiss.variables?.a.id === s.a.id && dismiss.variables?.b.id === s.b.id)

  return (
    <section aria-labelledby="suggestions-title" className="mb-6">
      <h2 id="suggestions-title" className="mb-2 flex items-center gap-2 text-base font-semibold">
        <UsersRound className="size-4" aria-hidden />
        Could these be the same person?
      </h2>
      <ul className="-mx-4 flex gap-3 overflow-x-auto px-4 pb-2 sm:mx-0 sm:px-0">
        {items.map((s) => (
          <li key={`${s.a.id}-${s.b.id}`} className="w-[19rem] shrink-0 rounded-2xl border border-line bg-surface p-3">
            <div className="mb-2 flex justify-center">
              <Badge tone={s.similarity >= 0.6 ? 'ok' : 'neutral'}>{Math.round(s.similarity * 100)}% alike</Badge>
            </div>
            <div className="flex gap-3">
              <Side person={s.a} samples={s.a_samples} />
              <Side person={s.b} samples={s.b_samples} />
            </div>
            <div className="mt-3 flex gap-2">
              <Button size="md" className="flex-1" disabled={!!pairBusy(s) || merge.isPending} onClick={() => startMerge(s)}>
                <GitMerge className="size-4" aria-hidden />
                Same person
              </Button>
              <Button size="md" variant="secondary" className="flex-1" disabled={!!pairBusy(s) || dismiss.isPending} onClick={() => dismiss.mutate(s)}>
                Different
              </Button>
            </div>
          </li>
        ))}
      </ul>
      <MergeModal
        open={choose !== null}
        people={choose ? [choose.a, choose.b] : []}
        busy={merge.isPending}
        onConfirm={(target, sources) => merge.mutate({ target, sources })}
        onClose={() => setChoose(null)}
      />
    </section>
  )
}
