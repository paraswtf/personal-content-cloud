// Confirm or reject guesses from a list (review page, person page), one or many at a time.
import { useMutation } from '@tanstack/react-query'
import { api } from '../../lib/api'
import { formatCount } from '../../lib/format'
import { useRefreshLibrary } from '../../lib/hooks'
import type { GuessItem } from '../../lib/types'
import { useToast } from '../Toast'
import { errorMessage } from '../admin/adminHooks'

export const guessKey = (g: Pick<GuessItem, 'kind' | 'id'>) => `${g.kind}:${g.id}`

export function useGuessReview() {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const m = useMutation({
    mutationFn: (v: { items: GuessItem[]; action: 'confirm' | 'reject' }) => {
      const body = {
        faces: v.items.filter((g) => g.kind === 'face').map((g) => g.id),
        bodies: v.items.filter((g) => g.kind === 'body').map((g) => g.id),
      }
      return v.action === 'confirm' ? api.people.confirmGuesses(body) : api.people.rejectGuesses(body)
    },
    onSuccess: (r, v) => {
      if (v.items.length > 1) toast.success(`${formatCount(r.updated, 'guess', 'guesses')} ${v.action === 'confirm' ? 'confirmed' : 'rejected'}.`)
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'That did not work. Try again.')),
  })
  const busyFor = (g: GuessItem): 'confirm' | 'reject' | null =>
    m.isPending && m.variables && m.variables.items.some((x) => guessKey(x) === guessKey(g)) ? m.variables.action : null
  return {
    confirm: (items: GuessItem[]) => m.mutate({ items, action: 'confirm' }),
    reject: (items: GuessItem[]) => m.mutate({ items, action: 'reject' }),
    busyFor,
    pending: m.isPending,
  }
}
