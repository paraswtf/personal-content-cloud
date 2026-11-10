// Confirm, reject or correct ("Someone else") guesses from a list (review page, person page), one or many at a time.
import { useMutation } from '@tanstack/react-query'
import { api } from '../../lib/api'
import { formatCount, personName } from '../../lib/format'
import { useRefreshLibrary } from '../../lib/hooks'
import type { AssignOut, GuessItem } from '../../lib/types'
import { useToast } from '../Toast'
import { errorMessage } from '../admin/adminHooks'
import { choiceBody, type PersonChoice } from './PersonPicker'

type Action = 'confirm' | 'reject' | 'correct'

export const guessKey = (g: Pick<GuessItem, 'kind' | 'id'>) => `${g.kind}:${g.id}`

/** Per-call callbacks of a decision. */
export interface DecideOpts {
  onError?: () => void
}

export function useGuessReview() {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const m = useMutation({
    mutationFn: (v: { items: GuessItem[]; action: Action; choice?: PersonChoice }) => {
      const body = {
        faces: v.items.filter((g) => g.kind === 'face').map((g) => g.id),
        bodies: v.items.filter((g) => g.kind === 'body').map((g) => g.id),
      }
      // "Someone else": the guess is learned as wrong and the box goes to the person picked, in one step
      if (v.action === 'correct') return api.people.correctGuesses({ ...body, ...choiceBody(v.choice ?? null) })
      return v.action === 'confirm' ? api.people.confirmGuesses(body) : api.people.rejectGuesses(body)
    },
    onSuccess: (r, v) => {
      if (v.action === 'correct') {
        const person = (r as AssignOut).person
        toast.success(person ? `Tagged as ${personName(person)}.` : 'Saved.')
      } else if (v.items.length > 1) {
        toast.success(`${formatCount(r.updated, 'guess', 'guesses')} ${v.action === 'confirm' ? 'confirmed' : 'rejected'}.`)
      }
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'That did not work. Try again.')),
  })
  const busyFor = (g: GuessItem): Action | null =>
    m.isPending && m.variables && m.variables.items.some((x) => guessKey(x) === guessKey(g)) ? m.variables.action : null
  // opts.onError: the caller's own reaction to a failed decision (the Review page puts focus back on that card)
  return {
    confirm: (items: GuessItem[], opts?: DecideOpts) => m.mutate({ items, action: 'confirm' }, opts),
    reject: (items: GuessItem[], opts?: DecideOpts) => m.mutate({ items, action: 'reject' }, opts),
    correct: (items: GuessItem[], choice: PersonChoice, opts?: DecideOpts) =>
      m.mutate({ items, action: 'correct', choice }, opts),
    busyFor,
    pending: m.isPending,
  }
}
