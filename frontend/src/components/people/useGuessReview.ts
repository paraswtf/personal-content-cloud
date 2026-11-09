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
  return {
    confirm: (items: GuessItem[]) => m.mutate({ items, action: 'confirm' }),
    reject: (items: GuessItem[]) => m.mutate({ items, action: 'reject' }),
    correct: (items: GuessItem[], choice: PersonChoice) => m.mutate({ items, action: 'correct', choice }),
    busyFor,
    pending: m.isPending,
  }
}
