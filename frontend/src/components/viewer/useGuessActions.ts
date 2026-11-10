// Confirm or reject a person guess on a face or body, then refresh what depends on it.
import { useMutation } from '@tanstack/react-query'
import { api } from '../../lib/api'
import { personName } from '../../lib/format'
import type { GuessOut } from '../../lib/types'
import { errorMessage } from '../admin/adminHooks'
import type { PersonBoxTarget } from './boxes'
import { useInvalidateItem } from './shared'

interface Vars {
  kind: PersonBoxTarget['kind']
  id: number
  guess: GuessOut
  action: 'confirm' | 'reject'
}

export function useGuessActions(uid: string, onToast: (msg: string) => void) {
  const invalidate = useInvalidateItem(uid)
  const m = useMutation({
    mutationFn: (v: Vars) => {
      const body = v.kind === 'face' ? { faces: [v.id] } : { bodies: [v.id] }
      return v.action === 'confirm' ? api.people.confirmGuesses(body) : api.people.rejectGuesses(body)
    },
    onSuccess: (_r, v) => {
      onToast(v.action === 'confirm' ? `Tagged as ${personName(v.guess.person)}` : `Not ${personName(v.guess.person)}`)
      // Pending until the item is fetched again: until then the answered guess is still listed, and a quick second
      // key in the Review viewer would answer it again instead of the next one.
      return invalidate()
    },
    onError: (e) => onToast(errorMessage(e, 'That did not work. Try again.')),
  })
  const pendingKey = m.isPending && m.variables ? `${m.variables.kind}:${m.variables.id}` : null
  return {
    confirm: (kind: PersonBoxTarget['kind'], id: number, guess: GuessOut) => m.mutate({ kind, id, guess, action: 'confirm' }),
    reject: (kind: PersonBoxTarget['kind'], id: number, guess: GuessOut) => m.mutate({ kind, id, guess, action: 'reject' }),
    pendingKey,
  }
}
