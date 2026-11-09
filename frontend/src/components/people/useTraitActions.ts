// Owner corrections of body traits (section 34): "Not a tattoo" (or nails, mehndi), "Not this person's" and
// "Show again" (this photo only). Each is one undoable action: the request carries the mutation's X-Action-Id
// (lib/api.ts and the MutationCache in main.tsx), and the success message, sent from onSuccess, is the one the
// toast claims for its Undo button (lib/undo.ts claimForToast). Everything that shows traits or guesses is asked
// again afterwards, since a hidden region drops out of the guesses that cited it.
import { useMutation } from '@tanstack/react-query'
import { api } from '../../lib/api'
import { useRefreshLibrary } from '../../lib/hooks'
import type { TraitKind, TraitOut } from '../../lib/types'
import { errorMessage } from '../admin/adminHooks'

export type TraitVerdict = 'not_trait' | 'not_theirs'

/** "Not a tattoo", "Not nail polish", "Not mehndi". */
export function notTraitLabel(kind: TraitKind): string {
  return kind === 'tattoo' ? 'Not a tattoo' : kind === 'mehndi' ? 'Not mehndi' : 'Not nail polish'
}

/** "tattoo", "nail polish", "mehndi". */
export function traitWord(kind: TraitKind): string {
  return kind === 'tattoo' ? 'tattoo' : kind === 'mehndi' ? 'mehndi' : 'nail polish'
}

/** Which button of which trait is busy. */
export function traitActionKey(id: number, verdict: TraitVerdict | null): string {
  return `${id}:${verdict ?? 'show'}`
}

interface Vars {
  trait: Pick<TraitOut, 'id' | 'kind' | 'label'>
  /** null: show it again on this photo. */
  verdict: TraitVerdict | null
  /** Whose box it is on, for the message ("Not Priya's"). */
  owner?: string | null
}

export function useTraitActions({ onDone, onError }: { onDone: (message: string) => void; onError: (message: string) => void }) {
  const refresh = useRefreshLibrary()
  const m = useMutation({
    mutationFn: (v: Vars) =>
      api.traits.hide(v.trait.id, v.verdict ? { hidden: true, verdict: v.verdict } : { hidden: false }),
    onSuccess: (_r, v) => {
      void refresh()
      const word = traitWord(v.trait.kind)
      // one sentence: the undo history names the action by it (lib/undo.ts labelOf)
      if (!v.verdict) onDone(`Shown again on this photo: ${v.trait.label}.`)
      else if (v.verdict === 'not_trait') onDone(`Not counted as ${v.trait.kind === 'tattoo' ? 'a tattoo' : word} any more: ${v.trait.label}.`)
      else onDone(`Not counted as ${v.owner ? `${v.owner}'s` : "this person's"} ${word} any more: ${v.trait.label}.`)
    },
    onError: (e) => onError(errorMessage(e, 'That did not work. Try again.')),
  })
  return {
    hide: (trait: Vars['trait'], verdict: TraitVerdict, owner?: string | null) => m.mutate({ trait, verdict, owner }),
    show: (trait: Vars['trait']) => m.mutate({ trait, verdict: null }),
    /** The change being saved: "12:not_trait", "12:not_theirs" or "12:show". */
    pendingKey: m.isPending && m.variables ? traitActionKey(m.variables.trait.id, m.variables.verdict) : null,
    pending: m.isPending,
  }
}
