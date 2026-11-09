// Who put a person or pet on a box, a video track, an item or a selection (section 10): the owner (a tag, a box
// they tagged or drew) or the system (recognition, grouping, a guess). OriginIcon shows it: a person icon for
// the owner, a bot for the system. Naming a group or merging people does not make its faces the owner's:
// the system grouped them, so they keep the bot until the owner tags one of them.
import type { AssignedBy, PersonRef } from './types'

/** 'drawn': a box the owner drew. 'confirmed': a guess the owner said yes to, still the system's find. 'guess': the
 *  system suggests someone and waits for an answer. */
export type Origin = 'you' | 'drawn' | 'confirmed' | 'auto' | 'guess'

export const ORIGIN_LABEL: Record<Origin, string> = {
  you: 'Tagged by you',
  drawn: 'Drawn by you',
  confirmed: 'Guessed automatically, confirmed by you',
  auto: 'Tagged automatically',
  guess: 'Suggested automatically',
}

/** The owner's origins: the person icon. The rest show the bot. */
export function isMine(origin: Origin): boolean {
  return origin === 'you' || origin === 'drawn'
}

interface OriginBox {
  person: { id: number } | null
  assigned_by: AssignedBy
  /** Bodies and pet boxes: 'user' when the owner drew the box. Older servers omit it. */
  source?: 'detector' | 'user'
  /** The person came from a guess the owner confirmed. */
  by_guess?: boolean
  guess?: unknown
}

/** Who put the person (or the guess) on one box; null for a box the detector found with nobody on it. Named by the
 *  owner: 'you'; a box the owner drew with someone the system matched on it: 'drawn'; a guess the owner confirmed:
 *  'confirmed', still the system's find (bot); found by the system: 'auto'. Moving or resizing a box changes none
 *  of this, and a guess waiting for an answer stays the system's suggestion even on a box the owner drew. */
export function boxOrigin(b: OriginBox): Origin | null {
  if (b.person && b.assigned_by === 'user' && !b.by_guess) return 'you'
  if (b.person && b.source === 'user') return 'drawn'
  if (b.person && b.assigned_by === 'user') return 'confirmed'
  if (b.person) return 'auto'
  if (b.guess) return 'guess'
  if (b.source === 'user') return 'drawn'
  return null
}

/** One origin for several boxes of one person (the tracks of a video group): the owner's when any is. */
export function mergeOrigins(list: (Origin | null)[]): Origin | null {
  for (const o of ['you', 'drawn', 'confirmed', 'auto', 'guess'] as const) if (list.includes(o)) return o
  return null
}

/** A person's origin on a whole item: the owner's tag or box, a box the owner drew, a guess they confirmed, or the
 *  system's alone. */
export type ItemOrigin = Exclude<Origin, 'guess'>
const RANK: Record<ItemOrigin, number> = { auto: 0, confirmed: 1, drawn: 2, you: 3 }

/** Per person or pet on an item: 'you' when the owner tagged them there by hand or put them on a box, 'drawn' when
 *  their box there is one the owner drew and the system matched, 'confirmed' when a guess the owner said yes to
 *  holds them (a box, or the tag of a faceless one), else 'auto'. One answer per person, so their badges agree. */
export function itemOrigins(detail: {
  faces: OriginBox[]
  bodies: OriginBox[]
  animals: OriginBox[]
  tags: PersonRef[]
}): Map<number, ItemOrigin> {
  const out = new Map<number, ItemOrigin>()
  const take = (id: number, v: ItemOrigin) => {
    const cur = out.get(id)
    if (cur === undefined || RANK[v] > RANK[cur]) out.set(id, v)
  }
  for (const t of detail.tags) take(t.id, t.by_guess ? 'confirmed' : 'you')
  for (const b of [...detail.faces, ...detail.bodies, ...detail.animals]) {
    if (!b.person) continue
    const o = boxOrigin(b)
    take(b.person.id, o === 'you' || o === 'drawn' || o === 'confirmed' ? o : 'auto')
  }
  return out
}

/** Words for a control's label: "Show Asha in the photo, tagged by you". */
export function originPhrase(origin: Origin | null | undefined): string {
  return origin ? ORIGIN_LABEL[origin].toLowerCase() : ''
}
