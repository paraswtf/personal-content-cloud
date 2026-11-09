// Who put a person or pet on a box, a video track, an item or a selection (section 10): the owner (a tag, a box
// they tagged or drew) or the system (recognition, grouping, a guess). OriginIcon shows it: a person icon for
// the owner, a bot for the system. Naming a group or merging people does not make its faces the owner's:
// the system grouped them, so they keep the bot until the owner tags one of them.
import type { AssignedBy, PersonRef } from './types'

/** 'drawn': a box the owner drew. 'guess': the system suggests someone and waits for an answer. */
export type Origin = 'you' | 'drawn' | 'auto' | 'guess'

export const ORIGIN_LABEL: Record<Origin, string> = {
  you: 'Tagged by you',
  drawn: 'Drawn by you',
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
  guess?: unknown
}

/** Who put the person (or the guess) on one box; null for a box the detector found with nobody on it. A box
 *  the owner drew and the system matched (a pet box) is 'drawn'; drawn and named by the owner is 'you'. */
export function boxOrigin(b: OriginBox): Origin | null {
  if (b.person && b.assigned_by === 'user') return 'you'
  if (b.source === 'user') return 'drawn'
  if (b.person) return 'auto'
  if (b.guess) return 'guess'
  return null
}

/** One origin for several boxes of one person (the tracks of a video group): the owner's when any is. */
export function mergeOrigins(list: (Origin | null)[]): Origin | null {
  for (const o of ['you', 'drawn', 'auto', 'guess'] as const) if (list.includes(o)) return o
  return null
}

/** A person's origin on a whole item: the owner's tag or box, a box the owner drew, or the system's alone. */
export type ItemOrigin = Exclude<Origin, 'guess'>
const RANK: Record<ItemOrigin, number> = { auto: 0, drawn: 1, you: 2 }

/** Per person or pet on an item: 'you' when the owner tagged them there (every tag is the owner's) or put them on
 *  a box, 'drawn' when their box there is one the owner drew and the system matched, else 'auto'. One answer per
 *  person, so their badges agree. */
export function itemOrigins(detail: {
  faces: OriginBox[]
  bodies: OriginBox[]
  animals: OriginBox[]
  tags: PersonRef[]
}): Map<number, ItemOrigin> {
  const out = new Map<number, ItemOrigin>()
  for (const t of detail.tags) out.set(t.id, 'you')
  for (const b of [...detail.faces, ...detail.bodies, ...detail.animals]) {
    if (!b.person) continue
    const o = boxOrigin(b)
    const v: ItemOrigin = o === 'you' || o === 'drawn' ? o : 'auto'
    const cur = out.get(b.person.id)
    if (cur === undefined || RANK[v] > RANK[cur]) out.set(b.person.id, v)
  }
  return out
}

/** Words for a control's label: "Show Asha in the photo, tagged by you". */
export function originPhrase(origin: Origin | null | undefined): string {
  return origin ? ORIGIN_LABEL[origin].toLowerCase() : ''
}
