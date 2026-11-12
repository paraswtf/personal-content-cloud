// The evidence of a person guess in words. A clothes guess reads as one line ("25 min apart, same scene 0.86,
// same clothes 0.91"); body traits (section 34) add their own reasons ("Same tattoo on the forearm as Priya's photo
// from 14 Mar 2026", "Same blue nail art, long, 2 days earlier"), always first. A guess from traits alone
// (basis "trait") carries no clothes or time-window keys and shows no percentage: its score is not calibrated, so
// it reads "strong match". A guess of recognition trained on one person (basis "face", section 10) reads "By face"
// with its probability as the percentage and one line: "Looks like Priya's photos, 0.61, and the same clothes".
import type { GuessBasis, GuessOut, PersonRef, TraitAgainst, TraitEvidence, TraitKind } from '../../lib/types'
import { formatDate, formatPercent, personName } from '../../lib/format'

type Ev = Pick<GuessOut, 'evidence'>

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function str(v: unknown): string | undefined {
  return typeof v === 'string' && v ? v : undefined
}

const KINDS: readonly TraitKind[] = ['tattoo', 'nails', 'mehndi']

function isKind(v: unknown): v is TraitKind {
  return typeof v === 'string' && (KINDS as readonly string[]).includes(v)
}

/** A server path for an image, nothing else. */
function path(v: unknown): string | undefined {
  return typeof v === 'string' && v.startsWith('/') && !v.startsWith('//') ? v : undefined
}

/** "#a1b2c3" only, so a colour can go into a style safely. */
export function safeHex(v: unknown): string | undefined {
  return typeof v === 'string' && /^#[0-9a-fA-F]{6}$/.test(v) ? v : undefined
}

/** What the guess rests on: clothes (no basis in the evidence, as before traits), traits alone, both, or
 *  recognition trained on the person. */
export function guessBasis(guess: Ev): GuessBasis {
  const b = guess.evidence?.basis
  return b === 'trait' || b === 'clothes+trait' || b === 'face' ? b : 'clothes'
}

/** The positive trait entries of the evidence, strongest first as the server sorted them; malformed ones dropped. */
export function guessTraits(guess: Ev): TraitEvidence[] {
  const raw = guess.evidence?.traits
  if (!Array.isArray(raw)) return []
  const out: TraitEvidence[] = []
  for (const r of raw) {
    if (!r || typeof r !== 'object') continue
    const o = r as Record<string, unknown>
    if (!isKind(o.kind) || typeof o.part !== 'string') continue
    out.push({
      kind: o.kind,
      part: o.part,
      ref_part: str(o.ref_part),
      mirrored: o.mirrored === true,
      trait_id: num(o.trait_id) ?? undefined,
      ref_trait_id: num(o.ref_trait_id) ?? undefined,
      ref_body_id: num(o.ref_body_id) ?? undefined,
      ref_item_uid: str(o.ref_item_uid),
      ref_taken_at: str(o.ref_taken_at),
      days_apart: num(o.days_apart) ?? undefined,
      ref_before: typeof o.ref_before === 'boolean' ? o.ref_before : undefined,
      inliers: num(o.inliers) ?? undefined,
      support: num(o.support) ?? undefined,
      strong: o.strong === true,
      llr: num(o.llr) ?? undefined,
      color: str(o.color),
      hex: safeHex(o.hex),
      finish: str(o.finish),
      length: str(o.length),
      crop_url: path(o.crop_url),
      ref_crop_url: path(o.ref_crop_url),
    })
  }
  return out
}

/** The traits the guessed person has that this box shows bare (they count against, softly). */
export function guessAgainst(guess: Ev): TraitAgainst[] {
  const raw = guess.evidence?.against
  if (!Array.isArray(raw)) return []
  const out: TraitAgainst[] = []
  for (const r of raw) {
    if (!r || typeof r !== 'object') continue
    const o = r as Record<string, unknown>
    if (!isKind(o.kind) || typeof o.part !== 'string') continue
    out.push({ kind: o.kind, part: o.part, llr: num(o.llr) ?? undefined, why: str(o.why) })
  }
  return out
}

const PART_WORDS: Record<string, string> = { forearm: 'forearm', upper_arm: 'upper arm', calf: 'calf', hand: 'hand' }

/** "forearm_l" -> "forearm": matching never looks at the side (a mirror photo swaps it). */
export function partWords(part: string): string {
  const fam = /_[lr]$/.test(part) ? part.slice(0, -2) : part
  return PART_WORDS[fam] ?? fam.replace(/_/g, ' ')
}

/** "Tattoo", "Nails", "Mehndi". */
export function traitNoun(kind: TraitKind): string {
  return kind === 'tattoo' ? 'Tattoo' : kind === 'mehndi' ? 'Mehndi' : 'Nails'
}

/** How far apart two photos were, from the reference's side: "the same day", "2 days earlier", "1 day later". */
export function daysPhrase(days: number | null | undefined, before?: boolean | null): string {
  if (days === null || days === undefined || !Number.isFinite(days)) return ''
  const n = Math.round(Math.abs(days))
  if (n === 0) return 'the same day'
  const unit = n === 1 ? 'day' : 'days'
  if (before === true) return `${n} ${unit} earlier`
  if (before === false) return `${n} ${unit} later`
  return `${n} ${unit} apart`
}

/** "blue nail art", "red glitter nails", "French nails", "nail polish". */
function nailWords(t: TraitEvidence): string {
  const color = t.color ? t.color.toLowerCase() : ''
  switch (t.finish) {
    case 'art':
      return color ? `${color} nail art` : 'nail art'
    case 'glitter':
      return color ? `${color} glitter nails` : 'glitter nails'
    case 'french':
      return 'French nails'
    case 'multi':
      return 'multicoloured nails'
    default:
      return color ? `${color} nails` : 'nail polish'
  }
}

export interface TraitReason {
  /** The reason as a sentence without a full stop. */
  text: string
  kind: TraitKind
  /** The part of this photo's box, as stored ("forearm_l"). */
  part: string
  cropUrl?: string
  refCropUrl?: string
  refItemUid?: string
  refTakenAt?: string
  mirrored: boolean
  /** Nails: the polish colour, for a swatch. */
  hex?: string
}

/** "Priya's", or "their" when the guess names no one or an unnamed group. */
function whose(person: PersonRef | null | undefined): string {
  return person?.name ? `${personName(person)}'s` : 'their'
}

/** One reason per positive trait entry, strongest first. */
export function traitReasons(guess: Ev & { person?: PersonRef | null }): TraitReason[] {
  return guessTraits(guess).map((t) => {
    const where = partWords(t.part)
    let text: string
    if (t.kind === 'tattoo') {
      const sameDay = (t.days_apart ?? 99) < 1
      const when = t.ref_taken_at ? formatDate(t.ref_taken_at) : ''
      text = sameDay
        ? `Same tattoo on the ${where} as in ${whose(guess.person)} photo taken the same day`
        : `Same tattoo on the ${where} as ${whose(guess.person)} photo${when ? ` from ${when}` : ''}`
      const more = t.support ? Math.round(t.support) : 0
      if (more > 0) text += more === 1 ? ' and 1 other photo' : ` and ${more} other photos`
      if (t.mirrored) text += ', probably a mirror photo'
    } else if (t.kind === 'mehndi') {
      const days = daysPhrase(t.days_apart, t.ref_before)
      text = `Same mehndi on the ${where}${days ? `, ${days}` : ''}`
      if (t.mirrored) text += ', probably a mirror photo'
    } else {
      const days = daysPhrase(t.days_apart, t.ref_before)
      text = `Same ${nailWords(t)}${t.length ? `, ${t.length}` : ''}${days ? `, ${days}` : ''}`
    }
    return {
      text,
      kind: t.kind,
      part: t.part,
      cropUrl: t.crop_url,
      refCropUrl: t.ref_crop_url,
      refItemUid: t.ref_item_uid,
      refTakenAt: t.ref_taken_at,
      mirrored: !!t.mirrored,
      hex: t.hex,
    }
  })
}

/** What counts against the guess, in words: "No tattoo on the upper arm here, where Priya has one". */
export function againstReasons(guess: Ev & { person?: PersonRef | null }): string[] {
  const who = guess.person ? personName(guess.person) : 'they'
  return guessAgainst(guess).map((a) => {
    const where = partWords(a.part)
    const noun = a.kind === 'tattoo' ? 'tattoo' : a.kind === 'mehndi' ? 'mehndi' : 'nail polish'
    return a.why === 'bare'
      ? `No ${noun} on the ${where} here, where ${who} ${guess.person ? 'has' : 'have'} one`
      : `No matching ${noun} on the ${where}`
  })
}

/** The kinds a guess shows, leading kind first: tattoo, then mehndi; nails never lead. */
function traitKinds(guess: Ev): TraitKind[] {
  const seen = new Set(guessTraits(guess).map((t) => t.kind))
  return (['tattoo', 'mehndi', 'nails'] as const).filter((k) => seen.has(k))
}

function joinWords(words: string[]): string {
  if (words.length <= 1) return words.join('')
  return `${words.slice(0, -1).join(', ')} and ${words[words.length - 1]}`
}

/** "Face", "By clothes", "By tattoo", "By mehndi and nails", "By clothes and tattoo", "By face". */
export function guessLabel(kind: 'face' | 'body', guess: Ev): string {
  const basis = guessBasis(guess)
  if (basis === 'face') return 'By face'
  const kinds = traitKinds(guess)
  if (basis === 'trait' && kinds.length) return `By ${joinWords(kinds)}`
  if (basis === 'clothes+trait' && kinds.length) return `By ${joinWords(['clothes', ...kinds])}`
  if (basis === 'trait') return 'By tattoo'
  return kind === 'face' ? 'Face' : 'By clothes'
}

/** The icons that go with the label: clothes, a tattoo, a hand (mehndi or nails), a face (trained recognition). */
export function guessIcons(kind: 'face' | 'body', guess: Ev): Array<'clothes' | 'tattoo' | 'hand' | 'face'> {
  const basis = guessBasis(guess)
  if (basis === 'face') return alsoClothes(guess) ? ['face', 'clothes'] : ['face']
  const kinds = traitKinds(guess)
  const out: Array<'clothes' | 'tattoo' | 'hand' | 'face'> = []
  if (basis !== 'trait' && (kind === 'body' || basis === 'clothes+trait')) out.push('clothes')
  if (kinds.includes('tattoo')) out.push('tattoo')
  else if (kinds.length) out.push('hand')
  return out
}

/** "85% sure" for a clothes guess (its score is learned) and for trained recognition (its probability, checked on
 *  photos it was not trained on); "strong match" for traits alone (no percentage). */
export function guessSure(guess: Pick<GuessOut, 'evidence' | 'score'>): string {
  return guessBasis(guess) === 'trait' ? 'strong match' : `${formatPercent(guess.score)} sure`
}

function alsoClothes(guess: Ev): boolean {
  const also = guess.evidence?.also
  return also === 'clothes' || also === 'clothes+trait'
}

/** A guess of recognition trained on the person: "Looks like Priya's photos, 0.61" and what else names them. */
export function faceLine(guess: Ev & { person?: PersonRef | null }): string {
  const e = guess.evidence ?? {}
  const face = num(e.face)
  let text = `Looks like ${whose(guess.person)} photos${face !== null ? `, ${face.toFixed(2)}` : ''}`
  if (e.also === 'clothes') text += ', and the same clothes'
  else if (e.also === 'trait') text += ', and the same tattoo or nails'
  else if (e.also === 'clothes+trait') text += ', and the same clothes and tattoo or nails'
  return text
}

/** The line under a guess. Clothes: time window, scene, place, clothes, face (and the trained recognition when it
 *  agrees). Trained recognition: faceLine. Empty for a guess from traits alone. */
export function clothesLine(guess: Ev & { person?: PersonRef | null }): string {
  const basis = guessBasis(guess)
  if (basis === 'trait') return ''
  if (basis === 'face') return faceLine(guess)
  const e = guess.evidence ?? {}
  const parts: string[] = []
  const minutes = num(e.minutes_apart)
  if (minutes !== null) parts.push(minutes < 1 ? 'same minute' : `${Math.round(minutes)} min apart`)
  else if (e.same_day === true) parts.push('same day') // guesses made before the one-hour window
  else if (e.same_day === false) parts.push('within hours')
  const scene = num(e.scene)
  if (scene !== null) parts.push(`same scene ${scene.toFixed(2)}`)
  const gps = num(e.gps_m)
  if (gps !== null) parts.push(`${Math.round(gps)} m apart`)
  const clothes = num(e.clothes)
  if (clothes !== null) parts.push(`same clothes ${clothes.toFixed(2)}`)
  const face = num(e.face)
  if (face !== null && face >= 0.2) parts.push(`face ${face.toFixed(2)}`)
  const trained = num(e.p)
  if (trained !== null) parts.push(`trained face ${formatPercent(trained)}`)
  return parts.join(', ')
}

/** The short trait phrases of a one-line summary: "same tattoo on the forearm", "same red nails 2 days earlier". */
function traitPhrases(guess: Ev): string[] {
  return guessTraits(guess).map((t) => {
    if (t.kind === 'tattoo') return `same tattoo on the ${partWords(t.part)}`
    const days = daysPhrase(t.days_apart, t.ref_before)
    const what = t.kind === 'mehndi' ? `same mehndi on the ${partWords(t.part)}` : `same ${nailWords(t)}`
    return days ? `${what} ${days}` : what
  })
}

/** The whole evidence as one line: trait phrases first, then the clothes parts (never for traits alone) or the
 *  trained recognition's line. */
export function evidenceLine(guess: Ev & { person?: PersonRef | null }): string {
  const parts = traitPhrases(guess)
  const clothes = clothesLine(guess)
  if (clothes) parts.push(clothes)
  return parts.join(', ')
}

/** uid of the item the guess was matched against, when the evidence names it. */
export function evidenceItemUid(guess: Ev): string | null {
  const v = guess.evidence?.ref_item_uid
  return typeof v === 'string' ? v : null
}
