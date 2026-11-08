// The evidence of a person guess as one readable line: "25 min apart, same scene 0.86, same clothes 0.91".
import type { GuessOut } from '../../lib/types'

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

export function evidenceLine(guess: Pick<GuessOut, 'evidence'>): string {
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
  return parts.join(', ')
}

/** uid of the item the guess was matched against, when the evidence names it. */
export function evidenceItemUid(guess: Pick<GuessOut, 'evidence'>): string | null {
  const v = guess.evidence?.ref_item_uid
  return typeof v === 'string' ? v : null
}
