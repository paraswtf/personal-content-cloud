// Face, body and animal boxes of an item as one list of targets for the stage, the chips and the editor.
import { personName, speciesLabel } from '../../lib/format'
import type { AnimalOut, BodyOut, FaceOut, ItemDetail } from '../../lib/types'

export type BoxKind = 'face' | 'body' | 'animal'
/** What the owner is drawing a box for: a pet the detector missed (section 22), a person or a face (section 10). */
export type DrawKind = 'pet' | 'person' | 'face'
/** An animal box has no guesses; the optional field lets shared code read `box.guess` on any box. */
export type AnimalBox = AnimalOut & { guess?: null }
export type BoxTarget =
  | { kind: 'face'; box: FaceOut }
  | { kind: 'body'; box: BodyOut }
  | { kind: 'animal'; box: AnimalBox }
/** A face or body target: the ones with guesses and the person editor. */
export type PersonBoxTarget = Exclude<BoxTarget, { kind: 'animal' }>

export function boxKey(t: { kind: BoxKind; box: { id: number } }): string {
  return `${t.kind}:${t.box.id}`
}

/** Label of an animal box: the pet's name when it has one, else the species. */
export function animalLabel(box: AnimalOut): string {
  return box.person?.name?.trim() ? box.person.name : speciesLabel(box.species)
}

/** Whose box it is, for the box adjuster's label (section 10): "the face box of Asha", "this pet box". */
export function adjustLabel(t: BoxTarget): string {
  const b = t.box
  const noun = t.kind === 'animal' ? 'pet box' : t.kind === 'body' ? 'person box' : 'face box'
  if (t.kind === 'animal') return t.box.person ? `the ${noun} of ${animalLabel(t.box)}` : `this ${noun}`
  if (b.person) return `the ${noun} of ${personName(b.person)}`
  return b.guess ? `this ${noun}, maybe ${personName(b.guess.person)}` : `this ${noun}`
}

/** Boxes drawn on a still picture: faces, plus bodies whose face was not found (a back, a far shot), plus animals. */
export function stillBoxes(detail: ItemDetail): BoxTarget[] {
  const faces: BoxTarget[] = detail.faces.filter((f) => f.frame_time_s === null).map((box) => ({ kind: 'face', box }))
  const bodies: BoxTarget[] = detail.bodies
    .filter((b) => b.frame_time_s === null && b.face_id === null)
    .map((box) => ({ kind: 'body', box }))
  const animals: BoxTarget[] = detail.animals.filter((a) => a.frame_time_s === null).map((box) => ({ kind: 'animal', box }))
  return [...faces, ...bodies, ...animals]
}

/** A box's area in the picture's own units (a share of the width times a share of the height). Every box of one
 *  picture has the same scale, so ordering by this area is ordering by the area on screen. */
export function boxArea(box: { w: number; h: number }): number {
  const side = (v: number) => (Number.isFinite(v) && v > 0 ? v : 0)
  return side(box.w) * side(box.h)
}

/**
 * Overlapping boxes (section 14): the smaller box is drawn above a larger one and gets the pointer first, so a
 * face inside a person box, or a child in front of a grown-up, can always be tapped. The layer comes from the
 * area alone: being edited, hovered, lit, focused or picked never raises a box (those show by its border and
 * ring). Returns each target's layer by `boxKey`: 1 for the largest, `targets.length` for the smallest. Equal
 * areas keep the list's order, the later one above, as the page would paint them.
 */
export function stackLayers(targets: readonly { kind: BoxKind; box: { id: number; w: number; h: number } }[]): Map<string, number> {
  const order = targets.map((t, i) => ({ key: boxKey(t), area: boxArea(t.box), i }))
  order.sort((a, b) => b.area - a.area || a.i - b.i)
  return new Map(order.map((o, n) => [o.key, n + 1]))
}

/** Boxes found in video frames, by time. */
export function videoBoxes(detail: ItemDetail): BoxTarget[] {
  const faces: BoxTarget[] = detail.faces.filter((f) => f.frame_time_s !== null).map((box) => ({ kind: 'face', box }))
  const bodies: BoxTarget[] = detail.bodies
    .filter((b) => b.frame_time_s !== null && b.face_id === null && (b.person || b.guess))
    .map((box) => ({ kind: 'body', box }))
  const animals: BoxTarget[] = detail.animals.filter((a) => a.frame_time_s !== null).map((box) => ({ kind: 'animal', box }))
  return [...faces, ...bodies, ...animals].sort((a, b) => (a.box.frame_time_s ?? 0) - (b.box.frame_time_s ?? 0))
}

/** Find a target again in a fresher detail (after a save). */
export function findBox(detail: ItemDetail, kind: BoxKind, id: number): BoxTarget | null {
  if (kind === 'face') {
    const box = detail.faces.find((f) => f.id === id)
    return box ? { kind, box } : null
  }
  if (kind === 'body') {
    const box = detail.bodies.find((b) => b.id === id)
    return box ? { kind, box } : null
  }
  const box = detail.animals.find((a) => a.id === id)
  return box ? { kind, box } : null
}

/** The tracks of a video gathered by their group (section 10): one entry per person or look-alike, in order
 *  of first appearance, with the tracks sorted by time. A track without a group stands alone. */
export interface VideoGroup {
  key: string
  targets: BoxTarget[]
  first: BoxTarget
}

export function groupVideoBoxes(targets: BoxTarget[]): VideoGroup[] {
  const groups = new Map<string, BoxTarget[]>()
  for (const t of targets) {
    const g = t.box.group
    const key = g === null || g === undefined ? `solo:${boxKey(t)}` : `group:${g}`
    const list = groups.get(key)
    if (list) list.push(t)
    else groups.set(key, [t])
  }
  return [...groups.entries()].map(([key, list]) => {
    const sorted = [...list].sort((a, b) => (a.box.frame_time_s ?? 0) - (b.box.frame_time_s ?? 0))
    return { key, targets: sorted, first: sorted[0] }
  })
}

/** The open guesses of an item, faces first, then person boxes: the order of "Guesses and why" in the details. */
export function guessTargets(detail: ItemDetail): PersonBoxTarget[] {
  return [
    ...detail.faces.filter((f) => !f.person && f.guess).map((box) => ({ kind: 'face' as const, box })),
    ...detail.bodies.filter((b) => !b.person && b.guess).map((box) => ({ kind: 'body' as const, box })),
  ]
}
