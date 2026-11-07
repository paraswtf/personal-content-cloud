// Face and body boxes of an item as one list of targets for the stage, the chips and the editor.
import type { BodyOut, FaceOut, ItemDetail } from '../../lib/types'

export type BoxKind = 'face' | 'body'
export interface BoxTarget {
  kind: BoxKind
  box: FaceOut | BodyOut
}

export function boxKey(t: { kind: BoxKind; box: { id: number } }): string {
  return `${t.kind}:${t.box.id}`
}

/** Boxes drawn on a still picture: faces, plus bodies whose face was not found (a back, a far shot). */
export function stillBoxes(detail: ItemDetail): BoxTarget[] {
  const faces: BoxTarget[] = detail.faces.filter((f) => f.frame_time_s === null).map((box) => ({ kind: 'face', box }))
  const bodies: BoxTarget[] = detail.bodies
    .filter((b) => b.frame_time_s === null && b.face_id === null)
    .map((box) => ({ kind: 'body', box }))
  return [...faces, ...bodies]
}

/** Boxes found in video frames, by time. */
export function videoBoxes(detail: ItemDetail): BoxTarget[] {
  const faces: BoxTarget[] = detail.faces.filter((f) => f.frame_time_s !== null).map((box) => ({ kind: 'face', box }))
  const bodies: BoxTarget[] = detail.bodies
    .filter((b) => b.frame_time_s !== null && b.face_id === null && (b.person || b.guess))
    .map((box) => ({ kind: 'body', box }))
  return [...faces, ...bodies].sort((a, b) => (a.box.frame_time_s ?? 0) - (b.box.frame_time_s ?? 0))
}

/** Find a target again in a fresher detail (after a save). */
export function findBox(detail: ItemDetail, kind: BoxKind, id: number): BoxTarget | null {
  const box = kind === 'face' ? detail.faces.find((f) => f.id === id) : detail.bodies.find((b) => b.id === id)
  return box ? { kind, box } : null
}
