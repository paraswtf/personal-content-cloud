// 3D heads of a person (docs/ARCHITECTURE.md section 36): the client, its types, and the logic of the look
// sliders. Mesh and picture URLs come from the API and are never built here.
import { del, get, post } from './api'
import type { OkOut } from './types'

/** One look of the person: a time of their life with one hair and facial hair style, and the head made for it. */
export interface Look3d {
  key: string
  label: string
  age: number | null
  age_band: string
  hair: number | null
  hair_label: string | null
  facial_hair: number | null
  facial_hair_label: string | null
  /** How many photos show this look. */
  photos: number
  /** When the photo the head was made from was taken. */
  taken_at: string | null
  item_uid: string
  picture_url: string
  mesh_url: string
  bytes: number
  vertices: number
  /** The head carries the photo as a texture over its front. Absent in a model made before heads had one. */
  textured?: boolean
  /** The face's shape (nose, lips, chin) was corrected from the photo by the depth model. */
  relief?: boolean
  /** What the generator has to say about this head, when it has something to say ("the photo did not line up, so
   *  this head has colours only"). */
  note?: string | null
}

export interface Model3dOptions {
  max_looks: number
  detail: number
  framing: number
  /** How much the surface is evened out: 0 off, 1 light, 2 normal, 3 strong. */
  smoothing: number
  /** The photo laid over the front of the head: 0 off (colours only); 1 and 2 shrink a photo whose longest side is
   *  over 1024 or 2048 pixels; 3 keeps the photo's own size. This step never enlarges the picture it is given. */
  texture: number
  /** 1: a depth model made for pictures of people shapes the nose, lips and chin, when the server has one. 0: off. */
  relief: number
}

/** The options as a model's status carries them. A model made before an option existed does not say it, or says
 *  null (completeOptions fills it in). */
export type Model3dMade = { [K in keyof Model3dOptions]?: number | null }

export interface Model3dProgress {
  step: string
  done: number
  total: number
}

export type Model3dState = 'none' | 'running' | 'done' | 'error'

export interface Model3dStatus {
  /** False when nothing can be generated; `reason` says why in plain words. */
  available: boolean
  reason: string | null
  generator: string | null
  /** The depth model files that Face relief needs are on the server. False: no head gets its face corrected until
   *  they are installed. Null or absent: the server does not say. */
  relief_installed?: boolean | null
  usable_photos: number
  state: Model3dState
  /** Another person's model is being generated: one at a time. */
  busy: boolean
  /** Stop or Delete was pressed and the running job is on its way out. */
  stopping: boolean
  progress: Model3dProgress | null
  error: string | null
  generated_at: string | null
  /** As the model was made. */
  options: Model3dMade | null
  looks: Look3d[]
}

export const MODEL3D_DEFAULTS: Model3dOptions = { max_looks: 6, detail: 2, framing: 2.6, smoothing: 2, texture: 2, relief: 1 }
/** The ranges the server accepts (app.api.model3d.Model3dIn). */
export const MODEL3D_LIMITS = {
  max_looks: { min: 1, max: 12, step: 1 },
  detail: { min: 1, max: 3, step: 1 },
  framing: { min: 1.8, max: 3.2, step: 0.1 },
  smoothing: { min: 0, max: 3, step: 1 },
  texture: { min: 0, max: 3, step: 1 },
  relief: { min: 0, max: 1, step: 1 },
} as const
export const MODEL3D_POLL_MS = 2000

/** The options a model was made with, complete and inside the ranges. A model made before an option existed does
 *  not say it: that option takes its default, and so does a value that is no number. */
export function completeOptions(options: Model3dMade | null | undefined): Model3dOptions {
  const out = { ...MODEL3D_DEFAULTS }
  for (const key of Object.keys(MODEL3D_LIMITS) as (keyof Model3dOptions)[]) {
    const value = options?.[key]
    if (typeof value !== 'number' || !Number.isFinite(value)) continue
    const { min, max, step } = MODEL3D_LIMITS[key]
    const stepped = step >= 1 ? Math.round(value) : Math.round(value / step) * step
    out[key] = Math.min(max, Math.max(min, Math.round(stepped * 1000) / 1000))
  }
  return out
}

/** The model is from before heads carried their photo: no look of it has one, and its options do not say whether
 *  the photo was to be laid on (a model made since with the photo switched off says 0). Such heads only get the
 *  photo when they are generated again. */
export function beforePhotoTextures(status: Pick<Model3dStatus, 'looks' | 'options'>): boolean {
  return status.looks.length > 0 && !status.looks.some((l) => l.textured) && typeof status.options?.texture !== 'number'
}

/** Not one of LIBRARY_KEYS: tagging elsewhere does not refetch it. The 3D tab reads it again each time it is
 *  shown, and when a head or a look's picture does not load. */
export const model3dKey = (id: number) => ['model3d', id] as const

export const model3dApi = {
  status: (id: number, signal?: AbortSignal) => get<Model3dStatus>(`/api/people/${id}/model3d`, signal),
  generate: (id: number, options: Model3dOptions) => post<{ started: boolean }>(`/api/people/${id}/model3d`, options),
  /** Stops the running job; the model in use stays. */
  stop: (id: number) => post<{ stopped: boolean }>(`/api/people/${id}/model3d/stop`),
  /** Deletes the model, stopping the running job first. */
  remove: (id: number) => del<OkOut>(`/api/people/${id}/model3d`),
}

// ------------------------------------------------------------------ the look sliders
// Every look sits at one stop of each axis: its age band, its hair length and its facial hair. A slider's
// stops are the values that exist among the looks, in order. Moving one slider decides that axis first; among
// the looks at the new stop, the one nearest to the head on screen on the other axes is shown, and the other
// sliders follow it. So the three sliders always describe the head on screen, and every stop leads to a look.

export type LookAxisKey = 'age' | 'hair' | 'facial_hair'

export interface LookAxisStop {
  /** The words for this stop ("about 24", "short hair", "a beard"). */
  text: string
  /** The looks at this stop, in the server's order. */
  keys: string[]
}

export interface LookAxis {
  key: LookAxisKey
  label: string
  /** Empty when no look has a value on this axis (it was not read): the slider is left out. */
  stops: LookAxisStop[]
}

const NOT_READ = 'not read'

const AXES: { key: LookAxisKey; label: string; group: (l: Look3d) => string | null; level: (l: Look3d) => number | null }[] = [
  { key: 'age', label: 'Age', group: (l) => l.age_band, level: (l) => l.age },
  { key: 'hair', label: 'Hair', group: (l) => l.hair_label, level: (l) => l.hair },
  { key: 'facial_hair', label: 'Facial hair', group: (l) => l.facial_hair_label, level: (l) => l.facial_hair },
]

function mean(values: number[]): number | null {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null
}

/** The three axes of a model's looks, each with its stops in order (low to high). */
export function lookAxes(looks: Look3d[]): LookAxis[] {
  return AXES.map(({ key, label, group, level }) => {
    const groups = new Map<string, Look3d[]>()
    for (const l of looks) {
      const g = group(l) ?? ''
      groups.set(g, [...(groups.get(g) ?? []), l])
    }
    if (groups.size === 1 && groups.has('')) return { key, label, stops: [] }
    const stops = [...groups.entries()]
      .map(([g, members]) => {
        const at = mean(members.map(level).filter((v): v is number => v !== null))
        const text = g === '' ? NOT_READ : key === 'age' && at !== null ? `about ${Math.round(at)}` : g
        // a group nothing was read for sorts first; one without a level keeps the server's order
        return { text, keys: members.map((l) => l.key), at: g === '' ? -Infinity : (at ?? Infinity) }
      })
      .sort((a, b) => (a.at === b.at ? 0 : a.at < b.at ? -1 : 1))
    return { key, label, stops: stops.map(({ text, keys }) => ({ text, keys })) }
  })
}

/** Which stop of the axis the look sits at (-1 when the axis has no stops or the look is unknown). */
export function stopOf(axis: LookAxis, lookKey: string): number {
  return axis.stops.findIndex((s) => s.keys.includes(lookKey))
}

/** The words for the look's value on an axis: its own age when the axis is age, else the stop's text. */
export function lookAxisText(axis: LookAxis, look: Look3d): string {
  if (axis.key === 'age' && look.age !== null) return `about ${Math.round(look.age)}`
  const i = stopOf(axis, look.key)
  return i < 0 ? NOT_READ : axis.stops[i].text
}

/** The look to show after the owner moved one slider to a stop, starting from the look on screen. */
export function pickLook(axes: LookAxis[], currentKey: string, moved: LookAxisKey, stopIndex: number): string {
  const stop = axes.find((a) => a.key === moved)?.stops[stopIndex]
  if (!stop || stop.keys.length === 0 || stop.keys.includes(currentKey)) return currentKey
  let best = stop.keys[0]
  let bestDistance = Infinity
  for (const key of stop.keys) {
    let distance = 0
    for (const other of axes) {
      if (other.key === moved || other.stops.length < 2) continue
      const from = stopOf(other, currentKey)
      const to = stopOf(other, key)
      distance += from < 0 || to < 0 ? 0.5 : Math.abs(from - to) / (other.stops.length - 1)
    }
    if (distance < bestDistance) {
      best = key
      bestDistance = distance
    }
  }
  return best
}

/** The look a model opens on: the person as they are now, so the latest age, and there the look most photos show. */
export function firstLook(looks: Look3d[]): Look3d | undefined {
  const last = looks[looks.length - 1]
  if (!last) return undefined
  return looks.filter((l) => l.age_band === last.age_band).reduce((a, b) => (b.photos > a.photos ? b : a))
}

// ------------------------------------------------------------------ the viewer's controls
// Kept here, not beside the viewer, so the tab can use them without loading the 3D chunk.

/** How the head is seen, in the sliders' terms. turn: degrees the head is turned to the viewer's right (0 faces
 *  the viewer, 180 shows the back). tilt: degrees the viewer looks down from above (negative: up from below).
 *  zoom: 1 fits the whole head. */
export interface HeadView {
  turn: number
  tilt: number
  zoom: number
}

/** The lamp. direction: degrees to the viewer's right of the viewer (0 lights the head from where the viewer
 *  stands, 180 from behind it). height: degrees above the viewer's eye line. brightness: 1 is normal. */
export interface HeadLight {
  direction: number
  height: number
  brightness: number
}

export const HEAD_VIEW: HeadView = { turn: 0, tilt: 0, zoom: 1 }
export const HEAD_LIGHT: HeadLight = { direction: 30, height: 35, brightness: 1 }
export const HEAD_LIMITS = {
  turn: { min: -180, max: 180, step: 1 },
  tilt: { min: -75, max: 75, step: 1 },
  zoom: { min: 0.6, max: 2.5, step: 0.05 },
  direction: { min: -180, max: 180, step: 5 },
  height: { min: -90, max: 90, step: 5 },
  brightness: { min: 0.4, max: 1.8, step: 0.05 },
} as const

/** What the tab may ask of the viewer. The view is not a prop: the owner also changes it by dragging the head,
 *  so the viewer owns it, reports it (onView) and takes changes as commands (only the parts that change, so a
 *  slider never puts back a turn the viewer has already moved past). */
export interface Head3dHandle {
  setView: (change: Partial<HeadView>) => void
}
