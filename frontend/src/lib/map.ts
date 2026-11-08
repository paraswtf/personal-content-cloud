// The map API (docs/ARCHITECTURE.md section 25): points, items in a box, places, status and refresh,
// plus the basemap choice kept in localStorage.
import { get, post, qs, type ItemsQuery } from './api'
import type { ItemPage } from './types'

/** [lat, lon, uid, kind code]: 0 image, 1 video, 2 document, 3 other. */
export type MapPoint = [number, number, string, number]
export interface MapPoints {
  points: MapPoint[]
  total: number
  with_gps: number
}
export interface PlaceOut {
  place: string
  place_city: string | null
  place_admin: string | null
  place_country: string | null
  count: number
  lat: number
  lon: number
}
export interface MapStatus {
  gazetteer: boolean
  with_gps: number
  unchecked: number
  unplaced: number
  running: boolean
}
export type MapQuery = Omit<ItemsQuery, 'cursor' | 'limit'>
export type Tiles = 'offline' | 'online'

export const mapApi = {
  points: (q: MapQuery, signal?: AbortSignal) => get<MapPoints>(`/api/map/points${qs({ ...q })}`, signal),
  items: (q: MapQuery & { bbox: string; cursor?: string | null; limit?: number }, signal?: AbortSignal) =>
    get<ItemPage>(`/api/map/items${qs({ ...q })}`, signal),
  places: (q: MapQuery, signal?: AbortSignal) => get<PlaceOut[]>(`/api/map/places${qs({ ...q })}`, signal),
  status: () => get<MapStatus>('/api/map/status'),
  refresh: () => post<{ started: boolean }>('/api/map/refresh'),
}

export const MAP_TILES_KEY = 'pcc.mapTiles'

/** The basemap the owner chose on this browser, or null when they never chose. */
export function readTilesChoice(): Tiles | null {
  try {
    const v = localStorage.getItem(MAP_TILES_KEY)
    return v === 'online' || v === 'offline' ? v : null
  } catch {
    return null
  }
}

export function writeTilesChoice(v: Tiles): void {
  try {
    localStorage.setItem(MAP_TILES_KEY, v)
  } catch {
    // private mode or blocked storage: the choice lasts for the page
  }
}

/** "w,s,e,n" for the items endpoint, from a lon/lat box. */
export function bboxParam(w: number, s: number, e: number, n: number): string {
  const f = (v: number) => String(Math.round(v * 1e5) / 1e5)
  return `${f(w)},${f(s)},${f(e)},${f(n)}`
}
