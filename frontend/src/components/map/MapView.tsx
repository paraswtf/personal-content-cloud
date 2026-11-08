// The map itself (docs/ARCHITECTURE.md section 25): MapLibre GL with the offline basemap
// (/map/basemap.json drawn as fill and line layers in the theme's colours) or the online vector style,
// the points as a clustered GeoJSON source (MapLibre clusters client side, radius 60 px), clusters
// drawn as HTML markers with their counts (no glyph server is needed, so it works offline), thumbs on
// single points when zoomed in, and the clicks: a big cluster zooms in, a small cluster or a single
// point reports a box for the panel.
import { useEffect, useRef } from 'react'
import maplibregl, { type GeoJSONSource, type LngLatBoundsLike, type StyleSpecification } from 'maplibre-gl'
import type { Feature, FeatureCollection, Point } from 'geojson'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { MapPoint, Tiles } from '../../lib/map'

export interface Focus {
  lat: number
  lon: number
  zoom: number
  /** Changes force a new fly even when the coordinates repeat. */
  nonce?: number
}

export interface Selection {
  /** lon/lat box: west, south, east, north */
  bbox: [number, number, number, number]
  count: number
  lat: number
  lon: number
}

interface Props {
  points: MapPoint[]
  tiles: Tiles
  styleUrl: string
  focus: Focus | null
  onSelect: (sel: Selection) => void
}

/** Clusters this small open the panel instead of zooming in. */
const OPEN_BELOW = 12
const THUMB_ZOOM = 12
const MAX_THUMBS = 150
const SOURCE = 'points'

function cssVar(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

function offlineStyle(): StyleSpecification {
  const water = cssVar('--color-accent-soft', '#e5eafb')
  const land = cssVar('--color-surface', '#ffffff')
  const line = cssVar('--color-line', '#e0e0da')
  return {
    version: 8,
    sources: { basemap: { type: 'geojson', data: '/map/basemap.json' } },
    layers: [
      { id: 'bg', type: 'background', paint: { 'background-color': water } },
      { id: 'land', type: 'fill', source: 'basemap', filter: ['==', ['get', 'kind'], 'land'], paint: { 'fill-color': land } },
      { id: 'lakes', type: 'fill', source: 'basemap', filter: ['==', ['get', 'kind'], 'lake'], paint: { 'fill-color': water } },
      { id: 'coast', type: 'line', source: 'basemap', filter: ['==', ['get', 'kind'], 'land'], paint: { 'line-color': line, 'line-width': 0.6 } },
      { id: 'borders', type: 'line', source: 'basemap', filter: ['==', ['get', 'kind'], 'boundary'], paint: { 'line-color': line, 'line-width': 0.8 } },
    ],
  }
}

/** The points source, or undefined while the map has no style (none given yet, or the map was removed):
 * MapLibre's getSource and isSourceLoaded throw then instead of answering. */
function pointsSource(map: maplibregl.Map): GeoJSONSource | undefined {
  try {
    return map.getSource(SOURCE) as GeoJSONSource | undefined
  } catch {
    return undefined
  }
}

function sourceReady(map: maplibregl.Map): boolean {
  try {
    return !!map.getSource(SOURCE) && map.isSourceLoaded(SOURCE)
  } catch {
    return false
  }
}

function featureCollection(points: MapPoint[]): FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: points.map(([lat, lon, uid, k]) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [lon, lat] },
      properties: { uid, k },
    })),
  }
}

function clusterElement(count: number, onClick: () => void): HTMLElement {
  const el = document.createElement('button')
  el.type = 'button'
  const size = count >= 1000 ? 52 : count >= 100 ? 44 : count >= 10 ? 36 : 30
  el.className = 'flex items-center justify-center rounded-full border-2 border-surface bg-accent text-xs font-semibold text-white shadow-md'
  el.style.width = `${size}px`
  el.style.height = `${size}px`
  el.textContent = count >= 1000 ? `${Math.round(count / 100) / 10}k` : String(count)
  el.setAttribute('aria-label', `${count.toLocaleString()} items here`)
  el.addEventListener('click', (e) => {
    e.stopPropagation()
    onClick()
  })
  return el
}

function thumbElement(uid: string, onClick: () => void): HTMLElement {
  const el = document.createElement('button')
  el.type = 'button'
  el.className = 'size-10 overflow-hidden rounded-lg border-2 border-surface bg-surface-2 shadow-md'
  const img = document.createElement('img')
  img.src = `/files/thumb/${uid}`
  img.alt = ''
  img.loading = 'lazy'
  img.className = 'size-full object-cover'
  el.appendChild(img)
  el.setAttribute('aria-label', 'Open this photo')
  el.addEventListener('click', (e) => {
    e.stopPropagation()
    onClick()
  })
  return el
}

export default function MapView({ points, tiles, styleUrl, focus, onSelect }: Props) {
  const container = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const markers = useRef(new Map<string, maplibregl.Marker>())
  const pointsRef = useRef(points)
  const onSelectRef = useRef(onSelect)
  const fitted = useRef(false)
  const focusRef = useRef(focus)
  const tilesRef = useRef(tiles)
  pointsRef.current = points
  onSelectRef.current = onSelect
  tilesRef.current = tiles

  // create the map once; style, data and focus are applied by the effects below
  useEffect(() => {
    const el = container.current
    if (!el) return
    const initial = focusRef.current
    const map = new maplibregl.Map({
      container: el,
      style: tiles === 'online' && styleUrl ? styleUrl : offlineStyle(),
      center: initial ? [initial.lon, initial.lat] : [20, 15],
      zoom: initial ? initial.zoom : 1.4,
      attributionControl: false,
      maxZoom: 19,
    })
    if (initial) fitted.current = true
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right')
    mapRef.current = map

    const select = async (feature: Feature) => {
      const props = feature.properties as Record<string, unknown>
      const coords = (feature.geometry as Point).coordinates
      const source = pointsSource(map)
      if (props.cluster && source) {
        const id = props.cluster_id as number
        const count = Number(props.point_count ?? 0)
        const zoom = await source.getClusterExpansionZoom(id).catch(() => map.getZoom() + 2)
        if (count > OPEN_BELOW && zoom <= map.getMaxZoom()) {
          map.easeTo({ center: [coords[0], coords[1]], zoom: Math.min(zoom, map.getMaxZoom()) })
          return
        }
        const leaves = await source.getClusterLeaves(id, Math.max(count, 1), 0).catch(() => [] as Feature[])
        let w = coords[0], e = coords[0], s = coords[1], n = coords[1]
        for (const leaf of leaves) {
          const c = (leaf.geometry as Point).coordinates
          w = Math.min(w, c[0]); e = Math.max(e, c[0]); s = Math.min(s, c[1]); n = Math.max(n, c[1])
        }
        const pad = 0.0002
        onSelectRef.current({ bbox: [w - pad, s - pad, e + pad, n + pad], count, lat: coords[1], lon: coords[0] })
        return
      }
      const pad = 0.0001
      onSelectRef.current({ bbox: [coords[0] - pad, coords[1] - pad, coords[0] + pad, coords[1] + pad], count: 1, lat: coords[1], lon: coords[0] })
    }

    const sync = () => {
      if (!sourceReady(map)) return
      const feats = map.querySourceFeatures(SOURCE)
      // right after setData the source reports loaded while its tiles are still being rebuilt and the
      // query answers nothing; keep the markers until the idle event (below) runs sync again
      if (feats.length === 0 && pointsRef.current.length > 0 && !map.areTilesLoaded()) return
      const keep = new Set<string>()
      const showThumbs = map.getZoom() >= THUMB_ZOOM
      let thumbs = 0
      for (const f of feats) {
        const props = f.properties as Record<string, unknown>
        const coords = (f.geometry as Point).coordinates
        let key: string
        if (props.cluster) key = `c${props.cluster_id}`
        else {
          if (!showThumbs || thumbs >= MAX_THUMBS) continue
          thumbs += 1
          key = `p${props.uid}`
        }
        if (keep.has(key)) continue
        keep.add(key)
        const existing = markers.current.get(key)
        if (existing) {
          // MapLibre reuses cluster ids across data loads: keep the marker where the cluster is now
          existing.setLngLat([coords[0], coords[1]])
          if (props.cluster) {
            const count = Number(props.point_count ?? 0)
            const el = existing.getElement()
            el.textContent = count >= 1000 ? `${Math.round(count / 100) / 10}k` : String(count)
            el.setAttribute('aria-label', `${count.toLocaleString()} items here`)
          }
          continue
        }
        const element = props.cluster
          ? clusterElement(Number(props.point_count ?? 0), () => void select(f))
          : thumbElement(String(props.uid), () => void select(f))
        const marker = new maplibregl.Marker({ element }).setLngLat([coords[0], coords[1]]).addTo(map)
        markers.current.set(key, marker)
      }
      for (const [key, marker] of markers.current) {
        if (!keep.has(key)) {
          marker.remove()
          markers.current.delete(key)
        }
      }
    }

    const addPoints = () => {
      if (pointsSource(map)) return
      map.addSource(SOURCE, { type: 'geojson', data: featureCollection(pointsRef.current), cluster: true, clusterRadius: 60, clusterMaxZoom: 17 })
      map.addLayer({
        id: 'points-dots',
        type: 'circle',
        source: SOURCE,
        filter: ['!', ['has', 'point_count']],
        paint: {
          'circle-radius': 6,
          'circle-color': cssVar('--color-accent', '#3656d1'),
          'circle-stroke-color': cssVar('--color-surface', '#ffffff'),
          'circle-stroke-width': 2,
        },
      })
    }

    // layer events are registered once; MapLibre binds them by layer id, so the layer may come later
    map.on('click', 'points-dots', (e) => {
      const f = e.features?.[0]
      if (f) void select(f as unknown as Feature)
    })
    map.on('mouseenter', 'points-dots', () => {
      map.getCanvas().style.cursor = 'pointer'
    })
    map.on('mouseleave', 'points-dots', () => {
      map.getCanvas().style.cursor = ''
    })

    map.on('load', addPoints)
    map.on('style.load', addPoints)
    map.on('moveend', sync)
    map.on('zoomend', sync)
    map.on('sourcedata', (e) => {
      if (e.sourceId === SOURCE && e.isSourceLoaded) sync()
    })
    // idle fires once every tile is loaded and the frame is drawn: the one moment querySourceFeatures
    // is complete, so the marker set is settled here whatever the event order before was
    map.on('idle', sync)
    map.on('styledata', () => {
      // a style swap drops the markers' layers but not the markers: start again
      for (const m of markers.current.values()) m.remove()
      markers.current.clear()
    })

    // the theme toggles data-theme on <html>; the offline basemap follows the new colours
    const restyle = () => {
      if (mapRef.current === map && tilesRef.current === 'offline') map.setStyle(offlineStyle())
    }
    const observer = new MutationObserver(restyle)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    media.addEventListener('change', restyle)

    return () => {
      observer.disconnect()
      media.removeEventListener('change', restyle)
      for (const m of markers.current.values()) m.remove()
      markers.current.clear()
      map.remove()
      mapRef.current = null
    }
    // the map is created once; tiles and styleUrl changes are applied by the effect below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // basemap switch
  const firstStyle = useRef(true)
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (firstStyle.current) {
      firstStyle.current = false
      return
    }
    map.setStyle(tiles === 'online' && styleUrl ? styleUrl : offlineStyle())
  }, [tiles, styleUrl])

  // data; a refetch that brings the same points back must not rebuild the clusters
  const applied = useRef('')
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    let lat = 0
    let lon = 0
    for (const p of points) {
      lat += p[0]
      lon += p[1]
    }
    const signature = `${points.length}:${lat.toFixed(4)}:${lon.toFixed(4)}:${points[0]?.[2] ?? ''}:${points[points.length - 1]?.[2] ?? ''}`
    const apply = () => {
      const source = pointsSource(map)
      if (!source) return
      if (applied.current === signature) return
      applied.current = signature
      // new data means new clusters: drop the markers so none stays at an old spot with an old count
      for (const m of markers.current.values()) m.remove()
      markers.current.clear()
      source.setData(featureCollection(points))
      if (!fitted.current && points.length > 0) {
        fitted.current = true
        let w = 180, s = 90, e = -180, n = -90
        for (const [lat, lon] of points) {
          w = Math.min(w, lon); e = Math.max(e, lon); s = Math.min(s, lat); n = Math.max(n, lat)
        }
        const bounds: LngLatBoundsLike = [[w, s], [e, n]]
        map.fitBounds(bounds, { padding: 48, maxZoom: 11, duration: 0 })
      }
    }
    if (pointsSource(map)) {
      apply()
      return
    }
    // the source arrives with the style (addPoints on style.load, before load): a later points array
    // must not be overwritten by this closure firing at load, so it is removed when points change
    map.once('load', apply)
    return () => {
      map.off('load', apply)
    }
  }, [points])

  // fly to a place or the item the owner came from
  useEffect(() => {
    const map = mapRef.current
    if (!map || !focus) return
    fitted.current = true
    map.flyTo({ center: [focus.lon, focus.lat], zoom: focus.zoom, duration: 900 })
  }, [focus])

  return <div ref={container} className="h-full w-full" role="application" aria-label="Map of your photos" />
}
