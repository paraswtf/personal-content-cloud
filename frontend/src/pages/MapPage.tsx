// /map (docs/ARCHITECTURE.md section 25): the photos with a location on a map, clustered, with the
// library's filters in the URL, a Places list (beside the map on desktop, in a sheet on phones), a
// panel with the items under a cluster or point, the viewer, the Detailed map toggle (online tiles,
// remembered in localStorage) and a status line with the scan of older items.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Globe, MapPin, RefreshCw, X } from 'lucide-react'
import { formatCount } from '../lib/format'
import { useConfig } from '../lib/hooks'
import { bboxParam, mapApi, readTilesChoice, writeTilesChoice, type PlaceOut, type Tiles } from '../lib/map'
import { useItemList } from '../lib/useItemList'
import { Button, EmptyState, ErrorState, Modal } from '../components/ui'
import { useToast } from '../components/Toast'
import { usePageTitle } from '../components/shell/title'
import FilterBar from '../components/library/FilterBar'
import { itemsQuery, readFilters } from '../components/library/filters'
import { withParam } from '../components/public/params'
import { useViewerParam } from '../components/public/useViewerParam'
import MediaGrid from '../components/MediaGrid'
import Viewer from '../components/Viewer'
import MapView, { type Focus, type Selection } from '../components/map/MapView'
import PlacesList from '../components/map/PlacesList'
import { errorMessage } from '../components/admin/adminHooks'

const STATUS_KEY = ['map', 'status'] as const

/** `at=lat,lon` and `z=` from the URL (the viewer's Show on map link), or null. */
function readAt(params: URLSearchParams): Focus | null {
  const at = params.get('at')
  if (!at) return null
  const [a, b] = at.split(',').map(Number)
  if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a) > 90 || Math.abs(b) > 180) return null
  const z = Number(params.get('z'))
  return { lat: a, lon: b, zoom: Number.isFinite(z) && z > 0 ? Math.min(z, 19) : 14, nonce: 0 }
}

export default function MapPage() {
  usePageTitle('Map')
  const qc = useQueryClient()
  const toast = useToast()
  const [params, setParams] = useSearchParams()
  const filters = useMemo(() => readFilters(params, false), [params])
  const mq = useMemo(() => {
    const { limit: _limit, ...rest } = itemsQuery(filters, false)
    return rest
  }, [filters])
  const set = useCallback(
    (key: string, value: string | null) => setParams(withParam(params, key, value), { replace: true }),
    [params, setParams],
  )
  const clearAll = () => setParams(new URLSearchParams(), { replace: true })

  // the basemap: the browser's choice, else the server's default
  const config = useConfig()
  const [choice, setChoice] = useState<Tiles | null>(() => readTilesChoice())
  const tiles: Tiles = choice ?? config.data?.map?.tiles ?? 'offline'
  const styleUrl = config.data?.map?.style_url ?? ''
  const pickTiles = (v: Tiles) => {
    setChoice(v)
    writeTilesChoice(v)
  }

  const points = useQuery({ queryKey: ['map', 'points', mq], queryFn: ({ signal }) => mapApi.points(mq, signal) })
  const places = useQuery({ queryKey: ['map', 'places', mq], queryFn: ({ signal }) => mapApi.places(mq, signal) })
  const status = useQuery({
    queryKey: STATUS_KEY,
    queryFn: mapApi.status,
    refetchInterval: (q) => (q.state.data?.running ? 4000 : false),
  })
  const refresh = useMutation({
    mutationFn: mapApi.refresh,
    onSuccess: (r) => {
      toast.success(r.started ? 'Scanning older items for locations. Places appear as they are found.' : 'A scan is already running.')
      void qc.invalidateQueries({ queryKey: STATUS_KEY })
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not start the scan.')),
  })

  // the spot the owner came from (the viewer's Show on map link) opens its panel too
  const [focus, setFocus] = useState<Focus | null>(() => readAt(params))
  const [sel, setSel] = useState<Selection | null>(() => {
    const at = readAt(params)
    if (!at || !params.get('m')) return null
    const pad = 0.0005
    return { bbox: [at.lon - pad, at.lat - pad, at.lon + pad, at.lat + pad], count: 1, lat: at.lat, lon: at.lon }
  })
  const [activePlace, setActivePlace] = useState<string | null>(null)
  const [placesOpen, setPlacesOpen] = useState(false)

  const pickPlace = (p: PlaceOut) => {
    setActivePlace(p.place)
    setPlacesOpen(false)
    setFocus({ lat: p.lat, lon: p.lon, zoom: 11, nonce: Date.now() })
  }

  // the panel: items inside the chosen box, in library order
  const bbox = sel ? bboxParam(...sel.bbox) : null
  const fetchPage = useCallback(
    (cursor: string | null, signal?: AbortSignal) => mapApi.items({ ...mq, bbox: bbox ?? '', cursor, limit: 60 }, signal),
    [mq, bbox],
  )
  const list = useItemList(['map', 'items', mq, bbox], fetchPage, { enabled: !!bbox })
  const viewer = useViewerParam(list.items, list.hasMore, list.loadMore, list.total)
  const openItem = useCallback((uid: string) => viewer.open(uid), [viewer])
  const closePanel = () => {
    setSel(null)
    if (params.get('m')) viewer.close()
  }
  // a filter change empties the panel: its box may no longer match. The first run keeps the panel the
  // viewer's Show on map link opened (compared by value, so StrictMode's double run is harmless too).
  const lastMq = useRef(JSON.stringify(mq))
  useEffect(() => {
    const key = JSON.stringify(mq)
    if (key === lastMq.current) return
    lastMq.current = key
    setSel(null)
  }, [mq])

  const total = points.data?.total ?? 0
  const withGps = points.data?.with_gps ?? 0
  const countLine = points.data
    ? total === 0
      ? 'No items match these filters.'
      : `${formatCount(withGps, 'item')} of ${total.toLocaleString()} ${total === 1 ? 'has' : 'have'} a location.`
    : 'Loading...'
  const st = status.data

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-3 px-4 py-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Map</h1>
          <p className="text-sm text-ink-soft" aria-live="polite">
            {countLine}
          </p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <div className="flex items-center gap-2" role="group" aria-label="Basemap">
            <button
              type="button"
              aria-pressed={tiles === 'online'}
              onClick={() => pickTiles(tiles === 'online' ? 'offline' : 'online')}
              className={
                'inline-flex h-10 items-center gap-1.5 rounded-full border px-4 text-sm font-semibold transition-colors ' +
                (tiles === 'online' ? 'border-ink bg-ink text-surface' : 'border-line bg-surface text-ink-soft hover:border-ink-faint hover:text-ink')
              }
            >
              <Globe className="size-4" aria-hidden />
              Detailed map
            </button>
          </div>
          <p className="max-w-xs text-right text-xs text-ink-faint">
            {tiles === 'online'
              ? 'Detailed tiles come from OpenFreeMap over the internet; your photos never leave this server.'
              : 'The plain map works offline. Detailed tiles need the internet.'}
          </p>
        </div>
      </header>

      <FilterBar filters={filters} set={set} clearAll={clearAll} vault={false} searching={false} />

      {st && (st.unchecked > 0 || !st.gazetteer || st.running) && (
        <div className="flex flex-wrap items-center gap-3 rounded-xl bg-surface-2 px-3 py-2 text-sm text-ink-soft" role="status">
          {!st.gazetteer && <span>Place names are off: the GeoNames files are missing from the models folder (see the README).</span>}
          {st.running ? (
            <span className="inline-flex items-center gap-2">
              <RefreshCw className="size-4 animate-spin" aria-hidden />
              Scanning older items for locations{st.unchecked > 0 ? `, ${st.unchecked.toLocaleString()} to go` : ''}.
            </span>
          ) : st.unchecked > 0 ? (
            <>
              <span>{formatCount(st.unchecked, 'older item')} {st.unchecked === 1 ? 'has' : 'have'} not been checked for a location yet.</span>
              <Button size="sm" variant="secondary" loading={refresh.isPending} onClick={() => refresh.mutate()}>
                Scan now
              </Button>
            </>
          ) : null}
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[16rem_minmax(0,1fr)]">
        <aside className="hidden lg:block" aria-label="Places">
          <h2 className="mb-2 text-sm font-semibold text-ink-soft">Places</h2>
          <div className="max-h-[60dvh] overflow-y-auto pr-1">
            <PlacesList places={places.data ?? []} loading={places.isPending} onPick={pickPlace} active={activePlace} />
          </div>
        </aside>

        <div className="relative h-[60dvh] min-h-[22rem] overflow-hidden rounded-[var(--radius-card)] border border-line bg-surface-2">
          {points.isError ? (
            <ErrorState error={points.error} onRetry={() => void points.refetch()} />
          ) : (
            <MapView points={points.data?.points ?? []} tiles={tiles} styleUrl={styleUrl} focus={focus} onSelect={setSel} />
          )}
          {points.data && withGps === 0 && !points.isError && (
            <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center p-3">
              <div className="pointer-events-auto rounded-xl bg-surface/95 px-4 py-2 text-sm text-ink-soft shadow">
                {total === 0 ? 'Nothing matches these filters.' : 'None of these items has a location yet.'}
              </div>
            </div>
          )}
          <button
            type="button"
            onClick={() => setPlacesOpen(true)}
            className="absolute bottom-3 left-3 inline-flex h-10 items-center gap-1.5 rounded-full bg-surface px-4 text-sm font-semibold shadow lg:hidden"
          >
            <MapPin className="size-4" aria-hidden />
            Places{places.data ? ` (${places.data.length})` : ''}
          </button>

          {sel && (
            <section
              aria-label="Items here"
              className="absolute inset-x-0 bottom-0 flex max-h-[55%] flex-col rounded-t-2xl border-t border-line bg-surface shadow-lg"
            >
              <div className="flex items-center justify-between gap-2 px-4 py-2">
                <h2 className="text-sm font-semibold">
                  {list.total > 0 ? formatCount(list.total, 'item') : formatCount(sel.count, 'item')} here
                </h2>
                <button type="button" onClick={closePanel} aria-label="Close" className="flex size-9 items-center justify-center rounded-full hover:bg-surface-2">
                  <X className="size-4" aria-hidden />
                </button>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
                {list.error && !list.items.length ? (
                  <ErrorState error={list.error} onRetry={list.refetch} />
                ) : (
                  <MediaGrid
                    items={list.items}
                    onOpen={openItem}
                    hasMore={list.hasMore}
                    loadMore={list.loadMore}
                    loading={list.loading}
                    empty={<EmptyState title="Nothing here" />}
                  />
                )}
              </div>
            </section>
          )}
        </div>
      </div>

      <p className="text-xs text-ink-faint">
        Map data: Natural Earth (public domain). Place names: GeoNames (CC BY 4.0).
        {tiles === 'online' ? ' Tiles: OpenFreeMap, OpenMapTiles, OpenStreetMap contributors.' : ''}
      </p>

      <Modal open={placesOpen} onClose={() => setPlacesOpen(false)} title="Places">
        <PlacesList places={places.data ?? []} loading={places.isPending} onPick={pickPlace} active={activePlace} />
      </Modal>

      {viewer.uid && (
        <Viewer uid={viewer.uid} onClose={viewer.close} onPrev={viewer.onPrev} onNext={viewer.onNext} position={viewer.position} startAt={viewer.t} />
      )}
    </div>
  )
}
