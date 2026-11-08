// The places beside the map (docs/ARCHITECTURE.md section 25): one row per place, most photos first;
// choosing one flies there.
import { MapPin } from 'lucide-react'
import type { PlaceOut } from '../../lib/map'
import { formatCount } from '../../lib/format'

export default function PlacesList({
  places,
  loading,
  onPick,
  active,
}: {
  places: PlaceOut[]
  loading: boolean
  onPick: (place: PlaceOut) => void
  active: string | null
}) {
  if (loading && places.length === 0) {
    return (
      <ul className="flex flex-col gap-1" aria-busy="true" aria-label="Loading places">
        {Array.from({ length: 6 }, (_, i) => (
          <li key={i} className="h-10 animate-pulse rounded-lg bg-surface-2" />
        ))}
      </ul>
    )
  }
  if (places.length === 0) {
    return <p className="text-sm text-ink-soft">No places yet. Photos with a location get a place name within a few minutes.</p>
  }
  return (
    <ul className="flex flex-col" role="list" aria-label="Places">
      {places.map((p) => {
        const on = active === p.place
        return (
          <li key={p.place}>
            <button
              type="button"
              onClick={() => onPick(p)}
              aria-pressed={on}
              className={
                'flex w-full items-start gap-2 rounded-lg px-2 py-2 text-left text-sm hover:bg-surface-2 ' +
                (on ? 'bg-surface-2 font-semibold' : '')
              }
            >
              <MapPin className="mt-0.5 size-4 shrink-0 text-ink-faint" aria-hidden />
              <span className="min-w-0 flex-1">
                <span className="block truncate">{p.place_city ?? p.place}</span>
                {(p.place_admin || p.place_country) && (
                  <span className="block truncate text-xs text-ink-soft">{[p.place_admin, p.place_country].filter(Boolean).join(', ')}</span>
                )}
              </span>
              <span className="shrink-0 text-xs tabular-nums text-ink-faint">{formatCount(p.count, 'item')}</span>
            </button>
          </li>
        )
      })}
    </ul>
  )
}
