// The viewer's Location row (docs/ARCHITECTURE.md section 25): the place name when the server has
// one, else the coordinates, and a Show on map link that opens the map at the spot with the item open.
import { Link } from 'react-router'
import { MapPin } from 'lucide-react'
import { formatCoords } from '../../lib/format'
import type { ItemDetail } from '../../lib/types'
import { Row } from '../viewer/InfoPanel'

export default function LocationRow({ detail }: { detail: ItemDetail }) {
  if (!detail.gps) return null
  const { lat, lon } = detail.gps
  const place = detail.place ?? null
  const to = `/map?at=${lat.toFixed(5)},${lon.toFixed(5)}&z=14&m=${encodeURIComponent(detail.uid)}`
  return (
    <Row
      label="Location"
      value={
        <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span title={place ? formatCoords(lat, lon) : undefined}>{place ?? formatCoords(lat, lon)}</span>
          <Link to={to} className="inline-flex items-center gap-1 text-white/70 underline-offset-2 hover:text-white hover:underline">
            <MapPin className="size-3.5" aria-hidden />
            Show on map
          </Link>
        </span>
      }
    />
  )
}
