// /map (docs/ARCHITECTURE.md section 25).
import { usePageTitle } from '../components/shell/title'

export default function MapPage() {
  usePageTitle('Map')
  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-8">
      <h1 className="text-2xl font-bold tracking-tight">Map</h1>
      <p className="mt-2 text-sm text-ink-soft">The map is being built.</p>
    </div>
  )
}
