// /albums/:id (docs/ARCHITECTURE.md section 24): the library with one album's members and the album header.
import { usePageTitle } from '../components/shell/title'

export default function AlbumPage() {
  usePageTitle('Album')
  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-8">
      <h1 className="text-2xl font-bold tracking-tight">Album</h1>
      <p className="mt-2 text-sm text-ink-soft">Albums are being built.</p>
    </div>
  )
}
