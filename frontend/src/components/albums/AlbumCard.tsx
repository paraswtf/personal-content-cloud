// One album on the albums page (docs/ARCHITECTURE.md section 24): cover, name, count and date range.
import { Link } from 'react-router'
import { BookImage, Lock } from 'lucide-react'
import { albumDateRange, type AlbumOut } from '../../lib/albums'
import { formatCount } from '../../lib/format'

export default function AlbumCard({ album }: { album: AlbumOut }) {
  const range = albumDateRange(album.oldest, album.newest)
  return (
    <li>
      <Link
        to={`/albums/${album.id}`}
        className="group flex flex-col gap-2 rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <span className="relative block aspect-square w-full overflow-hidden rounded-2xl bg-surface-2 shadow-[var(--shadow-card)] transition-shadow group-hover:ring-2 group-hover:ring-accent/40">
          {album.cover_url ? (
            <img src={album.cover_url} alt="" className="size-full object-cover" loading="lazy" decoding="async" />
          ) : (
            <span className="flex size-full items-center justify-center text-ink-faint">
              <BookImage className="size-10" aria-hidden />
            </span>
          )}
          {album.vault_count > 0 && (
            <span
              className="absolute bottom-2 right-2 inline-flex items-center gap-1 rounded-full bg-black/60 px-2 py-0.5 text-[11px] font-semibold text-white"
              title={`${formatCount(album.vault_count, 'private item')} in the vault`}
            >
              <Lock className="size-3" aria-hidden />
              {album.vault_count}
            </span>
          )}
        </span>
        <span className="min-w-0 px-0.5">
          <span className="block truncate text-sm font-semibold text-ink">{album.name}</span>
          <span className="block truncate text-xs text-ink-soft">
            {formatCount(album.count, 'item')}
            {range ? `, ${range}` : ''}
          </span>
        </span>
      </Link>
    </li>
  )
}
