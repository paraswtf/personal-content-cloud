// The viewer's Albums section (docs/ARCHITECTURE.md section 24), rendered by InfoPanel on its dark panel:
// chips for the item's albums (each links to the album, can make the item the album's cover, or take
// the item out), and an inline picker to add it to an album. The picker is inline, not a Modal: the
// viewer owns Escape and Tab.
import { useEffect, useState } from 'react'
import { Link } from 'react-router'
import { useMutation } from '@tanstack/react-query'
import { Plus, Star, X } from 'lucide-react'
import { api } from '../../lib/api'
import { albumsApi } from '../../lib/albums'
import { useRefreshLibrary } from '../../lib/hooks'
import type { ItemDetail } from '../../lib/types'
import { errorMessage } from '../admin/adminHooks'
import { darkButton, useInvalidateItem } from '../viewer/shared'
import { AlbumPickerPanel } from './AlbumPicker'

export default function AlbumsSection({ detail, onToast }: { detail: ItemDetail; onToast: (message: string) => void }) {
  const invalidate = useInvalidateItem(detail.uid)
  const refresh = useRefreshLibrary()
  const [picking, setPicking] = useState(false)
  // While the picker is open, Escape closes it first: the viewer's own Escape handler (which would
  // close the whole viewer) listens on the document, so this one runs in the capture phase.
  useEffect(() => {
    if (!picking) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.preventDefault()
      e.stopImmediatePropagation()
      setPicking(false)
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [picking])
  const act = useMutation({
    mutationFn: (v: { albumId: number; name: string; action: 'album' | 'unalbum' }) =>
      api.items.bulk({ uids: [detail.uid], action: v.action, value: String(v.albumId) }).then(() => v),
    onSuccess: (v) => {
      onToast(v.action === 'album' ? `Added to ${v.name}.` : `Removed from ${v.name}.`)
      setPicking(false)
      invalidate()
    },
    onError: (e) => onToast(errorMessage(e, 'Could not change the album.')),
  })
  const cover = useMutation({
    mutationFn: (albumId: number) => albumsApi.update(albumId, { cover_item_uid: detail.uid }),
    onSuccess: (album) => {
      onToast(`Now the cover of ${album.name}.`)
      void refresh()
    },
    onError: (e) => onToast(errorMessage(e, 'Could not set the cover.')),
  })
  if (detail.deleted_at) return null
  const busy = act.isPending || cover.isPending
  return (
    <section className="border-t border-white/10 px-4 py-4" aria-label="Albums">
      <h3 className="mb-2.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-white/50">Albums</h3>
      {detail.albums.length > 0 ? (
        <ul className="flex flex-wrap gap-2">
          {detail.albums.map((a) => (
            <li key={a.id} className="inline-flex h-9 items-center gap-0.5 rounded-full border border-white/15 pl-3 pr-1 text-sm text-white">
              <Link to={`/albums/${a.id}`} className="max-w-[12rem] truncate hover:underline">
                {a.name}
              </Link>
              <button
                type="button"
                onClick={() => cover.mutate(a.id)}
                disabled={busy}
                className="flex size-7 items-center justify-center rounded-full text-white/60 hover:bg-white/10 hover:text-white disabled:opacity-50"
                aria-label={`Use as the cover of ${a.name}`}
                title="Use as cover"
              >
                <Star className="size-3.5" aria-hidden />
              </button>
              <button
                type="button"
                onClick={() => act.mutate({ albumId: a.id, name: a.name, action: 'unalbum' })}
                disabled={busy}
                className="flex size-7 items-center justify-center rounded-full text-white/60 hover:bg-white/10 hover:text-white disabled:opacity-50"
                aria-label={`Remove from ${a.name}`}
                title="Remove from this album"
              >
                <X className="size-3.5" aria-hidden />
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-white/60">Not in an album yet.</p>
      )}
      {picking ? (
        <div className="mt-3 rounded-2xl border border-white/10 p-3">
          <AlbumPickerPanel
            dark
            busy={act.isPending}
            excludeIds={detail.albums.map((a) => a.id)}
            onPick={(a) => act.mutate({ albumId: a.id, name: a.name, action: 'album' })}
          />
          <button type="button" className={darkButton('default', 'mt-3')} onClick={() => setPicking(false)}>
            Close
          </button>
        </div>
      ) : (
        <button type="button" className={darkButton('default', 'mt-3')} disabled={busy} onClick={() => setPicking(true)}>
          <Plus className="size-4" aria-hidden />
          Add to album
        </button>
      )}
    </section>
  )
}
