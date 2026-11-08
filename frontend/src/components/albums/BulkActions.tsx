// Bulk bar actions for albums (docs/ARCHITECTURE.md section 24), rendered inside BulkBar's SelectionBar:
// Add to album (opens AlbumPicker) everywhere, Remove from album on an album page. Both go through the
// album and unalbum bulk actions, then clear the selection and refresh the library.
import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { BookImage } from 'lucide-react'
import { api } from '../../lib/api'
import { albumKey, type AlbumOut } from '../../lib/albums'
import { formatCount } from '../../lib/format'
import { useRefreshLibrary } from '../../lib/hooks'
import { errorMessage } from '../admin/adminHooks'
import { Button } from '../ui'
import { useToast } from '../Toast'
import AlbumPicker from './AlbumPicker'

export default function AlbumBulkActions({
  uids,
  album,
  busy,
  onDone,
}: {
  uids: string[]
  vault: boolean
  album?: number
  busy: boolean
  onDone: () => void
}) {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const qc = useQueryClient()
  const [open, setOpen] = useState(false)
  const act = useMutation({
    mutationFn: (v: { albumId: number; name: string; action: 'album' | 'unalbum' }) =>
      api.items.bulk({ uids, action: v.action, value: String(v.albumId) }).then((r) => ({ ...r, ...v })),
    onSuccess: (r) => {
      toast.success(
        r.action === 'album'
          ? `${formatCount(r.updated, 'item')} added to ${r.name}.`
          : `${formatCount(r.updated, 'item')} removed from ${r.name}.`,
      )
      setOpen(false)
      onDone()
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not change the album.')),
  })
  const pending = busy || act.isPending
  if (album !== undefined) {
    const name = qc.getQueryData<AlbumOut>(albumKey(album))?.name ?? 'the album'
    return (
      <Button variant="secondary" disabled={pending} onClick={() => act.mutate({ albumId: album, name, action: 'unalbum' })}>
        <BookImage className="size-4" aria-hidden />
        Remove from album
      </Button>
    )
  }
  return (
    <>
      <Button variant="secondary" disabled={pending} aria-haspopup="dialog" onClick={() => setOpen(true)}>
        <BookImage className="size-4" aria-hidden />
        Add to album
      </Button>
      <AlbumPicker
        open={open}
        onClose={() => setOpen(false)}
        busy={act.isPending}
        onPick={(a) => act.mutate({ albumId: a.id, name: a.name, action: 'album' })}
      />
    </>
  )
}
