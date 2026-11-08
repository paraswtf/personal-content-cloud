// /albums/:id (docs/ARCHITECTURE.md section 24): the library with one album's members, under a header
// with the description, the count and date range, and the album's actions (Share album, Rename and
// description, Delete album). The bulk bar shows Remove from album here (Library passes `album`).
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { BookImage, ChevronLeft, Pencil, Trash2 } from 'lucide-react'
import { ALBUMS_KEY, albumDateRange, albumKey, albumsApi } from '../lib/albums'
import { ApiError } from '../lib/api'
import { formatCount } from '../lib/format'
import { useRefreshLibrary } from '../lib/hooks'
import { errorMessage } from '../components/admin/adminHooks'
import AlbumDialog from '../components/albums/AlbumDialog'
import Library from '../components/library/Library'
import ShareAlbumButton from '../components/share/ShareAlbumButton'
import { usePageTitle } from '../components/shell/title'
import { useToast } from '../components/Toast'
import { Button, ConfirmModal, EmptyState, ErrorState, PageSpinner } from '../components/ui'

export default function AlbumPage() {
  const { id } = useParams()
  const albumId = Number(id)
  const valid = Number.isInteger(albumId) && albumId > 0
  const navigate = useNavigate()
  const qc = useQueryClient()
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const q = useQuery({ queryKey: albumKey(albumId), queryFn: () => albumsApi.get(albumId), enabled: valid })
  const album = q.data
  usePageTitle(album?.name ?? 'Album')
  const [renaming, setRenaming] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const rename = useMutation({
    mutationFn: (v: { name: string; description: string }) => albumsApi.update(albumId, v),
    onSuccess: (next) => {
      qc.setQueryData(albumKey(albumId), next)
      void qc.invalidateQueries({ queryKey: [...ALBUMS_KEY] })
      setRenaming(false)
      toast.success('Album updated.')
    },
  })
  const remove = useMutation({
    mutationFn: () => albumsApi.remove(albumId),
    onSuccess: () => {
      setConfirmDelete(false)
      toast.success(`Album ${album?.name ?? ''} deleted. Its photos stay in the library.`)
      qc.removeQueries({ queryKey: albumKey(albumId) })
      void refresh()
      navigate('/albums', { replace: true })
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not delete the album.')),
  })
  const renameError = rename.isError
    ? rename.error instanceof ApiError && rename.error.code === 'album_exists'
      ? 'An album with this name exists already.'
      : errorMessage(rename.error, 'Could not update the album.')
    : null

  if (!valid || (q.isError && q.error instanceof ApiError && q.error.status === 404)) {
    return (
      <div className="mx-auto w-full max-w-7xl px-4 py-10">
        <EmptyState
          icon={<BookImage className="size-8" aria-hidden />}
          title="This album does not exist"
          action={
            <Link to="/albums" className="text-sm font-semibold text-accent underline-offset-2 hover:underline">
              All albums
            </Link>
          }
        >
          It may have been deleted.
        </EmptyState>
      </div>
    )
  }
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />
  if (!album) return <PageSpinner />

  const range = albumDateRange(album.oldest, album.newest)
  const facts = [formatCount(album.count, 'item'), range, album.vault_count > 0 ? formatCount(album.vault_count, 'private item') + ' in the vault' : '']
    .filter(Boolean)
    .join(', ')

  return (
    <>
      <div className="mx-auto w-full max-w-7xl px-4 pt-5">
        <Link to="/albums" className="inline-flex items-center gap-1 text-sm font-semibold text-ink-soft hover:text-ink">
          <ChevronLeft className="size-4" aria-hidden />
          Albums
        </Link>
      </div>
      <Library
        album={albumId}
        title={album.name}
        headerExtra={
          <div className="flex w-full flex-col gap-2 sm:w-auto sm:items-end">
            {(album.description || facts) && (
              <p className="max-w-xl text-sm text-ink-soft sm:text-right">
                {album.description && <span className="text-ink">{album.description}</span>}
                {album.description && facts ? ' ' : ''}
                {facts && <span>{facts}</span>}
              </p>
            )}
            <div className="flex flex-wrap items-center gap-2">
              <ShareAlbumButton albumId={albumId} title={album.name} />
              <Button variant="secondary" onClick={() => setRenaming(true)}>
                <Pencil className="size-4" aria-hidden />
                Rename
              </Button>
              <Button variant="secondary" onClick={() => setConfirmDelete(true)}>
                <Trash2 className="size-4" aria-hidden />
                Delete album
              </Button>
            </div>
          </div>
        }
      />
      <AlbumDialog
        open={renaming}
        onClose={() => setRenaming(false)}
        onSubmit={(v) => rename.mutate(v)}
        busy={rename.isPending}
        title="Rename album"
        submitLabel="Save"
        initial={{ name: album.name, description: album.description }}
        error={renameError}
      />
      <ConfirmModal
        open={confirmDelete}
        title={`Delete the album ${album.name}?`}
        body="The album goes away. Its photos, videos and documents stay in the library."
        confirmLabel="Delete album"
        danger
        loading={remove.isPending}
        onConfirm={() => remove.mutate()}
        onClose={() => !remove.isPending && setConfirmDelete(false)}
      />
    </>
  )
}
