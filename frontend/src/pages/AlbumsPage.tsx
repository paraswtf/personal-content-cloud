// /albums (docs/ARCHITECTURE.md section 24): the album cards (cover, name, count, date range), a sort
// menu, a New album dialog, and an empty state that says how albums are filled.
import { useCallback, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { BookImage, Plus } from 'lucide-react'
import { ALBUMS_KEY, albumsApi, type AlbumOut, type AlbumSort } from '../lib/albums'
import { ApiError } from '../lib/api'
import { formatCount } from '../lib/format'
import { useInfiniteSentinel } from '../lib/hooks'
import { useItemList } from '../lib/useItemList'
import { errorMessage } from '../components/admin/adminHooks'
import AlbumCard from '../components/albums/AlbumCard'
import AlbumDialog from '../components/albums/AlbumDialog'
import Select from '../components/Select'
import { usePageTitle } from '../components/shell/title'
import { useToast } from '../components/Toast'
import { Button, EmptyState, ErrorState } from '../components/ui'

const SORTS: { value: AlbumSort; label: string }[] = [
  { value: 'updated', label: 'Last changed' },
  { value: 'name', label: 'Name' },
  { value: 'created', label: 'Newest first' },
]
const GRID = 'grid grid-cols-2 gap-3 sm:grid-cols-3 sm:gap-4 md:grid-cols-4 lg:grid-cols-5'

export default function AlbumsPage() {
  usePageTitle('Albums')
  const navigate = useNavigate()
  const qc = useQueryClient()
  const toast = useToast()
  const [params, setParams] = useSearchParams()
  const raw = params.get('sort')
  const sort: AlbumSort = raw === 'name' || raw === 'created' ? raw : 'updated'
  const fetchPage = useCallback(
    (cursor: string | null, signal?: AbortSignal) => albumsApi.list({ sort, cursor, limit: 60 }, signal),
    [sort],
  )
  const list = useItemList<AlbumOut>([...ALBUMS_KEY, 'list', sort], fetchPage, { keyOf: (a) => a.id })
  const sentinel = useInfiniteSentinel(list.loadMore, list.hasMore && !list.loading)
  const [creating, setCreating] = useState(false)
  const create = useMutation({
    mutationFn: (v: { name: string; description: string }) => albumsApi.create(v),
    onSuccess: (album) => {
      void qc.invalidateQueries({ queryKey: [...ALBUMS_KEY] })
      setCreating(false)
      toast.success(`Album ${album.name} created.`)
      navigate(`/albums/${album.id}`)
    },
  })
  const createError = create.isError
    ? create.error instanceof ApiError && create.error.code === 'album_exists'
      ? 'An album with this name exists already.'
      : errorMessage(create.error, 'Could not create the album.')
    : null

  return (
    <div className="mx-auto w-full max-w-7xl px-4">
      <div className="flex flex-wrap items-end justify-between gap-3 pb-4 pt-6">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Albums</h1>
          <p className="mt-1 text-sm tabular-nums text-ink-soft" aria-live="polite">
            {list.loading && list.items.length === 0 ? 'Loading' : formatCount(list.total, 'album')}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select
            label="Sort albums"
            value={sort}
            onChange={(v) => {
              const next = new URLSearchParams(params)
              if (v === 'updated') next.delete('sort')
              else next.set('sort', v)
              setParams(next, { replace: true })
            }}
            options={SORTS}
          />
          <Button onClick={() => setCreating(true)}>
            <Plus className="size-4" aria-hidden />
            New album
          </Button>
        </div>
      </div>

      {list.error && list.items.length === 0 ? (
        <ErrorState error={list.error} onRetry={list.refetch} />
      ) : list.loading && list.items.length === 0 ? (
        <ul className={GRID} aria-busy="true" aria-label="Loading">
          {Array.from({ length: 10 }, (_, i) => (
            <li key={i} className="aspect-square animate-pulse rounded-2xl bg-surface-2" />
          ))}
        </ul>
      ) : list.items.length === 0 ? (
        <EmptyState
          icon={<BookImage className="size-8" aria-hidden />}
          title="No albums yet"
          action={
            <Button onClick={() => setCreating(true)}>
              <Plus className="size-4" aria-hidden />
              New album
            </Button>
          }
        >
          Select photos in the library and choose Add to album, or make an empty album here and fill it later.
        </EmptyState>
      ) : (
        <ul className={GRID}>
          {list.items.map((a) => (
            <AlbumCard key={a.id} album={a} />
          ))}
        </ul>
      )}
      <div ref={sentinel} className="h-px" />
      {list.hasMore && !list.loading && (
        <div className="flex justify-center py-6">
          <Button variant="secondary" onClick={list.loadMore}>
            Load more
          </Button>
        </div>
      )}

      <AlbumDialog
        open={creating}
        onClose={() => setCreating(false)}
        onSubmit={(v) => create.mutate(v)}
        busy={create.isPending}
        title="New album"
        submitLabel="Create"
        error={createError}
      />
    </div>
  )
}
