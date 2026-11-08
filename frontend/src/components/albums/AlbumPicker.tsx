// Choose an album, or make a new one in place (docs/ARCHITECTURE.md section 24). AlbumPickerPanel is the
// content (search, the albums newest change first, Create new); AlbumPicker wraps it in a Modal for the
// bulk bar. The viewer uses the panel inline (dark) because the viewer owns Escape and Tab.
import { useState, type FormEvent } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { BookImage, Plus } from 'lucide-react'
import { ALBUMS_KEY, albumsApi, type AlbumOut } from '../../lib/albums'
import { formatCount } from '../../lib/format'
import { useDebounced } from '../../lib/hooks'
import { errorMessage } from '../admin/adminHooks'
import { Button, Modal, Spinner } from '../ui'

interface PanelProps {
  onPick: (album: AlbumOut) => void
  /** Albums the item is already in: shown, but not pickable. */
  excludeIds?: number[]
  busy?: boolean
  dark?: boolean
  onError?: (message: string) => void
}

export function AlbumPickerPanel({ onPick, excludeIds = [], busy = false, dark = false, onError }: PanelProps) {
  const qc = useQueryClient()
  const [q, setQ] = useState('')
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const query = useDebounced(q.trim(), 200)
  const list = useQuery({
    queryKey: [...ALBUMS_KEY, 'picker', query],
    queryFn: ({ signal }) => albumsApi.list({ q: query || undefined, sort: 'updated', limit: 50 }, signal),
  })
  const [error, setError] = useState('')
  const fail = (e: unknown) => {
    const text = errorMessage(e, 'Could not create the album.')
    setError(text)
    onError?.(text)
  }
  const create = useMutation({
    mutationFn: (name: string) => albumsApi.create({ name }),
    onSuccess: (album) => {
      void qc.invalidateQueries({ queryKey: [...ALBUMS_KEY] })
      setNewName('')
      setCreating(false)
      onPick(album)
    },
    onError: fail,
  })
  const submitNew = (e: FormEvent) => {
    e.preventDefault()
    const name = newName.trim()
    if (!name || create.isPending) return
    setError('')
    create.mutate(name)
  }
  const excluded = new Set(excludeIds)
  const input = clsx(
    'h-10 w-full rounded-full border px-4 text-sm focus-visible:outline-none focus-visible:ring-2',
    dark
      ? 'border-white/15 bg-white/5 text-white placeholder:text-white/40 focus-visible:ring-white/30'
      : 'border-line bg-paper text-ink placeholder:text-ink-faint focus-visible:border-brand focus-visible:ring-brand/20',
  )
  const row = clsx(
    'flex w-full items-center gap-3 rounded-xl px-2 py-1.5 text-left text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-50',
    dark ? 'text-white hover:bg-white/10' : 'text-ink hover:bg-surface-2',
  )
  const quiet = dark ? 'text-white/55' : 'text-ink-soft'
  const albums = list.data?.items ?? []
  return (
    <div className="flex flex-col gap-3">
      <input
        type="search"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Find an album"
        aria-label="Find an album"
        className={input}
        autoComplete="off"
      />
      <div className="max-h-64 overflow-y-auto">
        {list.isPending ? (
          <div className="py-4">
            <Spinner className={dark ? 'text-white/70' : undefined} />
          </div>
        ) : list.isError ? (
          <p className={clsx('px-2 py-3 text-sm', quiet)}>{errorMessage(list.error, 'Could not load the albums.')}</p>
        ) : albums.length === 0 ? (
          <p className={clsx('px-2 py-3 text-sm', quiet)}>{query ? 'No album matches.' : 'No albums yet. Make one below.'}</p>
        ) : (
          <ul className="flex flex-col gap-0.5" aria-label="Albums">
            {albums.map((a) => {
              const inAlready = excluded.has(a.id)
              return (
                <li key={a.id}>
                  <button type="button" className={row} disabled={busy || inAlready} onClick={() => onPick(a)}>
                    <span className={clsx('flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-lg', dark ? 'bg-white/10' : 'bg-surface-2')}>
                      {a.cover_url ? (
                        <img src={a.cover_url} alt="" className="size-full object-cover" loading="lazy" />
                      ) : (
                        <BookImage className={clsx('size-5', quiet)} aria-hidden />
                      )}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate font-semibold">{a.name}</span>
                      <span className={clsx('block text-xs', quiet)}>
                        {inAlready ? 'Already in this album' : formatCount(a.count, 'item')}
                      </span>
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </div>
      {creating ? (
        <form onSubmit={submitNew} className="flex flex-col gap-2 sm:flex-row">
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="New album name"
            aria-label="New album name"
            className={input}
            maxLength={120}
            autoFocus
          />
          <div className="flex gap-2">
            <Button type="submit" loading={create.isPending} disabled={!newName.trim() || busy}>
              Create
            </Button>
            <Button type="button" variant={dark ? 'ghost' : 'secondary'} onClick={() => setCreating(false)} className={dark ? 'text-white' : undefined}>
              Cancel
            </Button>
          </div>
        </form>
      ) : (
        <button
          type="button"
          onClick={() => setCreating(true)}
          disabled={busy}
          className={clsx(row, 'font-semibold', dark ? 'text-white' : 'text-accent')}
        >
          <span className={clsx('flex size-10 shrink-0 items-center justify-center rounded-lg', dark ? 'bg-white/10' : 'bg-accent-soft')}>
            <Plus className="size-5" aria-hidden />
          </span>
          Create a new album
        </button>
      )}
      {error && (
        <p role="alert" className="text-sm text-danger">
          {error}
        </p>
      )}
    </div>
  )
}

export default function AlbumPicker({
  open,
  onClose,
  title = 'Add to album',
  ...panel
}: PanelProps & { open: boolean; onClose: () => void; title?: string }) {
  return (
    <Modal open={open} onClose={onClose} title={title}>
      {open && <AlbumPickerPanel {...panel} />}
    </Modal>
  )
}
