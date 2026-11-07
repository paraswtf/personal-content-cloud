// /trash: items deleted in the last TRASH_DAYS days (api.trash.list), as a plain grid of thumbnails.
// Each item can be restored or deleted for good; "Empty trash" deletes everything here for good.
// Private (NSFW) items show a locked tile: their thumbnails are only served inside the vault.
import { useMemo, useState } from 'react'
import { useInfiniteQuery, useMutation, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { EyeOff, File as FileIcon, FileText, Film, RotateCcw, Trash2 } from 'lucide-react'
import { api } from '../lib/api'
import { formatDuration } from '../lib/format'
import { useRefreshLibrary } from '../lib/hooks'
import type { ItemCard } from '../lib/types'
import { Button, ConfirmModal, EmptyState, ErrorState, PageSpinner } from '../components/ui'
import { useToast } from '../components/Toast'
import { usePageTitle } from '../components/shell/title'

const PAGE = 60

function errorText(e: unknown, fallback: string): string {
  return e instanceof Error && e.message ? e.message : fallback
}

function Tile({
  item,
  busy,
  onRestore,
  onPurge,
}: {
  item: ItemCard
  busy: boolean
  onRestore: () => void
  onPurge: () => void
}) {
  const [broken, setBroken] = useState(false)
  const isDoc = item.kind === 'pdf' || item.kind === 'document'
  const Icon = isDoc ? FileText : item.kind === 'video' ? Film : FileIcon
  return (
    <li className="flex flex-col overflow-hidden rounded-xl bg-surface shadow-[var(--shadow-card)]">
      <div className="relative aspect-square bg-surface-2">
        {item.nsfw ? (
          <div className="flex size-full flex-col items-center justify-center gap-1 text-ink-faint">
            <EyeOff className="size-6" aria-hidden />
            <span className="text-xs font-medium">Private</span>
          </div>
        ) : broken ? (
          <div className="flex size-full items-center justify-center text-ink-faint">
            <Icon className="size-7" aria-hidden />
          </div>
        ) : (
          <img
            src={item.urls.thumb}
            alt=""
            loading="lazy"
            decoding="async"
            onError={() => setBroken(true)}
            className={clsx('size-full', isDoc ? 'bg-white object-contain p-1' : 'object-cover')}
          />
        )}
        {item.kind === 'video' && item.duration_s ? (
          <span className="absolute bottom-1.5 right-1.5 rounded-md bg-black/65 px-1.5 py-0.5 text-[11px] font-semibold tabular-nums text-white">
            {formatDuration(item.duration_s)}
          </span>
        ) : null}
      </div>
      <p className="truncate px-2 pt-2 text-xs font-medium" title={item.filename}>
        {item.filename}
      </p>
      <div className="flex items-center gap-1 p-1">
        <button
          type="button"
          onClick={onRestore}
          disabled={busy}
          className="inline-flex h-10 flex-1 items-center justify-center gap-1.5 rounded-lg text-sm font-semibold text-ink hover:bg-ink/5 disabled:opacity-50"
        >
          <RotateCcw className="size-4" aria-hidden />
          Restore
          <span className="sr-only"> {item.filename}</span>
        </button>
        <button
          type="button"
          onClick={onPurge}
          disabled={busy}
          aria-label={`Delete ${item.filename} forever`}
          title="Delete forever"
          className="inline-flex size-10 shrink-0 items-center justify-center rounded-lg text-danger hover:bg-danger/10 disabled:opacity-50"
        >
          <Trash2 className="size-4" aria-hidden />
        </button>
      </div>
    </li>
  )
}

export default function TrashPage() {
  usePageTitle('Trash')
  const refreshLibrary = useRefreshLibrary()
  const toast = useToast()
  const config = useQuery({ queryKey: ['config'], queryFn: api.config, staleTime: 60_000 })
  const list = useInfiniteQuery({
    queryKey: ['trash'],
    queryFn: ({ pageParam }) => api.trash.list(pageParam, PAGE),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
  })
  const [purging, setPurging] = useState<ItemCard | null>(null)
  const [emptying, setEmptying] = useState(false)
  const [busyUid, setBusyUid] = useState<string | null>(null)

  const items = useMemo(() => {
    const seen = new Set<string>()
    const out: ItemCard[] = []
    for (const p of list.data?.pages ?? [])
      for (const it of p.items) {
        if (seen.has(it.uid)) continue
        seen.add(it.uid)
        out.push(it)
      }
    return out
  }, [list.data])
  const total = list.data?.pages[0]?.total ?? items.length

  // restoring or purging changes people counts, person lists and review bins as well as the lists
  const refresh = refreshLibrary

  const restore = useMutation({
    mutationFn: (uid: string) => api.items.restore(uid),
    onMutate: (uid) => setBusyUid(uid),
    onSuccess: (_d, uid) => {
      const it = items.find((i) => i.uid === uid)
      toast.success(it ? `${it.filename} is back in the library.` : 'Restored.')
      void refresh()
    },
    onError: (e) => toast.error(errorText(e, 'Could not restore. Try again.')),
    onSettled: () => setBusyUid(null),
  })

  const purge = useMutation({
    mutationFn: (uid: string) => api.trash.purge(uid),
    onMutate: (uid) => setBusyUid(uid),
    onSuccess: () => {
      setPurging(null)
      toast.success('Deleted for good.')
      void refresh()
    },
    onError: (e) => toast.error(errorText(e, 'Could not delete. Try again.')),
    onSettled: () => setBusyUid(null),
  })

  const empty = useMutation({
    mutationFn: api.trash.empty,
    onSuccess: (r) => {
      setEmptying(false)
      toast.success(r.updated === 1 ? '1 item deleted for good.' : `${r.updated.toLocaleString()} items deleted for good.`)
      void refresh()
    },
    onError: (e) => toast.error(errorText(e, 'Could not empty the trash. Try again.')),
  })

  const days = config.data?.trash_days ?? 30

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-6">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-x-4 gap-y-3">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold tracking-tight">Trash</h1>
          <p className="mt-1 max-w-2xl text-sm text-ink-soft">
            Deleted items stay here for {days} days, then they and their files are removed for good. Restore puts an item
            back exactly as it was.
          </p>
        </div>
        {items.length > 0 && (
          <Button variant="danger" onClick={() => setEmptying(true)}>
            <Trash2 className="size-4" aria-hidden />
            Empty trash
          </Button>
        )}
      </div>

      {list.isPending ? (
        <PageSpinner />
      ) : list.isError ? (
        <ErrorState error={list.error} onRetry={() => void list.refetch()} />
      ) : items.length === 0 ? (
        <EmptyState icon={<Trash2 className="size-10" aria-hidden />} title="The trash is empty">
          Items you delete appear here for {days} days before they are removed for good.
        </EmptyState>
      ) : (
        <>
          <p className="mb-3 text-sm text-ink-soft">
            {total.toLocaleString()} {total === 1 ? 'item' : 'items'}
          </p>
          <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
            {items.map((item) => (
              <Tile
                key={item.uid}
                item={item}
                busy={busyUid === item.uid}
                onRestore={() => restore.mutate(item.uid)}
                onPurge={() => setPurging(item)}
              />
            ))}
          </ul>
          {list.hasNextPage && (
            <div className="mt-6 flex justify-center">
              <Button variant="secondary" loading={list.isFetchingNextPage} onClick={() => void list.fetchNextPage()}>
                Load more
              </Button>
            </div>
          )}
        </>
      )}

      <ConfirmModal
        open={!!purging}
        title="Delete forever"
        body={
          <>
            <span className="font-semibold text-ink">{purging?.filename}</span> and all its files will be deleted for good.
            This cannot be undone.
          </>
        }
        confirmLabel="Delete forever"
        danger
        loading={purge.isPending}
        onConfirm={() => purging && purge.mutate(purging.uid)}
        onClose={() => setPurging(null)}
      />
      <ConfirmModal
        open={emptying}
        title="Empty the trash"
        body={`All ${total.toLocaleString()} ${total === 1 ? 'item' : 'items'} in the trash and their files will be deleted for good. This cannot be undone.`}
        confirmLabel="Empty trash"
        danger
        loading={empty.isPending}
        onConfirm={() => empty.mutate()}
        onClose={() => setEmptying(false)}
      />
    </div>
  )
}
