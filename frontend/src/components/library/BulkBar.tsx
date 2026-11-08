// Actions for the selected library items: favourite, category, add or remove a tag, people (move,
// remove, tag or rename who the items show, section 10), keep (mis-clicks), vault, trash. The library,
// search results, albums and the vault share this bar.
import AlbumBulkActions from '../albums/BulkActions'
import PeopleBulkAction from '../people/PeopleBulkAction'
import ShareBulkAction from '../share/BulkAction'
import { useId, useState, type FormEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Forward, Lock, Star, StarOff, Tag, Trash2, Unlock, X } from 'lucide-react'
import { api } from '../../lib/api'
import { CATEGORIES, CATEGORY_LABELS, type BulkAction, type ItemCard } from '../../lib/types'
import { formatCount } from '../../lib/format'
import { useRefreshLibrary } from '../../lib/hooks'
import Select from '../Select'
import { useToast } from '../Toast'
import { Confirm } from '../admin/AdminModal'
import { errorMessage } from '../admin/adminHooks'
import { SelectionBar } from '../admin/fields'
import { Button } from '../ui'
import { MAX_TAG_LENGTH, useInvalidateTags, useTagList } from '../viewer/TagsPanel'

interface Props {
  items: ItemCard[]
  selected: Set<string>
  onClear: () => void
  onSelectAll: () => void
  total: number
  vault: boolean
  /** The album page's id: its actions swap Add to album for Remove from album (section 24). */
  album?: number
}

const DONE: Partial<Record<BulkAction, string>> = {
  delete: 'moved to the trash',
  keep: 'kept',
  forward: 'set apart as forwards',
  unforward: 'kept in the library',
  favorite: 'added to favourites',
  unfavorite: 'removed from favourites',
  category: 'moved to a new category',
}

/** The small tag form under the actions: Add tag tags the selection, Remove tag untags it. */
function TagForm({
  mode,
  busy,
  onSubmit,
  onCancel,
}: {
  mode: 'tag' | 'untag'
  busy: boolean
  onSubmit: (name: string) => void
  onCancel: () => void
}) {
  const [text, setText] = useState('')
  const id = useId()
  const tags = useTagList()
  const name = text.trim()
  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (name && !busy) onSubmit(name)
  }
  const label = mode === 'tag' ? 'Tag to add' : 'Tag to remove'
  return (
    <form onSubmit={submit} className="flex w-full min-w-0 items-center gap-2" aria-label={mode === 'tag' ? 'Add a tag' : 'Remove a tag'}>
      <label htmlFor={id} className="sr-only">
        {label}
      </label>
      <input
        id={id}
        type="text"
        list={`${id}-tags`}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            e.preventDefault()
            onCancel()
          }
        }}
        maxLength={MAX_TAG_LENGTH}
        placeholder={mode === 'tag' ? 'Tag name, new or existing' : 'Tag name'}
        autoComplete="off"
        autoFocus
        enterKeyHint="done"
        className="h-10 min-w-0 flex-1 rounded-full border border-line bg-surface px-4 text-base text-ink placeholder:text-ink-faint hover:border-ink-faint focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20 sm:text-sm"
      />
      <datalist id={`${id}-tags`}>
        {(tags.data ?? []).map((t) => (
          <option key={t.id} value={t.name} />
        ))}
      </datalist>
      <Button type="submit" variant={mode === 'tag' ? 'primary' : 'secondary'} loading={busy} disabled={!name}>
        {mode === 'tag' ? 'Add' : 'Remove'}
      </Button>
      <button
        type="button"
        onClick={onCancel}
        aria-label="Cancel"
        className="flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft hover:bg-ink/5 hover:text-ink"
      >
        <X className="size-4" aria-hidden />
      </button>
    </form>
  )
}

export default function BulkBar({ items, selected, onClear, onSelectAll, total, vault, album }: Props) {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const invalidateTags = useInvalidateTags()
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [tagMode, setTagMode] = useState<'tag' | 'untag' | null>(null)
  const uids = [...selected]
  const anyMisclick = items.some((i) => selected.has(i.uid) && i.misclick)
  const anyForward = items.some((i) => selected.has(i.uid) && i.forward)
  const allFav = uids.length > 0 && items.filter((i) => selected.has(i.uid)).every((i) => i.favorite)

  const bulk = useMutation({
    mutationFn: (v: { action: BulkAction; value?: string | null }) => api.items.bulk({ uids, action: v.action, value: v.value ?? null }),
    onSuccess: (r, v) => {
      const what =
        v.action === 'nsfw'
          ? v.value === '1'
            ? 'moved to the vault'
            : 'moved out of the vault'
          : v.action === 'tag'
            ? `tagged ${v.value}`
            : v.action === 'untag'
              ? `no longer tagged ${v.value}`
              : (DONE[v.action] ?? 'updated')
      toast.success(`${formatCount(r.updated, 'item')} ${what}.`)
      setConfirmDelete(false)
      setTagMode(null)
      onClear()
      void refresh()
      if (v.action === 'tag' || v.action === 'untag') invalidateTags()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not update the selection.')),
  })
  const busy = bulk.isPending

  return (
    <>
      <SelectionBar count={selected.size} noun={selected.size === 1 ? 'item' : 'items'} onClear={onClear} onSelectAll={onSelectAll} allSelected={selected.size >= total} total={total}>
        <Button
         
          variant="secondary"
          disabled={busy}
          onClick={() => bulk.mutate({ action: allFav ? 'unfavorite' : 'favorite' })}
        >
          {allFav ? <StarOff className="size-4" aria-hidden /> : <Star className="size-4" aria-hidden />}
          {allFav ? 'Unfavourite' : 'Favourite'}
        </Button>
        {anyMisclick && (
          <Button variant="secondary" disabled={busy} onClick={() => bulk.mutate({ action: 'keep' })}>
            Keep
          </Button>
        )}
        <Button variant="secondary" disabled={busy} onClick={() => bulk.mutate({ action: anyForward ? 'unforward' : 'forward' })}>
          <Forward className="size-4" aria-hidden />
          {anyForward ? 'Not forwards' : 'Forwards'}
        </Button>
        {!vault && (
          <Select
            label="Set category"
            placeholder="Set category"
            value=""
           
            disabled={busy}
            onChange={(v) => v && bulk.mutate({ action: 'category', value: v })}
            options={CATEGORIES.filter((c) => c !== 'nsfw').map((c) => ({ value: c, label: CATEGORY_LABELS[c] }))}
          />
        )}
        <PeopleBulkAction uids={uids} vault={vault} busy={busy} onDone={onClear} />
        <Button variant="secondary" disabled={busy} aria-expanded={tagMode === 'tag'} onClick={() => setTagMode(tagMode === 'tag' ? null : 'tag')}>
          <Tag className="size-4" aria-hidden />
          Add tag
        </Button>
        <Button variant="secondary" disabled={busy} aria-expanded={tagMode === 'untag'} onClick={() => setTagMode(tagMode === 'untag' ? null : 'untag')}>
          Remove tag
        </Button>
        <Button variant="secondary" disabled={busy} onClick={() => bulk.mutate({ action: 'nsfw', value: vault ? '0' : '1' })}>
          {vault ? <Unlock className="size-4" aria-hidden /> : <Lock className="size-4" aria-hidden />}
          {vault ? 'Not private' : 'Mark private'}
        </Button>
        <AlbumBulkActions uids={[...selected]} vault={vault} album={album} busy={busy} onDone={onClear} />
        <ShareBulkAction uids={[...selected]} vault={vault} busy={busy} />
        <Button variant="danger" disabled={busy} onClick={() => setConfirmDelete(true)}>
          <Trash2 className="size-4" aria-hidden />
          Delete
        </Button>
        {tagMode && (
          <TagForm
            key={tagMode}
            mode={tagMode}
            busy={busy}
            onSubmit={(name) => bulk.mutate({ action: tagMode, value: name })}
            onCancel={() => setTagMode(null)}
          />
        )}
      </SelectionBar>
      <Confirm
        open={confirmDelete}
        title={`Move ${formatCount(selected.size, 'item')} to the trash?`}
        body="They leave the library now and can be restored from Trash until it is emptied."
        confirmLabel="Move to trash"
        danger
        loading={busy}
        onConfirm={() => bulk.mutate({ action: 'delete' })}
        onClose={() => !busy && setConfirmDelete(false)}
      />
    </>
  )
}
