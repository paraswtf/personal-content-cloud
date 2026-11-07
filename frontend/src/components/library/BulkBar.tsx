// Actions for the selected library items: favourite, category, keep (mis-clicks), vault, trash.
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Lock, Star, StarOff, Trash2, Unlock } from 'lucide-react'
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

interface Props {
  items: ItemCard[]
  selected: Set<string>
  onClear: () => void
  onSelectAll: () => void
  total: number
  vault: boolean
}

const DONE: Partial<Record<BulkAction, string>> = {
  delete: 'moved to the trash',
  keep: 'kept',
  favorite: 'added to favourites',
  unfavorite: 'removed from favourites',
  category: 'moved to a new category',
}

export default function BulkBar({ items, selected, onClear, onSelectAll, total, vault }: Props) {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const [confirmDelete, setConfirmDelete] = useState(false)
  const uids = [...selected]
  const anyMisclick = items.some((i) => selected.has(i.uid) && i.misclick)
  const allFav = uids.length > 0 && items.filter((i) => selected.has(i.uid)).every((i) => i.favorite)

  const bulk = useMutation({
    mutationFn: (v: { action: BulkAction; value?: string | null }) => api.items.bulk({ uids, action: v.action, value: v.value ?? null }),
    onSuccess: (r, v) => {
      const what =
        v.action === 'nsfw' ? (v.value === '1' ? 'moved to the vault' : 'moved out of the vault') : (DONE[v.action] ?? 'updated')
      toast.success(`${formatCount(r.updated, 'item')} ${what}.`)
      setConfirmDelete(false)
      onClear()
      void refresh()
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
        <Button variant="secondary" disabled={busy} onClick={() => bulk.mutate({ action: 'nsfw', value: vault ? '0' : '1' })}>
          {vault ? <Unlock className="size-4" aria-hidden /> : <Lock className="size-4" aria-hidden />}
          {vault ? 'Not private' : 'Mark private'}
        </Button>
        <Button variant="danger" disabled={busy} onClick={() => setConfirmDelete(true)}>
          <Trash2 className="size-4" aria-hidden />
          Delete
        </Button>
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
