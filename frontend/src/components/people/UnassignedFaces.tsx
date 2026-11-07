// Drawer with the faces that belong to nobody yet: select some and assign them to a person (or a
// new one), or remove them as "not a face". Can also re-run the automatic grouping.
import { useCallback, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Ban, RefreshCw, UserPlus } from 'lucide-react'
import { api } from '../../lib/api'
import { formatCount, personName } from '../../lib/format'
import { useRefreshLibrary } from '../../lib/hooks'
import { useItemList } from '../../lib/useItemList'
import type { FaceListItem } from '../../lib/types'
import { useToast } from '../Toast'
import { AdminModal } from '../admin/AdminModal'
import { errorMessage, runBulk, useSelection } from '../admin/adminHooks'
import { Button, EmptyState, Spinner } from '../ui'
import FaceGrid from './FaceGrid'
import PersonChoiceModal from './PersonChoiceModal'
import { choiceBody } from './PersonPicker'

export default function UnassignedFaces({ open, onClose }: { open: boolean; onClose: () => void }) {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const fetchPage = useCallback((c: string | null) => api.people.unassignedFaces(c, 120), [])
  const list = useItemList<FaceListItem>(['faces-unassigned'], fetchPage, { enabled: open, keyOf: (f) => f.id })
  const sel = useSelection<number>()
  const [picking, setPicking] = useState(false)

  const assign = useMutation({
    mutationFn: (choice: Parameters<typeof choiceBody>[0]) => api.people.assignFaces({ face_ids: [...sel.selected], ...choiceBody(choice) }),
    onSuccess: (r) => {
      toast.success(`${formatCount(r.updated, 'face')} tagged${r.person ? ` as ${personName(r.person)}` : ''}.`)
      setPicking(false)
      sel.clear()
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not assign the faces.')),
  })
  const remove = useMutation({
    mutationFn: () => runBulk([...sel.selected], (id) => api.people.deleteFace(id)),
    onSuccess: (r) => {
      if (r.failed) toast.error(`${r.failed} could not be removed. ${errorMessage(r.firstError)}`)
      else toast.success(`${formatCount(r.done, 'face')} removed.`)
      sel.clear()
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not remove the faces.')),
  })
  const recluster = useMutation({
    mutationFn: () => api.people.recluster(),
    onSuccess: () => {
      toast.success('Grouping faces again. New groups appear in a minute.')
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not start grouping.')),
  })
  const busy = assign.isPending || remove.isPending

  return (
    <>
      <AdminModal
        open={open}
        onClose={onClose}
        wide
        title={`Unassigned faces${list.total ? ` (${list.total.toLocaleString()})` : ''}`}
        footer={
          <div className="flex w-full flex-wrap items-center gap-2">
            <span className="mr-auto text-sm text-ink-soft" aria-live="polite">
              {sel.count ? `${sel.count} selected` : 'Select faces to tag them'}
            </span>
            <Button variant="ghost" onClick={() => recluster.mutate()} loading={recluster.isPending}>
              <RefreshCw className="size-4" aria-hidden />
              Group again
            </Button>
            <Button variant="danger" disabled={!sel.count || busy} loading={remove.isPending} onClick={() => remove.mutate()}>
              <Ban className="size-4" aria-hidden />
              Not faces
            </Button>
            <Button disabled={!sel.count || busy} onClick={() => setPicking(true)}>
              <UserPlus className="size-4" aria-hidden />
              Tag as
            </Button>
          </div>
        }
      >
        {list.loading && !list.items.length ? (
          <div className="flex justify-center py-10">
            <Spinner />
          </div>
        ) : list.items.length === 0 ? (
          <EmptyState title="Every face belongs to someone">New faces that match nobody show up here.</EmptyState>
        ) : (
          <FaceGrid
            faces={list.items}
            selecting
            selected={sel.selected}
            onToggle={sel.toggle}
            onOpen={() => undefined}
            hasMore={list.hasMore}
            loadMore={list.loadMore}
            loadingMore={list.loading}
          />
        )}
      </AdminModal>
      <PersonChoiceModal
        open={picking}
        title={`Tag ${formatCount(sel.count, 'face')} as`}
        busy={assign.isPending}
        onPick={(c) => assign.mutate(c)}
        onClose={() => setPicking(false)}
      />
    </>
  )
}
