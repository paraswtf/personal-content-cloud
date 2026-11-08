// The bulk bar's Share button (docs/ARCHITECTURE.md section 26): opens ShareModal for the selection.
// Hidden in the vault, whose items are never shared.
import { useState } from 'react'
import { Share2 } from 'lucide-react'
import { Button } from '../ui'
import ShareModal from './ShareModal'

export default function ShareBulkAction({ uids, vault, busy }: { uids: string[]; vault: boolean; busy: boolean }) {
  const [open, setOpen] = useState(false)
  if (vault) return null
  return (
    <>
      <Button variant="secondary" disabled={busy || uids.length === 0} onClick={() => setOpen(true)}>
        <Share2 className="size-4" aria-hidden />
        Share
      </Button>
      <ShareModal open={open} onClose={() => setOpen(false)} target={{ kind: 'items', uids }} />
    </>
  )
}
