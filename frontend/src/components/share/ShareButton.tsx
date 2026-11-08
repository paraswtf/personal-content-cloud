// The viewer's Share button (docs/ARCHITECTURE.md section 26): a link for the open item. The viewer
// is told when the dialog is open (onOpenChange) so its keyboard handling steps aside, as it does
// for the delete confirmation. Not rendered for vault items.
import { useEffect, useState } from 'react'
import { Share2 } from 'lucide-react'
import { iconButton } from '../viewer/shared'
import ShareModal from './ShareModal'

export default function ShareButton({ uid, onOpenChange }: { uid: string; onOpenChange?: (open: boolean) => void }) {
  const [open, setOpen] = useState(false)
  useEffect(() => {
    onOpenChange?.(open)
  }, [open, onOpenChange])
  useEffect(() => () => onOpenChange?.(false), [onOpenChange])
  return (
    <>
      <button type="button" onClick={() => setOpen(true)} className={iconButton} aria-label="Share" title="Share">
        <Share2 className="size-5" aria-hidden />
      </button>
      <ShareModal open={open} onClose={() => setOpen(false)} target={{ kind: 'items', uids: [uid] }} />
    </>
  )
}
