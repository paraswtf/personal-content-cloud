// The album page's Share album button (docs/ARCHITECTURE.md section 26): opens ShareModal for the album.
import { useState } from 'react'
import { Share2 } from 'lucide-react'
import { Button } from '../ui'
import ShareModal from './ShareModal'

export default function ShareAlbumButton({ albumId, title }: { albumId: number; title: string }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <Button variant="secondary" onClick={() => setOpen(true)}>
        <Share2 className="size-4" aria-hidden />
        Share album
      </Button>
      <ShareModal open={open} onClose={() => setOpen(false)} target={{ kind: 'album', albumId }} defaultTitle={title} />
    </>
  )
}
