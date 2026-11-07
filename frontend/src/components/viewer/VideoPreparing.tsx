// Shown instead of the player while the server is still making the compressed streaming copy.
// The Viewer keeps polling the detail query and swaps in the player once urls.video arrives.
import { Download, Loader2 } from 'lucide-react'
import type { ItemDetail } from '../../lib/types'
import { formatBytes } from '../../lib/format'

export default function VideoPreparing({ detail }: { detail: ItemDetail }) {
  const w = detail.width || 16
  const h = detail.height || 9
  return (
    <div
      className="relative overflow-hidden bg-black bg-contain bg-center bg-no-repeat"
      style={{ aspectRatio: `${w} / ${h}`, width: `min(100cqw, calc(100cqh * ${w / h}))`, backgroundImage: `url("${detail.urls.preview_small}")` }}
    >
      <div className="absolute inset-0 flex items-center justify-center bg-black/55 p-4">
        <div className="flex max-w-sm flex-col items-center gap-3 text-center text-white" role="status">
          <Loader2 className="size-7 animate-spin text-white/85" aria-hidden />
          <p className="text-sm font-semibold">Preparing a lighter version for smooth playback.</p>
          <p className="text-sm text-white/75">This usually takes a minute.</p>
          {detail.urls.download && (
            <a href={detail.urls.download} download className="inline-flex items-center gap-2 text-sm text-white/80 underline underline-offset-2 hover:text-white">
              <Download className="size-4" aria-hidden />
              Download original ({formatBytes(detail.size_bytes)})
            </a>
          )}
        </div>
      </div>
    </div>
  )
}
