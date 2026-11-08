// Settings > System > Video (docs/ARCHITECTURE.md section 27): the web copy encoder in use (NVENC,
// VAAPI, Quick Sync or libx264), the copy counts, Test encoders (probes again) and Re-run failed
// copies (queues the web copy for videos whose copy failed). `video` comes from GET /api/system.
import { useMutation } from '@tanstack/react-query'
import { Clapperboard } from 'lucide-react'
import { videoApi } from '../../../lib/video'
import type { VideoStatus } from '../../../lib/types'
import { Badge, Button } from '../../ui'
import { useToast } from '../../Toast'
import { Section, errorText } from './common'

const NAMES: Record<string, string> = {
  h264_nvenc: 'NVIDIA NVENC',
  h264_vaapi: 'VAAPI',
  h264_qsv: 'Intel Quick Sync',
  libx264: 'libx264 (software)',
}

export function encoderLine(v: VideoStatus): string {
  if (!v.encoder) return 'Not detected yet. The first video that needs a copy sets it, or press Test encoders.'
  const name = NAMES[v.encoder] ?? v.encoder
  if (!v.hw) return `${name} on the CPU`
  return v.device ? `${v.encoder} on ${v.device}` : `${v.encoder} (${name})`
}

export default function VideoCard({ video, onRefresh }: { video: VideoStatus | null; onRefresh: () => void }) {
  const toast = useToast()
  const detect = useMutation({
    mutationFn: () => videoApi.detect(),
    onSuccess: (v) => {
      toast.success(v.encoder ? `Web copies will use ${encoderLine(v)}.` : 'No encoder answered.')
      onRefresh()
    },
    onError: (e) => toast.error(errorText(e, 'The encoders could not be tested.')),
  })
  const requeue = useMutation({
    mutationFn: () => videoApi.requeue('failed'),
    onSuccess: (r) => {
      toast.success(r.requeued === 0 ? 'No failed copies to re-run.' : `${r.requeued} ${r.requeued === 1 ? 'copy' : 'copies'} queued again.`)
      onRefresh()
    },
    onError: (e) => toast.error(errorText(e, 'The copies could not be queued.')),
  })
  if (!video) return null
  const busy = detect.isPending || requeue.isPending
  const counts: [string, number][] = [
    ['done', video.done],
    ['pending', video.pending],
    ['queued', video.queued],
    ['running', video.running],
    ['failed', video.failed],
  ]
  return (
    <Section
      title="Video"
      description="Videos the browser cannot play as they are get an H.264 web copy, up to 1080p, made with the machine's encoder."
      aside={video.encoder ? <Badge tone={video.hw ? 'ok' : 'warn'}>{video.hw ? 'Hardware encoder' : 'Software encoder'}</Badge> : null}
    >
      <div className="flex items-start gap-3 text-sm">
        <Clapperboard className="mt-0.5 size-5 shrink-0 text-ink-faint" aria-hidden />
        <div className="min-w-0 flex-1">
          <p className="font-semibold">{encoderLine(video)}</p>
          {video.reason && <p className="mt-1 text-ink-soft">Not used: {video.reason}</p>}
          <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-ink-soft sm:grid-cols-5">
            {counts.map(([label, n]) => (
              <div key={label} className="flex items-baseline justify-between gap-2 sm:flex-col sm:items-start sm:gap-0">
                <dt className="capitalize">{label}</dt>
                <dd className={n > 0 && label === 'failed' ? 'font-semibold text-danger' : 'font-semibold text-ink'}>{n}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button variant="secondary" size="sm" disabled={busy} loading={detect.isPending} onClick={() => detect.mutate()}>
              Test encoders
            </Button>
            <Button variant="secondary" size="sm" disabled={busy || video.failed === 0} loading={requeue.isPending} onClick={() => requeue.mutate()}>
              Re-run failed copies
            </Button>
          </div>
        </div>
      </div>
    </Section>
  )
}
