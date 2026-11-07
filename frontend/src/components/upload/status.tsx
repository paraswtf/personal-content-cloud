// Labels and badges for upload states, shared by the queue rows and the summary.
import type { ReactNode } from 'react'
import { Ban, CircleAlert, CircleCheck, CircleX, Clock, CopyCheck } from 'lucide-react'
import { Badge } from '../ui'
import type { UploadStatus } from '../../lib/types'

type Tone = 'neutral' | 'brand' | 'ok' | 'warn'

interface Outcome {
  label: string
  tone: Tone
  icon: ReactNode
}

const ICON = 'size-3.5'

export const OUTCOMES: Record<UploadStatus, Outcome> = {
  uploading: { label: 'Not finished', tone: 'neutral', icon: <Clock className={ICON} aria-hidden /> },
  queued: { label: 'Processing', tone: 'neutral', icon: <Clock className={ICON} aria-hidden /> },
  processing: { label: 'Processing', tone: 'neutral', icon: <Clock className={ICON} aria-hidden /> },
  stored: { label: 'Added', tone: 'ok', icon: <CircleCheck className={ICON} aria-hidden /> },
  duplicate: { label: 'Already in your library', tone: 'neutral', icon: <CopyCheck className={ICON} aria-hidden /> },
  rejected: { label: 'Not accepted', tone: 'warn', icon: <Ban className={ICON} aria-hidden /> },
  error: { label: 'Processing failed', tone: 'warn', icon: <CircleAlert className={ICON} aria-hidden /> },
  cancelled: { label: 'Cancelled', tone: 'neutral', icon: <CircleX className={ICON} aria-hidden /> },
}

export function StatusBadge({ status }: { status: UploadStatus }) {
  const o = OUTCOMES[status] ?? OUTCOMES.error
  return (
    <Badge tone={o.tone}>
      {o.icon}
      {o.label}
    </Badge>
  )
}

export function FailedBadge({ label = 'Not uploaded' }: { label?: string }) {
  return (
    <Badge tone="warn">
      <CircleAlert className={ICON} aria-hidden />
      {label}
    </Badge>
  )
}

export function CancelledBadge() {
  return <StatusBadge status="cancelled" />
}
