// Totals for the current queue: overall progress and "18 added, 3 already in your library, ...".
import { CircleCheck, Loader2 } from 'lucide-react'
import type { QueueItem, UploadSummary } from '../../lib/uploader'
import { Button } from '../ui'

function plural(n: number, one: string, many: string) {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`
}

/** Bytes done over bytes to send, counting only files that are (or were) being uploaded. */
export function overallProgress(items: QueueItem[]): number {
  let total = 0
  let done = 0
  for (const i of items) {
    if (i.phase === 'failed' || i.phase === 'cancelled') continue
    total += i.size
    if (i.phase === 'uploading') done += i.size * (i.progress ?? 0)
    else if (i.phase === 'finishing' || i.phase === 'processing' || i.phase === 'done') done += i.size
  }
  return total ? done / total : 0
}

export default function SummaryBar({
  items,
  summary,
  onClearFinished,
}: {
  items: QueueItem[]
  summary: UploadSummary
  onClearFinished: () => void
}) {
  const parts: string[] = []
  if (summary.added) parts.push(`${summary.added.toLocaleString()} added`)
  if (summary.duplicates) parts.push(`${summary.duplicates.toLocaleString()} already in your library`)
  if (summary.failed) parts.push(`${summary.failed.toLocaleString()} not uploaded`)
  if (summary.cancelled) parts.push(`${summary.cancelled.toLocaleString()} cancelled`)

  const busy = summary.sending > 0 || summary.processing > 0
  const finishedCount = summary.total - summary.sending - summary.processing
  const pct = Math.round(overallProgress(items) * 100)

  let headline: string
  if (summary.sending > 0) headline = `Uploading ${plural(summary.sending, 'file', 'files')}, ${pct}%`
  else if (summary.processing > 0) headline = `Processing ${plural(summary.processing, 'file', 'files')}`
  else headline = 'All done'

  return (
    <div className="flex flex-col gap-3 border-b border-line px-3 py-3 sm:flex-row sm:items-center sm:px-4">
      <div className="flex min-w-0 flex-1 items-start gap-2.5">
        {busy ? (
          <Loader2 className="mt-0.5 size-5 shrink-0 animate-spin text-accent" aria-hidden />
        ) : (
          <CircleCheck className="mt-0.5 size-5 shrink-0 text-ok" aria-hidden />
        )}
        <div className="min-w-0">
          <p className="text-sm font-semibold">{headline}</p>
          {/* only the outcome counts are live: they change when a file settles, not on every progress tick */}
          <p className="text-sm text-ink-soft empty:hidden" aria-live="polite">
            {parts.join(' · ')}
          </p>
          {summary.sending > 0 && (
            <p className="text-xs text-ink-faint">
              Keep this tab open until uploading finishes. Analysis and indexing continue on the server after that.
            </p>
          )}
        </div>
      </div>
      {finishedCount > 0 && (
        <Button variant="secondary" onClick={onClearFinished} className="self-start sm:self-auto">
          Clear finished
        </Button>
      )}
    </div>
  )
}
