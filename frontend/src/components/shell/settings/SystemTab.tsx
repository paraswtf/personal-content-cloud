// Settings > System: GPU, models, queues, storage tiers, backups and helpers, from GET /api/system polled every 8 s
// while this tab is open (the hub's dashboard cards, adapted). Missing models can be downloaded from
// here; housekeeping runs the cleanup jobs now instead of waiting for their schedule, and failed
// uploads can be queued for processing again.
import VideoCard from './VideoCard'
import { BackupSection, StorageSection } from './StorageCards'
import ThroughputCard from './ThroughputCard'
import { useState, type ReactNode } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Cpu, Download, ListTodo, RefreshCw, RotateCcw, Server, Sparkles } from 'lucide-react'
import { api } from '../../../lib/api'
import { formatBytes } from '../../../lib/format'
import type { ModelStatus, SystemOut } from '../../../lib/types'
import { Badge, Button, ErrorState } from '../../ui'
import { useToast } from '../../Toast'
import { Dot, Section, errorText } from './common'

export const SYSTEM_KEY = ['system'] as const
const POLL_MS = 8000

const QUEUE_LABELS: Record<string, string> = {
  main: 'Analysis (photos, videos, faces)',
  text: 'Text and documents',
  video: 'Video previews',
}

function Stat({ label, value, tone }: { label: string; value: ReactNode; tone?: 'danger' }) {
  return (
    <div>
      <p className={clsx('text-2xl font-bold tabular-nums', tone === 'danger' && 'text-danger')}>{value}</p>
      <p className="text-sm text-ink-soft">{label}</p>
    </div>
  )
}

function modelState(m: ModelStatus): { text: string; tone: 'ok' | 'warn' | 'danger' | 'idle' } {
  if (m.error) return { text: 'Error', tone: 'danger' }
  if (m.downloading) return { text: 'Downloading', tone: 'warn' }
  if (!m.present) return { text: 'Missing', tone: 'danger' }
  if (m.loaded) return { text: 'Loaded', tone: 'ok' }
  return { text: 'Ready, loads on first use', tone: 'idle' }
}

function providerLabel(p: string | null): string {
  if (!p) return '-'
  if (p.startsWith('CUDA')) return 'GPU (CUDA)'
  if (p.startsWith('CPU')) return 'CPU'
  return p.replace(/ExecutionProvider$/, '')
}

export function useSystem(poll: boolean) {
  return useQuery({
    queryKey: SYSTEM_KEY,
    queryFn: () => api.system(),
    refetchInterval: poll ? POLL_MS : false,
    refetchIntervalInBackground: false,
  })
}

function queueBusy(s: SystemOut): boolean {
  return s.queues.queued + s.queues.running > 0
}

export default function SystemTab() {
  const qc = useQueryClient()
  const toast = useToast()
  const q = useSystem(true)
  const [downloading, setDownloading] = useState(false)
  const [cleaning, setCleaning] = useState(false)
  const [retrying, setRetrying] = useState(false)

  const download = async () => {
    setDownloading(true)
    try {
      await api.downloadModels()
      toast.info('Model download started. Progress shows here.')
      void qc.invalidateQueries({ queryKey: SYSTEM_KEY })
    } catch (e) {
      toast.error(errorText(e, 'Could not start the download.'))
    } finally {
      setDownloading(false)
    }
  }

  const housekeeping = async () => {
    setCleaning(true)
    try {
      await api.housekeeping()
      toast.success('Housekeeping started.')
      void qc.invalidateQueries({ queryKey: SYSTEM_KEY })
      void qc.invalidateQueries({ queryKey: ['stats'] })
      void qc.invalidateQueries({ queryKey: ['trash'] })
    } catch (e) {
      toast.error(errorText(e, 'Housekeeping failed.'))
    } finally {
      setCleaning(false)
    }
  }

  const retryFailed = async () => {
    setRetrying(true)
    try {
      const { requeued } = await api.system.retryFailed()
      if (requeued === 0) toast.info('There were no failed uploads to retry.')
      else toast.success(`${requeued.toLocaleString()} failed ${requeued === 1 ? 'upload was' : 'uploads were'} queued again.`)
      void qc.invalidateQueries({ queryKey: SYSTEM_KEY })
      void qc.invalidateQueries({ queryKey: ['stats'] })
    } catch (e) {
      toast.error(errorText(e, 'Could not retry the failed uploads.'))
    } finally {
      setRetrying(false)
    }
  }

  if (q.isPending)
    return (
      <div className="grid gap-4 sm:grid-cols-2" aria-busy="true" aria-label="Loading system status">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="h-36 animate-pulse rounded-[var(--radius-card)] bg-ink/5" />
        ))}
      </div>
    )
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />
  const s = q.data

  const missing = s.models.filter((m) => !m.present || m.error)
  const dl = s.download
  const dlRunning = !!dl?.running || s.models.some((m) => m.downloading)
  const byQueue = Object.entries(s.queues.by_queue ?? {})

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <Section
          title="Processor"
          aside={s.gpu.available ? <Badge tone="ok">GPU in use</Badge> : <Badge tone="warn">CPU only</Badge>}
        >
          <div className="flex items-start gap-3 text-sm">
            <Cpu className="mt-0.5 size-5 shrink-0 text-ink-faint" aria-hidden />
            <div className="min-w-0">
              <p className="font-semibold">{s.gpu.name || (s.gpu.available ? 'GPU' : 'No GPU found')}</p>
              <p className="text-ink-soft">
                {s.gpu.available
                  ? `Models run on ${providerLabel(s.gpu.provider)}.`
                  : 'Models run on the CPU. Analysis works, just more slowly.'}
              </p>
            </div>
          </div>
        </Section>

        <Section
          title="Server"
          aside={
            <Button variant="secondary" size="sm" className="h-10" onClick={() => void q.refetch()} loading={q.isFetching}>
              {!q.isFetching && <RefreshCw className="size-4" aria-hidden />}
              Refresh
            </Button>
          }
        >
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
            <div>
              <dt className="text-ink-soft">Version</dt>
              <dd className="font-semibold">{s.version}</dd>
            </div>
            <div>
              <dt className="text-ink-soft">Time zone</dt>
              <dd className="font-semibold">{s.timezone}</dd>
            </div>
            <div>
              <dt className="text-ink-soft">Background workers</dt>
              <dd className="flex items-center gap-2 font-semibold">
                <Dot tone={s.workers ? 'ok' : 'danger'} />
                {s.workers ? 'Running' : 'Stopped'}
              </dd>
            </div>
            <div>
              <dt className="text-ink-soft">ffmpeg (videos)</dt>
              <dd className="flex items-center gap-2 font-semibold">
                <Dot tone={s.ffmpeg ? 'ok' : 'danger'} />
                {s.ffmpeg ? 'Found' : 'Missing'}
              </dd>
            </div>
          </dl>
          {!s.ffmpeg && (
            <p className="mt-3 text-sm text-ink-soft">Without ffmpeg, videos are stored but get no previews, frames or faces.</p>
          )}
        </Section>
      </div>

      <Section
        title="Models"
        description="All analysis runs on this server. Nothing is sent to outside services."
        aside={
          missing.length > 0 || dlRunning ? (
            <Button onClick={() => void download()} loading={downloading || dlRunning} disabled={dlRunning}>
              {!(downloading || dlRunning) && <Download className="size-4" aria-hidden />}
              {dlRunning ? 'Downloading' : missing.length === 1 ? 'Download the missing model' : 'Download missing models'}
            </Button>
          ) : (
            <Badge tone="ok">All present</Badge>
          )
        }
      >
        {dl && (dl.running || dl.progress || dl.error) && (
          <p className={clsx('mb-3 rounded-xl px-3 py-2 text-sm', dl.error ? 'bg-danger/10 text-danger' : 'bg-surface-2 text-ink-soft')}>
            {dl.error ? `Download failed: ${dl.error}` : dl.progress || 'Downloading'}
          </p>
        )}
        <ul className="divide-y divide-line">
          {s.models.map((m) => {
            const st = modelState(m)
            return (
              <li key={m.key} className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2.5">
                <Sparkles className="size-4 shrink-0 text-ink-faint" aria-hidden />
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-semibold">{m.label || m.key}</p>
                  {m.error && <p className="text-xs text-danger">{m.error}</p>}
                </div>
                <span className="text-xs text-ink-soft tabular-nums">{m.size_bytes ? formatBytes(m.size_bytes) : ''}</span>
                <span className="w-24 text-xs text-ink-soft">{m.loaded ? providerLabel(m.provider) : ''}</span>
                <span className="flex min-w-36 items-center gap-2 text-xs font-medium">
                  <Dot tone={st.tone} />
                  {st.text}
                </span>
              </li>
            )
          })}
        </ul>
      </Section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section
          title="Processing queue"
          aside={
            queueBusy(s) ? (
              <Badge tone="warn">
                <ListTodo className="size-3" aria-hidden /> Working
              </Badge>
            ) : (
              <Badge tone="ok">Idle</Badge>
            )
          }
        >
          <div className="grid grid-cols-3 gap-4">
            <Stat label="Waiting" value={s.queues.queued.toLocaleString()} />
            <Stat label="Running" value={s.queues.running.toLocaleString()} />
            <Stat label="Failed" value={s.queues.failed.toLocaleString()} tone={s.queues.failed > 0 ? 'danger' : undefined} />
          </div>
          {byQueue.length > 0 && (
            <ul className="mt-4 flex flex-col gap-1 border-t border-line pt-3 text-sm">
              {byQueue.map(([name, v]) => (
                <li key={name} className="flex items-center justify-between gap-3">
                  <span className="text-ink-soft">{QUEUE_LABELS[name] ?? name}</span>
                  <span className="tabular-nums">
                    {v.queued.toLocaleString()} waiting, {v.running.toLocaleString()} running
                  </span>
                </li>
              ))}
            </ul>
          )}
          {s.queues.failed > 0 && (
            <p className="mt-3 text-sm text-ink-soft">
              Failed jobs were retried and gave up. Open the item and choose to process it again, or run housekeeping.
            </p>
          )}
        </Section>

        <VideoCard video={s.video} onRefresh={() => void q.refetch()} />
        <StorageSection s={s} />
        <BackupSection s={s} />
      </div>

      <ThroughputCard />

      <Section
        title="Failed uploads"
        description="Uploads whose processing failed, for example because ffmpeg was missing or timed out, are kept. Retry them once the cause is fixed."
        aside={
          <Button variant="secondary" onClick={() => void retryFailed()} loading={retrying}>
            {!retrying && <RotateCcw className="size-4" aria-hidden />}
            Retry failed uploads
          </Button>
        }
      />

      <Section
        title="Housekeeping"
        description="Runs the cleanup tasks now: unfinished uploads that expired, trash past its keep period and files no item uses. It also runs on its own schedule."
        aside={
          <Button variant="secondary" onClick={() => void housekeeping()} loading={cleaning}>
            {!cleaning && <Server className="size-4" aria-hidden />}
            Run housekeeping
          </Button>
        }
      />
    </div>
  )
}
