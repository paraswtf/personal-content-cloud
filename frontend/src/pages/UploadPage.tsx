// /upload: drop zone for any file, the in-app camera, files shared to the installed app
// (?shared=1, drained from lib/shared.ts) and the live queue with its summary (docs/ARCHITECTURE.md
// sections 6 and 14). The upload engine is a module singleton, so uploads continue while the owner
// browses elsewhere; settled files refresh the library, stats and documents queries.
import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Camera, CopyCheck, Info, ShieldCheck, Sparkles } from 'lucide-react'
import { api } from '../lib/api'
import { formatBytes } from '../lib/format'
import { takeSharedFiles } from '../lib/shared'
import { checkInterrupted, formatLimit, useUploader, type AddResult } from '../lib/uploader'
import type { UploadSource } from '../lib/types'
import { Button, Card, ErrorState, PageSpinner } from '../components/ui'
import { useToast } from '../components/Toast'
import DropZone from '../components/upload/DropZone'
import QueueRow from '../components/upload/QueueRow'
import SummaryBar from '../components/upload/SummaryBar'
import CameraCapture, { CameraFallbackInput, cameraSupported } from '../components/upload/CameraCapture'
import { usePageTitle } from '../components/shell/title'

function addNotice(r: AddResult, source: UploadSource): string {
  const out: string[] = []
  const what = source === 'share' ? 'shared ' : ''
  if (r.added) out.push(`${r.added.toLocaleString()} ${what}${r.added === 1 ? 'file' : 'files'} added to the queue.`)
  if (r.rejected)
    out.push(`${r.rejected.toLocaleString()} ${r.rejected === 1 ? 'file was' : 'files were'} skipped because ${r.rejected === 1 ? 'it is' : 'they are'} empty or too large.`)
  if (r.skipped) out.push(`${r.skipped.toLocaleString()} ${r.skipped === 1 ? 'file is' : 'files are'} already in the list.`)
  return out.join(' ')
}

const NOTES = [
  { icon: ShieldCheck, text: 'Originals are kept byte for byte. Downloads give back the exact file.' },
  { icon: Sparkles, text: 'Photos are sorted, faces and people found, and documents read and indexed after upload.' },
  { icon: CopyCheck, text: 'Exact copies of files already here are not stored twice.' },
]

const SETTLE_INVALIDATE = [['items'], ['stats'], ['documents'], ['uploads-interrupted']] as const

export default function UploadPage() {
  usePageTitle('Upload')
  const qc = useQueryClient()
  const toast = useToast()
  const config = useQuery({ queryKey: ['config'], queryFn: api.config, staleTime: 60_000 })
  const { items, summary, engine } = useUploader()
  const [params, setParams] = useSearchParams()
  const [notice, setNotice] = useState('')
  const [cameraOpen, setCameraOpen] = useState(false)
  const fallbackInput = useRef<HTMLInputElement>(null)
  const liveCamera = cameraSupported()

  // keep the engine in line with the server's limits
  useEffect(() => {
    if (!config.data) return
    engine.configure({
      maxUploadBytes: config.data.max_upload_bytes,
      chunkSize: config.data.chunk_size,
      accept: '*/*',
    })
  }, [config.data, engine])

  // refresh the library, counts and documents as files settle (debounced)
  useEffect(() => {
    let t: ReturnType<typeof setTimeout> | null = null
    const off = engine.onSettled(() => {
      if (t) clearTimeout(t)
      t = setTimeout(() => {
        for (const key of SETTLE_INVALIDATE) void qc.invalidateQueries({ queryKey: [...key] })
      }, 1000)
    })
    return () => {
      off()
      if (t) clearTimeout(t)
    }
  }, [engine, qc])

  // a file dropped next to the drop zone must not make the browser navigate away to it
  useEffect(() => {
    const stop = (e: DragEvent) => {
      if (Array.from(e.dataTransfer?.types ?? []).includes('Files')) e.preventDefault()
    }
    window.addEventListener('dragover', stop)
    window.addEventListener('drop', stop)
    return () => {
      window.removeEventListener('dragover', stop)
      window.removeEventListener('drop', stop)
    }
  }, [])

  const noticeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(
    () => () => {
      if (noticeTimer.current) clearTimeout(noticeTimer.current)
    },
    [],
  )
  const addFiles = useCallback(
    (files: File[], source: UploadSource) => {
      const r = engine.add(files, source)
      setNotice(addNotice(r, source))
      if (noticeTimer.current) clearTimeout(noticeTimer.current)
      noticeTimer.current = setTimeout(() => setNotice(''), 8000)
    },
    [engine],
  )
  const onFiles = useCallback((files: File[]) => addFiles(files, 'web'), [addFiles])
  const onCameraFiles = useCallback((files: File[]) => addFiles(files, 'camera'), [addFiles])
  const onCapture = useCallback((file: File) => engine.add([file], 'camera'), [engine])

  // files shared to the installed app wait in IndexedDB until this page takes them
  const shared = params.get('shared') === '1'
  useEffect(() => {
    if (!shared) return
    let cancelled = false
    takeSharedFiles()
      .then((files) => {
        // the store is already cleared, so queue the files even if the page has gone; the engine
        // outlives it. Only the page's own state updates are skipped.
        if (cancelled) {
          if (files.length) engine.add(files, 'share')
          return
        }
        if (files.length) addFiles(files, 'share')
        else setNotice('Nothing new was shared. Files shared earlier are already in the list below.')
      })
      .catch(() => {
        if (!cancelled) toast.error('The shared files could not be read. Share them again, or choose them here.')
      })
      .finally(() => {
        if (cancelled) return
        const next = new URLSearchParams(params)
        next.delete('shared')
        setParams(next, { replace: true })
      })
    return () => {
      cancelled = true
    }
    // run once per arrival; params and setParams change when the flag is removed
  }, [shared])

  const interrupted = useQuery({
    queryKey: ['uploads-interrupted'],
    queryFn: checkInterrupted,
    staleTime: Infinity,
    retry: false,
  })
  const queuedIds = new Set(items.map((i) => i.uploadId).filter((id): id is string => !!id))
  const stillInterrupted = (interrupted.data ?? []).filter((e) => !queuedIds.has(e.id))

  const onCancel = useCallback((key: string) => engine.cancel(key), [engine])
  const onRetry = useCallback((key: string) => engine.retry(key), [engine])
  const onRemove = useCallback((key: string) => engine.remove(key), [engine])

  const openCamera = () => {
    if (liveCamera) setCameraOpen(true)
    else fallbackInput.current?.click()
  }
  const useSystemCamera = () => {
    setCameraOpen(false)
    fallbackInput.current?.click()
  }

  if (config.isPending) return <PageSpinner />
  if (config.isError) return <ErrorState error={config.error} onRetry={() => void config.refetch()} />
  const cfg = config.data

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-6 sm:py-10">
      <header className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Upload</h1>
        <p className="text-sm text-ink-soft sm:text-base">
          Add photos, videos and documents from this device, or take them with the camera.
        </p>
      </header>

      <DropZone
        accept={cfg.accept}
        maxLabel={formatLimit(cfg.max_upload_bytes)}
        onFiles={onFiles}
        extra={
          <Button size="lg" variant="secondary" type="button" onClick={openCamera}>
            <Camera className="size-5" aria-hidden />
            Take a photo or video
          </Button>
        }
      />
      <CameraFallbackInput ref={fallbackInput} onFiles={onCameraFiles} />
      <CameraCapture
        open={cameraOpen}
        onClose={() => setCameraOpen(false)}
        onCapture={onCapture}
        onUseSystemCamera={useSystemCamera}
      />

      <p className={notice ? '-mt-3 text-sm text-ink-soft' : 'sr-only'} role="status" aria-live="polite">
        {notice}
      </p>

      {stillInterrupted.length > 0 && (
        <Card className="flex gap-3 p-4">
          <Info className="mt-0.5 size-5 shrink-0 text-warn" aria-hidden />
          <div className="min-w-0 text-sm">
            <p className="font-semibold">
              {stillInterrupted.length === 1 ? 'An upload was interrupted' : `${stillInterrupted.length} uploads were interrupted`}
            </p>
            <p className="text-ink-soft">
              Choose the same {stillInterrupted.length === 1 ? 'file' : 'files'} again and{' '}
              {stillInterrupted.length === 1 ? 'it continues' : 'they continue'} where {stillInterrupted.length === 1 ? 'it' : 'they'}{' '}
              stopped:
            </p>
            <ul className="mt-1 list-inside list-disc text-ink-soft">
              {stillInterrupted.slice(0, 5).map((e) => (
                <li key={e.id} className="truncate">
                  {e.name} <span className="text-ink-faint">({formatBytes(e.size)})</span>
                </li>
              ))}
              {stillInterrupted.length > 5 && <li>and {stillInterrupted.length - 5} more</li>}
            </ul>
          </div>
        </Card>
      )}

      {items.length > 0 && (
        <section aria-labelledby="upload-queue">
          <h2 id="upload-queue" className="sr-only">
            Upload queue
          </h2>
          <Card className="overflow-hidden">
            <SummaryBar items={items} summary={summary} onClearFinished={() => engine.clearFinished()} />
            <ul className="divide-y divide-line">
              {items.map((item) => (
                <QueueRow
                  key={item.key}
                  item={item}
                  canCancel={engine.canCancel(item)}
                  canRetry={engine.canRetry(item)}
                  onCancel={onCancel}
                  onRetry={onRetry}
                  onRemove={onRemove}
                />
              ))}
            </ul>
          </Card>
          {summary.total > 0 && summary.sending === 0 && summary.processing === 0 && summary.added > 0 && (
            <p className="mt-3 text-center text-sm text-ink-soft">
              New files are in the library. Sorting and indexing may take a little longer.{' '}
              <Link to="/" className="font-semibold text-accent underline-offset-2 hover:underline">
                Open the library
              </Link>
            </p>
          )}
        </section>
      )}

      <ul className="grid gap-2 sm:grid-cols-3 sm:gap-3">
        {NOTES.map(({ icon: Icon, text }) => (
          <li key={text} className="flex items-start gap-2.5 rounded-xl bg-surface-2 px-3 py-2.5 text-sm text-ink-soft">
            <Icon className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden />
            <span>{text}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
