// Settings > System: the storage tiers and the nightly backups (docs/ARCHITECTURE.md section 31).
// StorageSection lists each disk once with its free space, what it holds and, while the originals of an
// earlier layout are still being moved to ORIGINALS_DIR, that move's progress. BackupSection shows the
// last snapshot, the next run and a "Back up now" button (POST /api/system/backup).
import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Archive, HardDrive } from 'lucide-react'
import { api } from '../../../lib/api'
import { formatBytes, formatDate, formatRelative, formatUntil } from '../../../lib/format'
import type { BackupOut, StorageOut, SystemOut, TierOut } from '../../../lib/types'
import { Badge, Button } from '../../ui'
import { useToast } from '../../Toast'
import { Section, errorText } from './common'
import { SYSTEM_KEY } from './SystemTab'

const TIER_TITLES: Record<string, string> = {
  originals: 'Originals',
  data: 'Database and previews',
  backup: 'Backups',
}

function tierTitle(key: string): string {
  return TIER_TITLES[key] ?? key
}

function Meter({ pct, label, tone }: { pct: number; label: string; tone?: 'danger' }) {
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      className="h-2.5 overflow-hidden rounded-full bg-ink/10"
    >
      <div className={clsx('h-full rounded-full', tone === 'danger' ? 'bg-danger' : 'bg-ink')} style={{ width: `${pct}%` }} />
    </div>
  )
}

function usedPct(t: { free_bytes: number; total_bytes: number }): number {
  return t.total_bytes > 0 ? Math.min(100, Math.round((1 - t.free_bytes / t.total_bytes) * 100)) : 0
}

function lowSpace(t: { free_bytes: number; total_bytes: number }): boolean {
  return t.total_bytes > 0 && t.free_bytes / t.total_bytes < 0.1
}

function Tier({ tier }: { tier: TierOut }) {
  const low = lowSpace(tier)
  const pct = usedPct(tier)
  return (
    <li className="py-3 first:pt-0 last:pb-0">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <p className="flex items-center gap-2 text-sm font-semibold">
          <HardDrive className="size-4 text-ink-faint" aria-hidden />
          {tierTitle(tier.key)}
        </p>
        {tier.shared_with ? (
          <span className="text-xs text-ink-soft">Same disk as {tierTitle(tier.shared_with).toLowerCase()}</span>
        ) : (
          <span className={clsx('text-sm tabular-nums', low ? 'font-medium text-danger' : 'text-ink-soft')}>
            {formatBytes(tier.free_bytes)} free of {formatBytes(tier.total_bytes)}
          </span>
        )}
      </div>
      <p className="mt-0.5 text-xs text-ink-soft">
        <span className="break-all">{tier.path}</span>
      </p>
      <p className="text-xs text-ink-faint">Holds {tier.holds}.</p>
      {!tier.shared_with && (
        <>
          <div className="mt-2">
            <Meter pct={pct} label={`${tierTitle(tier.key)} disk used`} tone={low ? 'danger' : undefined} />
          </div>
          <p className={clsx('mt-1.5 text-xs', low ? 'font-medium text-danger' : 'text-ink-soft')}>
            {pct}% used
            {low && tier.key === 'originals' ? '. Free space is low; new uploads may be refused.' : ''}
            {low && tier.key === 'data' ? '. Free space is low; previews and the database need room here.' : ''}
            {low && tier.key === 'backup' ? '. Free space is low; old snapshots are removed after each run.' : ''}
          </p>
        </>
      )}
    </li>
  )
}

function Migration({ storage }: { storage: StorageOut }) {
  const m = storage.migration
  if (!m.active && !m.running) return null
  const total = m.pending_files
  const done = m.moved_files
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0
  return (
    <div className="mt-4 rounded-xl bg-surface-2 px-3 py-2.5 text-sm">
      <p className="font-medium">
        {m.running
          ? `Moving the originals to their own disk: ${done.toLocaleString()} of ${total.toLocaleString()} files (${formatBytes(m.moved_bytes)} of ${formatBytes(m.pending_bytes)})`
          : 'Some originals still sit in the data folder. They are moved when the server starts.'}
      </p>
      {m.running && (
        <div className="mt-2">
          <Meter pct={pct} label="Originals moved" />
        </div>
      )}
      {m.error && <p className="mt-1.5 text-xs text-danger">Last problem: {m.error}</p>}
    </div>
  )
}

export function StorageSection({ s }: { s: SystemOut }) {
  const storage = s.storage
  const tiers: TierOut[] = storage?.tiers ?? [
    { key: 'originals', path: '', holds: 'everything', free_bytes: s.disk.free_bytes, total_bytes: s.disk.total_bytes, shared_with: null },
  ]
  return (
    <Section
      title="Storage"
      description={
        storage?.tiered
          ? 'Originals live on their own disk; the database, previews and thumbnails stay with the app.'
          : 'One folder holds the database, the originals and the previews. Set ORIGINALS_DIR to put the originals on another disk.'
      }
    >
      <ul className="divide-y divide-line">
        {tiers.map((t) => (
          <Tier key={t.key} tier={t} />
        ))}
      </ul>
      <dl className="mt-4 border-t border-line pt-3 text-sm">
        <dt className="text-ink-soft">Original files</dt>
        <dd className="text-lg font-semibold">{formatBytes(s.disk.originals_bytes)}</dd>
      </dl>
      {storage && <Migration storage={storage} />}
    </Section>
  )
}

function lastLine(b: BackupOut): string {
  const last = b.last
  if (!last) return 'No snapshot yet.'
  const when = last.finished_at ? `${formatRelative(last.finished_at)} (${formatDate(last.finished_at, true)})` : 'just now'
  if (!last.ok) return `The last run ${when} failed: ${last.error ?? 'unknown error'}.`
  const linked = last.bytes_linked > 0 ? `, ${formatBytes(last.bytes_linked)} unchanged and shared with the night before` : ''
  return `Last snapshot ${when}: ${last.files.toLocaleString()} originals, ${formatBytes(last.bytes_copied)} copied${linked}.`
}

export function BackupSection({ s }: { s: SystemOut }) {
  const qc = useQueryClient()
  const toast = useToast()
  const [starting, setStarting] = useState(false)
  const b = s.backup
  const run = async () => {
    setStarting(true)
    try {
      const r = await api.system.backupNow()
      if (r.started) toast.success('Backup started. Progress shows here.')
      else if (r.running) toast.info('A backup is already running.')
      else toast.info('Backups are off: set BACKUP_DIR first.')
      void qc.invalidateQueries({ queryKey: SYSTEM_KEY })
    } catch (e) {
      toast.error(errorText(e, 'Could not start the backup.'))
    } finally {
      setStarting(false)
    }
  }
  if (!b || !b.enabled) {
    return (
      <Section title="Backups" aside={<Badge tone="warn">Off</Badge>}>
        <div className="flex items-start gap-3 text-sm text-ink-soft">
          <Archive className="mt-0.5 size-5 shrink-0 text-ink-faint" aria-hidden />
          <p>
            Set <code className="rounded bg-surface-2 px-1">BACKUP_DIR</code> to a folder on another disk and restart. A snapshot of
            the database, the originals and the learned models is then taken every night; the newest ones are kept and
            unchanged files are shared between them.
          </p>
        </div>
      </Section>
    )
  }
  const p = b.progress
  const pct = p && p.files_total > 0 ? Math.min(100, Math.round((p.files_done / p.files_total) * 100)) : 0
  const hour = `${String(b.hour).padStart(2, '0')}:00`
  return (
    <Section
      title="Backups"
      aside={
        b.running ? (
          <Badge tone="warn">Running</Badge>
        ) : (
          <Button variant="secondary" onClick={() => void run()} loading={starting}>
            {!starting && <Archive className="size-4" aria-hidden />}
            Back up now
          </Button>
        )
      }
    >
      <p className={clsx('text-sm', b.last && !b.last.ok ? 'font-medium text-danger' : 'text-ink-soft')}>{lastLine(b)}</p>
      {b.running && p && (
        <div className="mt-3">
          <Meter pct={pct} label="Backup progress" />
          <p className="mt-1.5 text-xs text-ink-soft">
            {p.phase === 'originals'
              ? `${p.files_done.toLocaleString()} of ${p.files_total.toLocaleString()} originals, ${formatBytes(p.bytes_copied)} copied`
              : p.phase === 'database'
                ? 'Copying the database and the learned models'
                : 'Starting'}
          </p>
        </div>
      )}
      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-line pt-3 text-sm">
        <div>
          <dt className="text-ink-soft">Next run</dt>
          <dd className="font-semibold">
            {b.next_run_at ? `${formatDate(b.next_run_at, true)} (${formatUntil(b.next_run_at)})` : `every night at ${hour}`}
          </dd>
        </div>
        <div>
          <dt className="text-ink-soft">Snapshots kept</dt>
          <dd className="font-semibold">
            {b.snapshots.length.toLocaleString()} of {b.keep}
          </dd>
        </div>
        <div className="col-span-2">
          <dt className="text-ink-soft">Backup folder</dt>
          <dd className="break-all font-semibold">{b.dir}</dd>
          {b.total_bytes != null && b.free_bytes != null && (
            <p className="mt-1 text-xs text-ink-soft">
              {formatBytes(b.free_bytes)} free of {formatBytes(b.total_bytes)} on that disk
            </p>
          )}
        </div>
      </dl>
      <p className="mt-3 text-xs text-ink-soft">
        Every night at {hour}: the database, the originals, the share-link key, the learned models and the Wallet
        certificates. Previews and thumbnails are not copied; they are rebuilt from the originals.
      </p>
    </Section>
  )
}
