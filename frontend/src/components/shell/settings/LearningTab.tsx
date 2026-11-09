// Settings > Learning: what the small models learned from the owner's corrections (section 18 of
// docs/ARCHITECTURE.md), from GET /api/system/learning. The switch turns learning on and off; Train
// now fits the models at once instead of waiting for housekeeping; Apply queues a fresh look at
// items and documents that were decided automatically; Reset deletes the learned models (never the
// choices themselves). The Vault card also holds the vault sensitivity (low, normal, high).
// Tattoos and nails (section 34): the thresholds in use and, until every people box was read once, how far the
// reading got (asked again every 15 s while it runs), or that it is paused for a missing model.
import { useState, type ReactNode } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { EyeOff, FileText, Forward, GraduationCap, ImageOff, PenTool, RotateCcw, ScanSearch, Shapes, Tags, UsersRound, Wand2 } from 'lucide-react'
import { api } from '../../../lib/api'
import { formatCount, formatDate, formatRelative } from '../../../lib/format'
import { useRefreshLibrary } from '../../../lib/hooks'
import { CATEGORY_LABELS, type Category, type LearningEventKind, type LearningStatus, type MisclickThresholdKey, type VaultSensitivity } from '../../../lib/types'
import { docTypeLabel } from '../../documents/docMeta'
import { Badge, Button, ConfirmModal, ErrorState } from '../../ui'
import { useToast } from '../../Toast'
import { SYSTEM_KEY } from './SystemTab'
import { Section, errorText } from './common'

export const LEARNING_KEY = ['system', 'learning'] as const

// The built-in mis-click cuts: BLUR_VAR_MIN in backend/app/config.py (env, default 40) and DARK_MEAN,
// CLIPPED_FRAC, ACCIDENT_MIN in backend/app/analysis/quality.py. The status reports learned cuts only,
// so these are shown next to them as "default".
const MISCLICK_DEFAULTS: Record<MisclickThresholdKey, number> = {
  blur_var_min: 40,
  dark_mean: 28,
  clipped_frac: 0.6,
  accident_min: 0.6,
}
const MISCLICK_NAMES: Record<MisclickThresholdKey, string> = {
  blur_var_min: 'blur threshold',
  dark_mean: 'darkness threshold',
  clipped_frac: 'blown-out threshold',
  accident_min: 'pocket-shot threshold',
}
// People: the guess weights need this many confirm and reject outcomes (people.MIN_GUESS_EVENTS).
const MIN_GUESS_EVENTS = 30
// Vault: a calibrated cut needs this many labels (images.py, section 18).
const MIN_VAULT_LABELS = 8
// Forwards: the fitted cut needs this many keep or forward decisions, with both answers (section 29).
const MIN_FORWARD_EXAMPLES = 10

const SENSITIVITY_OPTIONS: { value: VaultSensitivity; label: string }[] = [
  { value: 'low', label: 'Low' },
  { value: 'normal', label: 'Normal' },
  { value: 'high', label: 'High' },
]

/** The toast line after a sensitivity change: what the immediate re-sort did. */
function movedText(moved: LearningStatus['vault_moved']): string {
  if (!moved) return ''
  const parts: string[] = []
  if (moved.moved_in) parts.push(`${moved.moved_in} moved into the vault`)
  if (moved.moved_out) parts.push(`${moved.moved_out} moved out`)
  const head = parts.length ? parts.join(', ') + '.' : 'Nothing moved.'
  const stale = moved.stale
    ? ` ${moved.stale} older item${moved.stale === 1 ? '' : 's'} need "Apply to existing items" to be checked with the new signals.`
    : ''
  return head + stale
}
const SENSITIVITY_HINTS: Record<VaultSensitivity, string> = {
  low: 'Low puts fewer photos in the vault: only clearly explicit ones.',
  normal: 'Normal puts explicit photos in the vault. Low puts in fewer, High puts in more.',
  high: 'High puts more photos in the vault, including suggestive but not explicit ones.',
}

const EVENT_LABELS: Record<LearningEventKind, string> = {
  doc_type: 'Document type changes',
  category: 'Category changes',
  vault: 'Vault moves',
  misclick: 'Mis-click decisions',
  guess: 'Guesses confirmed or rejected',
  tag: 'Suggested tags removed',
  forward: 'Forward decisions',
  species: 'Species corrections',
}

/** "Aadhaar card, PAN card, Passport and 2 more" */
export function labelList(labels: readonly string[], toLabel: (x: string) => string, max = 4): string {
  const names = labels.map(toLabel)
  if (names.length <= max) return names.join(', ')
  return `${names.slice(0, max).join(', ')} and ${names.length - max} more`
}

function fmtCut(key: MisclickThresholdKey, v: number): string {
  return key === 'blur_var_min' || key === 'dark_mean' ? String(Math.round(v)) : v.toFixed(2)
}

/** Learned cuts that differ from the built-in ones: "blur threshold now 55 (default 40)". */
export function misclickChanges(thresholds: Partial<Record<MisclickThresholdKey, number>>): string[] {
  const out: string[] = []
  for (const key of Object.keys(MISCLICK_DEFAULTS) as MisclickThresholdKey[]) {
    const v = thresholds[key]
    if (typeof v !== 'number' || !Number.isFinite(v)) continue
    const d = MISCLICK_DEFAULTS[key]
    if (Math.abs(v - d) <= Math.abs(d) * 0.005) continue
    out.push(`${MISCLICK_NAMES[key]} now ${fmtCut(key, v)} (default ${fmtCut(key, d)})`)
  }
  return out
}

function nothingLearned(s: LearningStatus): boolean {
  const l = s.learners
  return (
    (l.doc_type?.examples ?? 0) === 0 &&
    (l.category?.examples ?? 0) === 0 &&
    (l.vault?.examples ?? 0) === 0 &&
    !l.vault?.weights &&
    (l.misclick?.examples ?? 0) === 0 &&
    (l.people?.persons ?? 0) === 0 &&
    !l.people?.weights &&
    (l.tags?.tags ?? 0) === 0 &&
    (l.tags?.examples ?? 0) === 0 &&
    (l.forwards?.examples ?? 0) === 0 &&
    !l.forwards?.learned
  )
}

function Switch({
  checked,
  onChange,
  disabled,
  label,
}: {
  checked: boolean
  onChange: (next: boolean) => void
  disabled?: boolean
  label: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="inline-flex h-10 shrink-0 items-center gap-3 rounded-full pl-1 pr-1 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-60"
    >
      <span className="hidden sm:inline">{checked ? 'On' : 'Off'}</span>
      <span
        aria-hidden
        className={clsx(
          'relative inline-flex h-7 w-12 items-center rounded-full transition-colors',
          checked ? 'bg-brand' : 'bg-ink/20',
        )}
      >
        <span
          className={clsx(
            'inline-block size-5 rounded-full bg-white shadow transition-transform',
            checked ? 'translate-x-6' : 'translate-x-1',
          )}
        />
      </span>
    </button>
  )
}

function LearnerCard({
  icon,
  title,
  learned,
  children,
}: {
  icon: ReactNode
  title: string
  learned: boolean
  children: ReactNode
}) {
  return (
    <div className="flex min-w-0 items-start gap-3 rounded-2xl border border-line bg-paper p-4">
      <span className="mt-0.5 text-ink-faint" aria-hidden>
        {icon}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold">{title}</h3>
          {learned ? <Badge tone="ok">Learned</Badge> : <Badge>Not yet</Badge>}
        </div>
        <div className="mt-1 text-sm text-ink-soft">{children}</div>
      </div>
    </div>
  )
}

function SensitivityControl({
  value,
  onChange,
  disabled,
}: {
  value: VaultSensitivity | undefined
  onChange: (v: VaultSensitivity) => void
  disabled?: boolean
}) {
  return (
    <div className="mt-3">
      <div
        role="radiogroup"
        aria-label="Vault sensitivity"
        className="grid w-full grid-cols-3 rounded-full border border-line bg-surface-2 p-0.5 sm:inline-grid sm:w-auto"
      >
        {SENSITIVITY_OPTIONS.map((o) => (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={value === o.value}
            disabled={disabled}
            onClick={() => value !== o.value && onChange(o.value)}
            className={clsx(
              'h-9 min-w-0 rounded-full px-3.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60',
              value === o.value ? 'bg-ink text-paper' : 'text-ink-soft hover:bg-ink/5',
            )}
          >
            {o.label}
          </button>
        ))}
      </div>
      <p className="mt-2">{SENSITIVITY_HINTS[value ?? 'normal']}</p>
    </div>
  )
}

type SensitivityProps = { onSensitivity: (v: VaultSensitivity) => void; savingSensitivity: boolean }

/** Section 29: the keep or forward decisions, and the check of older items (GET /api/system forwards). */
function ForwardsCard({ forwards }: { forwards: LearningStatus['learners']['forwards'] }) {
  const system = useQuery({ queryKey: SYSTEM_KEY, queryFn: () => api.system(), staleTime: 15_000 })
  const n = forwards?.examples ?? 0
  const backlog = system.data?.forwards
  return (
    <LearnerCard icon={<Forward className="size-5" />} title="Forwards" learned={!!forwards?.learned}>
      <p>
        {forwards?.learned
          ? `Learned from ${formatCount(n, 'decision')}: what comes through a messaging app is set apart or kept the way you decided.`
          : n > 0
            ? `${formatCount(n, 'decision')} so far. The built-in rule decides until ${MIN_FORWARD_EXAMPLES} decisions with both answers exist.`
            : 'Nothing yet. Keep a forward, or mark one, in the forwards review to teach it.'}
      </p>
      {backlog && (backlog.unchecked > 0 || backlog.running) && (
        <p className="mt-1">
          {backlog.running ? 'Checking older items for forwards' : 'Older items waiting for a check'}
          {backlog.unchecked > 0 ? `: ${backlog.unchecked.toLocaleString()} to go.` : '.'}
        </p>
      )}
    </LearnerCard>
  )
}

function VaultCard({ s, onSensitivity, savingSensitivity }: { s: LearningStatus } & SensitivityProps) {
  const vault = s.learners.vault
  const vaultN = vault?.examples ?? 0
  return (
    <LearnerCard icon={<EyeOff className="size-5" />} title="Vault" learned={vaultN > 0 || !!vault?.weights}>
      <p>
        {vaultN > 0 ? (
          vault?.threshold != null ? (
            <>
              {formatCount(vaultN, 'example')}, threshold {vault.threshold.toFixed(2)}.
            </>
          ) : (
            <>
              {formatCount(vaultN, 'example')}. Close look-alikes follow your choice. A threshold of its own needs at least{' '}
              {MIN_VAULT_LABELS} examples, with 3 or more on each side.
            </>
          )
        ) : (
          'Nothing yet. Move a photo into or out of the vault to teach it.'
        )}
        {vault?.weights && ' Weights learned from your corrections.'}
      </p>
      <SensitivityControl value={s.vault_sensitivity} onChange={onSensitivity} disabled={savingSensitivity} />
    </LearnerCard>
  )
}

/** Reading tattoos and nails is still going: the status is worth asking again. */
function traitsScanning(s: LearningStatus | undefined): boolean {
  const scan = s?.learners.traits?.scan
  if (!scan || scan.enabled === false || scan.ready || scan.paused?.length) return false
  return !(scan.eligible > 0 && scan.scanned >= scan.eligible)
}

function TraitsCard({ traits }: { traits: NonNullable<LearningStatus['learners']['traits']> }) {
  const scan = traits.scan
  const off = scan?.enabled === false
  const reading = !!scan && !off && !scan.ready && !scan.paused?.length
  const adjusted = traits.source !== 'default'
  const outcomes = traits.outcomes
  return (
    <LearnerCard icon={<PenTool className="size-5" />} title="Tattoos and nails" learned={adjusted}>
      <p>
        {adjusted
          ? outcomes !== undefined && (traits.pairs ?? traits.hard_negatives ?? 0) > 0
            ? `Tattoos and nails: ${formatCount(outcomes, 'outcome')}, thresholds from ${formatCount(traits.pairs ?? traits.hard_negatives ?? 0, 'pair')}.`
            : outcomes !== undefined
              ? `Tattoos and nails: ${formatCount(outcomes, 'outcome')}. Thresholds adjusted to your answers.`
              : 'Thresholds adjusted to your library and your answers.'
          : reading
            ? 'Using the defaults. Guesses from tattoos alone start once every photo is read.'
            : 'Using the defaults.'}
        {adjusted && ` A tattoo match needs ${traits.tattoo_min_inliers} matching points.`}
        {!adjusted && outcomes !== undefined && outcomes > 0 && ` ${formatCount(outcomes, 'guess', 'guesses')} from tattoos alone answered so far.`}
      </p>
      {off ? (
        <p className="mt-1">Reading tattoos and nails is turned off.</p>
      ) : scan && scan.paused && scan.paused.length > 0 ? (
        <p className="mt-1" role="status">
          Paused: a model is missing ({scan.paused.join(', ')}).
        </p>
      ) : scan && reading ? (
        <>
          <p className="mt-1" role="status">
            Reading people boxes: {scan.scanned.toLocaleString()} of {scan.eligible.toLocaleString()}. Tattoos and nails count in guesses once all
            are read.
          </p>
          {scan.last_error && (
            <p className="mt-1 text-danger" role="status">
              Stopped on an error: {scan.last_error}. It tries again later.
            </p>
          )}
        </>
      ) : null}
    </LearnerCard>
  )
}

function Learners({ s, ...sensitivity }: { s: LearningStatus } & SensitivityProps) {
  const { doc_type, category, misclick, people, tags, forwards } = s.learners
  const docN = doc_type?.examples ?? 0
  const catN = category?.examples ?? 0
  const missN = misclick?.examples ?? 0
  const persons = people?.persons ?? 0
  const guessN = people?.events ?? 0
  const changes = misclickChanges(misclick?.thresholds ?? {})
  const docLabels = doc_type?.labels ?? []
  const catLabels = category?.labels ?? []

  return (
    <div className="grid gap-3 md:grid-cols-2">
      <LearnerCard icon={<FileText className="size-5" />} title="Document types" learned={docN > 0}>
        {docN > 0 ? (
          <>
            Learned from {formatCount(docN, 'correction')}, {formatCount(docLabels.length, 'type')}
            {docLabels.length > 0 && `: ${labelList(docLabels, docTypeLabel)}`}.
          </>
        ) : (
          'Nothing yet. Change the type of a document, or edit its fields, to teach it.'
        )}
      </LearnerCard>

      <LearnerCard icon={<Shapes className="size-5" />} title="Categories" learned={catN > 0}>
        {catN > 0 ? (
          <>
            {formatCount(catN, 'example')}
            {catLabels.length > 0 && ` across ${labelList(catLabels, (c) => CATEGORY_LABELS[c as Category] ?? c)}`}.
          </>
        ) : (
          'Nothing yet. Change the category of a photo to teach it.'
        )}
      </LearnerCard>

      <VaultCard s={s} {...sensitivity} />

      <LearnerCard icon={<ImageOff className="size-5" />} title="Mis-clicks" learned={changes.length > 0}>
        {changes.length > 0 ? (
          <>
            {changes.join('; ').replace(/^./, (c) => c.toUpperCase())}, from {formatCount(missN, 'decision')}.
          </>
        ) : missN > 0 ? (
          <>
            {formatCount(missN, 'decision')} so far. The built-in thresholds still fit them; a change needs at least 5
            kept and 5 confirmed photos of the same kind.
          </>
        ) : (
          'Nothing yet. Keep a flagged photo, or delete it, in the mis-click review to teach it.'
        )}
      </LearnerCard>

      <LearnerCard
        icon={<UsersRound className="size-5" />}
        title="People"
        learned={persons > 0 || !!people?.weights}
      >
        <p>
          {persons > 0
            ? `${formatCount(persons, 'person', 'people')} with their own match threshold.`
            : 'No person has their own match threshold yet. Assign faces to people to teach it.'}
        </p>
        <p className="mt-1">
          {people?.weights
            ? `Guess weights learned from ${formatCount(guessN, 'confirmation and rejection', 'confirmations and rejections')}.`
            : guessN >= MIN_GUESS_EVENTS
              ? `Guess weights: not yet. ${formatCount(guessN, 'guess', 'guesses')} decided; both confirmed and rejected guesses are needed.`
              : `Guess weights: not yet, ${guessN} of ${MIN_GUESS_EVENTS} confirmed or rejected guesses needed.`}
        </p>
      </LearnerCard>

      <LearnerCard icon={<Tags className="size-5" />} title="Tags" learned={(tags?.tags ?? 0) > 0}>
        {tags && tags.tags > 0 ? (
          <>
            {formatCount(tags.tags, 'tag')} learned from {formatCount(tags.examples, 'tagged item')},{' '}
            {formatCount(tags.rejections, 'removed suggestion')}.
          </>
        ) : tags && tags.examples > 0 ? (
          <>
            {formatCount(tags.examples, 'tagged item')} so far
            {tags.rejections > 0 && `, ${formatCount(tags.rejections, 'removed suggestion')}`}. A tag is learned once 3 photos
            and videos, or 3 documents, carry it.
          </>
        ) : (
          'Nothing yet. Add your own tags to items, and look-alikes get them suggested.'
        )}
      </LearnerCard>

      <ForwardsCard forwards={forwards} />

      {s.learners.traits && <TraitsCard traits={s.learners.traits} />}
    </div>
  )
}

export default function LearningTab() {
  const qc = useQueryClient()
  const refreshLibrary = useRefreshLibrary()
  const toast = useToast()
  const q = useQuery({
    queryKey: LEARNING_KEY,
    queryFn: () => api.system.learning(),
    refetchInterval: (query) => (traitsScanning(query.state.data) ? 15_000 : false),
  })
  const [switching, setSwitching] = useState(false)
  const [savingSensitivity, setSavingSensitivity] = useState(false)
  const [training, setTraining] = useState(false)
  const [confirmApply, setConfirmApply] = useState(false)
  const [applying, setApplying] = useState(false)
  const [confirmReset, setConfirmReset] = useState(false)
  const [resetting, setResetting] = useState(false)

  const setEnabled = async (enabled: boolean) => {
    const prev = qc.getQueryData<LearningStatus>(LEARNING_KEY)
    if (prev) qc.setQueryData<LearningStatus>(LEARNING_KEY, { ...prev, enabled })
    setSwitching(true)
    try {
      await api.system.setLearningEnabled(enabled)
      // read the server's answer directly: the cache already holds the optimistic value
      const next = await api.system.learning()
      qc.setQueryData(LEARNING_KEY, next)
      if (enabled && !next.enabled) toast.info('Learning is turned off on the server (LEARNING_ENABLED).')
      else toast.success(enabled ? 'Learning is on.' : 'Learning is off. The built-in rules decide on their own.')
    } catch (e) {
      if (prev) qc.setQueryData(LEARNING_KEY, prev)
      toast.error(errorText(e, 'Could not change the setting.'))
    } finally {
      setSwitching(false)
    }
  }

  const setSensitivity = async (vault_sensitivity: VaultSensitivity) => {
    const prev = qc.getQueryData<LearningStatus>(LEARNING_KEY)
    if (prev) qc.setQueryData<LearningStatus>(LEARNING_KEY, { ...prev, vault_sensitivity })
    setSavingSensitivity(true)
    try {
      const status = await api.system.setVaultSensitivity(vault_sensitivity)
      qc.setQueryData(LEARNING_KEY, { ...status, vault_moved: null })
      if (status.vault_moved?.moved_in || status.vault_moved?.moved_out) void refreshLibrary()
      const label = SENSITIVITY_OPTIONS.find((o) => o.value === vault_sensitivity)?.label ?? vault_sensitivity
      toast.success(`Vault sensitivity set to ${label}. ${movedText(status.vault_moved)}`)
    } catch (e) {
      if (prev) qc.setQueryData(LEARNING_KEY, prev)
      toast.error(errorText(e, 'Could not change the vault sensitivity.'))
    } finally {
      setSavingSensitivity(false)
    }
  }

  const train = async () => {
    setTraining(true)
    try {
      const s = await api.system.trainLearning()
      qc.setQueryData(LEARNING_KEY, s)
      toast.success('Training finished.')
    } catch (e) {
      toast.error(errorText(e, 'Training failed.'))
    } finally {
      setTraining(false)
    }
  }

  const apply = async () => {
    setApplying(true)
    try {
      const { items, documents, tags } = await api.system.applyLearning()
      const tagged = tags ?? 0
      if (items + documents + tagged === 0) toast.info('Nothing needed a fresh look.')
      else if (items + documents === 0) toast.success(`${formatCount(tagged, 'tag')} suggested on existing items.`)
      else
        toast.success(
          `Queued ${formatCount(items, 'item')} and ${formatCount(documents, 'document')} for a fresh look${
            tagged > 0 ? `, and suggested ${formatCount(tagged, 'tag')}` : ''
          }. Progress shows under System.`,
        )
      if (tagged > 0) {
        void qc.invalidateQueries({ queryKey: ['tags'] })
        void qc.invalidateQueries({ queryKey: ['item'] })
      }
      setConfirmApply(false)
      void qc.invalidateQueries({ queryKey: SYSTEM_KEY, exact: true })
      void qc.invalidateQueries({ queryKey: ['stats'] })
    } catch (e) {
      toast.error(errorText(e, 'Could not queue the items.'))
    } finally {
      setApplying(false)
    }
  }

  const reset = async () => {
    setResetting(true)
    try {
      await api.system.resetLearning()
      setConfirmReset(false)
      toast.success('What was learned has been deleted. Your corrections are kept.')
      void qc.invalidateQueries({ queryKey: LEARNING_KEY })
    } catch (e) {
      toast.error(errorText(e, 'Could not reset.'))
    } finally {
      setResetting(false)
    }
  }

  if (q.isPending)
    return (
      <div className="grid gap-4 sm:grid-cols-2" aria-busy="true" aria-label="Loading learning status">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="h-36 animate-pulse rounded-[var(--radius-card)] bg-ink/5" />
        ))}
      </div>
    )
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />
  const s = q.data
  const empty = nothingLearned(s)
  const pending = s.pending ?? 0
  const eventRows = (Object.keys(EVENT_LABELS) as LearningEventKind[]).map((k) => [k, s.events?.[k] ?? 0] as const)
  const eventTotal = eventRows.reduce((n, [, v]) => n + v, 0)

  return (
    <div className="flex flex-col gap-4">
      <Section
        title="Learn from my choices"
        description="Every correction you make teaches small models that run on this server, so the next automatic decision of the same kind follows your choice. Nothing is sent anywhere."
        aside={
          <Switch checked={s.enabled} onChange={(v) => void setEnabled(v)} disabled={switching} label="Learn from my choices" />
        }
      >
        <p className="text-sm text-ink-soft">
          {s.trained_at ? (
            <>
              Last trained{' '}
              <time dateTime={s.trained_at} title={formatDate(s.trained_at, true)}>
                {formatRelative(s.trained_at)}
              </time>
              .
            </>
          ) : (
            'Not trained yet.'
          )}{' '}
          {pending > 0
            ? `${formatCount(pending, 'correction')} ${s.trained_at ? 'since then' : 'waiting for the first training'}${
                s.enabled ? ', learned at the next training (every 10 minutes, or Train now).' : '.'
              }`
            : s.trained_at
              ? 'No new corrections since then.'
              : 'No corrections yet.'}
        </p>
        {!s.enabled && (
          <p className="mt-2 rounded-xl bg-surface-2 px-3 py-2 text-sm text-ink-soft">
            Learning is off. The built-in rules decide on their own, and corrections are still recorded for later.
          </p>
        )}
      </Section>

      <Section title="What has been learned">
        {empty ? (
          <div className="flex flex-col items-center gap-2 px-2 py-8 text-center">
            <GraduationCap className="size-8 text-ink-faint" aria-hidden />
            <h3 className="text-base font-semibold">Nothing learned yet</h3>
            <p className="max-w-md text-sm text-ink-soft">
              Correct a category, a document type, a mis-click, a forward or a person, or add your own tags, and the next training learns from it.
            </p>
          </div>
        ) : (
          <Learners s={s} onSensitivity={(v) => void setSensitivity(v)} savingSensitivity={savingSensitivity} />
        )}
        {empty && (
          <div className="mt-2 md:w-1/2 md:pr-1.5">
            <VaultCard s={s} onSensitivity={(v) => void setSensitivity(v)} savingSensitivity={savingSensitivity} />
          </div>
        )}
      </Section>

      <Section
        title="Corrections recorded"
        description="Corrections and guess outcomes recorded since learning was added. Choices saved earlier on items, documents and people are used for training as well."
      >
        {eventTotal === 0 ? (
          <p className="text-sm text-ink-soft">None yet.</p>
        ) : (
          <ul className="grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
            {eventRows.map(([k, v]) => (
              <li key={k} className="flex items-center justify-between gap-3 py-1">
                <span className="text-ink-soft">{EVENT_LABELS[k]}</span>
                <span className="tabular-nums font-medium">{v.toLocaleString()}</span>
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section
        title="Use it now"
        description="Training also runs on its own every 10 minutes when there are new corrections. Learned models only shape new decisions; apply them to reach items already in the library."
      >
        <div className="flex flex-wrap gap-2">
          <Button onClick={() => void train()} loading={training} disabled={!s.enabled}>
            {!training && <ScanSearch className="size-4" aria-hidden />}
            Train now
          </Button>
          <Button variant="secondary" onClick={() => setConfirmApply(true)} disabled={!s.enabled || empty}>
            <Wand2 className="size-4" aria-hidden />
            Apply to existing items
          </Button>
          <Button variant="danger" onClick={() => setConfirmReset(true)}>
            <RotateCcw className="size-4" aria-hidden />
            Reset
          </Button>
        </div>
      </Section>

      <Section
        title="How to teach it"
        description={
          <>
            Make corrections where you always have: change a photo&apos;s category or move it to the vault on the item, keep or
            delete flagged photos in the mis-click review, keep or mark forwards in the forwards review, fix a document&apos;s type or fields on the document, and assign faces,
            confirm or reject guesses on the people pages. Add your own tags on an item, and accept or remove the tags it
            suggests. Your own choices are never changed by what is learned.
          </>
        }
      />

      <ConfirmModal
        open={confirmApply}
        title="Apply to existing items"
        body="Items whose category, vault or mis-click decision was made automatically are analysed again, documents typed automatically are read again, and learned tags are suggested on look-alikes, so they follow what was learned. Choices you made yourself stay as they are. On a large library this takes a while."
        confirmLabel="Apply"
        loading={applying}
        onConfirm={() => void apply()}
        onClose={() => setConfirmApply(false)}
      />
      <ConfirmModal
        open={confirmReset}
        title="Reset what was learned"
        body="The learned models are deleted and automatic decisions go back to the built-in rules. Your corrections are kept, so the next training learns them again. To stop learning, turn the switch off."
        confirmLabel="Reset"
        danger
        loading={resetting}
        onConfirm={() => void reset()}
        onClose={() => setConfirmReset(false)}
      />
    </div>
  )
}
