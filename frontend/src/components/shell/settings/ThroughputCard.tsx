// Settings > System: processing speed (docs/ARCHITECTURE.md section 32). One small column chart per job type
// from GET /api/system/throughput: finished runs per minute (last hour), per hour (last 24 hours) or per six
// hours (last 7 days), failures stacked on top in the failure colour. Each chart has hover and keyboard
// tooltips (arrow keys move along the buckets); "Show as table" lists every number.
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { Gauge } from 'lucide-react'
import { api } from '../../../lib/api'
import type { ThroughputOut, ThroughputRange, ThroughputType } from '../../../lib/types'
import { ErrorState } from '../../ui'
import { Section } from './common'

const POLL_MS = 30_000
const RANGES: { key: ThroughputRange; label: string; per: string; window: string }[] = [
  { key: 'hour', label: 'Last hour', per: 'per minute', window: 'in the last hour' },
  { key: 'day', label: '24 hours', per: 'per hour', window: 'in the last 24 hours' },
  { key: 'week', label: '7 days', per: 'per 6 hours', window: 'in the last 7 days' },
]
const PLOT_H = 88
const AXIS_W = 30
const X_AXIS_H = 18
const TOP_PAD = 8 // room for the top tick label, centred on its gridline
const BAR_MAX = 24
const GAP = 2
const RADIUS = 4

function niceMax(v: number): number {
  if (v <= 4) return Math.max(1, v)
  const pow = 10 ** Math.floor(Math.log10(v))
  for (const m of [1, 2, 2.5, 5, 10]) if (m * pow >= v) return m * pow
  return 10 * pow
}

function formatTick(n: number): string {
  return n >= 10_000 ? `${Math.round(n / 1000)}K` : n.toLocaleString()
}

function formatMs(ms: number | null): string {
  if (ms == null) return ''
  if (ms < 1000) return `${ms} ms`
  const s = ms / 1000
  return s < 60 ? `${s < 10 ? s.toFixed(1) : Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`
}

function bucketLabel(iso: string, bucketS: number, withDay: boolean): string {
  const d = new Date(iso)
  const end = new Date(d.getTime() + bucketS * 1000)
  const t = (x: Date) => x.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const day = (x: Date) => x.toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short' })
  return withDay ? `${day(d)}, ${t(d)} to ${t(end)}` : `${t(d)} to ${t(end)}`
}

function axisLabel(iso: string, range: ThroughputRange): string {
  const d = new Date(iso)
  if (range === 'week') return d.toLocaleDateString([], { weekday: 'short', day: 'numeric' })
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/** A column with a rounded data end (top) and a square base. */
function columnPath(x: number, y: number, w: number, h: number, round: boolean): string {
  if (h <= 0) return ''
  const r = round ? Math.min(RADIUS, w / 2, h) : 0
  return `M${x},${y + h}V${y + r}${r ? `Q${x},${y} ${x + r},${y}` : ''}H${x + w - r}${r ? `Q${x + w},${y} ${x + w},${y + r}` : ''}V${y + h}Z`
}

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T | null>(null)
  const [w, setW] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver((entries) => setW(Math.floor(entries[0].contentRect.width)))
    ro.observe(el)
    setW(Math.floor(el.getBoundingClientRect().width))
    return () => ro.disconnect()
  }, [])
  return [ref, w]
}

function TypeChart({ t, data, rangeLabel }: { t: ThroughputType; data: ThroughputOut; rangeLabel: string }) {
  const [wrap, width] = useWidth<HTMLDivElement>()
  const [active, setActive] = useState<number | null>(null)
  const n = data.buckets.length
  const peak = Math.max(0, ...t.counts)
  const top = niceMax(peak)
  const plotW = Math.max(0, width - AXIS_W)
  const band = n > 0 ? plotW / n : 0
  const barW = Math.max(1, Math.min(BAR_MAX, band - GAP))
  const y = (v: number) => PLOT_H - (v / top) * PLOT_H
  const withDay = data.range === 'week'
  const ticks = [0, Math.round(top / 2) === top / 2 ? top / 2 : null, top].filter((v): v is number => v !== null)
  const xLabels = n > 0 ? [0, Math.floor((n - 1) / 2), n - 1] : []

  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left - AXIS_W
    if (x < 0 || band <= 0) return setActive(null)
    setActive(Math.min(n - 1, Math.max(0, Math.floor(x / band))))
  }
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    if (n === 0) return
    const cur = active ?? n - 1
    let next = cur
    if (e.key === 'ArrowLeft') next = Math.max(0, cur - 1)
    else if (e.key === 'ArrowRight') next = Math.min(n - 1, cur + 1)
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = n - 1
    else if (e.key === 'Escape') return setActive(null)
    else return
    e.preventDefault()
    setActive(next)
  }

  const tip = active !== null ? { i: active, done: t.counts[active] - t.failed_counts[active], failed: t.failed_counts[active] } : null
  const tipLeft = tip ? Math.min(Math.max(AXIS_W + (tip.i + 0.5) * band, 70), Math.max(70, width - 70)) : 0

  return (
    <div ref={wrap} className="relative">
      {width > 0 && (
        <svg
          width={width}
          height={PLOT_H + X_AXIS_H + TOP_PAD}
          role="img"
          tabIndex={0}
          aria-label={`${t.label}: ${t.total.toLocaleString()} ${rangeLabel}, ${t.failed.toLocaleString()} failed. Use the arrow keys to read each ${data.range === 'hour' ? 'minute' : data.range === 'day' ? 'hour' : 'six hours'}.`}
          className="block touch-pan-y rounded-md outline-none focus-visible:ring-2 focus-visible:ring-accent"
          onPointerMove={pick}
          onPointerDown={pick}
          onPointerLeave={() => setActive(null)}
          onFocus={() => setActive((a) => a ?? n - 1)}
          onBlur={() => setActive(null)}
          onKeyDown={onKey}
        >
          <g transform={`translate(0, ${TOP_PAD})`}>
            {ticks.map((v) => (
              <g key={v}>
                <line x1={AXIS_W} x2={width} y1={y(v) + 0.5} y2={y(v) + 0.5} stroke="var(--color-line)" strokeWidth={1} />
                <text x={AXIS_W - 6} y={y(v)} dy="0.32em" textAnchor="end" className="fill-ink-faint text-[10px] tabular-nums">
                  {formatTick(v)}
                </text>
              </g>
            ))}
            {active !== null && (
              <rect x={AXIS_W + active * band} y={0} width={band} height={PLOT_H} fill="var(--color-surface-2)" />
            )}
            {t.counts.map((c, i) => {
              if (c <= 0) return null
              const failed = t.failed_counts[i]
              const ok = c - failed
              const x = AXIS_W + i * band + (band - barW) / 2
              const okH = (ok / top) * PLOT_H
              const failH = (failed / top) * PLOT_H
              const gap = ok > 0 && failed > 0 ? Math.min(GAP, failH / 2) : 0
              return (
                <g key={i} opacity={active === null || active === i ? 1 : 0.55}>
                  {ok > 0 && (
                    <path d={columnPath(x, PLOT_H - okH, barW, okH, failed === 0)} fill="var(--color-chart-1)" />
                  )}
                  {failed > 0 && (
                    <path
                      d={columnPath(x, PLOT_H - okH - failH, barW, Math.max(1, failH - gap), true)}
                      fill="var(--color-chart-fail)"
                    />
                  )}
                </g>
              )
            })}
            <line x1={AXIS_W} x2={width} y1={PLOT_H + 0.5} y2={PLOT_H + 0.5} stroke="var(--color-line)" strokeWidth={1} />
            {xLabels.map((i, k) => (
              <text
                key={`${i}-${k}`}
                x={AXIS_W + (i + 0.5) * band}
                y={PLOT_H + 14}
                textAnchor={k === 0 ? 'start' : k === xLabels.length - 1 ? 'end' : 'middle'}
                className="fill-ink-faint text-[10px] tabular-nums"
              >
                {k === xLabels.length - 1 ? 'now' : axisLabel(data.buckets[i], data.range)}
              </text>
            ))}
          </g>
        </svg>
      )}
      {tip && (
        <div
          role="status"
          className="pointer-events-none absolute -top-2 z-10 -translate-x-1/2 -translate-y-full rounded-lg bg-ink px-2.5 py-1.5 text-xs whitespace-nowrap text-surface shadow-lg"
          style={{ left: tipLeft }}
        >
          <p className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-3 rounded-full bg-chart-1" aria-hidden />
            <strong className="font-semibold tabular-nums">{tip.done.toLocaleString()}</strong> finished
          </p>
          {tip.failed > 0 && (
            <p className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-3 rounded-full bg-chart-fail" aria-hidden />
              <strong className="font-semibold tabular-nums">{tip.failed.toLocaleString()}</strong> failed
            </p>
          )}
          <p className="mt-0.5 opacity-75">{bucketLabel(data.buckets[tip.i], data.bucket_s, withDay)}</p>
        </div>
      )}
    </div>
  )
}

function TypeBlock({ t, data, rangeLabel, rate }: { t: ThroughputType; data: ThroughputOut; rangeLabel: string; rate: number }) {
  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 className="text-sm font-semibold">{t.label}</h3>
        <p className="text-xs text-ink-soft tabular-nums">
          {rate > 0 ? `${rate.toLocaleString(undefined, { maximumFractionDigits: 1 })} a minute now` : 'idle now'}
        </p>
      </div>
      <p className="mb-2 text-xs text-ink-soft tabular-nums">
        {t.total.toLocaleString()} {rangeLabel}
        {t.avg_ms != null ? `, ${formatMs(t.avg_ms)} each on average` : ''}
        {t.failed > 0 ? `, ${t.failed.toLocaleString()} failed` : ''}
      </p>
      <TypeChart t={t} data={data} rangeLabel={rangeLabel} />
    </div>
  )
}

function Table({ data }: { data: ThroughputOut }) {
  const withDay = data.range === 'week'
  return (
    <div className="max-h-80 overflow-auto rounded-lg border border-line">
      <table className="w-full text-left text-xs tabular-nums">
        <thead className="sticky top-0 bg-surface-2 text-ink-soft">
          <tr>
            <th scope="col" className="px-2 py-1.5 font-medium">Time</th>
            {data.types.map((t) => (
              <th key={t.type} scope="col" className="px-2 py-1.5 text-right font-medium">
                {t.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.buckets
            .map((b, i) => ({ b, i }))
            .reverse()
            .map(({ b, i }) => (
              <tr key={b} className="border-t border-line">
                <th scope="row" className="px-2 py-1 font-normal whitespace-nowrap text-ink-soft">
                  {bucketLabel(b, data.bucket_s, withDay)}
                </th>
                {data.types.map((t) => (
                  <td key={t.type} className="px-2 py-1 text-right">
                    {t.counts[i] ? t.counts[i].toLocaleString() : ''}
                    {t.failed_counts[i] ? ` (${t.failed_counts[i]} failed)` : ''}
                  </td>
                ))}
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  )
}

export default function ThroughputCard() {
  const [range, setRange] = useState<ThroughputRange>('hour')
  const [table, setTable] = useState(false)
  const q = useQuery({
    queryKey: ['system', 'throughput', range],
    queryFn: () => api.system.throughput(range),
    refetchInterval: POLL_MS,
    refetchIntervalInBackground: false,
    placeholderData: keepPreviousData,
  })
  const meta = RANGES.find((r) => r.key === range) ?? RANGES[0]
  const data = q.data
  const busy = useMemo(() => (data ? data.types.filter((t) => t.total > 0 || (data.per_minute[t.type] ?? 0) > 0) : []), [data])
  const quiet = data ? data.types.filter((t) => !busy.includes(t)) : []

  return (
    <Section
      title="Processing speed"
      description={`Jobs the server finished, ${meta.per}. Kept for 14 days.`}
      aside={
        <div role="group" aria-label="Period" className="inline-flex rounded-full bg-surface-2 p-1">
          {RANGES.map((r) => (
            <button
              key={r.key}
              type="button"
              aria-pressed={range === r.key}
              onClick={() => setRange(r.key)}
              className={clsx(
                'h-8 rounded-full px-3 text-sm font-medium transition-colors',
                range === r.key ? 'bg-surface text-ink shadow-sm' : 'text-ink-soft hover:text-ink',
              )}
            >
              {r.label}
            </button>
          ))}
        </div>
      }
    >
      {q.isError && !data ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : !data ? (
        <div className="h-40 animate-pulse rounded-xl bg-ink/5" aria-busy="true" aria-label="Loading processing speed" />
      ) : (
        <div className={clsx('transition-opacity', q.isPlaceholderData && 'opacity-60')}>
          <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-soft">
            <span className="flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-sm bg-chart-1" aria-hidden /> Finished
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-sm bg-chart-fail" aria-hidden /> Failed
            </span>
          </div>
          {busy.length === 0 ? (
            <p className="flex items-center gap-2 py-6 text-sm text-ink-soft">
              <Gauge className="size-4 text-ink-faint" aria-hidden />
              Nothing was processed {meta.window}.
            </p>
          ) : (
            <div className="grid gap-x-8 gap-y-6 md:grid-cols-2">
              {busy.map((t) => (
                <TypeBlock key={t.type} t={t} data={data} rangeLabel={meta.window} rate={data.per_minute[t.type] ?? 0} />
              ))}
            </div>
          )}
          {quiet.length > 0 && busy.length > 0 && (
            <p className="mt-4 text-xs text-ink-soft">
              No runs {meta.window}: {quiet.map((t) => t.label.toLowerCase()).join(', ')}.
            </p>
          )}
          <div className="mt-4 border-t border-line pt-3">
            <button
              type="button"
              className="text-sm font-medium text-accent hover:underline"
              aria-expanded={table}
              onClick={() => setTable((v) => !v)}
            >
              {table ? 'Hide the table' : 'Show as table'}
            </button>
            {table && (
              <div className="mt-3">
                <Table data={data} />
              </div>
            )}
          </div>
        </div>
      )}
    </Section>
  )
}
