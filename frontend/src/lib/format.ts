// Display formatting helpers. Dates are shown in the browser's local time.
import type { PersonKind, Species } from './types'

export function formatBytes(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '-'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`
}

/** 75 -> "1:15", 3725 -> "1:02:05" */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return ''
  const s = Math.max(0, Math.round(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = String(s % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`
}

export function formatDate(iso: string | null | undefined, withTime = false): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    ...(withTime ? { hour: 'numeric', minute: '2-digit' } : {}),
  })
}

/** The tail of a merge toast: the look-alikes merged along with the owner's merge (section 10). */
export function followedTail(followed: number | undefined): string {
  return followed ? ` ${formatCount(followed, 'look-alike')} came along.` : ''
}

export function formatCount(n: number, singular: string, plural = `${singular}s`): string {
  return `${n.toLocaleString()} ${n === 1 ? singular : plural}`
}

/** A person's items: "4 items". Accepts the hub shape ({media_count, video_count}) too. */
export function formatMediaCount(p: { item_count?: number; media_count?: number; video_count?: number }): string {
  if (p.item_count !== undefined) return formatCount(p.item_count, 'item')
  const videos = p.video_count ?? 0
  const photos = Math.max(0, (p.media_count ?? 0) - videos)
  if (!videos) return formatCount(photos, 'photo')
  if (!photos) return formatCount(videos, 'video')
  return `${formatCount(photos, 'photo')}, ${formatCount(videos, 'video')}`
}

export function formatResolution(w: number | null | undefined, h: number | null | undefined): string {
  if (!w || !h) return ''
  const mp = (w * h) / 1e6
  return `${w} x ${h}${mp >= 1 ? ` (${mp.toFixed(1)} MP)` : ''}`
}

export function initials(name: string | null | undefined): string {
  if (!name) return '?'
  const parts = name.trim().split(/\s+/).filter(Boolean)
  return ((parts[0]?.[0] ?? '') + (parts.length > 1 ? parts[parts.length - 1][0] : '')).toUpperCase() || '?'
}

/** "18.5204° N, 73.8567° E" from decimal degrees. */
export function formatCoords(lat: number, lon: number): string {
  const part = (v: number, pos: string, neg: string) => `${Math.abs(v).toFixed(4)}° ${v >= 0 ? pos : neg}`
  return `${part(lat, 'N', 'S')}, ${part(lon, 'E', 'W')}`
}

/** "just now", "5 min ago", "3 h ago", "2 days ago", then a date. */
export function formatRelative(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return ''
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return ''
  const s = Math.max(0, Math.round((now - t) / 1000))
  if (s < 45) return 'just now'
  const m = Math.round(s / 60)
  if (m < 60) return `${m} min ago`
  const h = Math.round(m / 60)
  if (h < 24) return `${h} h ago`
  const d = Math.round(h / 24)
  if (d < 7) return `${d} ${d === 1 ? 'day' : 'days'} ago`
  return formatDate(iso)
}

/** "in 8 h", "in 25 min", "in 2 days" for a time ahead; "now" once it has come. */
export function formatUntil(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return ''
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return ''
  const s = Math.round((t - now) / 1000)
  if (s < 45) return 'now'
  const m = Math.round(s / 60)
  if (m < 60) return `in ${m} min`
  const h = Math.round(m / 60)
  if (h < 24) return `in ${h} h`
  const d = Math.round(h / 24)
  return `in ${d} ${d === 1 ? 'day' : 'days'}`
}

/** Heading for a day group: "Today", "Yesterday", or "Mon, 3 Nov 2026". */
export function formatDayHeading(iso: string, now = new Date()): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const day = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime()
  const diff = Math.round((day(now) - day(d)) / 86_400_000)
  if (diff === 0) return 'Today'
  if (diff === 1) return 'Yesterday'
  return d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })
}

/** Local calendar day of a timestamp, as a stable key for grouping. */
export function dayKey(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso.slice(0, 10) : `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`
}

/** Exact time for a tooltip. */
export function formatExact(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'medium' })
}

/** A person's display name: their name, or "Unnamed" when they have none yet. */
export function personName(p: { name: string | null; kind?: PersonKind; species?: Species | null } | null | undefined): string {
  if (p?.name?.trim()) return p.name
  if (p?.kind === 'pet') return p.species ? `Unnamed ${speciesLabel(p.species)}` : 'Unnamed pet'
  return 'Unnamed'
}

/** "dog" -> "Dog". */
export function speciesLabel(species: string | null | undefined): string {
  if (!species) return 'Pet'
  const words = species.replace(/_/g, ' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

/** "a cat", "an alpaca": a species in a sentence. */
export function aSpecies(species: string | null | undefined): string {
  const word = speciesLabel(species).toLowerCase()
  return `${/^[aeiou]/.test(word) ? 'an' : 'a'} ${word}`
}

/** 0.861 -> "86%". */
export function formatPercent(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return ''
  return `${(v * 100).toFixed(digits)}%`
}

/** 0.861 -> "0.86". */
export function formatScore(v: unknown): string {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(2) : ''
}

/** "2026-11-07" (a plain date, no time zone shift) -> "7 Nov 2026". Other strings are returned as they are. */
export function formatPlainDate(value: string | null | undefined): string {
  if (!value) return ''
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (!m) return formatDate(value) || value
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]))
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

/** Local calendar date of a timestamp as "YYYY-MM-DD" (for date inputs). */
export function localDateInput(d: Date = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

/** "3 min", "1 h 5 min": a span of minutes for countdowns. */
export function formatMinutes(min: number): string {
  const m = Math.max(0, Math.ceil(min))
  if (m < 60) return `${m} min`
  const h = Math.floor(m / 60)
  return m % 60 ? `${h} h ${m % 60} min` : `${h} h`
}
