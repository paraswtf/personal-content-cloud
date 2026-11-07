// Mis-click reasons (items.misclick_reasons: {reason: detail}) as display labels.
export const REASON_LABELS: Record<string, string> = {
  blurry: 'Blurry',
  dark: 'Too dark',
  blown_out: 'Blown out',
  pocket: 'Pocket shot',
  burst: 'Burst copy',
}

/** Reason keys of an item, in a stable order. Accepts {reason: detail}, {reasons: [...]} or a list. */
export function reasonKeys(reasons: unknown): string[] {
  if (!reasons) return []
  if (Array.isArray(reasons)) return reasons.map(String)
  if (typeof reasons !== 'object') return []
  const r = reasons as Record<string, unknown>
  if (Array.isArray(r.reasons)) return r.reasons.map(String)
  const order = Object.keys(REASON_LABELS)
  return Object.keys(r)
    .filter((k) => k !== 'burst_best' && k !== 'best_uid')
    .sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99))
}

export function reasonLabel(key: string): string {
  return REASON_LABELS[key] ?? key.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())
}

/** uid of the sharpest shot of a burst, when the reasons name it. */
export function burstBestUid(reasons: unknown): string | null {
  if (!reasons || typeof reasons !== 'object' || Array.isArray(reasons)) return null
  const r = reasons as Record<string, unknown>
  const b = r.burst
  if (b && typeof b === 'object' && typeof (b as { best_uid?: unknown }).best_uid === 'string') return (b as { best_uid: string }).best_uid
  return typeof r.best_uid === 'string' ? r.best_uid : null
}
