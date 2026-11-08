// Media received through messaging apps (docs/ARCHITECTURE.md section 29): the app's name for the
// viewer and the review bin, and the forward reasons (items.forward_reasons) as display labels.
import type { ReceivedVia } from '../../lib/types'

export const APP_LABELS: Record<ReceivedVia, string> = {
  whatsapp: 'WhatsApp',
  telegram: 'Telegram',
  facebook: 'Facebook',
  snapchat: 'Snapchat',
  messenger: 'Messenger',
  app: 'a messaging app',
}

export function appLabel(via: string): string {
  return APP_LABELS[via as ReceivedVia] ?? via
}

/** "Received on WhatsApp", or "Received through a messaging app" when the app is not known. */
export function receivedLabel(via: ReceivedVia | string | null | undefined): string | null {
  if (!via) return null
  return via === 'app' ? 'Received through a messaging app' : `Received on ${appLabel(via)}`
}

/** The reasons an item was set apart, in a stable order: "Meme 0.81", "Text", "Small, no camera data",
 *  "JPEG quality 75", "Filename from WhatsApp". */
export function forwardReasons(reasons: unknown): string[] {
  if (!reasons || typeof reasons !== 'object' || Array.isArray(reasons)) return []
  const r = reasons as Record<string, unknown>
  const out: string[] = []
  if (typeof r.meme_p === 'number' && Number.isFinite(r.meme_p)) out.push(`Meme ${r.meme_p.toFixed(2)}`)
  if (r.text) out.push('Text')
  const small = !!r.small
  const stripped = !!r.stripped
  if (small && stripped) out.push('Small, no camera data')
  else if (small) out.push('Small')
  else if (stripped) out.push('No camera data')
  if (r.video_small) out.push('Small video')
  if (typeof r.jpeg_quality === 'number' && Number.isFinite(r.jpeg_quality)) out.push(`JPEG quality ${Math.round(r.jpeg_quality)}`)
  if (typeof r.name === 'string' && r.name) out.push(`Filename from ${appLabel(r.name)}`)
  return out
}
