// Share links (docs/ARCHITECTURE.md section 26): the owner's /api/shares calls and the public
// /api/s/{token} calls, plus the small helpers the share UI shares. The public calls never answer
// 401, so lib/api's wrapper is safe to use from the page without a session.
import { ApiError, del, get, patch, post } from './api'
import type { OkOut } from './types'

export type ShareKind = 'album' | 'items'

export interface ShareAlbumRef {
  id: number
  name: string
}

export interface ShareOut {
  id: number
  token: string
  /** "/s/<token>"; the full link is location.origin + path (shareUrl). */
  path: string
  kind: ShareKind
  album: ShareAlbumRef | null
  title: string
  /** Live, non-vault members. */
  count: number
  has_password: boolean
  expires_at: string | null
  allow_download: boolean
  created_at: string
  revoked_at: string | null
  view_count: number
  last_viewed_at: string | null
}

export interface ShareCreate {
  kind: ShareKind
  album_id?: number
  item_uids?: string[]
  title?: string
  password?: string | null
  expires_days?: number | null
  allow_download?: boolean
}

/** Fields left out stay as they are; an empty password clears it; expires_days null means never. */
export interface ShareUpdate {
  title?: string
  password?: string
  expires_days?: number | null
  allow_download?: boolean
}

export interface ShareItemUrls {
  thumb: string
  preview: string
  video: string | null
  download: string | null
}

export interface ShareItem {
  uid: string
  kind: string
  width: number | null
  height: number | null
  duration_s: number | null
  taken_at: string | null
  filename: string
  urls: ShareItemUrls
}

export interface SharePublic {
  title: string
  kind: ShareKind
  count: number
  allow_download: boolean
  expires_at: string | null
  items: ShareItem[]
}

export const SHARES_KEY = ['shares'] as const
export const SHARE_PUBLIC_KEY = 'share-public'

export const shares = {
  list: () => get<ShareOut[]>('/api/shares'),
  create: (body: ShareCreate) => post<ShareOut>('/api/shares', body),
  update: (id: number, body: ShareUpdate) => patch<ShareOut>(`/api/shares/${id}`, body),
  revoke: (id: number) => del<OkOut>(`/api/shares/${id}`),
}

export const sharePublic = (token: string) => get<SharePublic>(`/api/s/${encodeURIComponent(token)}`)
export const shareUnlock = (token: string, password: string) =>
  post<SharePublic>(`/api/s/${encodeURIComponent(token)}/unlock`, { password })

export function shareUrl(path: string): string {
  return `${window.location.origin}${path}`
}

export type ShareStatus = 'active' | 'expired' | 'revoked'

export function shareStatus(s: Pick<ShareOut, 'revoked_at' | 'expires_at'>, now = Date.now()): ShareStatus {
  if (s.revoked_at) return 'revoked'
  if (s.expires_at && Date.parse(s.expires_at) <= now) return 'expired'
  return 'active'
}

/** The expiry choices of the share dialogs: '' is never. */
export const EXPIRY_CHOICES: { value: string; label: string }[] = [
  { value: '', label: 'Never' },
  { value: '1', label: 'After 1 day' },
  { value: '7', label: 'After 7 days' },
  { value: '30', label: 'After 30 days' },
]

export function expiryText(expiresAt: string | null, now = Date.now()): string {
  if (!expiresAt) return 'Never expires'
  const ms = Date.parse(expiresAt) - now
  if (Number.isNaN(ms)) return ''
  if (ms <= 0) return 'Expired'
  const hours = Math.ceil(ms / 3600000)
  if (hours < 48) return `Expires in ${hours} ${hours === 1 ? 'hour' : 'hours'}`
  const days = Math.ceil(ms / 86400000)
  return `Expires in ${days} days`
}

/** Copy text to the clipboard; falls back to a selection copy where the Clipboard API is missing. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // fall through to the selection copy
  }
  try {
    const area = document.createElement('textarea')
    area.value = text
    area.setAttribute('readonly', '')
    area.style.position = 'fixed'
    area.style.opacity = '0'
    document.body.appendChild(area)
    area.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(area)
    return ok
  } catch {
    return false
  }
}

/** Plain words for the share API's error codes. */
export function shareErrorText(e: unknown, fallback = 'Something went wrong. Try again.'): string {
  if (e instanceof ApiError) {
    switch (e.code) {
      case 'vault_item':
        return 'Private items cannot be shared. Take them out of the selection first.'
      case 'no_items':
        return 'Nothing to share: the selection has no photos or videos.'
      case 'album_not_found':
        return 'That album no longer exists.'
      case 'item_not_found':
        return 'One of the items no longer exists.'
      case 'share_not_found':
        return 'This link does not exist or was taken down.'
      case 'share_expired':
        return 'This link has expired.'
      case 'share_password_required':
        return 'This link needs a password.'
      case 'invalid_password':
        return 'That password is not right.'
      case 'too_many_attempts':
        return 'Too many tries. Wait a few minutes, then try again.'
      default:
        if (e.status === 0) return 'The server could not be reached.'
    }
    return e.message || fallback
  }
  return e instanceof Error && e.message ? e.message : fallback
}
