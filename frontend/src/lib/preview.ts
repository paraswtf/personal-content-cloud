// Sizes of the compressed previews the server makes and helpers around file URLs.
// Keep PREVIEW_SIZE and PREVIEW_SMALL_SIZE equal to the backend settings (PREVIEW_SIZE, PREVIEW_SMALL_SIZE);
// posterTime mirrors app.media.video.poster_time. Originals are never shown inline, only downloaded.
export const PREVIEW_SIZE = 2048
export const PREVIEW_SMALL_SIZE = 1280

/** Pixel width of a preview copy whose longer side is capped at `size` (never upscaled). */
export function previewWidth(width: number | null | undefined, height: number | null | undefined, size: number): number {
  const w = width ?? 0
  const h = height ?? 0
  const longest = Math.max(w, h)
  if (!(w > 0) || !(longest > 0)) return 0
  return Math.round(w * Math.min(1, size / longest))
}

/** srcSet with the small and large copies, or undefined when the dimensions are unknown. */
export function previewSrcSet(
  urls: { preview: string; preview_small: string },
  width: number | null | undefined,
  height: number | null | undefined,
): string | undefined {
  const small = previewWidth(width, height, PREVIEW_SMALL_SIZE)
  const large = previewWidth(width, height, PREVIEW_SIZE)
  if (!small || !large) return undefined
  // A small picture has one copy at its own size: both candidates would be identical.
  if (small >= large) return `${urls.preview} ${large}w`
  return `${urls.preview_small} ${small}w, ${urls.preview} ${large}w`
}

/** Time of the poster frame as the server picks it: 10 percent in, at most 3 s, 0 for clips under a second. */
export function posterTime(duration: number | null | undefined): number {
  if (!duration || duration < 1) return 0
  return Math.round(Math.min(3, duration * 0.1) * 1000) / 1000
}

/** `url` with `vault=1` added when `vault` is set and the URL points at /files (vault items need it). */
export function withVault(url: string, vault: boolean): string
export function withVault(url: string | null, vault: boolean): string | null
export function withVault(url: string | null, vault: boolean): string | null {
  if (!url || !vault || !url.startsWith('/files/') || /[?&]vault=1(&|$)/.test(url)) return url
  return `${url}${url.includes('?') ? '&' : '?'}vault=1`
}

/**
 * Deep copy of `value` with every /files URL string carrying `vault=1`. Used for vault items, whose
 * thumbnails, previews, pages, crops and downloads are refused without it. Returns `value` itself
 * when `vault` is false.
 */
export function vaultify<T>(value: T, vault: boolean): T {
  if (!vault) return value
  const walk = (v: unknown): unknown => {
    if (typeof v === 'string') return withVault(v, true)
    if (Array.isArray(v)) return v.map(walk)
    if (v && typeof v === 'object') {
      const out: Record<string, unknown> = {}
      for (const [k, x] of Object.entries(v as Record<string, unknown>)) out[k] = walk(x)
      return out
    }
    return v
  }
  return walk(value) as T
}
