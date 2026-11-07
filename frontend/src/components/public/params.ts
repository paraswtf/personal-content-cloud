// URL search-param helpers for the gallery pages.

/** "1,2,x,2" -> [1, 2] (positive integers, unique, in order). */
export function parseIds(raw: string | null): number[] {
  if (!raw) return []
  const out: number[] = []
  for (const part of raw.split(',')) {
    const n = Number(part)
    if (Number.isInteger(n) && n > 0 && !out.includes(n)) out.push(n)
  }
  return out
}

/** Copy of `params` with `key` set (or removed when value is empty) and the viewer params (m, t) dropped. */
export function withParam(params: URLSearchParams, key: string, value: string | null | undefined): URLSearchParams {
  const p = new URLSearchParams(params)
  if (value) p.set(key, value)
  else p.delete(key)
  p.delete('m')
  p.delete('t')
  return p
}
