// A search snippet rendered as plain text. The server HTML-escapes the text and wraps matched terms
// in <b></b> (search/service.py); <mark> and the raw \x02/\x03 markers are accepted too. The
// markers become <mark> elements, the basic entities are decoded, and nothing else is interpreted
// (OCR text is untrusted, so no innerHTML).
import { Fragment } from 'react'
import clsx from 'clsx'

const MARK = /<mark>|<\/mark>|<b>|<\/b>|\u0002|\u0003/g
const OPEN = new Set(['<mark>', '<b>', '\u0002'])

const ENTITIES: Record<string, string> = { '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#x27;': "'", '&#39;': "'" }

function decode(text: string): string {
  return text.replace(/&(amp|lt|gt|quot|#x27|#39);/g, (m) => ENTITIES[m] ?? m)
}

export function snippetParts(text: string): { text: string; hit: boolean }[] {
  const out: { text: string; hit: boolean }[] = []
  let hit = false
  let last = 0
  for (const m of text.matchAll(MARK)) {
    const i = m.index ?? 0
    if (i > last) out.push({ text: decode(text.slice(last, i)), hit })
    hit = OPEN.has(m[0])
    last = i + m[0].length
  }
  if (last < text.length) out.push({ text: decode(text.slice(last)), hit })
  return out
}

export default function Snippet({ text, className }: { text: string; className?: string }) {
  return (
    <span className={className}>
      {snippetParts(text).map((p, i) =>
        p.hit ? (
          <mark key={i} className={clsx('rounded-sm bg-accent-soft px-0.5 text-inherit')}>
            {p.text}
          </mark>
        ) : (
          <Fragment key={i}>{p.text}</Fragment>
        ),
      )}
    </span>
  )
}
