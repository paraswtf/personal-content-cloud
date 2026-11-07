// The text read from an item (PDF text layer, OCR, office text), collapsed to a few lines with
// "Show all" and a copy button.
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Check, Copy, Loader2 } from 'lucide-react'
import { get, qs } from '../../lib/api'
import type { ItemTextOut } from '../../lib/types'
import { formatCount } from '../../lib/format'

const SOURCE_LABELS: Record<string, string> = {
  pdf: 'from the PDF text',
  ocr: 'read from the picture',
  office: 'from the document',
  plain: 'plain text',
  mixed: 'from the text and the picture',
}

export default function TextPanel({ uid, vault }: { uid: string; vault: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const q = useQuery({
    queryKey: ['item-text', uid, vault],
    // api.items.text has no vault flag; vault items need ?vault=1 like every other read.
    queryFn: () => get<ItemTextOut>(`/api/items/${encodeURIComponent(uid)}/text${qs({ vault: vault || undefined })}`),
  })

  if (q.isPending)
    return (
      <p className="flex items-center gap-2 text-sm text-white/60" role="status">
        <Loader2 className="size-4 animate-spin" aria-hidden /> Loading text
      </p>
    )
  if (q.isError) return <p className="text-sm text-white/60">The text could not be loaded.</p>
  const text = q.data.text.trim()
  if (!text) return <p className="text-sm text-white/60">No text was found.</p>

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }
  const long = text.length > 600 || text.split('\n').length > 12

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-2">
        <p className="text-xs text-white/50">
          {formatCount(q.data.char_count, 'character')}, {SOURCE_LABELS[q.data.source] ?? q.data.source}
        </p>
        <button
          type="button"
          onClick={() => void copy()}
          className="flex size-10 items-center justify-center rounded-full text-white/70 hover:bg-white/10 hover:text-white"
          aria-label="Copy the text"
        >
          {copied ? <Check className="size-4" aria-hidden /> : <Copy className="size-4" aria-hidden />}
        </button>
      </div>
      <pre
        className={
          'whitespace-pre-wrap break-words rounded-xl bg-black/30 p-3 font-sans text-[13px] leading-relaxed text-white/85 ' +
          (expanded || !long ? 'max-h-[60vh] overflow-y-auto' : 'max-h-48 overflow-hidden')
        }
      >
        {text.replace(/\f/g, '\n\n')}
      </pre>
      {long && (
        <button type="button" onClick={() => setExpanded((v) => !v)} className="mt-1 h-10 text-sm font-semibold text-white/80 hover:text-white">
          {expanded ? 'Show less' : 'Show all'}
        </button>
      )}
    </div>
  )
}
