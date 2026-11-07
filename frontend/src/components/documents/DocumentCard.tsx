// A parsed document as a compact "virtual document" tile: the type's colour and icon, title, person,
// date, a few key fields (identifiers masked) and the ID photo when one was found.
import clsx from 'clsx'
import type { DocumentOut, ItemCard } from '../../lib/types'
import { formatPlainDate } from '../../lib/format'
import { docTypeLabel, groupMeta, keyFields } from './docMeta'

interface Props {
  item: ItemCard
  doc: DocumentOut
  onOpen?: () => void
  className?: string
}

export default function DocumentCard({ item, doc, onOpen, className }: Props) {
  const meta = groupMeta(doc.doc_group)
  const Icon = meta.icon
  const fields = keyFields(doc, 3)
  const title = doc.title || doc.label || docTypeLabel(doc.doc_type)
  const body = (
    <>
      <span
        className="flex items-center gap-2 px-3 py-2 text-white"
        style={{ backgroundImage: `linear-gradient(120deg, ${meta.from}, ${meta.to})` }}
      >
        <Icon className="size-4 shrink-0" aria-hidden />
        <span className="min-w-0 flex-1 truncate text-xs font-semibold uppercase tracking-wide">{doc.label || docTypeLabel(doc.doc_type)}</span>
        {doc.doc_date && <span className="shrink-0 text-xs text-white/85">{formatPlainDate(doc.doc_date)}</span>}
      </span>
      <span className="flex gap-3 p-3">
        {doc.photo_url ? (
          <img src={doc.photo_url} alt="" loading="lazy" className="h-20 w-16 shrink-0 rounded-md bg-surface-2 object-cover ring-1 ring-line" />
        ) : (
          <img src={item.urls.thumb} alt="" loading="lazy" className="h-20 w-16 shrink-0 rounded-md bg-white object-contain object-top ring-1 ring-line" />
        )}
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-semibold text-ink">{title}</span>
          {doc.person_name && <span className="block truncate text-sm text-ink-soft">{doc.person_name}</span>}
          {doc.issuer && !doc.person_name && <span className="block truncate text-sm text-ink-soft">{doc.issuer}</span>}
          <span className="mt-1.5 block space-y-0.5">
            {fields.map((f) => (
              <span key={f.key} className="flex gap-2 text-xs">
                <span className="w-20 shrink-0 truncate text-ink-faint">{f.label}</span>
                <span className="min-w-0 truncate font-medium tabular-nums text-ink">{f.value}</span>
              </span>
            ))}
          </span>
        </span>
      </span>
    </>
  )
  const cls = clsx(
    'flex w-full flex-col overflow-hidden rounded-2xl border border-line bg-surface text-left shadow-sm transition',
    onOpen && 'hover:-translate-y-0.5 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
    className,
  )
  if (!onOpen) return <div className={cls}>{body}</div>
  return (
    <button type="button" onClick={onOpen} className={cls} aria-label={`Open ${title}${doc.person_name ? `, ${doc.person_name}` : ''}`}>
      {body}
    </button>
  )
}
