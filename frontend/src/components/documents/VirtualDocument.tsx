// The full virtual document of an item: every parsed field, editable in place (saved with
// source 'user'), low-confidence values shaded, the barcode payload, Reparse, the Apple Wallet
// button for passes, and for a traffic challan (section 30) the status chip, "Pay or check status"
// (copies the challan number, opens the portal) and "Mark as paid".
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { Link } from 'react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Barcode, Check, Copy, ExternalLink, Loader2, Pencil, RefreshCw, Wallet } from 'lucide-react'
import { api } from '../../lib/api'
import type { DocumentOut, DocumentUpdate } from '../../lib/types'
import { formatPercent, formatPlainDate, localDateInput } from '../../lib/format'
import { useRefreshLibrary } from '../../lib/hooks'
import Select from '../Select'
import { useToast } from '../Toast'
import { errorMessage } from '../admin/adminHooks'
import { DOC_TYPE_LABELS, LOW_CONFIDENCE, PARIVAHAN_CHALLAN_URL, displayValue, docTypeIcon, docTypeLabel, fieldLabel, groupMeta, orderedKeys } from './docMeta'

interface Props {
  uid: string
  doc: DocumentOut
  /** Show the ID photo (it is hidden when the viewer already shows the picture). */
  showPhoto?: boolean
  /** Where feedback goes; defaults to the app toasts (the viewer passes its own, drawn above it). */
  notify?: { success: (text: string) => void; error: (text: string) => void }
}

function EditableValue({
  label,
  value,
  display,
  confidence,
  placeholder = 'Not found',
  saving,
  onSave,
  mono = false,
  inputType = 'text',
}: {
  label: string
  value: string
  display?: ReactNode
  confidence?: number
  placeholder?: string
  saving: boolean
  onSave: (next: string) => void
  mono?: boolean
  /** 'date' for YYYY-MM-DD values (the API takes doc_date in that form). */
  inputType?: 'text' | 'date'
}) {
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState(value)
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (!editing) setText(value)
  }, [value, editing])
  useEffect(() => {
    if (editing && inputType === 'text') input.current?.select()
    else if (editing) input.current?.focus()
  }, [editing])

  const commit = () => {
    setEditing(false)
    if (text.trim() !== value) onSave(text.trim())
  }
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      commit()
    } else if (e.key === 'Escape') {
      e.preventDefault()
      e.stopPropagation()
      setText(value)
      setEditing(false)
    }
  }
  const low = confidence !== undefined && confidence < LOW_CONFIDENCE

  return (
    <div className={clsx('group rounded-lg px-2 py-1.5', low && 'bg-warn/10')}>
      <dt className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-ink-faint">
        {label}
        {low && (
          <span className="normal-case tracking-normal text-warn" title={`Read with ${formatPercent(confidence)} confidence`}>
            check this
          </span>
        )}
      </dt>
      <dd className="mt-0.5 flex min-h-8 items-center gap-1">
        {editing ? (
          <>
            <input
              ref={input}
              type={inputType}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={onKey}
              onBlur={commit}
              aria-label={label}
              className="h-9 min-w-0 flex-1 rounded-lg border border-accent bg-surface px-2 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-accent/20"
            />
            <button
              type="button"
              onMouseDown={(e) => e.preventDefault()}
              onClick={commit}
              className="flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft hover:bg-surface-2"
              aria-label={`Save ${label}`}
            >
              <Check className="size-4" aria-hidden />
            </button>
          </>
        ) : (
          <>
            <span className={clsx('min-w-0 flex-1 break-words text-sm', value ? 'text-ink' : 'text-ink-faint', mono && 'font-mono tabular-nums')}>
              {value ? (display ?? value) : placeholder}
            </span>
            {saving ? (
              <Loader2 className="size-4 shrink-0 animate-spin text-ink-faint" aria-hidden />
            ) : (
              <button
                type="button"
                onClick={() => setEditing(true)}
                className="flex size-10 shrink-0 items-center justify-center rounded-full text-ink-faint opacity-70 hover:bg-surface-2 hover:text-ink group-hover:opacity-100"
                aria-label={`Edit ${label}`}
              >
                <Pencil className="size-3.5" aria-hidden />
              </button>
            )}
          </>
        )}
      </dd>
    </div>
  )
}

/** A challan's payment status as a chip: Unpaid (or pending, due) in the warning colour, anything else plain. */
function StatusChip({ value }: { value: string }) {
  const v = value.trim()
  const unpaid = /unpaid|pending|due/i.test(v)
  return (
    <span className={clsx('inline-flex h-6 items-center rounded-full px-2.5 text-xs font-semibold', unpaid ? 'bg-warn/20 text-warn' : 'bg-surface-2 text-ink')}>
      {v}
    </span>
  )
}

export default function VirtualDocument({ uid, doc, showPhoto = true, notify }: Props) {
  const appToast = useToast()
  const toast = notify ?? appToast
  const qc = useQueryClient()
  const refresh = useRefreshLibrary()
  const meta = groupMeta(doc.doc_group)
  const Icon = docTypeIcon(doc.doc_type, doc.doc_group)
  const isChallan = doc.doc_type === 'traffic_challan'
  const [savingKey, setSavingKey] = useState<string | null>(null)

  const types = useQuery({ queryKey: ['document-types'], queryFn: api.documents.types, staleTime: 5 * 60_000 })
  const typeOptions = useMemo(() => {
    const seen = new Map<string, string>()
    for (const t of types.data ?? []) seen.set(t.type, t.label || docTypeLabel(t.type))
    for (const [k, v] of Object.entries(DOC_TYPE_LABELS)) if (!seen.has(k)) seen.set(k, v)
    if (!seen.has(doc.doc_type)) seen.set(doc.doc_type, doc.label || docTypeLabel(doc.doc_type))
    return [...seen.entries()].map(([value, label]) => ({ value, label }))
  }, [types.data, doc.doc_type, doc.label])

  const update = useMutation({
    mutationFn: (v: { key: string; body: DocumentUpdate }) => api.documents.update(uid, v.body),
    onMutate: (v) => setSavingKey(v.key),
    onSettled: () => setSavingKey(null),
    onSuccess: (next) => {
      qc.setQueriesData({ queryKey: ['item', uid] }, (old: unknown) =>
        old && typeof old === 'object' ? { ...(old as object), document: { ...next, photo_url: doc.photo_url ?? next.photo_url } } : old,
      )
      void refresh()
      toast.success('Saved')
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not save the change.')),
  })
  const reparse = useMutation({
    mutationFn: () => api.documents.reparse(uid),
    onSuccess: () => {
      void refresh()
      toast.success('Reading the document again. This can take a minute.')
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not start reading it again.')),
  })

  const save = (key: string, body: DocumentUpdate) => update.mutate({ key, body })
  const keys = orderedKeys(doc).filter((k) => k !== 'pay_url')   // the link is the button below
  // section 30: the challan's portal and status
  const challanNumber = doc.fields.challan_number?.value?.trim() ?? ''
  const payUrl = doc.fields.pay_url?.value?.trim() || PARIVAHAN_CHALLAN_URL
  const paid = /^paid$/i.test(doc.fields.payment_status?.value?.trim() ?? '')
  const copyChallan = async () => {
    if (!challanNumber) return
    try {
      await navigator.clipboard.writeText(challanNumber)
      toast.success('Challan number copied; the portal opens in a new tab')
    } catch {
      toast.error('Could not copy the challan number.')
    }
  }
  const markPaid = () => save('f:payment_status', { fields: { payment_status: 'Paid', paid_on: localDateInput() } })
  const [copied, setCopied] = useState(false)
  const copyBarcode = async () => {
    if (!doc.barcode) return
    try {
      await navigator.clipboard.writeText(doc.barcode)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      toast.error('Could not copy.')
    }
  }

  return (
    <article className="overflow-hidden rounded-2xl border border-line bg-surface text-ink shadow-sm" aria-label={`${doc.label || docTypeLabel(doc.doc_type)} details`}>
      <header className="flex items-start gap-3 px-4 py-3 text-white" style={{ backgroundImage: `linear-gradient(120deg, ${meta.from}, ${meta.to})` }}>
        <span className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-xl bg-white/20">
          <Icon className="size-5" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-white/80">{meta.label}</p>
          <p className="truncate text-base font-semibold">{doc.title || doc.label || docTypeLabel(doc.doc_type)}</p>
          <p className="text-xs text-white/80">
            {doc.source === 'user' ? 'Checked by you' : `Read automatically, ${formatPercent(doc.confidence)} sure`}
          </p>
        </div>
      </header>

      <div className="flex flex-col gap-3 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <Select
            label="Document type"
            value={doc.doc_type}
            options={typeOptions}
            onChange={(v) => v !== doc.doc_type && save('doc_type', { doc_type: v })}
           
            className="max-w-full"
          />
        </div>

        <div className="flex gap-3">
          {showPhoto && doc.photo_url && (
            <img src={doc.photo_url} alt="Photo on the document" className="h-32 w-24 shrink-0 rounded-lg bg-surface-2 object-cover ring-1 ring-line" />
          )}
          <dl className="grid min-w-0 flex-1 grid-cols-1 gap-0.5">
            <EditableValue label="Title" value={doc.title ?? ''} saving={savingKey === 'title'} onSave={(v) => save('title', { title: v })} />
            <EditableValue
              label="Name"
              value={doc.person_name ?? ''}
              saving={savingKey === 'person_name'}
              onSave={(v) => save('person_name', { person_name: v })}
            />
            <EditableValue
              label="Date"
              value={doc.doc_date ?? ''}
              display={formatPlainDate(doc.doc_date)}
              placeholder="No date"
              inputType="date"
              saving={savingKey === 'doc_date'}
              onSave={(v) => save('doc_date', { doc_date: v })}
            />
          </dl>
        </div>

        {keys.length > 0 ? (
          <dl className="grid grid-cols-1 gap-0.5 border-t border-line pt-2 sm:grid-cols-2">
            {keys.map((k) => {
              const f = doc.fields[k]
              return (
                <EditableValue
                  key={k}
                  label={fieldLabel(doc, k)}
                  value={f.value}
                  display={isChallan && k === 'payment_status' && f.value ? <StatusChip value={f.value} /> : displayValue(k, f.value)}
                  confidence={f.confidence}
                  saving={savingKey === `f:${k}`}
                  onSave={(v) => save(`f:${k}`, { fields: { [k]: v } })}
                  mono={/number|no$|pnr|id$/i.test(k)}
                />
              )
            })}
          </dl>
        ) : (
          <p className="border-t border-line pt-3 text-sm text-ink-soft">No fields were read from this document yet.</p>
        )}

        {doc.barcode && (
          <div className="rounded-xl bg-surface-2 p-3">
            <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-ink-faint">
              <Barcode className="size-3.5" aria-hidden />
              Barcode{doc.barcode_format ? `, ${doc.barcode_format}` : ''}
            </p>
            <div className="mt-1 flex items-start gap-1">
              <p className="max-h-28 min-w-0 flex-1 overflow-y-auto break-all font-mono text-xs text-ink">{doc.barcode}</p>
              <button
                type="button"
                onClick={() => void copyBarcode()}
                className="flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft hover:bg-surface"
                aria-label="Copy the barcode content"
              >
                {copied ? <Check className="size-4" aria-hidden /> : <Copy className="size-4" aria-hidden />}
              </button>
            </div>
          </div>
        )}

        {isChallan && (
          <div className="flex flex-wrap items-center gap-2 border-t border-line pt-3">
            <a
              href={payUrl}
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => void copyChallan()}
              className="inline-flex h-10 items-center gap-2 rounded-full bg-ink px-4 text-sm font-semibold text-surface hover:opacity-90"
            >
              <ExternalLink className="size-4" aria-hidden />
              Pay or check status
            </a>
            {!paid && (
              <button
                type="button"
                onClick={markPaid}
                disabled={savingKey === 'f:payment_status'}
                className="inline-flex h-10 items-center gap-2 rounded-full border border-line px-4 text-sm font-semibold text-ink hover:bg-surface-2 disabled:opacity-50"
              >
                <Check className="size-4" aria-hidden />
                Mark as paid
              </button>
            )}
            <p className="basis-full text-xs text-ink-soft">
              The portal asks for the challan number and a captcha; the number is copied when you open it. The status shown is the one printed on the challan until you mark it paid.
            </p>
          </div>
        )}

        <div className="flex flex-wrap gap-2 border-t border-line pt-3">
          <button
            type="button"
            onClick={() => reparse.mutate()}
            disabled={reparse.isPending}
            className="inline-flex h-10 items-center gap-2 rounded-full border border-line px-4 text-sm font-semibold text-ink hover:bg-surface-2 disabled:opacity-50"
          >
            <RefreshCw className={clsx('size-4', reparse.isPending && 'animate-spin')} aria-hidden />
            Reparse
          </button>
          {doc.wallet_available ? (
            <a
              href={api.documents.walletUrl(uid)}
              download
              className="inline-flex h-10 items-center gap-2 rounded-full bg-ink px-4 text-sm font-semibold text-surface hover:opacity-90"
            >
              <Wallet className="size-4" aria-hidden />
              Add to Apple Wallet
            </a>
          ) : (
            doc.doc_group === 'pass' && (
              <span className="flex flex-col gap-1">
                <button
                  type="button"
                  disabled
                  aria-describedby={`wallet-hint-${uid}`}
                  className="inline-flex h-10 cursor-not-allowed items-center gap-2 rounded-full border border-line px-4 text-sm font-semibold text-ink-faint"
                >
                  <Wallet className="size-4" aria-hidden />
                  Add to Apple Wallet
                </button>
                <span id={`wallet-hint-${uid}`} className="text-xs text-ink-soft">
                  Wallet signing is not set up yet. See{' '}
                  <Link to="/settings" className="font-semibold text-accent underline underline-offset-2">
                    Settings
                  </Link>
                  .
                </span>
              </span>
            )
          )}
        </div>
      </div>
    </article>
  )
}

export function LowConfidenceLegend() {
  return (
    <p className="flex items-center gap-2 text-xs text-ink-soft">
      <span className="inline-block size-3 rounded bg-warn/20" aria-hidden />
      Shaded values were hard to read. Check them.
    </p>
  )
}
