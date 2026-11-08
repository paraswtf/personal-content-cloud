// Display metadata for parsed document types: labels, group colours and icons, key fields and masking.
import {
  BadgeCheck,
  Banknote,
  FileText,
  GraduationCap,
  HeartPulse,
  IdCard,
  SquareParking,
  Ticket,
  TrafficCone,
  type LucideIcon,
} from 'lucide-react'
import type { DocumentOut } from '../../lib/types'
import { formatDate } from '../../lib/format'

export type DocGroup = DocumentOut['doc_group']

export const GROUP_META: Record<DocGroup, { label: string; icon: LucideIcon; from: string; to: string }> = {
  id: { label: 'Identity', icon: IdCard, from: '#1d4ed8', to: '#3b82f6' },
  money: { label: 'Money', icon: Banknote, from: '#047857', to: '#10b981' },
  education_work: { label: 'Education and work', icon: GraduationCap, from: '#6d28d9', to: '#8b5cf6' },
  health: { label: 'Health and insurance', icon: HeartPulse, from: '#be123c', to: '#f43f5e' },
  pass: { label: 'Passes and tickets', icon: Ticket, from: '#c2410c', to: '#f59e0b' },
  other: { label: 'Other', icon: FileText, from: '#374151', to: '#6b7280' },
}

export const GROUP_ORDER: DocGroup[] = ['id', 'money', 'education_work', 'health', 'pass', 'other']

export function groupMeta(group: string | null | undefined) {
  return GROUP_META[(group as DocGroup) in GROUP_META ? (group as DocGroup) : 'other']
}

export const DOC_TYPE_LABELS: Record<string, string> = {
  aadhaar: 'Aadhaar card',
  pan: 'PAN card',
  passport: 'Passport',
  driving_licence: 'Driving licence',
  voter_id: 'Voter ID',
  vehicle_rc: 'Vehicle RC',
  id_card: 'ID card',
  bank_statement: 'Bank statement',
  invoice: 'Invoice',
  receipt: 'Receipt',
  traffic_challan: 'Traffic challan',
  parking_receipt: 'Parking receipt',
  salary_slip: 'Salary slip',
  marksheet: 'Marksheet',
  degree: 'Degree',
  certificate: 'Certificate',
  offer_letter: 'Offer letter',
  insurance_policy: 'Insurance policy',
  prescription: 'Prescription',
  lab_report: 'Lab report',
  vaccination: 'Vaccination record',
  event_pass: 'Event pass',
  boarding_pass: 'Boarding pass',
  other: 'Document',
}

export function docTypeLabel(type: string | null | undefined): string {
  if (!type) return 'Document'
  return DOC_TYPE_LABELS[type] ?? type.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())
}

/** Icons for the types whose group icon says too little (section 30); the group's icon otherwise. */
export const TYPE_ICONS: Record<string, LucideIcon> = {
  id_card: IdCard,
  traffic_challan: TrafficCone,
  parking_receipt: SquareParking,
}

export function docTypeIcon(type: string | null | undefined, group: string | null | undefined): LucideIcon {
  return (type && TYPE_ICONS[type]) || groupMeta(group).icon
}

/** Field order and labels the frontend knows on its own (section 30); the server's field_order and
 *  labels win when they are there. */
export const TYPE_FIELDS: Record<string, { key: string; label: string }[]> = {
  traffic_challan: [
    { key: 'challan_number', label: 'Challan number' },
    { key: 'date', label: 'Date' },
    { key: 'vehicle_number', label: 'Vehicle number' },
    { key: 'vehicle_owner', label: 'Vehicle owner' },
    { key: 'owner_address', label: 'Owner address' },
    { key: 'violations', label: 'Violations' },
    { key: 'amount', label: 'Amount' },
    { key: 'payment_status', label: 'Payment status' },
    { key: 'location', label: 'Location' },
    { key: 'officer', label: 'Officer' },
    { key: 'mobile', label: 'Mobile' },
    { key: 'evidences', label: 'Evidences' },
    { key: 'pay_url', label: 'Pay at' },
  ],
  parking_receipt: [
    { key: 'receipt_number', label: 'Receipt number' },
    { key: 'vehicle_number', label: 'Vehicle number' },
    { key: 'vehicle_type', label: 'Vehicle type' },
    { key: 'in_time', label: 'In' },
    { key: 'out_time', label: 'Out' },
    { key: 'duration', label: 'Duration' },
    { key: 'amount', label: 'Amount' },
    { key: 'operator', label: 'Operator' },
    { key: 'location', label: 'Location' },
    { key: 'tariff', label: 'Tariff' },
  ],
}

/** The fields a tile shows first for these types. */
export const KEY_FIELDS: Record<string, string[]> = {
  traffic_challan: ['challan_number', 'amount', 'payment_status'],
  parking_receipt: ['receipt_number', 'amount', 'in_time'],
}

/** The portal a challan without a pay_url opens (section 30). */
export const PARIVAHAN_CHALLAN_URL = 'https://echallan.parivahan.gov.in/index/accused-challan'

/** A field's label: the server's, else the type's own, else the key. */
export function fieldLabel(doc: Pick<DocumentOut, 'fields' | 'doc_type'>, key: string): string {
  return doc.fields[key]?.label || TYPE_FIELDS[doc.doc_type]?.find((f) => f.key === key)?.label || key
}

/** Timestamp fields (in_time, out_time) are shown like other dates, with the time. */
const TIME_FIELDS = new Set(['in_time', 'out_time'])

/** The value as the card shows it: a timestamp field formatted, anything else as stored. */
export function displayValue(key: string, value: string): string {
  if (TIME_FIELDS.has(key)) return formatDate(value, true) || value
  return value
}

/** Short file-type label for a document tile without a parsed type ("PDF", "DOCX"). */
export function extLabel(filename: string): string {
  const m = /\.([a-z0-9]{1,5})$/i.exec(filename)
  return m ? m[1].toUpperCase() : 'FILE'
}

const NUMBER_HINTS = ['number', 'no', 'pnr', 'epic', 'pan', 'uid', 'booking_id', 'policy', 'account', 'roll']

/** Whether a field holds an identifier that should be masked on tiles. */
export function isNumberField(key: string): boolean {
  const k = key.toLowerCase()
  return NUMBER_HINTS.some((h) => k === h || k.endsWith(`_${h}`) || k.startsWith(`${h}_`) || k.includes('number'))
}

/** "1234 5678 9012" -> "XXXX XXXX 9012": keeps the last four letters or digits and the spacing. */
export function maskValue(value: string): string {
  const total = (value.match(/[A-Za-z0-9]/g) ?? []).length
  if (total <= 4) return value
  let seen = 0
  return value.replace(/[A-Za-z0-9]/g, (c) => (++seen <= total - 4 ? 'X' : c))
}

/** Field keys in display order: the server's order (else the type's own, section 30) first, then any others. */
export function orderedKeys(doc: Pick<DocumentOut, 'fields' | 'field_order'> & { doc_type?: string }): string[] {
  const order = doc.field_order.length ? doc.field_order : (TYPE_FIELDS[doc.doc_type ?? ''] ?? []).map((f) => f.key)
  const keys = order.filter((k) => k in doc.fields)
  for (const k of Object.keys(doc.fields)) if (!keys.includes(k)) keys.push(k)
  return keys
}

const SKIP_ON_TILE = new Set(['name', 'full_name', 'holder_name', 'address', 'pay_url'])

/** Up to `n` key fields for a tile: the identifier (masked) first, then the next fields in order. */
export function keyFields(doc: DocumentOut, n = 3): { key: string; label: string; value: string }[] {
  const keys = orderedKeys(doc).filter((k) => doc.fields[k]?.value && !SKIP_ON_TILE.has(k))
  const wanted = (KEY_FIELDS[doc.doc_type] ?? []).filter((k) => keys.includes(k))
  const num = keys.find(isNumberField)
  const picked = wanted.length ? [...wanted, ...keys.filter((k) => !wanted.includes(k))] : num ? [num, ...keys.filter((k) => k !== num)] : keys
  return picked.slice(0, n).map((k) => ({
    key: k,
    label: fieldLabel(doc, k),
    value: isNumberField(k) && doc.doc_group === 'id' ? maskValue(doc.fields[k].value) : displayValue(k, doc.fields[k].value),
  }))
}

/** Confidence under which a value is shaded as "check this". */
export const LOW_CONFIDENCE = 0.6

export const VerifiedIcon = BadgeCheck
