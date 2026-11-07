// Mirrors backend/app/schemas.py exactly. Keep the two in sync.

export type ItemKind = 'image' | 'video' | 'pdf' | 'document' | 'other'
export type Category =
  | 'documents'
  | 'screenshots'
  | 'personal'
  | 'places'
  | 'food'
  | 'pets'
  | 'objects'
  | 'misc'
  | 'nsfw'
export const CATEGORIES: readonly Category[] = [
  'personal',
  'places',
  'food',
  'pets',
  'objects',
  'documents',
  'screenshots',
  'misc',
  'nsfw',
]
export const CATEGORY_LABELS: Record<Category, string> = {
  personal: 'People',
  places: 'Places',
  food: 'Food',
  pets: 'Pets',
  objects: 'Things',
  documents: 'Documents',
  screenshots: 'Screenshots',
  misc: 'Other',
  nsfw: 'Private',
}
export type AssignedBy = 'auto' | 'user' | 'guess'
export type AnalysisStatus = 'pending' | 'done' | 'error'
export type TextStatus = 'none' | 'pending' | 'done' | 'error'
export type PreviewStatus = 'none' | 'pending' | 'done' | 'error'

// auth
export interface AuthState {
  setup_required: boolean
  authenticated: boolean
  username: string | null
  passkeys: number
  /** ISO time until which the NSFW vault is unlocked for this session, or null. */
  vault_until: string | null
  session_id: number | null
  site_title: string
}
export interface SessionOut {
  id: number
  created_at: string
  last_seen_at: string
  user_agent: string | null
  ip: string | null
  current: boolean
}
export interface PasskeyOut {
  id: number
  name: string
  created_at: string
  last_used_at: string | null
  backup_eligible: boolean
}
export interface ChallengeOut {
  challenge_id: string
  options: Record<string, unknown>
}
export interface OkOut {
  ok: boolean
}

// people
export interface PersonRef {
  id: number
  name: string | null
  cover_url: string | null
}
export interface GuessOut {
  person: PersonRef
  score: number
  evidence: Record<string, unknown>
}
export interface FaceOut {
  id: number
  x: number
  y: number
  w: number
  h: number
  det_score: number
  size_px: number
  quality: number
  frame_time_s: number | null
  spans: [number, number][] | null
  person: PersonRef | null
  assigned_by: AssignedBy
  guess: GuessOut | null
  url: string
}
export interface BodyOut {
  id: number
  x: number
  y: number
  w: number
  h: number
  det_score: number
  face_id: number | null
  person: PersonRef | null
  assigned_by: AssignedBy
  guess: GuessOut | null
  frame_time_s: number | null
  url: string
}
export interface PersonOut {
  id: number
  name: string | null
  cover_url: string | null
  face_count: number
  item_count: number
  guess_count: number
  created_at: string
}
export interface PersonPage {
  items: PersonOut[]
  next_cursor: string | null
  total: number
}
export interface PersonCreate {
  name: string
  face_ids?: number[]
}
export interface PersonUpdate {
  name?: string | null
  cover_face_id?: number | null
}
export interface MergeIn {
  target_id: number
  source_ids: number[]
}
export interface FaceListItem {
  id: number
  item_uid: string
  item_kind: ItemKind
  thumb_url: string
  url: string
  det_score: number
  size_px: number
  quality: number
  frame_time_s: number | null
  person_id: number | null
  assigned_by: AssignedBy
  guess: GuessOut | null
  created_at: string
}
export interface FacePage {
  items: FaceListItem[]
  next_cursor: string | null
  total: number
}
export interface AssignFacesIn {
  face_ids: number[]
  person_id?: number | null
  new_person_name?: string | null
}
export interface AssignBodiesIn {
  body_ids: number[]
  person_id?: number | null
  new_person_name?: string | null
}
export interface AssignOut {
  updated: number
  person: PersonOut | null
}
export interface Suggestion {
  a: PersonOut
  b: PersonOut
  similarity: number
  a_samples: string[]
  b_samples: string[]
}
export interface SuggestionList {
  items: Suggestion[]
}
export interface TagIn {
  person_id?: number | null
  new_person_name?: string | null
}

// items
export interface ItemUrls {
  thumb: string
  preview: string
  preview_small: string
  original: string | null
  download: string | null
  video: string | null
  pages: string[]
}
export interface ItemCard {
  uid: string
  kind: ItemKind
  version: number
  filename: string
  width: number | null
  height: number | null
  duration_s: number | null
  taken_at: string | null
  created_at: string
  category: Category
  nsfw: boolean
  misclick: boolean
  misclick_reasons: Record<string, unknown> | null
  favorite: boolean
  /** uid of the group's best item when the item is a near-duplicate of others. */
  dup_group: string | null
  dup_count: number
  page_count: number | null
  doc_type: string | null
  urls: ItemUrls
}
export interface ItemPage {
  items: ItemCard[]
  next_cursor: string | null
  total: number
}
export interface DocumentField {
  value: string
  label: string
  confidence: number
}
export interface DocumentOut {
  doc_type: string
  doc_group: 'id' | 'money' | 'education_work' | 'health' | 'pass' | 'other'
  label: string
  confidence: number
  fields: Record<string, DocumentField>
  field_order: string[]
  title: string | null
  doc_date: string | null
  issuer: string | null
  person_name: string | null
  barcode: string | null
  barcode_format: string | null
  photo_url: string | null
  source: 'auto' | 'user'
  wallet_available: boolean
  updated_at: string
}
export interface GpsOut {
  lat: number
  lon: number
  alt: number | null
}
export interface LabelOut {
  label: string
  score: number
}
export interface ItemDetail extends ItemCard {
  mime: string
  size_bytes: number
  bitrate: number | null
  fps: number | null
  video_codec: string | null
  has_audio: boolean | null
  camera: string | null
  gps: GpsOut | null
  exif_datetime: string | null
  category_source: 'auto' | 'user'
  category_scores: Record<string, number> | null
  nsfw_score: number | null
  quality_scores: Record<string, unknown> | null
  labels: LabelOut[]
  faces: FaceOut[]
  bodies: BodyOut[]
  tags: PersonRef[]
  people: PersonRef[]
  document: DocumentOut | null
  text_status: TextStatus
  analysis_status: AnalysisStatus
  preview_status: PreviewStatus
  deleted_at: string | null
  updated_at: string
}
export interface ItemUpdate {
  category?: Category
  nsfw?: boolean
  misclick?: boolean
  favorite?: boolean
  filename?: string
}
export type BulkAction = 'delete' | 'restore' | 'keep' | 'category' | 'nsfw' | 'favorite' | 'unfavorite' | 'purge'
export interface BulkIn {
  uids: string[]
  action: BulkAction
  value?: string | null
}
export interface BulkOut {
  updated: number
}
export interface PageOut {
  page_no: number
  width: number | null
  height: number | null
  text: string | null
  text_source: string
  url: string
}
export interface ItemTextOut {
  text: string
  source: string
  char_count: number
}
export interface DupGroup {
  best: ItemCard
  copies: ItemCard[]
}
export interface DupGroupPage {
  groups: DupGroup[]
  next_cursor: string | null
  total: number
}
export interface GuessItem {
  kind: 'face' | 'body'
  id: number
  item: ItemCard
  box: FaceOut | BodyOut
  guess: GuessOut
}
export interface GuessPage {
  items: GuessItem[]
  next_cursor: string | null
  total: number
}
export interface GuessIdsIn {
  faces?: number[]
  bodies?: number[]
}
export interface DiskStats {
  free_bytes: number
  total_bytes: number
  originals_bytes: number
}
export interface Stats {
  items: number
  by_kind: Record<string, number>
  by_category: Record<string, number>
  people: number
  named_people: number
  documents: number
  misclicks: number
  guesses: number
  duplicates: number
  trash: number
  nsfw: number
  queue: { queued: number; running: number; failed: number; by_queue: Record<string, { queued: number; running: number }> }
  disk: DiskStats
  analysis: Record<string, number>
  text: Record<string, number>
  recent: ItemCard[]
}

// documents
export interface DocumentListItem {
  item: ItemCard
  document: DocumentOut
}
export interface DocumentPage {
  items: DocumentListItem[]
  next_cursor: string | null
  total: number
}
export interface DocumentTypeOut {
  type: string
  group: string
  label: string
  count: number
}
export interface DocumentUpdate {
  doc_type?: string
  fields?: Record<string, string>
  title?: string
  doc_date?: string
  person_name?: string
}

// search
export type SearchMode = 'auto' | 'semantic' | 'text'
export interface SearchHit extends ItemCard {
  score: number
  match_time_s: number | null
  snippet: string | null
  source: 'semantic' | 'text' | 'both'
}
export interface SearchPage {
  query: string
  mode: SearchMode
  items: SearchHit[]
  total: number
  next_offset: number | null
  indexed: number
  indexing: boolean
}

// uploads
export type UploadStatus =
  | 'uploading'
  | 'queued'
  | 'processing'
  | 'stored'
  | 'duplicate'
  | 'rejected'
  | 'error'
  | 'cancelled'
export const TERMINAL_UPLOAD_STATUSES: ReadonlySet<UploadStatus> = new Set([
  'stored',
  'duplicate',
  'rejected',
  'error',
  'cancelled',
])
export type UploadSource = 'web' | 'share' | 'camera' | 'api'
export interface UploadCreate {
  filename: string
  size: number
  mime?: string | null
  client_id?: string | null
  sha256?: string | null
  source?: UploadSource
}
export interface UploadOut {
  id: string
  filename: string
  size: number
  received_bytes: number
  status: UploadStatus
  message: string | null
  chunk_size: number
  item_uid: string | null
  item_thumb_url: string | null
  created_at: string
}
export interface UploadList {
  uploads: UploadOut[]
}
export interface ChunkOut {
  received_bytes: number
}

// system
export interface ModelStatus {
  key: string
  label: string
  present: boolean
  loaded: boolean
  provider: string | null
  size_bytes: number
  downloading: boolean
  error: string | null
}
export interface SystemOut {
  version: string
  site_title: string
  gpu: { available: boolean; name: string | null; provider: string | null }
  models: ModelStatus[]
  queues: Stats['queue']
  disk: DiskStats
  wallet: { configured: boolean; missing: string[]; dir: string }
  timezone: string
  ffmpeg: boolean
  workers: boolean
  download: { running: boolean; progress: string | null; error: string | null } | null
}
export interface ConfigOut {
  site_title: string
  max_upload_bytes: number
  simple_upload_max_bytes: number
  chunk_size: number
  accept: string
  categories: Category[]
  vault_unlock_min: number
  trash_days: number
}
