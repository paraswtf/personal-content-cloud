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
  /** The server's API build id; the shell compares it with __API_BUILD__ (older servers omit it). */
  api_build?: string
}
export interface SessionOut {
  id: number
  created_at: string
  last_seen_at: string
  user_agent: string | null
  ip: string | null
  current: boolean
  /** The name given to an iOS app device token; null for a browser session. */
  device_name: string | null
}
/** POST /api/auth/devices: the token is returned this once and never again. */
export interface DeviceCreated {
  id: number
  name: string
  token: string
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
export type Species = 'bird' | 'cat' | 'dog' | 'horse' | 'sheep' | 'cow' | 'elephant' | 'bear' | 'zebra' | 'giraffe'
export type PersonKind = 'person' | 'pet'
export interface PersonRef {
  id: number
  name: string | null
  cover_url: string | null
  /** Older servers omit it: treat a missing kind as 'person'. */
  kind: PersonKind
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
/** An animal box (section 22): a pet found by its look, assigned to a pet identity. */
export interface AnimalOut {
  id: number
  x: number
  y: number
  w: number
  h: number
  det_score: number
  species: Species
  frame_time_s: number | null
  person: PersonRef | null
  assigned_by: 'auto' | 'user'
  url: string
}
export interface PersonOut {
  id: number
  name: string | null
  cover_url: string | null
  cover_face_id: number | null
  kind: PersonKind
  /** Set for a pet, null for a person. */
  species: Species | null
  animal_count: number
  face_count: number
  item_count: number
  guess_count: number
  created_at: string
  /** Faces (or crops) the owner assigned: the anchors of the person's reference. */
  confirmed_faces?: number
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
export interface AnimalListItem {
  id: number
  item_uid: string
  item_kind: ItemKind
  thumb_url: string
  url: string
  det_score: number
  species: Species
  frame_time_s: number | null
  person_id: number | null
  assigned_by: 'auto' | 'user'
  created_at: string
}
export interface AnimalPage {
  items: AnimalListItem[]
  next_cursor: string | null
  total: number
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
export interface AssignAnimalsIn {
  animal_ids: number[]
  person_id?: number | null
  new_person_name?: string
}
export interface AssignOut {
  updated: number
  person: PersonOut | null
  /** Faces that looked like the ones taken away from a person and went the same way (section 10). */
  followed?: number
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

// free-form tags (section 19): the owner's own tags, and the ones the tag learner suggests
export interface TagOut {
  id: number
  name: string
  /** Live items the session may see with this tag added by the owner. */
  items: number
  /** Live items where the tag is only suggested. */
  auto_items: number
}
export type ItemTagSource = 'user' | 'auto'
export interface ItemTagOut {
  id: number
  name: string
  source: ItemTagSource
  /** Similarity of the suggestion (auto rows); null for tags the owner added. */
  score: number | null
}
/** POST /api/items/{uid}/user-tags: an existing tag by id, or a name (created when new). */
export interface UserTagIn {
  tag_id?: number
  name?: string
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
  /** Live Photo or motion photo clip (section 20): the companion's web video or /files/live/{uid}?v=; null otherwise. */
  live: string | null
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
  /** The still has a clip: a paired iPhone MOV or an embedded Android motion clip (section 20). */
  live: boolean
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
/** The clip of a Live Photo or motion photo (section 20). */
export interface LiveVideoOut {
  /** The hidden companion item's uid for an iPhone pair; null for a clip extracted from a motion photo. */
  uid: string | null
  /** Playable URL: the companion's web video, or /files/live/{uid}?v= for an extracted clip. */
  url: string
  duration_s: number | null
}
/** An album the item belongs to (section 24). */
export interface AlbumRef {
  id: number
  name: string
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
  animals: AnimalOut[]
  /** Albums containing the item (section 24). */
  albums: AlbumRef[]
  /** Place name from the gazetteer, when the photo has GPS and the spot is known (section 25). */
  place: string | null
  tags: PersonRef[]
  people: PersonRef[]
  /** Free-form tags: the owner's first, then suggestions by score. */
  user_tags: ItemTagOut[]
  document: DocumentOut | null
  text_status: TextStatus
  analysis_status: AnalysisStatus
  preview_status: PreviewStatus
  deleted_at: string | null
  updated_at: string
  live_video: LiveVideoOut | null
}
export interface ItemUpdate {
  category?: Category
  nsfw?: boolean
  misclick?: boolean
  favorite?: boolean
  filename?: string
}
export type BulkAction =
  | 'delete'
  | 'restore'
  | 'keep'
  | 'category'
  | 'nsfw'
  | 'favorite'
  | 'unfavorite'
  | 'purge'
  | 'album'
  | 'unalbum'
  /** value: the tag name (created when new) */
  | 'tag'
  /** value: the tag name */
  | 'untag'
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
  /** The device's id for the asset; both halves of a Live Photo share it (sections 20 and 28). */
  pair_key?: string | null
  resource?: 'still' | 'video' | 'file'
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
  /** True only while the vault is open and the stored item is in it: link with ?vault=1. */
  item_nsfw: boolean
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
/** The web copy encoder and queue (section 27). */
export interface VideoStatus {
  encoder: string | null
  hw: boolean
  device: string | null
  reason: string | null
  checked_at: string | null
  queued: number
  running: number
  failed: number
  done: number
  pending: number
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
  video: VideoStatus | null
}
// learning (section 18): what the small models learned from the owner's corrections
export type LearningEventKind = 'category' | 'vault' | 'misclick' | 'doc_type' | 'guess' | 'tag'
export type MisclickThresholdKey = 'blur_var_min' | 'dark_mean' | 'clipped_frac' | 'accident_min'
/** How readily photos go to the vault: low sends fewer, high also sends suggestive but not explicit ones. */
export type VaultSensitivity = 'low' | 'normal' | 'high'
export interface LearningStatus {
  enabled: boolean
  /** The owner's vault setting, changed with PATCH /api/system/learning {vault_sensitivity}. */
  vault_sensitivity: VaultSensitivity
  /** On the reply to a sensitivity change only: what the immediate re-sort of automatically decided
   *  items did. `stale` items were analysed before the signals existed and need "apply to existing items". */
  vault_moved?: { checked: number; moved_in: number; moved_out: number; stale: number } | null
  trained_at: string | null
  pending: number
  learners: {
    doc_type: { examples: number; labels: string[] }
    category: { examples: number; labels: string[] }
    /** `weights` is set once enough vault decisions were corrected to learn them; null before that. */
    vault: { examples: number; threshold: number | null; weights: Record<string, number> | null }
    misclick: { examples: number; thresholds: Partial<Record<MisclickThresholdKey, number>> }
    people: { persons: number; events: number; weights: Record<string, number> | null }
    /** Section 19: tags learned, items tagged by the owner, suggestions removed. Optional: older servers lack it. */
    tags?: { tags: number; examples: number; rejections: number }
  }
  events: Partial<Record<LearningEventKind, number>>
}
export interface LearningApplyOut {
  items: number
  documents: number
  /** Suggested tags written for existing items (section 19). */
  tags?: number
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
  /** Map defaults (section 25). */
  map: { tiles: 'offline' | 'online'; style_url: string }
}
