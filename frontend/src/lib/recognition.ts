// Recognition trained on one person (docs/ARCHITECTURE.md section 10): the status of the model and of a run, the
// two lists the owner settles (conflicts, faces on other people), and the actions. What it finds becomes guesses
// with basis 'face' (components/people/evidence.ts); nothing is tagged by it.
import { del, get, post, qs } from './api'
import type { FaceListItem, OkOut, PersonRef } from './types'

/** One cut of the check: of the faces it was not trained on, how many of the person's it found at probability p
 *  or more and how many faces of other people it mistook for them (unclear: blurred, small or turned faces). */
export interface RecognitionCut {
  p: number
  found: number
  mistaken: number
  unclear_found: number
  unclear_mistaken: number
}
export interface RecognitionCheck {
  /** What it was tested on: days it was not trained on, or (when the days do not spread the faces) photos. */
  folds?: 'day' | 'photo' | null
  positives: number
  negatives: number
  unclear_positives: number
  unclear_negatives: number
  cuts: RecognitionCut[]
}
export interface RecognitionScan {
  at: string
  looked: number
  guesses: number
  sure: number
  detached: number
  elsewhere: number
}
export interface RecognitionOut {
  available: boolean
  /** Why it cannot be trained or used (learning off, too few clear faces). */
  reason: string | null
  /** Learning is on in Settings. Off, a trained recognition is not used. */
  learning: boolean
  trained: boolean
  trained_at: string | null
  counts: Record<string, number>
  check: RecognitionCheck | null
  guess_p: number | null
  sure_p: number | null
  /** Faces on the person that look the same as faces the owner took off them, still to settle. */
  conflicts: number
  /** Faces the app put on other named people that look like the person. */
  elsewhere: number
  /** The model's open guesses now. */
  guesses: number
  /** Its guesses were taken back: it makes none until the next look. */
  withdrawn: boolean
  scan: RecognitionScan | null
  running: 'train' | 'scan' | null
  progress: { done: number; total: number; step: string } | null
  /** Why the last run failed, or that it stopped when the app was restarted. */
  error: string | null
  /** The owner's answers about this person since the training. */
  answers_since: number
  /** A run for another person is going. */
  busy: boolean
}
export interface RecognitionConflict {
  face: FaceListItem
  /** The face it looks like; null when that one cannot be shown. */
  like: FaceListItem | null
  like_person: PersonRef | null
  sim: number
}
export interface RecognitionConflictPage {
  items: RecognitionConflict[]
  next_cursor: string | null
  total: number
}
export interface RecognitionElsewhere {
  face: FaceListItem
  person: PersonRef
  p: number
}

export const recognitionApi = {
  status: (id: number) => get<RecognitionOut>(`/api/people/${id}/recognition`),
  /** Train on the faces now on the person, then look again (scan) unless told not to. Runs in the background. */
  train: (id: number, scan = true) => post<{ started: boolean }>(`/api/people/${id}/recognition/train`, { scan }),
  scan: (id: number) => post<{ started: boolean }>(`/api/people/${id}/recognition/scan`),
  conflicts: (id: number, cursor?: string | null, limit = 60) =>
    get<RecognitionConflictPage>(`/api/people/${id}/recognition/conflicts${qs({ cursor, limit })}`),
  elsewhere: (id: number) => get<{ items: RecognitionElsewhere[] }>(`/api/people/${id}/recognition/elsewhere`),
  /** "Leave": these faces on other people are not this person. Remembered; they leave the list. */
  leave: (id: number, faceIds: number[]) => post<{ updated: number }>(`/api/people/${id}/recognition/leave`, { face_ids: faceIds }),
  withdraw: (id: number) => post<{ updated: number }>(`/api/people/${id}/recognition/withdraw`),
  forget: (id: number) => del<OkOut>(`/api/people/${id}/recognition`),
}
