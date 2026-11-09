// "Who is it?" for a person box the owner drew (section 10): pick someone or type a new name, and the box goes to
// the server with the photo's (or the paused frame's) coordinates. A helmet, a back or someone far away works too:
// the box tags them, and when the detector finds a face inside it, that face is tagged as well. Placed like the
// face editor (next to the drawn box on wide screens, a bottom sheet on phones) and rendered into the viewer, not
// the stage: the stage's size containment would pin a fixed card to the stage instead of the screen.
// The same card names a face box the owner drew around a face the detector missed (kind 'face'): the detector
// must find the face inside it (422 no_face otherwise), and a stored face there is tagged instead.
import { useEffect, useLayoutEffect, useState, type CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { useMutation } from '@tanstack/react-query'
import { ScanFace, SquareUser, X } from 'lucide-react'
import { ApiError, api } from '../../lib/api'
import type { FaceOut, PersonBoxOut } from '../../lib/types'
import { personName } from '../../lib/format'
import { errorMessage } from '../admin/adminHooks'
import PersonPicker, { type PersonChoice } from '../people/PersonPicker'
import { Button } from '../ui'
import { placement } from './FaceEditor'
import { useInvalidateItem } from './shared'

/** A drawn box, normalised to the picture. */
export interface DrawnBox {
  x: number
  y: number
  w: number
  h: number
}

interface Props {
  uid: string
  box: DrawnBox
  /** Videos: the time of the paused frame the box was drawn on. */
  t?: number
  /** The drawn box on the stage: the card sits next to it. */
  anchor: HTMLElement
  /** The box was added; the message says what happened (the stage leaves draw mode and shows it). */
  onAdded: (message: string) => void
  onRedraw: () => void
  onCancel: () => void
  /** A person box (the default) or a face box. */
  kind?: 'person' | 'face'
}

/** What went wrong, in words: the server's codes for this call, else its message. */
function failure(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.code === 'frame_time_needed') return 'Pause the video on a frame, then draw the box again.'
    if (e.code === 'person_needed') return 'Pick someone, or type a name.'
    if (e.code === 'no_face') return 'No face found in that box. Draw it closer around the face, or add a person box when the face does not show.'
    if (e.code === 'models_unavailable') return 'The face model is not loaded right now. Try again in a while.'
    if (e.status === 409) return 'The photo or frame could not be opened.'
    if (e.status === 422) return 'Draw a larger box, inside the picture.'
  }
  return errorMessage(e, 'Could not add the box')
}

function faceAddedMessage(r: FaceOut, choice: PersonChoice, video: boolean): string {
  const who = r.person ? personName(r.person) : 'personId' in choice ? 'them' : choice.newName
  return video ? `Tagged ${who} with a face box on this frame.` : `Tagged ${who} with a face box.`
}

function addedMessage(r: PersonBoxOut, choice: PersonChoice, video: boolean): string {
  const who = r.body.person ? personName(r.body.person) : 'personId' in choice ? 'them' : choice.newName
  if (r.face) return `Tagged ${who}. Their face in the box is tagged too.`
  return video ? `Tagged ${who} on this frame. The box shows as a chip under the video.` : `Tagged ${who} with a person box.`
}

export default function PersonBoxCard({ uid, box, t, anchor, onAdded, onRedraw, onCancel, kind = 'person' }: Props) {
  const face = kind === 'face'
  const [style, setStyle] = useState<CSSProperties>(() => placement(anchor))
  const sheet = style.bottom === 0 && style.left === 0
  const invalidate = useInvalidateItem(uid)
  const video = t !== undefined

  useLayoutEffect(() => {
    setStyle(placement(anchor))
  }, [anchor])
  useEffect(() => {
    const onResize = () => setStyle(placement(anchor))
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [anchor])

  const add = useMutation({
    mutationFn: async (choice: PersonChoice): Promise<string> => {
      const body = {
        ...box,
        ...(video ? { t } : {}),
        ...('personId' in choice ? { person_id: choice.personId } : { new_person_name: choice.newName }),
      }
      return face
        ? faceAddedMessage(await api.items.addFaceBox(uid, body), choice, video)
        : addedMessage(await api.items.addPersonBox(uid, body), choice, video)
    },
    onSuccess: (message) => {
      invalidate()
      onAdded(message)
    },
  })

  const root = anchor.closest<HTMLElement>('[aria-modal="true"]') ?? document.body
  return createPortal(
    <div
      data-no-swipe
      role="dialog"
      aria-label={face ? 'Whose face is in the box?' : 'Who is in the box?'}
      style={style}
      className={
        'fixed z-50 flex max-h-[85dvh] flex-col overflow-y-auto bg-surface p-3 text-ink shadow-2xl ' +
        (sheet ? 'rounded-t-2xl pb-[max(0.75rem,env(safe-area-inset-bottom))]' : 'rounded-2xl')
      }
    >
      <div className="flex items-center gap-3">
        <span className="flex size-12 shrink-0 items-center justify-center rounded-full bg-surface-2 text-ink-soft ring-1 ring-line" aria-hidden>
          {face ? <ScanFace className="size-6" /> : <SquareUser className="size-6" />}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-xs text-ink-faint">
            {face ? (video ? 'Face box on this frame' : 'Face box') : video ? 'Person box on this frame' : 'Person box'}
          </p>
          <p className="truncate text-sm font-semibold">Who is it?</p>
        </div>
        <button
          type="button"
          onClick={onCancel}
          disabled={add.isPending}
          className="flex size-10 shrink-0 items-center justify-center rounded-full text-ink-soft hover:bg-surface-2 disabled:opacity-50"
          aria-label={face ? 'Cancel the face box' : 'Cancel the person box'}
        >
          <X className="size-5" aria-hidden />
        </button>
      </div>
      <div className="mt-3">
        <PersonPicker
          autoFocus={!sheet}
          placeholder="Search people or type a new name"
          onPick={(choice) => {
            if (!add.isPending) add.mutate(choice)
          }}
        />
      </div>
      <p className="mt-2 text-xs text-ink-soft">
        {face
          ? 'Draw it around the face: the face has to show. For a helmet, a back or someone far away, add a person box instead.'
          : 'A helmet, a back or someone far away works too. A face inside the box is tagged as well.'}
      </p>
      {add.isPending && (
        <p className="mt-2 text-xs text-ink-soft" role="status">
          Adding the box
        </p>
      )}
      {add.error && (
        <p className="mt-2 text-sm text-danger" role="alert">
          {failure(add.error)}
        </p>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        <Button variant="secondary" size="sm" disabled={add.isPending} onClick={onRedraw}>
          Draw again
        </Button>
        <Button variant="ghost" size="sm" disabled={add.isPending} onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>,
    root,
  )
}
