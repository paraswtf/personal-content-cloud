// Recognition trained on one person (docs/ARCHITECTURE.md section 10), on that person's page. The owner says the
// faces now on NAME are right; "Train and look again" fits a model on them (and on the faces taken off NAME as
// wrong), checks it on days it was not trained on, then looks at every other face and adds what looks like NAME as
// guesses (basis 'face'). Nothing is tagged until the owner confirms. The run is in the background: the card polls
// its progress. Two lists help the owner clean up: faces on NAME that look the same as faces they took off NAME
// (conflicts), and faces the app put on other people that look like NAME. Each is settled with the usual face
// assignment, so every answer is an owner action with its undo.
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ScanFace } from 'lucide-react'
import { ApiError, api } from '../../lib/api'
import { followedTail, formatCount, formatDate, personName } from '../../lib/format'
import { useRefreshLibrary } from '../../lib/hooks'
import { recognitionApi, type RecognitionConflict, type RecognitionElsewhere, type RecognitionOut } from '../../lib/recognition'
import { Confirm } from '../admin/AdminModal'
import { errorMessage } from '../admin/adminHooks'
import { useToast } from '../Toast'
import { Button, Spinner } from '../ui'

const POLL_MS = 2000
const BUSY_POLL_MS = 6000

function startError(e: unknown, fallback: string): string {
  if (e instanceof ApiError && e.code === 'busy') return "Another person's training is running. Try again when it is done."
  if (e instanceof ApiError && e.code === 'not_trained') return 'Train it first.'
  return errorMessage(e, fallback)
}

/** What a run is doing, in words. */
function stepText(s: RecognitionOut): string {
  const p = s.progress
  if (s.running === 'scan') {
    if (p?.step === 'write') return 'Writing the guesses.'
    return p && p.total > 0
      ? `Looking at other faces: ${p.done.toLocaleString()} of ${p.total.toLocaleString()}.`
      : 'Looking at other faces.'
  }
  if (p?.step === 'check') return 'Checking it on days it was not trained on.'
  if (p?.step === 'save') return 'Saving.'
  if (p?.step === 'fit') return `Training: step ${Math.min(p.done + 1, p.total)} of ${p.total}.`
  return 'Reading the faces.'
}

function Bar({ value, label }: { value: number | null; label: string }) {
  const pct = value === null ? null : Math.max(0, Math.min(100, Math.round(value * 100)))
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct ?? undefined}
      className="relative mt-2 h-1.5 w-full overflow-hidden rounded-full bg-ink/10"
    >
      {pct === null ? (
        <div className="h-full w-full animate-pulse rounded-full bg-ink-faint/50" />
      ) : (
        <div className="h-full rounded-full bg-accent transition-[width] duration-200" style={{ width: `${pct}%` }} />
      )}
    </div>
  )
}

/** The check in a sentence, with the counts as they are (no rounding to a rate). */
function checkText(s: RecognitionOut, name: string): string | null {
  const cut = s.check?.cuts[0]
  if (!s.check || !cut) return null
  let text =
    `On days it was not trained on, it found ${cut.found.toLocaleString()} of ${name}'s ${s.check.positives.toLocaleString()} faces ` +
    `and mistook ${cut.mistaken.toLocaleString()} of ${s.check.negatives.toLocaleString()} faces of other people for ${name}.`
  if (s.check.unclear_positives > 0) {
    text += ` Of the ${s.check.unclear_positives.toLocaleString()} blurred, small or turned ones it found ${cut.unclear_found.toLocaleString()}.`
  }
  return text
}

function Crop({ src, uid, label }: { src: string; uid: string; label: string }) {
  return (
    <Link
      to={`/m/${uid}`}
      aria-label={label}
      className="block size-16 shrink-0 overflow-hidden rounded-lg bg-surface-2 ring-1 ring-line hover:ring-ink-faint"
    >
      <img src={src} alt="" loading="lazy" className="size-full object-cover" />
    </Link>
  )
}

export default function RecognitionCard({ id, name, faceCount }: { id: number; name: string; faceCount: number }) {
  const qc = useQueryClient()
  const toast = useToast()
  const refresh = useRefreshLibrary()
  const key = ['person', id, 'recognition']
  const status = useQuery({
    queryKey: key,
    queryFn: () => recognitionApi.status(id),
    // while another person's run is going the buttons wait for it: look now and then for its end
    refetchInterval: (q) => (q.state.data?.running ? POLL_MS : q.state.data?.busy ? BUSY_POLL_MS : false),
  })
  const s = status.data
  const [show, setShow] = useState<'conflicts' | 'elsewhere' | null>(null)
  const [confirm, setConfirm] = useState<'withdraw' | 'forget' | null>(null)

  // a run that ends: say what it did, and let the Guesses tab and the counts catch up
  const wasRunning = useRef<'train' | 'scan' | null>(null)
  useEffect(() => {
    if (!s) return
    if (wasRunning.current && !s.running) {
      const waiting = formatCount(s.guesses, 'guess', 'guesses')
      if (s.error) toast.error(s.error)
      else if (wasRunning.current === 'train') toast.success(`Recognition trained on ${name}: ${waiting} to review.`)
      else toast.success(`Looked again for ${name}: ${waiting} to review.`)
      void refresh()
      wasRunning.current = null
    }
    // a run goes from training to looking: it stays the run that was started
    if (s.running && !wasRunning.current) wasRunning.current = s.running
  }, [s, name, toast, refresh])

  const started = (kind: 'train' | 'scan') => (r: { started: boolean }) => {
    if (!r.started) toast.info('A training is already running.')
    else wasRunning.current = kind
    void qc.invalidateQueries({ queryKey: key })
  }
  const train = useMutation({
    mutationFn: () => recognitionApi.train(id),
    onSuccess: started('train'),
    onError: (e) => toast.error(startError(e, 'Could not start the training.')),
  })
  const look = useMutation({
    mutationFn: () => recognitionApi.scan(id),
    onSuccess: started('scan'),
    onError: (e) => toast.error(startError(e, 'Could not start looking.')),
  })
  const withdraw = useMutation({
    mutationFn: () => recognitionApi.withdraw(id),
    onSuccess: (r) => {
      setConfirm(null)
      toast.success(`Took back ${formatCount(r.updated, 'guess', 'guesses')} of the recognition trained on ${name}.`)
      void refresh()
    },
    onError: (e) => toast.error(startError(e, 'Could not take the guesses back.')),
  })
  const forget = useMutation({
    mutationFn: () => recognitionApi.forget(id),
    onSuccess: () => {
      setConfirm(null)
      setShow(null)
      toast.success(`Forgot the recognition trained on ${name}.`)
      void refresh()
    },
    onError: (e) => toast.error(startError(e, 'Could not forget it.')),
  })

  if (!s) return null
  if (!s.trained && !s.available && !s.running && !s.error) return null // too few faces, or learning is off
  const busy = !!s.running || train.isPending || look.isPending || s.busy
  const progress = s.progress && s.progress.total > 0 ? s.progress.done / s.progress.total : null
  const check = checkText(s, name)

  return (
    <section aria-label={`Recognition trained on ${name}`} className="mt-4 rounded-2xl bg-surface-2 p-3">
      <p className="flex items-center gap-1.5 text-sm font-semibold text-ink">
        <ScanFace className="size-4" aria-hidden />
        {s.trained ? `Recognition trained on ${name}` : `Train recognition on ${name} alone`}
      </p>

      {s.running ? (
        <div role="status" className="mt-1">
          <p className="flex items-center gap-2 text-xs text-ink-soft">
            <Spinner className="[&_svg]:size-3.5" label="Working" />
            {stepText(s)}
          </p>
          <Bar value={progress} label={s.running === 'scan' ? 'Looking at other faces' : 'Training'} />
        </div>
      ) : !s.trained ? (
        <>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-soft">
            It takes the {formatCount(faceCount, 'face')} now on {name} as right and the faces you took off {name} as wrong, so
            take wrong faces off first. Then it looks again at every other face and adds what looks like {name} as guesses.
            Nothing is tagged until you confirm.
          </p>
          {s.error && <p className="mt-1 text-xs text-danger">{s.error}</p>}
          {!s.available && s.reason && <p className="mt-1 text-xs text-ink-soft">{s.reason}</p>}
          {s.busy && <p className="mt-1 text-xs text-ink-soft">Another person's training is running. This one can start when it is done.</p>}
          <Button className="mt-3" variant="secondary" disabled={!s.available || busy} loading={train.isPending} onClick={() => train.mutate()}>
            Train and look again
          </Button>
        </>
      ) : (
        <>
          <p className="mt-0.5 max-w-3xl text-xs text-ink-soft">
            Trained {formatDate(s.trained_at, true)} on {formatCount(s.counts.positives ?? 0, 'face')} of {name}, against{' '}
            {formatCount(s.counts.negatives ?? 0, 'face')} known not to be {name}. {check}
          </p>
          {s.scan && (
            <p className="mt-1 max-w-3xl text-xs text-ink-soft">
              Last look, {formatDate(s.scan.at, true)}: {formatCount(s.scan.looked, 'other face')},{' '}
              {s.guesses > 0 ? (
                <Link to="?tab=guesses" className="font-medium text-accent-strong hover:underline">
                  {formatCount(s.guesses, 'guess', 'guesses')} waiting for you
                </Link>
              ) : (
                'no guess waiting'
              )}
              {s.scan.guesses === 0 ? '. It found no face that looks like ' + name + ' and is not tagged yet.' : '.'}
            </p>
          )}
          {s.withdrawn && (
            <p className="mt-1 text-xs text-ink-soft">Its guesses were taken back. Look again to bring them back.</p>
          )}
          {!s.available && s.reason && <p className="mt-1 text-xs text-ink-soft">{s.reason} The trained recognition is not used meanwhile.</p>}
          {s.error && <p className="mt-1 text-xs text-danger">{s.error}</p>}
          {s.busy && <p className="mt-1 text-xs text-ink-soft">Another person's training is running. This one can run when it is done.</p>}
          {s.answers_since > 0 && (
            <p className="mt-1 text-xs text-ink-soft">
              {formatCount(s.answers_since, 'answer')} about {name} since the training. Train again to use them.
            </p>
          )}
          <div className="mt-3 flex flex-wrap gap-2">
            <Button variant="secondary" disabled={!s.available || busy} loading={look.isPending} onClick={() => look.mutate()}>
              Look again
            </Button>
            <Button variant="secondary" disabled={!s.available || busy} loading={train.isPending} onClick={() => train.mutate()}>
              Train again
            </Button>
            {s.guesses > 0 && (
              <Button variant="ghost" disabled={busy} onClick={() => setConfirm('withdraw')}>
                Take back its guesses
              </Button>
            )}
            <Button variant="ghost" disabled={busy} onClick={() => setConfirm('forget')}>
              Forget
            </Button>
          </div>

          {s.conflicts > 0 && (
            <div className="mt-3 border-t border-line pt-3">
              <p className="text-xs text-ink">
                {formatCount(s.conflicts, 'face')} on {name} {s.conflicts === 1 ? 'looks' : 'look'} the same as faces you took off {name}.
                Settle {s.conflicts === 1 ? 'it' : 'these'} and train again.
              </p>
              <Button className="mt-2" size="sm" variant="secondary" aria-expanded={show === 'conflicts'} onClick={() => setShow(show === 'conflicts' ? null : 'conflicts')}>
                {show === 'conflicts' ? 'Hide them' : 'Show them'}
              </Button>
              {show === 'conflicts' && <Conflicts id={id} name={name} />}
            </div>
          )}
          {s.elsewhere > 0 && (
            <div className="mt-3 border-t border-line pt-3">
              <p className="text-xs text-ink">
                {formatCount(s.elsewhere, 'face')} the app put on other people {s.elsewhere === 1 ? 'looks' : 'look'} like {name}.
              </p>
              <Button className="mt-2" size="sm" variant="secondary" aria-expanded={show === 'elsewhere'} onClick={() => setShow(show === 'elsewhere' ? null : 'elsewhere')}>
                {show === 'elsewhere' ? 'Hide them' : 'Show them'}
              </Button>
              {show === 'elsewhere' && <Elsewhere id={id} name={name} />}
            </div>
          )}
        </>
      )}

      <Confirm
        open={confirm === 'withdraw'}
        title={`Take back the guesses for ${name}?`}
        body={`The ${formatCount(s.guesses, 'guess', 'guesses')} this recognition made go away, and it makes no new ones until you look again. Faces you already confirmed stay.`}
        confirmLabel="Take them back"
        loading={withdraw.isPending}
        onConfirm={() => withdraw.mutate()}
        onClose={() => setConfirm(null)}
      />
      <Confirm
        open={confirm === 'forget'}
        title={`Forget the recognition trained on ${name}?`}
        body="Its open guesses go away and faces are matched as before the training. The faces and your answers stay, so you can train again at any time."
        confirmLabel="Forget"
        danger
        loading={forget.isPending}
        onConfirm={() => forget.mutate()}
        onClose={() => setConfirm(null)}
      />
    </section>
  )
}

/** One face answered through the usual assignment: to the person, to someone else, or to nobody. */
function useSettle(done: string) {
  const toast = useToast()
  const refresh = useRefreshLibrary()
  return useMutation({
    mutationFn: (v: { face: number; person: number | null; text: string }) =>
      api.people.assignFaces({ face_ids: [v.face], person_id: v.person }),
    onSuccess: (r, v) => {
      toast.success(`${v.text}${followedTail(r.followed ?? 0)}`)
      void refresh()
    },
    onError: (e) => toast.error(errorMessage(e, done)),
  })
}

function Conflicts({ id, name }: { id: number; name: string }) {
  const q = useQuery({ queryKey: ['person', id, 'recognition', 'conflicts'], queryFn: () => recognitionApi.conflicts(id) })
  const settle = useSettle('Could not save the answer.')
  if (q.isPending) return <p className="mt-2 text-xs text-ink-soft">Loading.</p>
  if (q.isError) return <p className="mt-2 text-xs text-danger">{errorMessage(q.error, 'Could not load them.')}</p>
  const items: RecognitionConflict[] = q.data.items
  return (
    <>
      <ul className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3" aria-label={`Faces on ${name} to settle`}>
        {items.map((c) => {
          const other = c.like_person
          const busy = settle.isPending && settle.variables?.face === c.face.id
          return (
            <li key={c.face.id} className="rounded-xl border border-line bg-surface p-2">
              <div className="flex items-center gap-2">
                <Crop src={c.face.url} uid={c.face.item_uid} label={`Open the photo of the face on ${name}`} />
                <span className="text-xs text-ink-soft">looks like</span>
                {c.like ? (
                  <Crop src={c.like.url} uid={c.like.item_uid} label="Open the photo of the face it looks like" />
                ) : (
                  <span className="text-xs text-ink-soft">a face that cannot be shown</span>
                )}
                <span className="min-w-0 text-xs text-ink-soft">
                  {other ? `now on ${personName(other)}` : 'which you took off'}
                </span>
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                <Button size="sm" variant="secondary" disabled={busy} onClick={() => settle.mutate({ face: c.face.id, person: id, text: `Kept on ${name}.` })}>
                  This is {name}
                </Button>
                <Button size="sm" variant="secondary" disabled={busy} onClick={() => settle.mutate({ face: c.face.id, person: null, text: `Taken off ${name}.` })}>
                  Not {name}
                </Button>
                {other && (
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={busy}
                    onClick={() => settle.mutate({ face: c.face.id, person: other.id, text: `Moved to ${personName(other)}.` })}
                  >
                    Move to {personName(other)}
                  </Button>
                )}
              </div>
            </li>
          )
        })}
      </ul>
      {q.data.total > items.length && (
        <p className="mt-2 text-xs text-ink-soft">
          Showing {items.length} of {q.data.total.toLocaleString()}. The rest shows as you settle these.
        </p>
      )}
    </>
  )
}

function Elsewhere({ id, name }: { id: number; name: string }) {
  const q = useQuery({ queryKey: ['person', id, 'recognition', 'elsewhere'], queryFn: () => recognitionApi.elsewhere(id) })
  const settle = useSettle('Could not move the face.')
  const [left, setLeft] = useState<Set<number>>(() => new Set())
  if (q.isPending) return <p className="mt-2 text-xs text-ink-soft">Loading.</p>
  if (q.isError) return <p className="mt-2 text-xs text-danger">{errorMessage(q.error, 'Could not load them.')}</p>
  const items: RecognitionElsewhere[] = q.data.items.filter((e) => !left.has(e.face.id))
  if (!items.length) return <p className="mt-2 text-xs text-ink-soft">Nothing left here.</p>
  return (
    <ul className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3" aria-label={`Faces on other people that look like ${name}`}>
      {items.map((e) => {
        const busy = settle.isPending && settle.variables?.face === e.face.id
        return (
          <li key={e.face.id} className="rounded-xl border border-line bg-surface p-2">
            <div className="flex items-center gap-2">
              <Crop src={e.face.url} uid={e.face.item_uid} label="Open the photo" />
              <span className="min-w-0 text-xs text-ink-soft">
                on {personName(e.person)}, {Math.round(e.p * 100)}% sure it is {name}
              </span>
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              <Button size="sm" variant="secondary" disabled={busy} onClick={() => settle.mutate({ face: e.face.id, person: id, text: `Moved to ${name}.` })}>
                This is {name}
              </Button>
              <Button size="sm" variant="ghost" disabled={busy} onClick={() => setLeft((prev) => new Set(prev).add(e.face.id))}>
                Leave
              </Button>
            </div>
          </li>
        )
      })}
    </ul>
  )
}
