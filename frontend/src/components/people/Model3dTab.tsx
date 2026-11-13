// A person's "3D" tab (docs/ARCHITECTURE.md section 36): a 3D head for each look found in their photos.
// Before a model exists it says what it does and shows the options and the "Generate" button; nothing is
// generated before that button is used. While the server works it shows the step and "Stop". With a model it
// shows the head (Head3dViewer, a lazy chunk with all the three.js code) and the sliders that pick the look,
// turn the head and move the light. Under the head it says what the owner is looking at: which photo the head
// was made from, how far round that photo reaches, and what the generator noted about it. A model from before
// heads carried their photo says so above the head, with the way to make it again.
//
// The status is read again each time the tab is shown, and when a head or a look's picture does not load: the
// server stops listing a look once its photo is in the trash or the vault or its face left the person, and that
// look then drops out here instead of staying on screen from the cache.
import { Suspense, lazy, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Box, CircleStop, RefreshCw, RotateCcw, Trash2 } from 'lucide-react'
import { ApiError } from '../../lib/api'
import { formatCount, formatDate, formatRelative } from '../../lib/format'
import {
  HEAD_LIGHT,
  HEAD_LIMITS,
  HEAD_VIEW,
  MODEL3D_DEFAULTS,
  MODEL3D_LIMITS,
  MODEL3D_POLL_MS,
  beforePhotoTextures,
  completeOptions,
  firstLook,
  lookAxes,
  lookAxisText,
  model3dApi,
  model3dKey,
  pickLook,
  stopOf,
  type Head3dHandle,
  type HeadLight,
  type HeadView,
  type Look3d,
  type Model3dOptions,
  type Model3dProgress,
  type Model3dStatus,
} from '../../lib/model3d'
import { useToast } from '../Toast'
import { errorMessage } from '../admin/adminHooks'
import { Button, ConfirmModal, ErrorState, PageSpinner, Slider, Spinner, Switch } from '../ui'

const Head3dViewer = lazy(() => import('./Head3dViewer'))

/** How often to look again while someone else's model is being generated. */
const BUSY_POLL_MS = 5000
/** Files of one model that fail to load within this time ask for the status once. */
const LOAD_ERROR_ASK_MS = 1000
const DETAIL_WORDS = ['coarse', 'normal', 'fine']
const SMOOTHING_WORDS = ['off', 'light', 'normal', 'strong']
const TEXTURE_WORDS = ['off', 'standard', 'fine', 'full']

function startError(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.code === 'busy') return 'Another model is being generated. Try again when it is done.'
    if (e.code === 'generator_missing') return 'The 3D generator is not installed on the server.'
    if (e.code === 'not_enough_photos') return 'There are too few clear, large face photos of this person.'
  }
  return errorMessage(e, 'Could not start.')
}

function degreesText(value: number, zero: string, half: string, positive: string, negative: string): string {
  const d = Math.round(value)
  if (d === 0) return zero
  if (Math.abs(d) >= 180) return half
  return `${Math.abs(d)}° ${d > 0 ? positive : negative}`
}
const percentText = (value: number) => `${Math.round(value * 100)}%`

/** A note of the generator as a sentence of its own: a capital first, a full stop last. */
function sentence(text: string | null | undefined): string | null {
  const t = text?.trim()
  if (!t) return null
  const s = t[0].toUpperCase() + t.slice(1)
  return /[.!?]$/.test(s) ? s : `${s}.`
}

/** "About 33" and "short hair, a beard": a look's label in two lines. */
function lookLines(look: Look3d): [string, string] {
  const first = look.age !== null ? `About ${Math.round(look.age)}` : look.age_band
  return [first, [look.hair_label, look.facial_hair_label].filter(Boolean).join(', ')]
}

function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={clsx('rounded-2xl border border-line bg-surface p-4', className)}>{children}</section>
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className="min-w-0">
      <legend className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-soft">{title}</legend>
      {children}
    </fieldset>
  )
}

/** The choices of a generation, as sliders. */
function OptionSliders({
  value,
  onChange,
  disabled,
  reliefInstalled,
}: {
  value: Model3dOptions
  onChange: (next: Model3dOptions) => void
  disabled?: boolean
  /** False when the server says the depth models of Face relief are not installed. */
  reliefInstalled?: boolean | null
}) {
  return (
    <div>
      <div className="grid gap-x-6 gap-y-1 sm:grid-cols-3">
        <Slider
          label="Looks"
          {...MODEL3D_LIMITS.max_looks}
          value={value.max_looks}
          onChange={(max_looks) => onChange({ ...value, max_looks })}
          valueText={`at most ${value.max_looks}`}
          disabled={disabled}
        />
        <Slider
          label="Detail"
          {...MODEL3D_LIMITS.detail}
          value={value.detail}
          onChange={(detail) => onChange({ ...value, detail })}
          valueText={DETAIL_WORDS[value.detail - 1] ?? String(value.detail)}
          ticks={[1, 2, 3]}
          disabled={disabled}
        />
        <Slider
          label="Framing"
          {...MODEL3D_LIMITS.framing}
          value={value.framing}
          onChange={(framing) => onChange({ ...value, framing: Math.round(framing * 10) / 10 })}
          valueText={`${value.framing.toFixed(1)} times the face`}
          disabled={disabled}
        />
        <Slider
          label="Smoothing"
          {...MODEL3D_LIMITS.smoothing}
          value={value.smoothing}
          onChange={(smoothing) => onChange({ ...value, smoothing })}
          valueText={SMOOTHING_WORDS[value.smoothing] ?? String(value.smoothing)}
          ticks={[0, 1, 2, 3]}
          disabled={disabled}
        />
        <Slider
          label="Photo texture"
          {...MODEL3D_LIMITS.texture}
          value={value.texture}
          onChange={(texture) => onChange({ ...value, texture })}
          valueText={TEXTURE_WORDS[value.texture] ?? String(value.texture)}
          ticks={[0, 1, 2, 3]}
          disabled={disabled}
        />
        {/* in the grid its label lines up with the sliders' labels; in one column it is a row like any switch */}
        <div className="flex items-center justify-between gap-3 sm:items-start">
          <span className={clsx('text-sm font-medium', disabled ? 'text-ink-soft' : 'text-ink')}>Face relief</span>
          <Switch
            checked={value.relief > 0}
            onChange={(on) => onChange({ ...value, relief: on ? 1 : 0 })}
            label="Face relief"
            disabled={disabled}
          />
        </div>
      </div>
      <p className="mt-1 text-xs text-ink-soft">
        More looks, finer detail and a larger texture take longer and make larger files. Framing is how much of the photo
        around the face is used: a wider frame keeps more of the hair and the shoulders. Smoothing evens out the surface.
        Face relief uses a depth model made for faces to shape the nose, lips and chin, when it is installed on the
        server.
      </p>
      {reliefInstalled === false && value.relief > 0 && (
        <p className="mt-1 text-xs text-ink-soft">
          The depth model is not installed on this server, so heads are made without Face relief. Running
          scripts/setup_3d.sh on the server once more fetches it (0.9 GB).
        </p>
      )}
      <p className="mt-1 text-xs text-ink-soft">
        {value.texture > 0
          ? "The photo is laid over the front of the head, and the sides and back, which one photo does not show, are filled in. The texture is the photo's own pixels and is never enlarged: full keeps them all, standard and fine shrink a large photo."
          : 'With Photo texture off the head has plain colours only, without the photo laid over its front.'}
      </p>
    </div>
  )
}

function Progress({ progress }: { progress: Model3dProgress }) {
  const share = progress.total > 0 ? Math.min(1, progress.done / progress.total) : null
  return (
    <div
      role="progressbar"
      aria-label={progress.step}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={share === null ? undefined : Math.round(share * 100)}
      className="relative h-1.5 w-full overflow-hidden rounded-full bg-ink/8"
    >
      {share === null ? (
        <div className="h-full w-full animate-pulse rounded-full bg-ink-faint/50" />
      ) : (
        <div className="h-full rounded-full bg-accent transition-[width] duration-200" style={{ width: `${share * 100}%` }} />
      )}
    </div>
  )
}

/** Why nothing can be started right now, in plain words; null when it can. */
function blocked(status: Model3dStatus): string | null {
  if (!status.available) return status.reason ?? 'A model cannot be generated right now.'
  if (status.busy) return "Another person's model is being generated now. One is made at a time."
  return null
}

function ModelView({ looks, onLoadError, checking }: { looks: Look3d[]; onLoadError: () => void; checking: boolean }) {
  const viewer = useRef<Head3dHandle>(null)
  const [lookKey, setLookKey] = useState<string | null>(null)
  const [view, setView] = useState<HeadView>(HEAD_VIEW)
  const [light, setLight] = useState<HeadLight>(HEAD_LIGHT)
  const [colour, setColour] = useState(true)
  const [photo, setPhoto] = useState(true)
  const [spin, setSpin] = useState(false)
  const [can3d, setCan3d] = useState(true)
  const axes = useMemo(() => lookAxes(looks), [looks])
  const active = looks.find((l) => l.key === lookKey) ?? firstLook(looks)
  if (!active) return null

  /** A slider moved: the sliders show it at once and the viewer is told. */
  const change = (part: Partial<HeadView>) => {
    setView((v) => ({ ...v, ...part }))
    viewer.current?.setView(part)
  }
  const reset = () => {
    setSpin(false)
    setColour(true)
    setPhoto(true)
    setLight(HEAD_LIGHT)
    change(HEAD_VIEW)
  }
  const taken = formatDate(active.taken_at)
  const note = sentence(active.note)

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
      <div className="min-w-0">
        <div className="mx-auto w-full max-w-xl">
          <Suspense
            fallback={
              <div className="flex aspect-square w-full items-center justify-center rounded-card bg-surface-2">
                <Spinner label="Loading the 3D viewer" />
              </div>
            }
          >
            <Head3dViewer
              ref={viewer}
              looks={looks}
              lookKey={active.key}
              initialView={view}
              onView={setView}
              light={light}
              colour={colour}
              photo={photo}
              spin={spin}
              onSupport={setCan3d}
              onLoadError={onLoadError}
              checking={checking}
            />
          </Suspense>
          <p className="mt-3 text-sm text-ink-soft">
            <span className="font-medium text-ink">{active.label}.</span> Made from one photo{taken && ` taken on ${taken}`};{' '}
            {formatCount(active.photos, 'photo')} {active.photos === 1 ? 'shows' : 'show'} this look.{' '}
            <Link to={`/m/${encodeURIComponent(active.item_uid)}`} className="font-medium text-accent hover:underline">
              Open the photo
            </Link>
          </p>
          {active.textured && (
            <p className="mt-1 text-sm text-ink-soft">
              From the front this is the photo; past about 50 degrees the sides and back are filled in.
            </p>
          )}
          {active.relief && (
            <p className="mt-1 text-sm text-ink-soft">The nose, lips and chin were shaped from the photo by a depth model.</p>
          )}
          {note && <p className="mt-1 text-xs text-ink-soft">{note}</p>}
        </div>

        <div role="group" aria-label="Looks" className="-mx-1 mt-4 flex gap-2 overflow-x-auto px-1 pb-2 pt-1">
          {looks.map((look) => {
            const [first, rest] = lookLines(look)
            const on = look.key === active.key
            return (
              <button
                key={look.key}
                type="button"
                aria-pressed={on}
                aria-label={look.label}
                title={look.label}
                onClick={() => setLookKey(look.key)}
                className={clsx(
                  'w-24 shrink-0 rounded-xl p-1 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
                  on ? 'bg-accent-soft ring-2 ring-accent' : 'hover:bg-surface-2',
                )}
              >
                <img
                  src={look.picture_url}
                  alt=""
                  loading="lazy"
                  onError={onLoadError}
                  className="aspect-square w-full rounded-lg bg-surface-2 object-cover"
                />
                <span className="mt-1 block truncate text-xs font-medium text-ink">{first}</span>
                <span className="block truncate text-xs text-ink-soft">{rest || '\u00a0'}</span>
              </button>
            )
          })}
        </div>
      </div>

      <div className="flex min-w-0 flex-col gap-5">
        <Group title="Look">
          {axes
            .filter((axis) => axis.stops.length > 0)
            .map((axis) => {
              const many = axis.stops.length > 1
              return (
                <Slider
                  key={axis.key}
                  label={axis.label}
                  min={0}
                  max={many ? axis.stops.length - 1 : 2}
                  value={many ? Math.max(0, stopOf(axis, active.key)) : 1}
                  onChange={(stop) => setLookKey(pickLook(axes, active.key, axis.key, stop))}
                  valueText={lookAxisText(axis, active)}
                  ticks={many ? axis.stops.map((_, i) => i) : undefined}
                  fill={false}
                  disabled={!many}
                />
              )
            })}
        </Group>

        {can3d && (
          <>
            <Group title="View">
              <Slider
                label="Turn"
                {...HEAD_LIMITS.turn}
                value={Math.round(view.turn)}
                onChange={(turn) => {
                  setSpin(false)
                  change({ turn })
                }}
                valueText={degreesText(view.turn, 'facing you', 'from behind', 'to the right', 'to the left')}
                ticks={[0]}
                fill={false}
              />
              <Slider
                label="Tilt"
                {...HEAD_LIMITS.tilt}
                value={Math.round(view.tilt)}
                onChange={(tilt) => change({ tilt })}
                valueText={degreesText(view.tilt, 'level', 'level', 'from above', 'from below')}
                ticks={[0]}
                fill={false}
              />
              <Slider
                label="Zoom"
                {...HEAD_LIMITS.zoom}
                value={Math.round(view.zoom * 20) / 20}
                onChange={(zoom) => change({ zoom })}
                valueText={percentText(view.zoom)}
              />
            </Group>

            <Group title="Light">
              <Slider
                label="Light direction"
                {...HEAD_LIMITS.direction}
                value={light.direction}
                onChange={(direction) => setLight((l) => ({ ...l, direction }))}
                valueText={degreesText(light.direction, 'from the front', 'from behind', 'to the right', 'to the left')}
                ticks={[0]}
                fill={false}
              />
              <Slider
                label="Light height"
                {...HEAD_LIMITS.height}
                value={light.height}
                onChange={(height) => setLight((l) => ({ ...l, height }))}
                valueText={degreesText(light.height, 'level', 'level', 'above', 'below')}
                ticks={[0]}
                fill={false}
              />
              <Slider
                label="Brightness"
                {...HEAD_LIMITS.brightness}
                value={light.brightness}
                onChange={(brightness) => setLight((l) => ({ ...l, brightness }))}
                valueText={percentText(light.brightness)}
              />
            </Group>

            <div>
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-medium">Colour</span>
                <Switch checked={colour} onChange={setColour} label="Colour" />
              </div>
              {active.textured && (
                <div className="flex items-center justify-between gap-3">
                  <span className={clsx('text-sm font-medium', !colour && 'text-ink-soft')}>Photo</span>
                  {/* without colours there is no photo either: the switch waits, and comes back as it was */}
                  <Switch checked={colour && photo} onChange={setPhoto} label="Photo" disabled={!colour} />
                </div>
              )}
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-medium">Spin slowly</span>
                <Switch checked={spin} onChange={setSpin} label="Spin slowly" />
              </div>
              <p className="mt-1 text-xs text-ink-soft">
                Drag the head to turn it; scroll or pinch to zoom. With Colour off you see the bare shape.
                {active.textured && ' With Photo off you see the colours under the photo.'}
              </p>
              <Button variant="secondary" size="sm" className="mt-3" onClick={reset}>
                <RotateCcw className="size-4" aria-hidden />
                Reset view
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default function Model3dTab({ id, name }: { id: number; name: string | null }) {
  const qc = useQueryClient()
  const toast = useToast()
  const key = model3dKey(id)
  const q = useQuery({
    queryKey: key,
    queryFn: ({ signal }) => model3dApi.status(id, signal),
    // not refreshed by library changes (model3dKey), so every showing of the tab reads it again; what is cached
    // shows meanwhile
    refetchOnMount: 'always',
    refetchInterval: (query) => {
      const data = query.state.data
      return data?.state === 'running' ? MODEL3D_POLL_MS : data?.busy ? BUSY_POLL_MS : false
    },
  })
  const [options, setOptions] = useState<Model3dOptions>(MODEL3D_DEFAULTS)
  const [again, setAgain] = useState(false)
  const [deleteOpen, setDeleteOpen] = useState(false)
  // The "Generate again" panel sits under the head and its sliders. Opened from the line above the head, it is
  // brought into view and takes the focus, so the next key press is in it.
  const panel = useRef<HTMLDivElement>(null)
  const toPanel = useRef(false)
  const showPanel = () => {
    panel.current?.focus({ preventScroll: true })
    panel.current?.scrollIntoView({ block: 'nearest' })
  }
  useEffect(() => {
    if (!again || !toPanel.current) return
    toPanel.current = false
    showPanel()
  }, [again])
  const refetch = () => qc.invalidateQueries({ queryKey: key })
  // A head or a picture that does not load may belong to a look the server stopped listing: read the status again,
  // so that look drops out. Files failing together (every picture of a model deleted elsewhere) ask once.
  const [checking, setChecking] = useState(false)
  const askedAt = useRef(0)
  const onLoadError = () => {
    const now = Date.now()
    if (now - askedAt.current < LOAD_ERROR_ASK_MS) return
    askedAt.current = now
    setChecking(true)
    void refetch().finally(() => setChecking(false))
  }

  const generate = useMutation({
    mutationFn: () => model3dApi.generate(id, options),
    onSuccess: () => {
      setAgain(false)
      // show the job at once; the next read brings the real step
      qc.setQueryData<Model3dStatus>(key, (old) =>
        old && { ...old, state: 'running', error: null, progress: old.progress ?? { step: 'Starting', done: 0, total: 0 } },
      )
      void refetch()
    },
    onError: (e) => {
      toast.error(startError(e))
      void refetch()
    },
  })
  const stop = useMutation({
    mutationFn: () => model3dApi.stop(id),
    onSuccess: () => {
      // the job ends within about a second; until the next read says so, the card says that it is stopping
      qc.setQueryData<Model3dStatus>(key, (old) => (old && old.state === 'running' ? { ...old, stopping: true } : old))
      void refetch()
    },
    onError: (e) => toast.error(errorMessage(e, 'Could not stop.')),
  })
  const remove = useMutation({
    mutationFn: () => model3dApi.remove(id),
    onSuccess: () => {
      setDeleteOpen(false)
      setAgain(false)
      // the server stops listing and serving the model at once; a job that was running is on its way out
      qc.setQueryData<Model3dStatus>(key, (old) =>
        old && { ...old, looks: [], generated_at: null, options: null, error: null, stopping: old.state === 'running' },
      )
      toast.success('The 3D model is deleted.')
      void refetch()
    },
    onError: (e) => {
      setDeleteOpen(false)
      toast.error(errorMessage(e, 'Could not delete.'))
    },
  })

  if (q.isPending) return <PageSpinner />
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />
  const status = q.data
  const running = status.state === 'running'
  const hasModel = status.looks.length > 0
  const why = blocked(status)
  const whose = name ? `${name}'s` : "this person's"

  const failure = status.error && !running && (
    <p role="alert" className="rounded-xl border border-warn/40 bg-warn/10 px-3 py-2 text-sm text-ink">
      About the last run: {status.error}
    </p>
  )
  /** Open the panel with the options as this model was made; what a model from before an option existed does not
   *  say starts at its default. */
  const openAgain = (jump: boolean) => {
    if (again) return showPanel()
    setOptions(completeOptions(status.options))
    toPanel.current = jump
    setAgain(true)
  }
  const generateButton = (
    <Button onClick={() => generate.mutate()} loading={generate.isPending} disabled={why !== null}>
      <Box className="size-4" aria-hidden />
      Generate
    </Button>
  )

  return (
    <div className="flex flex-col gap-5">
      {running && status.progress && (
        <Card>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="min-w-0">
              <h2 className="text-sm font-semibold">
                {status.stopping ? 'Stopping' : status.progress.step}
                {!status.stopping && status.progress.total > 0 && (
                  <span className="font-normal text-ink-soft tabular-nums">
                    , {status.progress.done.toLocaleString()} of {status.progress.total.toLocaleString()}
                  </span>
                )}
              </h2>
              <p className="mt-0.5 text-xs text-ink-soft">
                {status.stopping
                  ? 'The work is being stopped. That takes a moment.'
                  : `The server is working. You can leave this page; the work goes on${
                      hasModel ? ', and the model below stays until the new one is ready.' : '.'
                    }`}
              </p>
            </div>
            {!status.stopping && (
              <Button variant="secondary" size="sm" onClick={() => stop.mutate()} loading={stop.isPending}>
                <CircleStop className="size-4" aria-hidden />
                Stop
              </Button>
            )}
          </div>
          <div className="mt-3">
            <Progress progress={status.progress} />
          </div>
        </Card>
      )}

      {!hasModel && !running && (
        <Card className="flex flex-col gap-4">
          <div>
            <h2 className="text-base font-semibold">A 3D head for each look</h2>
            <p className="mt-1 text-sm text-ink-soft">
              Builds a 3D head for each look found in {whose} photos: ages, hair and facial hair. It runs on this server and
              takes a few minutes. Each head is made from one photo, so it is the model's reading of that photo and not
              a scan: the back of the head is a guess.
            </p>
            <p className="mt-2 text-sm text-ink-soft">
              {formatCount(status.usable_photos, 'face photo')} {status.usable_photos === 1 ? 'is' : 'are'} clear and large
              enough to choose from.
            </p>
          </div>
          {failure}
          {why !== null && <p className="rounded-xl bg-surface-2 px-3 py-2 text-sm text-ink">{why}</p>}
          <OptionSliders value={options} onChange={setOptions} disabled={!status.available} reliefInstalled={status.relief_installed} />
          <div>{generateButton}</div>
        </Card>
      )}

      {hasModel && (
        <>
          {failure}
          {!running && beforePhotoTextures(status) && (
            <p className="rounded-xl bg-surface-2 px-3 py-2 text-sm text-ink">
              These heads were made before photo textures.{' '}
              <button type="button" className="font-medium text-accent hover:underline" onClick={() => openAgain(true)}>
                Generate again
              </button>{' '}
              to lay the photo over the face.
            </p>
          )}
          <ModelView key={status.generated_at ?? ''} looks={status.looks} onLoadError={onLoadError} checking={checking} />
          <Card className="flex flex-col gap-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm text-ink-soft">
                {formatCount(status.looks.length, 'look')}
                {status.generated_at && `, made ${formatRelative(status.generated_at)}`}. The heads are kept on this server only.
              </p>
              <div className="flex flex-wrap gap-2">
                {!running && (
                  <Button
                    variant="secondary"
                    size="sm"
                    aria-expanded={again}
                    onClick={() => (again ? setAgain(false) : openAgain(false))}
                  >
                    <RefreshCw className="size-4" aria-hidden />
                    Generate again
                  </Button>
                )}
                <Button variant="danger" size="sm" onClick={() => setDeleteOpen(true)}>
                  <Trash2 className="size-4" aria-hidden />
                  Delete this model
                </Button>
              </div>
            </div>
            {again && !running && (
              <div
                ref={panel}
                tabIndex={-1}
                role="group"
                aria-label="Generate again"
                // the margins keep it clear of the bar at the top, and on a phone of the one at the bottom
                className="flex scroll-mt-20 scroll-mb-24 flex-col gap-4 border-t border-line pt-4 outline-none"
              >
                <p className="text-sm text-ink-soft">
                  The photos are read again and new heads are built. The new model takes this one's place when it is ready.
                </p>
                {why !== null && <p className="rounded-xl bg-surface-2 px-3 py-2 text-sm text-ink">{why}</p>}
                <OptionSliders value={options} onChange={setOptions} disabled={!status.available} reliefInstalled={status.relief_installed} />
                <div className="flex flex-wrap gap-2">
                  {generateButton}
                  <Button variant="ghost" onClick={() => setAgain(false)}>
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </Card>
        </>
      )}

      <ConfirmModal
        open={deleteOpen}
        title="Delete this 3D model?"
        body={
          running
            ? 'The heads are removed from the server and the work that is running stops. The photos are not touched, and the model can be generated again.'
            : 'The heads are removed from the server. The photos are not touched, and the model can be generated again.'
        }
        confirmLabel="Delete"
        danger
        loading={remove.isPending}
        onConfirm={() => remove.mutate()}
        onClose={() => setDeleteOpen(false)}
      />
    </div>
  )
}
