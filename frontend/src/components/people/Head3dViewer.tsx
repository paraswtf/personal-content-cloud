// The 3D picture of a person's head (docs/ARCHITECTURE.md section 36). All three.js code is in this file, which
// is a lazy chunk of its own: the 3D tab loads it only when a model exists.
//
// HeadStage is the drawing: one WebGL canvas, the looks' meshes (fetched when first wanted, a few kept), a lamp
// that follows the viewer and a soft light from above. It draws only when something changed. Going from one
// look to another is a cross-fade of two whole pictures: the old look is drawn and copied to a plain canvas on
// top, the new look is drawn underneath, and the copy fades out. Both pictures are opaque, so nothing shows
// through anything else and no triangle has to be sorted.
//
// A head whose file does not load is reported (onLoadError): its look may be gone on the server, and the tab then
// reads the model's status again before "Try again" is offered.
//
// The wrapper carries data-ready, data-look and data-fade for checks.
import { useEffect, useImperativeHandle, useLayoutEffect, useRef, useState, type Ref } from 'react'
import {
  Box3,
  Color,
  DirectionalLight,
  DoubleSide,
  HemisphereLight,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  Scene,
  Vector3,
  WebGLRenderer,
  type BufferGeometry,
  type Material,
  type Object3D,
} from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { useStableCallback } from '../../lib/hooks'
import { HEAD_LIMITS, type Head3dHandle, type HeadLight, type HeadView, type Look3d } from '../../lib/model3d'
import { useTheme } from '../../lib/theme'
import { Button, Spinner } from '../ui'

const FOV = 30
/** Half of what has to fit in the picture at zoom 1. A head is scaled into a box one unit across. */
const FIT_HALF = 0.62
const FADE_MS = 300
const SPIN_DEGREES_A_SECOND = 14
/** The sliders hear about a drag or the spin at most this often. */
const REPORT_MS = 60
/** Meshes kept ready besides the ones on screen. */
const KEEP_MESHES = 4
const MAX_PIXEL_RATIO = 2
/** The lamp and the soft light, as shares of full light, at brightness 1: a surface that faces the lamp comes
 *  out close to its own colour, one turned away keeps about half of it. */
const LAMP = 0.62
const SOFT = 0.5
const CLAY = 0xcfccc6
const GROUND = 0x77736d
const BACKGROUND = 0x808080

const rad = (degrees: number) => (degrees * Math.PI) / 180
const deg = (radians: number) => (radians * 180) / Math.PI
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
/** Into -180..180, keeping 180 itself (the back of the head) at the slider's right end. */
const wrap = (degrees: number) => {
  const w = ((((degrees + 180) % 360) + 360) % 360) - 180
  return w === -180 && degrees > 0 ? 180 : w
}
const smooth = (t: number) => t * t * (3 - 2 * t)

function clampView(v: HeadView): HeadView {
  return {
    turn: wrap(v.turn),
    tilt: clamp(v.tilt, HEAD_LIMITS.tilt.min, HEAD_LIMITS.tilt.max),
    zoom: clamp(v.zoom, HEAD_LIMITS.zoom.min, HEAD_LIMITS.zoom.max),
  }
}

interface StageState {
  /** The look on screen (during a fade: the one fading out). */
  shown: string | null
  fading: boolean
  /** The wanted look is still being fetched. */
  loading: boolean
  /** The wanted look could not be fetched or read. */
  failed: boolean
}

interface Entry {
  key: string
  url: string
  state: 'loading' | 'ready' | 'error'
  mesh: Mesh | null
  hasColour: boolean
  abort: AbortController
  /** When it was last on screen (a counter): the least recent goes first. */
  used: number
}

interface StageEvents {
  onView: (view: HeadView) => void
  onState: (state: StageState) => void
  onLost: (lost: boolean) => void
}

class HeadStage {
  private readonly renderer: WebGLRenderer
  private readonly canvas: HTMLCanvasElement
  private readonly copy: HTMLCanvasElement
  private readonly copyContext: CanvasRenderingContext2D | null
  private readonly scene = new Scene()
  private readonly camera = new PerspectiveCamera(FOV, 1, 0.05, 30)
  private readonly controls: OrbitControls
  private readonly lamp = new DirectionalLight(0xffffff, 1)
  private readonly soft = new HemisphereLight(0xffffff, GROUND, 1)
  private readonly skin = new MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0, side: DoubleSide })
  private readonly clay = new MeshStandardMaterial({ color: CLAY, roughness: 0.9, metalness: 0, side: DoubleSide })
  private readonly loader = new GLTFLoader()
  private readonly observer: ResizeObserver
  private readonly entries = new Map<string, Entry>() // by mesh URL
  private urls = new Map<string, string>() // look key -> mesh URL
  private wanted: string | null = null
  private shown: string | null = null
  private fade: { from: string; to: string; start: number } | null = null
  private view: HeadView
  private colour = true
  private spin = false
  private dragging = false
  private placing = false
  private disposed = false
  private frameId = 0
  private lastFrame = 0
  private reportedAt = 0
  private reportTimer = 0
  private uses = 0
  private width = 0
  private height = 0
  private ratio = 0
  private last: StageState = { shown: null, fading: false, loading: false, failed: false }

  /** Throws when the browser cannot give a WebGL 2 context. */
  constructor(
    private readonly host: HTMLElement,
    view: HeadView,
    private readonly events: StageEvents,
  ) {
    this.view = clampView(view)
    this.canvas = document.createElement('canvas')
    this.canvas.setAttribute('aria-hidden', 'true')
    this.canvas.style.cssText = 'display:block;width:100%;height:100%;outline:none'
    this.renderer = new WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false })
    this.copy = document.createElement('canvas')
    this.copy.setAttribute('aria-hidden', 'true')
    this.copy.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;opacity:0'
    this.copyContext = this.copy.getContext('2d')
    host.append(this.canvas, this.copy)

    // The lamp hangs on the camera, so it keeps its place for the viewer while the head turns.
    this.scene.add(this.camera, this.soft)
    this.camera.add(this.lamp, this.lamp.target)

    this.controls = new OrbitControls(this.camera, this.canvas)
    this.controls.enablePan = false
    this.controls.enableDamping = false
    this.controls.rotateSpeed = 0.9
    this.controls.minPolarAngle = rad(90 - HEAD_LIMITS.tilt.max)
    this.controls.maxPolarAngle = rad(90 - HEAD_LIMITS.tilt.min)
    this.controls.addEventListener('change', this.onChange)
    this.controls.addEventListener('start', this.onStart)
    this.controls.addEventListener('end', this.onEnd)

    this.canvas.addEventListener('webglcontextlost', this.onContextLost)
    this.canvas.addEventListener('webglcontextrestored', this.onContextRestored)
    this.observer = new ResizeObserver(this.resize)
    this.observer.observe(host)
    window.addEventListener('resize', this.resize) // a zoomed page changes the pixel ratio, not the box
    this.setBackground()
    this.resize()
  }

  // ---------------------------------------------------------------- what the component sets
  setLooks(looks: Look3d[]): void {
    this.urls = new Map(looks.map((l) => [l.key, l.mesh_url]))
    const live = new Set(this.urls.values())
    for (const entry of [...this.entries.values()]) {
      if (live.has(entry.url)) continue
      // a look of an older model, or one that is no longer listed
      if (this.fade && (this.fade.from === entry.key || this.fade.to === entry.key)) this.stopFade()
      if (this.shown === entry.key) this.shown = null
      this.drop(entry)
    }
    this.advance()
    this.requestFrame()
  }

  /** Show this look. One whose file could not be fetched before is fetched again. */
  want(key: string | null): void {
    this.wanted = key
    this.retry()
  }

  /** Fetch the wanted look again after it failed. */
  retry(): void {
    const entry = this.entryOf(this.wanted)
    if (entry && entry.state === 'error') this.drop(entry)
    this.advance()
  }

  setView(change: Partial<HeadView>): void {
    this.view = clampView({ ...this.view, ...change })
    this.place()
    this.requestFrame()
  }

  setLight(light: HeadLight): void {
    const a = rad(light.direction)
    const h = rad(light.height)
    // in the camera's own space: +X is the viewer's right, +Y up, +Z towards the viewer
    this.lamp.position.set(Math.sin(a) * Math.cos(h), Math.sin(h), Math.cos(a) * Math.cos(h))
    this.lamp.intensity = Math.PI * LAMP * light.brightness
    this.soft.intensity = Math.PI * SOFT * light.brightness
    this.requestFrame()
  }

  setColour(on: boolean): void {
    this.colour = on
    for (const entry of this.entries.values()) if (entry.mesh) entry.mesh.material = this.materialFor(entry)
    this.requestFrame()
  }

  setSpin(on: boolean): void {
    const was = this.spin
    this.spin = on
    this.lastFrame = 0
    if (on) this.requestFrame()
    else if (was) this.reportNow() // the sliders get the place it stopped at
  }

  /** Take the background from the wrapper, which carries the theme's surface colour. */
  setBackground(): void {
    const colour = new Color(BACKGROUND)
    const css = getComputedStyle(this.host).backgroundColor
    if (/^rgba?\(/.test(css)) colour.setStyle(css)
    this.renderer.setClearColor(colour, 1)
    this.requestFrame()
  }

  dispose(): void {
    this.disposed = true
    cancelAnimationFrame(this.frameId)
    window.clearTimeout(this.reportTimer)
    this.observer.disconnect()
    window.removeEventListener('resize', this.resize)
    this.canvas.removeEventListener('webglcontextlost', this.onContextLost)
    this.canvas.removeEventListener('webglcontextrestored', this.onContextRestored)
    this.controls.removeEventListener('change', this.onChange)
    this.controls.removeEventListener('start', this.onStart)
    this.controls.removeEventListener('end', this.onEnd)
    this.controls.dispose()
    for (const entry of [...this.entries.values()]) this.drop(entry)
    this.skin.dispose()
    this.clay.dispose()
    this.renderer.dispose()
    this.renderer.forceContextLoss() // hand the context back now: a browser allows only a few at a time
    this.canvas.remove()
    this.copy.remove()
  }

  // ---------------------------------------------------------------- looks
  private entryOf(key: string | null): Entry | undefined {
    const url = key ? this.urls.get(key) : undefined
    return url ? this.entries.get(url) : undefined
  }

  private materialFor(entry: Entry): Material {
    return this.colour && entry.hasColour ? this.skin : this.clay
  }

  /** Move towards showing the wanted look: fetch it (also while a fade runs), then, once no fade is running,
   *  show it at once or start the fade to it. */
  private advance(): void {
    if (this.disposed) return
    const key = this.wanted
    const url = key ? this.urls.get(key) : undefined
    if (key && url && (this.fade || this.shown !== key)) {
      const entry = this.entries.get(url) ?? this.load(key, url)
      if (!this.fade && entry.state === 'ready') {
        const from = this.entryOf(this.shown)
        if (from?.mesh && this.fadeMs() > 0) this.fade = { from: from.key, to: key, start: performance.now() }
        else this.show(key)
        this.requestFrame()
      }
    }
    this.tell()
  }

  private show(key: string): void {
    this.shown = key
    const entry = this.entryOf(key)
    if (entry) entry.used = ++this.uses
    this.tidy()
  }

  private stopFade(): void {
    this.fade = null
    this.copy.style.opacity = '0'
  }

  private fadeMs(): number {
    if (!this.copyContext) return 0
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : FADE_MS
  }

  private load(key: string, url: string): Entry {
    const entry: Entry = { key, url, state: 'loading', mesh: null, hasColour: false, abort: new AbortController(), used: 0 }
    this.entries.set(url, entry)
    void (async () => {
      try {
        const response = await fetch(url, { credentials: 'same-origin', signal: entry.abort.signal })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const gltf = await this.loader.parseAsync(await response.arrayBuffer(), '')
        if (this.disposed || this.entries.get(url) !== entry) {
          release(gltf.scene)
          return
        }
        const geometry = takeGeometry(gltf.scene)
        entry.hasColour = geometry.hasAttribute('color')
        entry.mesh = this.meshOf(geometry, this.materialFor(entry))
        entry.state = 'ready'
        this.scene.add(entry.mesh)
      } catch {
        if (this.disposed || this.entries.get(url) !== entry) return
        entry.state = 'error'
      }
      this.advance()
    })()
    return entry
  }

  /** A mesh for the geometry, centred and scaled into a box one unit across, whatever the file's own size. */
  private meshOf(geometry: BufferGeometry, material: Material): Mesh {
    if (!geometry.hasAttribute('normal')) geometry.computeVertexNormals()
    geometry.computeBoundingBox()
    const box = geometry.boundingBox ?? new Box3()
    const size = box.getSize(new Vector3())
    const centre = box.getCenter(new Vector3())
    const scale = 1 / Math.max(size.x, size.y, size.z, 1e-6)
    const mesh = new Mesh(geometry, material)
    mesh.scale.setScalar(scale)
    mesh.position.copy(centre).multiplyScalar(-scale)
    mesh.visible = false
    return mesh
  }

  private drop(entry: Entry): void {
    entry.abort.abort()
    if (entry.mesh) {
      this.scene.remove(entry.mesh)
      entry.mesh.geometry.dispose()
      entry.mesh = null
    }
    if (this.entries.get(entry.url) === entry) this.entries.delete(entry.url)
  }

  /** Let go of the meshes that were on screen longest ago. */
  private tidy(): void {
    const busy = new Set([this.shown, this.wanted, this.fade?.from, this.fade?.to])
    const idle = [...this.entries.values()].filter((e) => e.state === 'ready' && !busy.has(e.key)).sort((a, b) => a.used - b.used)
    for (const entry of idle.slice(0, Math.max(0, idle.length - KEEP_MESHES))) this.drop(entry)
  }

  private tell(): void {
    const wanted = this.entryOf(this.wanted)
    const next: StageState = {
      shown: this.shown,
      fading: this.fade !== null,
      loading: wanted?.state === 'loading',
      failed: wanted?.state === 'error',
    }
    const was = this.last
    if (next.shown === was.shown && next.fading === was.fading && next.loading === was.loading && next.failed === was.failed) return
    this.last = next
    this.events.onState(next)
  }

  // ---------------------------------------------------------------- the view
  /** How far the camera stands at zoom 1: the head fits the picture's smaller side. */
  private base(): number {
    return FIT_HALF / Math.tan(rad(FOV / 2)) / Math.min(1, this.camera.aspect)
  }

  /** Put the camera where the view says. */
  private place(): void {
    const around = -rad(this.view.turn) // the camera goes left when the head is to turn right
    const up = rad(this.view.tilt)
    const far = this.base() / this.view.zoom
    this.camera.position.set(far * Math.sin(around) * Math.cos(up), far * Math.sin(up), far * Math.cos(around) * Math.cos(up))
    this.placing = true
    this.controls.update()
    this.placing = false
  }

  /** The view the camera is at, after the owner dragged or zoomed (hundredths of a degree, thousandths of zoom). */
  private read(): HeadView {
    return clampView({
      turn: Math.round(-deg(this.controls.getAzimuthalAngle()) * 100) / 100,
      tilt: Math.round((90 - deg(this.controls.getPolarAngle())) * 100) / 100,
      zoom: Math.round((this.base() / this.controls.getDistance()) * 1000) / 1000,
    })
  }

  private readonly onChange = (): void => {
    if (!this.placing) {
      this.view = this.read()
      this.report()
    }
    this.requestFrame()
  }

  private readonly onStart = (): void => {
    this.dragging = true
  }

  private readonly onEnd = (): void => {
    this.dragging = false
    this.lastFrame = 0
    if (this.spin) this.requestFrame()
  }

  /** Tell the sliders where the view is, not more often than they can use. The last place always gets told. */
  private report(): void {
    const wait = REPORT_MS - (performance.now() - this.reportedAt)
    if (wait <= 0) this.reportNow()
    else if (!this.reportTimer) this.reportTimer = window.setTimeout(() => this.reportNow(), wait)
  }

  private reportNow(): void {
    window.clearTimeout(this.reportTimer)
    this.reportTimer = 0
    if (this.disposed) return
    this.reportedAt = performance.now()
    this.events.onView({ ...this.view })
  }

  // ---------------------------------------------------------------- drawing
  private readonly resize = (): void => {
    if (this.disposed) return
    const width = this.host.clientWidth
    const height = this.host.clientHeight
    const ratio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO)
    if (width < 2 || height < 2 || (width === this.width && height === this.height && ratio === this.ratio)) return
    this.width = width
    this.height = height
    this.ratio = ratio
    this.renderer.setPixelRatio(ratio)
    this.renderer.setSize(width, height, false)
    this.copy.width = this.canvas.width
    this.copy.height = this.canvas.height
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.controls.minDistance = this.base() / HEAD_LIMITS.zoom.max
    this.controls.maxDistance = this.base() / HEAD_LIMITS.zoom.min
    this.place()
    this.draw(performance.now()) // now, not next frame: a resized canvas is blank until it is drawn
  }

  private requestFrame(): void {
    if (!this.frameId && !this.disposed) this.frameId = requestAnimationFrame(this.frame)
  }

  private readonly frame = (now: number): void => {
    this.frameId = 0
    if (this.disposed) return
    const turning = this.spin && !this.dragging
    if (turning) {
      const seconds = this.lastFrame && now - this.lastFrame < 100 ? (now - this.lastFrame) / 1000 : 1 / 60
      this.view = { ...this.view, turn: wrap(this.view.turn + SPIN_DEGREES_A_SECOND * seconds) }
      this.place()
      this.report()
    }
    this.lastFrame = now
    this.draw(now)
    if (this.fade || turning) this.requestFrame()
  }

  private draw(now: number): void {
    if (this.fade) {
      const t = (now - this.fade.start) / FADE_MS
      if (t < 1) {
        // the old look, copied on top; then the new look underneath; the copy fades out
        this.only(this.fade.from)
        this.renderer.render(this.scene, this.camera)
        this.copyContext?.drawImage(this.canvas, 0, 0, this.copy.width, this.copy.height)
        this.copy.style.opacity = String(1 - smooth(Math.max(0, t)))
        this.only(this.fade.to)
        this.renderer.render(this.scene, this.camera)
        return
      }
      const to = this.fade.to
      this.stopFade()
      this.show(to)
      this.advance() // the owner may have moved on to another look meanwhile
    }
    this.only(this.shown)
    this.renderer.render(this.scene, this.camera)
  }

  private only(key: string | null): void {
    for (const entry of this.entries.values()) if (entry.mesh) entry.mesh.visible = entry.key === key
  }

  private readonly onContextLost = (event: Event): void => {
    event.preventDefault() // so the browser may give the context back
    this.events.onLost(true)
  }

  private readonly onContextRestored = (): void => {
    this.events.onLost(false)
    this.requestFrame()
  }
}

/** The first mesh's geometry of a loaded file; its other parts are let go. */
function takeGeometry(root: Object3D): BufferGeometry {
  let found: Mesh | null = null
  root.traverse((node) => {
    if (found === null && (node as Mesh).isMesh && (node as Mesh).geometry.hasAttribute('position')) found = node as Mesh
  })
  const mesh = found as Mesh | null
  if (mesh === null) {
    release(root)
    throw new Error('no mesh in the file')
  }
  const geometry = mesh.geometry
  release(root, geometry)
  return geometry
}

function release(root: Object3D, keep?: BufferGeometry): void {
  root.traverse((node) => {
    const mesh = node as Mesh
    if (!mesh.isMesh) return
    if (mesh.geometry !== keep) mesh.geometry.dispose()
    for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) material.dispose()
  })
}

export default function Head3dViewer({
  ref,
  looks,
  lookKey,
  initialView,
  onView,
  light,
  colour,
  spin,
  onSupport,
  onLoadError,
  checking = false,
}: {
  ref?: Ref<Head3dHandle>
  looks: Look3d[]
  /** The look to show. */
  lookKey: string
  /** Where the view starts. Later changes come through the handle's setView. */
  initialView: HeadView
  /** The owner turned, tilted or zoomed the head by hand, or it is spinning. */
  onView: (view: HeadView) => void
  light: HeadLight
  /** Off shows the bare shape. */
  colour: boolean
  spin: boolean
  /** Told once: whether this browser can draw 3D at all. */
  onSupport?: (ok: boolean) => void
  /** The shown look's head (or, without WebGL, its picture) did not load. The server may have stopped listing that
   *  look (its photo went to the trash or the vault, or the face left the person): the model's status is worth
   *  reading again. */
  onLoadError?: () => void
  /** That reading is going on: a head that failed waits for the answer before "Try again" is offered, because a
   *  look that is gone drops out of `looks` instead. */
  checking?: boolean
}) {
  const host = useRef<HTMLDivElement>(null)
  const stage = useRef<HeadStage | null>(null)
  const startView = useRef(initialView)
  const [supported, setSupported] = useState(true)
  const [lost, setLost] = useState(false)
  const [state, setState] = useState<StageState>({ shown: null, fading: false, loading: false, failed: false })
  const { resolved } = useTheme()
  const tellView = useStableCallback((view: HeadView) => {
    startView.current = view // a stage made again (React runs effects twice in development) starts where this one was
    onView(view)
  })
  const tellSupport = useStableCallback((ok: boolean) => onSupport?.(ok))
  const tellLoadError = useStableCallback(() => onLoadError?.())
  const active = looks.find((l) => l.key === lookKey) ?? looks[0]

  useEffect(() => {
    const el = host.current
    if (!el) return
    let made: HeadStage
    try {
      made = new HeadStage(el, startView.current, { onView: tellView, onState: setState, onLost: setLost })
    } catch {
      setSupported(false)
      tellSupport(false)
      return
    }
    stage.current = made
    tellSupport(true)
    return () => {
      stage.current = null
      made.dispose()
    }
  }, [tellView, tellSupport])

  // These run after the stage is made (effects run in order), and again whenever their value changes.
  useEffect(() => {
    stage.current?.setLooks(looks)
  }, [looks])
  useEffect(() => {
    stage.current?.want(active?.key ?? null)
  }, [active?.key])
  useEffect(() => {
    stage.current?.setLight({ direction: light.direction, height: light.height, brightness: light.brightness })
  }, [light.direction, light.height, light.brightness])
  useEffect(() => {
    stage.current?.setColour(colour)
  }, [colour])
  useEffect(() => {
    stage.current?.setSpin(spin)
  }, [spin])
  useEffect(() => {
    stage.current?.setBackground()
  }, [resolved])
  // Each time the wanted head fails (the first time, and again after "Try again"). Before the paint, so the
  // parent's `checking` is set by the time the failure would show.
  useLayoutEffect(() => {
    if (state.failed) tellLoadError()
  }, [state.failed, tellLoadError])

  useImperativeHandle(
    ref,
    () => ({
      setView: (change) => {
        startView.current = { ...startView.current, ...change }
        stage.current?.setView(change)
      },
    }),
    [],
  )

  if (!supported) {
    return (
      <div className="rounded-card border border-line bg-surface p-4" data-ready="false" data-look={active?.key ?? ''} data-fade="false">
        <p className="text-sm text-ink-soft">
          This browser cannot draw 3D here (WebGL is turned off or not available), so the head cannot be shown. This is the
          photo it was made from.
        </p>
        {active && (
          <img
            src={active.picture_url}
            alt={active.label}
            onError={tellLoadError}
            className="mt-3 aspect-square w-full rounded-card bg-surface-2 object-cover"
          />
        )}
      </div>
    )
  }

  const ready = !!active && state.shown === active.key && !state.fading && !lost
  return (
    <div className="relative">
      <div
        ref={host}
        role="img"
        aria-label={active ? `3D head: ${active.label}` : '3D head'}
        data-ready={ready ? 'true' : 'false'}
        data-look={state.shown ?? ''}
        data-fade={state.fading ? 'true' : 'false'}
        className="relative aspect-square w-full cursor-grab overflow-hidden rounded-card bg-surface-2 active:cursor-grabbing"
      />
      {lost ? (
        <p className="absolute inset-0 flex items-center justify-center rounded-card bg-surface-2 p-6 text-center text-sm text-ink-soft">
          The 3D picture stopped because the browser took its graphics memory back. It returns by itself; if it does not,
          reload the page.
        </p>
      ) : state.failed && checking ? (
        <div className="absolute inset-0 flex items-center justify-center rounded-card bg-surface-2/90">
          <Spinner label="Checking the model" />
        </div>
      ) : state.failed ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 rounded-card bg-surface-2/90 p-6 text-center">
          <p className="text-sm text-ink-soft">This head could not be loaded.</p>
          <Button variant="secondary" size="sm" onClick={() => stage.current?.retry()}>
            Try again
          </Button>
        </div>
      ) : (
        state.loading && (
          <div className={state.shown ? 'absolute right-3 top-3' : 'absolute inset-0 flex items-center justify-center'}>
            <Spinner label="Loading the head" />
          </div>
        )
      )}
    </div>
  )
}
