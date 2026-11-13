// The 3D picture of a person's head (docs/ARCHITECTURE.md section 36). All three.js code is in this file, which
// is a lazy chunk of its own: the 3D tab loads it only when a model exists.
//
// HeadStage is the drawing: one WebGL canvas, the looks' meshes (fetched when first wanted, a few kept), a lamp
// that follows the viewer and a soft light from above. It draws only when something changed. Going from one
// look to another is a cross-fade of two whole pictures: the old look is drawn and copied to a plain canvas on
// top, the new look is drawn underneath, and the copy fades out. Both pictures are opaque, so nothing shows
// through anything else and no triangle has to be sorted.
//
// A head file has one or two parts. The first is the head itself, a colour at every vertex. The second, in files
// made since heads carry the photo, is that photo as a texture over the front: the same surface again, its
// vertex colours' alpha saying how much of the photo shows at each corner, so the photo fades into the colours
// underneath instead of ending at a line. It is drawn after the head, a hair in front of it, without writing
// depth; a head's picture is still one opaque picture when it is done. The layer hangs on the head's mesh, so it
// is placed, sized, shown and hidden with it; Colour off and Photo off hide it. A layer whose picture could not
// be read is left out. Each layer has a material and a texture of its own, let go with its head; of the heads
// that are not on screen, at most KEEP_PHOTOS keep one.
//
// A photo has its own light in it, so a model with such heads is lit more evenly (PHOTO_LAMP, PHOTO_SOFT,
// PHOTO_GROUND): all of its looks, also one without the photo, so the light never changes from look to look.
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
  FrontSide,
  HemisphereLight,
  LoadingManager,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  Scene,
  Vector3,
  WebGLRenderer,
  type BufferGeometry,
  type Material,
  type Object3D,
  type Texture,
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
/** A fade waits for the graphics card to have drawn its first picture, but not longer than this. */
const SENT_MS = 1500
const SPIN_DEGREES_A_SECOND = 14
/** The sliders hear about a drag or the spin at most this often. */
const REPORT_MS = 60
/** Meshes kept ready besides the ones on screen. */
const KEEP_MESHES = 4
/** ... and how many of those may carry a photo: a photo is tens of megabytes of graphics memory, a head without
 *  one a few. A head that was let go is fetched again when the owner comes back to it (the browser has the file). */
const KEEP_PHOTOS = 2
const MAX_PIXEL_RATIO = 2
/** The lamp and the soft light, as shares of full light, at brightness 1: a surface that faces the lamp comes
 *  out close to its own colour, one turned away keeps about half of it. */
const LAMP = 0.62
const SOFT = 0.5
/** The same for a model whose heads carry the photo. The photo has its own light and shade in it, so the lamp
 *  only adds a little shape and the soft light carries the rest, from below nearly as much as from above (its
 *  ground is light, where a model without photos has a dark one): with the lamp where it starts, a surface that
 *  faces the viewer comes out at the photo's own colours (0.25 * 0.71 from the lamp, 0.9 * 0.92 from the soft
 *  light), one turned away from the lamp about a sixth darker. With the dark ground the same two numbers draw
 *  the face an eighth darker than the photo. */
const PHOTO_LAMP = 0.25
const PHOTO_SOFT = 0.9
const PHOTO_GROUND = 0xebebeb
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

interface Fade {
  from: string
  to: string
  /** When the fade began to move: null until the graphics card has drawn its first picture. */
  start: number | null
  /** The mark put behind that first picture, and when. */
  mark: { sync: WebGLSync | null; at: number } | null
}

interface Entry {
  key: string
  url: string
  state: 'loading' | 'ready' | 'error'
  mesh: Mesh | null
  hasColour: boolean
  /** The photo over the front of the head, a child of `mesh`; null in a file without one. */
  photo: Mesh | null
  abort: AbortController
  /** When it was last on screen (a counter): the least recent goes first. */
  used: number
}

interface StageEvents {
  onView: (view: HeadView) => void
  onState: (state: StageState) => void
  onLost: (lost: boolean) => void
  /** A look's head was read: whether it has a photo over it that can be drawn. The server may list a look as
   *  carrying its photo when the picture inside the file cannot be decoded here; the head then shows its own
   *  colours, and whoever offers a "Photo" switch should know. */
  onPhoto?: (key: string, drawn: boolean) => void
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
  private readonly loader = new GLTFLoader(keptInside())
  private readonly observer: ResizeObserver
  private readonly entries = new Map<string, Entry>() // by mesh URL
  private urls = new Map<string, string>() // look key -> mesh URL
  private wanted: string | null = null
  private shown: string | null = null
  private fade: Fade | null = null
  private view: HeadView
  private brightness = 1
  private colour = true
  private photo = true
  /** A look of the model carries its photo: the whole model gets the even light. */
  private photoModel = false
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
    this.relight()
    this.setBackground()
    this.resize()
  }

  // ---------------------------------------------------------------- what the component sets
  setLooks(looks: Look3d[]): void {
    this.urls = new Map(looks.map((l) => [l.key, l.mesh_url]))
    this.photoModel = looks.some((l) => l.textured === true)
    this.relight()
    const live = new Set(this.urls.values())
    for (const entry of [...this.entries.values()]) {
      // a look of an older model, or one that is no longer listed
      if (!live.has(entry.url)) this.forget(entry)
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
    this.brightness = light.brightness
    this.relight()
    this.requestFrame()
  }

  /** Off shows the bare shape: no colours and no photo. */
  setColour(on: boolean): void {
    this.colour = on
    for (const entry of this.entries.values()) {
      if (entry.mesh) entry.mesh.material = this.materialFor(entry)
      if (entry.photo) entry.photo.visible = this.colour && this.photo
    }
    this.relight()
    this.requestFrame()
  }

  /** Off hides the photo over the heads at once: what shows is the head's own colours underneath. */
  setPhoto(on: boolean): void {
    this.photo = on
    for (const entry of this.entries.values()) if (entry.photo) entry.photo.visible = this.colour && this.photo
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
    this.stopFade()
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

  /** How strong the two lights are. One rule for the whole model, so the light is the same for every look and on
   *  both sides of a fade: a model with a head that carries its photo is lit evenly, one without as before. The
   *  bare shape (Colour off) has no light of its own in it and is always lit as before. */
  private relight(): void {
    const even = this.photoModel && this.colour
    this.lamp.intensity = Math.PI * (even ? PHOTO_LAMP : LAMP) * this.brightness
    this.soft.intensity = Math.PI * (even ? PHOTO_SOFT : SOFT) * this.brightness
    this.soft.groundColor.setHex(even ? PHOTO_GROUND : GROUND)
  }

  /** Let a head go that may be on screen: the picture goes with it. */
  private forget(entry: Entry): void {
    if (this.fade && (this.fade.from === entry.key || this.fade.to === entry.key)) this.stopFade()
    if (this.shown === entry.key) this.shown = null
    this.drop(entry)
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
        if (from?.mesh && this.fadeMs() > 0) this.fade = { from: from.key, to: key, start: null, mark: null }
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
    if (this.fade?.mark?.sync) this.gl().deleteSync(this.fade.mark.sync)
    this.fade = null
    this.copy.style.opacity = '0'
  }

  private fadeMs(): number {
    if (!this.copyContext) return 0
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : FADE_MS
  }

  private load(key: string, url: string): Entry {
    const entry: Entry = {
      key, url, state: 'loading', mesh: null, hasColour: false, photo: null, abort: new AbortController(), used: 0,
    }
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
        const parts = takeParts(gltf.scene)
        entry.hasColour = parts.head.hasAttribute('color')
        entry.mesh = this.meshOf(parts.head, this.materialFor(entry))
        if (parts.photo) entry.photo = this.photoOf(parts.photo.geometry, parts.photo.map, entry.mesh)
        entry.state = 'ready'
        this.scene.add(entry.mesh)
        this.events.onPhoto?.(key, entry.photo !== null)
      } catch {
        if (this.disposed || this.entries.get(url) !== entry) return
        entry.state = 'error'
      }
      this.advance()
      this.tidy() // a head the owner has moved on from meanwhile counts among the kept ones at once
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

  /** The photo over the front of a head: the head's own surface again, so it hangs on the head's mesh (the same
   *  place and size, from the head's own box, and hidden with it). Its vertex colours carry how much of the photo
   *  shows (alpha); it is blended over the head, a hair in front of it so the two never fight for a pixel, and
   *  writes no depth. The texture is the file's own, as the loader set it up (its colours, its way up). */
  private photoOf(geometry: BufferGeometry, map: Texture, head: Mesh): Mesh {
    if (!geometry.hasAttribute('normal')) geometry.computeVertexNormals()
    map.anisotropy = this.renderer.capabilities.getMaxAnisotropy() // sharp when the surface turns away
    map.needsUpdate = true
    // Once the picture is on the graphics card the decoded copy is not needed: closing it gives its memory back
    // (as much again as the texture). It cannot be put there a second time, so when the browser takes the
    // graphics memory away such a head is let go and fetched again (onContextLost).
    map.onUpdate = () => {
      map.onUpdate = null
      closePicture(map)
    }
    const material = new MeshStandardMaterial({
      map,
      vertexColors: geometry.hasAttribute('color'),
      roughness: 0.9,
      metalness: 0,
      side: FrontSide,
      transparent: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -2,
    })
    const mesh = new Mesh(geometry, material)
    mesh.renderOrder = 1
    mesh.visible = this.colour && this.photo
    head.add(mesh)
    return mesh
  }

  private drop(entry: Entry): void {
    entry.abort.abort()
    if (entry.photo) {
      const material = entry.photo.material as MeshStandardMaterial
      entry.photo.removeFromParent()
      entry.photo.geometry.dispose()
      freeTexture(material.map) // the material does not let go of its texture by itself
      material.dispose()
      entry.photo = null
    }
    if (entry.mesh) {
      this.scene.remove(entry.mesh)
      entry.mesh.geometry.dispose()
      entry.mesh = null
    }
    if (this.entries.get(entry.url) === entry) this.entries.delete(entry.url)
  }

  /** Let go of the heads that are not on screen, but for the KEEP_MESHES that were there last, at most KEEP_PHOTOS
   *  of them with a photo. Of heads that were never on screen, the ones fetched last count as the latest. */
  private tidy(): void {
    const busy = new Set([this.shown, this.wanted, this.fade?.from, this.fade?.to])
    const idle = [...this.entries.values()]
      .reverse()
      .filter((e) => e.state === 'ready' && !busy.has(e.key))
      .sort((a, b) => b.used - a.used)
    let kept = 0
    let photos = 0
    for (const entry of idle) {
      if (kept >= KEEP_MESHES || (entry.photo && photos >= KEEP_PHOTOS)) {
        this.drop(entry)
        continue
      }
      kept += 1
      if (entry.photo) photos += 1
    }
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
      const fade = this.fade
      // The fade begins to move once the graphics card has drawn its first picture. A look's shapes and its photo
      // go there when the look is first drawn, which takes a moment on a phone, and drawing does not wait for
      // it: that moment passes under the old look, still whole on top, and not as a jump in the middle of the
      // fade.
      if (fade.start === null && fade.mark && this.reached(fade.mark, now)) {
        fade.start = now
        fade.mark = null
      }
      const t = fade.start === null ? 0 : (now - fade.start) / FADE_MS
      if (t < 1) {
        // the old look, copied on top; then the new look underneath; the copy fades out
        this.only(fade.from)
        this.renderer.render(this.scene, this.camera)
        this.copyContext?.drawImage(this.canvas, 0, 0, this.copy.width, this.copy.height)
        this.copy.style.opacity = String(1 - smooth(Math.max(0, t)))
        this.only(fade.to)
        this.renderer.render(this.scene, this.camera)
        if (fade.start === null && !fade.mark) fade.mark = this.markNow(now)
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

  /** Make this look the one that is drawn (the photo over a head hangs on it and goes with it). */
  private only(key: string | null): void {
    for (const entry of this.entries.values()) if (entry.mesh) entry.mesh.visible = entry.key === key
  }

  private gl(): WebGL2RenderingContext {
    return this.renderer.getContext() as WebGL2RenderingContext // three.js draws with WebGL 2 only
  }

  /** Put a mark behind everything the graphics card was sent so far. */
  private markNow(now: number): NonNullable<Fade['mark']> {
    const gl = this.gl()
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
    gl.flush()
    return { sync, at: now }
  }

  /** Whether the graphics card has come to the mark (asked once a frame, without waiting), or will not say:
   *  no mark could be made, the context is gone, or it took too long. */
  private reached(mark: NonNullable<Fade['mark']>, now: number): boolean {
    const gl = this.gl()
    const done =
      !mark.sync || now - mark.at > SENT_MS || gl.isContextLost() || gl.getSyncParameter(mark.sync, gl.SYNC_STATUS) === gl.SIGNALED
    if (done && mark.sync) gl.deleteSync(mark.sync)
    return done
  }

  private readonly onContextLost = (event: Event): void => {
    event.preventDefault() // so the browser may give the context back
    // When it comes back, the shapes go to the graphics card again from what the page holds. A photo cannot: its
    // picture was closed once it was there (photoOf). So the heads that carry one are let go now, while nothing
    // is on the card to be taken off it, and the wanted one is fetched again when the context is back.
    for (const entry of [...this.entries.values()]) if (entry.photo) this.forget(entry)
    this.events.onLost(true)
    this.tell()
  }

  private readonly onContextRestored = (): void => {
    this.setBackground() // three.js starts over with a black one
    this.events.onLost(false)
    this.advance()
    this.requestFrame()
  }
}

/** The parts of a loaded file. The head is the first mesh without a texture (the first mesh when all have one).
 *  The photo is the first other mesh that has both a texture and the coordinates to place it: a layer whose
 *  texture could not be read is left out, and the head then shows its own colours there. The rest is let go. */
/** A head file is whole in itself: its picture sits inside it, and the loader reads that through a blob: address
 *  of its own making. So a loader for heads follows no other address. A file whose description named one (for a
 *  buffer or a picture) would otherwise have this page fetch it, with whatever the address carries, and the files
 *  come from a generator that runs other people's model code without a network of its own. The server refuses
 *  such a file as well; this is the second lock. */
function keptInside(): LoadingManager {
  const manager = new LoadingManager()
  manager.setURLModifier((url) => (/^(blob|data):/i.test(url) ? url : 'data:,'))
  return manager
}

function takeParts(root: Object3D): { head: BufferGeometry; photo: { geometry: BufferGeometry; map: Texture } | null } {
  const meshes: Mesh[] = []
  root.traverse((node) => {
    const mesh = node as Mesh
    if (mesh.isMesh && mesh.geometry.hasAttribute('position')) meshes.push(mesh)
  })
  const head = meshes.find((m) => !mapOf(m)) ?? meshes[0]
  if (!head) {
    release(root)
    throw new Error('no mesh in the file')
  }
  const layer = meshes.find((m) => m !== head && mapOf(m) && m.geometry.hasAttribute('uv'))
  const map = layer ? mapOf(layer) : null
  const photo = layer && map ? { geometry: layer.geometry, map } : null
  release(root, photo ? [head.geometry, photo.geometry] : [head.geometry], photo?.map)
  return { head: head.geometry, photo }
}

function mapOf(mesh: Mesh): Texture | null {
  const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material
  return (material as MeshStandardMaterial | undefined)?.map ?? null
}

/** Close a texture's decoded picture: it holds its pixels until then. Closing twice does nothing. */
function closePicture(map: Texture): void {
  ;(map.image as { close?: () => void } | null)?.close?.()
}

/** Let a texture go, with its picture (but not a picture that the kept texture shows too). */
function freeTexture(map: Texture | null | undefined, keep?: Texture): void {
  if (!map) return
  map.dispose()
  if (map.image !== keep?.image) closePicture(map)
}

/** Let go of everything a loaded file holds, but for the geometries and the texture that are kept. */
function release(root: Object3D, keep: BufferGeometry[] = [], keepMap?: Texture): void {
  root.traverse((node) => {
    const mesh = node as Mesh
    if (!mesh.isMesh) return
    if (!keep.includes(mesh.geometry)) mesh.geometry.dispose()
    for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      const map = (material as MeshStandardMaterial).map
      if (map && map !== keepMap) freeTexture(map, keepMap)
      material.dispose()
    }
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
  photo = true,
  spin,
  onSupport,
  onLoadError,
  onPhoto,
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
  /** Off hides the photo over a head that carries one, and shows the colours underneath. */
  photo?: boolean
  spin: boolean
  /** Told once: whether this browser can draw 3D at all. */
  onSupport?: (ok: boolean) => void
  /** The shown look's head (or, without WebGL, its picture) did not load. The server may have stopped listing that
   *  look (its photo went to the trash or the vault, or the face left the person): the model's status is worth
   *  reading again. */
  onLoadError?: () => void
  /** A look's head was read: whether a photo is drawn over it. False for a look listed with its photo means the
   *  picture in its file could not be decoded here. */
  onPhoto?: (key: string, drawn: boolean) => void
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
  const tellPhoto = useStableCallback((key: string, drawn: boolean) => onPhoto?.(key, drawn))
  const active = looks.find((l) => l.key === lookKey) ?? looks[0]

  useEffect(() => {
    const el = host.current
    if (!el) return
    let made: HeadStage
    try {
      made = new HeadStage(el, startView.current, { onView: tellView, onState: setState, onLost: setLost, onPhoto: tellPhoto })
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
  }, [tellView, tellSupport, tellPhoto])

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
    stage.current?.setPhoto(photo)
  }, [photo])
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
