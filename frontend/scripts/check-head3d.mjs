// A check of the 3D head viewer without a browser (docs/ARCHITECTURE.md section 36): the real HeadStage class from
// src/components/people/Head3dViewer.tsx, the real three.js scene objects, OrbitControls and GLTFLoader, reading
// .glb files that the Python writer (gen3d/glb.py) made. Only the WebGL renderer and the page are stand-ins, so
// this checks the logic (which look is drawn when, the cross-fade, the camera, the lamp, what is kept and let go)
// and that three.js reads the files; what the pictures look like is for a browser. It also checks the look
// sliders' logic in src/lib/model3d.ts and the value src/lib/theme.ts gives React to compare.
//
// Run by backend/tests/test_gen3d.py, which writes the files:
//   node scripts/check-head3d.mjs <glb dir> <scratch dir>
// <glb dir> holds l1.glb, l2.glb and l3.glb (the stand-in backend's heads; l1 from a picture with the skin colour
// 200, 150, 120), plain.glb (four vertices far from the origin, no colours) and broken.glb (not a .glb).
// A head has one part (a colour at every vertex) or two (the photo over its front as a second, blended part with
// a texture); every check below holds for both, and says which it met. A look is told to the stage as the server
// would tell it: `textured` when its file carries a picture. When <glb dir> also holds photo.glb, a head that is
// known to have two parts, one more group checks the photo layer itself: how it is drawn, the light of a model
// with such a head, Colour off and Photo off, what is kept and let go, and what happens when the browser takes
// the graphics memory away.
import assert from 'node:assert/strict'
import { resolveObjectURL } from 'node:buffer'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

if (process.argv.length < 4) {
  console.error('usage: node check-head3d.mjs <glb dir> <scratch dir>')
  process.exit(2)
}
const FRONT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const GLB = path.resolve(process.argv[2])
const HERE = path.resolve(process.argv[3]) // the bundle and the stand-in renderer are written here
mkdirSync(HERE, { recursive: true })
const { build } = await import(pathToFileURL(path.join(FRONT, 'node_modules/esbuild/lib/main.js')).href)

const VIEWER = path.join(FRONT, 'src/components/people/Head3dViewer.tsx')
const WRAPPER = path.join(HERE, 'three-wrapper.js')
writeFileSync(
  WRAPPER,
  `export * from ${JSON.stringify(path.join(FRONT, 'node_modules/three/build/three.module.js'))}
export class WebGLRenderer {
  constructor(options) {
    if (globalThis.__noWebGL) throw new Error('Error creating WebGL context.')
    this.options = options; this.domElement = options.canvas; this.ratio = 1; this.renders = []; this.clear = null
    this.capabilities = { getMaxAnisotropy: () => 8 }
    this.marks = [] // the marks put behind what was sent to the graphics card (a fade waits for its first picture)
    this.sent = new WeakSet() // the textures on the graphics card
    this.sentClosed = 0 // ... that were sent from a picture that was closed already: must stay 0
    this.context = 0 // counts up each time the browser gives the graphics memory back
    this.staleLetGo = 0 // textures let go after that, though they were on the card before: the browser complains
    globalThis.__renderers.push(this)
  }
  getContext() { // the card comes to a mark once it was asked more often than globalThis.__cardLag (0: at once)
    const r = this
    return {
      SYNC_GPU_COMMANDS_COMPLETE: 0x9117, SYNC_STATUS: 0x9114, SIGNALED: 0x9119,
      fenceSync() { const mark = { asked: 0, flushed: false, deleted: false }; r.marks.push(mark); return mark },
      flush() { for (const mark of r.marks) mark.flushed = true },
      isContextLost: () => false,
      getSyncParameter(mark) { mark.asked += 1; return mark.asked > (globalThis.__cardLag ?? 0) ? 0x9119 : 0x9118 },
      deleteSync(mark) { mark.deleted = true },
    }
  }
  /** The browser gave the graphics memory back: nothing is on the card any more, and three.js starts over with a
   *  black background. */
  restore() { this.sent = new WeakSet(); this.context += 1; this.clear = ['000000', 1] }
  setPixelRatio(r) { this.ratio = r }
  setSize(w, h, style) { this.size = [w, h, style]; this.domElement.width = Math.floor(w * this.ratio); this.domElement.height = Math.floor(h * this.ratio) }
  setClearColor(c, a) { this.clear = [c.getHexString(), a] }
  render(scene, camera) {
    scene.updateMatrixWorld()
    const visible = []
    const light = {}
    const walk = (o) => { // as the real renderer: what hangs on a hidden object is not drawn
      if (!o.visible) return
      if (o.isMesh) {
        visible.push(o)
        const map = o.material.map
        if (map && !this.sent.has(map)) { // as the real renderer: a texture goes to the card when it is first drawn
          this.sent.add(map)
          if (map.image?.closed) this.sentClosed += 1
          const context = this.context
          map.addEventListener('dispose', () => { if (this.context !== context) this.staleLetGo += 1 })
          map.onUpdate?.(map)
        }
      }
      if (o.isDirectionalLight) light.lamp = o.intensity
      if (o.isHemisphereLight) { light.soft = o.intensity; light.ground = o.groundColor.getHex() }
      for (const child of o.children) walk(child)
    }
    walk(scene)
    this.renders.push({ visible, light, at: globalThis.__clock(), camera: camera.position.toArray() })
  }
  dispose() { this.disposed = true }
  forceContextLoss() { this.lostOnPurpose = true }
}
`,
)

const STUBS = {
  react: 'export const useEffect=()=>{}, useLayoutEffect=()=>{}, useImperativeHandle=()=>{}, useRef=(v)=>({current:v}), useState=(v)=>[v,()=>{}], useSyncExternalStore=(s,g)=>g()',
  'react/jsx-runtime': 'export const jsx=()=>null, jsxs=()=>null, Fragment=null',
  ui: 'export const Button=()=>null, Spinner=()=>null',
  hooks: 'export const useStableCallback=(f)=>f',
  theme: "export const useTheme=()=>({resolved:'light'})",
  api: 'export const get=()=>{}, post=()=>{}, del=()=>{}',
}
const OUT = path.join(HERE, 'bundle.mjs')
await build({
  stdin: {
    contents: `export { HeadStage } from ${JSON.stringify(VIEWER)}
export * as model from ${JSON.stringify(path.join(FRONT, 'src/lib/model3d.ts'))}
export * as theme from ${JSON.stringify(path.join(FRONT, 'src/lib/theme.ts'))}
export * as THREE from 'three'`,
    resolveDir: FRONT,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: OUT,
  jsx: 'automatic',
  logLevel: 'warning',
  plugins: [
    {
      name: 'stand-ins',
      setup(b) {
        b.onLoad({ filter: /Head3dViewer\.tsx$/ }, (a) => ({ contents: readFileSync(a.path, 'utf8') + '\nexport { HeadStage }\n', loader: 'tsx' }))
        b.onResolve({ filter: /^three$/ }, () => ({ path: WRAPPER }))
        const stub = (filter, name) => b.onResolve({ filter }, () => ({ path: name, namespace: 'stub' }))
        stub(/^react$/, 'react')
        stub(/^react\/jsx-runtime$/, 'react/jsx-runtime')
        stub(/\/ui$/, 'ui')
        stub(/lib\/hooks$/, 'hooks')
        stub(/lib\/theme$/, 'theme')
        stub(/^\.\/api$/, 'api')
        b.onLoad({ filter: /.*/, namespace: 'stub' }, (a) => ({ contents: STUBS[a.path], loader: 'js' }))
      },
    },
  ],
})

// ---------------------------------------------------------------- the stand-in page
let clock = 1000
let reduced = false
let dark = false
let rafs = []
let rafId = 0
let timers = []
let timerId = 0
const drawn = []
globalThis.__clock = () => clock
globalThis.__renderers = []
Object.defineProperty(globalThis, 'performance', { value: { now: () => clock }, configurable: true })
function element(extra = {}) {
  return {
    style: {},
    heard: {},
    addEventListener(type, f) {
      ;(this.heard[type] ??= []).push(f)
    },
    removeEventListener(type, f) {
      this.heard[type] = (this.heard[type] ?? []).filter((x) => x !== f)
    },
    fire(type, event = {}) {
      for (const f of [...(this.heard[type] ?? [])]) f(event)
    },
    setAttribute() {},
    remove() {
      this.removed = true
    },
    ...extra,
  }
}
const doc = element()
const canvases = []
globalThis.document = {
  createElement() {
    const c = element({
      width: 300,
      height: 150,
      clientWidth: 600,
      clientHeight: 600,
      ownerDocument: doc,
      getRootNode: () => doc,
      setPointerCapture() {},
      releasePointerCapture() {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 600, height: 600 }),
      getContext: (type) => (type === '2d' ? { drawImage: (source, ...rest) => drawn.push({ source, rest, at: clock }) } : null),
    })
    canvases.push(c)
    return c
  },
}
const win = element({
  devicePixelRatio: 3,
  matchMedia: (query) => ({ matches: String(query).includes('prefers-color-scheme') ? dark : reduced }),
  setTimeout: (f, ms) => (timers.push({ id: ++timerId, at: clock + ms, f }), timerId),
  clearTimeout: (id) => {
    timers = timers.filter((t) => t.id !== id)
  },
})
globalThis.window = win
globalThis.getComputedStyle = () => ({ backgroundColor: dark ? 'rgb(35, 39, 43)' : 'rgb(239, 239, 235)' })
globalThis.document.documentElement = element({ removeAttribute() {} })
globalThis.document.querySelector = () => null
const observers = []
globalThis.ResizeObserver = class {
  constructor(cb) {
    this.cb = cb
    observers.push(this)
  }
  observe(target) {
    this.target = target
  }
  disconnect() {
    this.off = true
  }
}
globalThis.requestAnimationFrame = (f) => (rafs.push({ id: ++rafId, f }), rafId)
globalThis.cancelAnimationFrame = (id) => {
  rafs = rafs.filter((r) => r.id !== id)
}
// A head's photo sits inside its file. three.js hands it to the browser as a blob address, fetches that and asks
// for the decoded picture; here the "decoded picture" is its size, read from the file's own header.
globalThis.self = globalThis
const bitmaps = []
function pictureSize(bytes) {
  if (bytes[0] === 0x89 && bytes[1] === 0x50) return [bytes.readUInt32BE(16), bytes.readUInt32BE(20)] // PNG
  for (let at = 2; at + 9 < bytes.length; ) { // JPEG: walk the segments to the frame header
    if (bytes[at] !== 0xff) return [0, 0]
    const marker = bytes[at + 1]
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) return [bytes.readUInt16BE(at + 7), bytes.readUInt16BE(at + 5)]
    at += 2 + bytes.readUInt16BE(at + 2)
  }
  return [0, 0]
}
globalThis.createImageBitmap = async (blob) => {
  const [width, height] = pictureSize(Buffer.from(await blob.arrayBuffer()))
  const bitmap = { width, height, closed: false, close() { this.closed = true } }
  bitmaps.push(bitmap)
  return bitmap
}
const files = new Map()
let fetches = 0
globalThis.fetch = async (url, { signal } = {}) => {
  if (String(url).startsWith('blob:')) {
    const blob = resolveObjectURL(String(url))
    return blob ? { ok: true, status: 200, blob: async () => blob } : { ok: false, status: 404, blob: async () => { throw new Error('gone') } }
  }
  fetches += 1
  await new Promise((r) => setImmediate(r))
  if (signal?.aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' })
  const file = files.get(url)
  if (!file) return { ok: false, status: 404 }
  const buf = readFileSync(file)
  return { ok: true, status: 200, arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) }
}

const settle = async () => {
  for (let i = 0; i < 12; i++) await new Promise((r) => setImmediate(r))
}
/** Let time pass: timers that came due run, then one animation frame. */
async function step(ms = 16) {
  clock += ms
  for (const t of timers.filter((t) => t.at <= clock)) {
    timers = timers.filter((x) => x !== t)
    t.f()
  }
  const now = rafs
  rafs = []
  for (const r of now) r.f(clock)
  await settle()
}
async function steps(n, ms = 16) {
  for (let i = 0; i < n; i++) await step(ms)
}

const { HeadStage, model, theme, THREE } = await import(pathToFileURL(OUT).href)
const near = (a, b, eps = 1e-3) => assert.ok(Math.abs(a - b) <= eps, `${a} is not near ${b}`)
/** Whether a head file carries a picture: what the server says in a look's `textured`. */
function carriesPicture(file) {
  try {
    const bytes = readFileSync(file)
    if (bytes.readUInt32LE(0) !== 0x46546c67) return false
    const json = JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)).toString('utf8'))
    return (json.images?.length ?? 0) > 0
  } catch {
    return false
  }
}
const look = (key, file = key, rev = 'r1') => {
  const url = `/files/model3d/7/${rev}/${key}.glb`
  files.set(url, path.join(GLB, `${file}.glb`))
  return { key, mesh_url: url, textured: carriesPicture(path.join(GLB, `${file}.glb`)) }
}
const BASE = 0.62 / Math.tan((15 * Math.PI) / 180)
const told = { views: [], states: [], lost: [] }
const events = { onView: (v) => told.views.push(v), onState: (s) => told.states.push(s), onLost: (l) => told.lost.push(l) }
const host = element({ clientWidth: 600, clientHeight: 600, append: (...els) => (host.children = els) })
const keyOf = (stage, mesh) => [...stage.entries.values()].find((e) => e.mesh === mesh || e.photo === mesh)?.key
/** The looks drawn in one picture (a head and the photo over it are one look). */
const shownIn = (stage, render) => [...new Set(render.visible.map((m) => keyOf(stage, m)))]
const LIGHT = { plain: [0.62, 0.5, 0x77736d], photo: [0.25, 0.9, 0xebebeb] }
/** The lights a picture was drawn with, as shares of full light at this brightness. */
const litAs = (render, kind, brightness = 1) => {
  const [lamp, soft, ground] = LIGHT[kind]
  near(render.light.lamp, Math.PI * lamp * brightness)
  near(render.light.soft, Math.PI * soft * brightness)
  assert.equal(render.light.ground, ground, `the soft light of a ${kind} model`)
}
/** The looks the stage was given last, and the light they call for: one rule for the whole model. */
let model3d = []
const give = (stage, list) => {
  model3d = list
  stage.setLooks(list)
}
const kind = () => (model3d.some((l) => l.textured) ? 'photo' : 'plain')
/** What must hold for the photo over a head, whichever file it came from. */
function checkPhoto(stage, entry) {
  const photo = entry.photo
  assert.equal(photo.parent, entry.mesh, 'the photo hangs on the head: the same place and size, hidden with it')
  assert.ok(photo.geometry.hasAttribute('uv') && photo.geometry.hasAttribute('normal'))
  const weight = photo.geometry.getAttribute('color')
  assert.equal(weight.itemSize, 4, 'the photo carries its weight as the alpha of its vertex colours')
  const m = photo.material
  assert.ok(m !== stage.skin && m !== stage.clay)
  assert.ok(m.map && m.map.isTexture, 'the texture of the file')
  assert.equal(m.map.colorSpace, THREE.SRGBColorSpace)
  assert.equal(m.map.anisotropy, 8, 'as sharp at a slant as the card allows')
  assert.equal(m.map.flipY, false)
  assert.deepEqual([m.transparent, m.depthWrite, m.side, m.vertexColors], [true, false, THREE.FrontSide, true])
  assert.deepEqual([m.polygonOffset, m.polygonOffsetFactor, m.polygonOffsetUnits], [true, -1, -2], 'a hair in front of the head')
  assert.equal(photo.renderOrder, 1)
  assert.equal(photo.visible, stage.colour && stage.photo)
  assert.ok(photo.geometry.getAttribute('position').count <= entry.mesh.geometry.getAttribute('position').count)
}
const results = []
const ok = (name) => results.push(name)

// ---------------------------------------------------------------- A: making it
const stage = new HeadStage(host, { turn: 0, tilt: 0, zoom: 1 }, events)
const renderer = globalThis.__renderers[0]
const [canvas, copy] = canvases
assert.deepEqual(host.children, [canvas, copy])
assert.equal(renderer.options.antialias, true)
assert.equal(renderer.options.alpha, false)
assert.equal(renderer.ratio, 2, 'the pixel ratio is capped at 2')
assert.deepEqual(renderer.size, [600, 600, false])
assert.equal(canvas.width, 1200)
assert.equal(copy.width, 1200)
assert.equal(copy.height, 1200)
assert.deepEqual(renderer.clear, ['efefeb', 1], 'the background comes from the wrapper')
assert.equal(renderer.renders.length, 1, 'a resized canvas is drawn at once')
assert.equal(canvas.style.touchAction, 'none')
near(stage.camera.position.z, BASE)
near(stage.camera.position.x, 0)
near(stage.controls.minDistance, BASE / 2.5)
near(stage.controls.maxDistance, BASE / 0.6)
ok('A making it')

// ---------------------------------------------------------------- B: the first look
const looks = [look('l1'), look('l2'), look('l3')]
give(stage, looks)
stage.want('l1')
assert.deepEqual(told.states.at(-1), { shown: null, fading: false, loading: true, failed: false })
await settle()
assert.deepEqual(told.states.at(-1), { shown: 'l1', fading: false, loading: false, failed: false }, 'the first look shows without a fade')
await step()
assert.deepEqual(shownIn(stage, renderer.renders.at(-1)), ['l1'])
{
  const e = stage.entryOf('l1')
  const colour = e.mesh.geometry.getAttribute('color')
  assert.equal(colour.itemSize, 4)
  assert.equal(colour.normalized, true)
  assert.ok(colour.array instanceof Uint16Array, 'colours arrive as 16-bit values')
  assert.equal(e.mesh.geometry.getAttribute('position').count, 963)
  assert.equal(e.mesh.geometry.index.count, 1900 * 3)
  assert.ok(e.mesh.geometry.hasAttribute('normal'))
  assert.equal(e.hasColour, true)
  assert.equal(e.mesh.material, stage.skin)
  assert.equal(stage.skin.vertexColors, true)
  if (e.photo) checkPhoto(stage, e)
  assert.equal(e.photo !== null, looks[0].textured, 'the look says what its file holds')
  litAs(renderer.renders.at(-1), kind())
  near(e.mesh.scale.x, 1, 1e-4) // the writer's unit box
  near(e.mesh.position.length(), 0, 1e-4)
  // the face (the picture's middle colour, in linear light) is on the +Z side, towards the camera at turn 0
  const pos = e.mesh.geometry.getAttribute('position')
  let front = 0
  for (let i = 1; i < pos.count; i++) if (pos.getZ(i) > pos.getZ(front)) front = i
  const lin = (c) => ((c / 255 + 0.055) / 1.055) ** 2.4
  near(colour.getX(front), lin(200), 0.02)
  near(colour.getY(front), lin(150), 0.02)
  near(colour.getZ(front), lin(120), 0.02)
}
assert.equal(rafs.length, 0, 'nothing is drawn while nothing changes')
const parts = (key) => (stage.entryOf(key)?.photo ? 'two parts' : 'one part')
ok(`B first look (${parts('l1')}), colours linear, face towards the camera`)

// ---------------------------------------------------------------- C: a cross-fade
drawn.length = 0
let before = renderer.renders.length
let marks = renderer.marks.length
stage.want('l2')
assert.equal(told.states.at(-1).loading, true)
await settle()
assert.deepEqual(told.states.at(-1), { shown: 'l1', fading: true, loading: false, failed: false })
const opacities = []
let fadeFrames = 0
while (stage.fade && fadeFrames < 60) {
  const n = renderer.renders.length
  await step()
  if (stage.fade) {
    fadeFrames += 1
    assert.equal(renderer.renders.length, n + 2, 'a fade frame draws both looks')
    assert.deepEqual(shownIn(stage, renderer.renders.at(-2)), ['l1'])
    assert.deepEqual(shownIn(stage, renderer.renders.at(-1)), ['l2'])
    opacities.push(Number(copy.style.opacity))
  }
}
assert.ok(fadeFrames >= 19 && fadeFrames <= 21, `the fade took ${fadeFrames} frames`)
assert.equal(drawn.length, fadeFrames, 'the old look is copied once a frame')
assert.ok(drawn.every((d) => d.source === canvas && d.rest.join() === '0,0,1200,1200'))
// The first picture has the old look whole on top, and so has the next: the fade begins to move when the graphics
// card has drawn the first one, so sending the new look there is not taken from the fade.
assert.deepEqual(opacities.slice(0, 2), [1, 1])
assert.equal(renderer.marks.length, marks + 1, 'one mark a fade, behind its first picture')
assert.ok(opacities[2] > 0.98 && opacities.at(-1) < 0.05, `opacity went ${opacities[2]} to ${opacities.at(-1)}`)
assert.ok(opacities.every((o, i) => i < 2 || o < opacities[i - 1]), 'the copy only fades out')
assert.equal(copy.style.opacity, '0')
assert.deepEqual(told.states.at(-1), { shown: 'l2', fading: false, loading: false, failed: false })
assert.deepEqual(shownIn(stage, renderer.renders.at(-1)), ['l2'])
await step()
assert.equal(rafs.length, 0)
const fadeDraws = renderer.renders.length - before
{
  // a slow graphics card: the old look stays whole on top until the card has drawn the first picture
  globalThis.__cardLag = 6
  stage.want('l1')
  await settle()
  const slow = []
  for (let i = 0; i < 9; i++) {
    await step()
    slow.push(Number(copy.style.opacity))
  }
  assert.deepEqual(slow.slice(0, 8), Array(8).fill(1), 'the fade waits')
  assert.ok(slow[8] < 1 && slow[8] > 0.98, 'and then starts from its beginning')
  assert.deepEqual(shownIn(stage, renderer.renders.at(-1)), ['l1'])
  while (stage.fade) await step()
  // one that never says so: the fade runs after a second and a half all the same
  globalThis.__cardLag = Infinity
  stage.want('l2')
  await settle()
  let waited = 0
  do {
    await step()
    waited += 16
  } while (Number(copy.style.opacity) === 1 && waited < 3000)
  assert.ok(waited > 1500 && waited < 1600, `the fade began to move after ${waited} ms`)
  while (stage.fade) await step()
  globalThis.__cardLag = 0
  assert.equal(stage.shown, 'l2')
  assert.ok(renderer.marks.every((m) => m.flushed && m.deleted), 'every mark is sent and taken away again')
  await step()
  assert.equal(rafs.length, 0)
}
ok(`C cross-fade: ${fadeFrames} frames of two opaque pictures, ${fadeDraws} draws; it waits for a slow graphics card`)

// ---------------------------------------------------------------- D: changing one's mind during a fade
stage.want('l3')
await settle()
assert.equal(stage.fade.to, 'l3')
await steps(5)
stage.want('l1') // already loaded; waits for the running fade
assert.equal(stage.fade.to, 'l3', 'a running fade is finished first')
let guard = 0
while ((stage.fade || stage.shown !== 'l1') && guard++ < 80) await step()
assert.equal(stage.shown, 'l1')
const order = told.states.map((s) => `${s.shown}${s.fading ? '~' : ''}`).slice(-4)
assert.deepEqual(order, ['l2', 'l2~', 'l3~', 'l1'], order.join(' ')) // one fade follows the other without a still frame told between
// a look that is not loaded yet is fetched while the fade runs
stage.want('l2')
await settle()
const n0 = fetches
const extra = look('l4', 'l1')
give(stage, [...looks, extra])
stage.want('l4')
assert.equal(fetches, n0 + 1, 'the next look is fetched during the fade')
guard = 0
while ((stage.fade || stage.shown !== 'l4') && guard++ < 80) await step()
assert.equal(stage.shown, 'l4')
ok('D a fade finishes before the next one starts; the next look loads meanwhile')

// ---------------------------------------------------------------- E: reduced motion
reduced = true
before = renderer.renders.length
stage.want('l1')
await settle()
assert.equal(stage.fade, null)
assert.equal(stage.shown, 'l1')
await step()
assert.deepEqual(shownIn(stage, renderer.renders.at(-1)), ['l1'])
reduced = false
ok('E reduced motion: no fade')

// ---------------------------------------------------------------- F: the view
told.views.length = 0
stage.setView({ turn: 30 })
near(stage.camera.position.x, -Math.sin(Math.PI / 6) * BASE)
near(stage.camera.position.z, Math.cos(Math.PI / 6) * BASE)
near(stage.controls.getAzimuthalAngle(), -Math.PI / 6)
stage.setView({ tilt: 20 })
near(stage.camera.position.y, Math.sin((20 * Math.PI) / 180) * BASE)
near(stage.controls.getPolarAngle(), (70 * Math.PI) / 180)
stage.setView({ zoom: 2 })
near(stage.camera.position.length(), BASE / 2)
assert.deepEqual(stage.read(), stage.view)
near(stage.view.turn, 30, 1e-6)
stage.setView({ tilt: 200, zoom: 9, turn: 540 })
assert.deepEqual(stage.view, { turn: 180, tilt: 75, zoom: 2.5 })
near(stage.controls.getDistance(), BASE / 2.5)
stage.setView({ turn: 0, tilt: 0, zoom: 1 })
assert.equal(told.views.length, 0, 'a view set from outside is not reported back')
await step()
near(renderer.renders.at(-1).camera[2], BASE)
ok('F view: turn, tilt and zoom place the camera and are clamped')

// ---------------------------------------------------------------- G: dragging, the wheel
const pointer = (x, y) => ({ pointerId: 1, pointerType: 'mouse', button: 0, clientX: x, clientY: y, ctrlKey: false, metaKey: false, shiftKey: false, preventDefault() {} })
canvas.fire('pointerdown', pointer(300, 300))
assert.equal(stage.dragging, true)
doc.fire('pointermove', pointer(400, 300))
assert.ok(stage.view.turn > 50 && stage.view.turn < 58, `a drag to the right turns the head to the right: ${stage.view.turn}`)
near(stage.view.tilt, 0, 1e-6)
assert.equal(told.views.length, 1, 'the first move is reported at once')
doc.fire('pointermove', pointer(400, 360))
assert.ok(stage.view.tilt > 30 && stage.view.tilt < 35, `a drag down shows the head from above: ${stage.view.tilt}`)
assert.equal(told.views.length, 1, 'the next report waits')
doc.fire('pointerup', pointer(400, 360))
assert.equal(stage.dragging, false)
await step(80)
assert.equal(told.views.length, 2, 'the last place is always reported')
assert.deepEqual(told.views.at(-1), stage.view)
const wheel = (deltaY) => ({ deltaY, deltaMode: 0, clientX: 300, clientY: 300, ctrlKey: false, preventDefault() {} })
const z0 = stage.view.zoom
canvas.fire('wheel', wheel(-100))
assert.ok(stage.view.zoom > z0, `scrolling up zooms in: ${z0} to ${stage.view.zoom}`)
for (let i = 0; i < 40; i++) canvas.fire('wheel', wheel(-100))
near(stage.view.zoom, 2.5, 1e-6)
for (let i = 0; i < 80; i++) canvas.fire('wheel', wheel(100))
near(stage.view.zoom, 0.6, 1e-6)
await step(80)
stage.setView({ turn: 0, tilt: 0, zoom: 1 })
ok('G drag right turns right, drag down looks from above, the wheel zooms within 60% to 250%')

// ---------------------------------------------------------------- H: spinning
await step()
told.views.length = 0
stage.setSpin(true)
await steps(30)
near(stage.view.turn, 14 * 0.016 * 29 + 14 / 60, 0.02) // the first frame has no earlier one to measure from
assert.ok(told.views.length >= 7 && told.views.length <= 9, `reports while spinning: ${told.views.length}`)
canvas.fire('pointerdown', pointer(300, 300))
const held = stage.view.turn
await steps(5)
assert.equal(stage.view.turn, held, 'the spin waits while the head is held')
doc.fire('pointerup', pointer(300, 300))
await steps(5)
assert.ok(stage.view.turn > held, 'and goes on after')
stage.setSpin(false)
assert.deepEqual(told.views.at(-1), stage.view)
await step()
await step()
assert.equal(rafs.length, 0, 'no frames once the spin is off')
ok('H spin: 14 degrees a second, waits during a drag, stops drawing when off')

// ---------------------------------------------------------------- I, J: colour and light
stage.setColour(false)
assert.ok([...stage.entries.values()].every((e) => e.mesh.material === stage.clay && (!e.photo || e.photo.visible === false)))
await step()
assert.ok(renderer.renders.at(-1).visible.every((m) => m.material === stage.clay), 'with Colour off only the bare shape is drawn')
litAs(renderer.renders.at(-1), 'plain') // the bare shape is lit the same in every model
stage.setColour(true)
assert.ok([...stage.entries.values()].every((e) => e.mesh.material === stage.skin && (!e.photo || e.photo.visible === true)))
stage.setView({ turn: 0, tilt: 0, zoom: 1 })
stage.setLight({ direction: 90, height: 0, brightness: 1 })
await step()
const from = () => stage.lamp.getWorldPosition(new THREE.Vector3()).sub(stage.lamp.target.getWorldPosition(new THREE.Vector3()))
assert.deepEqual(from().toArray().map((v) => Math.round(v * 1e3) / 1e3), [1, 0, 0], 'at 90 degrees the lamp is at the viewer\'s right')
litAs(renderer.renders.at(-1), kind())
near(stage.lamp.intensity, renderer.renders.at(-1).light.lamp)
stage.setView({ turn: 90 })
await step()
assert.deepEqual(from().toArray().map((v) => Math.round(v * 1e3) / 1e3 + 0), [0, 0, 1], 'and stays there for the viewer when the head turns')
stage.setLight({ direction: 0, height: 90, brightness: 1.5 })
await step()
near(from().y, 1)
litAs(renderer.renders.at(-1), kind(), 1.5)
stage.setView({ turn: 0 })
ok(`I colour switch; J the lamp follows the viewer (${kind()} light)`)

// ---------------------------------------------------------------- K, L: odd files
reduced = true
const plain = look('plain')
const broken = look('broken')
const missing = { key: 'gone', mesh_url: '/files/model3d/7/r1/gone.glb' }
give(stage, [...looks, plain, broken, missing])
stage.want('plain')
await settle()
{
  const e = stage.entryOf('plain')
  assert.equal(stage.shown, 'plain')
  assert.equal(e.hasColour, false)
  assert.equal(e.mesh.material, stage.clay, 'a head without colours is drawn as the bare shape')
  near(e.mesh.scale.x, 1 / 8)
  e.mesh.updateMatrixWorld()
  const box = new THREE.Box3().setFromObject(e.mesh)
  near(box.getCenter(new THREE.Vector3()).length(), 0, 1e-5)
  near(box.getSize(new THREE.Vector3()).y, 1, 1e-5)
}
for (const bad of ['broken', 'gone']) {
  stage.want(bad)
  await settle()
  assert.deepEqual(told.states.at(-1), { shown: 'plain', fading: false, loading: false, failed: true }, bad)
  const f0 = fetches
  stage.retry()
  assert.equal(fetches, f0 + 1)
  assert.equal(told.states.at(-1).loading, true)
  await settle()
  assert.equal(told.states.at(-1).failed, true)
}
stage.want('l1')
await settle()
assert.deepEqual(told.states.at(-1), { shown: 'l1', fading: false, loading: false, failed: false })
ok('K a file of any size and place is framed; L a file that cannot be read says so and can be tried again')

// ---------------------------------------------------------------- Q: a head that carries its photo
if (existsSync(path.join(GLB, 'photo.glb'))) {
  const last = () => renderer.renders.at(-1)
  const p1 = look('p1', 'photo')
  assert.equal(p1.textured, true, 'photo.glb is expected to carry a picture')
  stage.setLight({ direction: 30, height: 35, brightness: 1 })
  // one rule for the whole model: with a look that carries its photo, every look is lit evenly, also one without
  give(stage, [plain])
  stage.want('plain')
  await settle()
  await step()
  assert.deepEqual(shownIn(stage, last()), ['plain'])
  litAs(last(), 'plain')
  give(stage, [plain, p1])
  await step()
  assert.deepEqual(shownIn(stage, last()), ['plain'])
  litAs(last(), 'photo')
  // a fade to the head with the photo: two whole pictures under the same light
  reduced = false
  marks = renderer.marks.length
  stage.want('p1')
  await settle()
  assert.equal(stage.fade?.to, 'p1')
  let frames = 0
  while (stage.fade && frames < 60) {
    await step()
    if (!stage.fade) break
    frames += 1
    const [old, fresh] = renderer.renders.slice(-2)
    assert.deepEqual(shownIn(stage, old), ['plain'])
    litAs(old, 'photo')
    assert.deepEqual(shownIn(stage, fresh), ['p1'])
    assert.equal(fresh.visible.length, 2, 'the head and the photo over it are drawn into one picture')
    litAs(fresh, 'photo')
  }
  assert.ok(frames >= 19 && frames <= 21, `the fade took ${frames} frames`)
  assert.equal(renderer.marks.length, marks + 1)
  reduced = true
  const e = stage.entryOf('p1')
  checkPhoto(stage, e)
  assert.equal(e.mesh.material, stage.skin, 'under the photo the head keeps its own colours')
  assert.deepEqual(last().visible, [e.mesh, e.photo], 'the head first, then the photo over it')
  // once the texture is on the graphics card, the decoded picture is closed (it holds as much memory again)
  const { map } = e.photo.material
  const bitmap = map.image
  assert.ok(bitmap.width > 0 && bitmap.closed === true, 'the decoded picture is closed after it was sent')
  assert.equal(map.onUpdate, null)
  // Colour off: the bare shape, the photo hidden, the light of any bare shape
  stage.setColour(false)
  await step()
  assert.deepEqual(last().visible, [e.mesh])
  assert.equal(e.mesh.material, stage.clay)
  litAs(last(), 'plain')
  stage.setColour(true)
  await step()
  assert.deepEqual(last().visible, [e.mesh, e.photo])
  litAs(last(), 'photo')
  // Photo off: the head's own colours, at once and under the same light; a head loaded meanwhile has it hidden too
  stage.setPhoto(false)
  await step()
  assert.deepEqual(last().visible, [e.mesh])
  assert.equal(e.mesh.material, stage.skin)
  litAs(last(), 'photo')
  const p2 = look('p2', 'photo')
  give(stage, [plain, p1, p2])
  stage.want('p2')
  await settle()
  await step()
  assert.equal(stage.shown, 'p2')
  assert.deepEqual(last().visible, [stage.entryOf('p2').mesh])
  stage.setColour(false)
  stage.setColour(true)
  assert.equal(stage.entryOf('p2').photo.visible, false, 'Colour back on does not bring back a photo that is switched off')
  stage.setPhoto(true)
  await step()
  assert.deepEqual(last().visible, [stage.entryOf('p2').mesh, stage.entryOf('p2').photo])
  stage.want('p1')
  await settle()
  await step()
  // the brightness slider moves both lights of such a model too
  stage.setLight({ direction: 30, height: 35, brightness: 1.4 })
  await step()
  litAs(last(), 'photo', 1.4)
  stage.setLight({ direction: 30, height: 35, brightness: 1 })
  await step()
  // letting go: the layer's geometry, its material, its texture and the decoded picture
  const gone = []
  e.photo.geometry.addEventListener('dispose', () => gone.push('layer'))
  e.photo.material.addEventListener('dispose', () => gone.push('material'))
  map.addEventListener('dispose', () => gone.push('texture'))
  e.mesh.geometry.addEventListener('dispose', () => gone.push('head'))
  give(stage, [plain])
  assert.deepEqual(gone.sort(), ['head', 'layer', 'material', 'texture'])
  assert.deepEqual([...stage.entries.values()].map((x) => x.key), ['plain'], 'nothing of the heads with a photo is kept')
  assert.equal(stage.scene.children.filter((c) => c.isMesh).length, 1)
  stage.want('plain')
  await settle()
  await step()
  litAs(last(), 'plain') // a model without photos again
  // kept ready: of the heads that are not on screen, two with a photo at most
  const some = Array.from({ length: 4 }, (_, i) => look(`s${i}`, 'photo'))
  give(stage, [plain, ...some])
  const let_go = []
  for (const one of some) {
    stage.want(one.key)
    await settle()
    stage.entryOf(one.key).photo.material.map.addEventListener('dispose', () => let_go.push(one.key))
  }
  assert.deepEqual([stage.shown, stage.entries.size], ['s3', 4], 'the head on screen, two more with a photo, and the one without')
  assert.deepEqual(let_go, ['s0'])
  stage.want('plain')
  await settle()
  assert.deepEqual([stage.shown, stage.entries.size, let_go.join()], ['plain', 3, 's0,s1'])
  // heads the owner moved on from while they were on their way count at once
  const quick = Array.from({ length: 4 }, (_, i) => look(`q${i}`, 'photo', 'r5'))
  give(stage, [plain, ...quick])
  assert.equal(stage.entries.size, 1)
  for (const one of quick) stage.want(one.key)
  await settle()
  assert.equal(stage.shown, 'q3')
  const waiting = [...stage.entries.values()].filter((x) => x.photo && x.key !== 'q3').map((x) => x.key)
  assert.deepEqual(waiting.sort(), ['q1', 'q2'], 'two heads with a photo wait beside the one on screen: the ones asked for last')
  await step()
  assert.equal(bitmaps.filter((b) => !b.closed).length, 2, 'and only theirs are decoded pictures still open: they were never drawn')
  // The browser takes the graphics memory away: the heads with a photo are let go at once (their pictures were
  // closed, so they could not go back to the card). When it gives the memory back, the wanted one is fetched again.
  const lostBefore = fetches
  const kept = stage.entryOf('plain')
  canvas.fire('webglcontextlost', { preventDefault() {} })
  assert.deepEqual([...stage.entries.values()].map((x) => x.key), ['plain'], 'a head without a photo stays: it goes back to the card from the page')
  assert.deepEqual(told.states.at(-1), { shown: null, fading: false, loading: false, failed: false })
  assert.equal(fetches, lostBefore, 'nothing is fetched while the context is gone')
  renderer.restore()
  canvas.fire('webglcontextrestored', {})
  assert.deepEqual(told.states.at(-1), { shown: null, fading: false, loading: true, failed: false })
  assert.equal(fetches, lostBefore + 1, 'the head that was on screen is fetched again; the others when they are wanted')
  assert.equal(stage.entryOf('plain'), kept)
  await settle()
  await step()
  assert.equal(stage.shown, 'q3')
  assert.deepEqual(last().visible, [stage.entryOf('q3').mesh, stage.entryOf('q3').photo])
  assert.equal(renderer.sentClosed, 0, 'no texture was ever sent from a closed picture')
  assert.equal(renderer.staleLetGo, 0, 'and none was taken off a card that had lost it already')
  // a photo that cannot be decoded: the head shows with its own colours, without the layer
  const real = globalThis.createImageBitmap
  const say = console.error
  globalThis.createImageBitmap = async () => {
    throw new Error('cannot decode')
  }
  console.error = () => {} // three.js says that the texture did not load; that is what this step is about
  const dull = look('dull', 'photo', 'r10')
  give(stage, [plain, dull])
  stage.want('dull')
  await settle()
  globalThis.createImageBitmap = real
  console.error = say
  assert.deepEqual(told.states.at(-1), { shown: 'dull', fading: false, loading: false, failed: false })
  assert.equal(stage.entryOf('dull').photo, null)
  assert.equal(stage.entryOf('dull').mesh.material, stage.skin)
  await step()
  assert.equal(last().visible.length, 1, 'never a white sheet in place of the photo')
  litAs(last(), 'photo') // the model's light: the look is listed as one with a photo
  give(stage, [plain])
  assert.ok(bitmaps.every((b) => b.closed), 'every decoded picture was closed')
  give(stage, [...looks, plain])
  stage.want('l1')
  await settle()
  ok('Q a head with its photo: one picture, the model lit evenly, Colour and Photo off, two kept, let go whole, fetched again after a lost context')
}

// ---------------------------------------------------------------- M: letting meshes go
const many = Array.from({ length: 9 }, (_, i) => look(`m${i}`, `l${(i % 3) + 1}`))
give(stage, many)
assert.equal(stage.entries.size, 0, 'looks that are no longer listed are let go')
assert.equal(stage.shown, null)
const disposed = []
for (const m of many) {
  stage.want(m.key)
  await settle()
  stage.entryOf(m.key).mesh.geometry.addEventListener('dispose', () => disposed.push(m.key))
}
assert.equal(stage.shown, 'm8')
// The rule, each time a head comes on screen: of the others, latest first, four stay, two at most with a photo.
const alive = []
const expected = []
for (const m of many) {
  let kept = 0
  let photos = 0
  for (const x of [...alive].reverse()) {
    if (kept >= 4 || (x.textured && photos >= 2)) {
      alive.splice(alive.indexOf(x), 1)
      expected.push(x.key)
      continue
    }
    kept += 1
    if (x.textured) photos += 1
  }
  alive.push(m)
}
assert.deepEqual(disposed, expected)
if (!many.some((m) => m.textured)) assert.deepEqual(disposed, ['m0', 'm1', 'm2', 'm3'], 'the one on screen and four more')
if (many.every((m) => m.textured)) assert.deepEqual(disposed, ['m0', 'm1', 'm2', 'm3', 'm4', 'm5'], 'the one on screen and two more')
assert.deepEqual([...stage.entries.values()].map((x) => x.key).sort(), alive.map((m) => m.key).sort())
assert.equal(stage.scene.children.filter((c) => c.isMesh).length, alive.length)
stage.want('m0') // gone: fetched again
await settle()
assert.equal(stage.shown, 'm0')
ok(`M the head on screen and ${alive.length - 1} more are kept (four at most, two with a photo); older ones go first`)

// ---------------------------------------------------------------- N: a new model, same keys
const again = [look('m0', 'l2', 'r2'), look('m1', 'l3', 'r2')]
give(stage, again)
assert.equal(stage.shown, null)
assert.equal(told.states.at(-1).loading, true, 'the wanted key is fetched from the new model')
await settle()
assert.equal(stage.shown, 'm0')
assert.equal(stage.entryOf('m0').url, '/files/model3d/7/r2/m0.glb')
ok('N a new model with the same keys replaces the meshes')

// ---------------------------------------------------------------- P: the context goes and comes back
let prevented = 0
canvas.fire('webglcontextlost', { preventDefault: () => (prevented += 1) })
assert.deepEqual([told.lost.at(-1), prevented], [true, 1])
renderer.restore()
canvas.fire('webglcontextrestored', {})
assert.equal(told.lost.at(-1), false)
await settle()
await step()
assert.equal(stage.shown, 'm0', 'the head is on screen again')
assert.deepEqual(shownIn(stage, renderer.renders.at(-1)), ['m0'])
assert.deepEqual(renderer.clear, ['efefeb', 1], 'on the background of the page, not on black')
assert.deepEqual([renderer.sentClosed, renderer.staleLetGo], [0, 0])

// a resize draws at once and keeps the view
host.clientWidth = 300
host.clientHeight = 600
before = renderer.renders.length
observers[0].cb()
assert.equal(renderer.renders.length, before + 1)
assert.deepEqual(renderer.size, [300, 600, false])
near(stage.camera.aspect, 0.5)
near(stage.camera.position.length(), BASE / 0.5, 1e-3) // a tall picture: the head must fit its width
ok('P context lost and restored; a resize draws at once and fits the narrow side')

// ---------------------------------------------------------------- the theme
assert.equal(theme.themeSnapshot(), 'system:light')
dark = true // the system goes dark while the choice is "System"
assert.equal(theme.themeSnapshot(), 'system:dark', 'what React compares changes with the system, so the viewer is drawn again')
stage.setBackground()
assert.deepEqual(renderer.clear, ['23272b', 1], 'and the background is read again from the wrapper')
dark = false
theme.setTheme('dark')
assert.equal(theme.themeSnapshot(), 'dark:dark')
theme.setTheme('system')
assert.equal(theme.themeSnapshot(), 'system:light')
stage.setBackground()
assert.deepEqual(renderer.clear, ['efefeb', 1])
ok('the theme: a system switch under "System" reaches the viewer, and the background follows')

// ---------------------------------------------------------------- O: letting go of everything
stage.want('m1')
stage.setSpin(true)
stage.dispose()
assert.ok(renderer.disposed && renderer.lostOnPurpose && canvas.removed && copy.removed && observers[0].off)
assert.equal(rafs.length, 0)
assert.equal(stage.entries.size, 0)
assert.equal(canvas.heard.pointerdown.length, 0, 'the controls let go of the canvas')
const statesBefore = told.states.length
await settle()
await steps(3)
assert.equal(told.states.length, statesBefore, 'nothing is told after it is disposed')
ok('O dispose: renderer, context, canvases, observers, timers, meshes')

// ---------------------------------------------------------------- no WebGL
globalThis.__noWebGL = true
const host2 = element({ clientWidth: 600, clientHeight: 600, append: (...els) => (host2.children = els) })
assert.throws(() => new HeadStage(host2, { turn: 0, tilt: 0, zoom: 1 }, events), /WebGL/)
assert.equal(host2.children, undefined, 'nothing is added to the page when 3D cannot start')
ok('no WebGL: the stage refuses to start and leaves the page alone')

// ---------------------------------------------------------------- the look sliders
const L = (key, age, band, hair, hairLabel, facial, facialLabel, photos = 3) => ({
  key, label: '', age, age_band: band, hair, hair_label: hairLabel, facial_hair: facial, facial_hair_label: facialLabel, photos,
  taken_at: null, item_uid: 'u', picture_url: '', mesh_url: '', bytes: 0, vertices: 0,
})
{
  const three = [
    L('l1', 33.1, '25 to 34', 0.41, 'short hair', 0.6, 'a beard', 3),
    L('l2', 32.4, '25 to 34', 0.6, 'medium hair', 0.48, 'stubble or a short beard', 27),
    L('l3', 33.0, '25 to 34', 0.8, 'long hair', 0.49, 'stubble or a short beard', 4),
  ]
  const axes = model.lookAxes(three)
  assert.deepEqual(axes.map((a) => [a.key, a.stops.map((s) => s.text)]), [
    ['age', ['about 33']],
    ['hair', ['short hair', 'medium hair', 'long hair']],
    ['facial_hair', ['stubble or a short beard', 'a beard']],
  ])
  assert.deepEqual(axes[2].stops.map((s) => s.keys), [['l2', 'l3'], ['l1']])
  assert.equal(model.firstLook(three).key, 'l2', 'the look most photos show, in the latest age')
  assert.equal(model.pickLook(axes, 'l1', 'hair', 1), 'l2')
  assert.equal(model.pickLook(axes, 'l1', 'hair', 2), 'l3')
  assert.equal(model.pickLook(axes, 'l1', 'hair', 0), 'l1')
  assert.equal(model.pickLook(axes, 'l3', 'facial_hair', 1), 'l1')
  assert.equal(model.pickLook(axes, 'l1', 'facial_hair', 0), 'l2', 'of two looks with stubble, the one nearer in hair')
  assert.equal(model.pickLook(axes, 'l1', 'age', 0), 'l1')
  assert.equal(model.pickLook(axes, 'l1', 'hair', 7), 'l1', 'a stop that does not exist changes nothing')
  assert.equal(model.lookAxisText(axes[0], three[1]), 'about 32')
  assert.equal(model.lookAxisText(axes[1], three[1]), 'medium hair')
  assert.equal(model.stopOf(axes[1], 'l3'), 2)
  // every look can be reached from every other by the sliders alone
  for (const a of three) for (const b of three) {
    let at = a.key
    for (const axis of axes) if (axis.stops.length > 1) at = model.pickLook(axes, at, axis.key, model.stopOf(axis, b.key))
    assert.equal(at, b.key, `${a.key} to ${b.key}`)
  }
}
{
  const life = [
    L('c', 5.2, 'under 10', 0.45, 'short hair', 0.2, 'little or no facial hair', 1),
    L('y1', 24.0, '18 to 24', 0.85, 'long hair', 0.3, 'little or no facial hair', 9),
    L('y2', 24.5, '18 to 24', 0.44, 'short hair', 0.7, 'a beard', 5),
    L('a1', 33.0, '25 to 34', 0.43, 'short hair', 0.72, 'a beard', 12),
    L('a2', 34.0, '25 to 34', 0.46, 'short hair', 0.25, 'little or no facial hair', 30),
  ]
  const axes = model.lookAxes(life)
  assert.deepEqual(axes[0].stops.map((s) => s.text), ['about 5', 'about 24', 'about 34'])
  assert.deepEqual(axes[1].stops.map((s) => s.text), ['short hair', 'long hair'])
  assert.deepEqual(axes[2].stops.map((s) => s.text), ['little or no facial hair', 'a beard'])
  assert.equal(model.firstLook(life).key, 'a2')
  assert.equal(model.pickLook(axes, 'a1', 'age', 1), 'y2', 'the younger look that keeps the short hair and the beard')
  assert.equal(model.pickLook(axes, 'a2', 'age', 0), 'c')
  assert.equal(model.pickLook(axes, 'c', 'hair', 1), 'y1')
  assert.equal(model.pickLook(axes, 'y1', 'facial_hair', 1), 'y2', 'the beard at the same age before the one at another')
  assert.equal(model.lookAxisText(axes[0], life[3]), 'about 33')
}
{
  const bare = [L('l1', 30, '25 to 34', null, null, null, null), L('l2', 50, '45 to 59', null, null, null, null)]
  const axes = model.lookAxes(bare)
  assert.deepEqual(axes.map((a) => a.stops.length), [2, 0, 0], 'an axis nothing was read for has no slider')
  assert.equal(model.pickLook(axes, 'l1', 'age', 1), 'l2')
  const mixed = model.lookAxes([L('l1', 30, '25 to 34', null, null, 0.5, 'a beard'), L('l2', 31, '25 to 34', 0.5, 'short hair', 0.5, 'a beard')])
  assert.deepEqual(mixed[1].stops.map((s) => s.text), ['not read', 'short hair'])
  assert.equal(model.firstLook([]), undefined)
  assert.deepEqual(model.lookAxes([]).map((a) => a.stops.length), [0, 0, 0])
}
ok('look sliders: stops in order, every look reachable, the nearest look on the other axes wins')

// the options of a model, complete: one made before an option existed does not say it (or says null)
const D = { max_looks: 6, detail: 2, framing: 2.6, smoothing: 2, texture: 2, relief: 1 }
assert.deepEqual(model.MODEL3D_DEFAULTS, D)
assert.deepEqual(model.completeOptions({ max_looks: 4, detail: 3, framing: 2.2 }), { ...D, max_looks: 4, detail: 3, framing: 2.2 })
assert.deepEqual(model.completeOptions({ max_looks: 4, detail: 3, framing: 2.2, smoothing: null, texture: null, relief: null }), { ...D, max_looks: 4, detail: 3, framing: 2.2 })
assert.deepEqual(model.completeOptions(null), D)
assert.deepEqual(model.completeOptions({ smoothing: 0, texture: 3, relief: 0, framing: 2.6000000000000005 }), { ...D, smoothing: 0, texture: 3, relief: 0 })
assert.deepEqual(model.completeOptions({ smoothing: 9, texture: -1, detail: 'fine', max_looks: 2.4, relief: 5 }), { ...D, max_looks: 2, smoothing: 3, texture: 0 })
assert.deepEqual(Object.keys(model.MODEL3D_LIMITS), Object.keys(D), 'every option has its range')
// a model from before heads carried their photo: no look has one and the options do not say `texture`
{
  const one = L('l1', 30, '25 to 34', 0.4, 'short hair', 0.2, 'little or no facial hair')
  const old = { max_looks: 6, detail: 2, framing: 2.6 }
  assert.equal(model.beforePhotoTextures({ looks: [one], options: old }), true)
  assert.equal(model.beforePhotoTextures({ looks: [one], options: { ...old, smoothing: null, texture: null } }), true)
  assert.equal(model.beforePhotoTextures({ looks: [one], options: null }), true)
  assert.equal(model.beforePhotoTextures({ looks: [one], options: { ...old, texture: 0 } }), false, 'made since, with the photo switched off')
  assert.equal(model.beforePhotoTextures({ looks: [one], options: { ...old, texture: 2 } }), false, 'made since; this photo did not line up')
  assert.equal(model.beforePhotoTextures({ looks: [one, { ...one, key: 'l2', textured: true }], options: old }), false)
  assert.equal(model.beforePhotoTextures({ looks: [], options: null }), false, 'no model at all')
}

for (const r of results) console.log('ok  ', r)
console.log(`${results.length} groups passed`)
