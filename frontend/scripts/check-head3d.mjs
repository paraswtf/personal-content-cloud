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
import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
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
    globalThis.__renderers.push(this)
  }
  setPixelRatio(r) { this.ratio = r }
  setSize(w, h, style) { this.size = [w, h, style]; this.domElement.width = Math.floor(w * this.ratio); this.domElement.height = Math.floor(h * this.ratio) }
  setClearColor(c, a) { this.clear = [c.getHexString(), a] }
  render(scene, camera) {
    scene.updateMatrixWorld()
    const visible = []
    scene.traverse((o) => { if (o.isMesh && o.visible) visible.push(o) })
    this.renders.push({ visible, at: globalThis.__clock(), camera: camera.position.toArray() })
  }
  dispose() { this.disposed = true }
  forceContextLoss() { this.lostOnPurpose = true }
}
`,
)

const STUBS = {
  react: 'export const useEffect=()=>{}, useImperativeHandle=()=>{}, useRef=(v)=>({current:v}), useState=(v)=>[v,()=>{}], useSyncExternalStore=(s,g)=>g()',
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
const files = new Map()
let fetches = 0
globalThis.fetch = async (url, { signal } = {}) => {
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
const look = (key, file = key, rev = 'r1') => {
  const url = `/files/model3d/7/${rev}/${key}.glb`
  files.set(url, path.join(GLB, `${file}.glb`))
  return { key, mesh_url: url }
}
const BASE = 0.62 / Math.tan((15 * Math.PI) / 180)
const told = { views: [], states: [], lost: [] }
const events = { onView: (v) => told.views.push(v), onState: (s) => told.states.push(s), onLost: (l) => told.lost.push(l) }
const host = element({ clientWidth: 600, clientHeight: 600, append: (...els) => (host.children = els) })
const keyOf = (stage, mesh) => [...stage.entries.values()].find((e) => e.mesh === mesh)?.key
const shownIn = (stage, render) => render.visible.map((m) => keyOf(stage, m))
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
stage.setLooks(looks)
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
ok('B first look, colours linear, face towards the camera')

// ---------------------------------------------------------------- C: a cross-fade
drawn.length = 0
let before = renderer.renders.length
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
assert.ok(fadeFrames >= 17 && fadeFrames <= 19, `the fade took ${fadeFrames} frames`)
assert.equal(drawn.length, fadeFrames, 'the old look is copied once a frame')
assert.ok(drawn.every((d) => d.source === canvas && d.rest.join() === '0,0,1200,1200'))
assert.ok(opacities[0] > 0.98 && opacities.at(-1) < 0.05, `opacity went ${opacities[0]} to ${opacities.at(-1)}`)
assert.ok(opacities.every((o, i) => i === 0 || o < opacities[i - 1]), 'the copy only fades out')
assert.equal(copy.style.opacity, '0')
assert.deepEqual(told.states.at(-1), { shown: 'l2', fading: false, loading: false, failed: false })
assert.deepEqual(shownIn(stage, renderer.renders.at(-1)), ['l2'])
await step()
assert.equal(rafs.length, 0)
ok(`C cross-fade: ${fadeFrames} frames of two opaque pictures, ${renderer.renders.length - before} draws`)

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
stage.setLooks([...looks, extra])
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
assert.ok([...stage.entries.values()].every((e) => e.mesh.material === stage.clay))
stage.setColour(true)
assert.ok([...stage.entries.values()].every((e) => e.mesh.material === stage.skin))
stage.setView({ turn: 0, tilt: 0, zoom: 1 })
stage.setLight({ direction: 90, height: 0, brightness: 1 })
await step()
const from = () => stage.lamp.getWorldPosition(new THREE.Vector3()).sub(stage.lamp.target.getWorldPosition(new THREE.Vector3()))
assert.deepEqual(from().toArray().map((v) => Math.round(v * 1e3) / 1e3), [1, 0, 0], 'at 90 degrees the lamp is at the viewer\'s right')
near(stage.lamp.intensity, Math.PI * 0.62)
near(stage.soft.intensity, Math.PI * 0.5)
stage.setView({ turn: 90 })
await step()
assert.deepEqual(from().toArray().map((v) => Math.round(v * 1e3) / 1e3 + 0), [0, 0, 1], 'and stays there for the viewer when the head turns')
stage.setLight({ direction: 0, height: 90, brightness: 1.5 })
await step()
near(from().y, 1)
near(stage.lamp.intensity, Math.PI * 0.62 * 1.5)
stage.setView({ turn: 0 })
ok('I colour switch; J the lamp follows the viewer')

// ---------------------------------------------------------------- K, L: odd files
reduced = true
const plain = look('plain')
const broken = look('broken')
const missing = { key: 'gone', mesh_url: '/files/model3d/7/r1/gone.glb' }
stage.setLooks([...looks, plain, broken, missing])
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

// ---------------------------------------------------------------- M: letting meshes go
const many = Array.from({ length: 9 }, (_, i) => look(`m${i}`, `l${(i % 3) + 1}`))
stage.setLooks(many)
assert.equal(stage.entries.size, 0, 'looks that are no longer listed are let go')
assert.equal(stage.shown, null)
const disposed = []
for (const m of many) {
  stage.want(m.key)
  await settle()
  stage.entryOf(m.key).mesh.geometry.addEventListener('dispose', () => disposed.push(m.key))
}
assert.equal(stage.shown, 'm8')
assert.equal(stage.entries.size, 5, 'the one on screen and four more')
assert.deepEqual(disposed, ['m0', 'm1', 'm2', 'm3'])
assert.equal(stage.scene.children.filter((c) => c.isMesh).length, 5)
stage.want('m0') // gone: fetched again
await settle()
assert.equal(stage.shown, 'm0')
ok('M at most five meshes are kept; the least recently shown goes first')

// ---------------------------------------------------------------- N: a new model, same keys
const again = [look('m0', 'l2', 'r2'), look('m1', 'l3', 'r2')]
stage.setLooks(again)
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
canvas.fire('webglcontextrestored', {})
assert.equal(told.lost.at(-1), false)
await step()

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

for (const r of results) console.log('ok  ', r)
console.log(`${results.length} groups passed`)
