// A check of how overlapping boxes stack on a photo (docs/ARCHITECTURE.md section 14), without a browser: the real
// `stackLayers`, `boxArea` and `stillBoxes` from src/components/viewer/boxes.ts, bundled with esbuild. A smaller box
// is drawn above a larger one, whatever order the boxes come in; the page then paints each box at its layer.
// Whether a tap really reaches the smaller box is for a browser; this checks the rule the page is given.
//
// Run by backend/tests/test_viewer_boxes.py:
//   node scripts/check-boxes.mjs <scratch dir>
import assert from 'node:assert/strict'
import { mkdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

if (process.argv.length < 3) {
  console.error('usage: node check-boxes.mjs <scratch dir>')
  process.exit(2)
}
const FRONT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const HERE = path.resolve(process.argv[2])
mkdirSync(HERE, { recursive: true })
const { build } = await import(pathToFileURL(path.join(FRONT, 'node_modules/esbuild/lib/main.js')).href)
const out = path.join(HERE, 'boxes.mjs')
await build({
  entryPoints: [path.join(FRONT, 'src/components/viewer/boxes.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: out,
  logLevel: 'error',
})
const { boxArea, boxKey, stackLayers, stillBoxes } = await import(pathToFileURL(out).href)

let groups = 0
function group(name, fn) {
  fn()
  groups += 1
  console.log(`ok ${String.fromCharCode(64 + groups)} ${name}`)
}

const face = (id, x, y, w, h) => ({ kind: 'face', box: { id, x, y, w, h } })
const body = (id, x, y, w, h) => ({ kind: 'body', box: { id, x, y, w, h } })
const pet = (id, x, y, w, h) => ({ kind: 'animal', box: { id, x, y, w, h } })
const layerOf = (layers, t) => layers.get(boxKey(t))

/** Every order of `list`. */
function orders(list) {
  if (list.length <= 1) return [list]
  return list.flatMap((t, i) => orders([...list.slice(0, i), ...list.slice(i + 1)]).map((rest) => [t, ...rest]))
}

group('a face inside a person box is above it, though the person box comes later', () => {
  const f = face(1, 0.4, 0.2, 0.1, 0.12)
  const b = body(2, 0.3, 0.1, 0.4, 0.8)
  const layers = stackLayers([f, b])
  assert.equal(layerOf(layers, b), 1)
  assert.equal(layerOf(layers, f), 2)
})

group('a small person box is above a large pet box, and a face in a pet box is above the pet', () => {
  const small = body(5, 0.45, 0.45, 0.1, 0.2)
  const dog = pet(9, 0.1, 0.3, 0.8, 0.6)
  const f = face(3, 0.2, 0.35, 0.06, 0.08)
  const layers = stackLayers([f, small, dog])
  assert.ok(layerOf(layers, small) > layerOf(layers, dog))
  assert.ok(layerOf(layers, f) > layerOf(layers, dog))
  assert.ok(layerOf(layers, f) > layerOf(layers, small))
})

group('the layers do not depend on the order the boxes come in (all 120 orders of five boxes)', () => {
  const boxes = [
    face(1, 0.1, 0.1, 0.1, 0.1),
    face(2, 0.12, 0.12, 0.05, 0.06),
    body(3, 0.05, 0.05, 0.5, 0.9),
    body(4, 0.3, 0.2, 0.2, 0.7),
    pet(5, 0.0, 0.5, 0.9, 0.45),
  ]
  const want = stackLayers(boxes)
  let n = 0
  for (const order of orders(boxes)) {
    const got = stackLayers(order)
    for (const t of boxes) assert.equal(layerOf(got, t), layerOf(want, t), `${boxKey(t)} in ${order.map(boxKey).join(' ')}`)
    n += 1
  }
  assert.equal(n, 120)
})

group('layers run from 1 for the largest to the count for the smallest, each once', () => {
  const boxes = [body(1, 0, 0, 0.9, 0.9), face(2, 0, 0, 0.2, 0.2), face(3, 0, 0, 0.1, 0.1), pet(4, 0, 0, 0.5, 0.5)]
  const layers = stackLayers(boxes)
  assert.deepEqual([...layers.values()].sort((a, b) => a - b), [1, 2, 3, 4])
  const byLayer = [...layers.entries()].sort((a, b) => a[1] - b[1]).map(([k]) => k)
  assert.deepEqual(byLayer, ['body:1', 'animal:4', 'face:2', 'face:3'])
})

group('for every pair, the smaller box has the higher layer (a crowd of 40 boxes)', () => {
  const boxes = []
  for (let i = 0; i < 40; i++) {
    const w = 0.02 + ((i * 37) % 23) / 40
    const h = 0.03 + ((i * 53) % 19) / 30
    boxes.push(i % 3 === 0 ? face(i, 0.1, 0.1, w, h) : i % 3 === 1 ? body(i, 0.1, 0.1, w, h) : pet(i, 0.1, 0.1, w, h))
  }
  const layers = stackLayers(boxes)
  for (const a of boxes) {
    for (const b of boxes) {
      if (boxArea(a.box) < boxArea(b.box)) assert.ok(layerOf(layers, a) > layerOf(layers, b), `${boxKey(a)} over ${boxKey(b)}`)
    }
  }
})

group('equal areas keep the list order, the later box above, every time', () => {
  const a = face(1, 0.1, 0.1, 0.2, 0.3)
  const b = body(1, 0.15, 0.1, 0.3, 0.2) // the same area, another shape and kind
  assert.equal(boxArea(a.box), boxArea(b.box))
  for (let i = 0; i < 3; i++) {
    const layers = stackLayers([a, b])
    assert.equal(layerOf(layers, a), 1)
    assert.equal(layerOf(layers, b), 2)
  }
  const flipped = stackLayers([b, a])
  assert.equal(layerOf(flipped, b), 1)
  assert.equal(layerOf(flipped, a), 2)
})

group('a box of no size or a broken size counts as the smallest and leaves the others in order', () => {
  const big = body(1, 0, 0, 0.8, 0.8)
  const mid = face(2, 0, 0, 0.3, 0.3)
  const broken = [face(3, 0, 0, Number.NaN, 0.2), face(4, 0, 0, -0.1, 0.2), face(5, 0, 0, 0, 0.2), face(6, 0, 0, Infinity, 0.1)]
  for (const t of broken) assert.equal(boxArea(t.box), 0)
  const layers = stackLayers([big, ...broken, mid])
  assert.equal(layerOf(layers, big), 1)
  assert.equal(layerOf(layers, mid), 2)
  for (const t of broken) assert.ok(layerOf(layers, t) > 2)
  assert.equal(stackLayers([]).size, 0)
})

group('the order by stored area is the order by area on screen, for a tall and a wide photo', () => {
  const boxes = [face(1, 0, 0, 0.3, 0.1), body(2, 0, 0, 0.1, 0.25), pet(3, 0, 0, 0.2, 0.14)]
  const layers = stackLayers(boxes)
  for (const [W, H] of [[3000, 4000], [4000, 1000], [1080, 1920]]) {
    const onScreen = [...boxes].sort((a, b) => b.box.w * W * b.box.h * H - a.box.w * W * a.box.h * H).map(boxKey)
    const byLayer = [...layers.entries()].sort((a, b) => a[1] - b[1]).map(([k]) => k)
    assert.deepEqual(byLayer, onScreen, `${W} x ${H}`)
  }
})

group('the boxes of a photo as the stage lists them: faces, person boxes without a face, pets', () => {
  const detail = {
    faces: [
      { id: 11, x: 0.42, y: 0.2, w: 0.08, h: 0.1, frame_time_s: null },
      { id: 12, x: 0.1, y: 0.1, w: 0.1, h: 0.1, frame_time_s: 3.2 }, // a video frame's face, not drawn on a still
    ],
    bodies: [
      { id: 21, x: 0.3, y: 0.1, w: 0.45, h: 0.85, frame_time_s: null, face_id: null }, // around the face, its own face not found
      { id: 22, x: 0.35, y: 0.15, w: 0.3, h: 0.8, frame_time_s: null, face_id: 11 }, // the face's own person box: not drawn
    ],
    animals: [{ id: 31, x: 0.0, y: 0.4, w: 1.0, h: 0.6, frame_time_s: null }],
  }
  const boxes = stillBoxes(detail)
  assert.deepEqual(boxes.map(boxKey), ['face:11', 'body:21', 'animal:31'])
  const layers = stackLayers(boxes)
  assert.deepEqual([layers.get('animal:31'), layers.get('body:21'), layers.get('face:11')], [1, 2, 3])
})

console.log(`${groups} groups passed`)
