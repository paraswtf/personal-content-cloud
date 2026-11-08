// The API build id, the same hash as backend/app/build_id.py: sha256 over (relative path, NUL, bytes,
// NUL) of backend/app/schemas.py, urls.py and api/*.py sorted by path, first 12 hex digits.
// vite.config.ts bakes it into __API_BUILD__; the shell compares it with GET /api/auth/state api_build.
// CLI: node api-build.mjs [backend/app dir]  (default: ../backend/app next to this frontend)
import { createHash } from 'node:crypto'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const TOP_FILES = ['schemas.py', 'urls.py']

function isFile(path) {
  try {
    return statSync(path).isFile()
  } catch {
    return false
  }
}

export function apiBuild(appDir) {
  const files = TOP_FILES.map((name) => join(appDir, name))
  const api = join(appDir, 'api')
  if (statSync(api).isDirectory()) for (const name of readdirSync(api)) if (name.endsWith('.py')) files.push(join(api, name))
  const present = files
    .filter(isFile)
    .map((path) => ({ rel: relative(appDir, path).split('\\').join('/'), path }))
    .sort((a, b) => (a.rel < b.rel ? -1 : a.rel > b.rel ? 1 : 0))
  const h = createHash('sha256')
  for (const { rel, path } of present) {
    h.update(rel)
    h.update('\0')
    h.update(readFileSync(path))
    h.update('\0')
  }
  return h.digest('hex').slice(0, 12)
}

/** The build id for the backend next to the frontend that holds `configUrl` (vite.config.ts's import.meta.url). */
export function apiBuildFrom(configUrl) {
  return apiBuild(fileURLToPath(new URL('../backend/app', configUrl)))
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const dir = process.argv[2] ? resolve(process.argv[2]) : fileURLToPath(new URL('../../backend/app', import.meta.url))
  process.stdout.write(apiBuild(dir) + '\n')
}
