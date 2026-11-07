// Service worker for the installed app. It does one thing: the PWA share target. A POST to
// /share-target (manifest.webmanifest share_target) carries the shared files; they are stored in
// IndexedDB (database 'pcc-share', store 'files', records {blob, name, type, lastModified}) and the
// browser is sent to /upload?shared=1, where src/lib/shared.ts drains the store into the uploader.
// Nothing is cached and no other request is intercepted.

const SHARE_DB = 'pcc-share'
const SHARE_STORE = 'files'

self.addEventListener('install', () => {
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim())
})

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(SHARE_DB, 1)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(SHARE_STORE)) db.createObjectStore(SHARE_STORE, { autoIncrement: true })
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

async function storeFiles(files) {
  const db = await openDb()
  try {
    await new Promise((resolve, reject) => {
      const tx = db.transaction(SHARE_STORE, 'readwrite')
      const store = tx.objectStore(SHARE_STORE)
      for (const f of files) {
        store.add({ blob: f, name: f.name || '', type: f.type || '', lastModified: f.lastModified || Date.now() })
      }
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
      tx.onabort = () => reject(tx.error)
    })
  } finally {
    db.close()
  }
}

async function handleShare(request) {
  try {
    const form = await request.formData()
    const files = form.getAll('files').filter((f) => f && typeof f === 'object' && 'size' in f && f.size > 0)
    if (files.length) await storeFiles(files)
  } catch (err) {
    // the upload page still opens; it shows nothing to add
    console.error('share target failed', err)
  }
  return Response.redirect('/upload?shared=1', 303)
}

self.addEventListener('fetch', (event) => {
  const req = event.request
  if (req.method !== 'POST') return
  const url = new URL(req.url)
  if (url.origin !== self.location.origin || url.pathname !== '/share-target') return
  event.respondWith(handleShare(req))
})
