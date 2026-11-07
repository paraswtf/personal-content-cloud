// Files shared to the installed app (PWA share target). public/sw.js answers POST /share-target by
// writing each file into IndexedDB (database 'pcc-share', object store 'files', auto-increment keys,
// values {blob, name, type, lastModified}) and redirecting to /upload?shared=1. The upload page drains
// the store here and hands the files to the upload engine. The record shape must match sw.js.

export const SHARE_DB = 'pcc-share'
export const SHARE_STORE = 'files'
const DB_VERSION = 1

interface SharedRecord {
  blob: Blob
  name: string
  type: string
  lastModified: number
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') return reject(new Error('IndexedDB is not available'))
    const req = indexedDB.open(SHARE_DB, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(SHARE_STORE)) db.createObjectStore(SHARE_STORE, { autoIncrement: true })
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error ?? new Error('Could not open the shared files store'))
    req.onblocked = () => reject(new Error('The shared files store is busy in another tab'))
  })
}

function toFile(r: SharedRecord, i: number): File {
  const name = r.name || `shared-${i + 1}`
  const type = r.type || r.blob.type || ''
  if (r.blob instanceof File && r.blob.name === name) return r.blob
  return new File([r.blob], name, { type, lastModified: r.lastModified || Date.now() })
}

/** Read every shared file and delete them from the store in the same transaction. */
async function readAndClear(): Promise<File[]> {
  const db = await openDb()
  try {
    return await new Promise<File[]>((resolve, reject) => {
      const tx = db.transaction(SHARE_STORE, 'readwrite')
      const store = tx.objectStore(SHARE_STORE)
      const all = store.getAll()
      let records: SharedRecord[] = []
      all.onsuccess = () => {
        records = (all.result as SharedRecord[]).filter((r) => r && r.blob instanceof Blob)
        store.clear()
      }
      tx.oncomplete = () => resolve(records.map(toFile))
      tx.onerror = () => reject(tx.error ?? new Error('Could not read the shared files'))
      tx.onabort = () => reject(tx.error ?? new Error('Reading the shared files was interrupted'))
    })
  } finally {
    db.close()
  }
}

let draining: Promise<File[]> | null = null

/**
 * Take the files waiting in the share store (each file is returned once). Concurrent callers, such
 * as a StrictMode double effect, share the same read.
 */
export function takeSharedFiles(): Promise<File[]> {
  if (!draining) {
    draining = readAndClear().finally(() => {
      draining = null
    })
  }
  return draining
}

/** Number of files waiting, without removing them (0 when the store cannot be opened). */
export async function countSharedFiles(): Promise<number> {
  try {
    const db = await openDb()
    try {
      return await new Promise<number>((resolve, reject) => {
        const req = db.transaction(SHARE_STORE, 'readonly').objectStore(SHARE_STORE).count()
        req.onsuccess = () => resolve(req.result)
        req.onerror = () => reject(req.error)
      })
    } finally {
      db.close()
    }
  } catch {
    return 0
  }
}
