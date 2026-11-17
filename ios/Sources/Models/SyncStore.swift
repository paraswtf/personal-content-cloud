import Foundation
import SwiftData

/// One row per photo library asset the app has dealt with.
///
/// The upload state of section 28 lives here too, so a transfer continues after the app was
/// suspended or killed: the exported files of the asset (`files`), the upload in flight (`uploadId`,
/// `uploadResource`, `nextOffset`, `chunkPath`) and the attempt count for the backoff. New fields
/// have default values, so SwiftData migrates an existing store in place.
@Model
final class SyncRecord {
    @Attribute(.unique) var assetId: String
    var filename: String
    var kind: String            // image, video, live
    var bytes: Int64
    var sha256: String?
    var uid: String?            // the item on the server once it is there
    var clipUid: String?        // the Live Photo clip's item, when it was paired
    var status: String          // pending, uploading, uploaded (sent, the server is storing it), synced, failed, skipped
    var error: String?
    var createdAt: Date         // the asset's creation date
    var updatedAt: Date
    var deletedFromLibrary: Bool

    // section 28: resumable background uploads
    var uploadId: String?       // the server upload in flight for `uploadResource`
    var uploadResource: String? // still or video
    var nextOffset: Int64 = 0   // the next byte to send (the server's received_bytes)
    var chunkSize: Int = 0      // the server's chunk size for this upload
    var chunkPath: String?      // the chunk file of the task in flight, relative to UploadPaths.root
    var attempts: Int = 0       // failed attempts in a row (backoff 1, 2, 4, 8, 16, 30 s; at most 8)
    var filesData: Data?        // [UploadFile] as JSON: the exported originals and their upload state
    var trashedOnServer: Bool = false  // the server holds it, in its trash (shown apart in Delete synced)

    // The Upload screen's queue (Features/Upload). A file chosen there from Files, a picker copy or the camera
    // gets a record of its own ("upload:<uuid>", origin file or camera); a photo chosen from the library uses
    // the asset's own record (origin library), so the phone's sync knows it is sent.
    var queuedAt: Date?         // listed in the Upload screen's queue since then; nil when not listed
    var origin: String = "library"   // library (an asset of the phone's library), file, camera
    var serverStatus: String?   // the server's last word on the upload (queued, processing, stored, duplicate,
                                // rejected, error, cancelled), read while the queue lists it
    var thumbRef: String?       // the stored item's thumb (item_thumb_url)
    var itemVault: Bool = false // the stored item is in the vault (item_nsfw): open it with vault=1

    init(assetId: String, filename: String, kind: String, bytes: Int64, createdAt: Date) {
        self.assetId = assetId
        self.filename = filename
        self.kind = kind
        self.bytes = bytes
        self.status = "pending"
        self.createdAt = createdAt
        self.updatedAt = Date()
        self.deletedFromLibrary = false
    }

    /// The exported files of the asset, in upload order (the still, then a Live Photo's clip).
    var files: [UploadFile] {
        get { filesData.flatMap { try? JSONDecoder().decode([UploadFile].self, from: $0) } ?? [] }
        set { filesData = newValue.isEmpty ? nil : try? JSONEncoder().encode(newValue) }
    }

    /// A file of the Upload screen, not an asset of the library: its copy stays until the queue lets it go
    /// (Retry needs it after a failure or a cancel), and the phone's sync leaves it to the queue.
    var isQueueFile: Bool { origin != "library" }

    /// Forgets the upload in flight (the files and their finished uploads stay) and drops its chunk
    /// file, so a transfer that never completes leaves nothing behind.
    func clearTransfer() {
        if let chunkPath { try? FileManager.default.removeItem(at: UploadPaths.url(chunkPath)) }
        uploadId = nil
        uploadResource = nil
        nextOffset = 0
        chunkSize = 0
        chunkPath = nil
        attempts = 0
    }
}

/// One original file of an asset, exported for upload, and how far its upload got.
struct UploadFile: Codable, Equatable {
    var resource: String        // still or video (sent as UploadCreate.resource)
    var path: String            // relative to UploadPaths.root
    var filename: String
    var mime: String?
    var bytes: Int64
    var sha256: String
    var uploadId: String?       // the server upload for this file, once created
    var done: Bool              // the server has all of it (complete answered, or a duplicate)
    var itemUid: String?        // the item it became, when the server said so
}

/// Where exported originals and chunk files live while they upload: Application Support (not
/// Caches, which iOS may clear while a transfer is pending), excluded from backups. Paths are stored
/// relative to this folder because the app container path changes between installs and updates.
enum UploadPaths {
    static let root: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        var dir = base.appendingPathComponent("Uploads", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? dir.setResourceValues(values)
        return dir
    }()

    static func url(_ relative: String) -> URL { root.appendingPathComponent(relative) }

    static func relative(_ url: URL) -> String {
        let base = root.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        return path.hasPrefix(base + "/") ? String(path.dropFirst(base.count + 1)) : url.lastPathComponent
    }

    /// The folder of one asset's exported files (asset ids contain slashes).
    static func assetDir(_ assetId: String) -> URL {
        let safe = assetId.map { $0.isLetter || $0.isNumber || $0 == "-" ? $0 : "_" }
        let dir = root.appendingPathComponent("assets/" + String(safe), isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static var chunks: URL {
        let dir = root.appendingPathComponent("chunks", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func removeAsset(_ assetId: String) {
        try? FileManager.default.removeItem(at: assetDir(assetId))
    }
}

/// A line of the sync log shown in the app.
@Model
final class SyncEvent {
    var at: Date
    var text: String
    var level: String   // info, error

    init(_ text: String, level: String = "info") {
        self.at = Date()
        self.text = text
        self.level = level
    }
}

final class SyncStore {
    static let shared = SyncStore()
    let container: ModelContainer
    private var linesSinceTrim = 0

    private init() {
        let schema = Schema([SyncRecord.self, SyncEvent.self])
        let config = ModelConfiguration("sync", schema: schema)
        container = Self.open(schema: schema, config: config)
    }

    /// Opens the store, and starts the app even when it cannot. A store file that can be read but not opened (a
    /// model change iOS could not migrate) is moved aside and a new store made: the sync then asks the server what it
    /// already holds, by asset and by file hash, so nothing is sent twice. A store file that cannot be read now (the
    /// phone has not been unlocked since it started) is left as it is, and this launch keeps its state in memory.
    private static func open(schema: Schema, config: ModelConfiguration) -> ModelContainer {
        if let container = try? ModelContainer(for: schema, configurations: [config]) { return container }
        let url = config.url
        if readable(url) {
            let aside = url.path + ".unopened-\(Int(Date().timeIntervalSince1970))"
            for suffix in ["", "-wal", "-shm"] where FileManager.default.fileExists(atPath: url.path + suffix) {
                try? FileManager.default.moveItem(atPath: url.path + suffix, toPath: aside + suffix)
            }
            if let fresh = try? ModelContainer(for: schema, configurations: [config]) { return fresh }
        }
        let memory = ModelConfiguration("sync-memory", schema: schema, isStoredInMemoryOnly: true)
        do {
            return try ModelContainer(for: schema, configurations: [memory])
        } catch {
            fatalError("Could not open the sync store: \(error)")
        }
    }

    /// The store's file exists and its first bytes can be read now.
    private static func readable(_ url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        guard let head = try? handle.read(upToCount: 16) else { return false }
        return head.count == 16
    }

    /// A context for background work (the main context is for the views).
    func newContext() -> ModelContext {
        let ctx = ModelContext(container)
        ctx.autosaveEnabled = false
        return ctx
    }

    @MainActor
    func log(_ text: String, level: String = "info") {
        let ctx = container.mainContext
        ctx.insert(SyncEvent(text, level: level))
        // Keep the log short: every 50 lines, drop all but the newest 500. The Sync screen lists only
        // the newest 100, so it never shows a line being removed.
        linesSinceTrim += 1
        if linesSinceTrim >= 50 {
            linesSinceTrim = 0
            var old = FetchDescriptor<SyncEvent>(sortBy: [SortDescriptor(\.at, order: .reverse)])
            old.fetchOffset = 500
            for e in (try? ctx.fetch(old)) ?? [] { ctx.delete(e) }
        }
        try? ctx.save()
    }
}
