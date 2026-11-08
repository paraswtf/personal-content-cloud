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

    private init() {
        let schema = Schema([SyncRecord.self, SyncEvent.self])
        let config = ModelConfiguration("sync", schema: schema)
        do {
            container = try ModelContainer(for: schema, configurations: [config])
        } catch {
            fatalError("Could not open the sync store: \(error)")
        }
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
        // keep the log short
        if let old = try? ctx.fetch(FetchDescriptor<SyncEvent>(sortBy: [SortDescriptor(\.at, order: .reverse)])), old.count > 500 {
            for e in old.dropFirst(500) { ctx.delete(e) }
        }
        try? ctx.save()
    }
}
