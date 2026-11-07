import Foundation
import SwiftData

/// One row per photo library asset the app has dealt with.
@Model
final class SyncRecord {
    @Attribute(.unique) var assetId: String
    var filename: String
    var kind: String            // image, video, live
    var bytes: Int64
    var sha256: String?
    var uid: String?            // the item on the server once it is there
    var clipUid: String?        // the Live Photo clip's item, when it was paired
    var status: String          // pending, synced, failed, skipped
    var error: String?
    var createdAt: Date         // the asset's creation date
    var updatedAt: Date
    var deletedFromLibrary: Bool

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
