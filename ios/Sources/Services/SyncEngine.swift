import Foundation
import Photos
import SwiftData
import Network
import UIKit

/// The sync run (docs/ARCHITECTURE.md sections 21 and 28): ask the server what it holds, continue
/// the uploads that were interrupted from the server's offset, then export, hash and upload the
/// rest through the background uploader, and record every outcome. One run at a time.
@MainActor
final class SyncEngine: ObservableObject {
    static let shared = SyncEngine()

    @Published var running = false
    @Published var statusLine = "Idle"
    @Published var progress: Double = 0
    @Published var lastError: String?

    private var task: Task<Void, Never>?
    private let monitor = NWPathMonitor()
    private var onWifi = true

    /// What the sync check answered for the assets of a run.
    struct ServerState {
        var assets: [String: String] = [:]
        var pending: [String: [PendingUpload]] = [:]
        var trashed: Set<String> = []
    }

    private init() {
        monitor.pathUpdateHandler = { [weak self] path in
            let wifi = path.usesInterfaceType(.wifi) || path.usesInterfaceType(.wiredEthernet)
            Task { @MainActor in self?.onWifi = wifi }
        }
        monitor.start(queue: DispatchQueue(label: "pcc.netpath"))
        UIDevice.current.isBatteryMonitoringEnabled = true
    }

    /// Starts a run unless one is going. `manual` ignores the Wi-Fi and charging options.
    func start(manual: Bool = false, deadline: Date? = nil) {
        guard !running else { return }
        running = true
        lastError = nil
        task = Task { [weak self] in
            guard let self else { return }
            await self.run(manual: manual, deadline: deadline)
            await MainActor.run {
                self.running = false
                self.statusLine = self.lastError == nil ? "Up to date" : "Stopped: \(self.lastError ?? "")"
                AppSettings.lastSyncAt = Date()
            }
        }
    }

    /// Ends the run (the background task's time is up). A transfer in flight goes on in the
    /// background session.
    func cancel() {
        task?.cancel()
    }

    /// The Stop button: ends the run and pauses the transfer in flight; the next run continues it.
    func stop() {
        task?.cancel()
        BackgroundUploader.shared.pause()
    }

    // MARK: the run

    private func allowedNow(manual: Bool) -> String? {
        if manual { return nil }
        if AppSettings.wifiOnly && !onWifi { return "Waiting for Wi-Fi" }
        if AppSettings.chargingOnly {
            let s = UIDevice.current.batteryState
            if s != .charging && s != .full { return "Waiting for the charger" }
        }
        return nil
    }

    private func run(manual: Bool, deadline: Date?) async {
        guard PhotoLibrary.hasAccess else {
            lastError = "No access to Photos"
            SyncStore.shared.log("No access to Photos. Allow it in Settings.", level: "error")
            return
        }
        guard let api = APIClient.current() else {
            lastError = "Not signed in"
            return
        }
        if let wait = allowedNow(manual: manual) {
            statusLine = wait
            SyncStore.shared.log(wait)
            return
        }
        do {
            _ = try await api.state()
        } catch {
            lastError = "Cannot reach the server: \(error.localizedDescription)"
            SyncStore.shared.log(lastError ?? "", level: "error")
            return
        }

        // One context for the engine, the uploader's delegate and the views: the main context.
        let ctx = SyncStore.shared.container.mainContext
        let uploader = BackgroundUploader.shared
        let assets = PhotoLibrary.assets(since: AppSettings.syncSince, includeVideos: AppSettings.includeVideos)
        statusLine = "Checking \(assets.count) items"
        let records = (try? ctx.fetch(FetchDescriptor<SyncRecord>())) ?? []
        var byId = Dictionary(records.map { ($0.assetId, $0) }, uniquingKeysWith: { a, _ in a })
        let inLibrary = Dictionary(assets.map { ($0.localIdentifier, $0) }, uniquingKeysWith: { a, _ in a })
        // Interrupted uploads come first, whatever the current filter says (they hold server state).
        let unfinished = records.filter { ($0.status == "uploading" || $0.status == "pending") && $0.files.contains { !$0.done } }
        let todo = assets.filter { byId[$0.localIdentifier]?.status != "synced" }
        SyncStore.shared.log("\(assets.count) items in the library, \(todo.count) to check")

        // 1. What the server holds, and this device's unfinished uploads, by asset id.
        var askIds: [String] = []
        var seen = Set<String>()
        for id in unfinished.map(\.assetId) + todo.map(\.localIdentifier) where seen.insert(id).inserted { askIds.append(id) }
        var server = ServerState()
        for batch in stride(from: 0, to: askIds.count, by: 200) {
            if Task.isCancelled { return }
            let ids = Array(askIds[batch..<min(batch + 200, askIds.count)])
            do {
                let out = try await api.check(hashes: [], assetIds: ids)
                server.assets.merge(out.assets) { a, _ in a }
                server.pending.merge(out.pending) { a, _ in a }
                server.trashed.formUnion(out.trashed)
            } catch {
                lastError = "The sync check failed: \(error.localizedDescription)"
                SyncStore.shared.log(lastError ?? "", level: "error")
                return
            }
        }

        // 1b. Assets sent earlier that the server lists now are synced, whatever the run does next, so
        // the "sent, waiting for the server" count on the Sync screen is right from the start.
        for rec in records where rec.status == "uploaded" {
            if let uid = server.assets[rec.assetId], rec.files.allSatisfy({ $0.done }) {
                markSynced(rec, uid: uid, trashed: server.trashed.contains(rec.assetId))
            }
        }

        // 2. Continue the interrupted uploads from the server's offset.
        let live = await uploader.liveAssets()
        var handled = Set<String>()
        for rec in unfinished {
            if Task.isCancelled { return }
            if let deadline, Date() > deadline { SyncStore.shared.log("Background time is up, continuing later"); return }
            handled.insert(rec.assetId)
            statusLine = "Continuing \(rec.filename.isEmpty ? "an upload" : rec.filename)"
            if !live.contains(rec.assetId) {
                let asset = inLibrary[rec.assetId] ?? PhotoLibrary.assets(ids: [rec.assetId]).first
                await resume(rec, asset: asset, server: server, api: api)
            }
            await uploader.wait(for: rec.assetId, until: deadline)
        }

        // 3. The rest, newest first.
        var done = 0
        for asset in todo {
            if Task.isCancelled { break }
            if let deadline, Date() > deadline { SyncStore.shared.log("Background time is up, continuing later"); break }
            if let wait = allowedNow(manual: manual) { statusLine = wait; break }
            done += 1
            progress = Double(done) / Double(max(1, todo.count))
            let id = asset.localIdentifier
            if handled.contains(id) { continue }
            let isNew = byId[id] == nil
            let rec = byId[id] ?? {
                let r = SyncRecord(assetId: id, filename: "", kind: asset.mediaType == .video ? "video" : "image",
                                   bytes: 0, createdAt: asset.creationDate ?? Date())
                ctx.insert(r)
                byId[id] = r
                return r
            }()
            // Held by the server, and no half of it left to send (a Live Photo whose clip failed is
            // listed by its still, so its record decides).
            if let uid = server.assets[id], rec.files.allSatisfy({ $0.done }),
               rec.status == "uploaded" || (server.pending[id] ?? []).isEmpty {
                markSynced(rec, uid: uid, trashed: server.trashed.contains(id))
                continue
            }
            if rec.status == "uploaded" {
                await verify(rec, api: api)
                if rec.status != "pending" { try? ctx.save(); continue }
            }
            statusLine = "Uploading \(done) of \(todo.count)"
            // A little background time so an export started in the foreground can hand its file to
            // the background session when the owner leaves the app. Ended at the hand-off: the
            // transfer itself needs none.
            let time = BackgroundTime("pcc.export")
            defer { time.end() }
            do {
                try await prepare(rec, asset: asset, api: api)
                if isNew, await knownByHash(rec, api: api) {
                    // After a reinstall, or a photo first uploaded from the web: the server has these
                    // files (maybe in its trash, which a create would undo), so nothing is sent.
                    try? ctx.save()
                    continue
                }
                uploader.start(rec)
                time.end()
                await uploader.wait(for: id, until: deadline)
            } catch {
                rec.status = "failed"
                rec.error = error.localizedDescription
                rec.updatedAt = Date()
                UploadPaths.removeAsset(id)
                SyncStore.shared.log("\(rec.filename.isEmpty ? id : rec.filename): \(error.localizedDescription)", level: "error")
            }
            try? ctx.save()
        }
        progress = 1
    }

    private func markSynced(_ rec: SyncRecord, uid: String, trashed: Bool) {
        if rec.uid == nil { rec.uid = uid }
        rec.status = "synced"
        rec.error = nil
        rec.trashedOnServer = trashed
        rec.clearTransfer()
        rec.updatedAt = Date()
        UploadPaths.removeAsset(rec.assetId)
        try? SyncStore.shared.container.mainContext.save()
    }

    /// Puts an interrupted upload back on the wire (section 28). The server's `pending` list gives
    /// the offset; an upload it does not list had its last reply lost, and its own state
    /// (`GET /api/uploads/{id}`) says what happened.
    private func resume(_ rec: SyncRecord, asset: PHAsset?, server: ServerState, api: APIClient) async {
        let uploader = BackgroundUploader.shared
        let present = rec.files.filter { !$0.done }.allSatisfy { f in
            let size = (try? FileManager.default.attributesOfItem(atPath: UploadPaths.url(f.path).path)[.size] as? Int64) ?? -1
            return size == f.bytes
        }
        if !present {
            guard let asset else {
                rec.status = "failed"
                rec.error = "The photo is no longer in the library"
                rec.clearTransfer()
                UploadPaths.removeAsset(rec.assetId)
                return
            }
            do {
                try await prepare(rec, asset: asset, api: api)
            } catch {
                rec.status = "failed"
                rec.error = error.localizedDescription
                return
            }
        }
        guard let id = rec.uploadId else {
            uploader.start(rec)          // the create was never answered: it is idempotent, ask again
            return
        }
        if let p = server.pending[rec.assetId]?.first(where: { $0.upload_id == id }) {
            uploader.resume(rec, at: p.received_bytes)
            return
        }
        do {
            let up = try await api.upload(id: id)
            uploader.resume(rec, with: up)
        } catch {
            rec.status = "pending"
            rec.error = "Cannot reach the server: \(error.localizedDescription)"
        }
    }

    /// An asset sent earlier that the server does not list yet: still being stored, turned down,
    /// or dropped (then it is sent again).
    private func verify(_ rec: SyncRecord, api: APIClient) async {
        var files = rec.files
        var waiting = false
        for i in files.indices {
            guard let id = files[i].uploadId else { continue }
            guard let up = try? await api.upload(id: id) else {
                if (try? await api.state()) != nil { files[i].done = false; files[i].uploadId = nil }
                else { waiting = true }   // the server is not reachable now
                continue
            }
            switch up.status {
            case "stored", "duplicate":
                files[i].itemUid = up.item_uid
            case "rejected", "error":
                rec.status = "failed"
                rec.error = up.message ?? "The server turned the file down"
                rec.files = files
                SyncStore.shared.log("\(rec.filename): \(rec.error ?? "")", level: "error")
                return
            case "cancelled":
                files[i].done = false
                files[i].uploadId = nil
            default:
                waiting = true      // queued, processing
            }
        }
        rec.files = files
        if files.contains(where: { !$0.done }) {
            rec.status = "pending"          // sent again by the caller
        } else if !waiting {
            rec.status = "synced"
            rec.uid = files.first?.itemUid ?? rec.uid
        }
        rec.updatedAt = Date()
    }

    /// Exports the asset's originals into its upload folder and hashes them. A file that is the
    /// same as before keeps its upload; the unfinished upload of a file that changed is cancelled.
    private func prepare(_ rec: SyncRecord, asset: PHAsset, api: APIClient) async throws {
        let old = rec.files
        UploadPaths.removeAsset(rec.assetId)
        let exported = try await PhotoLibrary.export(asset, to: UploadPaths.assetDir(rec.assetId))
        guard let main = exported.first(where: { !$0.isClip }) else { throw APIError(status: 0, detail: "No original file") }
        var files: [UploadFile] = []
        for f in exported {
            let url = f.url
            let sha = try await Task.detached(priority: .utility) { try PhotoLibrary.sha256(of: url) }.value
            let resource = f.isClip || asset.mediaType == .video ? "video" : "still"
            let prior = old.first { $0.resource == resource }
            let same = prior.map { $0.bytes == f.bytes && $0.sha256 == sha } ?? false
            if let prior, !same, !prior.done, let id = prior.uploadId {
                try? await api.cancelUpload(id: id)   // its bytes are of another version of the file
            }
            files.append(UploadFile(resource: resource, path: UploadPaths.relative(url), filename: f.filename, mime: f.mime,
                                    bytes: f.bytes, sha256: sha, uploadId: same ? prior?.uploadId : nil,
                                    done: same && (prior?.done ?? false), itemUid: same ? prior?.itemUid : nil))
        }
        if let current = rec.uploadResource, rec.uploadId != nil,
           !files.contains(where: { $0.resource == current && $0.uploadId == rec.uploadId }) {
            rec.clearTransfer()   // its file changed (the old upload was cancelled above)
        }
        rec.files = files
        rec.filename = main.filename
        rec.bytes = main.bytes
        rec.sha256 = files.first(where: { $0.resource == (asset.mediaType == .video ? "video" : "still") })?.sha256
        if asset.mediaSubtypes.contains(.photoLive) { rec.kind = "live" }
        rec.updatedAt = Date()
    }

    /// For an asset without a record: asks the sync check by file hash. True when the server
    /// holds every file (the record is then synced).
    private func knownByHash(_ rec: SyncRecord, api: APIClient) async -> Bool {
        let files = rec.files
        guard !files.isEmpty, let out = try? await api.check(hashes: files.map(\.sha256), assetIds: []),
              files.allSatisfy({ out.known[$0.sha256] != nil }) else { return false }
        rec.uid = out.known[files[0].sha256]
        if files.count > 1 { rec.clipUid = out.known[files[1].sha256] }
        rec.files = files.map { var f = $0; f.done = true; f.itemUid = out.known[f.sha256]; return f }
        rec.status = "synced"
        rec.error = nil
        rec.updatedAt = Date()
        UploadPaths.removeAsset(rec.assetId)
        SyncStore.shared.log("Already in the cloud: \(rec.filename)")
        return true
    }

    // MARK: delete synced

    /// A synced asset still in the library.
    struct SyncedRow: Identifiable {
        let asset: PHAsset
        let rec: SyncRecord
        var id: String { rec.assetId }
    }

    /// The synced assets still in the library, as the server confirms them right now: `stored`
    /// (held and not in the trash) and `trashed` (held, but in the server's trash, so the owner
    /// decides). An asset the server cannot confirm is in neither list.
    func confirmedSynced() async -> (stored: [SyncedRow], trashed: [SyncedRow]) {
        guard let api = APIClient.current() else { return ([], []) }
        let ctx = SyncStore.shared.container.mainContext
        let synced = ((try? ctx.fetch(FetchDescriptor<SyncRecord>(predicate: #Predicate<SyncRecord> { $0.status == "synced" && !$0.deletedFromLibrary }))) ?? [])
        let assets = PhotoLibrary.assets(ids: synced.map(\.assetId))
        let present = Dictionary(assets.map { ($0.localIdentifier, $0) }, uniquingKeysWith: { a, _ in a })
        let mine = synced.filter { present[$0.assetId] != nil }
        var server = ServerState()
        let ids = mine.map(\.assetId)
        for batch in stride(from: 0, to: ids.count, by: 200) {
            let slice = Array(ids[batch..<min(batch + 200, ids.count)])
            guard let out = try? await api.check(hashes: [], assetIds: slice) else { return ([], []) }
            server.assets.merge(out.assets) { a, _ in a }
            server.pending.merge(out.pending) { a, _ in a }
            server.trashed.formUnion(out.trashed)
        }
        // Assets the server knows only by file hash (synced after a reinstall, or first uploaded
        // from the web): the item itself says whether it sits in the trash.
        var byHash: [String: Bool] = [:]   // asset id -> in the trash
        let hashOnly = mine.filter { server.assets[$0.assetId] == nil }
        func hashes(_ rec: SyncRecord) -> [String] {
            rec.files.isEmpty ? [rec.sha256].compactMap { $0 } : rec.files.map(\.sha256)
        }
        let wanted = Array(Set(hashOnly.flatMap(hashes)))
        var known: [String: String] = [:]
        for batch in stride(from: 0, to: wanted.count, by: 1000) {
            let slice = Array(wanted[batch..<min(batch + 1000, wanted.count)])
            if let out = try? await api.check(hashes: slice, assetIds: []) { known.merge(out.known) { a, _ in a } }
        }
        for rec in hashOnly {
            let h = hashes(rec)
            guard !h.isEmpty, h.allSatisfy({ known[$0] != nil }), let uid = known[h[0]],
                  let item = try? await api.item(uid: uid) else { continue }   // a vault item answers 403 to a locked device: not confirmed
            byHash[rec.assetId] = item.deleted_at != nil
        }
        var stored: [SyncedRow] = []
        var trashed: [SyncedRow] = []
        for rec in mine {
            guard let asset = present[rec.assetId] else { continue }
            let id = rec.assetId
            let onServer = server.assets[id] != nil && (server.pending[id] ?? []).isEmpty
            if (onServer && server.trashed.contains(id)) || byHash[id] == true {
                rec.trashedOnServer = true
                trashed.append(SyncedRow(asset: asset, rec: rec))
            } else if onServer || byHash[id] == false {
                rec.trashedOnServer = false
                stored.append(SyncedRow(asset: asset, rec: rec))
            }
        }
        try? ctx.save()
        return (stored, trashed)
    }

    func markDeleted(_ records: [SyncRecord]) {
        let ctx = SyncStore.shared.container.mainContext
        for r in records { r.deletedFromLibrary = true; r.updatedAt = Date() }
        try? ctx.save()
    }
}

/// A short stretch of background time for work that must reach a hand-off point (an export that
/// ends with a task in the background session). Ended as soon as the work is handed over, or by
/// iOS when the time runs out.
@MainActor
private final class BackgroundTime {
    private var id: UIBackgroundTaskIdentifier = .invalid

    init(_ name: String) {
        id = UIApplication.shared.beginBackgroundTask(withName: name) { [weak self] in
            self?.end()
        }
    }

    func end() {
        guard id != .invalid else { return }
        UIApplication.shared.endBackgroundTask(id)
        id = .invalid
    }
}
