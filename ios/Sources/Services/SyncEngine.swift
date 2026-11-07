import Foundation
import Photos
import SwiftData
import Network
import UIKit

/// The sync run: find new assets, skip what the server already holds, upload the rest through the
/// chunked protocol, and record every outcome. One run at a time.
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

    func cancel() {
        task?.cancel()
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

        let ctx = SyncStore.shared.newContext()
        let assets = PhotoLibrary.assets(since: AppSettings.syncSince, includeVideos: AppSettings.includeVideos)
        statusLine = "Checking \(assets.count) items"
        let known = (try? ctx.fetch(FetchDescriptor<SyncRecord>())) ?? []
        var byId = Dictionary(uniqueKeysWithValues: known.map { ($0.assetId, $0) })
        let todo = assets.filter { byId[$0.localIdentifier]?.status != "synced" }
        SyncStore.shared.log("\(assets.count) items in the library, \(todo.count) to check")

        // Ask the server, 200 asset ids at a time, which it already holds.
        var serverHas: [String: String] = [:]
        for batch in stride(from: 0, to: todo.count, by: 200) {
            if Task.isCancelled { return }
            let ids = todo[batch..<min(batch + 200, todo.count)].map(\.localIdentifier)
            if let out = try? await api.check(hashes: [], assetIds: ids) {
                serverHas.merge(out.assets) { a, _ in a }
            }
        }

        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("pcc-sync", isDirectory: true)
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        var done = 0
        for asset in todo {
            if Task.isCancelled { break }
            if let deadline, Date() > deadline { SyncStore.shared.log("Background time is up, continuing later"); break }
            if let wait = allowedNow(manual: manual) { statusLine = wait; break }
            done += 1
            progress = Double(done) / Double(max(1, todo.count))
            let rec = byId[asset.localIdentifier] ?? {
                let r = SyncRecord(assetId: asset.localIdentifier, filename: "", kind: asset.mediaType == .video ? "video" : "image",
                                   bytes: 0, createdAt: asset.creationDate ?? Date())
                ctx.insert(r)
                byId[asset.localIdentifier] = r
                return r
            }()
            if let uid = serverHas[asset.localIdentifier] {
                rec.uid = uid; rec.status = "synced"; rec.updatedAt = Date()
                try? ctx.save()
                continue
            }
            statusLine = "Uploading \(done) of \(todo.count)"
            do {
                try await upload(asset: asset, record: rec, api: api, tmp: tmp)
                rec.status = "synced"; rec.error = nil
            } catch {
                rec.status = "failed"; rec.error = error.localizedDescription
                SyncStore.shared.log("\(rec.filename.isEmpty ? asset.localIdentifier : rec.filename): \(error.localizedDescription)", level: "error")
            }
            rec.updatedAt = Date()
            try? ctx.save()
        }
        try? FileManager.default.removeItem(at: tmp)
        progress = 1
    }

    /// Exports the asset's original files and sends each one; a Live Photo's two files share the
    /// asset identifier as pair key, so the server pairs them (section 20).
    private func upload(asset: PHAsset, record: SyncRecord, api: APIClient, tmp: URL) async throws {
        let files = try await PhotoLibrary.export(asset, to: tmp)
        defer { for f in files { try? FileManager.default.removeItem(at: f.url) } }
        guard let still = files.first(where: { !$0.isClip }) else { throw APIError(status: 0, detail: "No original file") }
        record.filename = still.filename
        record.bytes = still.bytes
        if asset.mediaSubtypes.contains(.photoLive) { record.kind = "live" }
        for file in files {
            let url = file.url
            let sha = try await Task.detached(priority: .utility) { try PhotoLibrary.sha256(of: url) }.value
            if !file.isClip { record.sha256 = sha }
            let uid = try await send(file: file, sha256: sha, pairKey: asset.localIdentifier, api: api)
            if file.isClip { record.clipUid = uid } else { record.uid = uid }
        }
        SyncStore.shared.log("Synced \(still.filename)")
    }

    /// The chunked upload protocol: create (a known hash answers at once), PUT chunks, complete,
    /// then wait for the item. Returns the item uid.
    private func send(file: PhotoLibrary.ExportedFile, sha256: String, pairKey: String, api: APIClient) async throws -> String? {
        var up = try await api.createUpload(filename: file.filename, size: file.bytes, mime: file.mime, sha256: sha256, pairKey: pairKey)
        if up.status == "duplicate" { return up.item_uid }
        let handle = try FileHandle(forReadingFrom: file.url)
        defer { try? handle.close() }
        var offset: Int64 = up.received_bytes
        let chunk = max(1, up.chunk_size)
        while offset < file.bytes {
            if Task.isCancelled { throw CancellationError() }
            try handle.seek(toOffset: UInt64(offset))
            guard let data = try handle.read(upToCount: chunk), !data.isEmpty else { break }
            var attempt = 0
            while true {
                do {
                    let out = try await api.putChunk(id: up.id, offset: offset, data: data)
                    offset = out.received_bytes
                    break
                } catch {
                    attempt += 1
                    if attempt >= 3 { throw error }
                    try await Task.sleep(nanoseconds: UInt64(attempt) * 2_000_000_000)
                    if let fresh = try? await api.uploadStatus(id: up.id) { offset = fresh.received_bytes }
                }
            }
        }
        up = try await api.complete(id: up.id)
        var waited = 0
        while !["stored", "duplicate", "rejected", "error", "cancelled"].contains(up.status) && waited < 120 {
            try await Task.sleep(nanoseconds: 2_000_000_000)
            waited += 2
            up = try await api.uploadStatus(id: up.id)
        }
        if up.status == "rejected" || up.status == "error" {
            throw APIError(status: 0, detail: up.message ?? "The server rejected the file")
        }
        return up.item_uid
    }

    // MARK: delete synced

    /// The synced assets still in the library, confirmed by the server right now.
    struct SyncedRow: Identifiable {
        let asset: PHAsset
        let rec: SyncRecord
        var id: String { rec.assetId }
    }

    func confirmedSynced() async -> [SyncedRow] {
        guard let api = APIClient.current() else { return [] }
        let ctx = SyncStore.shared.container.mainContext
        let synced = ((try? ctx.fetch(FetchDescriptor<SyncRecord>(predicate: #Predicate { $0.status == "synced" && !$0.deletedFromLibrary }))) ?? [])
        let assets = PhotoLibrary.assets(ids: synced.map(\.assetId))
        let present = Dictionary(uniqueKeysWithValues: assets.map { ($0.localIdentifier, $0) })
        var confirmed: Set<String> = []
        let ids = synced.map(\.assetId)
        for batch in stride(from: 0, to: ids.count, by: 200) {
            let slice = Array(ids[batch..<min(batch + 200, ids.count)])
            if let out = try? await api.check(hashes: [], assetIds: slice) {
                confirmed.formUnion(out.assets.keys)
            }
        }
        return synced.compactMap { rec in
            guard let asset = present[rec.assetId], confirmed.contains(rec.assetId) else { return nil }
            return SyncedRow(asset: asset, rec: rec)
        }
    }

    func markDeleted(_ records: [SyncRecord]) {
        let ctx = SyncStore.shared.container.mainContext
        for r in records { r.deletedFromLibrary = true; r.updatedAt = Date() }
        try? ctx.save()
    }
}
