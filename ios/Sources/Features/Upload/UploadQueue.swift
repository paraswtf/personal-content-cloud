import Foundation
import SwiftUI
import SwiftData
import Photos
import PhotosUI
import UniformTypeIdentifiers

/// The Upload screen's queue (the website's lib/uploader.ts engine, on the phone).
///
/// Files come from the Photos picker, Files, a drop or the camera. Each one is a SyncRecord with `queuedAt` set, so
/// the queue survives the app being closed. A photo the library can hand over by its asset id uses the asset's own
/// record and is exported as the phone's sync exports it (the original files, both halves of a Live Photo), so the
/// sync knows it is sent; anything else is copied into the app and gets a record of its own ("upload:<uuid>").
/// Every file goes through the background uploader (BackgroundUploader, docs/ARCHITECTURE.md section 28): once
/// its upload has started it keeps going while the app is in the background, as the sync's do. At most
/// `maxInFlight` files are on the wire at once, each watched until it ends. When a file is on the server, the queue
/// reads the server's answer until it is final (added, already in your library, not accepted, processing failed).
/// Opening the screen continues whatever the queue left when the app was closed.
@MainActor
@Observable
final class UploadQueue {
    static let shared = UploadQueue()
    static let maxInFlight = 3
    static let finalStatuses: Set<String> = ["stored", "duplicate", "rejected", "error", "cancelled"]

    /// The server's limit per file (GET /api/config); the server's default until it is read.
    var maxUploadBytes: Int64 = 16 * 1024 * 1024 * 1024
    /// The server's settings were read in this session (the Upload screen reads them when it opens).
    var configLoaded = false
    /// What the last add did ("2 files added to the queue."), shown under the buttons for a while.
    var notice: String?
    /// Files the Photos picker is still handing over.
    var importing = 0
    /// Records whose files are being exported from Photos or hashed (the "Checking" phase).
    private(set) var checking: Set<String> = []
    /// How much of a copied file has been hashed (0...1), while it is checked ("Checking 40%").
    private(set) var checkProgress: [String: Double] = [:]
    /// Records the queue had already started an upload for (Resuming instead of Uploading).
    private(set) var resumed: Set<String> = []

    @ObservationIgnored private var inFlight: Set<String> = []
    @ObservationIgnored private var poller: Task<Void, Never>?
    @ObservationIgnored private var pollRound = 0
    @ObservationIgnored private var noticeTimer: Task<Void, Never>?
    /// Files the uploader stopped while the phone was offline: they wait with their reason and are not started again
    /// until the connection is back (backOnline).
    @ObservationIgnored private var heldOffline: Set<String> = []

    /// The reason a file gets when its upload stopped for the connection and gave none of its own.
    nonisolated static let connectionLost = "Connection lost. Check your internet and try again."
    /// The starts of the reasons that mean the network stopped the upload (the queue's own and the uploader's stalls).
    nonisolated static let networkReasons = [connectionLost, "Cannot reach the server", "The transfer did not start"]

    /// A failed file the queue starts again when the connection comes back: it failed on the way (the server never
    /// gave it a final answer) and for the network, as the uploader recorded (`known`) or as its reason says.
    nonisolated static func failedForNetwork(status: String, serverStatus: String?, error: String?, known: Bool) -> Bool {
        guard status == "failed", serverStatus == nil else { return false }
        if known { return true }
        guard let error else { return false }
        return networkReasons.contains { error.hasPrefix($0) }
    }

    private var ctx: ModelContext { SyncStore.shared.container.mainContext }

    /// The queued records, oldest first.
    func queued() -> [SyncRecord] {
        let d = FetchDescriptor<SyncRecord>(predicate: #Predicate<SyncRecord> { $0.queuedAt != nil },
                                            sortBy: [SortDescriptor(\.queuedAt)])
        return (try? ctx.fetch(d)) ?? []
    }

    private func record(_ id: String) -> SyncRecord? {
        var d = FetchDescriptor<SyncRecord>(predicate: #Predicate<SyncRecord> { $0.assetId == id })
        d.fetchLimit = 1
        return try? ctx.fetch(d).first
    }

    private func save() {
        try? ctx.save()
    }

    // MARK: what a row shows

    func phase(_ rec: SyncRecord) -> UploadPhase {
        UploadPhase.of(status: rec.status, serverStatus: rec.serverStatus, checking: checking.contains(rec.assetId))
    }

    /// `progress` is the upload's; a file being checked shows how much of it is hashed instead.
    func info(_ rec: SyncRecord, progress: Double?) -> UploadRowInfo {
        let current = self.phase(rec)
        let shown = current == .checking ? checkProgress[rec.assetId] : progress
        return UploadRowInfo(phase: current, serverStatus: rec.serverStatus, bytes: max(rec.bytes, 0), progress: shown,
                             skipped: rec.status == "skipped")
    }

    // MARK: adding

    struct AddResult {
        var added = 0
        var rejected = 0
        var skipped = 0

        static func + (a: AddResult, b: AddResult) -> AddResult {
            AddResult(added: a.added + b.added, rejected: a.rejected + b.rejected, skipped: a.skipped + b.skipped)
        }
    }

    /// Copies already in the app (Files, a picker copy, a drop, the camera; `origin` file or camera). `unreadable`
    /// counts chosen files that could not be copied, which the notice counts as skipped.
    func add(_ files: [UploadIncoming], origin: String, unreadable: Int = 0) {
        var result = addFiles(files, origin: origin)
        result.rejected += unreadable
        say(result)
        pump()
    }

    /// Photos and videos chosen in the Photos picker.
    func add(picked items: [PhotosPickerItem]) async {
        guard !items.isEmpty else { return }
        // The library's own record only when the sync is not exporting at the same moment (it could be this asset).
        let library = PhotoLibrary.hasAccess && !SyncEngine.shared.running
        var assets: [PHAsset] = []
        var copies: [UploadIncoming] = []
        var unreadable = 0
        importing += items.count
        for item in items {
            if library, let id = item.itemIdentifier, let asset = PhotoLibrary.assets(ids: [id]).first {
                assets.append(asset)
            } else if let file = try? await item.loadTransferable(type: UploadPickedFile.self) {
                copies.append(UploadIncoming(url: file.url, name: file.name))
            } else {
                unreadable += 1
            }
            importing = max(0, importing - 1)
        }
        var result = addAssets(assets) + addFiles(copies, origin: "file")
        result.rejected += unreadable
        say(result)
        await resume()   // an asset the sync had left half sent continues too
    }

    private func addFiles(_ files: [UploadIncoming], origin: String) -> AddResult {
        var result = AddResult()
        let listed = queued()
        for file in files {
            let size = (try? FileManager.default.attributesOfItem(atPath: file.url.path)[.size] as? Int64) ?? 0
            // the website's "already in the list": the same name and size
            if listed.contains(where: { $0.isQueueFile && $0.filename == file.name && $0.bytes == size }) {
                result.skipped += 1
                UploadIncoming.discard(file.url)
                continue
            }
            let id = "upload:" + UUID().uuidString
            let kind = UploadKind.of(name: file.name)
            let rec = SyncRecord(assetId: id, filename: file.name, kind: kind.recordKind, bytes: size, createdAt: Date())
            rec.origin = origin
            rec.queuedAt = Date()
            if let reason = UploadText.rejection(size: size, limit: maxUploadBytes) {
                rec.status = "skipped"
                rec.error = reason
                result.rejected += 1
                UploadIncoming.discard(file.url)
            } else {
                let target = UploadPaths.assetDir(id).appendingPathComponent(UploadIncoming.safeName(file.name))
                do {
                    try FileManager.default.moveItem(at: file.url, to: target)
                    UploadIncoming.discard(file.url)
                    let mime = UTType(filenameExtension: UploadKind.fileExtension(file.name))?.preferredMIMEType
                    rec.files = [UploadFile(resource: "file", path: UploadPaths.relative(target), filename: file.name,
                                            mime: mime, bytes: size, sha256: "", uploadId: nil, done: false, itemUid: nil)]
                    result.added += 1
                    Task { await UploadThumbs.make(id, from: target, kind: kind) }
                } catch {
                    rec.status = "failed"
                    rec.error = "This file could not be read. It may have been moved or deleted."
                    result.added += 1
                }
            }
            ctx.insert(rec)
        }
        save()
        return result
    }

    /// Library assets chosen in the picker: their own records, sent the sync's way.
    private func addAssets(_ assets: [PHAsset]) -> AddResult {
        var result = AddResult()
        for asset in assets {
            let id = asset.localIdentifier
            let rec = record(id) ?? {
                let r = SyncRecord(assetId: id, filename: "", kind: asset.mediaType == .video ? "video" : "image",
                                   bytes: 0, createdAt: asset.creationDate ?? Date())
                self.ctx.insert(r)
                return r
            }()
            if rec.queuedAt != nil {
                result.skipped += 1
                continue
            }
            rec.queuedAt = Date()
            rec.serverStatus = nil
            rec.thumbRef = nil
            rec.itemVault = false
            switch rec.status {
            case "synced":
                rec.serverStatus = "duplicate"   // the phone's sync sent it earlier: already in your library
            case "uploading", "uploaded":
                break   // on its way already, or sent: the queue watches it or asks the server
            default:
                rec.status = "pending"
                rec.error = nil
            }
            if rec.filename.isEmpty {
                rec.filename = PHAssetResource.assetResources(for: asset).first?.originalFilename ?? "Photo"
            }
            result.added += 1
            Task { await UploadThumbs.make(id, asset: asset) }
        }
        save()
        return result
    }

    private func say(_ result: AddResult) {
        let text = UploadText.notice(added: result.added, rejected: result.rejected, skipped: result.skipped)
        notice = text.isEmpty ? nil : text
        noticeTimer?.cancel()
        noticeTimer = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 8_000_000_000)
            if !Task.isCancelled { self?.notice = nil }
        }
    }

    // MARK: sending

    /// Starts waiting files while fewer than `maxInFlight` are on the wire, and reads the server's answers for
    /// files it is storing.
    func pump() {
        guard APIClient.current() != nil else { return }
        // Held files wait only while the phone is offline: online (the screen opened again after the connection came
        // back elsewhere, say) they go on.
        if SyncEngine.shared.online { heldOffline.removeAll() }
        let all = queued()
        for rec in all where rec.status == "pending" && !inFlight.contains(rec.assetId) && !heldOffline.contains(rec.assetId) {
            if inFlight.count >= Self.maxInFlight { break }
            begin(rec)
        }
        if all.contains(where: waitsForServer) { startPolling() }
    }

    /// The connection is back: files held while offline go on, and files that failed for the network start again (the
    /// website's uploader waits for the connection instead of failing them).
    func backOnline() {
        heldOffline.removeAll()
        let known = BackgroundUploader.shared.networkFailures
        for rec in queued() where Self.failedForNetwork(status: rec.status, serverStatus: rec.serverStatus,
                                                         error: rec.error, known: known.contains(rec.assetId)) {
            retry(rec)
        }
        pump()
    }

    /// Opening the screen: continues what the queue left when the app was closed. A file whose request is still in
    /// the background session is watched again; one whose transfer stopped starts again, and the server's answer
    /// to its create (the same upload) or its offset makes it continue where it stopped.
    func resume() async {
        let uploader = BackgroundUploader.shared
        // Copies no add took (the app stopped between copying a file and queueing it) go after a day.
        Task.detached(priority: .utility) { UploadIncoming.sweep(olderThan: 24 * 3600) }
        let live = await uploader.liveAssets()
        // Copied files taken off the list earlier go now, when nothing works on them any more.
        let gone = FetchDescriptor<SyncRecord>(predicate: #Predicate<SyncRecord> { $0.queuedAt == nil && $0.origin != "library" })
        for rec in (try? ctx.fetch(gone)) ?? []
        where !live.contains(rec.assetId) && !inFlight.contains(rec.assetId) && !uploader.isDriving(rec.assetId) {
            UploadPaths.removeAsset(rec.assetId)
            ctx.delete(rec)
        }
        for rec in queued() where rec.status == "uploading" && !inFlight.contains(rec.assetId) {
            if live.contains(rec.assetId) || uploader.isDriving(rec.assetId) {
                uploader.showSending(rec)
                watch(rec.assetId)
            } else {
                rec.status = "pending"
                resumed.insert(rec.assetId)
            }
        }
        save()
        pump()
    }

    private func begin(_ rec: SyncRecord) {
        let id = rec.assetId
        inFlight.insert(id)
        checking.insert(id)
        if rec.uploadId != nil { resumed.insert(id) }
        Task { @MainActor [self] in
            var claimed = false
            defer {
                if claimed { SyncEngine.shared.release(id) }
                self.inFlight.remove(id)
                self.checking.remove(id)
                self.checkProgress[id] = nil
                self.pump()
            }
            guard let api = APIClient.current() else { return }
            do {
                if rec.isQueueFile {
                    try await hash(rec)
                } else {
                    // The phone's sync is exporting or starting this photo right now: follow its upload instead of
                    // exporting a second copy (an export starts by emptying the photo's folder).
                    guard SyncEngine.shared.claim(id) else {
                        await self.follow(id)
                        return
                    }
                    claimed = true
                    guard let asset = PhotoLibrary.assets(ids: [id]).first else {
                        throw UploadQueueError("The photo is no longer in the library.")
                    }
                    try await SyncEngine.shared.prepare(rec, asset: asset, api: api)
                }
            } catch {
                guard rec.status == "pending" else { return }   // cancelled or removed meanwhile
                rec.status = "failed"
                rec.error = (error as? UploadQueueError)?.text ?? "This file could not be read. It may have been moved or deleted."
                rec.updatedAt = Date()
                save()
                return
            }
            // Cancelled or taken off the list while it was checked.
            guard rec.status == "pending", rec.queuedAt != nil, self.record(id) != nil else { return }
            self.checking.remove(id)
            self.checkProgress[id] = nil
            BackgroundUploader.shared.start(rec)
            self.save()
            await BackgroundUploader.shared.wait(for: id, until: nil)
            self.settle(id)
        }
    }

    /// The upload left the wire. One the uploader only paused (it could not reach the server, or a request never
    /// started) fails here with its reason, so the list says what happened and Retry continues it; the queue does
    /// not start it again by itself. While the phone is offline it waits instead, with its reason, and goes on when
    /// the connection is back (backOnline).
    private func settle(_ id: String) {
        resumed.remove(id)
        guard let rec = record(id), rec.queuedAt != nil, rec.status == "pending" else { return }
        if !SyncEngine.shared.online {
            if rec.error == nil { rec.error = Self.connectionLost }
            heldOffline.insert(id)
            rec.updatedAt = Date()
            save()
            return
        }
        rec.status = "failed"
        if rec.error == nil { rec.error = Self.connectionLost }
        rec.updatedAt = Date()
        save()
    }

    private func watch(_ id: String) {
        inFlight.insert(id)
        Task { @MainActor [self] in
            await BackgroundUploader.shared.wait(for: id, until: nil)
            self.inFlight.remove(id)
            self.settle(id)
            self.pump()
        }
    }

    /// The phone's sync holds this photo: waits until the sync lets it go, then watches the upload it started. A photo
    /// the sync has sent is read from the server by the next pump; one it left waiting is begun by the next pump.
    private func follow(_ id: String) async {
        while SyncEngine.shared.claimed.contains(id) {
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        guard let rec = record(id), rec.queuedAt != nil else { return }
        if rec.status == "uploading" || BackgroundUploader.shared.isDriving(id) {
            checking.remove(id)
            await BackgroundUploader.shared.wait(for: id, until: nil)
            settle(id)
        }
    }

    /// The SHA-256 of each copied file, which the server uses to keep one copy of a file it already has.
    private func hash(_ rec: SyncRecord) async throws {
        var files = rec.files
        guard !files.isEmpty else { throw UploadQueueError("This file could not be read. It may have been moved or deleted.") }
        let id = rec.assetId
        let total = Double(max(1, files.reduce(Int64(0)) { $0 + max(0, $1.bytes) }))
        var before: Int64 = 0
        for i in files.indices {
            let size = max(0, files[i].bytes)
            defer { before += size }
            guard files[i].sha256.isEmpty else { continue }
            let url = UploadPaths.url(files[i].path)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw UploadQueueError("This file could not be read. It may have been moved or deleted.")
            }
            let done = Double(before)
            checkProgress[id] = done / total
            files[i].sha256 = try await Task.detached(priority: .utility) {
                var shown = -1
                return try PhotoLibrary.sha256(of: url) { share in
                    // Whole percents only, so the row redraws at most a hundred times.
                    let percent = Int(((done + share * Double(size)) / total * 100).rounded(.down))
                    guard percent != shown else { return }
                    shown = percent
                    Task { @MainActor in UploadQueue.shared.showChecked(id, Double(percent) / 100) }
                }
            }.value
        }
        rec.files = files
        rec.sha256 = files.first?.sha256
        rec.updatedAt = Date()
    }

    /// A hashing step's share, while the file is still being checked.
    fileprivate func showChecked(_ id: String, _ share: Double) {
        guard checking.contains(id) else { return }
        if share > (checkProgress[id] ?? -1) { checkProgress[id] = share }
    }

    // MARK: the server's answer

    private func waitsForServer(_ rec: SyncRecord) -> Bool {
        (rec.status == "uploaded" || rec.status == "synced") && !Self.finalStatuses.contains(rec.serverStatus ?? "")
    }

    private func startPolling() {
        guard poller == nil else { return }
        pollRound += 1
        let round = pollRound
        poller = Task { @MainActor [weak self] in
            while let self, !Task.isCancelled {
                let waiting = self.queued().filter(self.waitsForServer)
                if waiting.isEmpty { break }
                await self.readServer(waiting)
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
            // Only this round's poller: a newer one may have started after stopPolling().
            if let self, self.pollRound == round { self.poller = nil }
        }
    }

    /// The Upload screen went away: the server's answers are read again when it opens (resume() pumps), as the
    /// website reads them only while its page is open, so a file the server never finishes does not keep the phone
    /// asking every two seconds.
    func stopPolling() {
        poller?.cancel()
        poller = nil
    }

    /// GET /api/uploads?ids=... for files the server is storing: their status, item and thumb.
    private func readServer(_ records: [SyncRecord]) async {
        var byUpload: [String: SyncRecord] = [:]
        for rec in records {
            // The still of a Live Photo speaks for the asset; any other record has one file.
            let file = rec.files.first(where: { $0.resource != "video" }) ?? rec.files.first
            if let upload = file?.uploadId { byUpload[upload] = rec }
        }
        let ids = Array(byUpload.keys)
        for start in stride(from: 0, to: ids.count, by: 100) {
            let slice = ids[start..<min(start + 100, ids.count)]
            guard let list = try? await Cloud.api().manyUploads(ids: slice.joined(separator: ",")) else { return }
            var stored: [String] = []
            for up in list.uploads {
                guard let rec = byUpload[up.id] else { continue }
                let status = up.status.rawValue
                rec.serverStatus = status
                if let thumb = up.itemThumbUrl { rec.thumbRef = thumb }
                rec.itemVault = up.itemNsfw
                if let uid = up.itemUid { rec.uid = uid }
                if let message = up.message { rec.error = message }
                if Self.finalStatuses.contains(status) {
                    // The server's answer is final: a copy is no longer needed, except to try a processing error again.
                    if rec.isQueueFile && status != "error" { UploadPaths.removeAsset(rec.assetId) }
                    if status == "stored" || status == "duplicate", let uid = up.itemUid { stored.append(uid) }
                }
            }
            save()
            if !stored.isEmpty { CloudEvents.post() }
        }
    }

    // MARK: the row's buttons

    func cancel(_ rec: SyncRecord) {
        guard phase(rec).canCancel else { return }
        let id = rec.assetId
        // On the wire, or half on the server from a transfer that stopped (its upload id is kept to continue it): the
        // uploader stops its requests and cancels the server's upload too.
        if rec.status == "uploading" || BackgroundUploader.shared.isDriving(id) || rec.uploadId != nil {
            BackgroundUploader.shared.cancel(id)
        } else {
            rec.status = "cancelled"
            rec.error = nil
            rec.updatedAt = Date()
            save()
        }
        checking.remove(id)
    }

    func retry(_ rec: SyncRecord) {
        guard info(rec, progress: nil).canRetry else { return }
        var files = rec.files
        // A processing error on the server or a cancel needs a fresh upload; a failure on the way continues.
        if rec.serverStatus == "error" || rec.status == "cancelled" {
            files = files.map { var f = $0; f.uploadId = nil; f.done = false; f.itemUid = nil; return f }
            rec.clearTransfer()
        }
        if rec.isQueueFile, files.isEmpty || files.contains(where: { !FileManager.default.fileExists(atPath: UploadPaths.url($0.path).path) }) {
            rec.status = "failed"
            rec.error = "This file is no longer on the phone. Choose it again."
            save()
            return
        }
        rec.files = files
        rec.status = "pending"
        rec.error = nil
        rec.serverStatus = nil
        rec.thumbRef = nil
        rec.itemVault = false
        rec.updatedAt = Date()
        save()
        pump()
    }

    /// Off the list (cancelling it first when it is still on its way). A copied file's copy goes now and its record
    /// when the screen next opens (a request of it may still be answered meanwhile); a library asset keeps its
    /// record, which the phone's sync uses.
    func remove(_ rec: SyncRecord) {
        if phase(rec).canCancel { cancel(rec) }
        let id = rec.assetId
        rec.queuedAt = nil
        UploadThumbs.remove(id)
        resumed.remove(id)
        if rec.isQueueFile { UploadPaths.removeAsset(id) }
        save()
    }

    /// Every file that needs nothing more (done, failed, cancelled) leaves the list.
    func clearFinished() {
        for rec in queued() where phase(rec).isFinished { remove(rec) }
    }
}

/// A reason the queue gives for a file that could not be sent.
struct UploadQueueError: Error {
    let text: String
    init(_ text: String) { self.text = text }
}
