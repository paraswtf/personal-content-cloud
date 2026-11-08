import Foundation
import SwiftData
import UIKit

/// What a background task is for. It travels in the task's `taskDescription`, which the system keeps
/// with the task, so a process relaunched into the delegate knows which record and step a finished
/// task belongs to.
struct UploadTaskTag: Codable {
    let asset: String           // the record's asset id
    let step: String            // create, chunk, complete
    let resource: String        // still or video
    let upload: String?         // the upload id (nil for a create)
    let offset: Int64           // the chunk's offset (the record's nextOffset when the task started)
    let length: Int64           // the chunk's length (0 for create and complete)
    let body: String?           // the request body file to remove afterwards, relative to UploadPaths.root

    var text: String {
        (try? JSONEncoder().encode(self)).flatMap { String(data: $0, encoding: .utf8) } ?? ""
    }

    static func from(_ text: String?) -> UploadTaskTag? {
        guard let data = text?.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(UploadTaskTag.self, from: data)
    }
}

/// The JSON body of `POST /api/uploads` (UploadCreate, sections 6 and 28).
private struct CreateBody: Encodable {
    let filename: String
    let size: Int64
    let mime: String?
    let sha256: String
    let source = "api"
    let client_id = "ios"
    let pair_key: String
    let resource: String
}

/// Background transfers for the device uploads of docs/ARCHITECTURE.md section 28.
///
/// Every request of the upload protocol (create, each chunk, complete) is an upload task of one
/// background URLSession (`in.rivant.pcc.upload`) whose body is a file. The system carries the
/// task while the app is suspended and relaunches the app into this delegate when it has to. The
/// delegate moves the asset's `SyncRecord` one step and schedules the next request of the same file,
/// so a file keeps flowing without the app. The state lives in the record and in each task's
/// description, never only in memory, because a relaunched process starts with nothing else.
///
/// One asset is in flight at a time, one request at a time. Errors:
/// - transport failures, 408, 429 and 5xx retry the same request after 1, 2, 4, 8, 16, 30 s
///   (`earliestBeginDate`), at most eight attempts in a row;
/// - 409 offset_mismatch continues from the server's offset; another 409 reads the upload's state;
/// - 404 on a chunk or complete creates the upload again (create is idempotent per asset, half
///   and size, so the server answers with its own row while it has one);
/// - 413 and 507 fail the asset with the server's message.
@MainActor
final class BackgroundUploader: NSObject, ObservableObject {
    static let identifier = "in.rivant.pcc.upload"
    static let shared = BackgroundUploader()
    static let backoff: [TimeInterval] = [1, 2, 4, 8, 16, 30]
    static let maxAttempts = 8
    static let defaultChunk = 8 * 1024 * 1024

    /// The file being sent and how far it got, for the Sync screen.
    @Published private(set) var current: String?
    @Published private(set) var fileProgress: Double = 0

    /// Handed over by the AppDelegate when iOS relaunches the app for this session's events; called
    /// once the delegate has seen them all.
    var backgroundCompletion: (() -> Void)?

    /// The configuration follows the Wi-Fi only and charging only options.
    struct Options: Equatable {
        let cellular: Bool
        let discretionary: Bool

        static var wanted: Options {
            Options(cellular: !AppSettings.wifiOnly, discretionary: AppSettings.wifiOnly || AppSettings.chargingOnly)
        }
    }

    private var session: URLSession!
    private var applied = Options.wanted
    private var reconfigurePending = false
    private var invalidating = false
    private var bodies: [Int: Data] = [:]                       // task id -> response body so far
    private var waiters: [String: [UUID: CheckedContinuation<Void, Never>]] = [:]
    private var currentBytes: Int64 = 0

    private override init() {
        super.init()
        session = makeSession()
    }

    private func makeSession() -> URLSession {
        let want = Options.wanted
        let cfg = URLSessionConfiguration.background(withIdentifier: Self.identifier)
        cfg.sessionSendsLaunchEvents = true
        cfg.allowsCellularAccess = want.cellular
        cfg.isDiscretionary = want.discretionary
        cfg.httpShouldSetCookies = false
        cfg.httpCookieAcceptPolicy = .never
        cfg.timeoutIntervalForRequest = 120
        cfg.timeoutIntervalForResource = 7 * 24 * 3600
        applied = want
        return URLSession(configuration: cfg, delegate: self, delegateQueue: .main)
    }

    /// Reconnects to the background session (at launch, and when iOS relaunches the app for its
    /// events), so finished tasks are delivered to this delegate.
    func activate() {
        _ = session
        sweepChunks()
    }

    /// Removes chunk and body files that no task of the session refers to any more (left behind when
    /// the app was force-quit between a task's end and the record's update), leaving young files
    /// alone: a chunk being cut for the next task has no task yet.
    private func sweepChunks() {
        session.getAllTasks { tasks in
            let live = Set(tasks.compactMap { UploadTaskTag.from($0.taskDescription)?.body })
            Task.detached(priority: .utility) { Self.removeChunkFiles(except: live) }
        }
    }

    private nonisolated static func removeChunkFiles(except live: Set<String>) {
        let fm = FileManager.default
        let dir = UploadPaths.chunks
        guard let names = try? fm.contentsOfDirectory(atPath: dir.path) else { return }
        let cutoff = Date(timeIntervalSinceNow: -300)
        for name in names where name != "empty.body" {
            let url = dir.appendingPathComponent(name)
            let modified = (try? fm.attributesOfItem(atPath: url.path)[.modificationDate] as? Date) ?? .distantPast
            if !live.contains(UploadPaths.relative(url)) && modified < cutoff {
                try? fm.removeItem(at: url)
            }
        }
    }

    /// The Wi-Fi only or charging only option changed. A background session cannot change its
    /// configuration, so the session is replaced at the next step boundary; requests already pick
    /// up the Wi-Fi option through their own `allowsCellularAccess`.
    func settingsChanged() {
        guard Options.wanted != applied, !invalidating else { return }
        reconfigurePending = true
        Task { @MainActor in
            let tasks = await self.session.allTasks
            if tasks.isEmpty { self.replaceSession() }
        }
    }

    private func replaceSession() {
        guard reconfigurePending, !invalidating else { return }
        reconfigurePending = false
        invalidating = true
        session.finishTasksAndInvalidate()   // urlSession(_:didBecomeInvalidWithError:) makes the new one
    }

    private func sessionInvalidated() {
        session = makeSession()
        invalidating = false
        // continue what was waiting for the new session
        for rec in records(status: "uploading") { scheduleNext(rec) }
    }

    // MARK: what the engine calls

    /// The assets with a task in the session right now (in flight or waiting for its backoff).
    func liveAssets() async -> Set<String> {
        let tasks = await session.allTasks
        return Set(tasks.filter { $0.state == .running || $0.state == .suspended }
            .compactMap { UploadTaskTag.from($0.taskDescription)?.asset })
    }

    /// Starts the upload of a prepared record (its `files` are exported and hashed), or continues
    /// it from the state the record holds.
    func start(_ rec: SyncRecord) {
        begin(rec)
        scheduleNext(rec)
    }

    /// Continues an interrupted upload from the server's offset (`pending` of the sync check).
    func resume(_ rec: SyncRecord, at offset: Int64) {
        begin(rec)
        rec.nextOffset = offset
        scheduleNext(rec)
    }

    /// Continues an interrupted upload from its state on the server (`GET /api/uploads/{id}`);
    /// nil means the server no longer has it, so the file is created again.
    func resume(_ rec: SyncRecord, with up: UploadOut?) {
        begin(rec)
        if let up {
            applyUpload(rec, up)
        } else {
            forgetUpload(rec)
            scheduleNext(rec)
        }
    }

    /// Waits until the asset's upload ends (sent, failed or interrupted), the deadline passes or the
    /// calling task is cancelled. The transfer itself goes on in the background either way.
    func wait(for assetId: String, until deadline: Date?) async {
        guard !Task.isCancelled, let rec = record(assetId), rec.status == "uploading" else { return }
        let token = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                waiters[assetId, default: [:]][token] = cont
                if let deadline {
                    let wait = max(0, deadline.timeIntervalSinceNow)
                    Task { @MainActor [weak self] in
                        try? await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000))
                        self?.release(assetId, token: token)
                    }
                }
            }
        } onCancel: {
            Task { @MainActor [weak self] in self?.release(assetId, token: token) }
        }
    }

    /// The Stop button: cancels the tasks in flight. The records keep their state, and the next run
    /// continues them from the server's offset.
    func pause() {
        session.getAllTasks { tasks in tasks.forEach { $0.cancel() } }
        for rec in records(status: "uploading") {
            rec.status = "pending"
            rec.updatedAt = Date()
            releaseAll(rec.assetId)
        }
        save()
        current = nil
    }

    // MARK: the state machine

    private func begin(_ rec: SyncRecord) {
        rec.status = "uploading"
        rec.error = nil
        rec.attempts = 0
        rec.updatedAt = Date()
        save()
    }

    /// Sends the next request of the record's first unfinished file: create when it has no upload
    /// yet, the chunk at `nextOffset`, or complete once every byte is there.
    private func scheduleNext(_ rec: SyncRecord, after delay: TimeInterval = 0) {
        guard !invalidating else { return }           // sessionInvalidated() continues it
        if reconfigurePending {
            replaceSession()
            return
        }
        guard let base = AppSettings.serverURL, let token = Keychain.token() else {
            stall(rec, "Not signed in")
            return
        }
        let files = rec.files
        guard let index = files.firstIndex(where: { !$0.done }) else {
            finish(rec)
            return
        }
        let file = files[index]
        if rec.uploadResource != file.resource {
            // The next file of the asset starts with a create. When the server still has an
            // unfinished upload of it from this device, the create answers with that row and its
            // received_bytes (section 28), so nothing is sent twice.
            let attempts = rec.attempts
            rec.clearTransfer()
            rec.attempts = attempts
            rec.uploadResource = file.resource
        }
        let source = UploadPaths.url(file.path)
        guard FileManager.default.fileExists(atPath: source.path) else {
            stall(rec, "The exported file is gone; the next sync exports it again")
            return
        }
        current = file.filename
        currentBytes = file.bytes

        var req: URLRequest
        let bodyURL: URL
        let tag: UploadTaskTag
        do {
            if let id = rec.uploadId {
                if rec.nextOffset >= file.bytes {
                    req = APIClient.makeRequest(base: base, token: token, method: "POST", path: "/api/uploads/\(id)/complete")
                    bodyURL = try emptyFile()
                    tag = UploadTaskTag(asset: rec.assetId, step: "complete", resource: file.resource, upload: id,
                                        offset: rec.nextOffset, length: 0, body: nil)
                } else {
                    let chunk = Int64(rec.chunkSize > 0 ? rec.chunkSize : Self.defaultChunk)
                    let offset = rec.nextOffset
                    let length = min(chunk, file.bytes - offset)
                    let chunkURL = UploadPaths.chunks.appendingPathComponent("\(id)-\(offset).part")
                    var chunkReq = APIClient.makeRequest(base: base, token: token, method: "PUT", path: "/api/uploads/\(id)/chunk",
                                                         query: ["offset": String(offset)])
                    chunkReq.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
                    let chunkTag = UploadTaskTag(asset: rec.assetId, step: "chunk", resource: file.resource, upload: id,
                                                 offset: offset, length: length, body: UploadPaths.relative(chunkURL))
                    let resource = file.resource
                    // The chunk file (up to 8 MiB) is cut off the main actor; the task is made once it
                    // is ready, and only if nothing moved the upload meanwhile (a pause, a resync).
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        do {
                            try await Task.detached(priority: .utility) {
                                try Self.writeChunk(from: source, offset: offset, length: length, to: chunkURL)
                            }.value
                        } catch {
                            guard rec.status == "uploading", rec.uploadId == id, rec.nextOffset == offset else { return }
                            self.fail(rec, "Could not read the exported file: \(error.localizedDescription)")
                            return
                        }
                        guard rec.status == "uploading", rec.uploadId == id, rec.nextOffset == offset,
                              rec.uploadResource == resource, !self.invalidating else {
                            try? FileManager.default.removeItem(at: chunkURL)
                            return
                        }
                        self.enqueue(rec, chunkReq, body: chunkURL, tag: chunkTag, delay: delay)
                    }
                    return
                }
            } else {
                let body = CreateBody(filename: file.filename, size: file.bytes, mime: file.mime, sha256: file.sha256,
                                      pair_key: rec.assetId, resource: file.resource)
                bodyURL = UploadPaths.chunks.appendingPathComponent("create-\(UUID().uuidString).json")
                try JSONEncoder().encode(body).write(to: bodyURL, options: .atomic)
                req = APIClient.makeRequest(base: base, token: token, method: "POST", path: "/api/uploads")
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                tag = UploadTaskTag(asset: rec.assetId, step: "create", resource: file.resource, upload: nil,
                                    offset: 0, length: 0, body: UploadPaths.relative(bodyURL))
            }
        } catch {
            fail(rec, "Could not read the exported file: \(error.localizedDescription)")
            return
        }
        enqueue(rec, req, body: bodyURL, tag: tag, delay: delay)
    }

    /// Hands one request to the session and records it on the asset (the chunk file in flight, the
    /// file's upload id), so a relaunched process can match the finished task to the record.
    private func enqueue(_ rec: SyncRecord, _ request: URLRequest, body bodyURL: URL, tag: UploadTaskTag, delay: TimeInterval) {
        var req = request
        req.allowsCellularAccess = !AppSettings.wifiOnly
        let task = session.uploadTask(with: req, fromFile: bodyURL)
        task.taskDescription = tag.text
        if delay > 0 { task.earliestBeginDate = Date(timeIntervalSinceNow: delay) }
        task.countOfBytesClientExpectsToSend = tag.length + 2048
        task.countOfBytesClientExpectsToReceive = 2048
        rec.chunkPath = tag.step == "chunk" ? tag.body : nil
        var files = rec.files
        if let i = files.firstIndex(where: { $0.resource == tag.resource }) { files[i].uploadId = rec.uploadId }
        rec.files = files
        rec.updatedAt = Date()
        save()
        task.resume()
    }

    private func handle(_ task: URLSessionTask, status: Int, body: Data, error: Error?) {
        guard let tag = UploadTaskTag.from(task.taskDescription) else { return }
        if let rel = tag.body { try? FileManager.default.removeItem(at: UploadPaths.url(rel)) }
        // Paused, finished or failed meanwhile, or a task of an older state: nothing to do.
        guard let rec = record(tag.asset), rec.status == "uploading",
              rec.uploadResource == tag.resource, rec.uploadId == tag.upload,
              tag.step == "create" || rec.nextOffset == tag.offset else { return }
        if rec.chunkPath == tag.body { rec.chunkPath = nil }
        if let error {
            if (error as? URLError)?.code == .cancelled { return }   // paused, or the app was force-quit
            retry(rec, reason: error.localizedDescription)
            return
        }
        switch status {
        case 200..<300:
            succeeded(rec, tag, body)
        case 409:
            let e = APIError.from(status: status, body: body)
            if tag.step == "chunk", let received = e.receivedBytes {
                // offset_mismatch: the server has a different number of bytes; continue from there
                count(rec) { rec.nextOffset = received; scheduleNext(rec) }
            } else {
                resync(rec)
            }
        case 404 where tag.step != "create":
            count(rec) { forgetUpload(rec); scheduleNext(rec) }
        case 401:
            stall(rec, "The device token was not accepted. Sign in again.")
        case 413, 507:
            fail(rec, APIError.from(status: status, body: body).detail)
        case 0, 408, 425, 429, 500...599:
            retry(rec, reason: APIError.from(status: status, body: body).localizedDescription)
        default:
            fail(rec, APIError.from(status: status, body: body).localizedDescription)
        }
        save()
    }

    private func succeeded(_ rec: SyncRecord, _ tag: UploadTaskTag, _ body: Data) {
        rec.attempts = 0
        switch tag.step {
        case "chunk":
            // A lost body (rare) still means the whole chunk was written: the server answers 200 only then.
            let received = (try? JSONDecoder().decode(ChunkOut.self, from: body))?.received_bytes ?? tag.offset + tag.length
            rec.nextOffset = received
            if currentBytes > 0 { fileProgress = Double(received) / Double(currentBytes) }
            scheduleNext(rec)
        default:
            guard let up = try? JSONDecoder().decode(UploadOut.self, from: body) else {
                if tag.step == "create" { stall(rec, "The server's answer could not be read") } else { resync(rec) }
                return
            }
            applyUpload(rec, up)
        }
    }

    /// Moves the record by an upload's state as the server reports it (create, complete or a read).
    private func applyUpload(_ rec: SyncRecord, _ up: UploadOut) {
        switch up.status {
        case "uploading":
            rec.uploadId = up.id
            rec.chunkSize = up.chunk_size
            rec.nextOffset = up.received_bytes
            scheduleNext(rec)
        case "queued", "processing", "stored", "duplicate":
            fileDone(rec, uploadId: up.id, itemUid: up.item_uid)
            scheduleNext(rec)
        case "cancelled":
            count(rec) { forgetUpload(rec); scheduleNext(rec) }
        default:   // rejected, error
            fail(rec, up.message ?? "The server turned the file down")
        }
    }

    /// A 409 without an offset, or an answer that could not be read: the upload's own state says
    /// what happened (`GET /api/uploads/{id}`).
    private func resync(_ rec: SyncRecord) {
        guard let id = rec.uploadId, let api = APIClient.current() else {
            count(rec) { forgetUpload(rec); scheduleNext(rec) }
            return
        }
        count(rec) {
            Task { @MainActor in
                let answer: UploadOut??
                do { answer = .some(try await api.upload(id: id)) } catch { answer = .none }
                guard rec.status == "uploading", rec.uploadId == id else { return }
                switch answer {
                case .none: self.stall(rec, "Cannot reach the server; the upload continues on the next sync")
                case .some(.none): self.forgetUpload(rec); self.scheduleNext(rec)
                case .some(.some(let up)): self.applyUpload(rec, up)
                }
                self.save()
            }
        }
    }

    /// Counts an attempt that did not move the upload forward; the asset fails after eight in a row.
    private func count(_ rec: SyncRecord, then next: () -> Void) {
        rec.attempts += 1
        if rec.attempts >= Self.maxAttempts {
            fail(rec, "The upload did not make progress after \(Self.maxAttempts) attempts")
        } else {
            next()
        }
    }

    private func retry(_ rec: SyncRecord, reason: String) {
        rec.attempts += 1
        if rec.attempts >= Self.maxAttempts {
            fail(rec, "Gave up after \(Self.maxAttempts) attempts: \(reason)")
            return
        }
        let delay = Self.backoff[min(rec.attempts - 1, Self.backoff.count - 1)]
        scheduleNext(rec, after: delay)
    }

    /// The current file starts again with a new create (the server dropped or cancelled its upload).
    private func forgetUpload(_ rec: SyncRecord) {
        var files = rec.files
        if let i = files.firstIndex(where: { $0.resource == rec.uploadResource }) { files[i].uploadId = nil }
        rec.files = files
        rec.uploadId = nil
        rec.nextOffset = 0
    }

    private func fileDone(_ rec: SyncRecord, uploadId: String, itemUid: String?) {
        var files = rec.files
        if let i = files.firstIndex(where: { $0.resource == rec.uploadResource && !$0.done }) {
            files[i].done = true
            files[i].uploadId = uploadId
            files[i].itemUid = itemUid
            try? FileManager.default.removeItem(at: UploadPaths.url(files[i].path))   // the server has it
            if files[i].resource == "still" || files.count == 1 { rec.uid = itemUid ?? rec.uid } else { rec.clipUid = itemUid ?? rec.clipUid }
        }
        rec.files = files
        rec.clearTransfer()
        fileProgress = 0
    }

    /// Every file is on the server. 'uploaded' until the sync check confirms the item ('synced').
    private func finish(_ rec: SyncRecord) {
        let files = rec.files
        let confirmed = !files.isEmpty && files.allSatisfy { $0.itemUid != nil }
        rec.status = confirmed ? "synced" : "uploaded"
        rec.error = nil
        rec.clearTransfer()
        rec.updatedAt = Date()
        UploadPaths.removeAsset(rec.assetId)
        save()
        SyncStore.shared.log("Sent \(rec.filename)")
        current = nil
        releaseAll(rec.assetId)
    }

    /// The asset fails with a message; the next run starts it again (a fresh export, and a create
    /// that the server answers with its own row when it still has one of the same size).
    private func fail(_ rec: SyncRecord, _ message: String) {
        rec.status = "failed"
        rec.error = message
        rec.clearTransfer()
        rec.updatedAt = Date()
        UploadPaths.removeAsset(rec.assetId)
        rec.files = rec.files.map { var f = $0; if !f.done { f.uploadId = nil }; return f }
        save()
        SyncStore.shared.log("\(rec.filename.isEmpty ? rec.assetId : rec.filename): \(message)", level: "error")
        current = nil
        releaseAll(rec.assetId)
    }

    /// Stops for now, keeping the state: the next run continues from the server's offset.
    private func stall(_ rec: SyncRecord, _ message: String) {
        rec.status = "pending"
        rec.error = message
        rec.updatedAt = Date()
        save()
        SyncStore.shared.log("\(rec.filename.isEmpty ? rec.assetId : rec.filename): \(message)", level: "error")
        current = nil
        releaseAll(rec.assetId)
    }

    // MARK: helpers

    private nonisolated static func writeChunk(from source: URL, offset: Int64, length: Int64, to target: URL) throws {
        let handle = try FileHandle(forReadingFrom: source)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(offset))
        guard let data = try handle.read(upToCount: Int(length)), data.count == Int(length) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        try data.write(to: target, options: .atomic)
    }

    private func emptyFile() throws -> URL {
        let url = UploadPaths.chunks.appendingPathComponent("empty.body")
        if !FileManager.default.fileExists(atPath: url.path) { try Data().write(to: url) }
        return url
    }

    private func record(_ assetId: String) -> SyncRecord? {
        var d = FetchDescriptor<SyncRecord>(predicate: #Predicate<SyncRecord> { $0.assetId == assetId })
        d.fetchLimit = 1
        return try? SyncStore.shared.container.mainContext.fetch(d).first
    }

    private func records(status: String) -> [SyncRecord] {
        let d = FetchDescriptor<SyncRecord>(predicate: #Predicate<SyncRecord> { $0.status == status })
        return (try? SyncStore.shared.container.mainContext.fetch(d)) ?? []
    }

    private func save() {
        try? SyncStore.shared.container.mainContext.save()
    }

    private func release(_ assetId: String, token: UUID) {
        waiters[assetId]?.removeValue(forKey: token)?.resume()
    }

    private func releaseAll(_ assetId: String) {
        let all = waiters.removeValue(forKey: assetId) ?? [:]
        for cont in all.values { cont.resume() }
    }

    fileprivate func received(_ data: Data, task: Int) {
        bodies[task, default: Data()].append(data)
    }

    fileprivate func sent(_ tag: UploadTaskTag?, total: Int64) {
        guard let tag, tag.step == "chunk", currentBytes > 0 else { return }
        fileProgress = min(1, Double(tag.offset + total) / Double(currentBytes))
    }

    fileprivate func completed(_ task: URLSessionTask, error: Error?) {
        let body = bodies.removeValue(forKey: task.taskIdentifier) ?? Data()
        let status = (task.response as? HTTPURLResponse)?.statusCode ?? 0
        handle(task, status: status, body: body, error: error)
    }

    fileprivate func eventsFinished() {
        let done = backgroundCompletion
        backgroundCompletion = nil
        done?()
    }

    fileprivate func invalidated() {
        sessionInvalidated()
    }
}

// The session's delegate queue is the main queue, so every callback runs on the main actor.
extension BackgroundUploader: URLSessionDataDelegate {
    nonisolated func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        let id = dataTask.taskIdentifier
        MainActor.assumeIsolated { self.received(data, task: id) }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                                totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        let tag = UploadTaskTag.from(task.taskDescription)
        MainActor.assumeIsolated { self.sent(tag, total: totalBytesSent) }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        MainActor.assumeIsolated { self.completed(task, error: error) }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        MainActor.assumeIsolated { self.eventsFinished() }
    }

    nonisolated func urlSession(_ session: URLSession, didBecomeInvalidWithError error: Error?) {
        MainActor.assumeIsolated { self.invalidated() }
    }
}
