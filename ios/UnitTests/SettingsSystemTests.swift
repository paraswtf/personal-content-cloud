import XCTest
@testable import PersonalContentCloud

/// The wording and the readers of the system side of Settings (Features/Settings/System), against the website's
/// rules in components/shell/settings and components/undo.
final class SettingsSystemTests: XCTestCase {
    private func json(_ text: String) throws -> [String: API.JSONValue] {
        try JSONDecoder().decode([String: API.JSONValue].self, from: Data(text.utf8))
    }

    func testProviderLabels() {
        XCTAssertEqual(SettingsSystemText.providerLabel("CUDAExecutionProvider"), "GPU (CUDA)")
        XCTAssertEqual(SettingsSystemText.providerLabel("CPUExecutionProvider"), "CPU")
        XCTAssertEqual(SettingsSystemText.providerLabel("CoreMLExecutionProvider"), "CoreML")
        XCTAssertEqual(SettingsSystemText.providerLabel(nil), "-")
    }

    func testModelStates() {
        func model(present: Bool = true, loaded: Bool = false, downloading: Bool = false, error: String? = nil) -> API.ModelStatus {
            API.ModelStatus(key: "faces", label: "Faces", present: present, loaded: loaded, provider: nil, sizeBytes: 0,
                            downloading: downloading, error: error)
        }
        XCTAssertEqual(SettingsSystemText.modelState(model(present: false)).text, "Missing")
        XCTAssertEqual(SettingsSystemText.modelState(model(error: "broken")).text, "Error")
        XCTAssertEqual(SettingsSystemText.modelState(model(present: false, downloading: true)).text, "Downloading")
        XCTAssertEqual(SettingsSystemText.modelState(model(loaded: true)).text, "Loaded")
        XCTAssertEqual(SettingsSystemText.modelState(model()).text, "Ready, loads on first use")
        XCTAssertEqual(SettingsSystemText.downloadLabel(missing: 1, running: false), "Download the missing model")
        XCTAssertEqual(SettingsSystemText.downloadLabel(missing: 3, running: false), "Download missing models")
        XCTAssertEqual(SettingsSystemText.downloadLabel(missing: 3, running: true), "Downloading")
    }

    func testEncoderLines() throws {
        let none = SettingsSystemStatus.Video(try json(#"{"encoder": null, "hw": false}"#))
        XCTAssertTrue(SettingsSystemText.encoderLine(none).hasPrefix("Not detected yet."))
        let software = SettingsSystemStatus.Video(try json(#"{"encoder": "libx264", "hw": false, "failed": 2}"#))
        XCTAssertEqual(SettingsSystemText.encoderLine(software), "libx264 (software) on the CPU")
        XCTAssertEqual(software.failed, 2)
        let device = SettingsSystemStatus.Video(try json(#"{"encoder": "h264_vaapi", "hw": true, "device": "/dev/dri/renderD128"}"#))
        XCTAssertEqual(SettingsSystemText.encoderLine(device), "h264_vaapi on /dev/dri/renderD128")
        let nvenc = SettingsSystemStatus.Video(try json(#"{"encoder": "h264_nvenc", "hw": true}"#))
        XCTAssertEqual(SettingsSystemText.encoderLine(nvenc), "h264_nvenc (NVIDIA NVENC)")
        XCTAssertEqual(SettingsSystemText.requeueMessage(0), "No failed copies to re-run.")
        XCTAssertEqual(SettingsSystemText.requeueMessage(1), "1 copy queued again.")
        XCTAssertEqual(SettingsSystemText.retryMessage(0), "There were no failed uploads to retry.")
        XCTAssertEqual(SettingsSystemText.retryMessage(2), "2 failed uploads were queued again.")
    }

    func testTiers() {
        let low = SettingsTier(key: "originals", path: "/srv", holds: "originals", freeBytes: 5, totalBytes: 100, sharedWith: nil)
        XCTAssertEqual(low.usedPercent, 95)
        XCTAssertTrue(low.lowSpace)
        XCTAssertEqual(SettingsSystemText.tierUsage(low), "95% used. Free space is low; new uploads may be refused.")
        let roomy = SettingsTier(key: "data", path: "/srv", holds: "previews", freeBytes: 60, totalBytes: 100, sharedWith: nil)
        XCTAssertEqual(SettingsSystemText.tierUsage(roomy), "40% used")
        XCTAssertEqual(SettingsSystemText.tierTitle("data"), "Database and previews")
        XCTAssertEqual(SettingsSystemText.hour(3), "03:00")
    }

    func testBackupLines() {
        XCTAssertEqual(SettingsSystemText.backupLastLine(nil), "No snapshot yet.")
        let failed = SettingsSystemStatus.BackupRecord(finishedAt: nil, ok: false, error: "disk full", files: 0,
                                                       bytesCopied: 0, bytesLinked: 0)
        XCTAssertEqual(SettingsSystemText.backupLastLine(failed), "The last run just now failed: disk full.")
        let ok = SettingsSystemStatus.BackupRecord(finishedAt: nil, ok: true, error: nil, files: 1200, bytesCopied: 2048,
                                                   bytesLinked: 0)
        XCTAssertTrue(SettingsSystemText.backupLastLine(ok).hasPrefix("Last snapshot just now: "))
        XCTAssertTrue(SettingsSystemText.backupLastLine(ok).hasSuffix("copied."))
    }

    func testQueuesAndStatusRead() throws {
        let gpu = try json(#"{"available": false, "name": null, "provider": null}"#)
        let queues = try json(#"{"queued": 2, "running": 1, "failed": 0, "by_queue": {"video": {"queued": 0, "running": 0}, "main": {"queued": 2, "running": 1}}}"#)
        let system = API.SystemOut(version: "1.0", siteTitle: "Cloud", gpu: gpu, models: [], queues: queues,
                                   disk: API.DiskStats(freeBytes: 10, totalBytes: 100, originalsBytes: 5), wallet: [:],
                                   timezone: "UTC", ffmpeg: true, workers: true, download: nil, video: nil,
                                   forwards: nil, pets: nil, storage: nil, backup: nil)
        let status = SettingsSystemStatus(system)
        XCTAssertFalse(status.gpu.available)
        XCTAssertTrue(status.queues.busy)
        XCTAssertEqual(status.queues.byQueue.map(\.name), ["main", "video"])
        XCTAssertEqual(SettingsSystemText.queueLabel("main"), "Analysis (photos, videos, faces)")
        let tiers = SettingsTier.tiers(system)
        XCTAssertEqual(tiers.map(\.key), ["originals"], "an older server without storage tiers shows one disk")
        XCTAssertEqual(tiers.first?.holds, "everything")
    }

    func testThroughputWords() {
        XCTAssertEqual(SettingsSystemText.niceMax(0), 1)
        XCTAssertEqual(SettingsSystemText.niceMax(3), 3)
        XCTAssertEqual(SettingsSystemText.niceMax(7), 10)
        XCTAssertEqual(SettingsSystemText.niceMax(11), 20)
        XCTAssertEqual(SettingsSystemText.niceMax(23), 25)
        XCTAssertEqual(SettingsSystemText.niceMax(26), 50)
        XCTAssertEqual(SettingsSystemText.niceMax(100), 100)
        XCTAssertEqual(SettingsSystemText.niceMax(101), 200)
        XCTAssertEqual(SettingsSystemText.duration(ms: 164), "164 ms")
        XCTAssertEqual(SettingsSystemText.duration(ms: 1500), "1.5 s")
        XCTAssertEqual(SettingsSystemText.duration(ms: 12_000), "12 s")
        XCTAssertEqual(SettingsSystemText.duration(ms: 75_000), "1 min 15 s")
        XCTAssertEqual(SettingsSystemText.duration(ms: nil), "")
        XCTAssertEqual(SettingsSystemText.rateLine(0), "idle now")
        XCTAssertEqual(SettingsSystemText.rateLine(0.4), "0.4 a minute now")
        let t = API.ThroughputType(type: "ingest", label: "Files stored", total: 4, failed: 1, avgMs: 164,
                                   counts: [0, 4], failedCounts: [0, 1])
        XCTAssertEqual(SettingsSystemText.typeSummary(t, window: "in the last hour"),
                       "4 in the last hour, 164 ms each on average, 1 failed")
        XCTAssertEqual(SettingsSystemText.period(.day).per, "per hour")
        XCTAssertFalse(SettingsSystemText.bucketLabel("2026-11-16T22:10:00.000Z", bucketSeconds: 60, withDay: false).isEmpty)
    }

    func testLearningWords() throws {
        XCTAssertEqual(SettingsSystemText.misclickChanges(["blur_var_min": 55]), ["blur threshold now 55 (default 40)"])
        XCTAssertEqual(SettingsSystemText.misclickChanges(["blur_var_min": 40, "dark_mean": 28.1]), [],
                       "cuts within half a percent of the default are not changes")
        XCTAssertEqual(SettingsSystemText.misclickChanges(["clipped_frac": 0.7]), ["blown-out threshold now 0.70 (default 0.60)"])
        XCTAssertEqual(SettingsSystemText.labelList(["a", "b", "c", "d", "e"]) { $0.uppercased() }, "A, B, C, D and 1 more")
        XCTAssertEqual(SettingsSystemText.docTypeLabel("pan"), "PAN card")
        XCTAssertEqual(SettingsSystemText.docTypeLabel("some_new_type"), "Some new type")
        XCTAssertEqual(SettingsSystemText.docTypeLabel(nil), "Document")
        XCTAssertEqual(SettingsSystemText.categoryLabel("objects"), "Things")
        XCTAssertEqual(SettingsSystemText.movedText(nil), "")
        XCTAssertEqual(SettingsSystemText.movedText(["moved_in": 0, "moved_out": 0, "stale": 0]), "Nothing moved.")
        XCTAssertEqual(SettingsSystemText.movedText(["moved_in": 2, "moved_out": 1, "stale": 1]),
                       "2 moved into the vault, 1 moved out. 1 older item needs \"Apply to existing items\" to be checked with the new signals.")
        XCTAssertEqual(SettingsSystemText.applyMessage(items: 0, documents: 0, tags: 0), "Nothing needed a fresh look.")
        XCTAssertEqual(SettingsSystemText.applyMessage(items: 0, documents: 0, tags: 2), "2 tags suggested on existing items.")
        XCTAssertEqual(SettingsSystemText.applyMessage(items: 3, documents: 1, tags: 0),
                       "Queued 3 items and 1 document for a fresh look. Progress shows under System.")
        XCTAssertEqual(SettingsSystemText.sensitivityLabel(.high), "High")
    }

    func testLearnersRead() throws {
        let empty = SettingsLearners(try json(#"""
        {"doc_type": {"examples": 0, "labels": []}, "category": {"examples": 0, "labels": []},
         "vault": {"examples": 0, "threshold": null, "weights": null},
         "misclick": {"examples": 0, "thresholds": {}}, "people": {"persons": 0, "events": 0, "weights": null},
         "tags": {"tags": 0, "examples": 0, "rejections": 0}, "forwards": {"examples": 0, "learned": false, "weights": null}}
        """#))
        XCTAssertTrue(empty.nothingLearned)
        XCTAssertFalse(empty.scanning)
        XCTAssertNil(empty.traits)

        let busy = SettingsLearners(try json(#"""
        {"doc_type": {"examples": 3, "labels": ["pan", "aadhaar"]}, "category": {"examples": 0, "labels": []},
         "vault": {"examples": 0, "threshold": null, "weights": {"a": 1}},
         "misclick": {"examples": 2, "thresholds": {"blur_var_min": 60.0}}, "people": {"persons": 1, "events": 4, "weights": null},
         "traits": {"source": "default", "tattoo_min_inliers": 12, "outcomes": 0,
                    "scan": {"scanned": 0, "eligible": 1, "ready": false, "rev": null, "paused": ["pose"], "last_error": null}},
         "place": {"scan": {"scanned": 3, "eligible": 19, "ready": false, "rev": null, "last_error": null}}}
        """#))
        XCTAssertFalse(busy.nothingLearned)
        XCTAssertTrue(busy.vaultWeights)
        XCTAssertNil(busy.tags, "a server without tag learning")
        XCTAssertEqual(busy.traits?.scan?.paused, ["pose"])
        XCTAssertFalse(busy.traits?.reading ?? true, "a paused reading is not reading")
        XCTAssertTrue(busy.scanning, "lining photos up is still going")
        XCTAssertEqual(SettingsLearningText.docTypes(busy), "Learned from 3 corrections, 2 types: PAN card, Aadhaar card.")
        XCTAssertEqual(SettingsLearningText.misclicks(busy), "Blur threshold now 60 (default 40), from 2 decisions.")
        XCTAssertEqual(SettingsLearningText.forwardsBacklog(SettingsSystemStatus.Forwards(unchecked: 5, running: true)),
                       "Checking older items for forwards: 5 to go.")
    }

    func testUndoStatus() {
        let now = Date()
        let done = UndoCenter.Entry(id: "a", label: "Renamed", at: now.addingTimeInterval(-300), state: .done, activeId: "a")
        XCTAssertEqual(SettingsSystemText.undoStatus(done, now: now), "5 min ago")
        var undone = done
        undone.state = .undone
        undone.undoId = "b"
        XCTAssertEqual(SettingsSystemText.undoStatus(undone, now: now), "5 min ago, undone")
        undone.undoId = nil
        XCTAssertEqual(SettingsSystemText.undoStatus(undone, now: now), "5 min ago, undone elsewhere")
        var old = done
        old.expired = true
        old.partial = true
        XCTAssertEqual(SettingsSystemText.undoStatus(old, now: now), "5 min ago, too old to undo, partly, as some of it had changed since")
        XCTAssertEqual(SettingsSystemText.relative(now.addingTimeInterval(-10), now: now), "just now")
        XCTAssertEqual(SettingsSystemText.relative(now.addingTimeInterval(-3 * 3600), now: now), "3 h ago")
    }
}
