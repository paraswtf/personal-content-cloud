import XCTest
@testable import PersonalContentCloud

/// The Upload screen's rules match the website's (lib/uploader.ts, SummaryBar.tsx, status.tsx, UploadPage.tsx).
final class UploadTests: XCTestCase {
    func testKindsFollowTheMimeTypeThenTheExtension() {
        XCTAssertEqual(UploadKind.of(name: "IMG_0001.HEIC"), .image)
        XCTAssertEqual(UploadKind.of(name: "scan.dng"), .image)
        XCTAssertEqual(UploadKind.of(name: "clip.MOV"), .video)
        XCTAssertEqual(UploadKind.of(name: "Invoice.PDF"), .pdf)
        XCTAssertEqual(UploadKind.of(name: "notes.txt"), .document)
        XCTAssertEqual(UploadKind.of(name: "report", mime: "application/vnd.openxmlformats-officedocument.wordprocessingml.document"), .document)
        XCTAssertEqual(UploadKind.of(name: "noext", mime: "video/mp4"), .video)
        XCTAssertEqual(UploadKind.of(name: "archive.zip"), .other)
        XCTAssertEqual(UploadKind.of(name: ".hidden"), .other)
        XCTAssertEqual(UploadKind.fileExtension("a.b.JPG"), "jpg")
        XCTAssertEqual(UploadKind.fileExtension("README"), "")
    }

    func testPhasesFromTheRecordAndTheServer() {
        XCTAssertEqual(UploadPhase.of(status: "pending", serverStatus: nil, checking: false), .waiting)
        XCTAssertEqual(UploadPhase.of(status: "pending", serverStatus: nil, checking: true), .checking)
        XCTAssertEqual(UploadPhase.of(status: "uploading", serverStatus: nil, checking: false), .uploading)
        XCTAssertEqual(UploadPhase.of(status: "uploaded", serverStatus: nil, checking: false), .processing)
        XCTAssertEqual(UploadPhase.of(status: "uploaded", serverStatus: "queued", checking: false), .processing)
        XCTAssertEqual(UploadPhase.of(status: "synced", serverStatus: "stored", checking: false), .done)
        XCTAssertEqual(UploadPhase.of(status: "synced", serverStatus: "duplicate", checking: false), .done)
        XCTAssertEqual(UploadPhase.of(status: "uploaded", serverStatus: "error", checking: false), .done)
        XCTAssertEqual(UploadPhase.of(status: "uploaded", serverStatus: "cancelled", checking: false), .cancelled)
        XCTAssertEqual(UploadPhase.of(status: "failed", serverStatus: nil, checking: false), .failed)
        XCTAssertEqual(UploadPhase.of(status: "skipped", serverStatus: nil, checking: false), .failed)
        XCTAssertEqual(UploadPhase.of(status: "cancelled", serverStatus: nil, checking: true), .cancelled)
        XCTAssertTrue(UploadPhase.checking.canCancel)
        XCTAssertFalse(UploadPhase.processing.canCancel)
        XCTAssertTrue(UploadPhase.cancelled.isFinished)
        XCTAssertFalse(UploadPhase.processing.isFinished)
    }

    func testRetryAsTheWebsiteAllowsIt() {
        XCTAssertTrue(UploadRowInfo(phase: .failed, bytes: 1).canRetry)
        XCTAssertFalse(UploadRowInfo(phase: .failed, bytes: 0, skipped: true).canRetry)
        XCTAssertTrue(UploadRowInfo(phase: .cancelled, bytes: 1).canRetry)
        XCTAssertTrue(UploadRowInfo(phase: .done, serverStatus: "error", bytes: 1).canRetry)
        XCTAssertFalse(UploadRowInfo(phase: .done, serverStatus: "rejected", bytes: 1).canRetry)
        XCTAssertFalse(UploadRowInfo(phase: .done, serverStatus: "stored", bytes: 1).canRetry)
    }

    func testSummaryCountsAndWords() {
        let rows = [
            UploadRowInfo(phase: .uploading, bytes: 100, progress: 0.5),
            UploadRowInfo(phase: .waiting, bytes: 100),
            UploadRowInfo(phase: .processing, bytes: 200),
            UploadRowInfo(phase: .done, serverStatus: "stored", bytes: 100),
            UploadRowInfo(phase: .done, serverStatus: "duplicate", bytes: 100),
            UploadRowInfo(phase: .done, serverStatus: "rejected", bytes: 100),
            UploadRowInfo(phase: .failed, bytes: 0, skipped: true),
            UploadRowInfo(phase: .cancelled, bytes: 100),
        ]
        let s = UploadSummary.of(rows)
        XCTAssertEqual(s, UploadSummary(total: 8, added: 1, duplicates: 1, failed: 2, sending: 2, processing: 1, cancelled: 1))
        XCTAssertEqual(s.finished, 5)
        // (50 + 0 + 200 + 100 + 100 + 100) of 700: failed and cancelled files do not count
        XCTAssertEqual(UploadSummary.progress(rows), 550.0 / 700.0, accuracy: 0.0001)
        XCTAssertEqual(s.headline(progress: 0.5), "Uploading 2 files, 50%")
        XCTAssertEqual(s.outcomes, "1 added · 1 already in your library · 2 not uploaded · 1 cancelled")
        let processing = UploadSummary(total: 1, processing: 1)
        XCTAssertEqual(processing.headline(progress: 1), "Processing 1 file")
        XCTAssertEqual(UploadSummary(total: 1, added: 1).headline(progress: 1), "All done")
        XCTAssertEqual(UploadSummary.progress([]), 0)
    }

    func testNoticesLimitsAndRefusals() {
        XCTAssertEqual(UploadText.notice(added: 1, rejected: 0, skipped: 0), "1 file added to the queue.")
        XCTAssertEqual(UploadText.notice(added: 3, rejected: 2, skipped: 1),
                       "3 files added to the queue. 2 files were skipped because they are empty or too large. 1 file is already in the list.")
        XCTAssertEqual(UploadText.notice(added: 0, rejected: 1, skipped: 2),
                       "1 file was skipped because it is empty or too large. 2 files are already in the list.")
        XCTAssertEqual(UploadText.notice(added: 0, rejected: 0, skipped: 0), "")
        let gb: Int64 = 1024 * 1024 * 1024
        XCTAssertEqual(UploadText.limit(16 * gb), "16 GB")
        XCTAssertEqual(UploadText.limit(gb + gb / 2), "1.5 GB")
        XCTAssertEqual(UploadText.limit(512 * 1024 * 1024), "512 MB")
        XCTAssertEqual(UploadText.rejection(size: 0, limit: gb), "This file is empty.")
        XCTAssertEqual(UploadText.rejection(size: 2 * gb, limit: gb), "This file is larger than the 1 GB limit.")
        XCTAssertNil(UploadText.rejection(size: 10, limit: gb))
    }

    func testBadgesAndPhaseWords() {
        XCTAssertEqual(UploadText.badge(phase: .done, serverStatus: "stored", skipped: false), "Added")
        XCTAssertEqual(UploadText.badge(phase: .done, serverStatus: "duplicate", skipped: false), "Already in your library")
        XCTAssertEqual(UploadText.badge(phase: .done, serverStatus: "rejected", skipped: false), "Not accepted")
        XCTAssertEqual(UploadText.badge(phase: .done, serverStatus: "error", skipped: false), "Processing failed")
        XCTAssertEqual(UploadText.badge(phase: .failed, serverStatus: nil, skipped: true), "Skipped")
        XCTAssertEqual(UploadText.badge(phase: .failed, serverStatus: nil, skipped: false), "Not uploaded")
        XCTAssertEqual(UploadText.badge(phase: .cancelled, serverStatus: nil, skipped: false), "Cancelled")
        XCTAssertNil(UploadText.badge(phase: .uploading, serverStatus: nil, skipped: false))
        XCTAssertEqual(UploadText.phase(.uploading, progress: 0.42, resumed: false), "Uploading 42%")
        XCTAssertEqual(UploadText.phase(.uploading, progress: 0.42, resumed: true), "Resuming 42%")
        XCTAssertEqual(UploadText.phase(.checking, progress: nil, resumed: false), "Checking")
        XCTAssertEqual(UploadText.phase(.checking, progress: 0.4, resumed: false), "Checking 40%")
        XCTAssertEqual(UploadText.phase(.checking, progress: nil, resumed: true), "Resuming")
        XCTAssertEqual(UploadText.phase(.waiting, progress: nil, resumed: false), "Waiting")
        XCTAssertNil(UploadText.phase(.done, progress: nil, resumed: false))
    }

    func testCopyNamesAreSafe() {
        XCTAssertEqual(UploadIncoming.safeName("a/b:c.jpg"), "a_b_c.jpg")
        XCTAssertEqual(UploadIncoming.safeName("..hidden"), "hidden")
        XCTAssertEqual(UploadIncoming.safeName("  "), "Untitled")
        let date = Date(timeIntervalSince1970: 1_794_856_697)   // 16 Nov 2026, 19:18:17 UTC
        let name = UploadIncoming.cameraName(prefix: "IMG", ext: "jpg", at: date)
        XCTAssertTrue(name.hasPrefix("IMG_20261"), name)
        XCTAssertTrue(name.hasSuffix(".jpg"), name)
        XCTAssertEqual(name.count, "IMG_20261116_191817.jpg".count)
    }

    /// The phone's sync and the Upload screen's queue never take the same library photo at once: whoever claims it
    /// first sends it, the other waits until it is let go.
    @MainActor
    func testOneSideAtATimeTakesALibraryPhoto() {
        let engine = SyncEngine.shared
        let id = "upload-tests-claim-\(UUID().uuidString)"
        XCTAssertTrue(engine.claim(id))
        XCTAssertFalse(engine.claim(id), "a second claim of the same photo must fail")
        XCTAssertTrue(engine.claimed.contains(id))
        engine.release(id)
        XCTAssertFalse(engine.claimed.contains(id))
        XCTAssertTrue(engine.claim(id), "a photo that was let go can be taken again")
        engine.release(id)
    }

    /// Copies no add took are swept after a day; a fresh copy stays.
    func testLeftoverCopiesAreSweptAfterADay() throws {
        let fm = FileManager.default
        let old = UploadIncoming.newURL(for: "upload-tests-old.jpg")
        let fresh = UploadIncoming.newURL(for: "upload-tests-fresh.jpg")
        try Data([1, 2, 3]).write(to: old)
        try Data([4, 5, 6]).write(to: fresh)
        let oldFolder = old.deletingLastPathComponent()
        let freshFolder = fresh.deletingLastPathComponent()
        let twoDaysAgo = Date(timeIntervalSinceNow: -2 * 24 * 3600)
        try fm.setAttributes([.modificationDate: twoDaysAgo], ofItemAtPath: oldFolder.path)
        UploadIncoming.sweep(olderThan: 24 * 3600)
        XCTAssertFalse(fm.fileExists(atPath: oldFolder.path), "a copy left for two days is removed")
        XCTAssertTrue(fm.fileExists(atPath: fresh.path), "a copy made now stays")
        UploadIncoming.discard(fresh)
        XCTAssertFalse(fm.fileExists(atPath: freshFolder.path))
    }

    /// Time offline does not use up attempts (the website's uploader.ts: attempt-- while offline). Online, every
    /// failure counts, so a server that keeps dropping the connection ends in a failed row instead of retrying forever.
    func testOfflineFailuresDoNotUseUpAttempts() {
        XCTAssertFalse(BackgroundUploader.countsAttempt(online: false, error: nil))
        XCTAssertFalse(BackgroundUploader.countsAttempt(online: false, error: URLError(.timedOut)))
        XCTAssertFalse(BackgroundUploader.countsAttempt(online: false, error: URLError(.notConnectedToInternet)))
        XCTAssertTrue(BackgroundUploader.countsAttempt(online: true, error: URLError(.notConnectedToInternet)))
        XCTAssertTrue(BackgroundUploader.countsAttempt(online: true, error: URLError(.networkConnectionLost)))
        XCTAssertTrue(BackgroundUploader.countsAttempt(online: true, error: URLError(.timedOut)))
        XCTAssertTrue(BackgroundUploader.countsAttempt(online: true, error: nil))
    }

    /// Back online, the queue starts again the files that failed on the way for the network, never one the server
    /// answered or one that failed for another reason.
    func testFilesThatFailedForTheNetworkStartAgain() {
        XCTAssertTrue(UploadQueue.failedForNetwork(status: "failed", serverStatus: nil,
                                                   error: UploadQueue.connectionLost, known: false))
        XCTAssertTrue(UploadQueue.failedForNetwork(status: "failed", serverStatus: nil,
                                                   error: "Cannot reach the server; the upload continues on the next sync",
                                                   known: false))
        XCTAssertTrue(UploadQueue.failedForNetwork(status: "failed", serverStatus: nil,
                                                   error: "Gave up after 8 attempts: The request timed out.", known: true))
        XCTAssertFalse(UploadQueue.failedForNetwork(status: "failed", serverStatus: nil,
                                                    error: "Gave up after 8 attempts: The request timed out.", known: false))
        XCTAssertFalse(UploadQueue.failedForNetwork(status: "failed", serverStatus: nil,
                                                    error: "This file could not be read. It may have been moved or deleted.",
                                                    known: false))
        XCTAssertFalse(UploadQueue.failedForNetwork(status: "failed", serverStatus: "error",
                                                    error: UploadQueue.connectionLost, known: true))
        XCTAssertFalse(UploadQueue.failedForNetwork(status: "pending", serverStatus: nil,
                                                    error: UploadQueue.connectionLost, known: true))
    }

    func testNetworkNoticeSaysOfflineOrCountsDown() {
        XCTAssertEqual(UploadText.network(offline: true, retryIn: 4),
                       "You are offline. The upload continues when the connection is back.")
        XCTAssertEqual(UploadText.network(offline: false, retryIn: 4), "Connection problem. Trying again in 4 s.")
        XCTAssertNil(UploadText.network(offline: false, retryIn: nil))
        XCTAssertNil(UploadText.network(offline: false, retryIn: 0))
        let now = Date()
        XCTAssertEqual(UploadText.secondsUntil(now.addingTimeInterval(3.2), now: now), 4)
        XCTAssertEqual(UploadText.secondsUntil(now.addingTimeInterval(1), now: now), 1)
        XCTAssertNil(UploadText.secondsUntil(now.addingTimeInterval(-1), now: now))
        XCTAssertNil(UploadText.secondsUntil(nil, now: now))
    }
}
