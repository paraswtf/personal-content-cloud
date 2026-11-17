import Foundation

// The Upload screen's rules, as the website has them (lib/uploader.ts, components/upload/SummaryBar.tsx,
// components/upload/status.tsx, pages/UploadPage.tsx): what kind a file is, where a queued file is in its life,
// the queue's totals and the words for them. Pure, so the unit tests check them.

/// What a file is, for its icon only (the server's probe decides what it really is).
enum UploadKind: String, Equatable {
    case image, video, pdf, document, other

    private static let imageExtensions: Set<String> = [
        "jpg", "jpeg", "png", "gif", "webp", "bmp", "tif", "tiff", "heic", "heif", "avif", "jxl",
        "dng", "cr2", "cr3", "nef", "arw", "orf", "rw2", "raf", "srw", "pef",
    ]
    private static let videoExtensions: Set<String> = [
        "mp4", "mov", "m4v", "webm", "mkv", "avi", "3gp", "mts", "m2ts", "wmv", "mpg", "mpeg",
    ]
    private static let documentExtensions: Set<String> = [
        "docx", "doc", "xlsx", "xls", "pptx", "ppt", "txt", "md", "csv", "rtf", "odt", "ods", "odp",
    ]
    private static let documentMimePrefixes = [
        "text/", "application/msword", "application/vnd.openxmlformats-officedocument", "application/vnd.ms-",
        "application/vnd.oasis.opendocument", "application/rtf",
    ]

    /// The file's extension in lower case, without the dot ("" when it has none).
    static func fileExtension(_ name: String) -> String {
        guard let dot = name.lastIndex(of: "."), dot != name.startIndex else { return "" }
        return String(name[name.index(after: dot)...]).lowercased()
    }

    /// The website's kindOfName: the MIME type first, then the extension.
    static func of(name: String, mime: String? = nil) -> UploadKind {
        let type = (mime ?? "").lowercased()
        let ext = fileExtension(name)
        if type == "application/pdf" || ext == "pdf" { return .pdf }
        if type.hasPrefix("image/") { return .image }
        if type.hasPrefix("video/") { return .video }
        if videoExtensions.contains(ext) { return .video }
        if imageExtensions.contains(ext) { return .image }
        if documentExtensions.contains(ext) || documentMimePrefixes.contains(where: { type.hasPrefix($0) }) { return .document }
        return .other
    }

    /// The SF Symbol for the kind.
    var systemImage: String {
        switch self {
        case .image: return "photo"
        case .video: return "film"
        case .pdf, .document: return "doc.text"
        case .other: return "doc"
        }
    }

    /// The record's kind (SyncRecord.kind).
    var recordKind: String { self == .video ? "video" : "image" }
}

/// Where a queued file is in its life (the website's Phase).
enum UploadPhase: String, Equatable {
    /// In the queue, not started yet.
    case waiting
    /// Exported from Photos or hashed on the phone, before the upload is created.
    case checking
    /// Sending bytes.
    case uploading
    /// On the server, which is storing it (queued or processing).
    case processing
    /// The server's answer is final: stored, duplicate, rejected or error.
    case done
    /// Not uploaded: refused on the phone (empty or too large) or failed on the way.
    case failed
    case cancelled

    /// The phase of a queued record: its sync status (SyncRecord.status), the server's last word on it
    /// (`serverStatus`) and whether the queue is checking it right now.
    static func of(status: String, serverStatus: String?, checking: Bool) -> UploadPhase {
        switch status {
        case "cancelled": return .cancelled
        case "failed", "skipped": return .failed
        case "uploading": return .uploading
        case "uploaded", "synced":
            switch serverStatus {
            case "stored", "duplicate", "rejected", "error": return .done
            case "cancelled": return .cancelled
            default: return .processing
            }
        default: return checking ? .checking : .waiting
        }
    }

    /// Nothing more happens to it (the website's isFinished).
    var isFinished: Bool { self == .done || self == .failed || self == .cancelled }

    /// The website's canCancel.
    var canCancel: Bool { self == .waiting || self == .checking || self == .uploading }
}

/// What the queue's totals and the overall progress need from one row.
struct UploadRowInfo: Equatable {
    var phase: UploadPhase
    var serverStatus: String?
    var bytes: Int64
    /// 0...1 while uploading.
    var progress: Double?
    /// Refused on the phone (empty or too large): retrying is pointless.
    var skipped = false

    /// The website's canRetry: a failure on the way, a cancel, or a processing error on the server.
    var canRetry: Bool {
        if skipped { return false }
        if phase == .failed || phase == .cancelled { return true }
        return phase == .done && serverStatus == "error"
    }
}

/// The queue's totals (the website's UploadSummary).
struct UploadSummary: Equatable {
    var total = 0
    var added = 0
    var duplicates = 0
    var failed = 0
    /// Still waiting, checking or uploading: bytes still to send.
    var sending = 0
    var processing = 0
    var cancelled = 0

    static func of(_ rows: [UploadRowInfo]) -> UploadSummary {
        var s = UploadSummary()
        s.total = rows.count
        for r in rows {
            switch r.phase {
            case .waiting, .checking, .uploading: s.sending += 1
            case .processing: s.processing += 1
            case .failed: s.failed += 1
            case .cancelled: s.cancelled += 1
            case .done:
                switch r.serverStatus {
                case "stored": s.added += 1
                case "duplicate": s.duplicates += 1
                case "rejected", "error": s.failed += 1
                default: break
                }
            }
        }
        return s
    }

    /// Finished rows, which Clear finished takes away.
    var finished: Int { total - sending - processing }

    /// Bytes done over bytes to send, counting only files that are (or were) being uploaded (overallProgress).
    static func progress(_ rows: [UploadRowInfo]) -> Double {
        var total: Double = 0
        var done: Double = 0
        for r in rows where r.phase != .failed && r.phase != .cancelled {
            let size = Double(r.bytes)
            total += size
            switch r.phase {
            case .uploading: done += size * (r.progress ?? 0)
            case .processing, .done: done += size
            default: break
            }
        }
        return total > 0 ? done / total : 0
    }

    /// "Uploading 3 files, 42%", "Processing 2 files" or "All done".
    func headline(progress: Double) -> String {
        if sending > 0 { return "Uploading \(Fmt.count(sending, "file")), \(Int((progress * 100).rounded()))%" }
        if processing > 0 { return "Processing \(Fmt.count(processing, "file"))" }
        return "All done"
    }

    /// "18 added · 3 already in your library · 1 not uploaded" (the outcomes so far).
    var outcomes: String {
        var parts: [String] = []
        if added > 0 { parts.append("\(added.formatted()) added") }
        if duplicates > 0 { parts.append("\(duplicates.formatted()) already in your library") }
        if failed > 0 { parts.append("\(failed.formatted()) not uploaded") }
        if cancelled > 0 { parts.append("\(cancelled.formatted()) cancelled") }
        return parts.joined(separator: " · ")
    }
}

/// The words of the upload screen.
enum UploadText {
    /// What an add did (the website's addNotice).
    static func notice(added: Int, rejected: Int, skipped: Int) -> String {
        var out: [String] = []
        if added > 0 { out.append("\(added.formatted()) \(added == 1 ? "file" : "files") added to the queue.") }
        if rejected > 0 {
            out.append("\(rejected.formatted()) \(rejected == 1 ? "file was" : "files were") skipped because "
                       + "\(rejected == 1 ? "it is" : "they are") empty or too large.")
        }
        if skipped > 0 { out.append("\(skipped.formatted()) \(skipped == 1 ? "file is" : "files are") already in the list.") }
        return out.joined(separator: " ")
    }

    /// "16 GB" style label for a size limit (the website's formatLimit).
    static func limit(_ bytes: Int64) -> String {
        let gb = Double(bytes) / Double(1024 * 1024 * 1024)
        if gb >= 1 {
            return gb == gb.rounded() ? "\(Int(gb)) GB" : String(format: "%.1f GB", gb)
        }
        return "\(Int((Double(bytes) / Double(1024 * 1024)).rounded())) MB"
    }

    /// Why a file is refused before anything is sent (the website's localRejection), or nil.
    static func rejection(size: Int64, limit: Int64) -> String? {
        if size > limit { return "This file is larger than the \(UploadText.limit(limit)) limit." }
        if size <= 0 { return "This file is empty." }
        return nil
    }

    /// The badge of a finished row (the website's OUTCOMES, FailedBadge and CancelledBadge).
    static func badge(phase: UploadPhase, serverStatus: String?, skipped: Bool) -> String? {
        switch phase {
        case .done:
            switch serverStatus {
            case "stored": return "Added"
            case "duplicate": return "Already in your library"
            case "rejected": return "Not accepted"
            default: return "Processing failed"
            }
        case .failed: return skipped ? "Skipped" : "Not uploaded"
        case .cancelled: return "Cancelled"
        default: return nil
        }
    }

    /// The phase words beside the size ("Uploading 40%").
    static func phase(_ phase: UploadPhase, progress: Double?, resumed: Bool) -> String? {
        switch phase {
        case .waiting: return "Waiting"
        case .checking:
            if let progress { return "Checking \(Int((progress * 100).rounded()))%" }
            return resumed ? "Resuming" : "Checking"
        case .uploading: return "\(resumed ? "Resuming" : "Uploading") \(Int(((progress ?? 0) * 100).rounded()))%"
        case .processing: return "Processing"
        default: return nil
        }
    }

    /// Why a file on its way waits (the website's QueueRow): the phone is offline, or a request failed and is tried
    /// again in `retryIn` seconds. Nil when neither.
    static func network(offline: Bool, retryIn: Int?) -> String? {
        if offline { return "You are offline. The upload continues when the connection is back." }
        if let retryIn, retryIn > 0 { return "Connection problem. Trying again in \(retryIn) s." }
        return nil
    }

    /// Whole seconds until `date`, rounded up; nil once it has passed.
    static func secondsUntil(_ date: Date?, now: Date) -> Int? {
        guard let date else { return nil }
        let left = date.timeIntervalSince(now)
        return left > 0 ? Int(left.rounded(.up)) : nil
    }

    /// The three notes under the queue.
    static let notes: [UploadNote] = [
        UploadNote(systemImage: "checkmark.shield", text: "Originals are kept byte for byte. Downloads give back the exact file."),
        UploadNote(systemImage: "sparkles",
                   text: "Photos are sorted, faces and people found, and documents read and indexed after upload."),
        UploadNote(systemImage: "doc.on.doc", text: "Exact copies of files already here are not stored twice."),
    ]
}

/// One of the notes under the queue.
struct UploadNote: Hashable {
    let systemImage: String
    let text: String
}
