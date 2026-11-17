import Foundation

// The system side of Settings: reading the parts of GET /api/system and GET /api/system/learning that the generator
// leaves as JSON values, and the website's words for them (components/shell/settings: SystemTab, StorageCards,
// VideoCard, ThroughputCard, LearningTab; components/undo/UndoHistory). Pure functions, so the unit tests can
// check the wording.

/// Plain values out of a JSON value; a missing or null value reads as zero, false or nil.
enum SettingsJSON {
    static func int(_ v: API.JSONValue?) -> Int {
        guard let v else { return 0 }
        if let i = v.intValue { return i }
        if let d = v.doubleValue, d.isFinite { return Int(d) }
        return 0
    }

    static func optionalInt(_ v: API.JSONValue?) -> Int? {
        guard let v, !v.isNull else { return nil }
        if let i = v.intValue { return i }
        if let d = v.doubleValue, d.isFinite { return Int(d) }
        return nil
    }

    static func double(_ v: API.JSONValue?) -> Double? {
        guard let d = v?.doubleValue, d.isFinite else { return nil }
        return d
    }

    static func bool(_ v: API.JSONValue?) -> Bool { v?.boolValue ?? false }

    /// A non-empty string, else nil (the website treats "" as nothing).
    static func string(_ v: API.JSONValue?) -> String? {
        guard let s = v?.stringValue, !s.isEmpty else { return nil }
        return s
    }

    static func strings(_ v: API.JSONValue?) -> [String] { v?.arrayValue?.compactMap(\.stringValue) ?? [] }

    /// True for an object (even an empty one), the way the website tests `weights`.
    static func isObject(_ v: API.JSONValue?) -> Bool { v?.objectValue != nil }
}

/// A colour for a state: the website's ok, warn, danger and idle tones.
enum SettingsTone {
    case ok, warn, danger, idle
}

/// The free-form parts of GET /api/system, read into plain values.
struct SettingsSystemStatus {
    struct GPU {
        var available: Bool
        var name: String?
        var provider: String?
    }

    struct QueueLine: Identifiable {
        var name: String
        var queued: Int
        var running: Int
        var id: String { name }
    }

    struct Queues {
        var queued: Int
        var running: Int
        var failed: Int
        var byQueue: [QueueLine]
        var busy: Bool { queued + running > 0 }
    }

    struct Download {
        var running: Bool
        var progress: String?
        var error: String?
    }

    struct Video {
        var encoder: String?
        var hw: Bool
        var device: String?
        var reason: String?
        var queued: Int
        var running: Int
        var failed: Int
        var done: Int
        var pending: Int

        init(_ v: [String: API.JSONValue]) {
            encoder = SettingsJSON.string(v["encoder"])
            hw = SettingsJSON.bool(v["hw"])
            device = SettingsJSON.string(v["device"])
            reason = SettingsJSON.string(v["reason"])
            queued = SettingsJSON.int(v["queued"])
            running = SettingsJSON.int(v["running"])
            failed = SettingsJSON.int(v["failed"])
            done = SettingsJSON.int(v["done"])
            pending = SettingsJSON.int(v["pending"])
        }
    }

    struct Migration {
        var active: Bool
        var running: Bool
        var pendingFiles: Int
        var pendingBytes: Int
        var movedFiles: Int
        var movedBytes: Int
        var error: String?

        var percent: Int {
            pendingFiles > 0 ? min(100, Int((Double(movedFiles) / Double(pendingFiles) * 100).rounded())) : 0
        }
    }

    struct BackupProgress {
        var phase: String
        var filesTotal: Int
        var filesDone: Int
        var bytesCopied: Int

        var percent: Int {
            filesTotal > 0 ? min(100, Int((Double(filesDone) / Double(filesTotal) * 100).rounded())) : 0
        }
    }

    struct BackupRecord {
        var finishedAt: String?
        var ok: Bool
        var error: String?
        var files: Int
        var bytesCopied: Int
        var bytesLinked: Int
    }

    struct Forwards {
        var unchecked: Int
        var running: Bool
    }

    var gpu: GPU
    var queues: Queues
    var download: Download?
    var video: Video?
    var migration: Migration?
    var backupProgress: BackupProgress?
    var backupLast: BackupRecord?
    var forwards: Forwards?

    init(_ s: API.SystemOut) {
        gpu = GPU(available: SettingsJSON.bool(s.gpu["available"]), name: SettingsJSON.string(s.gpu["name"]),
                  provider: SettingsJSON.string(s.gpu["provider"]))
        let lines = (s.queues["by_queue"]?.objectValue ?? [:])
            .map { QueueLine(name: $0.key, queued: SettingsJSON.int($0.value["queued"]), running: SettingsJSON.int($0.value["running"])) }
            .sorted { SettingsSystemText.queueOrder($0.name) < SettingsSystemText.queueOrder($1.name) }
        queues = Queues(queued: SettingsJSON.int(s.queues["queued"]), running: SettingsJSON.int(s.queues["running"]),
                        failed: SettingsJSON.int(s.queues["failed"]), byQueue: lines)
        if let d = s.download {
            download = Download(running: SettingsJSON.bool(d["running"]), progress: SettingsJSON.string(d["progress"]),
                                error: SettingsJSON.string(d["error"]))
        }
        if let v = s.video { video = Video(v) }
        if let m = s.storage?.migration {
            migration = Migration(active: SettingsJSON.bool(m["active"]), running: SettingsJSON.bool(m["running"]),
                                  pendingFiles: SettingsJSON.int(m["pending_files"]),
                                  pendingBytes: SettingsJSON.int(m["pending_bytes"]),
                                  movedFiles: SettingsJSON.int(m["moved_files"]),
                                  movedBytes: SettingsJSON.int(m["moved_bytes"]), error: SettingsJSON.string(m["error"]))
        }
        if let p = s.backup?.progress {
            backupProgress = BackupProgress(phase: SettingsJSON.string(p["phase"]) ?? "",
                                            filesTotal: SettingsJSON.int(p["files_total"]),
                                            filesDone: SettingsJSON.int(p["files_done"]),
                                            bytesCopied: SettingsJSON.int(p["bytes_copied"]))
        }
        if let l = s.backup?.last {
            backupLast = BackupRecord(finishedAt: SettingsJSON.string(l["finished_at"]), ok: SettingsJSON.bool(l["ok"]),
                                      error: SettingsJSON.string(l["error"]), files: SettingsJSON.int(l["files"]),
                                      bytesCopied: SettingsJSON.int(l["bytes_copied"]),
                                      bytesLinked: SettingsJSON.int(l["bytes_linked"]))
        }
        if let f = s.forwards {
            forwards = Forwards(unchecked: SettingsJSON.int(f["unchecked"]), running: SettingsJSON.bool(f["running"]))
        }
    }

    /// The models that are missing or failed.
    static func missingModels(_ s: API.SystemOut) -> Int {
        s.models.filter { !$0.present || ($0.error?.isEmpty == false) }.count
    }

    /// A download runs (the server's download, or a model that says it is downloading).
    func downloadRunning(_ s: API.SystemOut) -> Bool {
        (download?.running ?? false) || s.models.contains { $0.downloading }
    }
}

/// One disk of the storage card: a tier of GET /api/system's storage, or the single folder of an older server.
struct SettingsTier: Identifiable {
    var key: String
    var path: String
    var holds: String
    var freeBytes: Int
    var totalBytes: Int
    var sharedWith: String?

    var id: String { key }

    var usedPercent: Int {
        totalBytes > 0 ? min(100, Int(((1 - Double(freeBytes) / Double(totalBytes)) * 100).rounded())) : 0
    }

    var lowSpace: Bool { totalBytes > 0 && Double(freeBytes) / Double(totalBytes) < 0.1 }

    static func tiers(_ s: API.SystemOut) -> [SettingsTier] {
        if let storage = s.storage {
            return storage.tiers.map {
                SettingsTier(key: $0.key, path: $0.path, holds: $0.holds, freeBytes: $0.freeBytes,
                             totalBytes: $0.totalBytes, sharedWith: $0.sharedWith)
            }
        }
        return [SettingsTier(key: "originals", path: "", holds: "everything", freeBytes: s.disk.freeBytes,
                             totalBytes: s.disk.totalBytes, sharedWith: nil)]
    }
}

/// What the learning models know (GET /api/system/learning `learners`), read into plain values.
struct SettingsLearners {
    struct Scan {
        /// False only when the server says the reading is turned off.
        var enabled: Bool
        var scanned: Int
        var eligible: Int
        var ready: Bool
        var paused: [String]
        var lastError: String?

        init?(_ v: API.JSONValue?) {
            guard let o = v?.objectValue else { return nil }
            enabled = o["enabled"]?.boolValue ?? true
            scanned = SettingsJSON.int(o["scanned"])
            eligible = SettingsJSON.int(o["eligible"])
            ready = SettingsJSON.bool(o["ready"])
            paused = SettingsJSON.strings(o["paused"])
            lastError = SettingsJSON.string(o["last_error"])
        }
    }

    struct Traits {
        var source: String
        var tattooMinInliers: Int
        var outcomes: Int?
        var pairs: Int?
        var hardNegatives: Int?
        var scan: Scan?

        var adjusted: Bool { source != "default" }
        var reading: Bool {
            guard let scan, scan.enabled else { return false }
            return !scan.ready && scan.paused.isEmpty
        }
    }

    var docTypeExamples: Int
    var docTypeLabels: [String]
    var categoryExamples: Int
    var categoryLabels: [String]
    var vaultExamples: Int
    var vaultThreshold: Double?
    var vaultWeights: Bool
    var misclickExamples: Int
    var misclickThresholds: [String: Double]
    var peoplePersons: Int
    var peopleEvents: Int
    var peopleWeights: Bool
    /// nil on servers without tag learning.
    var tags: (tags: Int, examples: Int, rejections: Int)?
    /// nil on servers without forwards.
    var forwards: (examples: Int, learned: Bool)?
    var traits: Traits?
    var placeScan: Scan?

    init(_ l: [String: API.JSONValue]) {
        docTypeExamples = SettingsJSON.int(l["doc_type"]?["examples"])
        docTypeLabels = SettingsJSON.strings(l["doc_type"]?["labels"])
        categoryExamples = SettingsJSON.int(l["category"]?["examples"])
        categoryLabels = SettingsJSON.strings(l["category"]?["labels"])
        vaultExamples = SettingsJSON.int(l["vault"]?["examples"])
        vaultThreshold = SettingsJSON.double(l["vault"]?["threshold"])
        vaultWeights = SettingsJSON.isObject(l["vault"]?["weights"])
        misclickExamples = SettingsJSON.int(l["misclick"]?["examples"])
        var cuts: [String: Double] = [:]
        for (key, value) in l["misclick"]?["thresholds"]?.objectValue ?? [:] {
            if let d = SettingsJSON.double(value) { cuts[key] = d }
        }
        misclickThresholds = cuts
        peoplePersons = SettingsJSON.int(l["people"]?["persons"])
        peopleEvents = SettingsJSON.int(l["people"]?["events"])
        peopleWeights = SettingsJSON.isObject(l["people"]?["weights"])
        if let t = l["tags"]?.objectValue {
            tags = (SettingsJSON.int(t["tags"]), SettingsJSON.int(t["examples"]), SettingsJSON.int(t["rejections"]))
        }
        if let f = l["forwards"]?.objectValue {
            forwards = (SettingsJSON.int(f["examples"]), SettingsJSON.bool(f["learned"]))
        }
        if let t = l["traits"]?.objectValue {
            traits = Traits(source: SettingsJSON.string(t["source"]) ?? "default",
                            tattooMinInliers: SettingsJSON.int(t["tattoo_min_inliers"]),
                            outcomes: SettingsJSON.optionalInt(t["outcomes"]),
                            pairs: SettingsJSON.optionalInt(t["pairs"]),
                            hardNegatives: SettingsJSON.optionalInt(t["hard_negatives"]),
                            scan: Scan(t["scan"]))
        }
        placeScan = Scan(l["place"]?["scan"])
    }

    /// Nothing learned anywhere yet (the website's nothingLearned).
    var nothingLearned: Bool {
        docTypeExamples == 0 && categoryExamples == 0 && vaultExamples == 0 && !vaultWeights && misclickExamples == 0
            && peoplePersons == 0 && !peopleWeights && (tags?.tags ?? 0) == 0 && (tags?.examples ?? 0) == 0
            && (forwards?.examples ?? 0) == 0 && !(forwards?.learned ?? false)
    }

    /// Reading tattoos and nails, or lining photos up, is still going: worth asking again.
    var scanning: Bool {
        if let place = placeScan, place.enabled, !place.ready, place.scanned < place.eligible { return true }
        guard let scan = traits?.scan, scan.enabled, !scan.ready, scan.paused.isEmpty else { return false }
        return !(scan.eligible > 0 && scan.scanned >= scan.eligible)
    }
}

/// The website's words for the system side of Settings.
enum SettingsSystemText {
    // MARK: System

    static func queueOrder(_ name: String) -> Int {
        ["main", "text", "video"].firstIndex(of: name) ?? 99
    }

    /// The queues' names (QUEUE_LABELS).
    static func queueLabel(_ name: String) -> String {
        switch name {
        case "main": return "Analysis (photos, videos, faces)"
        case "text": return "Text and documents"
        case "video": return "Video previews"
        default: return name
        }
    }

    /// "GPU (CUDA)", "CPU", or the provider without "ExecutionProvider"; "-" for none.
    static func providerLabel(_ provider: String?) -> String {
        guard let p = provider, !p.isEmpty else { return "-" }
        if p.hasPrefix("CUDA") { return "GPU (CUDA)" }
        if p.hasPrefix("CPU") { return "CPU" }
        if p.hasSuffix("ExecutionProvider") { return String(p.dropLast("ExecutionProvider".count)) }
        return p
    }

    /// A model's state in words, with its tone (modelState).
    static func modelState(_ m: API.ModelStatus) -> (text: String, tone: SettingsTone) {
        if let e = m.error, !e.isEmpty { return ("Error", .danger) }
        if m.downloading { return ("Downloading", .warn) }
        if !m.present { return ("Missing", .danger) }
        if m.loaded { return ("Loaded", .ok) }
        return ("Ready, loads on first use", .idle)
    }

    /// The download button's words.
    static func downloadLabel(missing: Int, running: Bool) -> String {
        if running { return "Downloading" }
        return missing == 1 ? "Download the missing model" : "Download missing models"
    }

    /// The encoder line of the video card (encoderLine).
    static func encoderLine(_ v: SettingsSystemStatus.Video) -> String {
        guard let encoder = v.encoder else {
            return "Not detected yet. The first video that needs a copy sets it, or press Test encoders."
        }
        let names = ["h264_nvenc": "NVIDIA NVENC", "h264_vaapi": "VAAPI", "h264_qsv": "Intel Quick Sync",
                     "libx264": "libx264 (software)"]
        let name = names[encoder] ?? encoder
        if !v.hw { return "\(name) on the CPU" }
        if let device = v.device { return "\(encoder) on \(device)" }
        return "\(encoder) (\(name))"
    }

    /// A storage tier's title (TIER_TITLES).
    static func tierTitle(_ key: String) -> String {
        switch key {
        case "originals": return "Originals"
        case "data": return "Database and previews"
        case "backup": return "Backups"
        default: return key
        }
    }

    /// "42% used", with the low-space warning of the tier.
    static func tierUsage(_ t: SettingsTier) -> String {
        var text = "\(t.usedPercent)% used"
        guard t.lowSpace else { return text }
        switch t.key {
        case "originals": text += ". Free space is low; new uploads may be refused."
        case "data": text += ". Free space is low; previews and the database need room here."
        case "backup": text += ". Free space is low; old snapshots are removed after each run."
        default: break
        }
        return text
    }

    /// The last backup in one sentence (lastLine).
    static func backupLastLine(_ last: SettingsSystemStatus.BackupRecord?, now: Date = Date()) -> String {
        guard let last else { return "No snapshot yet." }
        let when: String
        if let finished = last.finishedAt {
            when = "\(Fmt.relative(finished, now: now)) (\(Fmt.date(finished, withTime: true)))"
        } else {
            when = "just now"
        }
        if !last.ok { return "The last run \(when) failed: \(last.error ?? "unknown error")." }
        let linked = last.bytesLinked > 0
            ? ", \(Fmt.bytes(last.bytesLinked)) unchanged and shared with the night before" : ""
        return "Last snapshot \(when): \(last.files.formatted()) originals, \(Fmt.bytes(last.bytesCopied)) copied\(linked)."
    }

    /// What a running backup is doing.
    static func backupPhase(_ p: SettingsSystemStatus.BackupProgress) -> String {
        switch p.phase {
        case "originals":
            return "\(p.filesDone.formatted()) of \(p.filesTotal.formatted()) originals, \(Fmt.bytes(p.bytesCopied)) copied"
        case "database":
            return "Copying the database and the learned models"
        default:
            return "Starting"
        }
    }

    /// "03:00".
    static func hour(_ h: Int) -> String { String(format: "%02d:00", h) }

    /// The move of an earlier layout's originals.
    static func migrationLine(_ m: SettingsSystemStatus.Migration) -> String {
        if m.running {
            return "Moving the originals to their own disk: \(m.movedFiles.formatted()) of \(m.pendingFiles.formatted()) files "
                + "(\(Fmt.bytes(m.movedBytes)) of \(Fmt.bytes(m.pendingBytes)))"
        }
        return "Some originals still sit in the data folder. They are moved when the server starts."
    }

    static func retryMessage(_ requeued: Int) -> String {
        if requeued == 0 { return "There were no failed uploads to retry." }
        return "\(requeued.formatted()) failed \(requeued == 1 ? "upload was" : "uploads were") queued again."
    }

    static func requeueMessage(_ requeued: Int) -> String {
        requeued == 0 ? "No failed copies to re-run." : "\(requeued) \(requeued == 1 ? "copy" : "copies") queued again."
    }

    // MARK: Processing speed

    struct Period {
        let range: API.ThroughputRange
        let label: String
        let per: String
        let window: String
        let unit: String
    }

    static let periods: [Period] = [
        Period(range: .hour, label: "Last hour", per: "per minute", window: "in the last hour", unit: "minute"),
        Period(range: .day, label: "24 hours", per: "per hour", window: "in the last 24 hours", unit: "hour"),
        Period(range: .week, label: "7 days", per: "per 6 hours", window: "in the last 7 days", unit: "six hours"),
    ]

    static func period(_ range: API.ThroughputRange) -> Period {
        periods.first { $0.range == range } ?? periods[0]
    }

    /// The top of a chart's scale: 1, 2, 2.5 or 5 times a power of ten (niceMax).
    static func niceMax(_ v: Int) -> Int {
        if v <= 4 { return max(1, v) }
        let pow = pow10(Int(log10(Double(v)).rounded(.down)))
        for m in [1.0, 2.0, 2.5, 5.0, 10.0] where m * Double(pow) >= Double(v) {
            return Int(m * Double(pow))
        }
        return 10 * pow
    }

    private static func pow10(_ e: Int) -> Int {
        var p = 1
        for _ in 0..<max(0, e) { p *= 10 }
        return p
    }

    /// "164 ms", "1.5 s", "12 s", "1 min 15 s" (formatMs).
    static func duration(ms: Int?) -> String {
        guard let ms else { return "" }
        if ms < 1000 { return "\(ms) ms" }
        let s = Double(ms) / 1000
        if s < 60 { return s < 10 ? String(format: "%.1f s", s) : "\(Int(s.rounded())) s" }
        return "\(Int(s / 60)) min \(Int(s.truncatingRemainder(dividingBy: 60).rounded())) s"
    }

    /// "4 in the last hour, 164 ms each on average, 1 failed".
    static func typeSummary(_ t: API.ThroughputType, window: String) -> String {
        var text = "\(t.total.formatted()) \(window)"
        if let avg = t.avgMs { text += ", \(duration(ms: avg)) each on average" }
        if t.failed > 0 { text += ", \(t.failed.formatted()) failed" }
        return text
    }

    /// "0.4 a minute now" or "idle now".
    static func rateLine(_ rate: Double) -> String {
        rate > 0 ? "\(rate.formatted(.number.precision(.fractionLength(0...1)))) a minute now" : "idle now"
    }

    /// One bucket's time span: "10:10 to 10:11", with the day for the week view.
    static func bucketLabel(_ iso: String, bucketSeconds: Int, withDay: Bool) -> String {
        guard let start = ISODate.parse(iso) else { return iso }
        let end = start.addingTimeInterval(TimeInterval(bucketSeconds))
        let time: (Date) -> String = { $0.formatted(date: .omitted, time: .shortened) }
        if withDay {
            return "\(start.formatted(.dateTime.weekday(.abbreviated).day().month(.abbreviated))), \(time(start)) to \(time(end))"
        }
        return "\(time(start)) to \(time(end))"
    }

    /// The x axis label of a bucket: a time, or a day for the week view.
    static func axisLabel(_ iso: String, range: API.ThroughputRange) -> String {
        guard let d = ISODate.parse(iso) else { return "" }
        if range == .week { return d.formatted(.dateTime.weekday(.abbreviated).day()) }
        return d.formatted(date: .omitted, time: .shortened)
    }

    // MARK: Learning

    static let misclickKeys = ["blur_var_min", "dark_mean", "clipped_frac", "accident_min"]
    static let misclickDefaults: [String: Double] = ["blur_var_min": 40, "dark_mean": 28, "clipped_frac": 0.6,
                                                     "accident_min": 0.6]
    static let misclickNames: [String: String] = ["blur_var_min": "blur threshold", "dark_mean": "darkness threshold",
                                                  "clipped_frac": "blown-out threshold",
                                                  "accident_min": "pocket-shot threshold"]

    private static func cut(_ key: String, _ v: Double) -> String {
        key == "blur_var_min" || key == "dark_mean" ? String(Int(v.rounded())) : String(format: "%.2f", v)
    }

    /// Learned cuts that differ from the built-in ones: "blur threshold now 55 (default 40)".
    static func misclickChanges(_ thresholds: [String: Double]) -> [String] {
        misclickKeys.compactMap { (key: String) -> String? in
            guard let v = thresholds[key], v.isFinite, let d = misclickDefaults[key] else { return nil }
            if abs(v - d) <= abs(d) * 0.005 { return nil }
            return "\(misclickNames[key] ?? key) now \(cut(key, v)) (default \(cut(key, d)))"
        }
    }

    /// "Aadhaar card, PAN card, Passport, Voter ID and 2 more".
    static func labelList(_ labels: [String], max: Int = 4, _ toLabel: (String) -> String) -> String {
        let names = labels.map(toLabel)
        if names.count <= max { return names.joined(separator: ", ") }
        return "\(names.prefix(max).joined(separator: ", ")) and \(names.count - max) more"
    }

    /// The document types' names (components/documents/docMeta.ts).
    static func docTypeLabel(_ type: String?) -> String {
        guard let type, !type.isEmpty else { return "Document" }
        let labels = [
            "aadhaar": "Aadhaar card", "pan": "PAN card", "passport": "Passport", "driving_licence": "Driving licence",
            "voter_id": "Voter ID", "vehicle_rc": "Vehicle RC", "id_card": "ID card", "bank_statement": "Bank statement",
            "invoice": "Invoice", "receipt": "Receipt", "traffic_challan": "Traffic challan",
            "parking_receipt": "Parking receipt", "salary_slip": "Salary slip", "marksheet": "Marksheet",
            "degree": "Degree", "certificate": "Certificate", "offer_letter": "Offer letter",
            "insurance_policy": "Insurance policy", "prescription": "Prescription", "lab_report": "Lab report",
            "vaccination": "Vaccination record", "event_pass": "Event pass", "boarding_pass": "Boarding pass",
            "other": "Document",
        ]
        if let label = labels[type] { return label }
        let spaced = type.replacingOccurrences(of: "_", with: " ")
        return spaced.prefix(1).uppercased() + spaced.dropFirst()
    }

    /// The categories' names (CATEGORY_LABELS).
    static func categoryLabel(_ c: String) -> String {
        let labels = ["personal": "People", "places": "Places", "food": "Food", "pets": "Pets", "objects": "Things",
                      "documents": "Documents", "screenshots": "Screenshots", "misc": "Other", "nsfw": "Private"]
        return labels[c] ?? c
    }

    /// The kinds of corrections recorded, in the website's order (EVENT_LABELS).
    static let eventKinds: [(key: String, label: String)] = [
        ("doc_type", "Document type changes"), ("category", "Category changes"), ("vault", "Vault moves"),
        ("misclick", "Mis-click decisions"), ("guess", "Guesses confirmed or rejected"),
        ("tag", "Suggested tags removed"), ("forward", "Forward decisions"), ("species", "Species corrections"),
    ]

    static func sensitivityLabel(_ v: API.VaultSensitivity) -> String {
        switch v {
        case .low: return "Low"
        case .normal: return "Normal"
        case .high: return "High"
        case .unknown(let raw): return raw
        }
    }

    static func sensitivityHint(_ v: API.VaultSensitivity) -> String {
        switch v {
        case .low: return "Low puts fewer photos in the vault: only clearly explicit ones."
        case .high: return "High puts more photos in the vault, including suggestive but not explicit ones."
        default: return "Normal puts explicit photos in the vault. Low puts in fewer, High puts in more."
        }
    }

    /// What the immediate re-sort after a sensitivity change did (movedText).
    static func movedText(_ moved: [String: Int]?) -> String {
        guard let moved else { return "" }
        var parts: [String] = []
        let movedIn = moved["moved_in"] ?? 0, movedOut = moved["moved_out"] ?? 0, stale = moved["stale"] ?? 0
        if movedIn > 0 { parts.append("\(movedIn) moved into the vault") }
        if movedOut > 0 { parts.append("\(movedOut) moved out") }
        let head = parts.isEmpty ? "Nothing moved." : parts.joined(separator: ", ") + "."
        let tail = stale > 0
            ? " \(stale) older \(stale == 1 ? "item needs" : "items need") \"Apply to existing items\" to be checked with the new signals."
            : ""
        return head + tail
    }

    /// The line under the switch: when it last trained and what waits.
    static func trainedLine(_ s: API.LearningStatus, now: Date = Date()) -> String {
        let trained = s.trainedAt != nil
        let first = trained ? "Last trained \(Fmt.relative(s.trainedAt, now: now))." : "Not trained yet."
        let second: String
        if s.pending > 0 {
            let when = trained ? "since then" : "waiting for the first training"
            let tail = s.enabled ? ", learned at the next training (every 10 minutes, or Train now)." : "."
            second = "\(Fmt.count(s.pending, "correction")) \(when)\(tail)"
        } else {
            second = trained ? "No new corrections since then." : "No corrections yet."
        }
        return "\(first) \(second)"
    }

    /// The message after Apply to existing items.
    static func applyMessage(items: Int, documents: Int, tags: Int) -> String {
        if items + documents + tags == 0 { return "Nothing needed a fresh look." }
        if items + documents == 0 { return "\(Fmt.count(tags, "tag")) suggested on existing items." }
        let tagged = tags > 0 ? ", and suggested \(Fmt.count(tags, "tag"))" : ""
        return "Queued \(Fmt.count(items, "item")) and \(Fmt.count(documents, "document")) for a fresh look\(tagged). "
            + "Progress shows under System."
    }

    // MARK: Undo history

    /// "just now", "5 min ago", "3 h ago", "2 days ago", then a date (Fmt.relative for a Date).
    static func relative(_ date: Date, now: Date = Date()) -> String {
        let s = max(0, Int(now.timeIntervalSince(date).rounded()))
        if s < 45 { return "just now" }
        let m = Int((Double(s) / 60).rounded())
        if m < 60 { return "\(m) min ago" }
        let h = Int((Double(m) / 60).rounded())
        if h < 24 { return "\(h) h ago" }
        let d = Int((Double(h) / 24).rounded())
        if d < 7 { return "\(d) \(d == 1 ? "day" : "days") ago" }
        return Fmt.date(date)
    }

    /// The line under an undo history entry: when, and whether it was undone or is too old.
    static func undoStatus(_ e: UndoCenter.Entry, now: Date = Date()) -> String {
        var parts = [relative(e.at, now: now)]
        if e.expired {
            parts.append("too old to undo")
        } else if e.state == .undone {
            parts.append(e.undoId != nil ? "undone" : "undone elsewhere")
        }
        if e.partial { parts.append("partly, as some of it had changed since") }
        return parts.joined(separator: ", ")
    }
}
