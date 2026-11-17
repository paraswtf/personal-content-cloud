import Foundation

/// The website's words and rules for the viewer's details: categories (lib/types.ts), mis-click reasons
/// (components/library/misclick.ts), forwards (components/library/forwards.ts), the quality rows and score bars of
/// the info panel, the text sources of TextPanel, the document labels and field order (components/documents/
/// docMeta.ts), and the video's tracks gathered by person (components/viewer/boxes.ts).
enum ViewerFormat {
    // MARK: categories

    /// The categories the owner can pick, in the website's order (Private belongs to the vault, it is not a pick).
    static let categories: [API.Category] = [.personal, .places, .food, .pets, .objects, .documents, .screenshots, .misc]

    static func categoryLabel(_ raw: String) -> String {
        switch raw {
        case "personal": return "People"
        case "places": return "Places"
        case "food": return "Food"
        case "pets": return "Pets"
        case "objects": return "Things"
        case "documents": return "Documents"
        case "screenshots": return "Screenshots"
        case "misc": return "Other"
        case "nsfw": return "Private"
        default: return sentence(raw)
        }
    }

    /// A category's score (0 to 1).
    struct Score: Hashable {
        let key: String
        let value: Double
    }

    /// A labelled value of the details.
    struct Line: Hashable {
        let label: String
        let value: String
    }

    /// The five highest category scores, highest first (the website's ScoreBars).
    static func topScores(_ scores: [String: Double]?) -> [Score] {
        (scores ?? [:])
            .filter { $0.value.isFinite }
            .sorted { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }
            .prefix(5)
            .map { Score(key: $0.key, value: $0.value) }
    }

    // MARK: mis-clicks

    static let reasonOrder = ["blurry", "dark", "blown_out", "pocket", "burst"]

    static func reasonLabel(_ key: String) -> String {
        switch key {
        case "blurry": return "Blurry"
        case "dark": return "Too dark"
        case "blown_out": return "Blown out"
        case "pocket": return "Pocket shot"
        case "burst": return "Burst copy"
        default: return sentence(key)
        }
    }

    /// The reason keys of an item in a stable order: `{reason: detail}`, or `{reasons: [...]}`.
    static func reasonKeys(_ reasons: [String: API.JSONValue]?) -> [String] {
        guard let reasons else { return [] }
        if case .array(let list)? = reasons["reasons"] {
            return list.compactMap { $0.stringValue }
        }
        func rank(_ key: String) -> Int { reasonOrder.firstIndex(of: key) ?? 99 }
        return reasons.keys
            .filter { $0 != "burst_best" && $0 != "best_uid" }
            .sorted { rank($0) == rank($1) ? $0 < $1 : rank($0) < rank($1) }
    }

    /// The uid of the sharpest shot of a burst, when the reasons name it.
    static func burstBestUid(_ reasons: [String: API.JSONValue]?) -> String? {
        guard let reasons else { return nil }
        if case .object(let burst)? = reasons["burst"], let uid = burst["best_uid"]?.stringValue { return uid }
        return reasons["best_uid"]?.stringValue
    }

    // MARK: forwards and received media

    static func appLabel(_ via: String) -> String {
        switch via {
        case "whatsapp": return "WhatsApp"
        case "telegram": return "Telegram"
        case "facebook": return "Facebook"
        case "snapchat": return "Snapchat"
        case "messenger": return "Messenger"
        case "app": return "a messaging app"
        default: return via
        }
    }

    /// "Received on WhatsApp", or "Received through a messaging app" when the app is not known.
    static func receivedLabel(_ via: String?) -> String? {
        guard let via, !via.isEmpty else { return nil }
        return via == "app" ? "Received through a messaging app" : "Received on \(appLabel(via))"
    }

    /// Why an item was set apart, in the website's order: "Meme 0.81", "Text", "Small, no camera data", ...
    static func forwardReasons(_ reasons: [String: API.JSONValue]?) -> [String] {
        guard let r = reasons else { return [] }
        var out: [String] = []
        if case .number(let meme)? = r["meme_p"], meme.isFinite { out.append("Meme \(String(format: "%.2f", meme))") }
        if truthy(r["text"]) { out.append("Text") }
        let small = truthy(r["small"]), stripped = truthy(r["stripped"])
        if small && stripped { out.append("Small, no camera data") } else if small { out.append("Small") } else if stripped { out.append("No camera data") }
        if truthy(r["video_small"]) { out.append("Small video") }
        if case .number(let q)? = r["jpeg_quality"], q.isFinite { out.append("JPEG quality \(Int(q.rounded()))") }
        if case .string(let name)? = r["name"], !name.isEmpty { out.append("Filename from \(appLabel(name))") }
        return out
    }

    // MARK: quality

    /// The quality checks the website lists, with its words.
    static func qualityLines(_ scores: [String: API.JSONValue]?) -> [Line] {
        guard let scores else { return [] }
        let rows: [(String, String, (Double) -> String)] = [
            ("blur_var", "Sharpness", { "\(Int($0.rounded())) (blurry under 40)" }),
            ("motion_blur_p", "Motion blur", { Fmt.percent($0) }),
            ("mean_lum", "Brightness", { "\(Int($0.rounded())) of 255" }),
            ("p95_lum", "Highlights", { "\(Int($0.rounded())) of 255" }),
            ("clipped_frac", "Blown out", { Fmt.percent($0) }),
            ("accident_p", "Accidental shot", { Fmt.percent($0) }),
        ]
        return rows.compactMap { key, label, format in
            guard case .number(let v)? = scores[key], v.isFinite else { return nil }
            return Line(label: label, value: format(v))
        }
    }

    /// The checks were skipped for this item ("skipped" is there and not false).
    static func qualitySkipped(_ scores: [String: API.JSONValue]?) -> Bool {
        guard let value = scores?["skipped"] else { return false }
        if case .bool(false) = value { return false }
        if case .null = value { return false }
        return true
    }

    // MARK: text

    static func textSource(_ source: String) -> String {
        switch source {
        case "pdf": return "from the PDF text"
        case "ocr": return "read from the picture"
        case "office": return "from the document"
        case "plain": return "plain text"
        case "mixed": return "from the text and the picture"
        default: return source
        }
    }

    /// Long text starts folded (the website's "Show all").
    static func isLongText(_ text: String) -> Bool {
        text.count > 600 || text.split(separator: "\n", omittingEmptySubsequences: false).count > 12
    }

    // MARK: documents

    static let lowConfidence = 0.6
    static let challanPortal = "https://echallan.parivahan.gov.in/index/accused-challan"

    static func docTypeLabel(_ type: String?) -> String {
        guard let type, !type.isEmpty else { return "Document" }
        let labels: [String: String] = [
            "aadhaar": "Aadhaar card", "pan": "PAN card", "passport": "Passport", "driving_licence": "Driving licence",
            "voter_id": "Voter ID", "vehicle_rc": "Vehicle RC", "id_card": "ID card", "bank_statement": "Bank statement",
            "invoice": "Invoice", "receipt": "Receipt", "traffic_challan": "Traffic challan",
            "parking_receipt": "Parking receipt", "salary_slip": "Salary slip", "marksheet": "Marksheet",
            "degree": "Degree", "certificate": "Certificate", "offer_letter": "Offer letter",
            "insurance_policy": "Insurance policy", "prescription": "Prescription", "lab_report": "Lab report",
            "vaccination": "Vaccination record", "event_pass": "Event pass", "boarding_pass": "Boarding pass",
            "other": "Document",
        ]
        return labels[type] ?? sentence(type)
    }

    static func groupLabel(_ group: String?) -> String {
        switch group {
        case "id": return "Identity"
        case "money": return "Money"
        case "education_work": return "Education and work"
        case "health": return "Health and insurance"
        case "pass": return "Passes and tickets"
        default: return "Other"
        }
    }

    /// The symbol of a document's kind: its type's when the group's says too little, else its group's.
    static func docSymbol(type: String?, group: String?) -> String {
        switch type {
        case "id_card": return "person.text.rectangle"
        case "traffic_challan": return "exclamationmark.triangle"
        case "parking_receipt": return "parkingsign.circle"
        default: break
        }
        switch group {
        case "id": return "person.text.rectangle"
        case "money": return "banknote"
        case "education_work": return "graduationcap"
        case "health": return "heart.text.square"
        case "pass": return "ticket"
        default: return "doc.text"
        }
    }

    /// The field order and labels the website knows on its own; the server's order and labels win.
    static let typeFields: [String: [(key: String, label: String)]] = [
        "traffic_challan": [
            ("challan_number", "Challan number"), ("date", "Date"), ("vehicle_number", "Vehicle number"),
            ("vehicle_owner", "Vehicle owner"), ("owner_address", "Owner address"), ("violations", "Violations"),
            ("amount", "Amount"), ("payment_status", "Payment status"), ("location", "Location"), ("officer", "Officer"),
            ("mobile", "Mobile"), ("evidences", "Evidences"), ("pay_url", "Pay at"),
        ],
        "parking_receipt": [
            ("receipt_number", "Receipt number"), ("vehicle_number", "Vehicle number"), ("vehicle_type", "Vehicle type"),
            ("in_time", "In"), ("out_time", "Out"), ("duration", "Duration"), ("amount", "Amount"),
            ("operator", "Operator"), ("location", "Location"), ("tariff", "Tariff"),
        ],
    ]

    /// Field keys in display order: the server's order (else the type's own) first, then any others by name.
    static func orderedKeys(fieldOrder: [String], fieldKeys: [String], docType: String) -> [String] {
        let order = fieldOrder.isEmpty ? (typeFields[docType] ?? []).map(\.key) : fieldOrder
        let present = Set(fieldKeys)
        var keys = order.filter { present.contains($0) }
        for key in fieldKeys.sorted() where !keys.contains(key) { keys.append(key) }
        return keys
    }

    /// A field's label: the server's, else the type's own, else the key.
    static func fieldLabel(key: String, serverLabel: String?, docType: String) -> String {
        if let serverLabel, !serverLabel.isEmpty { return serverLabel }
        return typeFields[docType]?.first(where: { $0.key == key })?.label ?? key
    }

    /// The value as the card shows it: the in and out times of a receipt as dates with the time.
    static func displayValue(key: String, value: String) -> String {
        if key == "in_time" || key == "out_time" {
            let shown = Fmt.date(value, withTime: true)
            return shown.isEmpty ? value : shown
        }
        return value
    }

    /// Identifiers read better in a fixed-width font (numbers, PNRs, ids).
    static func isMonospaced(_ key: String) -> Bool {
        let k = key.lowercased()
        return k.contains("number") || k.hasSuffix("no") || k.contains("pnr") || k.hasSuffix("id")
    }

    /// A challan's payment status reads as unpaid (warning colour).
    static func isUnpaid(_ status: String) -> Bool {
        let s = status.lowercased()
        return s.contains("unpaid") || s.contains("pending") || s.contains("due")
    }

    // MARK: files

    /// "PDF", "DOCX": the file's extension, upper case, for a file without a preview.
    static func extLabel(_ filename: String) -> String {
        guard let dot = filename.lastIndex(of: "."), filename.distance(from: dot, to: filename.endIndex) <= 6 else { return "FILE" }
        let ext = filename[filename.index(after: dot)...]
        guard !ext.isEmpty, ext.allSatisfy({ $0.isLetter || $0.isNumber }) else { return "FILE" }
        return ext.uppercased()
    }

    /// The details panel's title for an item of this kind.
    static func detailsTitle(_ kind: MediaKind) -> String {
        switch kind {
        case .video: return "Video details"
        case .pdf, .document: return "Document details"
        case .image: return "Photo details"
        case .other: return "File details"
        }
    }

    /// The word the trash confirmation uses.
    static func trashNoun(_ kind: MediaKind) -> String {
        switch kind {
        case .video: return "video"
        case .image: return "photo"
        default: return "file"
        }
    }

    /// The video's technical line: "H264, 25 fps, 4.2 Mbit/s, no sound".
    static func videoLine(codec: String?, fps: Double?, bitrate: Int?, hasAudio: Bool?) -> String {
        var parts: [String] = []
        if let codec, !codec.isEmpty { parts.append(codec.uppercased()) }
        if let fps, fps > 0 {
            let rounded = (fps * 100).rounded() / 100
            parts.append(rounded == rounded.rounded() ? "\(Int(rounded)) fps" : "\(rounded) fps")
        }
        if let bitrate, bitrate > 0 { parts.append(String(format: "%.1f Mbit/s", Double(bitrate) / 1e6)) }
        if hasAudio == false { parts.append("no sound") }
        return parts.joined(separator: ", ")
    }

    // MARK: video tracks

    /// One face, person box or pet seen at one moment of a video.
    struct VideoTrack: Identifiable, Hashable {
        let box: StageBox
        let time: Double
        let crop: String
        /// The person's or pet's name, a guess's name with a question mark, or "Not tagged".
        let name: String
        let isGuess: Bool
        let isNamed: Bool
        let group: Int?
        let origin: BoxOrigin?

        var id: String { box.id }
    }

    /// One person or look-alike of a video: their tracks by time.
    struct VideoGroup: Identifiable, Hashable {
        let key: String
        let tracks: [VideoTrack]

        var id: String { key }
        var first: VideoTrack { tracks[0] }

        /// The owner's origin when any track is the owner's (the website's mergeOrigins).
        var origin: BoxOrigin? {
            let all = tracks.compactMap(\.origin)
            for o in [BoxOrigin.you, .drawn, .confirmed, .auto, .guess] where all.contains(o) { return o }
            return nil
        }
    }

    /// The video's faces, person boxes with someone on them and pets, by time (the website's videoBoxes). Only
    /// named people are named; a box of someone not named reads "Not tagged".
    static func videoTracks(_ item: API.ItemDetail) -> [VideoTrack] {
        var out: [VideoTrack] = []
        for f in item.faces {
            guard let t = f.frameTimeS else { continue }
            let named = f.person.flatMap { Fmt.isNamed($0.name) ? $0.name : nil }
            let guess = named == nil ? f.guess.flatMap { Fmt.isNamed($0.person.name) ? $0.person.name : nil } : nil
            let box = StageBox(kind: .face, boxId: f.id, x: f.x, y: f.y, w: f.w, h: f.h, name: named, guessName: guess,
                               personId: f.person?.id)
            let origin = BoxOrigin.of(hasPerson: f.person != nil, assignedByUser: f.assignedBy == .user,
                                      drawnByUser: f.source == .user, byGuess: f.byGuess, hasGuess: guess != nil)
            out.append(VideoTrack(box: box, time: t, crop: f.url, name: named ?? guess.map { "\($0)?" } ?? "Not tagged",
                                  isGuess: named == nil && guess != nil, isNamed: named != nil, group: f.group,
                                  origin: origin))
        }
        for b in item.bodies {
            guard let t = b.frameTimeS, b.faceId == nil else { continue }
            let named = b.person.flatMap { Fmt.isNamed($0.name) ? $0.name : nil }
            let guess = named == nil ? b.guess.flatMap { Fmt.isNamed($0.person.name) ? $0.person.name : nil } : nil
            guard named != nil || guess != nil else { continue }
            let box = StageBox(kind: .body, boxId: b.id, x: b.x, y: b.y, w: b.w, h: b.h, name: named, guessName: guess,
                               personId: b.person?.id)
            let origin = BoxOrigin.of(hasPerson: b.person != nil, assignedByUser: b.assignedBy == .user,
                                      drawnByUser: b.source == .user, byGuess: b.byGuess, hasGuess: guess != nil)
            out.append(VideoTrack(box: box, time: t, crop: b.url, name: named ?? guess.map { "\($0)?" } ?? "Not tagged",
                                  isGuess: named == nil, isNamed: named != nil, group: b.group, origin: origin))
        }
        for a in item.animals {
            guard let t = a.frameTimeS else { continue }
            let named = a.person.flatMap { Fmt.isNamed($0.name) ? $0.name : nil }
            let label = named ?? Fmt.speciesLabel(a.species)
            let box = StageBox(kind: .animal, boxId: a.id, x: a.x, y: a.y, w: a.w, h: a.h, name: label, guessName: nil,
                               personId: a.person?.id)
            let origin = BoxOrigin.of(hasPerson: a.person != nil, assignedByUser: a.assignedBy == .user,
                                      drawnByUser: a.source == .user)
            out.append(VideoTrack(box: box, time: t, crop: a.url, name: label, isGuess: false, isNamed: named != nil,
                                  group: a.group, origin: origin))
        }
        return out.sorted { $0.time == $1.time ? $0.id < $1.id : $0.time < $1.time }
    }

    /// The tracks gathered by their group, in order of first appearance; a track without a group stands alone.
    static func groupTracks(_ tracks: [VideoTrack]) -> [VideoGroup] {
        var order: [String] = []
        var groups: [String: [VideoTrack]] = [:]
        for t in tracks {
            let key = t.group.map { "group:\($0)" } ?? "solo:\(t.id)"
            if groups[key] == nil { order.append(key) }
            groups[key, default: []].append(t)
        }
        return order.map { key in VideoGroup(key: key, tracks: groups[key]!.sorted { $0.time < $1.time }) }
    }

    // MARK: helpers

    /// JavaScript's truthiness, for the reasons' flags.
    static func truthy(_ value: API.JSONValue?) -> Bool {
        switch value {
        case .bool(let b)?: return b
        case .number(let n)?: return n != 0 && !n.isNaN
        case .string(let s)?: return !s.isEmpty
        case .array?, .object?: return true
        default: return false
        }
    }

    /// "blown_out" -> "Blown out".
    static func sentence(_ key: String) -> String {
        let spaced = key.replacingOccurrences(of: "_", with: " ")
        guard let first = spaced.first else { return spaced }
        return first.uppercased() + spaced.dropFirst()
    }
}
