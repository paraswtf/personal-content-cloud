import Foundation

/// Where a library browser looks: the whole library, one album, or the vault (every request with vault=1).
enum LibraryScope: Hashable {
    case all
    case album(id: Int)
    case vault

    /// The album's id for the endpoints' `album` parameter.
    var albumId: Int? {
        if case .album(let id) = self { return id }
        return nil
    }

    var isVault: Bool { self == .vault }
}

/// The library's filters (the website's components/library/filters.ts), one field per filter parameter of the item
/// lists (`listItems`, `buckets`, `bucket`, `points`, `placeList`, `items(bbox:)`), so other screens pass them to their
/// own endpoints. Unset is nil. `normalized(for:)` is what the website sends for them; pass its fields on.
struct LibraryFilters: Hashable {
    var category: API.Category?
    var kind: API.Kind?
    var person: [Int]?
    var tag: [String]?
    var live: Bool?
    var from: String?
    var to: String?
    var favorite: Bool?
    var misclicks: API.Forwards?
    var forwards: API.Forwards?
    var received: String?
    var dups: API.Dups?
    var sort: API.Sort?

    /// Tag names are 1 to 40 characters, and a filter takes at most 20 of them (section 19).
    static let maxTagLength = 40
    static let maxTags = 20

    /// One choice of a filter with the website's words.
    struct Choice<Value: Hashable>: Identifiable, Hashable {
        let value: Value
        let label: String
        var id: String { label }
    }

    /// The category chips, in the website's order (the vault's own category is not one of them).
    static let categories: [Choice<API.Category>] = [
        Choice(value: .personal, label: "People"), Choice(value: .places, label: "Places"),
        Choice(value: .food, label: "Food"), Choice(value: .pets, label: "Pets"),
        Choice(value: .objects, label: "Things"), Choice(value: .documents, label: "Documents"),
        Choice(value: .screenshots, label: "Screenshots"), Choice(value: .misc, label: "Other"),
    ]

    /// The Type choices, "All types" being nil.
    static let kinds: [Choice<API.Kind?>] = [
        Choice(value: nil, label: "All types"), Choice(value: .image, label: "Photos"),
        Choice(value: .video, label: "Videos"), Choice(value: .pdf, label: "PDFs"),
        Choice(value: .document, label: "Office and text files"), Choice(value: .other, label: "Other files"),
    ]

    static func categoryLabel(_ category: API.Category) -> String {
        if category == .nsfw { return "Private" }
        return categories.first(where: { $0.value == category })?.label ?? category.rawValue.capitalized
    }

    // MARK: State

    /// Filters beyond the category chips (the website opens its filter panel when one is set).
    var hasExtra: Bool {
        kind != nil || !(person ?? []).isEmpty || !(tag ?? []).isEmpty || from != nil || to != nil || favorite == true
            || live == true || (misclicks ?? .exclude) != .exclude || (forwards ?? .exclude) != .exclude
            || dups == .all
    }

    /// Any filter is set (the website shows Clear filters).
    var any: Bool { hasExtra || category != nil || sort == .added || received != nil }

    /// Something narrows the list: an empty answer then says "Nothing matches these filters".
    var narrows: Bool {
        category != nil || kind != nil || !(person ?? []).isEmpty || !(tag ?? []).isEmpty || live == true
            || from != nil || to != nil || favorite == true
    }

    // MARK: What is sent

    /// What the website sends for these filters: unset values left out, the defaults too (mis-clicks and forwards
    /// excluded, copies collapsed, sorted by date taken), dates only as yyyy-MM-dd, people without repeats, tags
    /// trimmed, 1 to 40 characters, without repeats whatever their case and at most 20, and no category in the vault.
    func normalized(for scope: LibraryScope) -> LibraryFilters {
        var f = LibraryFilters()
        if !scope.isVault, let c = category, Self.categories.contains(where: { $0.value == c }) { f.category = c }
        if let k = kind, Self.kinds.contains(where: { $0.value == k }) { f.kind = k }
        var seen = Set<Int>()
        let people = (person ?? []).filter { $0 > 0 && seen.insert($0).inserted }
        f.person = people.isEmpty ? nil : people
        let tags = Self.cleanTags(tag ?? [])
        f.tag = tags.isEmpty ? nil : tags
        f.live = live == true ? true : nil
        f.from = Self.validDay(from)
        f.to = Self.validDay(to)
        f.favorite = favorite == true ? true : nil
        if let m = misclicks, m == .include || m == .only { f.misclicks = m }
        if let w = forwards, w == .include || w == .only { f.forwards = w }
        if let r = received?.trimmingCharacters(in: .whitespacesAndNewlines), !r.isEmpty { f.received = r }
        if dups == .all { f.dups = .all }
        if sort == .added { f.sort = .added }
        return f
    }

    /// The part of the filters a search takes (the website's searchQuery): category, type, people, tags, dates and
    /// Live Photos; favourites, mis-clicks, forwards, copies and the sort do not apply to search results.
    func forSearch(in scope: LibraryScope) -> LibraryFilters {
        var f = normalized(for: scope)
        f.favorite = nil
        f.misclicks = nil
        f.forwards = nil
        f.received = nil
        f.dups = nil
        f.sort = nil
        return f
    }

    /// Trimmed, non-empty, at most 40 characters, without repeats whatever their case, at most 20.
    static func cleanTags(_ raw: [String]) -> [String] {
        var out: [String] = []
        var seen = Set<String>()
        for value in raw {
            let t = value.trimmingCharacters(in: .whitespacesAndNewlines)
            let key = t.lowercased()
            guard !t.isEmpty, t.count <= maxTagLength, !seen.contains(key) else { continue }
            seen.insert(key)
            out.append(t)
            if out.count >= maxTags { break }
        }
        return out
    }

    /// The text when it is a day written yyyy-MM-dd, else nil.
    static func validDay(_ text: String?) -> String? {
        guard let text, text.range(of: #"^\d{4}-\d{2}-\d{2}$"#, options: .regularExpression) != nil else { return nil }
        return text
    }

    private static let dayFormatter: DateFormatter = {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()

    /// "2026-03-14" for a day picked on the phone.
    static func dayString(_ date: Date) -> String { dayFormatter.string(from: date) }

    /// The day a yyyy-MM-dd text names, for a date picker.
    static func day(_ text: String?) -> Date? {
        guard let text = validDay(text) else { return nil }
        return dayFormatter.date(from: text)
    }
}

/// How a search looks for matches (the website's Search by): best match, looks like, or text.
extension API.Mode {
    var libraryLabel: String {
        switch self {
        case .semantic: return "Looks like"
        case .text: return "Text"
        default: return "Best match"
        }
    }
}

/// Labels of the parsed document types (the website's components/documents/docMeta.ts), for the tiles.
enum LibraryDocTypes {
    static let labels: [String: String] = [
        "aadhaar": "Aadhaar card", "pan": "PAN card", "passport": "Passport", "driving_licence": "Driving licence",
        "voter_id": "Voter ID", "vehicle_rc": "Vehicle RC", "id_card": "ID card", "bank_statement": "Bank statement",
        "invoice": "Invoice", "receipt": "Receipt", "traffic_challan": "Traffic challan",
        "parking_receipt": "Parking receipt", "salary_slip": "Salary slip", "marksheet": "Marksheet",
        "degree": "Degree", "certificate": "Certificate", "offer_letter": "Offer letter",
        "insurance_policy": "Insurance policy", "prescription": "Prescription", "lab_report": "Lab report",
        "vaccination": "Vaccination record", "event_pass": "Event pass", "boarding_pass": "Boarding pass",
        "other": "Document",
    ]

    /// "Boarding pass" for "boarding_pass"; an unknown type in words with a capital first letter.
    static func label(_ type: String?) -> String {
        guard let type, !type.isEmpty else { return "Document" }
        if let known = labels[type] { return known }
        let words = type.replacingOccurrences(of: "_", with: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }

    /// "PDF" for "scan.pdf", "FILE" without an extension.
    static func extensionLabel(_ filename: String) -> String {
        guard let dot = filename.lastIndex(of: ".") else { return "FILE" }
        let ext = filename[filename.index(after: dot)...]
        guard (1...5).contains(ext.count), ext.allSatisfy({ $0.isLetter || $0.isNumber }) else { return "FILE" }
        return ext.uppercased()
    }
}
