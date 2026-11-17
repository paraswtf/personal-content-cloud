import SwiftUI

/// Who a picker lists: people, pets, or both (people first, then a Pets section; a new name offers both).
enum PickerKind: String {
    case person, pet, all

    var listKind: API.ListPeopleKind {
        switch self {
        case .person: return .person
        case .pet: return .pet
        case .all: return .all
        }
    }
}

/// What is being named, so the picker lists the most alike named people first (POST /api/people/similar,
/// docs/ARCHITECTURE.md section 10, naming by likeness).
struct LikenessSource: Hashable {
    var faceIds: [Int] = []
    var bodyIds: [Int] = []
    var animalIds: [Int] = []
    /// With `personId`: that person's boxes on these items. Without items, `personId` means the whole person.
    var uids: [String] = []
    var personId: Int?
    /// An item: its unnamed faces are compared (tagging a whole item).
    var aroundUid: String?

    static func faces(_ ids: [Int]) -> LikenessSource { LikenessSource(faceIds: ids) }
    static func bodies(_ ids: [Int]) -> LikenessSource { LikenessSource(bodyIds: ids) }
    static func animals(_ ids: [Int]) -> LikenessSource { LikenessSource(animalIds: ids) }
    static func person(_ id: Int, on uids: [String] = []) -> LikenessSource { LikenessSource(uids: uids, personId: id) }
    static func item(_ uid: String) -> LikenessSource { LikenessSource(aroundUid: uid) }

    /// The box under the finger in the viewer.
    static func box(_ box: StageBox) -> LikenessSource {
        switch box.kind {
        case .face: return faces([box.boxId])
        case .body: return bodies([box.boxId])
        case .animal: return animals([box.boxId])
        }
    }

    /// Nothing to compare.
    var isEmpty: Bool {
        faceIds.isEmpty && bodyIds.isEmpty && animalIds.isEmpty && personId == nil && (aroundUid ?? "").isEmpty
    }

    /// The request body, with sorted, unique ids so the same source always asks the same question.
    func request(limit: Int) -> API.SimilarIn {
        func ids(_ xs: [Int]) -> [Int]? { xs.isEmpty ? nil : Array(Set(xs)).sorted() }
        var body = API.SimilarIn()
        body.faceIds = ids(faceIds)
        body.bodyIds = ids(bodyIds)
        body.animalIds = ids(animalIds)
        body.uids = (uids.isEmpty || personId == nil) ? nil : Array(Set(uids)).sorted()
        body.personId = personId
        body.aroundUid = (aroundUid ?? "").isEmpty ? nil : aroundUid
        body.limit = limit
        return body
    }
}

/// One choice in the person picker.
struct PersonOption: Identifiable, Hashable {
    let id: Int
    /// The name to show ("Unnamed dog" for a pet nobody named; people without a name are never offered).
    let name: String
    var coverRef: String?
    /// "person" or "pet".
    var kind: String = "person"
    var species: String?
    /// The line under the name: why it is offered ("Looks like them, 4 items") or its counts ("12 faces, 30 items").
    var detail: String?
    /// How alike it is to what is being named (0 to 1), when the picker ranks by likeness.
    var score: Double?
    /// The matching would take this person: shown as "Likely".
    var likely = false

    /// A person or pet from the API, or nil for an unnamed person.
    init?(id: Int, name: String?, coverRef: String?, kind: String?, species: String?, detail: String? = nil,
          score: Double? = nil, likely: Bool = false) {
        let kindText = kind ?? "person"
        guard kindText == "pet" || Fmt.isNamed(name) else { return nil }
        self.id = id
        self.name = Fmt.personName(name, kind: kindText, species: species)
        self.coverRef = coverRef
        self.kind = kindText
        self.species = species
        self.detail = detail
        self.score = score
        self.likely = likely
    }
}

/// What the owner picked: an existing person, or a new name (with "person" or "pet").
enum PersonPick: Hashable {
    case existing(PersonOption)
    case new(name: String, kind: String)
}

/// A row of the picker: a person in a section, or "Create" a new name.
enum PickerRow: Identifiable, Hashable {
    enum Section: String { case alike, rest, pets, create }

    case person(PersonOption, section: Section)
    case create(name: String, kind: String)

    var id: String {
        switch self {
        case .person(let option, _): return "p\(option.id)"
        case .create(let name, let kind): return "c-\(kind)-\(name)"
        }
    }

    var section: Section {
        switch self {
        case .person(_, let section): return section
        case .create: return .create
        }
    }
}

/// The picker's rows, as the website builds them (components/people/PersonPicker.tsx): with nothing typed, the most
/// alike named people first, then everyone else; typing filters by name and keeps the order of likeness.
enum PickerOptions {
    /// How many of the most alike people the empty box lists first.
    static let alikeLimit = 6

    static func build(listed: [API.PersonOut], ranking: API.SimilarOut?, typed rawTyped: String, kind: PickerKind,
                      excluding: Set<Int>, species: String?, speciesOnly: Bool, allowCreate: Bool) -> [PickerRow] {
        let typed = rawTyped.trimmingCharacters(in: .whitespacesAndNewlines)
        func fits(id: Int, kind personKind: String, name: String?, species personSpecies: String?) -> Bool {
            !excluding.contains(id)
                && (kind == .all || personKind == kind.rawValue)
                && (personKind == "pet" || Fmt.isNamed(name))
                && !(speciesOnly && species != nil && personSpecies != species)
        }
        func fitsListed(_ p: API.PersonOut) -> Bool { fits(id: p.id, kind: p.kind.rawValue, name: p.name, species: p.species) }
        func isPet(_ p: API.PersonOut) -> Bool { p.kind.rawValue == "pet" }
        func restSection(_ p: API.PersonOut) -> PickerRow.Section { kind == .all && isPet(p) ? .pets : .rest }

        // Both kinds: people first, then pets. Pets of the box's species first.
        var byKind = kind == .all ? listed.filter { !isPet($0) } + listed.filter(isPet) : listed
        if let species {
            byKind = byKind.enumerated().sorted { a, b in
                let sa = a.element.species == species, sb = b.element.species == species
                return sa != sb ? sa : a.offset < b.offset
            }.map(\.element)
        }
        let found = byKind.filter(fitsListed)
        var rows: [PickerRow] = []

        if let ranking, typed.isEmpty {
            let listedById = Dictionary(found.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
            let top = ranking.items.filter {
                fits(id: $0.person.id, kind: $0.person.kind.rawValue, name: $0.person.name, species: $0.person.species)
            }
            for alike in top {
                if let option = option(listed: listedById[alike.person.id], alike: alike) {
                    rows.append(.person(option, section: .alike))
                }
            }
            let shown = Set(top.map { $0.person.id })
            for p in found where !shown.contains(p.id) {
                if let option = option(p, alike: nil) { rows.append(.person(option, section: restSection(p))) }
            }
        } else if let ranking {
            let alikeById = Dictionary(ranking.items.map { ($0.person.id, $0) }, uniquingKeysWith: { first, _ in first })
            func score(_ p: API.PersonOut) -> Double { ranking.scores[String(p.id)] ?? -1 }
            func group(_ p: API.PersonOut) -> Int { kind == .all && isPet(p) ? 1 : 0 }
            let ordered = found.enumerated().sorted { a, b in
                if group(a.element) != group(b.element) { return group(a.element) < group(b.element) }
                if score(a.element) != score(b.element) { return score(a.element) > score(b.element) }
                return a.offset < b.offset
            }.map(\.element)
            for p in ordered {
                if let option = option(p, alike: alikeById[p.id]) { rows.append(.person(option, section: restSection(p))) }
            }
        } else {
            for p in found {
                if let option = option(p, alike: nil) { rows.append(.person(option, section: restSection(p))) }
            }
        }

        if allowCreate, !typed.isEmpty {
            let exact = listed.contains { ($0.name ?? "").compare(typed, options: [.caseInsensitive]) == .orderedSame }
            if !exact {
                switch kind {
                case .all:
                    rows.append(.create(name: typed, kind: "person"))
                    rows.append(.create(name: typed, kind: "pet"))
                case .pet:
                    rows.append(.create(name: typed, kind: "pet"))
                case .person:
                    rows.append(.create(name: typed, kind: "person"))
                }
            }
        }
        return rows
    }

    /// A listed person's row: an alike one says why (and how many items), the others their counts.
    static func option(_ p: API.PersonOut, alike: API.SimilarOption?) -> PersonOption? {
        let pet = p.kind.rawValue == "pet"
        let detail: String
        if let alike {
            detail = [alike.reason.isEmpty ? "Alike" : alike.reason, Fmt.count(p.itemCount, "item")].joined(separator: ", ")
        } else if pet {
            detail = "\(Fmt.speciesLabel(p.species)), \(Fmt.count(p.itemCount, "item"))"
        } else {
            detail = "\(Fmt.count(p.faceCount, "face")), \(Fmt.count(p.itemCount, "item"))"
        }
        return PersonOption(id: p.id, name: p.name, coverRef: p.coverUrl, kind: p.kind.rawValue, species: p.species,
                            detail: detail, score: alike?.score, likely: alike?.likely ?? false)
    }

    /// An alike person's row: the listed details when the list has them, else the likeness answer's (no counts).
    static func option(listed: API.PersonOut?, alike: API.SimilarOption) -> PersonOption? {
        if let listed { return option(listed, alike: alike) }
        let p = alike.person
        return PersonOption(id: p.id, name: p.name, coverRef: p.coverUrl, kind: p.kind.rawValue, species: p.species,
                            detail: alike.reason.isEmpty ? "Alike" : alike.reason, score: alike.score, likely: alike.likely)
    }
}

/// Searches people and picks one, or creates a new name. With `like`, the most alike named people come first. It uses
/// the navigation bar's search field: embed it in a NavigationStack, or use PersonPickerSheet.
struct PersonPicker: View {
    var kind: PickerKind = .person
    /// What is being named: the most alike named people are listed first.
    var like: LikenessSource?
    var excluding: Set<Int> = []
    var allowCreate = true
    /// Pets: this species first; with `speciesOnly`, only this species.
    var species: String?
    var speciesOnly = false
    var prompt: String?
    let onPick: (PersonPick) -> Void

    @State private var text = ""
    @State private var query = ""
    @State private var listed: [API.PersonOut] = []
    @State private var listLoaded = false
    @State private var ranking: API.SimilarOut?
    @State private var rankingFor: LikenessSource?
    @State private var rankingPending = false
    @State private var rankGeneration = 0
    @State private var failure: String?
    /// When the last pick was taken: a second tap within a second is ignored (`pick(_:)`).
    @State private var pickedAt: Date?

    private var activeLike: LikenessSource? {
        guard let like, !like.isEmpty else { return nil }
        return like
    }

    /// The list waits for the likeness answer, so nothing moves under the finger once it shows.
    private var waiting: Bool { activeLike != nil && rankingPending }

    private var rows: [PickerRow] {
        guard !waiting else { return [] }
        return PickerOptions.build(listed: listed, ranking: rankingFor == activeLike ? ranking : nil, typed: text,
                                   kind: kind, excluding: excluding, species: species, speciesOnly: speciesOnly,
                                   allowCreate: allowCreate)
    }

    private var hint: String {
        if let prompt { return prompt }
        switch kind {
        case .all: return "Search people and pets or type a new name"
        case .pet: return "Search pets or type a new name"
        case .person: return "Search people or type a new name"
        }
    }

    var body: some View {
        let current = rows
        List {
            if let failure {
                Text(failure)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }
            ForEach(sections(current), id: \.key) { section in
                Section {
                    ForEach(section.rows) { row in rowView(row) }
                } header: {
                    if let title = section.title { Text(title) }
                }
            }
            if current.isEmpty && !waiting && listLoaded && failure == nil {
                Text(emptyText)
                    .foregroundStyle(.secondary)
            }
        }
        .listStyle(.insetGrouped)
        .searchable(text: $text, placement: .navigationBarDrawer(displayMode: .always), prompt: hint)
        .autocorrectionDisabled()
        .textInputAutocapitalization(.words)
        .overlay {
            if current.isEmpty && (waiting || !listLoaded) && failure == nil {
                ProgressView(waiting ? "Finding who looks alike" : "Searching")
            }
        }
        .task(id: text) { await loadList() }
        .task(id: activeLike) { await loadRanking() }
        .accessibilityIdentifier("picker")
    }

    private var emptyText: String {
        if !text.trimmingCharacters(in: .whitespaces).isEmpty {
            return kind == .pet ? "No pet matches that name." : "No one matches that name."
        }
        switch kind {
        case .all: return "No people or pets yet. Type a name to add one."
        case .pet: return "No pets yet. Type a name to add one."
        case .person: return "No people yet. Type a name to create one."
        }
    }

    private struct RowSection {
        let key: String
        let title: String?
        let rows: [PickerRow]
    }

    /// Consecutive rows of one section under one heading: "Most alike", "Everyone else" (after the alike ones),
    /// "Pets"; the plain list and "Create" have none.
    private func sections(_ rows: [PickerRow]) -> [RowSection] {
        var out: [RowSection] = []
        var bucket: [PickerRow] = []
        var current: PickerRow.Section?
        let hasAlike = rows.contains { $0.section == .alike }
        func flush() {
            guard let section = current, !bucket.isEmpty else { return }
            let title: String?
            switch section {
            case .alike: title = "Most alike"
            case .rest: title = hasAlike ? "Everyone else" : nil
            case .pets: title = "Pets"
            case .create: title = nil
            }
            out.append(RowSection(key: "\(section.rawValue)-\(out.count)", title: title, rows: bucket))
            bucket = []
        }
        for row in rows {
            if row.section != current {
                flush()
                current = row.section
            }
            bucket.append(row)
        }
        flush()
        return out
    }

    @ViewBuilder
    private func rowView(_ row: PickerRow) -> some View {
        switch row {
        case .person(let option, _):
            Button {
                pick(.existing(option))
            } label: {
                PersonOptionRow(option: option)
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("picker.option")
        case .create(let name, let newKind):
            Button {
                pick(.new(name: name, kind: newKind))
            } label: {
                Label(newKind == "pet" ? "New pet \u{201C}\(name)\u{201D}" : "Create \u{201C}\(name)\u{201D}",
                      systemImage: newKind == "pet" ? "pawprint" : "person.badge.plus")
            }
            .accessibilityIdentifier("picker.create")
        }
    }

    /// Hands a pick on, once: a fast second tap on a row or on Create (the sheet is still going away) would send the
    /// save twice or make two people of one new name. After a second the picker takes a pick again, for when the
    /// save failed and the picker is still there.
    private func pick(_ choice: PersonPick) {
        if let at = pickedAt, Date().timeIntervalSince(at) < 1 { return }
        pickedAt = Date()
        onPick(choice)
    }

    private func loadList() async {
        let typed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // A short pause while typing, so each key does not send a request.
        if !typed.isEmpty || listLoaded {
            try? await Task.sleep(nanoseconds: 200_000_000)
            if Task.isCancelled { return }
        }
        query = typed
        do {
            let page = try await Cloud.api().listPeople(q: typed.isEmpty ? nil : typed,
                                                        filter: kind == .person ? .named : .all,
                                                        sort: typed.isEmpty ? .items : .name,
                                                        kind: kind.listKind, limit: 30)
            if Task.isCancelled { return }
            listed = page.items
            listLoaded = true
            failure = nil
        } catch {
            if error is CancellationError || Task.isCancelled { return }
            listLoaded = true
            failure = "Could not load people. \(ErrorText.describe(error))"
        }
    }

    /// The likeness answer is read once per source and kept while the picker is open.
    private func loadRanking() async {
        guard let source = activeLike else {
            rankingPending = false
            return
        }
        if rankingFor == source, ranking != nil { return }
        rankGeneration += 1
        let generation = rankGeneration
        rankingPending = true
        defer { if generation == rankGeneration { rankingPending = false } }
        do {
            let answer = try await Cloud.api().similarPeople(body: source.request(limit: PickerOptions.alikeLimit))
            if Task.isCancelled { return }
            ranking = answer
            rankingFor = source
        } catch {
            if Task.isCancelled { return }
            // Without the likeness answer the plain list shows.
            ranking = nil
            rankingFor = nil
        }
    }
}

/// A person row: picture, name, why or how many, and "Likely" when the matching would take them.
struct PersonOptionRow: View {
    let option: PersonOption

    var body: some View {
        HStack(spacing: 12) {
            Avatar(name: option.name, ref: option.coverRef, size: 40)
            VStack(alignment: .leading, spacing: 2) {
                Text(option.name)
                    .font(.body)
                    .lineLimit(1)
                if let detail = option.detail {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 8)
            if option.likely {
                Chip(text: "Likely", tint: .accentColor)
            }
            if option.kind == "pet" {
                Image(systemName: "pawprint")
                    .foregroundStyle(.secondary)
                    .accessibilityHidden(true)
            }
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
    }
}

/// The picker in its own sheet, with a title and Cancel.
struct PersonPickerSheet: View {
    var title = "Who is this?"
    var kind: PickerKind = .person
    var like: LikenessSource?
    var excluding: Set<Int> = []
    var allowCreate = true
    var species: String?
    var speciesOnly = false
    let onPick: (PersonPick) -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            PersonPicker(kind: kind, like: like, excluding: excluding, allowCreate: allowCreate, species: species,
                         speciesOnly: speciesOnly) { pick in
                dismiss()
                onPick(pick)
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            }
        }
    }
}
