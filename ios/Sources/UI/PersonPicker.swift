import SwiftUI

/// One choice in the person picker: a named person or pet.
struct PersonOption: Identifiable, Hashable {
    let id: Int
    let name: String
    var coverRef: String?
    /// "person" or "pet".
    var kind: String = "person"
    var species: String?
    /// A short line under the name: "12 items", or why it is offered first ("Most alike").
    var detail: String?
    /// How alike this person is to what is being named, when the source ranks by likeness (0 to 1).
    var score: Double?

    /// A named person from the API, or nil for an unnamed one: the picker never offers unnamed people.
    init?(id: Int, name: String?, coverRef: String?, kind: String?, species: String?, detail: String? = nil,
          score: Double? = nil) {
        guard let name, Fmt.isNamed(name) else { return nil }
        self.id = id
        self.name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        self.coverRef = coverRef
        self.kind = kind ?? "person"
        self.species = species
        self.detail = detail
        self.score = score
    }
}

/// What the owner picked: an existing person, or a new name (with "person" or "pet").
enum PersonPick: Hashable {
    case existing(PersonOption)
    case new(name: String, kind: String)
}

/// Where the picker's options come from. The default lists named people by how many items they have and searches
/// them by name; a ranked source (the people most alike a face or a box, from the similarity route) plugs in here.
protocol PersonOptionSource {
    /// Options for the typed text ("" for the list shown before typing), best first. Never unnamed people.
    func options(matching text: String) async throws -> [PersonOption]
}

/// Named people (or pets) from `GET /api/people?filter=named`.
struct NamedPeopleSource: PersonOptionSource {
    /// "person", "pet" or "all".
    var kind = "person"
    var limit = 60

    func options(matching text: String) async throws -> [PersonOption] {
        guard let server = APIConnection.current else { return [] }
        let typed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        var query = [
            URLQueryItem(name: "filter", value: "named"),
            URLQueryItem(name: "kind", value: kind),
            URLQueryItem(name: "sort", value: typed.isEmpty ? "items" : "name"),
            URLQueryItem(name: "limit", value: String(limit)),
        ]
        if !typed.isEmpty { query.append(URLQueryItem(name: "q", value: typed)) }
        let page = try await CoreHTTP.get(CorePersonPage.self, server, "/api/people", query: query)
        return page.items.compactMap { p in
            PersonOption(id: p.id, name: p.name, coverRef: p.coverUrl, kind: p.kind, species: p.species,
                         detail: p.itemCount.map { Fmt.count($0, "item") })
        }
    }
}

/// Searches people and picks one, or creates a new name. Embed it in a NavigationStack (it uses the search
/// field of the navigation bar), or use PersonPickerSheet.
struct PersonPicker: View {
    var source: any PersonOptionSource = NamedPeopleSource()
    var excluding: Set<Int> = []
    var allowCreate = true
    /// The kind a new name is created as.
    var createKind = "person"
    var prompt = "Search people or type a new name"
    let onPick: (PersonPick) -> Void

    @State private var text = ""
    @State private var options: [PersonOption] = []
    @State private var loading = false
    @State private var failure: String?

    private var shown: [PersonOption] { options.filter { !excluding.contains($0.id) } }

    /// The typed name when it matches nobody exactly (offered as "Create").
    private var newName: String? {
        let name = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard allowCreate, !name.isEmpty else { return nil }
        let exists = options.contains { $0.name.compare(name, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame }
        return exists ? nil : name
    }

    var body: some View {
        List {
            if let failure {
                Text(failure).foregroundStyle(.red).font(.footnote)
            }
            if let name = newName {
                Button {
                    onPick(.new(name: name, kind: createKind))
                } label: {
                    Label("Create \u{201C}\(name)\u{201D}", systemImage: createKind == "pet" ? "pawprint" : "person.badge.plus")
                }
                .accessibilityIdentifier("picker.create")
            }
            ForEach(shown) { option in
                Button {
                    onPick(.existing(option))
                } label: {
                    PersonOptionRow(option: option)
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("picker.option")
            }
            if shown.isEmpty && newName == nil && !loading && failure == nil {
                Text(text.isEmpty ? "No named people yet." : "Nobody by that name.")
                    .foregroundStyle(.secondary)
            }
        }
        .listStyle(.plain)
        .searchable(text: $text, placement: .navigationBarDrawer(displayMode: .always), prompt: prompt)
        .autocorrectionDisabled()
        .overlay {
            if loading && options.isEmpty { ProgressView() }
        }
        .task(id: text) { await reload() }
    }

    private func reload() async {
        // A short pause while typing, so each key does not send a request.
        if !text.isEmpty {
            try? await Task.sleep(nanoseconds: 200_000_000)
            if Task.isCancelled { return }
        }
        loading = true
        defer { loading = false }
        do {
            let found = try await source.options(matching: text)
            if Task.isCancelled { return }
            options = found
            failure = nil
        } catch {
            if error is CancellationError || Task.isCancelled { return }
            failure = ErrorText.describe(error)
        }
    }
}

/// A person row: picture, name, and the detail line.
struct PersonOptionRow: View {
    let option: PersonOption

    var body: some View {
        HStack(spacing: 12) {
            Avatar(name: option.name, ref: option.coverRef, size: 40)
            VStack(alignment: .leading, spacing: 2) {
                Text(option.name).font(.body)
                if let detail = option.detail {
                    Text(detail).font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer()
            if option.kind == "pet" {
                Image(systemName: "pawprint").foregroundStyle(.secondary).accessibilityHidden(true)
            }
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
    }
}

/// The picker in its own sheet, with a title and Cancel.
struct PersonPickerSheet: View {
    var title = "Who is this?"
    var source: any PersonOptionSource = NamedPeopleSource()
    var excluding: Set<Int> = []
    var allowCreate = true
    var createKind = "person"
    let onPick: (PersonPick) -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            PersonPicker(source: source, excluding: excluding, allowCreate: allowCreate, createKind: createKind) { pick in
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
