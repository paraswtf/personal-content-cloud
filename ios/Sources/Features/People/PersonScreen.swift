import SwiftUI

/// The sections of a person's or pet's page, as the website's tabs.
enum PeoplePersonTab: String, CaseIterable, Identifiable {
    case items, faces, guesses, traits, related, threeD = "3d", crops

    var id: String { rawValue }

    var title: String {
        switch self {
        case .items: return "Items"
        case .faces: return "Faces"
        case .guesses: return "Guesses"
        case .traits: return "Tattoos and nails"
        case .related: return "Related"
        case .threeD: return "3D"
        case .crops: return "Crops"
        }
    }

    /// The tests and VoiceOver find the tabs by these.
    var identifier: String { "people.tab.\(rawValue)" }

    static func tabs(for p: API.PersonOut) -> [PeoplePersonTab] {
        if PeopleText.isPet(p) { return [.items, .crops, .related] }
        var tabs: [PeoplePersonTab] = [.items, .faces, .guesses]
        if (p.traitCount ?? 0) > 0 { tabs.append(.traits) }
        tabs.append(contentsOf: [.related, .threeD])
        return tabs
    }

    func count(_ p: API.PersonOut) -> Int? {
        switch self {
        case .items: return p.itemCount
        case .faces: return p.faceCount
        case .guesses: return p.guessCount
        case .traits: return p.traitCount
        case .related: return p.relatedCount
        case .crops: return p.animalCount
        case .threeD: return nil
        }
    }
}

/// One person's or pet's page (the website's pages/PersonPage.tsx): rename, cover, their items, their faces (move,
/// remove, use as cover), the guesses waiting for them, tattoos and nails, related items, the 3D head, merge into
/// someone else and delete. A pet shows its crops instead of faces and guesses. A named person has the card of
/// recognition trained on them alone.
struct PersonScreen: View {
    let personId: Int

    @State private var shownId: Int
    @State private var person: API.PersonOut?
    @State private var recognition: API.RecognitionOut?
    @State private var loadError: Error?
    @State private var tab: PeoplePersonTab = .items
    @State private var renaming = false
    @State private var nameText = ""
    @State private var mergePicking = false
    @State private var pendingTarget: PersonOption?
    @State private var mergeTarget: PersonOption?
    @State private var confirmDelete = false
    @Environment(\.dismiss) private var dismiss

    init(personId: Int) {
        self.personId = personId
        _shownId = State(initialValue: personId)
    }

    var body: some View {
        Group {
            if let person {
                page(person)
            } else if let loadError {
                failed(loadError)
            } else {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle(person.map { PeopleText.name($0) } ?? "Person")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button {
                        startRename()
                    } label: {
                        Label(Fmt.isNamed(person?.name) ? "Rename" : "Add a name", systemImage: "pencil")
                    }
                    Button {
                        mergePicking = true
                    } label: {
                        Label("Merge into", systemImage: "arrow.triangle.merge")
                    }
                    Button(role: .destructive) {
                        confirmDelete = true
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .disabled(person == nil)
                .accessibilityLabel("More")
                .accessibilityIdentifier("person.menu")
            }
        }
        .task(id: shownId) { await load() }
        .onCloudChange { Task { await load() } }
        .alert(Fmt.isNamed(person?.name) ? "Rename" : "Add a name", isPresented: $renaming) {
            TextField(isPet ? "Its name" : "Their name", text: $nameText)
                .accessibilityIdentifier("person.name.field")
            Button("Save") { Task { await saveName() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("A name someone else already has merges them.")
        }
        .sheet(isPresented: $mergePicking, onDismiss: {
            if let target = pendingTarget {
                pendingTarget = nil
                mergeTarget = target
            }
        }) {
            if let p = person {
                PersonPickerSheet(title: "Merge \(PeopleText.name(p)) into", kind: isPet ? .pet : .person,
                                  like: .person(p.id), excluding: [p.id], allowCreate: false, species: p.species,
                                  speciesOnly: isPet) { pick in
                    if case .existing(let option) = pick { pendingTarget = option }
                }
            }
        }
        .confirmationDialog("Merge into \(mergeTarget?.name ?? "this person")?",
                            isPresented: Binding(get: { mergeTarget != nil }, set: { if !$0 { mergeTarget = nil } }),
                            titleVisibility: .visible, presenting: mergeTarget) { target in
            Button("Merge") { Task { await merge(into: target) } }
            Button("Cancel", role: .cancel) {}
        } message: { target in
            Text("\(person.map { PeopleText.name($0) } ?? "This person") and everything tagged as \(isPet ? "it" : "them") becomes \(target.name).")
        }
        .confirmAction("Delete \(person.map { PeopleText.name($0) } ?? "")?", isPresented: $confirmDelete,
                       actionTitle: "Delete",
                       message: isPet
                           ? "The pet goes away. Its boxes stay in the photos, unassigned, and its tags are removed."
                           : "The person goes away. Their faces and bodies stay in the photos, unassigned, and their tags are removed.") {
            Task { await delete() }
        }
    }

    private var isPet: Bool { person.map(PeopleText.isPet) ?? false }

    // MARK: the page

    @ViewBuilder private func page(_ p: API.PersonOut) -> some View {
        let head = AnyView(header(p))
        let tabs = PeoplePersonTab.tabs(for: p)
        let current = tabs.contains(tab) ? tab : .items
        switch current {
        case .items:
            PeoplePersonItemsTab(person: p, related: false, header: head).id("items-\(p.id)")
        case .related:
            PeoplePersonItemsTab(person: p, related: true, header: head).id("related-\(p.id)")
        case .faces:
            PeoplePersonFacesTab(person: p, header: head).id("faces-\(p.id)")
        case .crops:
            PeoplePersonCropsTab(person: p, header: head).id("crops-\(p.id)")
        case .guesses:
            PeoplePersonGuessesTab(person: p, header: head).id("guesses-\(p.id)")
        case .traits:
            PeopleTraitsTab(person: p, header: head).id("traits-\(p.id)")
        case .threeD:
            VStack(spacing: 0) {
                tabBar(p)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                HeadsPersonTab(personId: p.id, name: PeopleText.name(p))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .id("3d-\(p.id)")
        }
    }

    private func header(_ p: API.PersonOut) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 14) {
                Avatar(name: p.name, ref: p.coverUrl, size: 72)
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 2) {
                        Text(PeopleText.name(p))
                            .font(.title2.weight(.bold))
                            .foregroundStyle(Fmt.isNamed(p.name) ? Color.primary : Color.secondary)
                            .accessibilityAddTraits(.isHeader)
                            .accessibilityIdentifier("person.name")
                        Button {
                            startRename()
                        } label: {
                            Image(systemName: "pencil").frame(width: 36, height: 36)
                        }
                        .accessibilityLabel(Fmt.isNamed(p.name) ? "Rename" : "Add a name")
                        .accessibilityIdentifier("person.rename")
                    }
                    Text(summary(p))
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("person.summary")
                    if PeopleText.isPet(p) { speciesMenu(p) }
                }
            }
            if !PeopleText.isPet(p) && Fmt.isNamed(p.name), let status = recognition,
               PeopleRecognitionCard.shows(status) {
                PeopleRecognitionCard(person: p, status: status, onStatus: { recognition = $0 })
            }
            tabBar(p)
        }
        .padding(.horizontal, 16)
        .padding(.top, 8)
    }

    private func summary(_ p: API.PersonOut) -> String {
        if PeopleText.isPet(p) {
            return "\(Fmt.speciesLabel(p.species)), \(Fmt.count(p.itemCount, "item")), \(Fmt.count(p.animalCount, "crop"))"
        }
        var parts = [Fmt.count(p.itemCount, "item"), Fmt.count(p.faceCount, "face")]
        if p.confirmedFaces > 0 { parts.append("\(p.confirmedFaces) confirmed by you") }
        if p.guessCount > 0 { parts.append("\(Fmt.count(p.guessCount, "guess", "guesses")) to review") }
        return parts.joined(separator: ", ")
    }

    private func speciesMenu(_ p: API.PersonOut) -> some View {
        Menu {
            ForEach(API.Species.allCases, id: \.self) { s in
                Button(PeopleText.speciesPlural(s.rawValue)) { Task { await setSpecies(s, on: p) } }
            }
        } label: {
            HStack(spacing: 4) {
                Text("These are \(p.species == nil ? "pets" : PeopleText.speciesPlural(p.species))")
                Image(systemName: "chevron.up.chevron.down").font(.caption2)
            }
            .font(.subheadline)
        }
        .accessibilityLabel("Species, \(p.species == nil ? "not set" : Fmt.speciesLabel(p.species))")
        .accessibilityIdentifier("person.species")
    }

    private func tabBar(_ p: API.PersonOut) -> some View {
        let tabs = PeoplePersonTab.tabs(for: p)
        let current = tabs.contains(tab) ? tab : .items
        return ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(tabs) { t in
                    Button {
                        tab = t
                    } label: {
                        HStack(spacing: 5) {
                            Text(t.title)
                            if let n = t.count(p), n > 0 {
                                Text(n.formatted())
                                    .font(.caption.monospacedDigit())
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 1)
                                    .background(Capsule().fill(Color(uiColor: .tertiarySystemFill)))
                            }
                        }
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(Capsule().fill(current == t ? Color.accentColor.opacity(0.18)
                                                               : Color(uiColor: .secondarySystemBackground)))
                        .foregroundStyle(current == t ? Color.accentColor : Color.primary)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(t.count(p).map { $0 > 0 ? "\(t.title), \($0)" : t.title } ?? t.title)
                    .accessibilityAddTraits(current == t ? .isSelected : [])
                    .accessibilityIdentifier(t.identifier)
                }
            }
        }
        .accessibilityIdentifier("people.tabs")
    }

    @ViewBuilder private func failed(_ error: Error) -> some View {
        if let api = error as? APIError, api.status == 404 {
            VStack(spacing: 16) {
                EmptyState(title: "This person is not here any more", systemImage: "person.crop.circle.badge.questionmark",
                           text: "They may have been merged into someone else or deleted.")
                Button("All people") { dismiss() }.buttonStyle(.bordered)
            }
        } else {
            ErrorState(error: error) { Task { await load() } }
        }
    }

    // MARK: loading and actions

    private func load() async {
        do {
            let p = try await Cloud.api().getPerson(personId: shownId)
            person = p
            loadError = nil
            if !PeopleText.isPet(p) && Fmt.isNamed(p.name) {
                recognition = try? await Cloud.api().personRecognition(personId: p.id)
            } else {
                recognition = nil
            }
        } catch {
            if error is CancellationError { return }
            if person == nil || (error as? APIError)?.status == 404 {
                person = nil
                loadError = error
            }
        }
    }

    private func startRename() {
        guard let p = person else { return }
        nameText = p.name ?? ""
        renaming = true
    }

    private func saveName() async {
        guard let p = person else { return }
        let text = nameText.trimmingCharacters(in: .whitespacesAndNewlines)
        let next: String? = text.isEmpty ? nil : String(text.prefix(120))
        if next == p.name { return }
        var update = API.PersonUpdate(name: next)
        if next == nil { update.nullKeys = [.name] }
        let pet = PeopleText.isPet(p)
        let answer = await PeopleAction.run(next == nil ? "Removed a name" : "Renamed someone", doing: "save the name") {
            try await Cloud.api().updatePerson(personId: p.id, body: update)
        } message: { r in
            if r.id != p.id { return "Merged into \(PeopleText.name(r)).\(Fmt.followedTail(r.followed))" }
            if !Fmt.isNamed(r.name) && !pet { return "Name removed. Their faces are back with the faces to name." }
            return Fmt.isNamed(r.name) ? "Named \(r.name ?? "")." : "Name removed."
        }
        guard let r = answer else { return }
        if r.id != p.id {
            // a name someone else already has merged this person into them: follow them
            shownId = r.id
        } else if !Fmt.isNamed(r.name) && !pet {
            // without a name, a person is a group the app keeps in the background: this page is gone
            dismiss()
        } else {
            person = r
        }
    }

    private func setSpecies(_ species: API.Species, on p: API.PersonOut) async {
        guard species.rawValue != p.species else { return }
        await PeopleAction.run("Changed a pet's species", doing: "change the species") {
            try await Cloud.api().setPersonSpecies(personId: p.id, body: API.PersonSpeciesIn(species: species))
        } message: { _ in
            "Changed to \(Fmt.speciesLabel(species.rawValue).lowercased())."
        }
    }

    private func merge(into target: PersonOption) async {
        guard let p = person else { return }
        if let merged = await PeopleMerging.merge(target: target.id, sources: [p.id]) {
            shownId = merged.id
        }
    }

    private func delete() async {
        guard let p = person else { return }
        let pet = PeopleText.isPet(p)
        let done = await PeopleAction.run(pet ? "Deleted a pet" : "Deleted a person", doing: "delete") {
            try await Cloud.api().deletePerson(personId: p.id)
        } message: { _ in
            pet ? "Pet deleted. Its crops are unassigned again." : "Person deleted. Their faces are unassigned again."
        }
        if done != nil { dismiss() }
    }
}
