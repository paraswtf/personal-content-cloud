import SwiftUI

/// "People in N items" for a selection (the website's components/people/PeopleBulkAction.tsx, docs/ARCHITECTURE.md
/// section 10): everyone the selected items show, each with Change to, Rename and Not this person (Not this pet),
/// and Tag someone in these; on a person's or pet's page (`fromPersonId`) first "Not NAME?", which moves their
/// appearances on the items to someone else or says they are not NAME there. One sheet with steps. A move, a "Not
/// NAME" or a tag closes the sheet and runs `onDone` (the caller ends its selection); a rename keeps it open, except
/// a rename that merges the page's own person into someone else: the sheet closes and `onMerged` gets the person
/// they joined, so the page can follow them.
/// The library's and the person page's selections present it.
struct PeopleSelectionSheet: View {
    let uids: [String]
    var vault = false
    var fromPersonId: Int? = nil
    var onDone: (() -> Void)? = nil
    var onMerged: ((Int) -> Void)? = nil

    @Environment(\.dismiss) private var dismiss
    @State private var step: PeopleSelectionStep = .list
    @State private var answer: API.ItemsPeopleOut?
    @State private var loadError: Error?
    @State private var from: PeopleWho?
    @State private var related = false
    @State private var nameText = ""
    @State private var working = false

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("People in \(Fmt.count(uids.count, "item"))")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        if step == .list {
                            Button("Close") { dismiss() }.disabled(working)
                        } else {
                            Button {
                                step = .list
                            } label: {
                                Label("Back", systemImage: "chevron.left")
                            }
                            .disabled(working)
                            .accessibilityIdentifier("peopleSheet.back")
                        }
                    }
                }
                .overlay {
                    if working { ProgressView().controlSize(.large) }
                }
        }
        .interactiveDismissDisabled(working)
        .task { await load() }
        .toastHost()
    }

    private var total: Int { answer?.items ?? uids.count }

    @ViewBuilder private var content: some View {
        switch step {
        case .list:
            listStep
        case .move(let who):
            VStack(alignment: .leading, spacing: 0) {
                Text("Move \(who.display) on \(PeopleText.itemsTail(total)) to \(who.isPet ? "another pet" : "someone else"). Their faces, boxes and tags there go to the one you pick\(who.isPet ? "" : "; look-alike faces may follow").")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding(16)
                PersonPicker(kind: who.isPet ? .pet : .person, like: .person(who.id, on: uids), excluding: [who.id],
                             species: who.species, speciesOnly: who.isPet && who.species != nil) { pick in
                    Task { await move(who, PeopleChoice(pick)) }
                }
            }
        case .rename(let who):
            Form {
                Section {
                    TextField(who.isPet ? "Its name" : "Their name", text: $nameText)
                        .textInputAutocapitalization(.words)
                        .submitLabel(.done)
                        .onSubmit { Task { await rename(who) } }
                        .accessibilityLabel("New name")
                        .accessibilityIdentifier("peopleSheet.name")
                } footer: {
                    Text("\(Fmt.isNamed(who.name) ? "Rename \(who.display) everywhere, not only on these items." : "Name this \(who.isPet ? "pet" : "person") everywhere.") A name someone else already has merges them.")
                }
                Button("Save") { Task { await rename(who) } }
                    .disabled(working)
                    .accessibilityIdentifier("peopleSheet.save")
            }
        case .not(let who):
            VStack(alignment: .leading, spacing: 12) {
                Text("Not \(who.notWho) on \(PeopleText.itemsTail(total))?")
                    .font(.headline)
                Text(who.isPet
                     ? "Its boxes here stay, unassigned and remembered as not \(who.notWho), so they are not matched to it again. Look-alike crops go along, and its tags here are removed."
                     : "Their faces and bodies here stay, unassigned and remembered as not \(who.notWho), so they are not matched to them again. Look-alike faces go along, and their tags here are removed.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                HStack {
                    Spacer()
                    Button("Cancel") { step = .list }.disabled(working)
                    Button {
                        Task { await notThem(who) }
                    } label: {
                        Label(who.notLabel, systemImage: "person.crop.circle.badge.xmark")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(working)
                    .accessibilityIdentifier("peopleSheet.notConfirm")
                }
                Spacer()
            }
            .padding(16)
        case .tag:
            VStack(alignment: .leading, spacing: 0) {
                Toggle(isOn: $related) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Not in the picture, related to them")
                        Text(related ? "Shown under Related on their page and found by search." : "Off: they are in the pictures.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(16)
                .accessibilityIdentifier("peopleSheet.related")
                PersonPicker(kind: .all, like: uids.count == 1 ? LikenessSource.item(uids[0]) : nil) { pick in
                    switch pick {
                    case .new(let name, let kind) where kind == "pet":
                        step = .tagSpecies(name)
                    default:
                        Task { await tag(choice: PeopleChoice(pick), species: nil) }
                    }
                }
            }
        case .tagSpecies(let name):
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("What is \(name)?").font(.headline)
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 100), spacing: 8)], spacing: 8) {
                        ForEach(API.Species.allCases, id: \.self) { s in
                            Button(Fmt.speciesLabel(s.rawValue)) {
                                var choice = PeopleChoice(.new(name: name, kind: "pet"))
                                choice.newKind = "pet"
                                Task { await tag(choice: choice, species: s.rawValue) }
                            }
                            .buttonStyle(.bordered)
                            .disabled(working)
                        }
                    }
                }
                .padding(16)
            }
        }
    }

    // MARK: the list

    private var listStep: some View {
        List {
            if let from {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Not \(from.display)?").font(.subheadline.weight(.semibold))
                        Text("Fix who \(uids.count == 1 ? "this item shows" : "these \(uids.count.formatted()) items show").")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 8) {
                            Button {
                                step = .move(from)
                            } label: {
                                Label("Move these to...", systemImage: "arrow.right.circle")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .accessibilityIdentifier("peopleSheet.fromMove")
                            Button {
                                step = .not(from)
                            } label: {
                                Label(from.notLabel, systemImage: "person.crop.circle.badge.xmark")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                            .accessibilityIdentifier("peopleSheet.fromNot")
                        }
                        .padding(.top, 4)
                    }
                    .padding(.vertical, 4)
                }
            }
            Section("In these items") {
                if let answer {
                    if answer.people.isEmpty {
                        Text("Nobody is named on \(PeopleText.itemsTail(total)) yet.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(answer.people, id: \.id) { p in row(p) }
                    }
                    if answer.unnamedFaces > 0 {
                        Text("\(Fmt.count(answer.unnamedFaces, "face")) here \(answer.unnamedFaces == 1 ? "is" : "are") not named yet. Name \(answer.unnamedFaces == 1 ? "it" : "them") from the photo or the People page.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else if let loadError {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(ErrorText.describe(loadError, doing: "load who is on these items"))
                            .font(.subheadline)
                            .foregroundStyle(.red)
                        Button("Try again") { Task { await load() } }
                    }
                } else {
                    HStack { Spacer(); ProgressView(); Spacer() }
                }
            }
            Section {
                Button {
                    step = .tag
                } label: {
                    Label("Tag someone in these", systemImage: "person.badge.plus")
                }
                .accessibilityIdentifier("peopleSheet.tag")
            } footer: {
                Label("Can also mark them as related, not in the pictures.", systemImage: "link")
                    .font(.caption)
            }
        }
        .accessibilityIdentifier("peopleSheet.list")
    }

    private func row(_ p: API.ItemPersonOut) -> some View {
        let who = PeopleWho(id: p.id, name: p.name, kind: p.kind.rawValue, species: p.species)
        let mine = min(p.byYou, p.items)
        let auto = p.items - mine
        var detail = ["in \(p.items.formatted()) of \(total.formatted())"]
        if mine > 0 && auto > 0 { detail.append("\(mine.formatted()) by you") }
        if p.related > 0 { detail.append("related on \(p.related.formatted())") }
        if who.isPet, let species = p.species { detail.append(Fmt.speciesLabel(species).lowercased()) }
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Avatar(name: p.name, ref: p.coverUrl, size: 36)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 5) {
                        Text(who.display)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(Fmt.isNamed(p.name) ? Color.primary : Color.secondary)
                        if mine > 0 {
                            OriginIcon(origin: .you, label: "Tagged or drawn by you on \(Fmt.count(mine, "item"))")
                                .foregroundStyle(.secondary)
                        }
                        if auto > 0 {
                            OriginIcon(origin: .auto, label: "Found automatically on \(Fmt.count(auto, "item"))")
                                .foregroundStyle(.secondary)
                        }
                    }
                    Text(detail.joined(separator: ", ")).font(.caption).foregroundStyle(.secondary)
                }
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    Button("Change to...") { step = .move(who) }
                    Button(Fmt.isNamed(p.name) ? "Rename" : "Name") {
                        nameText = p.name ?? ""
                        step = .rename(who)
                    }
                    Button(who.isPet ? "Not this pet" : "Not this person") { step = .not(who) }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(working)
            }
        }
        .padding(.vertical, 4)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(who.display)
    }

    // MARK: loading and actions

    private func load() async {
        do {
            let request = API.ItemsPeopleIn(uids: uids)
            let got: API.ItemsPeopleOut
            if vault {
                got = try await APIContext.$vault.withValue(true) { try await Cloud.api().peopleOnItems(body: request) }
            } else {
                got = try await Cloud.api().peopleOnItems(body: request)
            }
            answer = got
            loadError = nil
            if let id = fromPersonId, from == nil {
                if let p = got.people.first(where: { $0.id == id }) {
                    from = PeopleWho(id: p.id, name: p.name, kind: p.kind.rawValue, species: p.species)
                } else if let p = try? await Cloud.api().getPerson(personId: id) {
                    from = PeopleWho(id: p.id, name: p.name, kind: p.kind.rawValue, species: p.species)
                }
            }
        } catch {
            if answer == nil { loadError = error }
        }
    }

    /// A move, a "Not NAME" or a tag is done: say so, close, and let the caller end its selection.
    private func finish() {
        dismiss()
        onDone?()
    }

    private func move(_ who: PeopleWho, _ choice: PeopleChoice) async {
        guard !working else { return }
        working = true
        defer { working = false }
        let count = uids.count
        let r = await PeopleAction.run("Moved people on items", doing: "move them", items: uids, vault: vault) {
            try await Cloud.api().movePeopleOnItems(body: API.PeopleMoveIn(uids: uids, fromPersonId: who.id,
                                                                          personId: choice.personId,
                                                                          newPersonName: choice.newName))
        } message: { r in
            r.updated > 0
                ? "\(Fmt.count(r.updated, "item")) moved to \(r.person.map { PeopleText.name($0) } ?? "someone else").\(PeopleText.bulkTail(r, pet: who.isPet))"
                : "\(who.display) is not on \(PeopleText.itemsTail(count)): nothing moved."
        }
        if r != nil { finish() }
    }

    private func notThem(_ who: PeopleWho) async {
        guard !working else { return }
        working = true
        defer { working = false }
        let count = uids.count
        let r = await PeopleAction.run("Marked people as not on items", doing: "save that", items: uids, vault: vault) {
            try await Cloud.api().removePeopleFromItems(body: API.PeopleRemoveIn(uids: uids, personId: who.id))
        } message: { r in
            r.updated > 0
                ? "\(Fmt.count(r.updated, "item")) marked as not \(who.notWho).\(PeopleText.bulkTail(r, pet: who.isPet))"
                : "\(who.display) is not on \(PeopleText.itemsTail(count)): nothing changed."
        }
        if r != nil { finish() }
    }

    private func tag(choice: PeopleChoice, species: String?) async {
        guard !working else { return }
        working = true
        defer { working = false }
        let count = uids.count
        let relation: API.Relation = related ? .related : .in
        let pet = choice.newKind == "pet"
        let request = API.PeopleTagIn(personId: choice.personId, newPersonName: choice.newName,
                                      kind: pet ? .pet : nil, species: pet ? species : nil,
                                      relation: relation, uids: uids)
        let r = await PeopleAction.run("Tagged someone on items", doing: "tag them", items: uids, vault: vault) {
            try await Cloud.api().tagPeopleOnItems(body: request)
        } message: { r in
            if r.updated > 0 {
                return "\(Fmt.count(r.updated, "item")) tagged \(r.person.map { PeopleText.name($0) } ?? "")\(relation == .related ? " as related" : "")."
            }
            let who = r.person.map { PeopleText.name($0) } ?? "They"
            let verb = r.person?.kind == .pet ? "is" : "are"
            return "\(who) already \(verb) on \(PeopleText.itemsTail(count))."
        }
        if r != nil { finish() }
    }

    private func rename(_ who: PeopleWho) async {
        guard !working else { return }
        let text = nameText.trimmingCharacters(in: .whitespacesAndNewlines)
        let next: String? = text.isEmpty ? nil : String(text.prefix(120))
        if next == who.name {
            step = .list
            return
        }
        working = true
        defer { working = false }
        var update = API.PersonUpdate(name: next)
        if next == nil { update.nullKeys = [.name] }
        let r = await PeopleAction.run(next == nil ? "Removed a name" : "Renamed someone", doing: "save the name", vault: vault) {
            try await Cloud.api().updatePerson(personId: who.id, body: update)
        } message: { p in
            p.id != who.id ? "Merged into \(PeopleText.name(p))." : Fmt.isNamed(p.name) ? "Renamed to \(p.name ?? "")." : "Name removed."
        }
        guard let p = r else { return }
        if p.id != who.id && who.id == fromPersonId {
            // the page's own person went away into the one they joined: follow them
            finish()
            onMerged?(p.id)
            return
        }
        step = .list
        await load()
    }
}

/// Someone a step of the selection sheet is about.
struct PeopleWho: Hashable {
    let id: Int
    let name: String?
    let kind: String
    let species: String?

    var isPet: Bool { kind == "pet" }
    var display: String { PeopleText.display(name, kind: kind) }
    var notLabel: String { PeopleText.notLabel(name, kind: kind) }
    var notWho: String { PeopleText.notWho(name, kind: kind) }
}

/// The steps of the selection sheet.
enum PeopleSelectionStep: Hashable {
    case list
    case move(PeopleWho)
    case rename(PeopleWho)
    case not(PeopleWho)
    case tag
    case tagSpecies(String)
}
