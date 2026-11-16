import SwiftUI

/// The info panel's people section (the website's InfoPanel people part, PersonEdit, GuessChip): who is in the item
/// with what each one's menu changes, who it is related to, the guesses and why, the faces and animals nobody
/// named, tagging someone, and adding a box. The viewer area places it in its info panel.
struct ViewerBoxesInfoSection: View {
    let model: ViewerBoxesModel
    let context: ViewerPageContext

    var body: some View {
        if let item = context.item {
            BoxesInfoContent(model: model, context: context, item: item)
        }
    }
}

/// A person in the section's sheets.
struct BoxesPersonChoice: Identifiable {
    let person: API.PersonRef
    var id: Int { person.id }
}

struct BoxesInfoContent: View {
    let model: ViewerBoxesModel
    let context: ViewerPageContext
    let item: API.ItemDetail

    @Environment(\.dismiss) private var dismissPanel
    @State private var editing: BoxesEditorTarget?
    @State private var editingPicks = false
    @State private var changing: BoxesPersonChoice?
    @State private var speciesFor: BoxesPersonChoice?
    @State private var renaming: API.PersonRef?
    @State private var renameText = ""
    @State private var tagging = false
    @State private var moreBoxes = false
    @State private var busy = false

    private var uid: String { context.uid }
    private var vault: Bool { context.vault }
    private var kind: MediaKind { context.detail.mediaKind }
    private var isVideo: Bool { kind == .video }
    private var isDoc: Bool { kind == .pdf || kind == .document }

    private var inPeople: [API.PersonRef] { item.people.filter { $0.relation.rawValue != "related" } }
    private var relatedPeople: [API.PersonRef] { item.people.filter { $0.relation.rawValue == "related" } }
    private var guesses: [BoxesTarget] { BoxesLookup.guessTargets(item) }
    private var origins: [Int: BoxOrigin] { BoxesLookup.itemOrigins(item) }
    private var unmatched: Int { item.faces.filter { $0.person == nil && BoxesTarget.face($0).guess == nil }.count }
    private var unnamedAnimals: Int { item.animals.filter { $0.person == nil }.count }

    var body: some View {
        if !isDoc || !item.people.isEmpty || !item.faces.isEmpty {
            peopleSection
            if !relatedPeople.isEmpty {
                Section {
                    ForEach(relatedPeople, id: \.id) { personRow($0) }
                } header: {
                    Label("Related to", systemImage: "link")
                }
            }
            if !guesses.isEmpty {
                Section {
                    ForEach(guesses) { guessRow($0) }
                } header: {
                    Text("Guesses and why")
                }
            }
        }
    }

    // MARK: people

    private var sectionTitle: String {
        if isVideo { return "In this video" }
        return item.people.contains(where: { $0.kind == .pet }) || !item.animals.isEmpty ? "People and pets" : "People"
    }

    private var emptyText: String {
        if item.analysisStatus.rawValue == "pending" { return "Still looking for people." }
        if !relatedPeople.isEmpty { return isVideo ? "No one is in this video." : "No one is in this picture." }
        return "No one is tagged here yet."
    }

    private var unmatchedText: String {
        let count = unmatched == 1 ? "1 face is" : "\(unmatched) faces are"
        return "\(count) not matched to anyone. Tap a face \(isVideo ? "chip" : "box") to tag it."
    }

    private var unnamedAnimalsText: String {
        let count = unnamedAnimals == 1 ? "1 animal is" : "\(unnamedAnimals) animals are"
        return "\(count) not matched to a pet. Tap its \(isVideo ? "chip" : "box") to name it."
    }

    private var renamePlaceholder: String { renaming?.kind == .pet ? "Its name" : "Their name" }

    private var renameMessage: String {
        Fmt.isNamed(renaming?.name)
            ? "The new name shows everywhere, not only here. A name someone else already has merges them."
            : "The name shows everywhere they appear. A name someone else already has merges them."
    }

    private var peopleSection: some View {
        Section {
            if inPeople.isEmpty {
                Text(emptyText)
                    .foregroundStyle(.secondary)
            }
            ForEach(inPeople, id: \.id) { personRow($0) }
            if unmatched > 0 {
                Text(unmatchedText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            if unnamedAnimals > 0 {
                Text(unnamedAnimalsText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            tagButton
            addBoxRows
        } header: {
            Text(sectionTitle)
        }
    }

    /// "Tag someone", which also holds the section's sheets (one place that is always there).
    private var tagButton: some View {
        Button {
            tagging = true
        } label: {
            Label("Tag someone", systemImage: "person.badge.plus")
        }
        .disabled(busy)
        .accessibilityIdentifier("boxes.tag")
        .sheet(item: $editing) { target in
            BoxesEditorSheet(model: model, target: target, startPicking: editingPicks) {
                editing = nil
                editingPicks = false
            }
            .presentationDetents([.medium, .large])
        }
        .sheet(item: $changing) { choice in
            let p = choice.person
            let pet = p.kind == .pet
            let species = speciesOf(p)
            PersonPickerSheet(title: "Change \(BoxesWords.displayName(p.name, pet: pet)) to...", kind: pet ? .pet : .person,
                              like: .person(p.id, on: [uid]), excluding: [p.id], species: species,
                              speciesOnly: pet && species != nil) { pick in
                act { await BoxesActions.change(p, to: pick, uid: uid, vault: vault) }
            }
        }
        .sheet(item: $speciesFor) { choice in
            let p = choice.person
            BoxesSpeciesSheet(title: "What is \(BoxesWords.displayName(p.name, pet: true))?", current: speciesOf(p),
                              note: "The species changes everywhere, on all its boxes, not only in this \(isVideo ? "video" : "picture").") { species in
                act { await BoxesActions.personSpecies(p, species: species, uid: uid, vault: vault) }
            }
            .presentationDetents([.medium, .large])
        }
        .sheet(isPresented: $tagging) {
            BoxesTagSheet(item: item, uid: uid, vault: vault, isVideo: isVideo)
        }
        .alert(renameTitle, isPresented: renamingShown) {
            TextField(renamePlaceholder, text: $renameText)
                .accessibilityIdentifier("boxes.rename.field")
            Button("Save") {
                guard let p = renaming else { return }
                let text = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
                let next: String? = text.isEmpty ? nil : String(text.prefix(120))
                renaming = nil
                if next == p.name { return }
                act { await BoxesActions.rename(p, to: next, uid: uid, vault: vault) }
            }
            Button("Cancel", role: .cancel) { renaming = nil }
        } message: {
            Text(renameMessage)
        }
    }

    private var renameTitle: String {
        guard let p = renaming else { return "Rename" }
        if Fmt.isNamed(p.name) { return "Rename \(BoxesWords.displayName(p.name, pet: p.kind == .pet))" }
        return p.kind == .pet ? "Name this pet" : "Name this person"
    }

    private var renamingShown: Binding<Bool> {
        Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })
    }

    /// Adding a box by drawing it: on photos, where the core's stage draws.
    @ViewBuilder
    private var addBoxRows: some View {
        if kind == .image {
            Button { model.startDrawing(.face, in: context) } label: {
                Label("Add a face box", systemImage: "faceid")
            }
            .accessibilityIdentifier("boxes.info.addFace")
            Button { model.startDrawing(.person, in: context) } label: {
                Label("Add a person box", systemImage: "person.crop.rectangle")
            }
            .accessibilityIdentifier("boxes.info.addPerson")
        }
        if kind == .image || isVideo {
            if !item.animals.isEmpty && !moreBoxes {
                Button { moreBoxes = true } label: {
                    Label("More", systemImage: "ellipsis")
                }
            } else {
                if kind == .image {
                    Button { model.startDrawing(.pet, in: context) } label: {
                        Label("Add a pet box", systemImage: "pawprint")
                    }
                    .accessibilityIdentifier("boxes.info.addPet")
                }
                Button {
                    act { await BoxesActions.findPets(uid: uid, vault: vault); return false }
                } label: {
                    Label("Find pets again", systemImage: "arrow.clockwise")
                }
                .accessibilityIdentifier("boxes.info.findPets")
            }
            if item.animals.isEmpty {
                Text(isVideo ? "No pet box in this video. Let the detector look again more closely."
                     : "No pet box here. Draw one over the pet, or let the detector look again more closely.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func speciesOf(_ p: API.PersonRef) -> String? {
        item.animals.first(where: { $0.person?.id == p.id })?.species
    }

    /// Runs one change at a time.
    private func act(_ action: @escaping () async -> Bool) {
        guard !busy else { return }
        busy = true
        Task {
            _ = await action()
            busy = false
        }
    }

    // MARK: a person's row and menu

    private func personRow(_ p: API.PersonRef) -> some View {
        let pet = p.kind == .pet
        let name = Fmt.personName(p.name, kind: pet ? "pet" : "person", species: speciesOf(p))
        let faceIds = item.faces.filter { $0.person?.id == p.id }.map(\.id)
        // Person boxes without a face of their own (drawn ones, a back or a far shot); a face's body goes with the face.
        let bodyIds = item.bodies.filter { $0.person?.id == p.id && $0.faceId == nil }.map(\.id)
        let petIds = item.animals.filter { $0.person?.id == p.id }.map(\.id)
        let tagged = item.tags.contains(where: { $0.id == p.id }) || p.relation.rawValue == "related"
        let boxed = !faceIds.isEmpty || !bodyIds.isEmpty || !petIds.isEmpty
            || item.bodies.contains(where: { $0.person?.id == p.id })
        let tagOnly = tagged && !boxed
        let still = BoxesLookup.stillTargets(item).first(where: { $0.person?.id == p.id })
        let renameLabel: String = Fmt.isNamed(p.name) ? "Rename" : "Name"
        return HStack(spacing: 10) {
            RemoteImage(ref: BoxesLookup.thumb(of: p, in: item), maxPixel: 120, vault: vault)
                .frame(width: 36, height: 36)
                .clipShape(Circle())
                .accessibilityHidden(true)
            HStack(spacing: 4) {
                Text(name)
                    .lineLimit(1)
                if let origin = origins[p.id] {
                    OriginIcon(origin: origin)
                        .foregroundStyle(.secondary)
                }
            }
            .accessibilityElement(children: .combine)
            Spacer()
            Menu {
                Button("Change to...") { changing = BoxesPersonChoice(person: p) }
                Button(renameLabel) {
                    renameText = p.name ?? ""
                    renaming = p
                }
                if pet && !petIds.isEmpty {
                    Button("Species...") { speciesFor = BoxesPersonChoice(person: p) }
                }
                if tagged {
                    if p.relation.rawValue == "related" {
                        Button("Make this 'in'") {
                            act { await BoxesActions.setRelation(p, related: false, uid: uid, vault: vault) }
                        }
                    } else {
                        Button("Make this 'related'") {
                            act { await BoxesActions.setRelation(p, related: true, uid: uid, vault: vault) }
                        }
                    }
                }
                if let still, kind == .image {
                    Button {
                        let key = still.id
                        dismissPanel()
                        model.show(key)
                    } label: {
                        Label("Show in the photo", systemImage: "viewfinder")
                    }
                }
                if tagOnly {
                    Button("Remove from this item") {
                        act { await BoxesActions.removeTag(p, uid: uid, vault: vault) }
                    }
                } else {
                    Button {
                        act { await BoxesActions.notThem(p, uid: uid, vault: vault) }
                    } label: {
                        Text("Not \(name)")
                        Text(pet ? "Its boxes here stay, unnamed and remembered as not it"
                             : "Their faces here stay, unnamed and remembered as not them")
                    }
                }
                if !faceIds.isEmpty {
                    Button(role: .destructive) {
                        // A face inside a person box of theirs leaves them on the item through that box.
                        let stays = item.bodies.contains { b in
                            b.person?.id == p.id && b.faceId.map { faceIds.contains($0) } == true
                        }
                        act { await BoxesActions.removeBoxes(of: p, kind: .face, ids: faceIds, stays: stays, uid: uid, vault: vault) }
                    } label: {
                        Text("Not a face")
                        Text(faceIds.count == 1 ? "Removes this face box, and detection learns from it"
                             : "Removes their \(faceIds.count) face boxes here, and detection learns from it")
                    }
                }
                if !bodyIds.isEmpty {
                    Button(role: .destructive) {
                        act { await BoxesActions.removeBoxes(of: p, kind: .body, ids: bodyIds, stays: false, uid: uid, vault: vault) }
                    } label: {
                        Text("Not a person")
                        Text(bodyIds.count == 1 ? "Removes their person box here" : "Removes their \(bodyIds.count) person boxes here")
                    }
                }
                if !petIds.isEmpty {
                    Button(role: .destructive) {
                        act { await BoxesActions.removeBoxes(of: p, kind: .animal, ids: petIds, stays: false, uid: uid, vault: vault) }
                    } label: {
                        Text("Not a pet")
                        Text(petIds.count == 1 ? "Removes its box here" : "Removes its \(petIds.count) boxes here")
                    }
                }
            } label: {
                Image(systemName: "ellipsis.circle")
                    .font(.title3)
                    .frame(width: 44, height: 44)
            }
            .disabled(busy)
            .accessibilityLabel("Change \(name)")
            .accessibilityIdentifier("boxes.person.menu")
        }
    }

    // MARK: a guess's row (the website's GuessChip)

    private func guessRow(_ t: BoxesTarget) -> some View {
        let name = t.guessName ?? ""
        let noun = t.kind == .face ? "face" : "person"
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Button {
                    editingPicks = false
                    editing = BoxesEditorTarget(uid: uid, kind: t.kind, boxId: t.boxId, vault: vault)
                } label: {
                    RemoteImage(ref: t.cropRef, maxPixel: 120, vault: vault)
                        .frame(width: 40, height: 40)
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Edit this \(noun)")
                OriginIcon(origin: .guess)
                    .foregroundStyle(.secondary)
                Text("\(name)?")
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Spacer()
                Button {
                    act { await BoxesActions.answer([t], confirm: true, guessName: name, uid: uid, vault: vault) }
                } label: {
                    Image(systemName: "checkmark")
                }
                .buttonStyle(.bordered)
                .accessibilityLabel("Yes, this is \(name)")
                .accessibilityIdentifier("boxes.guessRow.yes")
                Button {
                    act { await BoxesActions.answer([t], confirm: false, guessName: name, uid: uid, vault: vault) }
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.bordered)
                .accessibilityLabel("No, this is not \(name)")
                .accessibilityIdentifier("boxes.guessRow.no")
                Button {
                    editingPicks = true
                    editing = BoxesEditorTarget(uid: uid, kind: t.kind, boxId: t.boxId, vault: vault)
                } label: {
                    Image(systemName: "person.crop.circle.badge.questionmark")
                }
                .buttonStyle(.bordered)
                .accessibilityLabel("Someone else, not \(name)")
                .accessibilityIdentifier("boxes.guessRow.other")
            }
            .disabled(busy)
            if let guess = t.guess {
                ReviewGuessEvidence(guess: guess)
                    .padding(.leading, 48)
            }
        }
    }
}

/// "Tag someone" (the website's tagging form): a person or pet in the item, or, with the switch, one it is about
/// without them in it. A new pet asks for its species first.
struct BoxesTagSheet: View {
    let item: API.ItemDetail
    let uid: String
    let vault: Bool
    let isVideo: Bool

    @Environment(\.dismiss) private var dismiss
    @State private var related = false
    @State private var newPet: String?

    var body: some View {
        NavigationStack {
            Group {
                if let newPet {
                    BoxesSpeciesList(current: nil, note: nil) { species in
                        add(.new(name: newPet, kind: "pet"), species: species.rawValue)
                    }
                    .navigationTitle("What is \(newPet)?")
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) { Button("Back") { self.newPet = nil } }
                    }
                } else {
                    PersonPicker(kind: .all, like: .item(uid), excluding: Set(item.people.map(\.id))) { pick in
                        if case .new(let name, let kind) = pick, kind == "pet" {
                            newPet = name
                            return
                        }
                        add(pick, species: nil)
                    }
                    .safeAreaInset(edge: .top) { relatedSwitch }
                    .navigationTitle("Tag a person or pet")
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var relatedSwitch: some View {
        Toggle(isOn: $related) {
            VStack(alignment: .leading, spacing: 2) {
                Text(switchTitle)
                Text(switchNote)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityIdentifier("boxes.tag.related")
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private var switchTitle: String { "Not in the \(isVideo ? "video" : "picture"), related to them" }

    private var switchNote: String {
        related ? "Shown under Related on their page and found by search." : "Off: they are in the \(isVideo ? "video" : "picture")."
    }

    private func add(_ pick: PersonPick, species: String?) {
        let related = related
        dismiss()
        Task { _ = await BoxesActions.tag(pick, related: related, species: species, uid: uid, vault: vault) }
    }
}
