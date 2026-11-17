import SwiftUI
import Observation

/// The pages an editor pushes: the person (or pet) picker and the species list.
enum BoxesEditorPage: Hashable {
    case picker
    case species
}

/// What an editor's pages share.
@MainActor
@Observable
final class BoxesEditorState {
    /// A video's box: change this track only, not every track of the person in the video.
    var onlyTrack = false
    var busy = false
    /// A pet of another species was picked: moving the box there needs that pet to become this species first.
    var mismatch: PersonOption?
}

/// A box's editor in a sheet of its own: from a tap on the stage, or from the info panel's people section (a sheet
/// over the panel). `startPicking` opens it at the picker ("Someone else" on a guess).
struct BoxesEditorSheet: View {
    let model: ViewerBoxesModel
    let target: BoxesEditorTarget
    var startPicking = false
    let onClose: () -> Void

    @State private var path: [BoxesEditorPage] = []
    @State private var state = BoxesEditorState()

    var body: some View {
        NavigationStack(path: $path) {
            BoxesEditorView(model: model, target: target, page: nil, state: state, path: $path, onClose: onClose)
                .navigationDestination(for: BoxesEditorPage.self) { page in
                    BoxesEditorView(model: model, target: target, page: page, state: state, path: $path, onClose: onClose)
                }
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Close") { onClose() }
                            .accessibilityIdentifier("boxes.editor.close")
                    }
                }
        }
        .onAppear {
            if startPicking && path.isEmpty { path = [.picker] }
        }
    }
}

/// One page of a box's editor: the editor itself (`page` nil), its picker or its species list. Faces and person
/// boxes get the website's FaceEditor, pets its AnimalEditor. The box is read from the item as it is now, so the
/// editor shows each save's answer.
struct BoxesEditorView: View {
    let model: ViewerBoxesModel
    let target: BoxesEditorTarget
    let page: BoxesEditorPage?
    @Bindable var state: BoxesEditorState
    @Binding var path: [BoxesEditorPage]
    let onClose: () -> Void

    private var context: ViewerPageContext? { model.stage.contexts[target.uid] }
    private var item: API.ItemDetail? { context?.item }
    private var box: BoxesTarget? {
        guard let item else { return nil }
        return BoxesLookup.target(in: item, kind: target.kind, id: target.boxId)
    }

    var body: some View {
        Group {
            if let item, let box {
                switch page {
                case nil:
                    if box.kind == .animal { petEditor(item, box) } else { personEditor(item, box) }
                case .picker?:
                    picker(item, box)
                case .species?:
                    BoxesSpeciesList(current: box.species, note: nil) { species in
                        path.removeAll()
                        run { await BoxesActions.setSpecies(scope(item, box), species: species, uid: target.uid,
                                                            vault: target.vault) }
                    }
                    .navigationTitle("This is a")
                }
            } else if item != nil && !target.fresh {
                ContentUnavailableView("This box is gone", systemImage: "rectangle.dashed",
                                       description: Text("It was removed, or the item changed."))
            } else if context != nil && item == nil && !target.fresh {
                ContentUnavailableView("Could not read this item", systemImage: "exclamationmark.triangle",
                                       description: Text("Close it and open it again."))
            } else {
                ProgressView()
            }
        }
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: shared

    /// The boxes a change applies to: every track of the person in a video, unless "Only this track".
    private func scope(_ item: API.ItemDetail, _ box: BoxesTarget) -> [BoxesTarget] {
        state.onlyTrack ? [box] : BoxesLookup.tracks(of: box, in: item)
    }

    private func allTracks(_ item: API.ItemDetail, _ box: BoxesTarget) -> [BoxesTarget] {
        BoxesLookup.tracks(of: box, in: item)
    }

    /// Runs an action once at a time; a change that went through closes the editor.
    private func run(_ action: @escaping () async -> Bool) {
        guard !state.busy else { return }
        state.busy = true
        Task {
            let ok = await action()
            state.busy = false
            if ok { onClose() }
        }
    }

    /// Moving or resizing needs the still picture the core's stage draws.
    private func canAdjust(_ box: BoxesTarget) -> Bool {
        box.frameTimeS == nil && context?.detail.mediaKind == .image
    }

    /// For a whole video's tracks, the owner's origin when any track is theirs (the website's mergeOrigins).
    private func origin(_ boxes: [BoxesTarget]) -> BoxOrigin? {
        if boxes.count <= 1 { return boxes.first?.origin }
        let all = boxes.compactMap(\.origin)
        return [BoxOrigin.you, .drawn, .confirmed, .auto, .guess].first { all.contains($0) }
    }

    private func header(caption: String, name: String, crop: String, origin: BoxOrigin?) -> some View {
        HStack(spacing: 12) {
            RemoteImage(ref: crop, maxPixel: 168, vault: target.vault)
                .frame(width: 56, height: 56)
                .clipShape(Circle())
                .overlay(Circle().stroke(Color(uiColor: .separator), lineWidth: 0.5))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(caption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(spacing: 4) {
                    Text(name)
                        .font(.headline)
                        .lineLimit(2)
                        .accessibilityIdentifier("boxes.editor.name")
                    if let origin {
                        OriginIcon(origin: origin)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private func tracksToggle(_ item: API.ItemDetail, _ box: BoxesTarget, pet: Bool) -> some View {
        let tracks = allTracks(item, box)
        if tracks.count > 1 {
            Toggle("Only this track", isOn: $state.onlyTrack)
                .accessibilityIdentifier("boxes.onlyTrack")
            let who = pet ? "pet" : "person"
            let note: String = state.onlyTrack ? "Applies to this track only."
                : "Applies to all \(tracks.count) tracks of this \(who) in the video."
            Text(note)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: faces and person boxes (the website's FaceEditor)

    private func personEditor(_ item: API.ItemDetail, _ box: BoxesTarget) -> some View {
        let isFace = box.kind == .face
        let guess = box.guess
        let boxes = scope(item, box)
        let traitBody: API.BodyOut? = {
            if case .body(let b) = box { return b }
            return BoxesLookup.body(of: box.boxId, in: item)
        }()
        let traits = traitBody?.traits ?? []
        let caption = box.person != nil ? "Currently tagged as" : guess != nil ? "Our guess" : "Not tagged yet"
        var name = "Who is this?"
        var pickLabel = "Who is this?"
        if let person = box.personName {
            name = person
            pickLabel = "Move to someone else"
        } else if box.person != nil {
            name = Fmt.personName(nil)
            pickLabel = "Move to someone else"
        } else if let guessed = box.guessName {
            name = guessed + "?"
            pickLabel = "Not \(guessed)? Pick who it is"
        }
        return List {
            Section {
                header(caption: caption, name: name, crop: box.cropRef, origin: origin(boxes))
            }
            if let guess {
                guessSection(guess, boxes: boxes)
            }
            if !traits.isEmpty {
                traitsSection(traits, owner: box.personName)
            }
            Section {
                NavigationLink(value: BoxesEditorPage.picker) {
                    Label(pickLabel, systemImage: "person.crop.circle.badge.plus")
                }
                .disabled(state.busy)
                .accessibilityIdentifier("viewer.box.who")
                tracksToggle(item, box, pet: false)
            }
            Section {
                if let current = box.person {
                    Button {
                        run { await BoxesActions.assign(boxes, pick: nil, current: current, uid: target.uid, vault: target.vault) }
                    } label: {
                        Label("Not this person", systemImage: "person.crop.circle.badge.xmark")
                    }
                    .accessibilityIdentifier("boxes.notPerson")
                }
                if canAdjust(box) {
                    Button {
                        model.startAdjusting(box.stageBox, uid: target.uid)
                    } label: {
                        Label("Move or resize", systemImage: "arrow.up.and.down.and.arrow.left.and.right")
                    }
                    .accessibilityIdentifier("boxes.adjust")
                }
                Button(role: .destructive) {
                    run { await BoxesActions.remove(boxes, noun: isFace ? "face" : "person", uid: target.uid, vault: target.vault) }
                } label: {
                    Label(isFace ? "Not a face" : "Not a person", systemImage: "nosign")
                }
                .accessibilityIdentifier("boxes.remove")
            }
            .disabled(state.busy)
        }
        .navigationTitle(isFace ? "Edit face" : "Edit person")
    }

    private func guessSection(_ guess: API.GuessOut, boxes: [BoxesTarget]) -> some View {
        let name = Fmt.personName(guess.person.name)
        return Section {
            (Text("Looks like ") + Text(name).bold() + Text("."))
                .accessibilityIdentifier("boxes.guess.text")
            ReviewGuessEvidence(guess: guess)
            HStack(spacing: 12) {
                Button {
                    run { await BoxesActions.answer(boxes, confirm: true, guessName: name, uid: target.uid, vault: target.vault) }
                } label: {
                    Label("Yes, it is", systemImage: "checkmark")
                }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier("boxes.guess.yes")
                Button {
                    run { await BoxesActions.answer(boxes, confirm: false, guessName: name, uid: target.uid, vault: target.vault) }
                } label: {
                    Label("No", systemImage: "xmark")
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("boxes.guess.no")
            }
            .disabled(state.busy)
        }
    }

    private func traitsSection(_ traits: [API.TraitOut], owner: String?) -> some View {
        Section {
            ForEach(traits, id: \.id) { trait in
                HStack(spacing: 10) {
                    RemoteImage(ref: trait.cropUrl, maxPixel: 120, vault: target.vault)
                        .frame(width: 36, height: 36)
                        .clipShape(Circle())
                        .opacity(trait.hidden ? 0.5 : 1)
                        .accessibilityHidden(true)
                    if trait.kind == .nails, let color = BoxesColor.hex(trait.hex) {
                        Circle().fill(color).frame(width: 12, height: 12).accessibilityHidden(true)
                    }
                    Text(trait.label)
                        .strikethrough(trait.hidden)
                        .foregroundStyle(trait.hidden ? Color.secondary : Color.primary)
                    Spacer()
                    Menu {
                        if trait.hidden {
                            Button {
                                run { await BoxesActions.trait(trait, verdict: nil, owner: owner, uid: target.uid, vault: target.vault) }
                            } label: {
                                Label("Show again", systemImage: "eye")
                            }
                        } else {
                            Button {
                                run { await BoxesActions.trait(trait, verdict: .notTrait, owner: owner, uid: target.uid, vault: target.vault) }
                            } label: {
                                Label(BoxesWords.notTraitLabel(trait.kind), systemImage: "eye.slash")
                            }
                            Button {
                                run { await BoxesActions.trait(trait, verdict: .notTheirs, owner: owner, uid: target.uid, vault: target.vault) }
                            } label: {
                                Label("Not this person's", systemImage: "person.crop.circle.badge.xmark")
                            }
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                            .font(.title3)
                    }
                    .accessibilityLabel(trait.label + (trait.hidden ? ", hidden" : "") + ": options")
                }
            }
        } header: {
            Text("Seen on this person")
        } footer: {
            if traits.contains(where: \.hidden) {
                Text("Show again is for this photo only. Hidden look-alikes elsewhere stay hidden.")
            }
        }
    }

    // MARK: pets (the website's AnimalEditor)

    private func petEditor(_ item: API.ItemDetail, _ box: BoxesTarget) -> some View {
        let boxes = scope(item, box)
        let species = Fmt.speciesLabel(box.species)
        var userSpecies = false
        if case .animal(let a) = box { userSpecies = a.speciesSource == .user }
        let caption = (box.person != nil ? "\(species), currently tagged as" : "\(species), not named yet")
            + (userSpecies ? " (your choice)" : "")
        let name = box.person.map { Fmt.personName($0.name, kind: "pet", species: box.species) } ?? "Which pet is this?"
        let aSpecies = BoxesWords.aSpecies(box.species)
        return List {
            Section {
                header(caption: caption, name: name, crop: box.cropRef, origin: origin(boxes))
            }
            Section {
                NavigationLink(value: BoxesEditorPage.picker) {
                    Label(box.person != nil ? "Move to another pet" : "Which pet is this?", systemImage: "pawprint")
                }
                .disabled(state.busy)
                .accessibilityIdentifier("viewer.box.who")
                tracksToggle(item, box, pet: true)
            } footer: {
                Text("Pets of the same species come first. A pet of another species can take this box once it becomes \(aSpecies) too.")
            }
            if let other = state.mismatch {
                Section {
                    Text("\(other.name) is \(BoxesWords.aSpecies(other.species)); this box is \(aSpecies).")
                        .foregroundStyle(.secondary)
                    Button {
                        let pick = PersonPick.existing(other)
                        run {
                            await BoxesActions.assignPet(boxes, pick: pick, fixSpecies: true, current: box.person,
                                                         species: box.species, uid: target.uid, vault: target.vault)
                        }
                    } label: {
                        Label("Move and make \(other.name) \(aSpecies)", systemImage: "shuffle")
                    }
                    .accessibilityIdentifier("boxes.pet.fix")
                }
            }
            Section {
                NavigationLink(value: BoxesEditorPage.species) {
                    Label("Not \(aSpecies)", systemImage: "pawprint")
                }
                .accessibilityIdentifier("boxes.pet.species")
                if let current = box.person {
                    Button {
                        run {
                            await BoxesActions.assignPet(boxes, pick: nil, fixSpecies: false, current: current,
                                                         species: box.species, uid: target.uid, vault: target.vault)
                        }
                    } label: {
                        Label("Not this pet", systemImage: "xmark.circle")
                    }
                    .accessibilityIdentifier("boxes.pet.notThis")
                }
                if canAdjust(box) {
                    Button {
                        model.startAdjusting(box.stageBox, uid: target.uid)
                    } label: {
                        Label("Move or resize", systemImage: "arrow.up.and.down.and.arrow.left.and.right")
                    }
                    .accessibilityIdentifier("boxes.adjust")
                }
                Button(role: .destructive) {
                    run { await BoxesActions.remove(boxes, noun: "pet", uid: target.uid, vault: target.vault) }
                } label: {
                    Label("Remove box", systemImage: "nosign")
                }
                .accessibilityIdentifier("boxes.remove")
            }
            .disabled(state.busy)
        }
        .navigationTitle("Edit pet")
    }

    // MARK: the picker

    private func picker(_ item: API.ItemDetail, _ box: BoxesTarget) -> some View {
        let boxes = scope(item, box)
        let pet = box.kind == .animal
        var like = LikenessSource()
        like.faceIds = boxes.filter { $0.kind == .face }.map(\.boxId)
        like.bodyIds = boxes.filter { $0.kind == .body }.map(\.boxId)
        like.animalIds = boxes.filter { $0.kind == .animal }.map(\.boxId)
        var excluding: Set<Int> = []
        if let id = box.person?.id { excluding.insert(id) } else if let id = box.guess?.person.id { excluding.insert(id) }
        // Any open guess among the boxes: a pick corrects it (learned as wrong) instead of overwriting it.
        let anyGuess = boxes.contains { $0.guess != nil }
        return PersonPicker(kind: pet ? .pet : .person, like: like, excluding: excluding, species: pet ? box.species : nil) { pick in
            path.removeAll()
            if pet {
                if case .existing(let option) = pick, let s = option.species, s != box.species {
                    state.mismatch = option
                    return
                }
                state.mismatch = nil
                run {
                    await BoxesActions.assignPet(boxes, pick: pick, fixSpecies: false, current: box.person,
                                                 species: box.species, uid: target.uid, vault: target.vault)
                }
            } else if anyGuess {
                run { await BoxesActions.correct(boxes, pick: pick, uid: target.uid, vault: target.vault) }
            } else {
                run { await BoxesActions.assign(boxes, pick: pick, current: box.person, uid: target.uid, vault: target.vault) }
            }
        }
        .navigationTitle(pet ? "Which pet is this?" : "Who is this?")
    }
}

/// The species a pet can be, to pick one (the website's species buttons).
struct BoxesSpeciesList: View {
    var current: String?
    var note: String?
    let onPick: (API.Species) -> Void

    var body: some View {
        List {
            if let note {
                Section {
                    Text(note)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            Section {
                ForEach(API.Species.allCases, id: \.rawValue) { species in
                    Button {
                        onPick(species)
                    } label: {
                        HStack {
                            Text(Fmt.speciesLabel(species.rawValue))
                                .foregroundStyle(.primary)
                            Spacer()
                            if species.rawValue == current {
                                Image(systemName: "checkmark")
                                    .foregroundStyle(.tint)
                            }
                        }
                    }
                    .disabled(species.rawValue == current)
                    .accessibilityIdentifier("species.\(species.rawValue)")
                }
            }
        }
    }
}

/// The species list in a sheet of its own.
struct BoxesSpeciesSheet: View {
    let title: String
    var current: String?
    var note: String?
    let onPick: (API.Species) -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            BoxesSpeciesList(current: current, note: note) { species in
                dismiss()
                onPick(species)
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            }
        }
    }
}

/// Colours the server sends as "#rrggbb" (a nail polish swatch).
enum BoxesColor {
    static func hex(_ text: String?) -> Color? {
        guard var s = text?.trimmingCharacters(in: .whitespaces), !s.isEmpty else { return nil }
        if s.hasPrefix("#") { s.removeFirst() }
        guard s.count == 6, let v = UInt32(s, radix: 16) else { return nil }
        return Color(red: Double((v >> 16) & 0xFF) / 255, green: Double((v >> 8) & 0xFF) / 255,
                     blue: Double(v & 0xFF) / 255)
    }
}
