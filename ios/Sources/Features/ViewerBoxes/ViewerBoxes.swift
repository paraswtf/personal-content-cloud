import SwiftUI
import Observation

// The boxes area's part of the viewer (the website's FaceEditor, PersonEdit, AnimalEditor, BoxAdjuster,
// PersonBoxCard, GuessChip and the box parts of PhotoStage). The viewer area wires these into ViewerHooks
// (Features/Viewer/ViewerScreen.swift), so their names and signatures stay: ViewerBoxesModel (with tap(_:in:) and
// selectedBoxId), ViewerBoxesToolbarItems(model:context:), ViewerBoxesInfoSection(model:context:) and
// .viewerBoxEditors(_:).

/// The box whose editor is open: which item, which box, and what the stage drew for it.
struct BoxesEditorTarget: Identifiable, Hashable {
    let uid: String
    let kind: StageBox.Kind
    let boxId: Int
    let vault: Bool
    /// A box just added: the editor waits for the item to load again instead of saying the box is gone.
    var fresh = false

    var id: String { "\(uid)#\(kind.rawValue):\(boxId)" }
    /// The box's key on the stage ("face:3").
    var boxKey: String { "\(kind.rawValue):\(boxId)" }
}

/// A box the owner drew on the stage, waiting for who is in it or for its species.
struct BoxesDrawn: Identifiable {
    let id = UUID()
    let uid: String
    let kind: StageDrawKind
    let rect: CGRect
    let vault: Bool
}

/// What the boxes area keeps while the viewer is open: the box being edited (drawn selected), the stage's edit
/// (moving a box, drawing one), and the sheets.
@MainActor
@Observable
final class ViewerBoxesModel {
    /// Shared with the viewer core and the stage through the environment.
    let stage = StageEditState()
    /// The box whose editor is open, from a tap on the stage.
    var editing: BoxesEditorTarget?
    /// A box shown for a moment from the info panel ("Show in the photo").
    var spotlight: String?
    /// A drawn person or face box whose picker is open.
    var naming: BoxesDrawn?
    /// A drawn pet box whose species list is open.
    var choosingSpecies: BoxesDrawn?
    /// What went wrong saving a moved box, or adding a drawn one: shown in the stage's bar.
    var adjustError: String?
    var drawError: String?
    var saving = false
    @ObservationIgnored private var spotlightTask: Task<Void, Never>?

    /// The box drawn selected on the stage (never raised above a smaller one).
    var selectedBoxId: String? { editing?.boxKey ?? spotlight }

    /// A tapped box, always the smallest under the finger: its editor opens.
    func tap(_ box: StageBox, in context: ViewerPageContext) {
        guard !stage.isActive else { return }
        stage.contexts[context.uid] = context
        spotlight = nil
        editing = BoxesEditorTarget(uid: context.uid, kind: box.kind, boxId: box.boxId, vault: context.vault)
    }

    /// "Move or resize": the editor closes and the box gets handles on the stage.
    func startAdjusting(_ box: StageBox, uid: String) {
        editing = nil
        adjustError = nil
        stage.startAdjusting(box, on: uid)
    }

    /// "Add a face box", "Add a person box", "Add a pet box": the owner drags a box over the picture.
    func startDrawing(_ kind: StageDrawKind, in context: ViewerPageContext) {
        editing = nil
        drawError = nil
        stage.contexts[context.uid] = context
        stage.startDrawing(kind, on: context.uid)
    }

    /// Leaves moving or drawing without saving.
    func cancelEdit() {
        stage.stop()
        adjustError = nil
        drawError = nil
        naming = nil
        choosingSpecies = nil
        saving = false
    }

    /// The item being edited came from the vault.
    var editVault: Bool { stage.uid.flatMap { stage.contexts[$0]?.vault } ?? false }

    /// Saves the moved or resized box; a box that did not move just leaves.
    func saveAdjust() async {
        guard let box = stage.adjustingBox, let rect = stage.rect, let uid = stage.uid, !saving else { return }
        let start = CGRect(x: box.x, y: box.y, width: box.w, height: box.h)
        guard StageEditGeometry.changed(start, rect) else {
            cancelEdit()
            return
        }
        saving = true
        adjustError = nil
        let error = await BoxesActions.move(box, to: rect, uid: uid, vault: editVault)
        saving = false
        if let error { adjustError = error } else { cancelEdit() }
    }

    /// The drawn box's next step: who is in it, or which species it is.
    func nameDrawn() {
        guard let kind = stage.drawingKind, let rect = stage.rect, let uid = stage.uid else { return }
        let drawn = BoxesDrawn(uid: uid, kind: kind, rect: rect, vault: editVault)
        drawError = nil
        if kind == .pet { choosingSpecies = drawn } else { naming = drawn }
    }

    /// Adds the drawn person or face box with who is in it.
    func addDrawn(_ drawn: BoxesDrawn, pick: PersonPick) async {
        saving = true
        let error = await BoxesActions.addBox(drawn.kind, rect: drawn.rect, pick: pick, uid: drawn.uid, vault: drawn.vault)
        saving = false
        if let error { drawError = error } else { cancelEdit() }
    }

    /// Adds the drawn pet box with its species, then opens its editor to name the pet.
    func addDrawnPet(_ drawn: BoxesDrawn, species: API.Species) async {
        saving = true
        let added = await BoxesActions.addPet(rect: drawn.rect, species: species, uid: drawn.uid, vault: drawn.vault)
        saving = false
        guard let added else { return }
        cancelEdit()
        // The species list is still going away: the editor opens once it has.
        try? await Task.sleep(nanoseconds: 600_000_000)
        editing = BoxesEditorTarget(uid: drawn.uid, kind: .animal, boxId: added.id, vault: drawn.vault, fresh: true)
    }

    /// Shows a box on the stage for a moment.
    func show(_ boxKey: String) {
        spotlight = boxKey
        spotlightTask?.cancel()
        spotlightTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            if !Task.isCancelled { self?.spotlight = nil }
        }
    }
}

/// The boxes area's buttons in the viewer's bottom bar: adding a box by drawing it, and asking the detector to
/// look for pets again. Drawing needs the still picture, so it is offered on photos.
struct ViewerBoxesToolbarItems: View {
    let model: ViewerBoxesModel
    let context: ViewerPageContext

    var body: some View {
        let kind = context.detail.mediaKind
        if kind == .image || kind == .video {
            Menu {
                if kind == .image {
                    Button { model.startDrawing(.face, in: context) } label: {
                        Label("Add a face box", systemImage: "faceid")
                    }
                    Button { model.startDrawing(.person, in: context) } label: {
                        Label("Add a person box", systemImage: "person.crop.rectangle")
                    }
                    Button { model.startDrawing(.pet, in: context) } label: {
                        Label("Add a pet box", systemImage: "pawprint")
                    }
                }
                Button {
                    Task { await BoxesActions.findPets(uid: context.uid, vault: context.vault) }
                } label: {
                    Label("Find pets again", systemImage: "arrow.clockwise")
                }
            } label: {
                Image(systemName: "plus.viewfinder")
                    .frame(width: 44, height: 44)
            }
            .accessibilityLabel("Add a box")
            .accessibilityIdentifier("boxes.add")
        }
    }
}

extension View {
    /// The boxes area's sheets and the stage's edit bars, attached once to the viewer.
    func viewerBoxEditors(_ model: ViewerBoxesModel) -> some View {
        modifier(ViewerBoxesSheets(model: model))
    }
}

/// The editors' sheets, the bars shown while a box is moved or drawn, and the edit state in the environment (the
/// viewer core and the stage read it).
struct ViewerBoxesSheets: ViewModifier {
    @Bindable var model: ViewerBoxesModel

    func body(content: Content) -> some View {
        content
            .environment(model.stage)
            .overlay(alignment: .top) { BoxesEditTopBar(model: model) }
            .overlay(alignment: .bottom) { BoxesEditBottomBar(model: model) }
            .sheet(item: $model.editing) { target in
                BoxesEditorSheet(model: model, target: target) { model.editing = nil }
                    .presentationDetents([.medium, .large])
            }
            .sheet(item: $model.naming) { drawn in
                PersonPickerSheet(title: drawn.kind == .face ? "Whose face is in the box?" : "Who is in the box?",
                                  kind: .person, like: .item(drawn.uid)) { pick in
                    Task { await model.addDrawn(drawn, pick: pick) }
                }
            }
            .sheet(item: $model.choosingSpecies) { drawn in
                BoxesSpeciesSheet(title: "What is it?", current: nil, note: nil) { species in
                    Task { await model.addDrawnPet(drawn, species: species) }
                }
                .presentationDetents([.medium, .large])
            }
    }
}

/// While drawing: what to do, "Whole photo" and Cancel, at the top of the picture.
struct BoxesEditTopBar: View {
    let model: ViewerBoxesModel

    var body: some View {
        let stage = model.stage
        if let kind = stage.drawingKind, stage.rect == nil || stage.dragging {
            HStack(spacing: 8) {
                Text("Drag over the \(kind.noun)")
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.black.opacity(0.7), in: Capsule())
                    .accessibilityIdentifier("boxes.draw.hint")
                if kind != .face {
                    Button("Whole photo") { stage.setRect(StageEditGeometry.wholePhoto) }
                        .buttonStyle(.bordered)
                        .accessibilityIdentifier("boxes.draw.whole")
                }
                Button("Cancel") { model.cancelEdit() }
                    .buttonStyle(.bordered)
                    .accessibilityIdentifier("boxes.draw.cancel")
            }
            .foregroundStyle(.white)
            .tint(.white)
            .padding(.top, 8)
            .padding(.horizontal, 12)
        }
    }
}

/// While moving a box: Save and Cancel. After drawing a box: who is in it (or its species), Draw again, Cancel.
struct BoxesEditBottomBar: View {
    let model: ViewerBoxesModel

    var body: some View {
        let stage = model.stage
        Group {
            if let box = stage.adjustingBox {
                adjustBar(box)
            } else if let kind = stage.drawingKind, stage.rect != nil, !stage.dragging {
                drawnCard(kind)
            }
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
    }

    private func adjustBar(_ box: StageBox) -> some View {
        let label = adjustLabel(box)
        return VStack(alignment: .leading, spacing: 8) {
            Text("Move or resize \(label)")
                .font(.subheadline.weight(.semibold))
            Text("Drag the box to move it, or a corner to resize it. Two fingers move and zoom the photo.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            if let error = model.adjustError {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .accessibilityIdentifier("boxes.adjust.error")
            }
            HStack {
                Button("Cancel") { model.cancelEdit() }
                    .buttonStyle(.bordered)
                    .disabled(model.saving)
                    .accessibilityIdentifier("boxes.adjust.cancel")
                Spacer()
                Button {
                    Task { await model.saveAdjust() }
                } label: {
                    if model.saving { ProgressView() } else { Text("Save") }
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.saving)
                .accessibilityIdentifier("boxes.adjust.save")
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18))
    }

    private func drawnCard(_ kind: StageDrawKind) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: kind == .face ? "faceid" : kind == .pet ? "pawprint" : "person.crop.rectangle")
                    .font(.title3)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(kind == .face ? "Face box" : kind == .pet ? "Pet box" : "Person box")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(kind == .pet ? "What is it?" : "Who is it?")
                        .font(.headline)
                }
            }
            Text(hint(kind))
                .font(.footnote)
                .foregroundStyle(.secondary)
            if let error = model.drawError {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .accessibilityIdentifier("boxes.draw.error")
            }
            HStack {
                Button(kind == .pet ? "Choose the species" : "Who is it?") { model.nameDrawn() }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.saving)
                    .accessibilityIdentifier("boxes.draw.next")
                Spacer()
                Button("Draw again") {
                    model.drawError = nil
                    model.stage.setRect(nil)
                }
                .buttonStyle(.bordered)
                .disabled(model.saving)
                .accessibilityIdentifier("boxes.draw.again")
                Button("Cancel") { model.cancelEdit() }
                    .buttonStyle(.bordered)
                    .disabled(model.saving)
                    .accessibilityIdentifier("boxes.draw.cancelBox")
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18))
    }

    private func hint(_ kind: StageDrawKind) -> String {
        switch kind {
        case .face:
            return "Draw it around the face: the face has to show. For a helmet, a back or someone far away, add a person box instead."
        case .person:
            return "A helmet, a back or someone far away works too. A face inside the box is tagged as well."
        case .pet:
            return "Pick what kind of animal it is, then name the pet."
        }
    }

    /// "the face box of Sam", from the item as it is now, else from what the stage drew.
    private func adjustLabel(_ box: StageBox) -> String {
        if let uid = model.stage.uid, let item = model.stage.contexts[uid]?.item,
           let target = BoxesLookup.target(in: item, kind: box.kind, id: box.boxId) {
            return target.adjustLabel
        }
        let text = box.accessibilityText
        return text.prefix(1).lowercased() + String(text.dropFirst())
    }
}
