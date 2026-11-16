import SwiftUI
import Observation

// The boxes area's part of the viewer (the website's FaceEditor, PersonEdit, AnimalEditor, BoxAdjuster,
// PersonBoxCard and GuessChip). The viewer area wires these into ViewerHooks (Features/Viewer/ViewerScreen.swift),
// so their names and signatures stay: ViewerBoxesModel (with tap(_:in:) and selectedBoxId),
// ViewerBoxesToolbarItems(model:context:), ViewerBoxesInfoSection(model:context:) and .viewerBoxEditors(_:).
// Until the editors are built, a tapped box says whose it is and opens the person picker ranked by likeness to
// that box, without saving anything.

/// What the boxes area keeps while the viewer is open: the box being edited (drawn selected) and the sheets.
@MainActor
@Observable
final class ViewerBoxesModel {
    /// The box whose editor is open.
    var editing: StageBox?
    /// The box whose person picker is open.
    var picking: StageBox?
    /// The box whose picker opens once the editor's sheet has gone (two sheets cannot change places at once).
    var pickNext: StageBox?

    /// The box drawn selected on the stage (never raised above a smaller one).
    var selectedBoxId: String? { editing?.id }

    /// A tapped box, always the smallest under the finger.
    func tap(_ box: StageBox, in context: ViewerPageContext) {
        editing = box
    }
}

/// The boxes area's buttons in the viewer's bottom bar (adding a box, for example). None yet.
struct ViewerBoxesToolbarItems: View {
    let model: ViewerBoxesModel
    let context: ViewerPageContext

    var body: some View {
        EmptyView()
    }
}

/// The info panel's section about the item's boxes: who is in it, the guesses, the faces nobody named. None yet.
struct ViewerBoxesInfoSection: View {
    let model: ViewerBoxesModel
    let context: ViewerPageContext

    var body: some View {
        EmptyView()
    }
}

extension View {
    /// The boxes area's sheets, attached once to the viewer.
    func viewerBoxEditors(_ model: ViewerBoxesModel) -> some View {
        modifier(ViewerBoxesSheets(model: model))
    }
}

/// The sheets a tapped box opens.
struct ViewerBoxesSheets: ViewModifier {
    @Bindable var model: ViewerBoxesModel

    func body(content: Content) -> some View {
        content
            .sheet(item: $model.editing, onDismiss: {
                model.picking = model.pickNext
                model.pickNext = nil
            }) { box in
                VStack(spacing: 12) {
                    Text(box.accessibilityText)
                        .font(.title3.weight(.semibold))
                        .accessibilityIdentifier("viewer.box.title")
                    Button("Who is this?") {
                        model.pickNext = box
                        model.editing = nil
                    }
                    .buttonStyle(.borderedProminent)
                    .accessibilityIdentifier("viewer.box.who")
                    Text("Saving a name comes with the viewer's editors.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding(24)
                .presentationDetents([.height(200)])
            }
            .sheet(item: $model.picking) { box in
                PersonPickerSheet(kind: box.kind == .animal ? .pet : .person, like: .box(box)) { pick in
                    switch pick {
                    case .existing(let option): Toaster.shared.ok("Picked \(option.name). Nothing was saved.")
                    case .new(let name, _): Toaster.shared.ok("Picked a new name, \(name). Nothing was saved.")
                    }
                }
            }
    }
}
