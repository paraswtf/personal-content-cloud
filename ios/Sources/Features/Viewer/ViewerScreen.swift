import SwiftUI

/// The viewer. Placeholder of the core (the viewer area replaces this file and fills ViewerHooks: the face,
/// person and pet editors for a tapped box, the info panel, the toolbar). Here a tapped box says whose it is and
/// opens the person picker ranked by likeness to that box, without saving anything.
struct ViewerScreen: View {
    let request: ViewerRequest

    @State private var tapped: StageBox?
    @State private var picking: StageBox?
    /// The box whose picker opens once the first sheet has gone (two sheets cannot change places at once).
    @State private var pickNext: StageBox?

    var body: some View {
        ViewerCore(request: request,
                   hooks: ViewerHooks(onBoxTap: { box, _ in tapped = box }, selectedBoxId: tapped?.id))
            .sheet(item: $tapped, onDismiss: {
                picking = pickNext
                pickNext = nil
            }) { box in
                VStack(spacing: 12) {
                    Text(box.accessibilityText)
                        .font(.title3.weight(.semibold))
                        .accessibilityIdentifier("viewer.box.title")
                    Button("Who is this?") {
                        pickNext = box
                        tapped = nil
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
            .sheet(item: $picking) { box in
                PersonPickerSheet(kind: box.kind == .animal ? .pet : .person, like: .box(box)) { pick in
                    switch pick {
                    case .existing(let option): Toaster.shared.ok("Picked \(option.name). Nothing was saved.")
                    case .new(let name, _): Toaster.shared.ok("Picked a new name, \(name). Nothing was saved.")
                    }
                }
            }
    }
}
