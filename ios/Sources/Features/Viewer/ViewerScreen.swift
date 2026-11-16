import SwiftUI

/// The viewer. Placeholder of the core (the viewer area replaces this file and fills ViewerHooks: the face,
/// person and pet editors for a tapped box, the info panel, the toolbar). Here a tapped box only says whose it is.
struct ViewerScreen: View {
    let request: ViewerRequest

    @State private var tapped: StageBox?

    var body: some View {
        ViewerCore(request: request,
                   hooks: ViewerHooks(onBoxTap: { box, _ in tapped = box }, selectedBoxId: tapped?.id))
            .sheet(item: $tapped) { box in
                VStack(spacing: 8) {
                    Text(box.accessibilityText)
                        .font(.title3.weight(.semibold))
                        .accessibilityIdentifier("viewer.box.title")
                    Text("Editing who is in a photo comes with the viewer's editors.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding(24)
                .presentationDetents([.height(160)])
            }
    }
}
