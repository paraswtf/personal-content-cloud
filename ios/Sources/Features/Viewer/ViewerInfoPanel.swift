import SwiftUI

/// The viewer's info panel (the website's InfoPanel). Placeholder until the viewer area builds it: the boxes
/// area's section about who is in the item, then what the item is. Keep `ViewerBoxesInfoSection` in the panel.
struct ViewerInfoPanel: View {
    let context: ViewerPageContext
    let boxes: ViewerBoxesModel

    var body: some View {
        let d = context.detail
        List {
            ViewerBoxesInfoSection(model: boxes, context: context)
            Section {
                LabeledContent("Name", value: d.filename)
                LabeledContent("Kind", value: d.mediaKind.label)
                if let taken = d.takenAt { LabeledContent("Taken", value: Fmt.date(taken, withTime: true)) }
                LabeledContent("Added", value: Fmt.date(d.createdAt, withTime: true))
                if let place = d.place, !place.isEmpty { LabeledContent("Place", value: place) }
            }
        }
        .navigationTitle("Details")
        .navigationBarTitleDisplayMode(.inline)
    }
}
