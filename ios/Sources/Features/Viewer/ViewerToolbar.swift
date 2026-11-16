import SwiftUI

/// The viewer's buttons in the bottom bar, next to the core's boxes button (favourite, share, albums, download,
/// vault, delete on the website). Placeholder until the viewer area builds it: only the boxes area's buttons.
/// Keep `ViewerBoxesToolbarItems` in the bar.
struct ViewerToolbar: View {
    let context: ViewerPageContext
    let boxes: ViewerBoxesModel

    var body: some View {
        ViewerBoxesToolbarItems(model: boxes, context: context)
    }
}
