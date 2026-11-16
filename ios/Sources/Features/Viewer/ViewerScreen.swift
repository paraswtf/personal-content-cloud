import SwiftUI

/// The viewer: the core viewer (UI/Viewer) with the viewer area's info panel, toolbar and own pages, and the boxes
/// area's editors (Features/ViewerBoxes). The viewer area owns this file and the files next to it; the lines that
/// hand the boxes area its part (`boxes`, `ViewerBoxesToolbarItems`, `ViewerBoxesInfoSection`,
/// `.viewerBoxEditors`) stay, whatever else changes.
struct ViewerScreen: View {
    let request: ViewerRequest

    @State private var boxes = ViewerBoxesModel()
    @State private var pages = ViewerPageState()

    var body: some View {
        ViewerCore(request: request,
                   hooks: ViewerHooks(onBoxTap: { box, context in boxes.tap(box, in: context) },
                                      infoPanel: { context in AnyView(ViewerInfoPanel(context: context, boxes: boxes)) },
                                      toolbar: { context in AnyView(ViewerToolbar(context: context, boxes: boxes, pages: pages)) },
                                      stage: { context, environment in
                                          ViewerStages.stage(for: context, environment, boxes: boxes, pages: pages)
                                      },
                                      selectedBoxId: boxes.selectedBoxId))
            .viewerBoxEditors(boxes)
    }
}
