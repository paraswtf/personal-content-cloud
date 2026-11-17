import SwiftUI

/// The viewer: the core viewer (UI/Viewer) with the viewer area's info panel, toolbar and own pages, and the boxes
/// area's editors (Features/ViewerBoxes). The viewer area owns this file and the files next to it; the lines that
/// hand the boxes area its part (`boxes`, `ViewerBoxesToolbarItems`, `ViewerBoxesInfoSection`,
/// `.viewerBoxEditors`) stay, whatever else changes.
struct ViewerScreen: View {
    let request: ViewerRequest

    @Environment(AppRouter.self) private var router
    @Environment(\.dismiss) private var dismiss
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
            .onAppear {
                // A box drawn on a video goes with the time of the frame on screen.
                let pages = pages
                boxes.frameClock = { uid in pages.playerTime(uid) }
            }
            .task(id: request.id) { await closeWhenTheVaultLocks() }
    }

    /// Vault items never stay on screen once the vault is locked (the website hides a private item when the vault's
    /// window ends): the viewer opened from the vault closes when the app's vault window runs out.
    private func closeWhenTheVaultLocks() async {
        guard request.vault else { return }
        while !Task.isCancelled {
            let session = VaultSession.shared
            if session.loaded && !session.isOpen() {
                // Read as locked: the vault's pictures leave memory too.
                session.expire()
                Toaster.shared.error("The vault is locked. Unlock the vault again to see private items.")
                if router.viewer?.id == request.id { router.viewer = nil } else { dismiss() }
                return
            }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
    }
}
