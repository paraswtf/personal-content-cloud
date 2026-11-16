import SwiftUI

/// The pages the viewer area draws itself (the website's DocumentStage and VideoStage: a document's pages, a note,
/// a video with its faces). Placeholder until the viewer area builds them: nil keeps the core's picture, video or
/// Live Photo for every item.
enum ViewerStages {
    static func stage(for context: ViewerPageContext, _ environment: ViewerStageEnvironment) -> AnyView? {
        nil
    }
}
