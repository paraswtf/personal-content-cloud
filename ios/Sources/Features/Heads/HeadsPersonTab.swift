import SwiftUI

/// A person's 3D head (the website's "3D" tab on a person's page: components/people/Model3dTab.tsx and
/// Head3dViewer.tsx). The heads area builds it; the people area shows it as the person page's 3D tab and gives it
/// the room the tab has. Its name and signature stay. Placeholder until then.
struct HeadsPersonTab: View {
    let personId: Int
    let name: String

    var body: some View {
        ContentUnavailableView {
            Label("3D head", systemImage: "cube")
        } description: {
            Text("The 3D head is on its way to the app. The website has it today.")
        }
    }
}
