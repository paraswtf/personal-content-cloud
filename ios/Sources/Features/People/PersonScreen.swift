import SwiftUI

/// One person's page (Route.person). Placeholder of the core (the people area replaces this file). Test scaffolding
/// of the 3D heads branch: the person's 3D tab in a scroll view, as the person page will hold it.
struct PersonScreen: View {
    let personId: Int

    @State private var name = ""

    var body: some View {
        ScrollView {
            HeadsPersonTab(personId: personId, name: name)
        }
        .navigationTitle(name.isEmpty ? "Person" : name)
        .navigationBarTitleDisplayMode(.inline)
        .task { name = (try? await Cloud.api().getPerson(personId: personId).name ?? "") ?? "" }
    }
}
