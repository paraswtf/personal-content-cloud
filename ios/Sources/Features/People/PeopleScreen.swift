import SwiftUI

/// The People tab. Placeholder of the core (the people area replaces this file). Test scaffolding of the 3D heads
/// branch: the named people as a list, each opening its page, so the heads' screen test can reach the 3D tab.
struct PeopleScreen: View {
    @State private var people: [API.PersonOut] = []

    var body: some View {
        List(people.filter { Fmt.isNamed($0.name) }, id: \.id) { person in
            NavigationLink(value: Route.person(id: person.id)) {
                Text(person.name ?? "")
            }
        }
        .navigationTitle("People")
        .task { people = (try? await Cloud.api().listPeople(limit: 100).items) ?? [] }
    }
}
