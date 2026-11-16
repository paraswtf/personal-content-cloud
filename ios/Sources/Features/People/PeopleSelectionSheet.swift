import SwiftUI

/// "People in N items" for a selection (the website's components/people/PeopleBulkAction.tsx): everyone the
/// selected items show, with Change to, Rename and Not this person, and Tag someone in these; on a person's or
/// pet's page (`fromPersonId`) first "Not NAME?". The people area builds it; the library's and the person page's
/// selections present it as a sheet and pass `onDone` to end the selection. Its name and parameters stay.
/// Placeholder until then.
struct PeopleSelectionSheet: View {
    let uids: [String]
    var vault = false
    var fromPersonId: Int? = nil
    var onDone: (() -> Void)? = nil

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            PlaceholderScreen(title: "People", systemImage: "person.2")
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                }
        }
    }
}
