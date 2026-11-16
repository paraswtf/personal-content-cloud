import SwiftUI

/// Adds items to an album or to a new one (the website's AlbumPicker). The albums area builds it; the library, the
/// viewer and the people screens present it as a sheet: `.sheet(isPresented: $adding) { AlbumsAddSheet(uids: uids) }`.
/// `onDone` runs after the items were added (to end a selection, for example). Placeholder until then.
struct AlbumsAddSheet: View {
    let uids: [String]
    var onDone: (() -> Void)? = nil

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            PlaceholderScreen(title: "Add to album", systemImage: "rectangle.stack.badge.plus")
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                }
        }
    }
}
