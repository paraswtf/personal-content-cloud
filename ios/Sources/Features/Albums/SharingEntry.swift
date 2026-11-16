import SwiftUI

// Share links, part of the albums area (the website's ShareModal and Settings > Sharing). Other areas use these
// two entry points; their names and signatures stay. Placeholders until the albums area builds them.

/// What a share link shows: one album, or items picked by the owner (`API.ShareCreate.kind`).
enum SharingTarget: Hashable {
    case album(id: Int, title: String)
    case items(uids: [String])
}

/// Makes a share link and shows it to copy or send. Presented as a sheet by the album screen, the library's
/// selection and the viewer: `.sheet(isPresented: $sharing) { SharingCreateSheet(target: .items(uids: uids)) }`.
struct SharingCreateSheet: View {
    let target: SharingTarget

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            PlaceholderScreen(title: "Share", systemImage: "square.and.arrow.up")
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                }
        }
    }
}

/// Every share link, to change or revoke. Settings opens it.
struct SharingLinksScreen: View {
    var body: some View {
        PlaceholderScreen(title: "Share links", systemImage: "link")
    }
}
