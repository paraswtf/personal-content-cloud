import SwiftUI

/// One album (Route.album; the website's AlbumPage, docs/ARCHITECTURE.md section 24): the library's browser with the
/// album's members, under a header with the description, the count and date range, and the album's actions (Share
/// album, Rename and description, Delete album). The browser's selection has the album's own bulk actions, such
/// as Remove from album (the library area's LibraryBrowser).
struct AlbumScreen: View {
    let albumId: Int

    @Environment(\.dismiss) private var dismiss
    @State private var album: API.AlbumOut?
    @State private var loadError: Error?
    @State private var missing = false
    @State private var deleted = false
    @State private var editing = false
    @State private var sharing = false
    @State private var confirmDelete = false

    var body: some View {
        content
            .navigationTitle(album?.name ?? "Album")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if album != nil { actionsMenu }
                }
            }
            .task { if album == nil { await load() } }
            .onCloudChange { Task { await load() } }
            .sheet(isPresented: $editing) {
                if let current = album {
                    AlbumsEditSheet(mode: .edit(current)) { saved in album = saved }
                }
            }
            .sheet(isPresented: $sharing) {
                if let current = album {
                    SharingCreateSheet(target: .album(id: current.id, title: current.name))
                }
            }
            .confirmAction("Delete the album \(album?.name ?? "")?", isPresented: $confirmDelete,
                           actionTitle: "Delete album",
                           message: "The album goes away. Its photos, videos and documents stay in the library.") {
                Task { await delete() }
            }
    }

    @ViewBuilder
    private var content: some View {
        if missing {
            ContentUnavailableView {
                Label("This album does not exist", systemImage: "photo.stack")
            } description: {
                Text("It may have been deleted.")
            } actions: {
                Button("All albums") { dismiss() }
                    .buttonStyle(.bordered)
            }
        } else if let current = album {
            LibraryBrowser(scope: .album(id: albumId), header: AnyView(AlbumsHeader(album: current)))
        } else if let loadError {
            ErrorState(error: loadError) { Task { await load() } }
        } else {
            LoadingState()
        }
    }

    private var actionsMenu: some View {
        Menu {
            Button {
                sharing = true
            } label: {
                Label("Share album", systemImage: "square.and.arrow.up")
            }
            Button {
                editing = true
            } label: {
                Label("Rename", systemImage: "pencil")
            }
            Button(role: .destructive) {
                confirmDelete = true
            } label: {
                Label("Delete album", systemImage: "trash")
            }
        } label: {
            Label("Album actions", systemImage: "ellipsis.circle")
        }
        .accessibilityIdentifier("albums.menu")
    }

    private func load() async {
        guard !deleted else { return }
        do {
            let fresh = try await Cloud.api().getAlbum(albumId: albumId)
            album = fresh
            loadError = nil
            missing = false
        } catch is CancellationError {
            return
        } catch let error as APIError where error.status == 404 {
            missing = true
        } catch {
            if album == nil { loadError = error }
        }
    }

    private func delete() async {
        guard let current = album else { return }
        do {
            try await AlbumsActions.run("Album \(current.name) deleted. Its photos stay in the library.") {
                try await Cloud.api().deleteAlbum(albumId: current.id)
            }
            deleted = true
            CloudEvents.post()
            dismiss()
        } catch {
            Toaster.shared.error(error, doing: "delete the album")
        }
    }
}

/// The album page's header: the description and the facts (count, date range, private items).
struct AlbumsHeader: View {
    let album: API.AlbumOut

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if !album.description.isEmpty {
                Text(album.description)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
            }
            Text(AlbumsText.facts(album))
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(.bar)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("albums.header")
    }
}
