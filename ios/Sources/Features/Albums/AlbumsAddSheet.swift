import SwiftUI

/// Adds items to an album or to a new one (the website's AlbumPicker and the bulk bar's Add to album). The library,
/// the viewer and the people screens present it as a sheet: `.sheet(isPresented: $adding) { AlbumsAddSheet(uids: uids) }`.
/// `onDone` runs after the items were added (to end a selection, for example). For one item, the albums it is in
/// already are shown but cannot be picked.
struct AlbumsAddSheet: View {
    let uids: [String]
    var onDone: (() -> Void)? = nil

    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @State private var albums: [API.AlbumOut] = []
    @State private var loaded = false
    @State private var loadError: Error?
    @State private var inAlready: Set<Int> = []
    @State private var creating = false
    @State private var newName = ""
    @State private var busy = false
    @State private var errorText: String?
    @FocusState private var newNameFocused: Bool

    private var trimmedQuery: String { query.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var trimmedNewName: String { newName.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        NavigationStack {
            List {
                if let errorText {
                    Section {
                        Text(errorText)
                            .foregroundStyle(.red)
                            .accessibilityIdentifier("albums.add.error")
                    }
                }
                Section {
                    if creating {
                        HStack(spacing: 12) {
                            TextField("New album name", text: $newName)
                                .focused($newNameFocused)
                                .submitLabel(.done)
                                .onSubmit { Task { await createAndAdd() } }
                                .accessibilityLabel("New album name")
                                .accessibilityIdentifier("albums.add.newName")
                            Button("Create") { Task { await createAndAdd() } }
                                .buttonStyle(.borderedProminent)
                                .disabled(trimmedNewName.isEmpty || busy)
                                .accessibilityIdentifier("albums.add.create")
                            // folds the field back to "Create a new album" (the website's Cancel); the sheet stays
                            Button("Cancel") {
                                creating = false
                                newName = ""
                                newNameFocused = false
                            }
                            .buttonStyle(.bordered)
                            .disabled(busy)
                            .accessibilityIdentifier("albums.add.cancelNew")
                        }
                    } else {
                        Button {
                            creating = true
                            newNameFocused = true
                        } label: {
                            Label("Create a new album", systemImage: "plus")
                        }
                        .disabled(busy)
                        .accessibilityIdentifier("albums.add.new")
                    }
                }
                Section {
                    if !loaded {
                        ProgressView()
                            .frame(maxWidth: .infinity)
                    } else if albums.isEmpty, let loadError {
                        Text(ErrorText.describe(loadError, doing: "load the albums"))
                            .foregroundStyle(.secondary)
                    } else if albums.isEmpty {
                        Text(trimmedQuery.isEmpty ? "No albums yet. Make one above." : "No album matches.")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(albums, id: \.id) { album in
                            row(album)
                        }
                    }
                } header: {
                    Text("Albums")
                }
            }
            .searchable(text: $query, placement: .navigationBarDrawer(displayMode: .always), prompt: "Find an album")
            .navigationTitle(uids.count == 1 ? "Add to album" : "Add \(Fmt.count(uids.count, "item")) to album")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                        .disabled(busy)
                }
                ToolbarItem(placement: .confirmationAction) {
                    if busy { ProgressView() }
                }
            }
            .task(id: trimmedQuery) { await load() }
            .task { await loadAlbumsOfItem() }
        }
        .interactiveDismissDisabled(busy)
    }

    private func row(_ album: API.AlbumOut) -> some View {
        let already = inAlready.contains(album.id)
        return Button {
            Task { await add(to: album) }
        } label: {
            HStack(spacing: 12) {
                AlbumsCover(album: album, cornerRadius: 8, maxPixel: 128, showsVaultCount: false)
                    .frame(width: 44, height: 44)
                VStack(alignment: .leading, spacing: 2) {
                    Text(album.name)
                        .font(.body.weight(.semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    Text(already ? "Already in this album" : Fmt.count(album.count, "item"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(busy || already)
        .accessibilityLabel(already ? "\(album.name), already in this album" : "\(album.name), \(Fmt.count(album.count, "item"))")
        .accessibilityIdentifier("albums.add.album")
    }

    /// The albums newest change first, filtered by the search once typing pauses.
    private func load() async {
        if loaded {
            try? await Task.sleep(nanoseconds: 200_000_000)
            if Task.isCancelled { return }
        }
        do {
            let text = trimmedQuery
            let page = try await Cloud.api().listAlbums(q: text.isEmpty ? nil : text, sort: .updated, limit: 50)
            if Task.isCancelled { return }
            albums = page.items
            loadError = nil
        } catch is CancellationError {
            return
        } catch {
            loadError = error
        }
        loaded = true
    }

    /// For one item: the albums it is in already (the item's details name them).
    private func loadAlbumsOfItem() async {
        guard uids.count == 1, let uid = uids.first else { return }
        if let detail = try? await Cloud.api().getItem(uid: uid) {
            inAlready = Set(detail.albums.map(\.id))
        }
    }

    private func add(to album: API.AlbumOut) async {
        guard !busy else { return }
        busy = true
        errorText = nil
        defer { busy = false }
        do {
            let items = uids
            try await AlbumsActions.runCounting("Added to \(album.name).",
                                                message: { (out: API.AlbumItemsOut) in AlbumsText.added(out.added, to: album.name) }) {
                try await Cloud.api().addItems(albumId: album.id, body: API.AlbumItemsIn(uids: items))
            }
            CloudEvents.post(items: items)
            onDone?()
            dismiss()
        } catch {
            errorText = AlbumsText.error(error, doing: "change the album")
        }
    }

    /// Makes the album with the items in it, in one request.
    private func createAndAdd() async {
        let name = trimmedNewName
        guard !name.isEmpty, !busy else { return }
        busy = true
        errorText = nil
        defer { busy = false }
        do {
            let items = uids
            try await AlbumsActions.runCounting("Album \(name) created.",
                                                message: { (album: API.AlbumOut) in AlbumsText.added(album.count, to: album.name) }) {
                try await Cloud.api().createAlbum(body: API.AlbumIn(name: name, itemUids: items))
            }
            CloudEvents.post(items: items)
            onDone?()
            dismiss()
        } catch {
            errorText = AlbumsText.error(error, doing: "create the album")
        }
    }
}
