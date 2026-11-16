import SwiftUI

/// The Albums tab (the website's AlbumsPage, docs/ARCHITECTURE.md section 24): the album cards (cover, name, count
/// and date range), a sort menu, a search, New album, and an empty state that says how albums are filled.
struct AlbumsScreen: View {
    @Environment(AppRouter.self) private var router
    @State private var sort: API.ListAlbumsSort = .updated
    @State private var query = ""
    @State private var pager = AlbumsScreen.makePager(sort: .updated, text: "")
    @State private var loadedKey: AlbumsQueryKey?
    @State private var creating = false
    @State private var created: API.AlbumOut?
    /// Screen tests only: "Add to album" opened straight away (see openTestAddSheet).
    @State private var testAdd: AlbumsTestAdd?

    private var key: AlbumsQueryKey { AlbumsQueryKey(sort: sort.rawValue, text: trimmedQuery) }
    private var trimmedQuery: String { query.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        PagerStateView(pager: pager,
                       emptyTitle: trimmedQuery.isEmpty ? "No albums yet" : "No album matches",
                       emptyImage: "photo.stack",
                       emptyText: trimmedQuery.isEmpty
                           ? "Select photos in the library and choose Add to album, or make an empty album here and fill it later."
                           : "No album has \u{201C}\(trimmedQuery)\u{201D} in its name.") {
            grid
        }
        .navigationTitle("Albums")
        .searchable(text: $query, placement: .navigationBarDrawer(displayMode: .always), prompt: "Find an album")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) { sortMenu }
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    creating = true
                } label: {
                    Label("New album", systemImage: "plus")
                }
                .accessibilityIdentifier("albums.new")
            }
        }
        .task(id: key) { await load(for: key) }
        .task { await openTestAddSheet() }
        .refreshable { await pager.refresh() }
        .onCloudChange { Task { await pager.refresh() } }
        .sheet(isPresented: $creating, onDismiss: {
            if let album = created {
                created = nil
                router.open(.album(id: album.id))
            }
        }) {
            AlbumsEditSheet(mode: .create) { album in created = album }
        }
        .sheet(item: $testAdd) { request in
            AlbumsAddSheet(uids: request.uids)
        }
    }

    private var grid: some View {
        ScrollView {
            if let total = pager.total {
                Text(Fmt.count(total, "album"))
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.top, 4)
                    .accessibilityIdentifier("albums.count")
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150, maximum: 260), spacing: 12)], spacing: 18) {
                ForEach(pager.items) { row in
                    NavigationLink(value: Route.album(id: row.id)) {
                        AlbumsCard(album: row.album)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(AlbumsText.cardLabel(row.album))
                    .accessibilityIdentifier("albums.card")
                    .onAppear { pager.loadMoreIfNeeded(row) }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            if pager.loading && pager.loaded {
                ProgressView().padding()
            }
        }
    }

    private var sortMenu: some View {
        Menu {
            Picker("Sort albums", selection: $sort) {
                Text("Last changed").tag(API.ListAlbumsSort.updated)
                Text("Name").tag(API.ListAlbumsSort.name)
                Text("Newest first").tag(API.ListAlbumsSort.created)
            }
        } label: {
            Label("Sort albums", systemImage: "arrow.up.arrow.down")
        }
        .accessibilityIdentifier("albums.sort")
    }

    /// Loads the list for the sort and search; a change after the first load waits for typing to pause.
    private func load(for key: AlbumsQueryKey) async {
        if loadedKey == key { return }
        if loadedKey != nil {
            try? await Task.sleep(nanoseconds: 250_000_000)
            if Task.isCancelled { return }
        }
        let next = Self.makePager(sort: sort, text: trimmedQuery)
        pager = next
        loadedKey = key
        await next.refresh()
    }

    private static func makePager(sort: API.ListAlbumsSort, text: String) -> Pager<AlbumsRow> {
        Pager<AlbumsRow> { cursor in
            let page = try await Cloud.api().listAlbums(q: text.isEmpty ? nil : text, sort: sort, cursor: cursor, limit: 60)
            return Pager<AlbumsRow>.Page(items: page.items.map { AlbumsRow(album: $0) }, next: page.nextCursor,
                                         total: page.total)
        }
    }

    /// Screen tests only (Debug builds): with PCC_TEST_ALBUMS_ADD_FROM set to an album's name, opens Add to album for
    /// that album's items. The buttons that open it in the app belong to the library, the viewer and the people
    /// screens, so this is how the albums area's own test reaches it.
    private func openTestAddSheet() async {
        #if DEBUG
        guard testAdd == nil, let name = ProcessInfo.processInfo.environment["PCC_TEST_ALBUMS_ADD_FROM"],
              !name.isEmpty else { return }
        do {
            let page = try await Cloud.api().listAlbums(q: name, limit: 10)
            guard let album = page.items.first(where: { $0.name == name }) else { return }
            let items = try await Cloud.api().listItems(album: album.id, limit: 50)
            testAdd = AlbumsTestAdd(uids: items.items.map(\.uid))
        } catch {
            Toaster.shared.error(error, doing: "open Add to album")
        }
        #endif
    }
}

/// The list's sort and search, which decide when it loads again.
struct AlbumsQueryKey: Hashable {
    let sort: String
    let text: String
}

/// The items "Add to album" was opened for by a screen test.
struct AlbumsTestAdd: Identifiable {
    let id = UUID()
    let uids: [String]
}

/// One album's card: the cover, the name, the count and the date range.
struct AlbumsCard: View {
    let album: API.AlbumOut

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            AlbumsCover(album: album)
                .shadow(color: .black.opacity(0.08), radius: 4, y: 2)
            VStack(alignment: .leading, spacing: 2) {
                Text(album.name)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Text(AlbumsText.cardLine(album))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, 2)
        }
        .contentShape(Rectangle())
    }
}
