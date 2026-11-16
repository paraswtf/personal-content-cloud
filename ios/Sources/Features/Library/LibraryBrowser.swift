import SwiftUI

// The library's browser and filters (the website's components/library/Library.tsx with its `album` and `vault`
// props, FilterBar.tsx and filters.ts). The library area builds them; the album screen shows
// LibraryBrowser(scope: .album(id:)), the vault screen LibraryBrowser(scope: .vault) while the vault is open,
// and the map uses LibraryFilters with LibraryFilterBar. Their names and parameters stay. Placeholders until
// then: the scope's items, newest first, in the media grid, with selection.

/// Where a library browser looks: the whole library, one album, or the vault (every request with vault=1).
enum LibraryScope: Hashable {
    case all
    case album(id: Int)
    case vault
}

/// The library's filters, one field per filter parameter of the item lists (`listItems`, `buckets`, `points`,
/// `placeList`), so other screens pass them to their own endpoints.
struct LibraryFilters: Hashable {
    var category: API.Category?
    var kind: API.Kind?
    var person: [Int]?
    var tag: [String]?
    var live: Bool?
    var from: String?
    var to: String?
    var favorite: Bool?
    var misclicks: API.Forwards?
    var forwards: API.Forwards?
    var received: String?
    var dups: API.Dups?
    var sort: API.Sort?
}

/// The library's filter controls for a scope. Placeholder: none yet.
struct LibraryFilterBar: View {
    @Binding var filters: LibraryFilters
    var scope: LibraryScope = .all

    var body: some View {
        EmptyView()
    }
}

/// The library's browser for a scope: the timeline, filters, search, selection and its actions. `header` shows
/// above the items (the album's title and buttons, for example).
struct LibraryBrowser: View {
    let scope: LibraryScope
    var header: AnyView?

    @Environment(AppRouter.self) private var router
    @State private var pager: Pager<API.ItemCard>
    @State private var selection = SelectionModel<String>()

    init(scope: LibraryScope, header: AnyView? = nil) {
        self.scope = scope
        self.header = header
        _pager = State(initialValue: Pager<API.ItemCard> { cursor in
            switch scope {
            case .all:
                return try await Cloud.api().listItems(cursor: cursor, limit: 120).pagerPage
            case .album(let id):
                return try await Cloud.api().listItems(album: id, cursor: cursor, limit: 120).pagerPage
            case .vault:
                return try await APIContext.$vault.withValue(true) {
                    try await Cloud.api().listItems(cursor: cursor, limit: 120).pagerPage
                }
            }
        })
    }

    var body: some View {
        PagerStateView(pager: pager, emptyTitle: "Nothing here yet", emptyImage: "photo.on.rectangle",
                       emptyText: "Photos and videos appear here.") {
            MediaGrid(items: pager.items, selection: selection,
                      onOpen: { item in open(item) },
                      onAppear: { item in pager.loadMoreIfNeeded(item) })
        }
        .safeAreaInset(edge: .top) {
            if let header { header }
        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if !selection.active {
                    Button("Select") { selection.begin() }
                        .disabled(pager.items.isEmpty)
                        .accessibilityIdentifier("library.select")
                }
            }
        }
        .safeAreaInset(edge: .bottom) {
            if selection.active {
                BulkBar(count: selection.count, actions: [],
                        onSelectAll: { selection.select(pager.items.map(\.uid)) },
                        onDone: { selection.end() })
            }
        }
        .task { if !pager.loaded { await pager.refresh() } }
        .refreshable { await pager.refresh() }
        .onCloudChange { Task { await pager.refresh() } }
    }

    private func open(_ item: API.ItemCard) {
        router.show(pager.items, start: item, vault: scope == .vault) {
            await pager.loadMore()
            return pager.items.map(\.viewerEntry)
        }
    }
}
