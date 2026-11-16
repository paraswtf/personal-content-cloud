import SwiftUI

/// The Library tab. Placeholder of the core (the library area replaces this file): every item, newest first, in
/// the media grid, opening in the viewer, with selection.
struct LibraryScreen: View {
    @Environment(AppRouter.self) private var router
    @State private var pager = Pager<API.ItemCard> { cursor in
        try await Cloud.api().listItems(cursor: cursor, limit: 120).pagerPage
    }
    @State private var selection = SelectionModel<String>()

    var body: some View {
        PagerStateView(pager: pager, emptyTitle: "No photos yet", emptyImage: "photo.on.rectangle",
                       emptyText: "Photos and videos you upload appear here.") {
            MediaGrid(items: pager.items, selection: selection,
                      onOpen: { item in open(item) },
                      onAppear: { item in pager.loadMoreIfNeeded(item) })
        }
        .navigationTitle("Library")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button(selection.active ? "Done" : "Select") {
                    if selection.active { selection.end() } else { selection.begin() }
                }
                .disabled(pager.items.isEmpty)
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
        router.show(pager.items, start: item) {
            await pager.loadMore()
            return pager.items.map(\.viewerEntry)
        }
    }
}
