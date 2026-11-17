import SwiftUI

// The library's browser (the website's components/library/Library.tsx with its `album` and `vault` props). The
// Library tab shows LibraryBrowser(scope: .all), the album screen LibraryBrowser(scope: .album(id:)) under its own
// header, and the vault screen LibraryBrowser(scope: .vault) while the vault is open. Their names and parameters
// stay.

/// The library's browser for a scope: the timeline by month with its scrubber, the search with its modes, the
/// filters, multi-select with every bulk action, and the viewer over the list. `header` shows above the items and
/// scrolls with them (the album's title and buttons, the vault's lock).
struct LibraryBrowser: View {
    let scope: LibraryScope
    var header: AnyView?
    /// The screen's title, which reads "Search results" while searching (the website's heading). Nil leaves the
    /// title to the screen around the browser.
    var title: String?

    @Environment(AppRouter.self) private var router
    @State private var model: LibraryModel
    @State private var sheet: LibraryBulkSheet?
    @State private var sheetUids: [String] = []
    @State private var confirmDelete = false
    @State private var showingFilters = false

    /// `filters`: filters set from the start (Route.libraryFiltered: a tag tapped in the viewer).
    init(scope: LibraryScope, header: AnyView? = nil, filters: LibraryFilters? = nil, title: String? = nil) {
        self.scope = scope
        self.header = header
        self.title = title
        let model = LibraryModel(scope: scope)
        if let filters { model.filters = filters }
        _model = State(initialValue: model)
    }

    var body: some View {
        content
            .modifier(LibraryTitle(title: title, searching: model.searching))
            .modifier(LibrarySearchField(model: model,
                                         prompt: scope.isVault ? "Search the vault" : "Search photos, videos and documents"))
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if !model.selection.active {
                        Button("Select") { model.selection.begin() }
                            .accessibilityIdentifier("library.select")
                    }
                }
                if scope == .all {
                    ToolbarItem(placement: .topBarLeading) {
                        Button {
                            router.open(.more(.upload))
                        } label: {
                            Label("Upload", systemImage: "icloud.and.arrow.up")
                        }
                        .accessibilityIdentifier("library.upload")
                    }
                }
            }
            .safeAreaInset(edge: .bottom) {
                if model.selection.active {
                    LibraryBulkBar(model: model, open: { kind in
                        sheetUids = Array(model.selection.selected)
                        sheet = kind
                    }, delete: { confirmDelete = true })
                }
            }
            .task(id: model.loadKey) {
                if model.loadedKey != model.loadKey { await model.reload() }
            }
            .task(id: model.searchText) {
                try? await Task.sleep(for: .milliseconds(400))
                if !Task.isCancelled { model.commitSearch() }
            }
            .refreshable { await model.reload() }
            .onCloudChange { Task { await model.refreshInPlace() } }
            .sheet(item: $sheet) { kind in
                sheetView(kind)
            }
            // here rather than in the filter bar, so the sheet stays open when the list under it changes
            .sheet(isPresented: $showingFilters) {
                LibraryFilterSheet(filters: filtersBinding, scope: scope, searching: model.searching)
            }
            .confirmationDialog("Move \(Fmt.count(model.selection.count, "item")) to the trash?",
                                isPresented: $confirmDelete, titleVisibility: .visible) {
                Button("Move to trash", role: .destructive) { Task { await model.bulk(.delete) } }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("They leave the library now and can be restored from Trash until it is emptied.")
            }
    }

    // MARK: Content

    /// Nothing to show yet or at all: the filters and the search stay on screen above the state.
    private var listEmpty: Bool { model.searching ? model.hits.isEmpty : model.buckets.isEmpty }

    /// The list on screen is the answer for the current filters and search.
    private var settled: Bool { model.loadedKey == model.loadKey && !model.loading }

    @ViewBuilder
    private var content: some View {
        if model.searching && model.searchUnavailable && settled {
            stateScroll {
                ContentUnavailableView {
                    Label("Search is not available right now", systemImage: "magnifyingglass")
                } description: {
                    Text("The search model is not installed or still loading. Settings shows its status.")
                } actions: {
                    Button("Browse instead") { model.clearSearch() }
                        .buttonStyle(.bordered)
                }
            }
        } else if let error = model.error, listEmpty, settled {
            stateScroll {
                ErrorState(error: error) { Task { await model.reload() } }
            }
        } else if listEmpty && settled {
            stateScroll { emptyState }
        } else if listEmpty {
            stateScroll {
                ProgressView(model.searching ? "Searching" : "Loading")
                    .frame(maxWidth: .infinity)
                    .padding(.top, 60)
            }
        } else if model.searching {
            LibrarySearchResults(model: model, vault: scope.isVault, header: AnyView(topBlock)) { hit in
                // a video opens at its matching moment
                router.show(entries: model.hitEntries, start: model.hits.firstIndex { $0.uid == hit.uid } ?? 0,
                            vault: scope.isVault, total: model.hitsTotal) { await model.loadMoreForViewer() }
            }
        } else {
            LibraryTimelineView(model: model, vault: scope.isVault, header: AnyView(topBlock)) { card in
                // the counter counts the whole library (album, vault), at the item's place in the whole timeline
                let library = model
                router.show(library.timelineItems, start: card, vault: scope.isVault, total: library.total,
                            loadMore: { await library.loadMoreForViewer() },
                            positionOf: { library.timelinePosition(of: $0) })
            }
        }
    }

    /// The header, the count, the search modes and the filters, above the items.
    private var topBlock: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let header { header }
            HStack(spacing: 8) {
                Text(countLine)
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("library.count")
                if model.loading && !listEmpty {
                    ProgressView().controlSize(.small)
                }
            }
            if model.searching {
                Picker("Search by", selection: modeBinding) {
                    ForEach([API.Mode.auto, .semantic, .text], id: \.self) { mode in
                        Text(mode.libraryLabel).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .accessibilityIdentifier("library.mode")
                if model.indexing && !model.searchUnavailable {
                    Text("Still indexing: \(model.indexed.formatted()) items can be found so far.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            LibraryFilterBar(filters: filtersBinding, scope: scope, searching: model.searching,
                             presentFilters: { showingFilters = true },
                             showOnMap: scope == .all && !model.searching
                                 ? { router.open(.mapAt(MapStart(filters: model.filters.normalized(for: .all)))) }
                                 : nil)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var countLine: String {
        if listEmpty && !settled { return model.searching ? "Searching" : "Loading" }
        if model.searching {
            return model.searchUnavailable ? "" : "\(Fmt.count(model.hitsTotal, "result")) for \"\(model.query)\""
        }
        return Fmt.count(model.total, "item")
    }

    /// An empty, failed or loading list, with the filters still there to change.
    private func stateScroll<StateContent: View>(@ViewBuilder _ state: () -> StateContent) -> some View {
        ScrollView {
            topBlock
            state()
                .frame(minHeight: 320)
        }
    }

    @ViewBuilder
    private var emptyState: some View {
        if model.searching {
            ContentUnavailableView {
                Label("Nothing matches. Try other words.", systemImage: "photo")
            } description: {
                Text(model.indexing
                     ? "Some items are still being indexed, so try again in a few minutes."
                     : "Describe what is in the picture, like \"beach at sunset\", or type words from a document, like a name or a number.")
            } actions: {
                Button("Clear search") { model.clearSearch() }
                    .buttonStyle(.bordered)
            }
        } else if model.filters.narrows {
            ContentUnavailableView {
                Label("Nothing matches these filters", systemImage: "photo")
            } actions: {
                Button("Clear filters") { model.filters = LibraryFilters() }
                    .buttonStyle(.bordered)
            }
        } else if scope.isVault {
            ContentUnavailableView {
                Label("The vault is empty", systemImage: "lock.shield")
            } description: {
                Text("Items marked private, and photos found to be explicit, are kept here.")
            }
        } else if scope.albumId != nil {
            ContentUnavailableView {
                Label("Nothing here yet", systemImage: "rectangle.stack")
            } description: {
                Text("Items you add to this album appear here. Select them in the library and choose Add to album.")
            }
        } else {
            ContentUnavailableView {
                Label("Nothing here yet", systemImage: "photo.on.rectangle")
            } description: {
                Text("Photos, videos, PDFs and other documents you upload appear here, sorted and searchable.")
            } actions: {
                Button("Upload") { router.open(.more(.upload)) }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    // MARK: Sheets

    @ViewBuilder
    private func sheetView(_ kind: LibraryBulkSheet) -> some View {
        let uids = sheetUids
        switch kind {
        case .album:
            AlbumsAddSheet(uids: uids) { model.selection.end() }
        case .share:
            SharingCreateSheet(target: .items(uids: uids))
        case .people:
            PeopleSelectionSheet(uids: uids, vault: scope.isVault) { model.selection.end() }
        case .tag, .untag:
            LibraryTagSheet(removing: kind == .untag) { name in
                Task { await model.bulk(kind == .tag ? .tag : .untag, value: name) }
            }
        }
    }

    // MARK: Bindings

    /// A filter change clears the selection, as on the website.
    private var filtersBinding: Binding<LibraryFilters> {
        Binding(get: { model.filters }, set: { value in
            if value != model.filters {
                model.selection.clear()
                model.filters = value
            }
        })
    }

    private var modeBinding: Binding<API.Mode> {
        Binding(get: { model.mode }, set: { model.mode = $0 })
    }
}

/// The browser's title when it has one: "Search results" while searching.
struct LibraryTitle: ViewModifier {
    let title: String?
    let searching: Bool

    func body(content: Content) -> some View {
        if let title {
            content.navigationTitle(searching ? "Search results" : title)
        } else {
            content
        }
    }
}

/// The search field in the navigation bar (not on an album, whose items the website's search cannot be limited to).
/// Typing runs the search after a pause, Search runs it at once.
struct LibrarySearchField: ViewModifier {
    @Bindable var model: LibraryModel
    let prompt: String

    func body(content: Content) -> some View {
        if model.canSearch {
            content
                .searchable(text: $model.searchText, placement: .navigationBarDrawer(displayMode: .always),
                            prompt: prompt)
                .onSubmit(of: .search) { model.commitSearch() }
        } else {
            content
        }
    }
}
