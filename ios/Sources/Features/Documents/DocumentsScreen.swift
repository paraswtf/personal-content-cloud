import SwiftUI

/// One row of the documents list: the item and its parsed document.
struct DocumentsEntry: Identifiable {
    let item: API.ItemCard
    let document: API.DocumentOut

    var id: String { item.uid }

    init(_ listed: API.DocumentListItem) {
        item = listed.item
        document = listed.document
    }
}

/// What the list shows: one type, or one group of types, and the words searched for.
struct DocumentsQuery: Equatable {
    var type: String?
    var group: String?
    var text: String?

    var isFiltered: Bool { type != nil || group != nil || text != nil }
}

/// Documents (the website's DocumentsPage): parsed documents as cards, filtered by type or group and searched by
/// their text and fields. A card opens the viewer.
struct DocumentsScreen: View {
    @Environment(AppRouter.self) private var router
    @State private var query = DocumentsQuery()
    @State private var searchText = ""
    @State private var types: [API.DocumentTypeOut]?
    @State private var pager = DocumentsScreen.makePager(DocumentsQuery())

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                Text(countLine)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .accessibilityIdentifier("documents.count")
                if let types, !types.isEmpty {
                    DocumentsTypeChips(types: types, query: query) { next in apply(next) }
                }
                content
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 24)
        }
        .background(Color(uiColor: .systemGroupedBackground))
        .navigationTitle("Documents")
        .searchable(text: $searchText, placement: .navigationBarDrawer(displayMode: .always),
                    prompt: "Search names, numbers and words in your documents")
        .onSubmit(of: .search) { commitSearch() }
        .task(id: searchText) {
            // typing settles for a moment before the list follows; Search commits at once
            try? await Task.sleep(for: .milliseconds(450))
            if !Task.isCancelled { commitSearch() }
        }
        .task {
            if types == nil { await loadTypes() }
            if !pager.loaded { await pager.refresh() }
        }
        .refreshable {
            await loadTypes()
            await pager.refresh()
        }
        .onCloudChange {
            Task {
                await loadTypes()
                await pager.refresh()
            }
        }
    }

    // MARK: content

    private var countLine: String {
        if !pager.loaded && pager.items.isEmpty { return "Loading" }
        let count = Fmt.count(pager.total ?? pager.items.count, "document")
        if let text = query.text { return "\(count) matching \"\(text)\"" }
        return count
    }

    @ViewBuilder
    private var content: some View {
        if pager.items.isEmpty {
            if let error = pager.error {
                ErrorState(error: error) { Task { await pager.retry() } }
                    .frame(minHeight: 280)
            } else if pager.loaded && !pager.loading {
                Group {
                    if query.isFiltered {
                        EmptyState(title: "No documents match", systemImage: "doc.text.magnifyingglass",
                                   text: "Try other words, or a different type.")
                    } else {
                        EmptyState(title: "No documents yet", systemImage: "doc.text",
                                   text: "Upload PDFs, office files or photos of IDs, bills and tickets. Their text is read and the details are filled in for you.")
                    }
                }
                .frame(minHeight: 280)
            } else {
                ProgressView("Loading documents")
                    .frame(maxWidth: .infinity, minHeight: 200)
            }
        } else {
            ForEach(pager.items) { entry in
                DocumentsCard(item: entry.item, document: entry.document) { open(entry) }
                    .onAppear { pager.loadMoreIfNeeded(entry) }
            }
            if pager.loading {
                ProgressView().frame(maxWidth: .infinity).padding(.vertical, 8)
            }
        }
    }

    // MARK: actions

    private static func makePager(_ query: DocumentsQuery) -> Pager<DocumentsEntry> {
        Pager<DocumentsEntry> { cursor in
            let page = try await Cloud.api().listDocuments(type: query.type, group: query.type == nil ? query.group : nil,
                                                           q: query.text, cursor: cursor, limit: 40)
            return .init(items: page.items.map(DocumentsEntry.init), next: page.nextCursor, total: page.total)
        }
    }

    private func apply(_ next: DocumentsQuery) {
        guard next != query else { return }
        query = next
        pager = Self.makePager(next)
        Task { await pager.refresh() }
    }

    private func commitSearch() {
        let words = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        var next = query
        next.text = words.isEmpty ? nil : words
        apply(next)
    }

    private func loadTypes() async {
        if let answer = try? await Cloud.api().documentTypes() { types = answer }
    }

    /// Opens the viewer in document mode (the website's mode="document"): the info panel opens at once with the
    /// document's section first, and the owner pages through the list.
    private func open(_ entry: DocumentsEntry) {
        router.show(pager.items.map(\.item), start: entry.item, mode: .document, total: pager.total) {
            await pager.loadMore()
            return pager.items.map { $0.item.viewerEntry }
        }
    }
}

/// The type filter (the website's TypeChips): All, then a row per group with the group and its types, with counts.
struct DocumentsTypeChips: View {
    let types: [API.DocumentTypeOut]
    let query: DocumentsQuery
    let onChange: (DocumentsQuery) -> Void

    /// One group of types with documents.
    struct TypeGroup: Identifiable {
        let key: String
        let types: [API.DocumentTypeOut]

        var id: String { key }
    }

    private var groups: [TypeGroup] {
        var by: [String: [API.DocumentTypeOut]] = [:]
        for t in types where t.count > 0 { by[t.group, default: []].append(t) }
        let order = DocumentsMeta.groupOrder + by.keys.filter { !DocumentsMeta.groupOrder.contains($0) }.sorted()
        return order.compactMap { key in by[key].map { TypeGroup(key: key, types: $0) } }
    }

    var body: some View {
        let total = types.reduce(0) { $0 + max(0, $1.count) }
        VStack(alignment: .leading, spacing: 8) {
            chip("All", count: total, systemImage: nil, on: query.type == nil && query.group == nil) {
                var next = query
                next.type = nil
                next.group = nil
                onChange(next)
            }
            .accessibilityIdentifier("documents.all")
            ForEach(groups) { group in
                let meta = DocumentsMeta.group(group.key)
                let groupOn = query.group == group.key && query.type == nil
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        chip(meta.label, count: group.types.reduce(0) { $0 + $1.count }, systemImage: meta.systemImage,
                             on: groupOn) {
                            var next = query
                            next.type = nil
                            next.group = groupOn ? nil : group.key
                            onChange(next)
                        }
                        .accessibilityIdentifier("documents.group.\(group.key)")
                        ForEach(group.types, id: \.type) { t in
                            let typeOn = query.type == t.type
                            chip(t.label.isEmpty ? DocumentsMeta.typeLabel(t.type) : t.label, count: t.count,
                                 systemImage: nil, on: typeOn) {
                                var next = query
                                next.type = typeOn ? nil : t.type
                                next.group = nil
                                onChange(next)
                            }
                            .accessibilityIdentifier("documents.type.\(t.type)")
                        }
                    }
                }
            }
        }
    }

    private func chip(_ text: String, count: Int, systemImage: String?, on: Bool,
                      action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 5) {
                if let systemImage { Image(systemName: systemImage).imageScale(.small) }
                Text(text).fontWeight(.semibold)
                Text(count.formatted()).monospacedDigit().opacity(0.7)
            }
            .font(.subheadline)
            .padding(.horizontal, 13)
            .frame(height: 36)
            .foregroundStyle(on ? Color(uiColor: .systemBackground) : Color.primary)
            .background(on ? Color.primary : Color(uiColor: .secondarySystemGroupedBackground), in: Capsule())
            .overlay(Capsule().strokeBorder(Color(uiColor: .separator), lineWidth: on ? 0 : 0.5))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(text), \(count)")
        .accessibilityAddTraits(on ? .isSelected : [])
    }
}
