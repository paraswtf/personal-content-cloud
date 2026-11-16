import SwiftUI
import Observation

/// One month of the timeline: its cards so far, where its next page starts, and whether more are coming.
struct LibraryMonth {
    var cards: [API.ItemCard] = []
    var cursor: String?
    var complete = false
    var loading = false
    var failed = false
}

/// What a library browser shows and does (the website's Library.tsx with useTimeline and useSearchList): the month
/// skeleton and each month's cards page by page (docs/ARCHITECTURE.md section 23), or the search results, for the
/// filters and the search; the selection and its bulk actions.
@MainActor
@Observable
final class LibraryModel {
    let scope: LibraryScope

    /// Cards per page of a month (the website asks for 500; a phone screen holds fewer).
    static let monthPage = 200
    /// Results per page of a search (the website's 60).
    static let searchPage = 60
    /// The longest search the server takes.
    static let maxQuery = 200

    var filters = LibraryFilters()
    /// The search box's text; it runs after a pause or on Search (`commitSearch`).
    var searchText = ""
    /// The search that runs ("" browses the timeline).
    private(set) var query = ""
    var mode: API.Mode = .auto

    // the timeline
    private(set) var buckets: [API.TimelineBucket] = []
    private(set) var total = 0
    private(set) var months: [String: LibraryMonth] = [:]

    // the search
    private(set) var hits: [API.SearchHit] = []
    private(set) var hitsTotal = 0
    private(set) var nextOffset: Int?
    private(set) var indexing = false
    private(set) var indexed = 0
    private(set) var searchUnavailable = false
    private(set) var loadingMoreHits = false
    private(set) var moreHitsFailed = false

    private(set) var loading = false
    private(set) var loaded = false
    private(set) var error: Error?
    /// What the last finished load was for: coming back to the screen loads again only when it changed.
    private(set) var loadedKey: LibraryLoadKey?

    let selection = SelectionModel<String>()
    @ObservationIgnored private var generation = 0
    /// The album's name, for "removed from Album" (asked for once, when needed).
    @ObservationIgnored private var albumName: String?

    init(scope: LibraryScope) {
        self.scope = scope
    }

    var searching: Bool { !query.isEmpty }

    /// The search box is offered: the website's search cannot be limited to one album.
    var canSearch: Bool { scope.albumId == nil }

    var loadKey: LibraryLoadKey { LibraryLoadKey(filters: filters, query: query, mode: mode) }

    /// Loaded cards in timeline order: what the viewer pages through and Select all picks.
    var timelineItems: [API.ItemCard] {
        buckets.flatMap { months[$0.key]?.cards ?? [] }
    }

    /// The uids Select all picks: every loaded item of the list on screen.
    var loadedUids: [String] { searching ? hits.map(\.uid) : timelineItems.map(\.uid) }

    /// Every month is loaded to its end.
    var timelineComplete: Bool {
        buckets.allSatisfy { months[$0.key]?.complete == true }
    }

    // MARK: Loading

    /// Runs the search box's text now (Search on the keyboard), or after the pause.
    func commitSearch() {
        let next = String(searchText.trimmingCharacters(in: .whitespacesAndNewlines).prefix(Self.maxQuery))
        if next != query {
            query = next
            selection.clear()
        }
    }

    func clearSearch() {
        searchText = ""
        commitSearch()
    }

    /// Loads the list for the filters and the search from the start.
    func reload() async {
        generation += 1
        let gen = generation
        let key = loadKey
        loading = true
        error = nil
        moreHitsFailed = false
        do {
            if searching {
                let f = filters.forSearch(in: scope)
                let q = query, m = mode
                let page = try await call {
                    try await Cloud.api().searchItems(q: q, mode: m, category: f.category, kind: f.kind,
                                                      person: f.person, tag: f.tag, live: f.live, from: f.from,
                                                      to: f.to, limit: Self.searchPage, offset: nil)
                }
                guard gen == generation else { return }
                hits = page.items
                hitsTotal = page.total
                nextOffset = page.nextOffset
                indexing = page.indexing
                indexed = page.indexed
                searchUnavailable = false
            } else {
                let result = try await fetchBuckets()
                guard gen == generation else { return }
                buckets = result.buckets.filter { $0.count > 0 }
                total = result.total
                months = [:]
                // the newest month at once, the rest as they come on screen
                if let first = buckets.first {
                    await loadMonth(first.key, gen: gen)
                }
            }
            guard gen == generation else { return }
            loaded = true
            loadedKey = key
        } catch {
            guard gen == generation else { return }
            if Self.isCancel(error) {
                loading = false
                return
            }
            if searching, let api = error as? APIError, api.status == 503 {
                searchUnavailable = true
                hits = []
                hitsTotal = 0
                nextOffset = nil
            } else {
                self.error = error
            }
            loaded = true
            loadedKey = key
        }
        if gen == generation { loading = false }
    }

    /// After a change on the server: the skeleton and the months already loaded come again, in place, so the list
    /// does not jump back to the top.
    func refreshInPlace() async {
        guard loaded, error == nil else {
            await reload()
            return
        }
        if searching {
            await reload()
            return
        }
        generation += 1
        let gen = generation
        do {
            let result = try await fetchBuckets()
            guard gen == generation else { return }
            let fresh = result.buckets.filter { $0.count > 0 }
            var next: [String: LibraryMonth] = [:]
            for bucket in fresh {
                guard let had = months[bucket.key], !had.cards.isEmpty else { continue }
                var month = LibraryMonth()
                let wanted = had.cards.count
                while month.cards.count < wanted && !month.complete {
                    let page = try await fetchMonthPage(bucket.key, cursor: month.cursor)
                    guard gen == generation else { return }
                    month.cards += page.items
                    month.cursor = page.nextCursor
                    month.complete = page.nextCursor == nil || page.items.isEmpty
                }
                next[bucket.key] = month
            }
            guard gen == generation else { return }
            buckets = fresh
            total = result.total
            months = next
            if let first = fresh.first, next[first.key] == nil {
                await loadMonth(first.key, gen: gen)
            }
        } catch {
            guard gen == generation, !Self.isCancel(error) else { return }
            await reload()
        }
    }

    /// A month's cells came on screen: loads its next page when it has one and nothing is loading.
    func ensureMonth(_ key: String) {
        let month = months[key] ?? LibraryMonth()
        guard !month.loading, !month.complete, !month.failed, loaded, !searching else { return }
        let gen = generation
        Task { await loadMonth(key, gen: gen) }
    }

    /// After a failed page: tries the month again.
    func retryMonth(_ key: String) {
        if var month = months[key] {
            month.failed = false
            months[key] = month
        }
        ensureMonth(key)
    }

    private func loadMonth(_ key: String, gen: Int) async {
        var month = months[key] ?? LibraryMonth()
        guard !month.loading, !month.complete else { return }
        month.loading = true
        month.failed = false
        months[key] = month
        do {
            let page = try await fetchMonthPage(key, cursor: month.cursor)
            guard gen == generation else { return }
            var current = months[key] ?? LibraryMonth()
            let known = Set(current.cards.map(\.uid))
            current.cards += page.items.filter { !known.contains($0.uid) }
            current.cursor = page.nextCursor
            current.complete = page.nextCursor == nil || page.items.isEmpty
            current.loading = false
            months[key] = current
        } catch {
            guard gen == generation else { return }
            var current = months[key] ?? LibraryMonth()
            current.loading = false
            current.failed = !Self.isCancel(error)
            months[key] = current
        }
    }

    /// The next page of results, as the last ones come on screen.
    func loadMoreHits() async {
        guard searching, let offset = nextOffset, !loadingMoreHits, !moreHitsFailed else { return }
        let gen = generation
        loadingMoreHits = true
        let f = filters.forSearch(in: scope)
        let q = query, m = mode
        do {
            let page = try await call {
                try await Cloud.api().searchItems(q: q, mode: m, category: f.category, kind: f.kind, person: f.person,
                                                  tag: f.tag, live: f.live, from: f.from, to: f.to,
                                                  limit: Self.searchPage, offset: offset)
            }
            guard gen == generation else { return }
            let known = Set(hits.map(\.uid))
            hits += page.items.filter { !known.contains($0.uid) }
            hitsTotal = page.total
            nextOffset = page.nextOffset
            indexing = page.indexing
            indexed = page.indexed
        } catch {
            guard gen == generation else { return }
            moreHitsFailed = !Self.isCancel(error)
        }
        if gen == generation { loadingMoreHits = false }
    }

    func retryMoreHits() async {
        moreHitsFailed = false
        await loadMoreHits()
    }

    /// For the viewer paging past the last loaded item: the next page in timeline order.
    func loadMoreForViewer() async -> [ViewerEntry] {
        if searching {
            await loadMoreHits()
            return hits.map(\.viewerEntry)
        }
        if let next = buckets.first(where: { months[$0.key]?.complete != true }) {
            await loadMonth(next.key, gen: generation)
        }
        return timelineItems.map(\.viewerEntry)
    }

    private func fetchBuckets() async throws -> API.TimelineBuckets {
        let f = filters.normalized(for: scope)
        let album = scope.albumId
        return try await call {
            try await Cloud.api().buckets(category: f.category, kind: f.kind, person: f.person, tag: f.tag,
                                          live: f.live, from: f.from, to: f.to, favorite: f.favorite,
                                          misclicks: f.misclicks, forwards: f.forwards, received: f.received,
                                          dups: f.dups, sort: f.sort, album: album)
        }
    }

    private func fetchMonthPage(_ key: String, cursor: String?) async throws -> API.ItemPage {
        let f = filters.normalized(for: scope)
        let album = scope.albumId
        return try await call {
            try await Cloud.api().bucket(key: key, cursor: cursor, limit: Self.monthPage, category: f.category,
                                         kind: f.kind, person: f.person, tag: f.tag, live: f.live, from: f.from,
                                         to: f.to, favorite: f.favorite, misclicks: f.misclicks,
                                         forwards: f.forwards, received: f.received, dups: f.dups, sort: f.sort,
                                         album: album)
        }
    }

    /// Runs a request in the scope's mode: the vault's requests carry vault=1.
    func call<T>(_ body: () async throws -> T) async throws -> T {
        if scope.isVault {
            return try await APIContext.$vault.withValue(true) { try await body() }
        }
        return try await body()
    }

    static func isCancel(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        if let u = error as? URLError, u.code == .cancelled { return true }
        return false
    }

    // MARK: The selection's actions

    /// The flags of the selected items that are loaded (they decide some buttons).
    private var selectedFlags: [LibraryFlags] {
        let picked = selection.selected
        if searching {
            return hits.filter { picked.contains($0.uid) }
                .map { LibraryFlags(favorite: $0.favorite, misclick: $0.misclick, forward: $0.forward) }
        }
        return timelineItems.filter { picked.contains($0.uid) }
            .map { LibraryFlags(favorite: $0.favorite, misclick: $0.misclick, forward: $0.forward) }
    }

    /// Every selected item is a favourite: the button removes them from the favourites.
    var allSelectedFavorite: Bool {
        let flags = selectedFlags
        return !flags.isEmpty && flags.allSatisfy { $0.favorite }
    }

    var anySelectedMisclick: Bool { selectedFlags.contains { $0.misclick } }

    var anySelectedForward: Bool { selectedFlags.contains { $0.forward } }

    /// One bulk action on the selection (POST /api/items/bulk), with the website's message, undoable.
    func bulk(_ action: API.Action, value: String? = nil) async {
        let uids = Array(selection.selected)
        guard !uids.isEmpty else { return }
        let what = await describe(action, value: value)
        let label = "\(Fmt.count(uids.count, "item")) \(what)."
        do {
            try await UndoCenter.shared.perform(label) {
                try await call {
                    try await Cloud.api().bulk(body: API.BulkIn(uids: uids, action: action, value: value))
                }
            }
            selection.end()
            CloudEvents.post(items: uids)
        } catch {
            Toaster.shared.error(error, doing: action == .unalbum ? "change the album" : "update the selection")
        }
    }

    /// The website's words for what a bulk action did ("added to favourites").
    private func describe(_ action: API.Action, value: String?) async -> String {
        switch action {
        case .delete: return "moved to the trash"
        case .keep: return "kept"
        case .forward: return "set apart as forwards"
        case .unforward: return "kept in the library"
        case .favorite: return "added to favourites"
        case .unfavorite: return "removed from favourites"
        case .category: return "moved to a new category"
        case .nsfw: return value == "1" ? "moved to the vault" : "moved out of the vault"
        case .tag: return "tagged \(value ?? "")"
        case .untag: return "no longer tagged \(value ?? "")"
        case .unalbum:
            let title = await albumTitle()
            return "removed from \(title)"
        default: return "updated"
        }
    }

    private func albumTitle() async -> String {
        if let albumName { return albumName }
        guard let id = scope.albumId else { return "the album" }
        if let album = try? await Cloud.api().getAlbum(albumId: id) {
            albumName = album.name
            return album.name
        }
        return "the album"
    }
}

/// The flags of one selected item.
struct LibraryFlags {
    let favorite: Bool
    let misclick: Bool
    let forward: Bool
}

/// What a load is for: when it changes, the list loads again.
struct LibraryLoadKey: Hashable {
    let filters: LibraryFilters
    let query: String
    let mode: API.Mode
}
