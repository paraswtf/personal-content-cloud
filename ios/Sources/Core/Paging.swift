import Foundation
import Observation

/// A list loaded a page at a time with the server's cursors (`next_cursor`). Screens give it a function that
/// loads one page for a cursor (nil for the first page) and call `loadMoreIfNeeded` as rows appear.
@MainActor
@Observable
final class Pager<Item: Identifiable> {
    struct Page {
        var items: [Item]
        var next: String?
        var total: Int?

        init(items: [Item], next: String?, total: Int? = nil) {
            self.items = items
            self.next = next
            self.total = total
        }
    }

    private(set) var items: [Item] = []
    private(set) var total: Int?
    private(set) var loading = false
    private(set) var error: Error?
    /// True once the last page has come (or the first load answered with no cursor).
    private(set) var finished = false
    /// True after the first page has been loaded once.
    private(set) var loaded = false

    @ObservationIgnored private var next: String?
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private let load: (String?) async throws -> Page

    init(load: @escaping (String?) async throws -> Page) {
        self.load = load
    }

    /// Nothing there: loaded, no rows, no error.
    var isEmpty: Bool { loaded && items.isEmpty && error == nil }

    /// Loads the first page again (pull to refresh, a change on the server).
    func refresh() async {
        generation += 1
        let gen = generation
        loading = true
        error = nil
        do {
            let page = try await load(nil)
            guard gen == generation else { return }
            items = Self.unique(page.items)
            next = page.next
            total = page.total
            finished = page.next == nil
            loaded = true
        } catch {
            guard gen == generation else { return }
            if !(error is CancellationError) { self.error = error }
        }
        if gen == generation { loading = false }
    }

    /// Loads the next page when there is one and nothing is loading.
    func loadMore() async {
        guard loaded, !loading, !finished, error == nil, let cursor = next else { return }
        let gen = generation
        loading = true
        do {
            let page = try await load(cursor)
            guard gen == generation else { return }
            items = Self.unique(items + page.items)
            next = page.next
            if let t = page.total { total = t }
            finished = page.next == nil
        } catch {
            guard gen == generation else { return }
            if !(error is CancellationError) { self.error = error }
        }
        if gen == generation { loading = false }
    }

    /// Call from a row's onAppear: loads the next page when the row is among the last ones.
    func loadMoreIfNeeded(_ item: Item) {
        guard !finished, !loading else { return }
        if items.suffix(24).contains(where: { $0.id == item.id }) {
            Task { await loadMore() }
        }
    }

    /// After an error: tries the failed load again.
    func retry() async {
        error = nil
        if loaded { await loadMore() } else { await refresh() }
    }

    /// Removes rows locally (after a delete), keeping the rest in place.
    func remove(where shouldRemove: (Item) -> Bool) {
        let before = items.count
        items.removeAll(where: shouldRemove)
        if let t = total { total = max(0, t - (before - items.count)) }
    }

    /// Replaces one row in place (after an edit).
    func replace(_ item: Item) {
        if let i = items.firstIndex(where: { $0.id == item.id }) { items[i] = item }
    }

    private static func unique(_ list: [Item]) -> [Item] {
        var seen = Set<Item.ID>()
        return list.filter { seen.insert($0.id).inserted }
    }
}
