import Foundation

// The generated API types (ios/Sources/API) in the core's shapes, so screens hand them straight to the media grid,
// the viewer and the person picker.

extension API.ItemCard: MediaItem {
    var mediaKind: MediaKind { MediaKind(kind.rawValue) }
    var thumbRef: String { urls.thumb }
    var takenAtText: String? { takenAt }
    var createdAtText: String { createdAt }
    var durationSeconds: Double? { durationS }
    var isLive: Bool { live }
    var isFavorite: Bool { favorite }
}

extension API.SearchHit: MediaItem {
    var mediaKind: MediaKind { MediaKind(kind.rawValue) }
    var thumbRef: String { urls.thumb }
    var takenAtText: String? { takenAt }
    var createdAtText: String { createdAt }
    var durationSeconds: Double? { durationS }
    var isLive: Bool { live }
    var isFavorite: Bool { favorite }
}

extension API.ItemPage {
    /// This page for a `Pager`.
    var pagerPage: Pager<API.ItemCard>.Page { .init(items: items, next: nextCursor, total: total) }
}

extension PersonOption {
    /// A named person from the API (nil for an unnamed one: the app never offers unnamed people).
    init?(_ person: API.PersonOut, detail: String? = nil, score: Double? = nil) {
        self.init(id: person.id, name: person.name, coverRef: person.coverUrl, kind: person.kind.rawValue,
                  species: person.species, detail: detail ?? Fmt.count(person.itemCount, "item"), score: score)
    }
}
