import Foundation

/// The kind of an item (`items.kind` on the server).
enum MediaKind: String {
    case image, video, pdf, document, other

    init(_ raw: String?) { self = MediaKind(rawValue: raw ?? "") ?? .other }

    var label: String {
        switch self {
        case .image: return "Photo"
        case .video: return "Video"
        case .pdf: return "PDF"
        case .document: return "Document"
        case .other: return "File"
        }
    }
}

/// What the media grid and the viewer need of an item, whatever API shape it comes in (the core's own
/// `CoreItemCard`, or the generated `API.ItemCard` and `API.SearchHit` through ios/Sources/Core/APIAdapters.swift).
protocol MediaItem: Identifiable where ID == String {
    var uid: String { get }
    var mediaKind: MediaKind { get }
    /// The thumb's address as the API gives it ("/files/thumb/<uid>?v=3").
    var thumbRef: String { get }
    var takenAtText: String? { get }
    var createdAtText: String { get }
    var durationSeconds: Double? { get }
    var isLive: Bool { get }
    var isFavorite: Bool { get }
}

extension MediaItem {
    var id: String { uid }

    /// When it was taken, else when it came to the server: the date the grid groups by.
    var sortDateText: String { takenAtText ?? createdAtText }

    /// "2026-11" from "2026-11-16T19:18:17.000Z" (the month the grid puts it under), cheap enough for every cell.
    var monthKey: String {
        let text = sortDateText
        guard text.count >= 7 else { return "" }
        return String(text.prefix(7))
    }

    /// A viewer entry for this item, so the viewer can show the thumb while the item loads.
    var viewerEntry: ViewerEntry { ViewerEntry(uid: uid, thumb: thumbRef, kind: mediaKind.rawValue) }

    /// What VoiceOver reads for a cell: "Video, 1:15, 3 Nov 2026".
    var accessibilitySummary: String {
        var parts = [mediaKind.label]
        if isLive { parts[0] = "Live Photo" }
        if mediaKind == .video, let d = durationSeconds { parts.append(Fmt.duration(d)) }
        let date = Fmt.date(sortDateText)
        if !date.isEmpty { parts.append(date) }
        if isFavorite { parts.append("favourite") }
        return parts.joined(separator: ", ")
    }
}

// MARK: - The core's own small views of the API's JSON (decoded with CoreHTTP.decoder: snake_case to camelCase)

struct CoreItemURLs: Decodable, Hashable {
    let thumb: String
    let preview: String
    let previewSmall: String
    let original: String?
    let download: String?
    let video: String?
    let live: String?
    let pages: [String]?
}

/// One row of `GET /api/items` (`ItemCard`).
struct CoreItemCard: Decodable, Hashable, MediaItem {
    let uid: String
    let kind: String
    let version: Int
    let filename: String
    let width: Int?
    let height: Int?
    let durationS: Double?
    let takenAt: String?
    let createdAt: String
    let favorite: Bool?
    let live: Bool?
    let urls: CoreItemURLs

    var mediaKind: MediaKind { MediaKind(kind) }
    var thumbRef: String { urls.thumb }
    var takenAtText: String? { takenAt }
    var createdAtText: String { createdAt }
    var durationSeconds: Double? { durationS }
    var isLive: Bool { live ?? false }
    var isFavorite: Bool { favorite ?? false }
}

/// `ItemPage`.
struct CoreItemPage: Decodable {
    let items: [CoreItemCard]
    let nextCursor: String?
    let total: Int
}

/// `PersonRef`: who a box or a tag is.
struct CorePersonRef: Decodable, Hashable {
    let id: Int
    let name: String?
    let coverUrl: String?
    let kind: String?
}

/// `GuessOut`: who a box might be.
struct CoreGuess: Decodable, Hashable {
    let person: CorePersonRef
    let score: Double
}

struct CoreFace: Decodable, Hashable {
    let id: Int
    let x: Double
    let y: Double
    let w: Double
    let h: Double
    let frameTimeS: Double?
    let person: CorePersonRef?
    let guess: CoreGuess?
}

struct CoreBody: Decodable, Hashable {
    let id: Int
    let x: Double
    let y: Double
    let w: Double
    let h: Double
    let faceId: Int?
    let frameTimeS: Double?
    let person: CorePersonRef?
    let guess: CoreGuess?
}

struct CoreAnimal: Decodable, Hashable {
    let id: Int
    let x: Double
    let y: Double
    let w: Double
    let h: Double
    let species: String
    let frameTimeS: Double?
    let person: CorePersonRef?
}

struct CoreLiveVideo: Decodable, Hashable {
    let uid: String?
    let url: String
    let durationS: Double?
}

/// The part of `GET /api/items/{uid}` (`ItemDetail`) the viewer core draws: the picture, the video, the Live
/// Photo clip and the boxes. The viewer area decodes the whole `API.ItemDetail` from the same bytes.
struct CoreItemDetail: Decodable, Hashable {
    let uid: String
    let kind: String
    let version: Int
    let filename: String
    let width: Int?
    let height: Int?
    let durationS: Double?
    let takenAt: String?
    let createdAt: String
    let favorite: Bool?
    let live: Bool?
    let urls: CoreItemURLs
    let faces: [CoreFace]?
    let bodies: [CoreBody]?
    let animals: [CoreAnimal]?
    let liveVideo: CoreLiveVideo?
    let place: String?
    let deletedAt: String?

    var mediaKind: MediaKind { MediaKind(kind) }

    /// The boxes drawn on the still picture (docs/ARCHITECTURE.md sections 10 and 14): every face, the person
    /// boxes whose face was not found, and the pets. Video frames' boxes are not drawn on the still.
    var stillBoxes: [StageBox] {
        var out: [StageBox] = []
        for f in faces ?? [] where f.frameTimeS == nil {
            out.append(StageBox(kind: .face, boxId: f.id, x: f.x, y: f.y, w: f.w, h: f.h,
                                name: Self.named(f.person?.name),
                                guessName: f.person == nil ? Self.named(f.guess?.person.name) : nil,
                                personId: f.person?.id))
        }
        for b in bodies ?? [] where b.frameTimeS == nil && b.faceId == nil {
            out.append(StageBox(kind: .body, boxId: b.id, x: b.x, y: b.y, w: b.w, h: b.h,
                                name: Self.named(b.person?.name),
                                guessName: b.person == nil ? Self.named(b.guess?.person.name) : nil,
                                personId: b.person?.id))
        }
        for a in animals ?? [] where a.frameTimeS == nil {
            out.append(StageBox(kind: .animal, boxId: a.id, x: a.x, y: a.y, w: a.w, h: a.h,
                                name: Self.named(a.person?.name) ?? Fmt.speciesLabel(a.species),
                                guessName: nil, personId: a.person?.id))
        }
        return out
    }

    /// A name to show, or nil: unnamed people are never named on screen.
    private static func named(_ name: String?) -> String? {
        Fmt.isNamed(name) ? name?.trimmingCharacters(in: .whitespacesAndNewlines) : nil
    }
}

/// `PersonOut`, as the person picker lists it.
struct CorePerson: Decodable, Hashable {
    let id: Int
    let name: String?
    let coverUrl: String?
    let kind: String?
    let species: String?
    let itemCount: Int?
}

/// `PersonPage`.
struct CorePersonPage: Decodable {
    let items: [CorePerson]
    let nextCursor: String?
    let total: Int
}

/// `DeviceCreated`: a new device token, shown once.
struct CoreDeviceCreated: Decodable {
    let id: Int
    let name: String
    let token: String
}

/// `AuthState`, as the sign-in checks it.
struct CoreAuthState: Decodable {
    let setupRequired: Bool
    let authenticated: Bool
    let username: String?
    let vaultUntil: String?
    let siteTitle: String
}
