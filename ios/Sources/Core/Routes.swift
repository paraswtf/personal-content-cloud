import SwiftUI
import Observation

/// The app's tabs. Each has its own navigation stack, so going to another tab and back keeps your place.
enum AppTab: String, Hashable, CaseIterable {
    case library, people, albums, review, more
}

/// The screens listed under More.
enum MoreDestination: String, Hashable, CaseIterable, Identifiable {
    case map, documents, vault, upload, trash, settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .map: return "Map"
        case .documents: return "Documents"
        case .vault: return "Vault"
        case .upload: return "Upload and sync"
        case .trash: return "Trash"
        case .settings: return "Settings"
        }
    }

    var systemImage: String {
        switch self {
        case .map: return "map"
        case .documents: return "doc.text"
        case .vault: return "lock.shield"
        case .upload: return "icloud.and.arrow.up"
        case .trash: return "trash"
        case .settings: return "gearshape"
        }
    }

    /// More's headings, as the website's More menu groups its entries (Browse, Tidy up, Account).
    struct Heading {
        let title: String
        let destinations: [MoreDestination]
    }

    static let groups: [Heading] = [
        Heading(title: "Browse", destinations: [.map, .documents, .vault, .upload]),
        Heading(title: "Tidy up", destinations: [.trash]),
        Heading(title: "Account", destinations: [.settings]),
    ]
}

/// A place in the app any screen can go to: `NavigationLink(value: Route.person(id: 3))`, or
/// `router.open(.album(id: 2))` from code. Items open in the viewer over everything.
enum Route: Hashable {
    case person(id: Int)
    case album(id: Int)
    case item(uid: String)
    case more(MoreDestination)
    /// The map with the library's filters or at one item's spot ("Show on map").
    case mapAt(MapStart)
    /// The library with filters set (a tag tapped in the viewer shows everything with that tag).
    case libraryFiltered(LibraryFilters)
    /// The vault with filters set (a tag of a private item tapped in the viewer: the website's /vault?tag=).
    case vaultFiltered(LibraryFilters)
}

/// One item the viewer can page to. The thumb shows at once while the item itself loads. `startAt` opens a video
/// paused at that second (a search hit's matching moment, the frame of a face or a guess).
struct ViewerEntry: Identifiable, Hashable {
    let uid: String
    var thumb: String?
    var kind: String?
    var startAt: Double? = nil

    var id: String { uid }
}

/// How the viewer opens: as it is, or in document mode (the Documents page: the info panel opens at once with
/// the document's section first).
enum ViewerMode: Hashable {
    case normal
    case document
}

/// A request to show the viewer: the items to page through, where to start, and how to load more when the
/// owner pages past the last one (the list behind it continues). `vault` opens vault items (the vault screen):
/// the viewer then asks for everything with `vault=1`. `total` is the whole list's size when the screen knows it
/// (the counter then says "3 of 120", the website's position), and `positionOf` an item's place in that whole list
/// (from 0) when the entries are only parts of it (the library's months loaded so far); nil keeps the entries' order.
struct ViewerRequest: Identifiable {
    let id = UUID()
    let entries: [ViewerEntry]
    let start: Int
    var loadMore: (() async -> [ViewerEntry])?
    var vault = false
    var mode: ViewerMode = .normal
    var total: Int?
    var positionOf: ((String) -> Int?)?

    init(entries: [ViewerEntry], start: Int, loadMore: (() async -> [ViewerEntry])? = nil, vault: Bool = false,
         mode: ViewerMode = .normal, total: Int? = nil, positionOf: ((String) -> Int?)? = nil) {
        self.entries = entries
        self.start = min(max(0, start), max(0, entries.count - 1))
        self.loadMore = loadMore
        self.vault = vault
        self.mode = mode
        self.total = total
        self.positionOf = positionOf
    }
}

/// Where the app is: the selected tab, each tab's navigation path, and the viewer when it is open.
@MainActor
@Observable
final class AppRouter {
    var tab: AppTab = .library
    var viewer: ViewerRequest?
    private var paths: [AppTab: NavigationPath] = [:]

    func path(_ tab: AppTab) -> Binding<NavigationPath> {
        Binding(get: { self.paths[tab] ?? NavigationPath() }, set: { self.paths[tab] = $0 })
    }

    /// Goes to a route: items open in the viewer, everything else is pushed on the current tab's stack.
    func open(_ route: Route) {
        switch route {
        case .item(let uid):
            viewer = ViewerRequest(entries: [ViewerEntry(uid: uid)], start: 0)
        default:
            var p = paths[tab] ?? NavigationPath()
            p.append(route)
            paths[tab] = p
        }
    }

    /// Opens the viewer on `items`, starting at `start` (`vault: true` from the vault screen, `mode: .document` from
    /// the Documents page). `total` and `positionOf`: ViewerRequest's. (`positionOf` comes after `loadMore`, so a
    /// trailing closure is still `loadMore`.)
    func show<Item: MediaItem>(_ items: [Item], start: Item, vault: Bool = false, mode: ViewerMode = .normal,
                               total: Int? = nil, loadMore: (() async -> [ViewerEntry])? = nil,
                               positionOf: ((String) -> Int?)? = nil) {
        let entries = items.map(\.viewerEntry)
        let index = items.firstIndex(where: { $0.uid == start.uid }) ?? 0
        viewer = ViewerRequest(entries: entries, start: index, loadMore: loadMore, vault: vault, mode: mode,
                               total: total, positionOf: positionOf)
    }

    /// Opens the viewer on entries the caller built (with `startAt` for a video's moment), starting at index `start`.
    func show(entries: [ViewerEntry], start: Int, vault: Bool = false, mode: ViewerMode = .normal,
              total: Int? = nil, loadMore: (() async -> [ViewerEntry])? = nil) {
        guard !entries.isEmpty else { return }
        viewer = ViewerRequest(entries: entries, start: start, loadMore: loadMore, vault: vault, mode: mode,
                               total: total)
    }

    /// Back to the root of a tab (tapping the selected tab again).
    func popToRoot(_ tab: AppTab) {
        paths[tab] = NavigationPath()
    }
}
