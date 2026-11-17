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
}

/// One item the viewer can page to. The thumb shows at once while the item itself loads.
struct ViewerEntry: Identifiable, Hashable {
    let uid: String
    var thumb: String?
    var kind: String?

    var id: String { uid }
}

/// A request to show the viewer: the items to page through, where to start, and how to load more when the
/// owner pages past the last one (the list behind it continues). `vault` opens vault items (the vault screen):
/// the viewer then asks for everything with `vault=1`.
struct ViewerRequest: Identifiable {
    let id = UUID()
    let entries: [ViewerEntry]
    let start: Int
    var loadMore: (() async -> [ViewerEntry])?
    var vault = false

    init(entries: [ViewerEntry], start: Int, loadMore: (() async -> [ViewerEntry])? = nil, vault: Bool = false) {
        self.entries = entries
        self.start = min(max(0, start), max(0, entries.count - 1))
        self.loadMore = loadMore
        self.vault = vault
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

    /// Opens the viewer on `items`, starting at `start` (`vault: true` from the vault screen).
    func show<Item: MediaItem>(_ items: [Item], start: Item, vault: Bool = false,
                               loadMore: (() async -> [ViewerEntry])? = nil) {
        let entries = items.map(\.viewerEntry)
        let index = items.firstIndex(where: { $0.uid == start.uid }) ?? 0
        viewer = ViewerRequest(entries: entries, start: index, loadMore: loadMore, vault: vault)
    }

    /// Back to the root of a tab (tapping the selected tab again).
    func popToRoot(_ tab: AppTab) {
        paths[tab] = NavigationPath()
    }
}
