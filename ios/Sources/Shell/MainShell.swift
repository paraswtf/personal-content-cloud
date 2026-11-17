import SwiftUI

/// The signed-in app: five tabs (Library, People, Albums, Review, More), each with its own navigation stack, and
/// the viewer over everything.
struct MainShell: View {
    @State private var router = AppRouter()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        TabView(selection: tabSelection) {
            TabRoot(tab: .library) { LibraryScreen() }
                .tabItem { Label("Library", systemImage: "photo.on.rectangle") }
                .tag(AppTab.library)
            TabRoot(tab: .people) { PeopleScreen() }
                .tabItem { Label("People", systemImage: "person.2") }
                .tag(AppTab.people)
            TabRoot(tab: .albums) { AlbumsScreen() }
                .tabItem { Label("Albums", systemImage: "rectangle.stack") }
                .tag(AppTab.albums)
            TabRoot(tab: .review) { ReviewScreen() }
                .tabItem { Label("Review", systemImage: "checkmark.bubble") }
                .tag(AppTab.review)
            TabRoot(tab: .more) { MoreMenu() }
                .tabItem { Label("More", systemImage: "ellipsis.circle") }
                .tag(AppTab.more)
        }
        .environment(router)
        .fullScreenCover(item: $router.viewer) { request in
            ViewerScreen(request: request)
                .environment(router)
                .toastHost()
        }
        .task { startAutomaticSync() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { startAutomaticSync() }
        }
    }

    /// The phone's photos sync while the app is open (ios/README.md), once Photos access was given on the upload and
    /// sync screens: this only checks the access, it never asks for it.
    private func startAutomaticSync() {
        guard AppSettings.autoSync, PhotoLibrary.hasAccess else { return }
        BackgroundSync.observeLibraryIfAllowed()
        SyncEngine.shared.start(manual: false)
    }

    /// Tapping the selected tab again goes back to its first screen.
    private var tabSelection: Binding<AppTab> {
        Binding(get: { router.tab },
                set: { tab in
                    if tab == router.tab { router.popToRoot(tab) }
                    router.tab = tab
                })
    }
}

/// One tab's navigation stack, which knows every Route.
struct TabRoot<Content: View>: View {
    let tab: AppTab
    @ViewBuilder let content: () -> Content

    @Environment(AppRouter.self) private var router

    var body: some View {
        NavigationStack(path: router.path(tab)) {
            content()
                .navigationDestination(for: Route.self) { route in
                    RouteDestination(route: route)
                }
        }
    }
}

/// The screen each Route shows.
struct RouteDestination: View {
    let route: Route

    var body: some View {
        switch route {
        case .person(let id):
            PersonScreen(personId: id)
        case .album(let id):
            AlbumScreen(albumId: id)
        case .item(let uid):
            ViewerScreen(request: ViewerRequest(entries: [ViewerEntry(uid: uid)], start: 0))
                .toolbar(.hidden, for: .navigationBar, .tabBar)
        case .more(let destination):
            MoreDestinationView(destination: destination)
        case .mapAt(let start):
            MapScreen(start: start)
        case .libraryFiltered(let filters):
            LibraryBrowser(scope: .all, filters: filters)
                .navigationTitle("Library")
        }
    }
}

/// The More tab: the screens that do not have a tab of their own.
struct MoreMenu: View {
    var body: some View {
        List {
            ForEach(MoreDestination.allCases) { destination in
                NavigationLink(value: Route.more(destination)) {
                    Label(destination.title, systemImage: destination.systemImage)
                }
                .accessibilityIdentifier("more.\(destination.rawValue)")
            }
        }
        .navigationTitle("More")
    }
}

/// The screen behind each entry of More.
struct MoreDestinationView: View {
    let destination: MoreDestination

    var body: some View {
        switch destination {
        case .map: MapScreen()
        case .documents: DocumentsScreen()
        case .vault: VaultScreen()
        case .upload: UploadScreen()
        case .trash: TrashScreen()
        case .settings: SettingsScreen()
        }
    }
}
