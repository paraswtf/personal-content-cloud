import SwiftUI

/// The signed-in app: five tabs (Library, People, Albums, Review, More), each with its own navigation stack, and
/// the viewer over everything. Review shows how many items wait there; while the vault is open, a bar over every
/// tab shows the time left (ShellVaultBar).
struct MainShell: View {
    @State private var router = AppRouter()
    @Environment(\.scenePhase) private var scenePhase

    private var status: ShellStatus { ShellStatus.shared }

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
                .badge(ShellStatus.badge(status.reviewWaiting).map { Text($0) })
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
        .task {
            startAutomaticSync()
            await status.refresh()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                startAutomaticSync()
                Task { await status.refresh() }
            }
        }
        // Review's count follows decisions made anywhere; changing tabs reads it again now and then.
        .onCloudChange { Task { await status.refreshStats(force: true) } }
        .onChange(of: router.tab) { _, _ in
            Task { await status.refreshStats(force: false) }
        }
        // The vault closed (locked, or its window ran out on the phone's clock): ask the server how things stand,
        // as the website's badge does at zero.
        .onChange(of: VaultSession.shared.until) { old, new in
            if old != nil && new == nil { Task { await status.refresh(force: false) } }
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
        VStack(spacing: 0) {
            ShellVaultBar()
            NavigationStack(path: router.path(tab)) {
                content()
                    .navigationDestination(for: Route.self) { route in
                        RouteDestination(route: route)
                    }
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
        case .vaultFiltered(let filters):
            VaultScreen(filters: filters)
        }
    }
}

/// The More tab: the screens that do not have a tab of their own, under the website's headings (Browse, Tidy up,
/// Account), and the theme.
struct MoreMenu: View {
    @AppStorage(SettingsAppearanceChoice.storageKey) private var theme: SettingsAppearanceChoice = .system

    var body: some View {
        List {
            ForEach(MoreDestination.groups, id: \.title) { group in
                Section(group.title) {
                    ForEach(group.destinations) { destination in
                        row(destination)
                    }
                }
            }
            Section("Theme") {
                Picker("Theme", selection: $theme) {
                    ForEach(SettingsAppearanceChoice.allCases) { option in
                        Text(option.label).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .accessibilityIdentifier("more.theme")
            }
        }
        .navigationTitle("More")
    }

    private func row(_ destination: MoreDestination) -> some View {
        let vault = VaultSession.shared
        let isVault = destination == .vault
        let open = isVault && vault.isOpen()
        // The vault's state, once the server said it (the website's "Open now" and "Locked").
        let hint: String? = isVault && vault.loaded ? (open ? "Open now" : "Locked") : nil
        return NavigationLink(value: Route.more(destination)) {
            Label {
                VStack(alignment: .leading, spacing: 2) {
                    Text(destination.title)
                    if let hint {
                        Text(hint)
                            .font(.caption)
                            .foregroundStyle(open ? Color.orange : Color.secondary)
                    }
                }
            } icon: {
                Image(systemName: open ? "lock.open" : destination.systemImage)
            }
        }
        .accessibilityIdentifier("more.\(destination.rawValue)")
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
