import SwiftUI
import SwiftData
import BackgroundTasks

@main
struct PersonalContentCloudApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var session = AppSession()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(session)
                .modelContainer(SyncStore.shared.container)
        }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        BackgroundSync.register()
        BackgroundUploader.shared.activate()   // reconnect to the background session (section 28)
        return true
    }

    /// iOS relaunched the app (or woke it) because tasks of the background upload session finished
    /// while it was suspended or not running. The uploader's delegate takes the events and calls the
    /// handler once it has seen them all.
    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        guard identifier == BackgroundUploader.identifier else {
            completionHandler()
            return
        }
        BackgroundUploader.shared.backgroundCompletion = completionHandler
        BackgroundUploader.shared.activate()
    }
}

/// Sign-in state shared by the views: server address and device token (the token lives in the Keychain).
@MainActor
final class AppSession: ObservableObject {
    @Published var serverURL: URL? = AppSettings.serverURL
    @Published var signedIn: Bool = Keychain.token() != nil && AppSettings.serverURL != nil

    func signIn(server: URL, token: String) {
        AppSettings.serverURL = server
        Keychain.save(token: token)
        serverURL = server
        signedIn = true
    }

    func signOut() {
        BackgroundUploader.shared.pause()
        Keychain.deleteToken()
        signedIn = false
    }
}
