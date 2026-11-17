import SwiftUI
import SwiftData
import BackgroundTasks

@main
struct PersonalContentCloudApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var session = AppSession()

    init() {
        #if DEBUG
        TestLaunch.apply()  // UI tests sign in from the launch environment (ios/README.md)
        #endif
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(session)
                .modelContainer(SyncStore.shared.container)
                .settingsAppearance()  // Settings > Appearance (Features/Settings/Account)
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

    /// Forgets the token on this phone only. Settings' Sign out ends the session on the server first
    /// (`signOutOnServer()`); this alone is for a token the server no longer accepts or has just revoked.
    func signOut() {
        BackgroundUploader.shared.pause()
        Keychain.deleteToken()
        signedIn = false
    }

    /// Settings' Sign out (the website's logout): the server ends this phone's session (POST /api/auth/logout), so
    /// its token stops working and it leaves Sessions and devices, then the phone forgets the token. When the server
    /// cannot be reached or refuses, the phone stays signed in and the caller says why.
    func signOutOnServer() async -> ServerSignOut {
        let outcome: ServerSignOut
        do {
            _ = try await Cloud.api().logout()
            outcome = .done
        } catch {
            outcome = ServerSignOut.of(error)
        }
        if outcome == .done { signOut() }
        return outcome
    }
}

/// How ending the session on the server went.
enum ServerSignOut: Equatable {
    /// The session is over (or the server no longer knew the token): the phone signed out.
    case done
    /// No answer from the server (no connection, or it is down): the owner may sign out on the phone anyway.
    case unreachable
    /// The server answered with an error: "Could not sign out. Try again."
    case failed
    /// The request was cancelled: nothing to say.
    case cancelled

    static func of(_ error: Error) -> ServerSignOut {
        if error is CancellationError { return .cancelled }
        if let api = error as? APIError {
            if api.status == 401 { return .done }   // the token was already revoked: nothing left to end
            if api.status == 0 || api.status >= 500 { return .unreachable }
            return .failed
        }
        if let u = error as? URLError { return u.code == .cancelled ? .cancelled : .unreachable }
        return .failed
    }
}
