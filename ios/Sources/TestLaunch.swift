#if DEBUG
import Foundation

/// How the UI tests on the CI machine start the app (ios/README.md, "Tests on GitHub"). Debug builds only:
/// none of this is compiled into a Release build.
///
/// - PCC_TEST_SERVER: the seeded server's address, stored as the server the app talks to.
/// - PCC_TEST_TOKEN: a device token; the app is signed in with it (Keychain.token() also answers with it,
///   so a simulator without a working Keychain still signs in).
/// - PCC_TEST_SIGNED_OUT=1: forget any stored token, so the sign-in screen shows (with the server filled in).
///
/// With a test server the automatic sync is off, so the simulator's own sample photos are not uploaded
/// while a test looks at a screen.
enum TestLaunch {
    private static var env: [String: String] { ProcessInfo.processInfo.environment }

    static var token: String? {
        guard let t = env["PCC_TEST_TOKEN"]?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty else { return nil }
        return t
    }

    static func apply() {
        if let text = env["PCC_TEST_SERVER"], let url = URL(string: text) {
            AppSettings.serverURL = url
            AppSettings.autoSync = false
        }
        if env["PCC_TEST_SIGNED_OUT"] == "1" {
            Keychain.deleteToken()
        } else if let token {
            Keychain.save(token: token)
        }
    }
}
#endif
