import Foundation

/// The signed-in server and token, read from the Keychain once and kept in memory (thumbnails ask for it often).
/// Reset on sign-in and sign-out.
final class ConnectionCache: @unchecked Sendable {
    static let shared = ConnectionCache()

    private let lock = NSLock()
    private var value: APIConnection?
    private var loaded = false

    func get() -> APIConnection? {
        lock.lock()
        defer { lock.unlock() }
        if !loaded {
            value = APIConnection.signedIn()
            loaded = true
        }
        return value
    }

    func reset() {
        lock.lock()
        value = nil
        loaded = false
        lock.unlock()
    }
}

/// The generated API for the screens: `try await Cloud.api().listItems(cursor: c)`. One client per signed-in
/// server, so its connection pool is shared.
@MainActor
enum Cloud {
    private static var cached: (connection: APIConnection, client: APIClient)?

    /// The client for the signed-in server; throws "login_required" (401) when signed out.
    static func api() throws -> APIClient {
        guard let connection = APIConnection.current else {
            cached = nil
            throw APIError(status: 401, detail: "login_required")
        }
        if let cached, cached.connection == connection { return cached.client }
        let client = APIClient(base: connection.base, token: connection.token)
        cached = (connection, client)
        return client
    }
}
