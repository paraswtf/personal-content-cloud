import Foundation

/// The JSON requests the core makes itself: sign-in, undo, the picker's people and the viewer's item. Screens
/// call the generated API (ios/Sources/API) instead. Every write made inside `UndoCenter.perform` carries that
/// action's id (`X-Action-Id`, docs/ARCHITECTURE.md section 33) and tells the action when the server changed
/// something, so it can be undone.
enum CoreHTTP {
    static let session: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 60
        cfg.httpCookieStorage = nil
        cfg.httpShouldSetCookies = false
        cfg.urlCache = nil
        return URLSession(configuration: cfg)
    }()

    /// Decodes the server's snake_case keys into camelCase properties (`duration_s` becomes `durationS`).
    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }()

    /// Sends one request and returns its body; throws `APIError` for an answer outside 2xx and
    /// `CancellationError` when the calling task was cancelled.
    @discardableResult
    static func raw(_ server: APIConnection, _ method: String, _ path: String, query: [URLQueryItem] = [],
                    body: Data? = nil) async throws -> (Data, HTTPURLResponse) {
        guard let url = server.apiURL(path, query: query) else { throw APIError(status: 0, detail: "Bad address") }
        var req = server.request(url, method: method)
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        if let body {
            req.httpBody = body
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let scope = method == "GET" ? nil : ActionScope.current
        if let scope { req.setValue(scope.id, forHTTPHeaderField: "X-Action-Id") }
        let answer: (Data, URLResponse)
        do {
            answer = try await session.data(for: req)
        } catch let e as URLError where e.code == .cancelled {
            throw CancellationError()
        }
        let (payload, response) = answer
        guard let http = response as? HTTPURLResponse else {
            throw APIError(status: 0, detail: "No answer from the server")
        }
        scope?.note(http)
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401, !path.hasPrefix("/api/auth/"), !server.token.isEmpty {
                NotificationCenter.default.post(name: .apiUnauthorized, object: nil)
            }
            throw APIError.from(status: http.statusCode, body: payload)
        }
        return (payload, http)
    }

    /// GET a route and decode its JSON.
    static func get<T: Decodable>(_ type: T.Type, _ server: APIConnection, _ path: String,
                                  query: [URLQueryItem] = []) async throws -> T {
        let (payload, _) = try await raw(server, "GET", path, query: query)
        return try decoder.decode(T.self, from: payload)
    }

    /// A write with an optional JSON body, decoding the answer.
    static func send<T: Decodable, B: Encodable>(_ type: T.Type, _ server: APIConnection, _ method: String, _ path: String,
                                                  json: B) async throws -> T {
        let body = try JSONEncoder().encode(json)
        let (payload, _) = try await raw(server, method, path, body: body)
        return try decoder.decode(T.self, from: payload)
    }

    /// A write without a body, decoding the answer.
    static func send<T: Decodable>(_ type: T.Type, _ server: APIConnection, _ method: String, _ path: String) async throws -> T {
        let (payload, _) = try await raw(server, method, path)
        return try decoder.decode(T.self, from: payload)
    }
}
