import Foundation

// The transport under the generated API layer (Models.generated.swift and Endpoints.generated.swift, written by
// scripts/gen_swift_api.py from the server's OpenAPI description). Hand-written: the generator only calls it.
//
// Every request carries the device token as a bearer token and `X-Requested-With: pcc` (the server's CSRF
// header, docs/ARCHITECTURE.md section 4). Writes carry the task's undo id when one is set (section 33), and
// an answer that names its action is announced with `Notification.Name.apiActionRecorded`, so the app can
// offer Undo as the website does. A 401 outside /api/auth/ is announced with `.apiUnauthorized`.

/// The namespace of the server's types.
enum API {}

// MARK: - Connection

/// The server address and device token requests are made with.
struct APIConnection: Sendable, Hashable {
    let base: URL
    let token: String

    /// The signed-in server and token: the values `APIClient.current()` builds a client from.
    static func signedIn() -> APIConnection? {
        guard let base = AppSettings.serverURL, let token = Keychain.token(), !token.isEmpty else { return nil }
        return APIConnection(base: base, token: token)
    }

    /// An address under the server's base address, from a path and query that are already percent-encoded
    /// ("/files/thumb/abc", "v=3"). A base address with a path of its own keeps it in front.
    func url(encodedPath: String, encodedQuery: String? = nil) -> URL? {
        var root = base.absoluteString
        if let cut = root.firstIndex(where: { $0 == "?" || $0 == "#" }) {
            root = String(root[..<cut])
        }
        while root.hasSuffix("/") {
            root.removeLast()
        }
        let path = encodedPath.hasPrefix("/") ? encodedPath : "/" + encodedPath
        let query = encodedQuery.map { $0.isEmpty ? "" : "?" + $0 } ?? ""
        return URL(string: root + path + query)
    }

    /// An address under the server's base address from an encoded path and plain query items.
    func url(path: String, query: [URLQueryItem]) -> URL? {
        guard !query.isEmpty else { return url(encodedPath: path) }
        var items = URLComponents()
        items.queryItems = query
        // URLComponents leaves "+" alone, and the server reads it as a space.
        let encoded = items.percentEncodedQuery?.replacingOccurrences(of: "+", with: "%2B")
        return url(encodedPath: path, encodedQuery: encoded)
    }
}

// MARK: - Task context

/// Values that ride along with every request made inside a task (Swift task-locals).
enum APIContext {
    /// The undo action the writes of this task belong to (section 33). The several requests of one owner
    /// action share it, so one Undo puts them all back. Unset, the server names each write on its own.
    @TaskLocal static var actionID: String? = nil

    /// Vault mode: every request and media address carries `vault=1`, as on the website's vault pages.
    /// The session must have unlocked the vault (`vaultUnlock`), or the server answers 403 `vault_locked`.
    @TaskLocal static var vault: Bool = false

    /// A new undo id (the server accepts 8 to 64 letters, digits, "-" and "_").
    static func newActionID() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }

    /// Runs `body` as one owner action: all its writes share one new undo id.
    static func action<T>(_ body: () async throws -> T) async rethrows -> T {
        try await $actionID.withValue(newActionID(), operation: body)
    }
}

extension Notification.Name {
    /// A write the server recorded as an undoable action. userInfo: "id", "method" and "path" (all String).
    /// Undo it with `undoAction(actionId:)`.
    static let apiActionRecorded = Notification.Name("in.rivant.pcc.api.actionRecorded")
    /// A 401 outside /api/auth/: the device token no longer works (revoked or expired).
    static let apiUnauthorized = Notification.Name("in.rivant.pcc.api.unauthorized")
}

// MARK: - Query and path encoding

/// Query items with the website's encoding: nil is left out, true and false are 1 and 0, arrays repeat the key.
struct APIQuery {
    private(set) var items: [URLQueryItem] = []

    mutating func add(_ name: String, _ value: String?) {
        if let value { items.append(URLQueryItem(name: name, value: value)) }
    }

    mutating func add(_ name: String, _ value: Int?) {
        if let value { add(name, String(value)) }
    }

    mutating func add(_ name: String, _ value: Double?) {
        if let value { add(name, String(value)) }
    }

    mutating func add(_ name: String, _ value: Bool?) {
        if let value { add(name, value ? "1" : "0") }
    }

    mutating func add<E: RawRepresentable>(_ name: String, _ value: E?) where E.RawValue == String {
        if let value { add(name, value.rawValue) }
    }

    mutating func add(_ name: String, _ values: [String]?) {
        for value in values ?? [] { add(name, value) }
    }

    mutating func add(_ name: String, _ values: [Int]?) {
        for value in values ?? [] { add(name, value) }
    }

    mutating func add(_ name: String, _ values: [Double]?) {
        for value in values ?? [] { add(name, value) }
    }

    mutating func add<E: RawRepresentable>(_ name: String, _ values: [E]?) where E.RawValue == String {
        for value in values ?? [] { add(name, value.rawValue) }
    }

    mutating func add(_ name: String, _ value: API.JSONValue?) {
        if let value, let text = value.queryText { add(name, text) }
    }
}

/// Path segments, percent-encoded ("/" included, so a value never adds a segment).
enum APIPath {
    private static let allowed: CharacterSet = {
        var set = CharacterSet.urlPathAllowed
        set.remove("/")
        return set
    }()

    static func segment(_ value: String) -> String {
        value.addingPercentEncoding(withAllowedCharacters: allowed) ?? value
    }

    static func segment(_ value: Int) -> String {
        String(value)
    }

    static func segment(_ value: Double) -> String {
        String(value)
    }

    static func segment<E: RawRepresentable>(_ value: E) -> String where E.RawValue == String {
        segment(value.rawValue)
    }
}

// MARK: - Requests

extension APIClient {
    /// Methods that change something: they carry the undo id, and their answer may name an action.
    private static let writeMethods: Set<String> = ["POST", "PUT", "PATCH", "DELETE"]

    /// The session of the generated endpoints: no cookies (the device token is the only credential), a request
    /// fails instead of waiting when there is no network, so a screen can say so, and redirects stay on the
    /// server's own host (`SameHostRedirects`).
    static let apiSession: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.httpCookieStorage = nil
        cfg.httpShouldSetCookies = false
        cfg.waitsForConnectivity = false
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 600
        return URLSession(configuration: cfg, delegate: SameHostRedirects.shared, delegateQueue: nil)
    }()

    /// The server and token the generated endpoints speak with: this client's own, so a client made for
    /// another server (the sign-in screen's check) sends its generated calls there too.
    nonisolated func connection() throws -> APIConnection {
        guard !token.isEmpty else { throw APIError(status: 401, detail: "login_required") }
        return APIConnection(base: base, token: token)
    }

    /// A request to the server: the token, the CSRF header, the undo id of a write, `vault=1` in vault mode,
    /// and the body as JSON.
    nonisolated func urlRequest(_ method: String, _ path: String, query: [URLQueryItem] = [],
                                body: (any Encodable)? = nil, headers: [String: String] = [:]) throws -> URLRequest {
        let connection = try connection()
        var items = query
        if APIContext.vault, !items.contains(where: { $0.name == "vault" }) {
            items.append(URLQueryItem(name: "vault", value: "1"))
        }
        guard let url = connection.url(path: path, query: items) else {
            throw APIError(status: 0, detail: "Not a valid address: \(path)")
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("Bearer \(connection.token)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("pcc", forHTTPHeaderField: "X-Requested-With")
        if Self.writeMethods.contains(method), let action = APIContext.actionID {
            request.setValue(action, forHTTPHeaderField: "X-Action-Id")
        }
        for (name, value) in headers {
            request.setValue(value, forHTTPHeaderField: name)
        }
        if let body {
            do {
                request.httpBody = try JSONEncoder().encode(body)
            } catch {
                throw APIError(status: 0, detail: "The request could not be written: \(error)")
            }
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return request
    }

    /// Sends a request and returns the answer's body. Throws `APIError` with the server's `detail` for an
    /// answer outside 200 to 299 (a 304 to an If-None-Match included).
    nonisolated func performData(_ method: String, _ path: String, query: [URLQueryItem] = [],
                                 body: (any Encodable)? = nil, headers: [String: String] = [:]) async throws -> Data {
        let request = try urlRequest(method, path, query: query, body: body, headers: headers)
        let (data, response) = try await Self.apiSession.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw APIError(status: 0, detail: "No answer from the server")
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401, !path.hasPrefix("/api/auth/") {
                NotificationCenter.default.post(name: .apiUnauthorized, object: nil)
            }
            throw APIError.from(status: http.statusCode, body: data)
        }
        if Self.writeMethods.contains(method), let action = http.value(forHTTPHeaderField: "X-Action-Id"),
           !action.isEmpty {
            NotificationCenter.default.post(name: .apiActionRecorded, object: nil,
                                            userInfo: ["id": action, "method": method, "path": path])
        }
        return data
    }

    /// Sends a request and decodes its JSON answer.
    nonisolated func perform<T: Decodable>(_ method: String, _ path: String, query: [URLQueryItem] = [],
                                           body: (any Encodable)? = nil, headers: [String: String] = [:],
                                           as type: T.Type) async throws -> T {
        let data = try await performData(method, path, query: query, body: body, headers: headers)
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw APIError(status: 0, detail: "The server's answer to \(method) \(path) could not be read: \(error)")
        }
    }

    /// Sends a request whose answer is not needed.
    nonisolated func performVoid(_ method: String, _ path: String, query: [URLQueryItem] = [],
                                 body: (any Encodable)? = nil, headers: [String: String] = [:]) async throws {
        _ = try await performData(method, path, query: query, body: body, headers: headers)
    }

    // MARK: media

    /// A media address from an answer (`thumb_url`, `preview_url`, a face crop's `url`, a share's file) as a
    /// full URL on the server. In vault mode it carries `vault=1`. An absolute address on another host is
    /// refused, so the token is never sent anywhere else.
    nonisolated func mediaURL(_ address: String, vault: Bool = APIContext.vault) throws -> URL {
        let connection = try connection()
        var path = address
        var query: String?
        if let absolute = URL(string: address), absolute.scheme != nil {
            // The same check as the picture loader's (an explicit :443 is the server's own port too).
            guard connection.sameOrigin(absolute) else {
                throw APIError(status: 0, detail: "An address outside the server was refused")
            }
            let encodedPath = absolute.path(percentEncoded: true)
            path = encodedPath.isEmpty ? "/" : encodedPath
            query = absolute.query(percentEncoded: true)
            let base = connection.base.path(percentEncoded: true)
            if !base.isEmpty, base != "/", path.hasPrefix(base) {
                path = String(path.dropFirst(base.count))
            }
        } else if let mark = address.firstIndex(of: "?") {
            path = String(address[..<mark])
            query = String(address[address.index(after: mark)...])
        }
        if vault {
            let pairs = (query ?? "").split(separator: "&").map { $0.split(separator: "=").first.map(String.init) ?? "" }
            if !pairs.contains("vault") {
                query = (query?.isEmpty ?? true) ? "vault=1" : query! + "&vault=1"
            }
        }
        guard let url = connection.url(encodedPath: path, encodedQuery: query) else {
            throw APIError(status: 0, detail: "Not a valid address: \(address)")
        }
        return url
    }

    /// A signed request for a media address from an answer, for image loaders.
    nonisolated func mediaRequest(_ address: String, vault: Bool = APIContext.vault) throws -> URLRequest {
        var request = URLRequest(url: try mediaURL(address, vault: vault))
        for (name, value) in try mediaHeaders() {
            request.setValue(value, forHTTPHeaderField: name)
        }
        return request
    }

    /// The headers a media request needs. For video: `AVURLAsset(url: try mediaURL(...),
    /// options: ["AVURLAssetHTTPHeaderFieldsKey": try mediaHeaders()])`.
    nonisolated func mediaHeaders() throws -> [String: String] {
        ["Authorization": "Bearer \(try connection().token)", "X-Requested-With": "pcc"]
    }
}

// MARK: - Any JSON

extension API {
    /// Any JSON value, for the few fields the generator cannot type (free-form options and progress,
    /// validation details, unions of plain values).
    enum JSONValue: Codable, Hashable, Sendable {
        case null
        case bool(Bool)
        case number(Double)
        case string(String)
        case array([JSONValue])
        case object([String: JSONValue])

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if c.decodeNil() {
                self = .null
            } else if let value = try? c.decode(Bool.self) {
                self = .bool(value)
            } else if let value = try? c.decode(Double.self) {
                self = .number(value)
            } else if let value = try? c.decode(String.self) {
                self = .string(value)
            } else if let value = try? c.decode([JSONValue].self) {
                self = .array(value)
            } else if let value = try? c.decode([String: JSONValue].self) {
                self = .object(value)
            } else {
                throw DecodingError.dataCorruptedError(in: c, debugDescription: "Not a JSON value")
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .null: try c.encodeNil()
            case .bool(let value): try c.encode(value)
            case .number(let value): try c.encode(value)
            case .string(let value): try c.encode(value)
            case .array(let value): try c.encode(value)
            case .object(let value): try c.encode(value)
            }
        }

        var boolValue: Bool? {
            if case .bool(let value) = self { return value }
            return nil
        }

        var doubleValue: Double? {
            if case .number(let value) = self { return value }
            return nil
        }

        var intValue: Int? {
            if case .number(let value) = self, value.rounded() == value, abs(value) < 9.0e15 { return Int(value) }
            return nil
        }

        var stringValue: String? {
            if case .string(let value) = self { return value }
            return nil
        }

        var arrayValue: [JSONValue]? {
            if case .array(let value) = self { return value }
            return nil
        }

        var objectValue: [String: JSONValue]? {
            if case .object(let value) = self { return value }
            return nil
        }

        var isNull: Bool {
            if case .null = self { return true }
            return false
        }

        subscript(key: String) -> JSONValue? {
            objectValue?[key]
        }

        subscript(index: Int) -> JSONValue? {
            guard let array = arrayValue, array.indices.contains(index) else { return nil }
            return array[index]
        }

        /// The value as a query parameter, for the plain kinds.
        var queryText: String? {
            switch self {
            case .bool(let value): return value ? "1" : "0"
            case .number(let value): return intValue.map { String($0) } ?? String(value)
            case .string(let value): return value
            default: return nil
            }
        }
    }

    /// A coding key made from any string, for the union types that look at an object's keys.
    struct AnyKey: CodingKey {
        var stringValue: String
        var intValue: Int?

        init(_ string: String) {
            stringValue = string
            intValue = nil
        }

        init?(stringValue: String) {
            self.init(stringValue)
        }

        init?(intValue: Int) {
            stringValue = String(intValue)
            self.intValue = intValue
        }
    }
}
