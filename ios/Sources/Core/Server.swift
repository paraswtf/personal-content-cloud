import Foundation

/// The signed-in server as the core uses it: the generated layer's `APIConnection` (base address and device
/// token, docs/ARCHITECTURE.md sections 4 and 21) with the helpers for the core's own requests and for media.
/// Every request goes to this one host: the token is never sent anywhere else.
extension APIConnection {
    /// The signed-in server, or nil before sign-in (kept in memory by ConnectionCache).
    static var current: APIConnection? { ConnectionCache.shared.get() }

    /// The headers every request carries: the bearer token (when there is one) and the CSRF header.
    var headers: [String: String] {
        var h = ["X-Requested-With": "pcc"]
        if !token.isEmpty { h["Authorization"] = "Bearer \(token)" }
        return h
    }

    /// The URL of an API route such as "/api/items" (under the base address, as the generated calls build it).
    func apiURL(_ path: String, query: [URLQueryItem] = []) -> URL? {
        url(path: path, query: query)
    }

    /// An absolute URL for a media address the API returns ("/files/thumb/a1?v=2", or a full URL on this server).
    /// A full URL on another host gives nil. With `vault`, the address carries vault=1 (vault items need it).
    func mediaURL(_ ref: String, vault: Bool = false) -> URL? {
        let trimmed = ref.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        var path = trimmed
        var query: String?
        if let absolute = URL(string: trimmed), absolute.scheme != nil {
            guard sameOrigin(absolute) else { return nil }
            path = absolute.path(percentEncoded: true)
            query = absolute.query(percentEncoded: true)
            let basePath = base.path(percentEncoded: true)
            if !basePath.isEmpty, basePath != "/", path.hasPrefix(basePath) { path = String(path.dropFirst(basePath.count)) }
        } else if let mark = trimmed.firstIndex(of: "?") {
            path = String(trimmed[..<mark])
            query = String(trimmed[trimmed.index(after: mark)...])
        }
        if vault {
            let names = (query ?? "").split(separator: "&").map { String($0.split(separator: "=").first ?? "") }
            if !names.contains("vault") { query = (query ?? "").isEmpty ? "vault=1" : "\(query ?? "")&vault=1" }
        }
        return url(encodedPath: path, encodedQuery: query)
    }

    /// True when `url` has the base address's scheme, host and port.
    func sameOrigin(_ url: URL) -> Bool {
        url.scheme?.lowercased() == base.scheme?.lowercased()
            && url.host?.lowercased() == base.host?.lowercased()
            && (url.port ?? Self.defaultPort(url.scheme)) == (base.port ?? Self.defaultPort(base.scheme))
    }

    private static func defaultPort(_ scheme: String?) -> Int? {
        switch scheme?.lowercased() {
        case "https": return 443
        case "http": return 80
        default: return nil
        }
    }

    /// A request to `url` with the token and the CSRF header (only for this server's own addresses).
    func request(_ url: URL, method: String = "GET") -> URLRequest {
        var req = URLRequest(url: url)
        req.httpMethod = method
        if sameOrigin(url) {
            for (name, value) in headers { req.setValue(value, forHTTPHeaderField: name) }
        }
        return req
    }
}
