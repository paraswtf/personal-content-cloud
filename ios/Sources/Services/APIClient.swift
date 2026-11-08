import Foundation

/// The server's HTTP API (docs/ARCHITECTURE.md sections 6, 13 and 21), spoken with a device token.
struct APIError: LocalizedError {
    let status: Int
    let detail: String
    /// Set on 409 offset_mismatch: where the server's copy ends.
    var receivedBytes: Int64? = nil
    var errorDescription: String? { status == 0 ? detail : "\(detail) (\(status))" }

    /// The server's error body: {"detail": "..."} and, for offset_mismatch, "received_bytes".
    /// A body that is not of that shape (a proxy page, a validation list) falls back to its text.
    static func from(status: Int, body: Data) -> APIError {
        struct Body: Decodable { let detail: String?; let received_bytes: Int64? }
        if let b = try? JSONDecoder().decode(Body.self, from: body), let detail = b.detail {
            return APIError(status: status, detail: detail, receivedBytes: b.received_bytes)
        }
        let text = String(data: body.prefix(200), encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return APIError(status: status, detail: text.isEmpty ? HTTPURLResponse.localizedString(forStatusCode: status) : text)
    }
}

struct AuthState: Decodable {
    let authenticated: Bool
    let username: String?
    let site_title: String?
}

struct UploadOut: Decodable {
    let id: String
    let status: String          // uploading, queued, processing, stored, duplicate, rejected, error, cancelled
    let message: String?
    let chunk_size: Int
    let received_bytes: Int64
    let item_uid: String?
}

struct ChunkOut: Decodable { let received_bytes: Int64 }

/// One of this device's unfinished uploads of an asset (section 28).
struct PendingUpload: Decodable {
    let upload_id: String
    let resource: String          // still, video or file
    let received_bytes: Int64
    let size: Int64
}

struct SyncCheckOut: Decodable {
    let known: [String: String]   // sha256 -> uid (live or in the trash)
    let assets: [String: String]  // asset id -> uid (live or in the trash)
    let pending: [String: [PendingUpload]]  // asset id -> this device's unfinished uploads
    let trashed: [String]         // asset ids whose item is in the server's trash

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        known = try c.decodeIfPresent([String: String].self, forKey: .known) ?? [:]
        assets = try c.decodeIfPresent([String: String].self, forKey: .assets) ?? [:]
        pending = try c.decodeIfPresent([String: [PendingUpload]].self, forKey: .pending) ?? [:]
        trashed = try c.decodeIfPresent([String].self, forKey: .trashed) ?? []
    }

    private enum CodingKeys: String, CodingKey { case known, assets, pending, trashed }
}

struct ItemBrief: Decodable {
    let uid: String
    let deleted_at: String?
}

actor APIClient {
    private let base: URL
    private let token: String
    private let session: URLSession

    init(base: URL, token: String) {
        self.base = base
        self.token = token
        let cfg = URLSessionConfiguration.default
        cfg.waitsForConnectivity = true
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 3600
        cfg.httpCookieStorage = nil
        session = URLSession(configuration: cfg)
    }

    static func current() -> APIClient? {
        guard let base = AppSettings.serverURL, let token = Keychain.token() else { return nil }
        return APIClient(base: base, token: token)
    }

    // MARK: requests

    /// A request to the server with the device token. Also used by the background uploader, which
    /// sends its bodies from files.
    static func makeRequest(base: URL, token: String, method: String, path: String,
                            query: [String: String] = [:]) -> URLRequest {
        var comps = URLComponents(url: base.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty { comps.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) } }
        var req = URLRequest(url: comps.url!)
        req.httpMethod = method
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        return req
    }

    private func request(_ method: String, _ path: String, query: [String: String] = [:],
                         json: Encodable? = nil, body: Data? = nil, contentType: String? = nil) -> URLRequest {
        var req = Self.makeRequest(base: base, token: token, method: method, path: path, query: query)
        if let json {
            req.httpBody = try? JSONEncoder().encode(AnyEncodable(json))
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        } else if let body {
            req.httpBody = body
            req.setValue(contentType ?? "application/octet-stream", forHTTPHeaderField: "Content-Type")
        }
        return req
    }

    private func send<T: Decodable>(_ req: URLRequest, as type: T.Type) async throws -> T {
        let (data, resp) = try await session.data(for: req)
        let status = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else { throw APIError.from(status: status, body: data) }
        return try JSONDecoder().decode(T.self, from: data)
    }

    // MARK: endpoints

    func state() async throws -> AuthState {
        try await send(request("GET", "/api/auth/state"), as: AuthState.self)
    }

    func check(hashes: [String], assetIds: [String]) async throws -> SyncCheckOut {
        try await send(request("POST", "/api/uploads/check", json: ["sha256": hashes, "asset_ids": assetIds]),
                       as: SyncCheckOut.self)
    }

    func item(uid: String) async throws -> ItemBrief? {
        do {
            return try await send(request("GET", "/api/items/\(uid)"), as: ItemBrief.self)
        } catch let e as APIError where e.status == 404 {
            return nil
        }
    }

    /// The state of one upload; nil when the server no longer has it (404).
    func upload(id: String) async throws -> UploadOut? {
        do {
            return try await uploadStatus(id: id)
        } catch let e as APIError where e.status == 404 {
            return nil
        }
    }

    /// Cancels an unfinished upload (its file changed on the phone); a finished one is left as it is.
    func cancelUpload(id: String) async throws {
        _ = try await send(request("DELETE", "/api/uploads/\(id)"), as: UploadOut.self)
    }

    func uploadStatus(id: String) async throws -> UploadOut {
        try await send(request("GET", "/api/uploads/\(id)"), as: UploadOut.self)
    }
}

// Small helpers so plain dictionaries can be sent as JSON.
enum AnyEncodableValue: Encodable {
    case string(String), int(Int64), bool(Bool), strings([String])

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let s): try c.encode(s)
        case .int(let i): try c.encode(i)
        case .bool(let b): try c.encode(b)
        case .strings(let a): try c.encode(a)
        }
    }
}

struct AnyEncodable: Encodable {
    private let encodeImpl: (Encoder) throws -> Void
    init(_ value: Encodable) { encodeImpl = value.encode }
    func encode(to encoder: Encoder) throws { try encodeImpl(encoder) }
}

