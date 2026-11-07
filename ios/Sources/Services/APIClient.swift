import Foundation

/// The server's HTTP API (docs/ARCHITECTURE.md sections 6, 13 and 21), spoken with a device token.
struct APIError: LocalizedError {
    let status: Int
    let detail: String
    var errorDescription: String? { "\(detail) (\(status))" }
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

struct SyncCheckOut: Decodable {
    let known: [String: String]   // sha256 -> uid
    let assets: [String: String]  // asset id -> uid
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

    private func request(_ method: String, _ path: String, query: [String: String] = [:],
                         json: Encodable? = nil, body: Data? = nil, contentType: String? = nil) -> URLRequest {
        var comps = URLComponents(url: base.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty { comps.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) } }
        var req = URLRequest(url: comps.url!)
        req.httpMethod = method
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Accept")
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
        guard (200..<300).contains(status) else {
            let detail = (try? JSONDecoder().decode([String: String].self, from: data))?["detail"]
                ?? String(data: data.prefix(200), encoding: .utf8) ?? "request failed"
            throw APIError(status: status, detail: detail)
        }
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

    func createUpload(filename: String, size: Int64, mime: String?, sha256: String, pairKey: String?) async throws -> UploadOut {
        var body: [String: AnyEncodableValue] = [
            "filename": .string(filename), "size": .int(size), "sha256": .string(sha256), "source": .string("api"),
            "client_id": .string("ios"),
        ]
        if let mime { body["mime"] = .string(mime) }
        if let pairKey { body["pair_key"] = .string(pairKey) }
        return try await send(request("POST", "/api/uploads", json: body), as: UploadOut.self)
    }

    func putChunk(id: String, offset: Int64, data: Data) async throws -> ChunkOut {
        try await send(request("PUT", "/api/uploads/\(id)/chunk", query: ["offset": String(offset)], body: data),
                       as: ChunkOut.self)
    }

    func complete(id: String) async throws -> UploadOut {
        try await send(request("POST", "/api/uploads/\(id)/complete"), as: UploadOut.self)
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

