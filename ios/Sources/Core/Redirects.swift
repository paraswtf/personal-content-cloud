import Foundation

/// The redirect rule of every session that sends the device token (the generated API's, the core's own requests,
/// the picture loader): a redirect is followed only to the same host, with the same scheme and port or from http to
/// https, so the token in the request's headers never reaches another host (the sign-in page of a proxy in front of
/// the server, for example). A refused redirect hands its 3xx answer back to the caller, which reads it as an error.
final class SameHostRedirects: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    static let shared = SameHostRedirects()

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest) async -> URLRequest? {
        guard let from = task.currentRequest?.url ?? task.originalRequest?.url, let to = request.url,
              Self.allowed(from: from, to: to) else { return nil }
        return request
    }

    /// True for a redirect that stays on the same host: the same scheme and port, or http to https.
    static func allowed(from: URL, to: URL) -> Bool {
        guard let a = from.host?.lowercased(), let b = to.host?.lowercased(), a == b else { return false }
        let fromScheme = from.scheme?.lowercased(), toScheme = to.scheme?.lowercased()
        if fromScheme == toScheme { return port(from) == port(to) }
        return fromScheme == "http" && toScheme == "https"
    }

    private static func port(_ url: URL) -> Int? {
        if let p = url.port { return p }
        switch url.scheme?.lowercased() {
        case "https": return 443
        case "http": return 80
        default: return nil
        }
    }
}
