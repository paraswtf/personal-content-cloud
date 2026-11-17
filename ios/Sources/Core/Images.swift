import SwiftUI
import UIKit
import ImageIO
import CryptoKit

/// Pictures from the server (thumbs, previews, face crops), sent with the device token. A memory cache, a disk
/// cache in Caches (answers marked no-store, such as vault items, stay in memory only), and decoding and downsizing
/// off the main thread. A load is cancelled with the task that asked for it (a view that scrolled away).
final class ImageLoader: @unchecked Sendable {
    static let shared = ImageLoader()

    private let memory = NSCache<NSString, UIImage>()
    /// Vault pictures, apart from the rest so locking the vault can forget them all (`forgetVault()`).
    private let vaultMemory = NSCache<NSString, UIImage>()
    private let folder: URL
    private let session: URLSession
    /// The disk cache is trimmed to this size at launch, oldest files first.
    static let diskLimitBytes: Int64 = 500 * 1024 * 1024

    init() {
        memory.totalCostLimit = 160 * 1024 * 1024
        vaultMemory.totalCostLimit = 64 * 1024 * 1024
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        folder = caches.appendingPathComponent("pcc-images", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let cfg = URLSessionConfiguration.default
        cfg.urlCache = nil
        cfg.httpCookieStorage = nil
        cfg.httpShouldSetCookies = false
        cfg.timeoutIntervalForRequest = 60
        cfg.httpMaximumConnectionsPerHost = 6
        // Redirects stay on the server's own host, so the token never follows one elsewhere.
        session = URLSession(configuration: cfg, delegate: SameHostRedirects.shared, delegateQueue: nil)
    }

    /// A picture already in memory, for showing it without a flash.
    func cached(_ ref: String, maxPixel: Int, vault: Bool = false) -> UIImage? {
        (vault ? vaultMemory : memory).object(forKey: Self.key(ref, maxPixel, vault) as NSString)
    }

    /// Loads a picture no larger than `maxPixel` on its long side. Vault items need `vault` (their addresses
    /// answer only with vault=1 on an unlocked session).
    func image(_ ref: String, maxPixel: Int, vault: Bool = false) async throws -> UIImage {
        let key = Self.key(ref, maxPixel, vault)
        if let hit = (vault ? vaultMemory : memory).object(forKey: key as NSString) { return hit }
        guard let server = APIConnection.current, let url = server.mediaURL(ref, vault: vault) else { throw URLError(.badURL) }
        let file = folder.appendingPathComponent(Self.diskName(url.absoluteString))
        if let data = try? Data(contentsOf: file), let image = Self.decode(data, maxPixel: maxPixel) {
            store(image, key, vault: vault)
            return image
        }
        try Task.checkCancellation()
        let answer: (Data, URLResponse)
        do {
            answer = try await session.data(for: server.request(url))
        } catch let e as URLError where e.code == .cancelled {
            throw CancellationError()
        }
        let (data, response) = answer
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        guard (200..<300).contains(http.statusCode) else { throw APIError.from(status: http.statusCode, body: data) }
        guard let image = Self.decode(data, maxPixel: maxPixel) else { throw URLError(.cannotDecodeContentData) }
        let cacheControl = (http.value(forHTTPHeaderField: "Cache-Control") ?? "").lowercased()
        if !cacheControl.contains("no-store") && !vault {
            try? data.write(to: file, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        }
        store(image, key, vault: vault)
        return image
    }

    /// Forgets the vault's pictures (the vault was found locked: locked by the owner, or its window ended).
    func forgetVault() {
        vaultMemory.removeAllObjects()
    }

    /// Forgets every cached picture (sign-out).
    func clear() {
        memory.removeAllObjects()
        vaultMemory.removeAllObjects()
        try? FileManager.default.removeItem(at: folder)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    }

    /// Keeps the disk cache under its limit, deleting the files used longest ago first.
    func trimDisk() {
        let keys: [URLResourceKey] = [.contentModificationDateKey, .fileSizeKey]
        guard let files = try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: keys)
        else { return }
        var entries: [(url: URL, date: Date, size: Int64)] = []
        var total: Int64 = 0
        for file in files {
            let values = try? file.resourceValues(forKeys: Set(keys))
            let size = Int64(values?.fileSize ?? 0)
            entries.append((url: file, date: values?.contentModificationDate ?? .distantPast, size: size))
            total += size
        }
        guard total > Self.diskLimitBytes else { return }
        for entry in entries.sorted(by: { $0.date < $1.date }) {
            try? FileManager.default.removeItem(at: entry.url)
            total -= entry.size
            if total <= Self.diskLimitBytes * 8 / 10 { break }
        }
    }

    private func store(_ image: UIImage, _ key: String, vault: Bool) {
        let pixels = Int(image.size.width * image.scale * image.size.height * image.scale)
        (vault ? vaultMemory : memory).setObject(image, forKey: key as NSString, cost: pixels * 4)
    }

    private static func key(_ ref: String, _ maxPixel: Int, _ vault: Bool) -> String { "\(maxPixel)|\(vault ? "v" : "")|\(ref)" }

    private static func diskName(_ url: String) -> String {
        SHA256.hash(data: Data(url.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    /// Decodes and downsizes in one step (ImageIO thumbnails never upscale), applying the EXIF orientation.
    static func decode(_ data: Data, maxPixel: Int) -> UIImage? {
        let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, sourceOptions) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: max(1, maxPixel),
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        return UIImage(cgImage: cg)
    }
}

/// A picture from the server, filling its frame (or fitting), with a quiet placeholder while it loads.
struct RemoteImage: View {
    let ref: String?
    var maxPixel: Int = 512
    var contentMode: ContentMode = .fill
    /// A vault item's picture (sent with vault=1).
    var vault = false

    @State private var image: UIImage?
    /// The address `image` was loaded from, so a reused view never shows another item's picture.
    @State private var shown: String?
    @State private var failed = false

    init(ref: String?, maxPixel: Int = 512, contentMode: ContentMode = .fill, vault: Bool = false) {
        self.ref = ref
        self.maxPixel = maxPixel
        self.contentMode = contentMode
        self.vault = vault
        let hit = ref.flatMap { ImageLoader.shared.cached($0, maxPixel: maxPixel, vault: vault) }
        _image = State(initialValue: hit)
        _shown = State(initialValue: hit == nil ? nil : ref)
    }

    var body: some View {
        ZStack {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: contentMode)
            } else {
                Rectangle().fill(Color(uiColor: .secondarySystemFill))
                if failed {
                    Image(systemName: "photo")
                        .foregroundStyle(.secondary)
                        .accessibilityHidden(true)
                }
            }
        }
        .task(id: "\(ref ?? "")|\(maxPixel)|\(vault)") { await load() }
    }

    private func load() async {
        guard let ref, !ref.isEmpty else {
            image = nil
            failed = true
            return
        }
        if let hit = ImageLoader.shared.cached(ref, maxPixel: maxPixel, vault: vault) {
            image = hit
            shown = ref
            failed = false
            return
        }
        if shown != ref {
            image = nil
            shown = nil
        }
        failed = false
        do {
            let loaded = try await ImageLoader.shared.image(ref, maxPixel: maxPixel, vault: vault)
            image = loaded
            shown = ref
        } catch {
            if !(error is CancellationError) && !Task.isCancelled { failed = true }
        }
    }
}
