import Foundation
import Photos
import CryptoKit
import UniformTypeIdentifiers

/// PhotoKit access: which assets exist, their original files, and deletion.
enum PhotoLibrary {
    static func requestAccess() async -> PHAuthorizationStatus {
        await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    }

    static var hasAccess: Bool {
        let s = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        return s == .authorized || s == .limited
    }

    /// Photos and videos, newest first, created after `since` when given.
    static func assets(since: Date?, includeVideos: Bool) -> [PHAsset] {
        let options = PHFetchOptions()
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        var predicates: [NSPredicate] = []
        if let since { predicates.append(NSPredicate(format: "creationDate > %@", since as NSDate)) }
        if includeVideos {
            predicates.append(NSPredicate(format: "mediaType == %d OR mediaType == %d",
                                          PHAssetMediaType.image.rawValue, PHAssetMediaType.video.rawValue))
        } else {
            predicates.append(NSPredicate(format: "mediaType == %d", PHAssetMediaType.image.rawValue))
        }
        options.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: predicates)
        let result = PHAsset.fetchAssets(with: options)
        var out: [PHAsset] = []
        out.reserveCapacity(result.count)
        result.enumerateObjects { asset, _, _ in out.append(asset) }
        return out
    }

    static func assets(ids: [String]) -> [PHAsset] {
        var out: [PHAsset] = []
        PHAsset.fetchAssets(withLocalIdentifiers: ids, options: nil).enumerateObjects { a, _, _ in out.append(a) }
        return out
    }

    /// The original files of an asset: the still (or video), plus the paired clip of a Live Photo.
    struct ExportedFile {
        let url: URL
        let filename: String
        let mime: String?
        let bytes: Int64
        let isClip: Bool
    }

    static func export(_ asset: PHAsset, to dir: URL) async throws -> [ExportedFile] {
        let resources = PHAssetResource.assetResources(for: asset)
        var wanted: [(PHAssetResource, Bool)] = []
        if asset.mediaType == .video {
            if let r = resources.first(where: { $0.type == .video }) ?? resources.first(where: { $0.type == .fullSizeVideo }) {
                wanted.append((r, false))
            }
        } else {
            if let r = resources.first(where: { $0.type == .photo }) ?? resources.first(where: { $0.type == .fullSizePhoto }) {
                wanted.append((r, false))
            }
            if asset.mediaSubtypes.contains(.photoLive), let clip = resources.first(where: { $0.type == .pairedVideo }) {
                wanted.append((clip, true))
            }
        }
        var out: [ExportedFile] = []
        for (resource, isClip) in wanted {
            let name = resource.originalFilename.isEmpty ? "\(asset.localIdentifier.prefix(8)).bin" : resource.originalFilename
            let url = dir.appendingPathComponent(UUID().uuidString + "-" + name)
            let options = PHAssetResourceRequestOptions()
            options.isNetworkAccessAllowed = true
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                PHAssetResourceManager.default().writeData(for: resource, toFile: url, options: options) { error in
                    if let error { cont.resume(throwing: error) } else { cont.resume() }
                }
            }
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
            let mime = UTType(resource.uniformTypeIdentifier)?.preferredMIMEType
            out.append(ExportedFile(url: url, filename: name, mime: mime, bytes: size, isClip: isClip))
        }
        return out
    }

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 4 * 1024 * 1024), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Deletes the assets through the system confirmation dialog. Returns true when the user confirmed.
    static func delete(_ assets: [PHAsset]) async -> Bool {
        guard !assets.isEmpty else { return false }
        do {
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.deleteAssets(assets as NSArray)
            }
            return true
        } catch {
            return false
        }
    }
}
