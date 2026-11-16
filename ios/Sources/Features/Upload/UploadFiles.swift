import Foundation
import SwiftUI
import UIKit
import Photos
import AVFoundation
import ImageIO
import UniformTypeIdentifiers
import CoreTransferable

/// A file handed to the Upload screen (from Files, the Photos picker, a drop or the camera), already copied into
/// the app: `url` is the copy, `name` the name the owner sees and the server stores.
struct UploadIncoming: Sendable {
    let url: URL
    let name: String

    /// Where copies wait until the queue takes them: Application Support (iOS may clear Caches while they wait),
    /// next to the uploads' own folders.
    static var folder: URL {
        let dir = UploadPaths.root.appendingPathComponent("incoming", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// A new place for a copy named `name` (the copy keeps the name; the folder keeps two copies apart).
    static func newURL(for name: String) -> URL {
        let dir = folder.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent(UploadIncoming.safeName(name))
    }

    /// Removes a copy's own folder once the queue has taken (or refused) the copy. Only folders made by newURL.
    static func discard(_ url: URL) {
        let dir = url.deletingLastPathComponent().standardizedFileURL
        guard dir.deletingLastPathComponent().path == folder.standardizedFileURL.path else { return }
        try? FileManager.default.removeItem(at: dir)
    }

    /// A file name without path separators or a leading dot, never empty.
    static func safeName(_ name: String) -> String {
        let cleaned = name.replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "_")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmed = cleaned.hasPrefix(".") ? String(cleaned.drop(while: { $0 == "." })) : cleaned
        return trimmed.isEmpty ? "Untitled" : trimmed
    }

    /// Copies a file the owner chose in Files (a security-scoped address) into the app. Off the main actor: a video
    /// can take a while.
    static func copy(fromPicked url: URL) async throws -> UploadIncoming {
        try await Task.detached(priority: .userInitiated) {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue {
                throw CocoaError(.fileReadUnsupportedScheme)
            }
            let name = UploadIncoming.safeName(url.lastPathComponent)
            let target = UploadIncoming.newURL(for: name)
            try FileManager.default.copyItem(at: url, to: target)
            return UploadIncoming(url: target, name: name)
        }.value
    }

    /// A camera file name as the website's camera makes them: IMG_20261117_093015.jpg, VID_20261117_093015.mov.
    static func cameraName(prefix: String, ext: String, at date: Date = Date()) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMdd_HHmmss"
        return "\(prefix)_\(f.string(from: date)).\(ext)"
    }
}

/// A photo or video from the Photos picker, or a file dropped on the screen, copied into the app as its file
/// (the picker hands over the current original with `preferredItemEncoding: .current`, so its date and place stay).
struct UploadPickedFile: Transferable {
    let url: URL
    let name: String

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .data) { received in
            let name = received.file.lastPathComponent
            let target = UploadIncoming.newURL(for: name)
            try FileManager.default.copyItem(at: received.file, to: target)
            return UploadPickedFile(url: target, name: UploadIncoming.safeName(name))
        }
    }
}

/// Small pictures of queued files until the server's thumb arrives: made once from the copy (or the library's
/// asset) and kept in Caches.
enum UploadThumbs {
    static var folder: URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let dir = base.appendingPathComponent("UploadThumbs", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func url(_ id: String) -> URL {
        let safe = String(id.map { $0.isLetter || $0.isNumber || $0 == "-" ? $0 : "_" })
        return folder.appendingPathComponent(safe + ".jpg")
    }

    static func image(_ id: String) -> UIImage? {
        UIImage(contentsOfFile: url(id).path)
    }

    static func remove(_ id: String) {
        try? FileManager.default.removeItem(at: url(id))
    }

    /// From a copied file: a picture's own thumbnail, or a frame of a video. Nothing for other files.
    static func make(_ id: String, from file: URL, kind: UploadKind) async {
        let target = url(id)
        await Task.detached(priority: .utility) {
            var cg: CGImage?
            if kind == .image, let src = CGImageSourceCreateWithURL(file as CFURL, nil) {
                let opts: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                             kCGImageSourceCreateThumbnailWithTransform: true,
                                             kCGImageSourceThumbnailMaxPixelSize: 240]
                cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
            } else if kind == .video {
                let generator = AVAssetImageGenerator(asset: AVURLAsset(url: file))
                generator.appliesPreferredTrackTransform = true
                generator.maximumSize = CGSize(width: 240, height: 240)
                cg = try? generator.copyCGImage(at: .zero, actualTime: nil)
            }
            guard let cg, let data = UIImage(cgImage: cg).jpegData(compressionQuality: 0.8) else { return }
            try? data.write(to: target, options: .atomic)
        }.value
    }

    /// From an asset of the library (a photo chosen in the picker that goes the sync's way).
    static func make(_ id: String, asset: PHAsset) async {
        let target = url(id)
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true
        options.resizeMode = .fast
        let image: UIImage? = await withCheckedContinuation { cont in
            PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: 240, height: 240),
                                                  contentMode: .aspectFill, options: options) { image, _ in
                cont.resume(returning: image)
            }
        }
        if let data = image?.jpegData(compressionQuality: 0.8) { try? data.write(to: target, options: .atomic) }
    }
}

/// Photos from the camera: written as JPEG with the camera's own metadata and orientation.
enum UploadCameraFile {
    static func writeJPEG(_ image: UIImage, metadata: [String: Any]?, to url: URL) throws {
        guard let cg = image.cgImage,
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        var props = metadata ?? [:]
        props[kCGImageDestinationLossyCompressionQuality as String] = 0.92
        props[kCGImagePropertyOrientation as String] = orientation(image.imageOrientation).rawValue
        CGImageDestinationAddImage(dest, cg, props as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }

    /// The picture's orientation as an image file says it (UIImage keeps it apart from the pixels).
    static func orientation(_ o: UIImage.Orientation) -> CGImagePropertyOrientation {
        switch o {
        case .up: return .up
        case .down: return .down
        case .left: return .left
        case .right: return .right
        case .upMirrored: return .upMirrored
        case .downMirrored: return .downMirrored
        case .leftMirrored: return .leftMirrored
        case .rightMirrored: return .rightMirrored
        @unknown default: return .up
        }
    }
}
