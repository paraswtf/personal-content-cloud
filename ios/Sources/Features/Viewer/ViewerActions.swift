import SwiftUI
import UIKit
import Photos
import PassKit
import QuickLook

/// The viewer's owner actions on one item: in the vault when the viewer was opened from it, through the undo history
/// with the website's message (and Undo when the server recorded a change), then the item is read again and other
/// screens hear of the change.
@MainActor
enum ViewerActions {
    /// Runs one action. `label` is the message on success; `doing` completes "Could not ..." on failure. With
    /// `alwaysSay`, the label shows even when the server kept nothing to undo (starting a job, for example).
    @discardableResult
    static func run(_ context: ViewerPageContext, _ label: String, doing: String, alwaysSay: Bool = false,
                    _ body: (APIClient) async throws -> Void) async -> Bool {
        do {
            try await APIContext.$vault.withValue(context.vault) {
                try await UndoCenter.shared.perform(label, toast: !alwaysSay) {
                    try await body(try Cloud.api())
                }
            }
        } catch {
            Toaster.shared.error(error, doing: doing)
            return false
        }
        if alwaysSay { Toaster.shared.ok(label) }
        context.reload()
        CloudEvents.post(items: [context.uid])
        return true
    }

    /// The item read again as the core reads it (both the core's shape and the generated one), for the panels that
    /// outlive the context they were opened with.
    static func reload(_ context: ViewerPageContext) async -> ViewerPageContext? {
        do {
            let data = try await APIContext.$vault.withValue(context.vault) {
                try await Cloud.api().performData("GET", "/api/items/\(context.uid)")
            }
            let detail = try CoreHTTP.decoder.decode(CoreItemDetail.self, from: data)
            let item = try? JSONDecoder().decode(API.ItemDetail.self, from: data)
            return ViewerPageContext(uid: context.uid, detail: detail, item: item, json: data, reload: context.reload,
                                     vault: context.vault)
        } catch {
            return nil
        }
    }

    /// The url version of an item's files (`?v=`): its version, with an "n" for a vault item.
    static func fileVersion(_ item: API.ItemDetail) -> String {
        "\(item.version)\(item.nsfw ? "n" : "")"
    }
}

/// A file the viewer fetched for the owner (the original, a frame, a page, the Live Photo clip), kept in a temporary
/// folder under its own name while the share sheet, Quick Look or Photos use it.
struct ViewerFile: Identifiable {
    let id = UUID()
    let url: URL
    let isVideo: Bool
    let isImage: Bool
}

@MainActor
enum ViewerFiles {
    /// Downloads a media address of the item (or a prepared request) into a temporary file named `filename`.
    static func fetch(_ context: ViewerPageContext, address: String? = nil, request: URLRequest? = nil,
                      filename: String) async throws -> ViewerFile {
        let req: URLRequest
        if let request {
            req = request
        } else {
            guard let address else { throw APIError(status: 0, detail: "Nothing to download") }
            req = try APIContext.$vault.withValue(context.vault) { try Cloud.api().mediaRequest(address, vault: context.vault) }
        }
        let (tmp, response) = try await URLSession.shared.download(for: req)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw APIError(status: http.statusCode, detail: "The server answered \(http.statusCode)")
        }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("viewer-\(UUID().uuidString)",
                                                                                     isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let safe = filename.replacingOccurrences(of: "/", with: "-")
        let target = folder.appendingPathComponent(safe.isEmpty ? "download" : safe)
        try FileManager.default.moveItem(at: tmp, to: target)
        let ext = target.pathExtension.lowercased()
        let videoExts: Set<String> = ["mov", "mp4", "m4v", "avi", "mkv", "webm", "3gp"]
        let imageExts: Set<String> = ["jpg", "jpeg", "png", "heic", "heif", "webp", "gif", "tif", "tiff", "dng", "bmp"]
        return ViewerFile(url: target, isVideo: videoExts.contains(ext), isImage: imageExts.contains(ext))
    }

    /// Saves a photo or video file to the phone's library (add-only access, asked for here when needed).
    static func saveToPhotos(_ file: ViewerFile) async throws {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            throw APIError(status: 0, detail: "Content Cloud may not add to your photos. Allow it in the Settings app")
        }
        try await PHPhotoLibrary.shared().performChanges {
            let request = PHAssetCreationRequest.forAsset()
            request.addResource(with: file.isVideo ? .video : .photo, fileURL: file.url, options: nil)
        }
    }
}

/// The system share sheet for files (Save Image, Save to Files, AirDrop, Messages, ...).
struct ViewerShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}

/// Quick Look for a fetched file (a PDF with its search and zoom, an office file, a picture).
struct ViewerQuickLook: UIViewControllerRepresentable {
    let url: URL

    func makeCoordinator() -> Coordinator { Coordinator(url: url) }

    func makeUIViewController(context: Context) -> UINavigationController {
        let controller = QLPreviewController()
        controller.dataSource = context.coordinator
        return UINavigationController(rootViewController: controller)
    }

    func updateUIViewController(_ controller: UINavigationController, context: Context) {}

    final class Coordinator: NSObject, QLPreviewControllerDataSource {
        let url: URL
        init(url: URL) { self.url = url }
        func numberOfPreviewItems(in controller: QLPreviewController) -> Int { 1 }
        func previewController(_ controller: QLPreviewController, previewItemAt index: Int) -> QLPreviewItem {
            url as NSURL
        }
    }
}

/// Apple Wallet's own sheet for a pass the server made.
struct ViewerAddPassSheet: UIViewControllerRepresentable {
    let pass: PKPass

    func makeUIViewController(context: Context) -> UIViewController {
        PKAddPassesViewController(pass: pass) ?? UIViewController()
    }

    func updateUIViewController(_ controller: UIViewController, context: Context) {}
}

/// A pass ready for the Wallet sheet.
struct ViewerPass: Identifiable {
    let id = UUID()
    let pass: PKPass
}
