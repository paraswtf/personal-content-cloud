import SwiftUI
import UIKit
import AVFoundation
import UniformTypeIdentifiers

/// The camera for the Upload screen (the website's CameraCapture): the system camera, photo or video. A photo is
/// kept as a JPEG with the camera's own metadata, a video as the camera recorded it; both join the queue as camera
/// uploads.
enum UploadCamera {
    /// This device has a camera (the simulator has none).
    static var available: Bool { UIImagePickerController.isSourceTypeAvailable(.camera) }

    /// The owner said no to the camera earlier (only the Settings app can change it now).
    static var refused: Bool {
        let s = AVCaptureDevice.authorizationStatus(for: .video)
        return s == .denied || s == .restricted
    }
}

struct UploadCameraPicker: UIViewControllerRepresentable {
    let onCapture: (UploadIncoming) -> Void
    let onError: (String) -> Void
    let onClose: () -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.mediaTypes = [UTType.image.identifier, UTType.movie.identifier]
        picker.videoQuality = .typeHigh
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ picker: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: UploadCameraPicker

        init(_ parent: UploadCameraPicker) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            let now = Date()
            var captured: UploadIncoming?
            var failure: String?
            if let movie = info[.mediaURL] as? URL {
                let ext = movie.pathExtension.isEmpty ? "mov" : movie.pathExtension.lowercased()
                let name = UploadIncoming.cameraName(prefix: "VID", ext: ext, at: now)
                let target = UploadIncoming.newURL(for: name)
                do {
                    try FileManager.default.moveItem(at: movie, to: target)
                    captured = UploadIncoming(url: target, name: name)
                } catch {
                    failure = "The video could not be saved. Try again."
                }
            } else if let image = info[.originalImage] as? UIImage {
                let name = UploadIncoming.cameraName(prefix: "IMG", ext: "jpg", at: now)
                let target = UploadIncoming.newURL(for: name)
                do {
                    try UploadCameraFile.writeJPEG(image, metadata: info[.mediaMetadata] as? [String: Any], to: target)
                    captured = UploadIncoming(url: target, name: name)
                } catch {
                    failure = "The photo could not be saved. Try again."
                }
            }
            let parent = self.parent
            Task { @MainActor in
                if let captured { parent.onCapture(captured) }
                if let failure { parent.onError(failure) }
                parent.onClose()
            }
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            let parent = self.parent
            Task { @MainActor in parent.onClose() }
        }
    }
}
