#if DEBUG
import Foundation

/// How the screen tests on the CI machine hand files to the upload queue (ios/UITests/UploadScreens.swift): the
/// test cannot use the Photos picker or Files, so it passes small files in the launch environment and taps "Add the
/// test files", which queues them exactly as files chosen in Files are queued. Debug builds only.
///
/// PCC_TEST_UPLOADS: a JSON list of {"name": "...", "base64": "..."}.
enum UploadTestHook {
    struct File: Decodable {
        let name: String
        let base64: String
    }

    static var files: [File]? {
        guard let text = ProcessInfo.processInfo.environment["PCC_TEST_UPLOADS"], let data = text.data(using: .utf8) else {
            return nil
        }
        return try? JSONDecoder().decode([File].self, from: data)
    }

    @MainActor
    static func add() {
        guard let files else { return }
        var copies: [UploadIncoming] = []
        for file in files {
            let url = UploadIncoming.newURL(for: file.name)
            guard let data = Data(base64Encoded: file.base64), (try? data.write(to: url)) != nil else { continue }
            copies.append(UploadIncoming(url: url, name: file.name))
        }
        UploadQueue.shared.add(copies, origin: "file")
    }
}
#endif
