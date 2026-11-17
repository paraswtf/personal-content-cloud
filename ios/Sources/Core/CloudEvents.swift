import Foundation
import SwiftUI

/// Something changed on the server (a save, an undo, a redo): screens that show it load again.
enum CloudEvents {
    static let changed = Notification.Name("pcc.cloud.changed")

    /// `items` are the uids touched when known; an empty list means "anything may have changed".
    @MainActor
    static func post(items: [String] = []) {
        NotificationCenter.default.post(name: changed, object: nil, userInfo: ["items": items])
    }

    /// The uids a notification names (empty when it names none).
    static func items(of note: Notification) -> [String] {
        note.userInfo?["items"] as? [String] ?? []
    }
}

extension View {
    /// Runs `action` whenever something changed on the server; with `uid`, only when that item is named or
    /// when the change names no items.
    func onCloudChange(uid: String? = nil, perform action: @escaping () -> Void) -> some View {
        onReceive(NotificationCenter.default.publisher(for: CloudEvents.changed)) { note in
            let items = CloudEvents.items(of: note)
            if let uid, !items.isEmpty, !items.contains(uid) { return }
            action()
        }
    }
}
