#if DEBUG
import Foundation

/// The undo history's screen test (UITests/SettingsSystemScreens.swift) needs a change this phone made, and the
/// system screens make none that can be undone (server jobs are not library edits). With the launch variable
/// PCC_TEST_SETSYS_UNDO=1, in Debug builds only, opening Undo history once per launch gives the album
/// "Setsys undo test" (made by ios/ci/seeds/setsys.py, nobody else's data) a new description through
/// UndoCenter, like any owner action, so the history lists it and the test can undo and redo it.
@MainActor
enum SettingsSystemTestHook {
    static let albumName = "Setsys undo test"
    static let label = "Changed the description of the album Setsys undo test"
    private static var done = false

    static func makeUndoableChangeIfAsked() async {
        guard !done, ProcessInfo.processInfo.environment["PCC_TEST_SETSYS_UNDO"] == "1" else { return }
        done = true
        do {
            let api = try Cloud.api()
            var album = try await api.listAlbums(q: albumName, limit: 50).items.first { $0.name == albumName }
            if album == nil {
                album = try await api.listAlbums(limit: 200).items.first { $0.name == albumName }
            }
            guard let album else {
                Toaster.shared.error("The test album \(albumName) is not on the server.")
                return
            }
            let stamp = Date().formatted(.iso8601)
            try await UndoCenter.shared.perform(label, toast: false) {
                _ = try await api.updateAlbum(albumId: album.id,
                                              body: API.AlbumUpdate(description: "Changed by the screen test at \(stamp)"))
            }
        } catch {
            Toaster.shared.error(error, doing: "make the test change")
        }
    }
}
#endif
