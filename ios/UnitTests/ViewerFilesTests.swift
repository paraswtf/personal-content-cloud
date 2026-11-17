import XCTest
@testable import PersonalContentCloud

/// The viewer's downloads (originals, frames, pages) do not stay on the phone once their sheet is done.
@MainActor
final class ViewerFilesTests: XCTestCase {
    func testDiscardRemovesTheDownloadAndItsFolder() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("viewer-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let file = folder.appendingPathComponent("IMG_0001.JPG")
        try Data([1, 2, 3]).write(to: file)

        ViewerFiles.discard(ViewerFile(url: file, isVideo: false, isImage: true))

        XCTAssertFalse(FileManager.default.fileExists(atPath: folder.path))
    }

    func testDiscardLeavesAnyOtherFolderAlone() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("other-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appendingPathComponent("keep.txt")
        try Data([1]).write(to: file)

        ViewerFiles.discard(ViewerFile(url: file, isVideo: false, isImage: false))

        XCTAssertTrue(FileManager.default.fileExists(atPath: file.path))
    }
}
