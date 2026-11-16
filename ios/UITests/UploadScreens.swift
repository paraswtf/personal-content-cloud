import XCTest
import UIKit

/// The upload queue against the seeded server: a drawn picture is sent through the background upload session and
/// stored, an empty file is skipped on the phone, the same files again are already in the list, and once the list
/// is cleared the picture comes back as already in your library and opens in the viewer. The files reach the app in
/// the launch environment (Sources/Features/Upload/UploadTestHook.swift), as the test cannot drive Files.
final class UploadScreens: ScreenTestCase {
    /// A small PNG with noise in it, different on every run (the server keeps one copy of identical files).
    private static func drawnPNG() -> Data {
        let size = CGSize(width: 96, height: 96)
        let renderer = UIGraphicsImageRenderer(size: size)
        var seed = UInt64(Date().timeIntervalSince1970 * 1000) | 1
        let image = renderer.image { ctx in
            UIColor(red: 0.35, green: 0.62, blue: 0.86, alpha: 1).setFill()
            ctx.fill(CGRect(origin: .zero, size: size))
            for y in stride(from: 0, to: 96, by: 4) {
                for x in stride(from: 0, to: 96, by: 4) {
                    seed = seed &* 6364136223846793005 &+ 1442695040888963407
                    let v = CGFloat((seed >> 33) % 255) / 255
                    UIColor(red: v, green: 0.5 * v + 0.3, blue: 1 - v, alpha: 1).setFill()
                    ctx.fill(CGRect(x: x, y: y, width: 4, height: 4))
                }
            }
        }
        return image.pngData() ?? Data()
    }

    /// Waits for an element of the queue, scrolling the list up once if it is below the screen.
    private func waitInList(_ element: XCUIElement, timeout: TimeInterval) -> Bool {
        if element.waitForExistence(timeout: timeout / 2) { return true }
        app.collectionViews.firstMatch.swipeUp()
        return element.waitForExistence(timeout: timeout / 2)
    }

    /// Waits until the element's label reads `text`.
    private func waitForLabel(_ element: XCUIElement, _ text: String, timeout: TimeInterval) -> Bool {
        let done = expectation(for: NSPredicate(format: "label == %@", text), evaluatedWith: element)
        return XCTWaiter().wait(for: [done], timeout: timeout) == .completed
    }

    func testUploadQueue() throws {
        let files: [[String: String]] = [
            ["name": "upload-test.png", "base64": Self.drawnPNG().base64EncodedString()],
            ["name": "upload-empty.txt", "base64": ""],
        ]
        let json = try XCTUnwrap(String(data: try JSONSerialization.data(withJSONObject: files), encoding: .utf8))
        try launchSignedIn(["PCC_TEST_UPLOADS": json])
        tab("More")
        app.buttons["more.upload"].tap()
        XCTAssertTrue(app.navigationBars["Upload and sync"].waitForExistence(timeout: 10))
        answerSystemAlert(timeout: 5)
        XCTAssertTrue(app.buttons["upload.files"].waitForExistence(timeout: 10))
        snap("upload-screen")

        // The picture is sent and stored; the empty file is skipped on the phone.
        app.buttons["upload.testFiles"].tap()
        let notice = app.staticTexts["upload.notice"]
        XCTAssertTrue(waitForLabel(notice, "1 file added to the queue. 1 file was skipped because it is empty or too large.",
                                   timeout: 10), "the notice said: \(notice.exists ? notice.label : "nothing")")
        XCTAssertTrue(waitInList(app.staticTexts["Skipped"], timeout: 20), "the empty file was not skipped")
        XCTAssertTrue(waitInList(app.staticTexts["Added"], timeout: 150), "the picture was not stored")
        XCTAssertTrue(app.staticTexts["This file is empty."].exists)
        snap("upload-added")

        // The same files again: both are already in the list.
        app.collectionViews.firstMatch.swipeDown()
        app.buttons["upload.testFiles"].tap()
        XCTAssertTrue(waitForLabel(notice, "2 files are already in the list.", timeout: 10),
                      "the notice said: \(notice.exists ? notice.label : "nothing")")

        // Cleared, then the picture again: the server has it already.
        let clear = app.buttons["upload.clearFinished"]
        XCTAssertTrue(waitInList(clear, timeout: 10))
        clear.tap()
        XCTAssertFalse(app.staticTexts["Added"].waitForExistence(timeout: 3))
        app.collectionViews.firstMatch.swipeDown()
        app.buttons["upload.testFiles"].tap()
        XCTAssertTrue(waitInList(app.staticTexts["Already in your library"], timeout: 90), "the picture was not found in the library")
        snap("upload-duplicate")

        // View opens the stored picture in the viewer.
        let view = app.buttons.matching(identifier: "upload.view").firstMatch
        XCTAssertTrue(view.waitForExistence(timeout: 10))
        view.tap()
        XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 20), "the viewer did not open")
        snap("upload-viewer")
        app.buttons["viewer.close"].tap()

        // Clear finished leaves the queue empty for the other tests.
        XCTAssertTrue(waitInList(clear, timeout: 10))
        clear.tap()
        XCTAssertFalse(app.buttons["upload.clearFinished"].waitForExistence(timeout: 3))
    }
}
