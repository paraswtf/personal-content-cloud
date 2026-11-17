import XCTest

/// Base class of the screen tests (ios/README.md, "Tests on GitHub").
///
/// On the CI machine the workflow seeds a real backend and passes its address and a device token to the
/// test runner as TEST_RUNNER_PCC_TEST_SERVER and TEST_RUNNER_PCC_TEST_TOKEN (xcodebuild strips the
/// prefix). `launchSignedIn()` hands them to the app, which signs in with them in Debug builds
/// (Sources/TestLaunch.swift). `snap(_:)` keeps a screenshot that the workflow exports as
/// `<name>.png` into the "screenshots" artifact.
///
/// A new screen test is a few lines:
///
///     final class AlbumsScreens: ScreenTestCase {
///         func testAlbums() throws {
///             try launchSignedIn()
///             tab("Albums")
///             XCTAssertTrue(app.staticTexts["Weekend trip"].waitForExistence(timeout: 15))
///             snap("albums")
///         }
///     }
class ScreenTestCase: XCTestCase {
    var app: XCUIApplication!

    /// The seeded server's address, for example http://127.0.0.1:8765.
    var server: String { ProcessInfo.processInfo.environment["PCC_TEST_SERVER"] ?? "" }
    /// A device token for the seeded server's owner account.
    var token: String { ProcessInfo.processInfo.environment["PCC_TEST_TOKEN"] ?? "" }

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        // The workflow grants Photos access before the tests; this answers the prompt if it shows anyway
        // in the middle of a test (answerSystemAlert covers the one at launch).
        addUIInterruptionMonitor(withDescription: "Photos access") { alert in
            for label in ScreenTestCase.allowLabels where alert.buttons[label].exists {
                alert.buttons[label].tap()
                return true
            }
            return false
        }
    }

    /// The buttons that grant a permission prompt (the simulator runs in English).
    static let allowLabels = ["Allow Full Access", "Allow Access to All Photos", "Allow", "OK"]

    /// Answers a permission prompt that shows over the app, so the next screenshot shows the app. The app
    /// asks for Photos access as it starts (BackgroundSync registers a library observer).
    func answerSystemAlert(timeout: TimeInterval = 2) {
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let allow = springboard.buttons.matching(NSPredicate(format: "label IN %@", ScreenTestCase.allowLabels)).firstMatch
        if allow.waitForExistence(timeout: timeout) { allow.tap() }
    }

    /// Starts the app signed in to the seeded server. `environment` adds launch variables for one test.
    func launchSignedIn(_ environment: [String: String] = [:]) throws {
        try requireServer()
        app.launchEnvironment["PCC_TEST_SERVER"] = server
        app.launchEnvironment["PCC_TEST_TOKEN"] = token
        for (key, value) in environment { app.launchEnvironment[key] = value }
        app.launch()
        answerSystemAlert()
    }

    /// Starts the app on its sign-in screen, with the seeded server's address filled in.
    func launchSignedOut() throws {
        try requireServer()
        app.launchEnvironment["PCC_TEST_SERVER"] = server
        app.launchEnvironment["PCC_TEST_SIGNED_OUT"] = "1"
        app.launch()
        answerSystemAlert()
    }

    /// Attaches a screenshot of the app, kept even when the test passes. Names are file names: keep
    /// them unique across tests ("albums", "album-detail").
    func snap(_ name: String) {
        let shot = XCTAttachment(screenshot: app.screenshot())
        shot.name = name
        shot.lifetime = .keepAlways
        add(shot)
    }

    /// Taps a tab bar button once it is there.
    func tab(_ label: String, file: StaticString = #filePath, line: UInt = #line) {
        let button = app.tabBars.buttons[label]
        XCTAssertTrue(button.waitForExistence(timeout: 20), "no tab named \(label)", file: file, line: line)
        button.tap()
    }

    private func requireServer() throws {
        if server.isEmpty || token.isEmpty { throw NoTestServer() }
    }
}

/// A screen test run without the seeded server fails (it is not skipped): on the CI machine that means the
/// workflow lost the server, which must not pass unnoticed.
struct NoTestServer: LocalizedError {
    var errorDescription: String? {
        "PCC_TEST_SERVER and PCC_TEST_TOKEN are not set: run the tests through the iOS workflow, or pass "
            + "TEST_RUNNER_PCC_TEST_SERVER and TEST_RUNNER_PCC_TEST_TOKEN to xcodebuild"
    }
}
