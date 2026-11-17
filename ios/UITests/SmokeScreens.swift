import XCTest

/// Signing in with a device token, and the phone's sync screens under More, Upload and sync, against the seeded server.
final class SmokeScreens: ScreenTestCase {
    /// The real sign-in: the token typed into the sign-in screen is checked by the server (GET /api/auth/state).
    func testSignInWithDeviceToken() throws {
        try launchSignedOut()
        let field = app.secureTextFields.firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 30), "the sign-in screen did not show")
        snap("sign-in")
        field.tap()
        field.typeText(token)
        app.buttons["Sign in"].tap()
        XCTAssertTrue(app.tabBars.buttons["Library"].waitForExistence(timeout: 30), "the server did not accept the token")
        snap("signed-in")
    }

    func testSyncScreensAndSettings() throws {
        try launchSignedIn()
        tab("More")
        app.buttons["more.upload"].tap()
        XCTAssertTrue(app.navigationBars["Upload and sync"].waitForExistence(timeout: 10))
        answerSystemAlert(timeout: 5)  // opening upload asks for Photos access
        snap("upload")
        app.buttons["upload.sync"].tap()
        XCTAssertTrue(app.navigationBars["Sync"].waitForExistence(timeout: 30))
        snap("sync")
        app.navigationBars["Sync"].buttons.firstMatch.tap()
        app.buttons["upload.syncSettings"].tap()
        XCTAssertTrue(app.navigationBars["Sync settings"].waitForExistence(timeout: 10))
        snap("sync-settings")
        app.navigationBars["Sync settings"].buttons.firstMatch.tap()
        app.buttons["upload.deleteSynced"].tap()
        XCTAssertTrue(app.navigationBars["Delete synced"].waitForExistence(timeout: 10))
        snap("delete-synced")
        tab("More")
        app.buttons["more.settings"].tap()
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 10))
        snap("settings")
        // the cloud's name from the server, beside its address at the bottom
        app.collectionViews.firstMatch.swipeUp()
        XCTAssertTrue(app.staticTexts["Name"].waitForExistence(timeout: 10), "Settings does not name the cloud")
        snap("settings-server")
    }
}
