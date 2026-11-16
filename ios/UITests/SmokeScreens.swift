import XCTest

/// The screens the app has today, against the seeded server.
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
        XCTAssertTrue(app.tabBars.buttons["Sync"].waitForExistence(timeout: 30), "the server did not accept the token")
        snap("signed-in")
    }

    func testSyncAndSettings() throws {
        try launchSignedIn()
        XCTAssertTrue(app.navigationBars["Sync"].waitForExistence(timeout: 30))
        snap("sync")
        tab("Settings")
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 10))
        snap("settings")
        tab("Delete synced")
        snap("delete-synced")
    }
}
