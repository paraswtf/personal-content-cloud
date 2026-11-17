import XCTest

/// Settings, account side (Features/Settings/Account): Account, Passkeys, Sessions and devices, Wallet and
/// Appearance, against the seeded server. The tests change only what ci/seeds/setacct.py made: the passkey
/// "Setacct test key", the "Firefox on Windows" session and the device "Setacct spare phone", plus a device the test
/// adds itself. They never end "iPhone simulator" (the tests' token), never change the password and never sign out
/// everywhere; the vault may be open or locked for the app (the seed opened it, it closes after 15 minutes), and the
/// tests accept both. The passkey test locks it on purpose to show the unlock sheet, and opens it again there.
final class SettingsAccountScreens: ScreenTestCase {
    /// The seed's throwaway account on a server that lives for one CI run (ci/seed_server.py).
    private var password: String { ProcessInfo.processInfo.environment["PCC_TEST_PASSWORD"] ?? "ci-only-password" }

    /// Locks the vault for the app from the vault screen (it may be locked already), then goes back to More.
    private func lockVault() {
        tab("More")
        let entry = app.buttons["more.vault"]
        XCTAssertTrue(entry.waitForExistence(timeout: 15), "no Vault under More")
        entry.tap()
        let lock = app.buttons["vault.lock"]
        let field = app.secureTextFields["vault.password"]
        var open = false
        for _ in 0..<30 {
            if lock.waitForExistence(timeout: 1) {
                open = true
                break
            }
            if field.exists { break }
        }
        if open { lock.tap() }
        XCTAssertTrue(field.waitForExistence(timeout: 20), "the vault did not lock")
        app.navigationBars["Vault"].buttons.element(boundBy: 0).tap()
        XCTAssertTrue(app.buttons["more.settings"].waitForExistence(timeout: 10), "did not go back to More")
    }

    private func openSettings(_ row: String) {
        tab("More")
        let settings = app.buttons["more.settings"]
        XCTAssertTrue(settings.waitForExistence(timeout: 15), "no Settings under More")
        settings.tap()
        let entry = app.buttons[row]
        XCTAssertTrue(entry.waitForExistence(timeout: 15), "no \(row) in Settings")
        entry.tap()
    }

    /// The button of the confirmation dialog on screen (an action sheet on a phone).
    private func dialogButton(_ title: String) -> XCUIElement {
        let inSheet = app.sheets.buttons[title]
        if inSheet.waitForExistence(timeout: 5) { return inSheet }
        return app.buttons[title].firstMatch
    }

    private func waitGone(_ element: XCUIElement, timeout: TimeInterval = 15) -> Bool {
        let gone = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == false"), object: element)
        return XCTWaiter.wait(for: [gone], timeout: timeout) == .completed
    }

    /// Replaces a field's text: a tap at its right end puts the cursor after the text, then it is deleted.
    private func replaceText(_ field: XCUIElement, with text: String) {
        field.coordinate(withNormalizedOffset: CGVector(dx: 0.97, dy: 0.5)).tap()
        let old = (field.value as? String) ?? ""
        field.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: max(old.count, 24)))
        field.typeText(text)
    }

    func testAccount() throws {
        try launchSignedIn()
        openSettings("settings.account")
        XCTAssertTrue(app.navigationBars["Account"].waitForExistence(timeout: 15))
        let username = app.staticTexts["account.username"]
        XCTAssertTrue(username.waitForExistence(timeout: 15), "no username row")
        let shown = XCTNSPredicateExpectation(predicate: NSPredicate(format: "label == 'owner'"), object: username)
        XCTAssertEqual(XCTWaiter.wait(for: [shown], timeout: 15), .completed, "the username did not show")
        // The change password form carries the account name for password managers, read-only.
        let account = app.textFields["account.usernameField"]
        XCTAssertTrue(account.waitForExistence(timeout: 5), "no username beside the password fields")
        XCTAssertEqual(account.value as? String, "owner", "the form does not carry the account name")
        snap("setacct-account")

        // The form's own checks run before anything is sent: a short new password and a mismatch never reach the
        // server (the test never changes the seeded password).
        let current = app.secureTextFields["account.current"]
        XCTAssertTrue(current.waitForExistence(timeout: 5))
        current.tap()
        current.typeText("not the password")
        let new = app.secureTextFields["account.new"]
        new.tap()
        new.typeText("short")
        let again = app.secureTextFields["account.again"]
        again.tap()
        again.typeText("different\n")
        XCTAssertTrue(app.staticTexts["Use at least 10 characters."].waitForExistence(timeout: 10),
                      "the length check did not show")
        XCTAssertTrue(app.staticTexts["The two passwords do not match."].exists, "the match check did not show")
        snap("setacct-account-checks")

        // Sign out everywhere asks first; the test only looks at the question and cancels.
        let signOutAll = app.buttons["account.signOutAll"]
        XCTAssertTrue(signOutAll.waitForExistence(timeout: 5))
        signOutAll.tap()
        let confirm = dialogButton("Sign out everywhere")
        XCTAssertTrue(confirm.exists, "no confirmation for signing out everywhere")
        snap("setacct-signout-all")
        dialogButton("Cancel").tap()
        XCTAssertTrue(app.navigationBars["Account"].waitForExistence(timeout: 5), "the account screen went away")
    }

    func testPasskeys() throws {
        try launchSignedIn()
        lockVault()
        openSettings("settings.passkeys")
        XCTAssertTrue(app.navigationBars["Passkeys"].waitForExistence(timeout: 15))
        XCTAssertTrue(app.staticTexts["Setacct test key"].waitForExistence(timeout: 20), "the seeded passkey is not listed")
        let website = app.buttons["passkeys.addOnWebsite"].exists ? app.buttons["passkeys.addOnWebsite"]
            : app.links["passkeys.addOnWebsite"]
        XCTAssertTrue(website.exists, "no way to add a passkey on the website")
        snap("setacct-passkeys")

        // Rename needs no vault.
        app.buttons["Rename Setacct test key"].tap()
        let alert = app.alerts["Rename passkey"]
        XCTAssertTrue(alert.waitForExistence(timeout: 10), "the rename question did not show")
        replaceText(alert.textFields.firstMatch, with: "Setacct renamed key")
        snap("setacct-passkey-rename")
        alert.buttons["Save"].tap()
        XCTAssertTrue(app.staticTexts["Setacct renamed key"].waitForExistence(timeout: 20), "the new name did not show")

        // Removing needs the vault open for the app. The test locked it, so the unlock sheet asks first and says
        // what it is for (the website's "Confirm it is you" dialog); the password opens the vault and the removal
        // goes through. The vault stays open for the tests after this one.
        app.buttons["Remove Setacct renamed key"].tap()
        let remove = dialogButton("Remove")
        XCTAssertTrue(remove.exists, "no confirmation for removing the passkey")
        remove.tap()
        XCTAssertTrue(app.navigationBars["Confirm it is you"].waitForExistence(timeout: 15),
                      "the unlock sheet did not show")
        let reason = "To remove \"Setacct renamed key\", unlock the vault with your password."
        XCTAssertTrue(app.staticTexts[reason].exists, "the unlock sheet does not say what it is for")
        snap("setacct-passkey-vault")
        let field = app.secureTextFields["vault.password"]
        XCTAssertTrue(field.waitForExistence(timeout: 5), "no password field on the unlock sheet")
        field.tap()
        field.typeText(password + "\n")
        XCTAssertTrue(app.staticTexts["No passkeys yet."].waitForExistence(timeout: 20),
                      "the passkey was not removed after the vault opened")
        snap("setacct-passkey-removed")
    }

    func testSessionsAndDevices() throws {
        try launchSignedIn()
        openSettings("settings.sessions")
        XCTAssertTrue(app.navigationBars["Sessions and devices"].waitForExistence(timeout: 15))
        XCTAssertTrue(app.staticTexts["Firefox on Windows"].waitForExistence(timeout: 20), "the seeded browser is not listed")
        XCTAssertTrue(app.staticTexts["iPhone simulator"].exists, "the app's own device is not listed")
        XCTAssertTrue(app.staticTexts["This device"].exists, "this phone is not marked")
        snap("setacct-sessions")

        // Revoke the seeded browser session.
        app.buttons["Revoke Firefox on Windows"].tap()
        dialogButton("Revoke").tap()
        XCTAssertTrue(waitGone(app.staticTexts["Firefox on Windows"]), "the revoked session is still listed")

        // Add a device of the test's own: its token shows once (vault open), or the unlock sheet (vault locked).
        let name = app.textFields["sessions.deviceName"]
        XCTAssertTrue(name.waitForExistence(timeout: 5))
        replaceText(name, with: "Setacct test phone\n")
        let tokenScreen = app.navigationBars["Device token"]
        if tokenScreen.waitForExistence(timeout: 15) {
            XCTAssertTrue(app.staticTexts["sessions.token"].exists, "no token on the token sheet")
            snap("setacct-token")
            app.buttons["sessions.tokenDone"].tap()
            let added = app.buttons["End Setacct test phone"]
            XCTAssertTrue(added.waitForExistence(timeout: 15), "the new device is not listed")
            added.tap()
            dialogButton("End").tap()
            XCTAssertTrue(waitGone(app.staticTexts["Setacct test phone"]), "the ended device is still listed")
        } else {
            let cancel = app.buttons["Cancel"].firstMatch
            XCTAssertTrue(cancel.waitForExistence(timeout: 10), "neither a token nor the unlock sheet showed")
            XCTAssertTrue(app.navigationBars["Confirm it is you"].exists, "the unlock sheet has no title")
            XCTAssertTrue(app.staticTexts["To add a device, unlock the vault with your password."].exists,
                          "the unlock sheet does not say what it is for")
            snap("setacct-device-vault")
            cancel.tap()
        }

        // End the seeded spare device.
        let spare = app.buttons["End Setacct spare phone"]
        XCTAssertTrue(spare.waitForExistence(timeout: 15), "the seeded device is not listed")
        spare.tap()
        dialogButton("End").tap()
        XCTAssertTrue(waitGone(app.staticTexts["Setacct spare phone"]), "the ended device is still listed")
        XCTAssertTrue(app.staticTexts["iPhone simulator"].exists, "the app's own device must stay")
        snap("setacct-devices")
    }

    func testWallet() throws {
        try launchSignedIn()
        openSettings("settings.wallet")
        XCTAssertTrue(app.navigationBars["Wallet"].waitForExistence(timeout: 15))
        let status = app.staticTexts["wallet.status"]
        XCTAssertTrue(status.waitForExistence(timeout: 20), "the wallet status did not load")
        XCTAssertEqual(status.label, "Not set up", "the CI server has no wallet set up")
        XCTAssertTrue(app.staticTexts["WALLET_PASS_TYPE_ID"].exists, "the missing settings are not listed")
        snap("setacct-wallet")
    }

    func testAppearance() throws {
        try launchSignedIn()
        openSettings("settings.appearance")
        XCTAssertTrue(app.navigationBars["Appearance"].waitForExistence(timeout: 15))
        let dark = app.buttons["appearance.dark"]
        XCTAssertTrue(dark.waitForExistence(timeout: 5))
        dark.tap()
        XCTAssertTrue(dark.isSelected, "Dark is not marked as chosen")
        snap("setacct-appearance-dark")
        // Back to System, so the tests after this one see the phone's own setting.
        let system = app.buttons["appearance.system"]
        system.tap()
        XCTAssertTrue(system.isSelected, "System is not marked as chosen")
        snap("setacct-appearance")
    }
}
