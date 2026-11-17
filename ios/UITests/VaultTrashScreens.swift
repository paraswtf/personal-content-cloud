import XCTest

/// The vault and the trash against the seeded server (ci/seeds/vaulttrash.py): the vault opens for the app, locks,
/// refuses a wrong password, opens again with the right one and stays open for the other areas' tests; the trash
/// restores and deletes for good only the photos this area's seed put there, shows one larger, and asks before
/// emptying (the test never empties it: other areas' items are there too).
final class VaultTrashScreens: ScreenTestCase {
    /// The seed's throwaway account on a server that lives for one CI run (ci/seed_server.py).
    private var password: String { ProcessInfo.processInfo.environment["PCC_TEST_PASSWORD"] ?? "ci-only-password" }

    func testVaultLocksAndOpensAgain() throws {
        try launchSignedIn()
        openMore("more.vault")
        let lock = app.buttons["vault.lock"]
        let locked = text("The vault is locked")
        // The seed opens the vault for the app; when its window has ended meanwhile, open it first.
        let first = waitForEither(lock, locked, timeout: 30)
        XCTAssertNotNil(first, "the vault screen showed neither state")
        if first === locked { unlock(with: password) }
        XCTAssertTrue(lock.waitForExistence(timeout: 20), "the vault did not open")
        let status = app.descendants(matching: .any).matching(identifier: "vault.status").firstMatch
        XCTAssertTrue(status.waitForExistence(timeout: 10), "no time left shows")
        XCTAssertTrue(status.label.hasPrefix("Unlocked"), "the time left reads \(status.label)")
        XCTAssertTrue(app.buttons.matching(identifier: "media.cell").firstMatch.waitForExistence(timeout: 30),
                      "the vault's photo did not show")
        snap("vaulttrash-vault-open")

        lock.tap()
        // the message first: it stays 5 seconds, and waiting for the locked screen first could outlast it
        XCTAssertTrue(toastShows("Vault locked"), "no message after Lock now")
        XCTAssertTrue(locked.waitForExistence(timeout: 20), "the vault did not lock")
        XCTAssertFalse(app.buttons.matching(identifier: "media.cell").firstMatch.exists, "vault items show while locked")
        XCTAssertTrue(app.secureTextFields["vault.password"].exists)
        snap("vaulttrash-vault-locked")

        unlock(with: "not-the-password")
        XCTAssertTrue(text("That password is not right.").waitForExistence(timeout: 20), "a wrong password was not refused")
        XCTAssertFalse(lock.exists)
        snap("vaulttrash-vault-wrong-password")

        unlock(with: password)
        XCTAssertTrue(lock.waitForExistence(timeout: 20), "the vault did not open with the password")
        XCTAssertTrue(app.buttons.matching(identifier: "media.cell").firstMatch.waitForExistence(timeout: 30))
        snap("vaulttrash-vault-open-again")
        // left open: Settings and the other areas' tests find it so
    }

    func testTrashRestoresDeletesAndShowsLarger() throws {
        try launchSignedIn()
        openMore("more.trash")
        let intro = app.staticTexts["trash.intro"]
        XCTAssertTrue(intro.waitForExistence(timeout: 30), "the trash did not load")
        XCTAssertTrue(intro.label.hasPrefix("Deleted items stay here for 30 days"), "the intro reads \(intro.label)")
        let restore = button("Restore VAULTTRASH_RESTORE.JPG")
        XCTAssertTrue(reveal(restore), "the seeded trash photo is not listed")
        snap("vaulttrash-trash")

        restore.tap()
        XCTAssertTrue(toastShows("VAULTTRASH_RESTORE.JPG is back in the library."), "no message after Restore")
        XCTAssertTrue(restore.waitForNonExistence(timeout: 20), "the restored photo stayed in the trash")
        snap("vaulttrash-trash-restored")

        let purge = button("Delete VAULTTRASH_PURGE.JPG forever")
        XCTAssertTrue(reveal(purge), "the photo to delete is not listed")
        purge.tap()
        let confirm = dialogButton("Delete forever")
        XCTAssertTrue(confirm.waitForExistence(timeout: 10), "Delete forever did not ask first")
        snap("vaulttrash-trash-confirm-delete")
        confirm.tap()
        XCTAssertTrue(purge.waitForNonExistence(timeout: 20), "the deleted photo stayed in the trash")
        XCTAssertTrue(toastShows("Deleted for good."), "no message after Delete forever")

        let open = button("Show VAULTTRASH_KEEP.JPG larger")
        XCTAssertTrue(reveal(open), "the photo to preview is not listed")
        open.tap()
        let close = app.buttons["trash.preview.close"]
        XCTAssertTrue(close.waitForExistence(timeout: 15), "the larger look did not open")
        XCTAssertTrue(app.buttons["trash.preview.restore"].waitForExistence(timeout: 10))
        XCTAssertEqual(app.buttons["trash.preview.restore"].label, "Restore VAULTTRASH_KEEP.JPG")
        snap("vaulttrash-trash-preview")
        close.tap()
        XCTAssertTrue(close.waitForNonExistence(timeout: 10))

        // Empty trash asks first; Cancel leaves everything (the core's and other areas' trashed items are here too).
        let empty = app.buttons["trash.empty"]
        XCTAssertTrue(empty.waitForExistence(timeout: 10))
        empty.tap()
        let cancel = dialogButton("Cancel")
        XCTAssertTrue(cancel.waitForExistence(timeout: 10), "Empty trash did not ask first")
        snap("vaulttrash-trash-confirm-empty")
        cancel.tap()
        XCTAssertTrue(button("Show VAULTTRASH_KEEP.JPG larger").waitForExistence(timeout: 10),
                      "the trash changed after Cancel")
    }

    // MARK: helpers

    private func openMore(_ identifier: String) {
        tab("More")
        let row = app.buttons[identifier]
        XCTAssertTrue(row.waitForExistence(timeout: 20), "no \(identifier) row")
        row.tap()
    }

    /// Types into the unlock form and sends it with Return (the keyboard may cover the Unlock button).
    private func unlock(with secret: String) {
        let field = app.secureTextFields["vault.password"]
        XCTAssertTrue(field.waitForExistence(timeout: 20), "the unlock form did not show")
        field.tap()
        field.typeText(secret + "\n")
    }

    private func button(_ label: String) -> XCUIElement {
        app.buttons.matching(NSPredicate(format: "label == %@", label)).firstMatch
    }

    private func text(_ label: String) -> XCUIElement {
        app.staticTexts.matching(NSPredicate(format: "label == %@", label)).firstMatch
    }

    /// The app's message bar shows a text (it reads as one element with its Undo button).
    private func toastShows(_ text: String, timeout: TimeInterval = 20) -> Bool {
        app.descendants(matching: .any)
            .matching(NSPredicate(format: "identifier == 'toast' AND label CONTAINS %@", text))
            .firstMatch
            .waitForExistence(timeout: timeout)
    }

    /// The first of two elements to appear.
    private func waitForEither(_ a: XCUIElement, _ b: XCUIElement, timeout: TimeInterval) -> XCUIElement? {
        let end = Date().addingTimeInterval(timeout)
        while Date() < end {
            if a.exists { return a }
            if b.exists { return b }
            RunLoop.current.run(until: Date().addingTimeInterval(0.25))
        }
        return nil
    }

    /// Scrolls the trash until an element is on the screen (the grid makes its tiles as they come into view).
    private func reveal(_ element: XCUIElement) -> Bool {
        if element.waitForExistence(timeout: 10), element.isHittable { return true }
        for _ in 0..<8 {
            app.swipeUp()
            if element.exists, element.isHittable { return true }
        }
        return element.exists
    }

    /// A button of the confirmation dialog on screen (an action sheet on the phone).
    private func dialogButton(_ label: String) -> XCUIElement {
        let inSheet = app.sheets.buttons.matching(NSPredicate(format: "label == %@", label)).firstMatch
        if inSheet.waitForExistence(timeout: 5) { return inSheet }
        return app.buttons.matching(NSPredicate(format: "label == %@", label)).firstMatch
    }
}
