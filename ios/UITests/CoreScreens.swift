import XCTest

/// The core's screens against the seeded server (ios/ci/seed_server.py): the tabs, the media grid, the viewer with
/// overlapping boxes where the smaller box gets the tap, the ranked person picker, and signing in with the password.
final class CoreScreens: ScreenTestCase {
    /// The first cell of the library grid (the seed's text note, the newest item).
    private var firstCell: XCUIElement { app.buttons.matching(identifier: "media.cell").firstMatch }

    /// The seed's photo with overlapping boxes, taken on 10 Nov 2026 (ci/seed_server.py), found by what VoiceOver
    /// reads for its cell.
    private var boxesCell: XCUIElement {
        app.buttons.matching(NSPredicate(format: "identifier == 'media.cell' AND label BEGINSWITH 'Photo, Nov 10, 2026'"))
            .firstMatch
    }

    func testTheTabsTheGridAndMore() throws {
        try launchSignedIn()
        XCTAssertTrue(firstCell.waitForExistence(timeout: 30), "the library grid did not show")
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        XCTAssertFalse(springboard.alerts.firstMatch.exists, "a permission prompt showed at launch")
        snap("core-library")
        for (label, name) in [("People", "core-people"), ("Albums", "core-albums"), ("Review", "core-review")] {
            tab(label)
            snap(name)
        }
        tab("More")
        XCTAssertTrue(app.navigationBars["More"].waitForExistence(timeout: 10))
        snap("core-more")
        // The phone's sync lives under More, Upload and sync, which is where the app first asks for Photos access
        // (never at launch: the library above showed no prompt).
        app.buttons["more.upload"].tap()
        XCTAssertTrue(app.navigationBars["Upload and sync"].waitForExistence(timeout: 10))
        answerSystemAlert(timeout: 5)
        app.buttons["upload.sync"].tap()
        XCTAssertTrue(app.navigationBars["Sync"].waitForExistence(timeout: 10))
        snap("core-sync")
        // Tapping the selected tab again goes back to its first screen.
        tab("More")
        XCTAssertTrue(app.navigationBars["More"].waitForExistence(timeout: 10))
    }

    /// What the shell shows on every screen: how many items wait in Review (a badge on its tab), and, while the vault
    /// is open (the seed opens it for the whole run), a bar with the time left that opens the vault. More groups its
    /// screens under headings and offers the theme. The bar's Lock is never tapped: the vault stays open for the others.
    func testTheShellShowsWhatWaitsAndTheOpenVault() throws {
        try launchSignedIn()
        XCTAssertTrue(firstCell.waitForExistence(timeout: 30), "the library grid did not show")
        let review = app.tabBars.buttons["Review"]
        XCTAssertTrue(review.waitForExistence(timeout: 10))
        let counted = NSPredicate { object, _ in
            guard let tab = object as? XCUIElement else { return false }
            let text = "\(tab.label) \(tab.value as? String ?? "")"
            return text.rangeOfCharacter(from: .decimalDigits) != nil
        }
        let badge = XCTNSPredicateExpectation(predicate: counted, object: review)
        XCTAssertEqual(XCTWaiter().wait(for: [badge], timeout: 30), .completed,
                       "the Review tab shows no count: \(review.label) \(review.value as? String ?? "")")

        let vault = app.buttons["shell.vault"]
        XCTAssertTrue(vault.waitForExistence(timeout: 30), "no vault bar while the vault is open")
        XCTAssertTrue(vault.label.hasPrefix("Vault unlocked, "), "the vault bar reads \(vault.label)")
        XCTAssertTrue(app.buttons["shell.vaultLock"].exists, "the vault bar has no Lock")
        snap("core-shell-vault-bar")
        vault.tap()
        XCTAssertTrue(app.navigationBars["Vault"].waitForExistence(timeout: 10), "the vault bar did not open the vault")
        snap("core-shell-vault-opened")

        tab("More")
        XCTAssertTrue(app.navigationBars["More"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["BROWSE"].exists || app.staticTexts["Browse"].exists, "More has no headings")
        let vaultEntry = app.buttons["more.vault"]
        XCTAssertTrue(vaultEntry.waitForExistence(timeout: 10))
        XCTAssertTrue(vaultEntry.label.contains("Open now"), "the vault entry reads \(vaultEntry.label)")
        XCTAssertTrue(app.descendants(matching: .any)["more.theme"].exists, "More has no theme choice")
        snap("core-shell-more")
    }

    func testSelectingInTheGrid() throws {
        try launchSignedIn()
        XCTAssertTrue(firstCell.waitForExistence(timeout: 30))
        app.navigationBars["Library"].buttons["Select"].tap()
        firstCell.tap()
        app.buttons.matching(identifier: "media.cell").element(boundBy: 1).tap()
        XCTAssertEqual(app.staticTexts["bulk.count"].label, "2 selected")
        snap("core-library-selecting")
        app.buttons["bulk.done"].tap()
        XCTAssertFalse(app.staticTexts["bulk.count"].exists)
    }

    func testTheViewerGivesTheTapToTheSmallerBox() throws {
        try launchSignedIn()
        XCTAssertTrue(boxesCell.waitForExistence(timeout: 30), "the seed's photo with boxes is not in the grid")
        boxesCell.tap()
        // A face inside a larger person box of someone else.
        let face = app.buttons["Face of Test Visitor"]
        let person = app.buttons["Person box of Test Person"]
        XCTAssertTrue(face.waitForExistence(timeout: 30), "the face box did not show")
        XCTAssertTrue(person.exists, "the person box did not show")
        XCTAssertTrue(person.frame.contains(face.frame), "the seed's face box should sit inside the person box")
        XCTAssertTrue(app.windows.firstMatch.frame.contains(face.frame), "the face box is not on the screen")
        snap("core-viewer-boxes")

        // The middle of the face is inside the person box too: the face gets the tap. (A box draws itself and takes
        // no touches; the stage hit-tests the boxes, so the tap goes by its place on the screen.)
        face.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        // The face's editor opens (the boxes area's FaceEditor), naming whose face it is.
        let name = app.staticTexts["boxes.editor.name"]
        XCTAssertTrue(app.navigationBars["Edit face"].waitForExistence(timeout: 10), "a tapped box opened nothing")
        XCTAssertEqual(name.label, "Test Visitor")
        snap("core-viewer-face-tapped")
        closeEditor()

        // A point of the person box away from the face goes to the person box.
        person.coordinate(withNormalizedOffset: CGVector(dx: 0.85, dy: 0.92)).tap()
        XCTAssertTrue(app.navigationBars["Edit person"].waitForExistence(timeout: 10), "the person box opened nothing")
        XCTAssertEqual(name.label, "Test Person")

        // "Move to someone else" opens the person picker ranked by likeness to the box (nothing is saved here).
        app.buttons["viewer.box.who"].tap()
        XCTAssertTrue(app.navigationBars["Who is this?"].waitForExistence(timeout: 15), "the picker did not open")
        XCTAssertTrue(app.buttons.matching(identifier: "picker.option").firstMatch.waitForExistence(timeout: 20),
                      "the picker listed nobody")
        snap("core-picker")
        app.navigationBars["Who is this?"].buttons.firstMatch.tap()
        XCTAssertTrue(app.navigationBars["Edit person"].waitForExistence(timeout: 10), "the picker did not go back")
        closeEditor()

        // Hiding the boxes, then closing the viewer. A tap above a sheet can reach the photo and hide the controls;
        // a tap on an empty part of the photo shows them again.
        let boxesButton = app.buttons["viewer.boxes"]
        if !boxesButton.waitForExistence(timeout: 3) {
            app.windows.firstMatch.coordinate(withNormalizedOffset: CGVector(dx: 0.08, dy: 0.5)).tap()
        }
        XCTAssertTrue(boxesButton.waitForExistence(timeout: 5), "the viewer's controls did not show")
        boxesButton.tap()
        XCTAssertTrue(face.waitForNonExistence(timeout: 5), "the boxes did not hide")
        snap("core-viewer-no-boxes")
        app.buttons["viewer.close"].tap()
        XCTAssertTrue(firstCell.waitForExistence(timeout: 10))
    }

    /// Closes a box's editor with its Close button.
    private func closeEditor(file: StaticString = #filePath, line: UInt = #line) {
        app.buttons["boxes.editor.close"].tap()
        XCTAssertTrue(app.buttons["boxes.editor.close"].waitForNonExistence(timeout: 10), "the editor did not close",
                      file: file, line: line)
    }

    /// Closes a short sheet: a tap on the dimmed screen above it, else a drag down from its text.
    private func dismissSheet(showing element: XCUIElement, file: StaticString = #filePath, line: UInt = #line) {
        app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.12)).tap()
        if !element.waitForNonExistence(timeout: 3) {
            element.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
                .press(forDuration: 0.1, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.99)))
        }
        XCTAssertTrue(element.waitForNonExistence(timeout: 10), "the sheet did not close", file: file, line: line)
    }

    func testSignInWithThePassword() throws {
        try launchSignedOut()
        let password = ProcessInfo.processInfo.environment["PCC_TEST_PASSWORD"] ?? "ci-only-password"
        XCTAssertTrue(app.buttons["Password"].waitForExistence(timeout: 30), "the sign-in screen did not show")
        app.buttons["Password"].tap()
        let user = app.textFields["signin.username"]
        XCTAssertTrue(user.waitForExistence(timeout: 5))
        user.tap()
        user.typeText("owner")
        let secret = app.secureTextFields["signin.password"]
        secret.tap()
        secret.typeText(password)
        snap("core-sign-in-password")
        app.buttons["signin.submit"].tap()
        XCTAssertTrue(app.tabBars.buttons["Library"].waitForExistence(timeout: 30), "the password sign-in did not finish")
        XCTAssertTrue(firstCell.waitForExistence(timeout: 30))
    }
}
