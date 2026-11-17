import XCTest

/// Settings > System, Learning and Undo history against the seeded server: models off, so the models read as
/// missing; backups on (the workflow sets BACKUP_DIR). The buttons that start work on the server (downloads,
/// housekeeping, backups, encoder tests, training, apply, reset) are checked for, never pressed.
final class SettingsSystemScreens: ScreenTestCase {
    private func openSettings(_ row: String) {
        tab("More")
        let settings = app.buttons["more.settings"]
        XCTAssertTrue(settings.waitForExistence(timeout: 15), "no Settings under More")
        settings.tap()
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 10))
        let link = app.buttons[row]
        XCTAssertTrue(link.waitForExistence(timeout: 10), "no \(row) row in Settings")
        link.tap()
    }

    private func element(containing text: String) -> XCUIElement {
        app.descendants(matching: .any).matching(NSPredicate(format: "label CONTAINS %@", text)).firstMatch
    }

    /// Moves the screen's list up by about a third of its height. A slow drag held at its end, not a swipe: a
    /// swipe flings the list on and can carry a row past the screen between two checks.
    private func dragUp() {
        let window = app.windows.firstMatch
        let start = window.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.72))
        let end = window.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.38))
        start.press(forDuration: 0.05, thenDragTo: end, withVelocity: .slow, thenHoldForDuration: 0.2)
    }

    /// Scrolls the screen's list until `element` is on screen, at most `drags` times.
    private func scroll(to element: XCUIElement, drags: Int = 30) -> Bool {
        for _ in 0..<drags {
            if element.exists && element.isHittable { return true }
            dragUp()
        }
        return element.exists && element.isHittable
    }

    /// Moves the list down by about a third of its height (the opposite of dragUp).
    private func dragDown() {
        let window = app.windows.firstMatch
        let start = window.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.38))
        let end = window.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.72))
        start.press(forDuration: 0.05, thenDragTo: end, withVelocity: .slow, thenHoldForDuration: 0.2)
    }

    /// The top and bottom of the part of the screen a list row can be tapped in: below the navigation bar, above
    /// the tab bar.
    private var openArea: (top: CGFloat, bottom: CGFloat) {
        let nav = app.navigationBars.firstMatch
        let tabs = app.tabBars.firstMatch
        let top = nav.exists ? nav.frame.maxY : 0
        let bottom = tabs.exists ? tabs.frame.minY : app.windows.firstMatch.frame.maxY
        return (top, bottom)
    }

    /// Scrolls until `element` is wholly between the navigation bar and the tab bar (isHittable alone is true for a
    /// chart whose middle is under the tab bar).
    private func scroll(toAboveTabBar element: XCUIElement, drags: Int = 10) -> Bool {
        for _ in 0...drags {
            if element.exists {
                let frame = element.frame
                let area = openArea
                if frame.minY >= area.top && frame.maxY <= area.bottom { return true }
                if frame.minY < area.top { dragDown() } else { dragUp() }
            } else {
                dragUp()
            }
        }
        return false
    }

    /// Scrolls back up until `element` is on screen and clear of the navigation bar, at most `drags` times.
    private func scrollBack(to element: XCUIElement, drags: Int = 10) -> Bool {
        for _ in 0..<drags {
            if element.exists && element.isHittable && element.frame.minY >= openArea.top { return true }
            dragDown()
        }
        return element.exists && element.isHittable
    }

    func testSystem() throws {
        try launchSignedIn()
        openSettings("settings.system")
        XCTAssertTrue(app.navigationBars["System"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["Processor"].waitForExistence(timeout: 30), "the system status did not load")
        XCTAssertTrue(element(containing: "Version").exists, "no server version")
        snap("setsys-system")

        // models off on the CI server: missing models, with the download offered (not pressed)
        XCTAssertTrue(scroll(to: app.buttons["system.models.download"]), "no download button for the missing models")
        XCTAssertTrue(element(containing: "Missing").exists, "no model reads as missing")
        snap("setsys-system-models")

        XCTAssertTrue(scroll(to: app.staticTexts["Processing queue"]), "no processing queue")
        snap("setsys-system-queue")

        XCTAssertTrue(scroll(to: app.buttons["system.backup.now"]), "backups are on in CI, but there is no Back up now")
        XCTAssertTrue(element(containing: "Originals").exists, "no storage tier")
        snap("setsys-system-storage")

        let range = app.segmentedControls["system.throughput.range"]
        XCTAssertTrue(scroll(to: range), "no processing speed")
        // the seed's uploads were stored minutes ago
        XCTAssertTrue(scroll(to: element(containing: "Files stored")), "no throughput for the stored files")
        snap("setsys-system-speed")
        // a tap on a chart reads one time's numbers in place of the period's. The chart is brought wholly above
        // the tab bar first: a tap that lands on the tab bar's More (the selected tab) goes back to the More list.
        let chart = app.descendants(matching: .any).matching(NSPredicate(format: "label BEGINSWITH %@", "Files stored:")).firstMatch
        if scroll(toAboveTabBar: chart) {
            chart.coordinate(withNormalizedOffset: CGVector(dx: 0.9, dy: 0.5)).tap()
            _ = element(containing: " finished, ").waitForExistence(timeout: 5)
            snap("setsys-system-speed-tap")
        }
        XCTAssertTrue(app.navigationBars["System"].exists, "the chart tap left the System screen")
        XCTAssertTrue(scrollBack(to: range), "the range control went off the screen")
        range.buttons["24 hours"].tap()
        XCTAssertTrue(element(containing: "in the last 24 hours").waitForExistence(timeout: 30), "the 24 hours view did not load")
        let table = app.buttons["system.throughput.table"]
        XCTAssertTrue(scroll(to: table), "no Show as table")
        table.tap()
        XCTAssertTrue(app.buttons["Hide the table"].waitForExistence(timeout: 5))
        snap("setsys-system-table")

        XCTAssertTrue(scroll(to: app.buttons["system.housekeeping"]), "no Run housekeeping")
        XCTAssertTrue(app.buttons["system.retryFailed"].exists, "no Retry failed uploads")
        snap("setsys-system-end")
    }

    func testLearning() throws {
        try launchSignedIn()
        openSettings("settings.learning")
        XCTAssertTrue(app.navigationBars["Learning"].waitForExistence(timeout: 10))
        let toggle = app.switches["learning.enabled"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 30), "the learning status did not load")
        XCTAssertTrue(element(containing: "trained").exists || element(containing: "Not trained").exists)
        snap("setsys-learning")

        XCTAssertTrue(scroll(to: app.segmentedControls["learning.sensitivity"]), "no vault sensitivity")
        snap("setsys-learning-learned")

        XCTAssertTrue(scroll(to: app.staticTexts["Corrections recorded"]), "no corrections section")
        XCTAssertTrue(scroll(to: app.buttons["learning.reset"]), "no Reset")
        XCTAssertTrue(app.buttons["learning.train"].exists, "no Train now")
        XCTAssertTrue(app.buttons["learning.apply"].exists, "no Apply to existing items")
        snap("setsys-learning-use")
    }

    func testUndoHistory() throws {
        try launchSignedIn(["PCC_TEST_SETSYS_UNDO": "1"])
        openSettings("settings.undo")
        XCTAssertTrue(app.navigationBars["Undo history"].waitForExistence(timeout: 10))
        let label = "Changed the description of the album Setsys undo test"
        let undo = app.buttons.matching(NSPredicate(format: "label == %@", "Undo: \(label)")).firstMatch
        XCTAssertTrue(undo.waitForExistence(timeout: 30), "the test change did not reach the undo history")
        snap("setsys-undo")
        undo.tap()
        let redo = app.buttons.matching(NSPredicate(format: "label == %@", "Redo: \(label)")).firstMatch
        XCTAssertTrue(redo.waitForExistence(timeout: 30), "the undo did not finish")
        XCTAssertTrue(element(containing: "undone").exists, "the entry does not say it was undone")
        snap("setsys-undo-undone")
        redo.tap()
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label == %@", "Undo: \(label)")).firstMatch
            .waitForExistence(timeout: 30), "the redo did not finish")
        snap("setsys-undo-redone")

        // Clear history asks first; Cancel keeps the list
        app.buttons["undo.clear"].tap()
        let cancel = app.buttons["Cancel"]
        XCTAssertTrue(cancel.waitForExistence(timeout: 5), "Clear history did not ask first")
        snap("setsys-undo-clear")
        cancel.tap()
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label == %@", "Undo: \(label)")).firstMatch.exists)
    }
}
