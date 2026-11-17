import XCTest

/// The 3D heads area (Features/Heads): "Heads Person" has a stored model of two looks with their photo textures,
/// "Heads Newcomer" has none (ci/seeds/heads.py; the CI server has no generator). Reached as the owner reaches
/// them: People, the person, the 3D tab.
final class HeadsScreens: ScreenTestCase {
    private func openPerson(_ name: String) {
        tab("People")
        let card = app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", name)).firstMatch
        XCTAssertTrue(card.waitForExistence(timeout: 20), "no card for \(name) on People")
        card.tap()
        let tab3d = app.buttons["people.tab.3d"]
        if tab3d.waitForExistence(timeout: 5) {
            // The person page's tabs scroll sideways and 3D, the last one, starts off the screen on a phone: drag the
            // row of tabs to the left, at its own height, until 3D can be touched. (A query for the row's scroll view
            // found nothing to swipe, and a swipe on a tab button could scroll that button back into view.)
            var drags = 0
            while !tab3d.isHittable && drags < 5 {
                let y = tab3d.frame.midY
                let width = app.frame.width
                let origin = app.coordinate(withNormalizedOffset: .zero)
                origin.withOffset(CGVector(dx: width * 0.85, dy: y))
                    .press(forDuration: 0.05, thenDragTo: origin.withOffset(CGVector(dx: width * 0.15, dy: y)))
                drags += 1
            }
            XCTAssertTrue(tab3d.isHittable, "could not reach the 3D tab")
            tab3d.tap()
        }
    }

    private func element(_ identifier: String) -> XCUIElement {
        app.descendants(matching: .any)[identifier]
    }

    private func waitGone(_ element: XCUIElement, timeout: TimeInterval) -> Bool {
        let gone = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == false"), object: element)
        return XCTWaiter().wait(for: [gone], timeout: timeout) == .completed
    }

    /// Scrolls the page back towards its top with drags down its right margin, beside the head (a drag that starts
    /// on the head turns the head instead).
    private func scrollToTop(drags: Int) {
        for _ in 0..<drags {
            app.coordinate(withNormalizedOffset: CGVector(dx: 0.97, dy: 0.3))
                .press(forDuration: 0.05, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.97, dy: 0.85)))
        }
    }

    /// Scrolls the page down with drags up its right margin until the element can be touched. The controls sit
    /// below the head, and a scroll that XCUITest makes by itself may start on the head (which turns it) or land a
    /// slider's drag on another slider.
    private func scrollTo(_ element: XCUIElement, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(element.waitForExistence(timeout: 10), "no \(element)", file: file, line: line)
        var drags = 0
        while !element.isHittable && drags < 10 {
            app.coordinate(withNormalizedOffset: CGVector(dx: 0.97, dy: 0.72))
                .press(forDuration: 0.05, thenDragTo: app.coordinate(withNormalizedOffset: CGVector(dx: 0.97, dy: 0.38)))
            drags += 1
        }
        XCTAssertTrue(element.isHittable, "could not scroll to \(element)", file: file, line: line)
    }

    /// Flips a switch by its knob (a tap on a switch's middle may land on its label).
    private func flip(_ identifier: String) {
        let toggle = app.switches[identifier]
        scrollTo(toggle)
        toggle.coordinate(withNormalizedOffset: CGVector(dx: 0.93, dy: 0.5)).tap()
    }

    func testAModelShowsItsHeadsAndTheirControls() throws {
        try launchSignedIn()
        openPerson("Heads Person")
        XCTAssertTrue(element("heads.viewer").waitForExistence(timeout: 30), "the head's stage did not show")
        let caption = element("heads.caption")
        XCTAssertTrue(caption.waitForExistence(timeout: 10))
        XCTAssertTrue(waitGone(element("heads.loading"), timeout: 40), "the head did not load")
        XCTAssertFalse(element("heads.loadRetry").exists, "the head file was refused")
        // the model opens on the latest age: the older look, whose face was shaped
        XCTAssertTrue(caption.label.hasPrefix("About 41, short hair, a beard."), caption.label)
        XCTAssertTrue(app.staticTexts["The nose, lips and chin were shaped from the photo by a depth model."].exists)
        XCTAssertTrue(app.staticTexts["From the front this is the photo; past about 50 degrees the sides and back are filled in."].exists)
        snap("heads-model")

        // the other look, with the generator's note
        let young = app.buttons["heads.look.heads-young"]
        XCTAssertTrue(young.waitForExistence(timeout: 10))
        young.tap()
        XCTAssertTrue(waitGone(element("heads.loading"), timeout: 40), "the second head did not load")
        let note = element("heads.note")
        XCTAssertTrue(note.waitForExistence(timeout: 10))
        XCTAssertEqual(note.label, "The face was not shaped: the depth model is not installed.")
        XCTAssertTrue(caption.label.hasPrefix("About 29"), caption.label)
        snap("heads-look-young")

        // turn the head and take its colours away: the sliders say what the head shows
        let turn = app.sliders["heads.turn"]
        scrollTo(turn)
        turn.adjust(toNormalizedSliderPosition: 0.72)
        XCTAssertTrue((turn.value as? String ?? "").hasSuffix("to the right"), "\(turn.value ?? "")")
        let age = app.sliders["heads.axis.age"]
        XCTAssertEqual(age.value as? String, "about 29")
        snap("heads-controls")
        flip("heads.colour")
        XCTAssertEqual(app.switches["heads.colour"].value as? String, "0")
        XCTAssertFalse(app.switches["heads.photo"].isEnabled, "Photo waits while Colour is off")
        scrollToTop(drags: 5)
        XCTAssertTrue(element("heads.viewer").isHittable, "the head is not back on screen")
        snap("heads-turned-bare")

        // Reset view puts it all back
        let reset = app.buttons["heads.reset"]
        scrollTo(reset)
        reset.tap()
        XCTAssertEqual(app.sliders["heads.turn"].value as? String, "facing you")
        XCTAssertEqual(app.switches["heads.colour"].value as? String, "1")

        // Generate again: the options, and why nothing can be made on this server
        let again = app.buttons["heads.again"]
        scrollTo(again)
        again.tap()
        let blocked = element("heads.blocked")
        XCTAssertTrue(blocked.waitForExistence(timeout: 10))
        XCTAssertTrue(blocked.label.contains("not installed"), blocked.label)
        XCTAssertFalse(app.buttons["heads.generate"].isEnabled)
        let cancel = app.buttons["heads.againCancel"]
        scrollTo(cancel)
        snap("heads-again")
        cancel.tap()

        // Delete asks first; cancelled, the model stays
        let delete = app.buttons["heads.delete"]
        scrollTo(delete)
        delete.tap()
        XCTAssertTrue(app.staticTexts["Delete this 3D model?"].waitForExistence(timeout: 10))
        snap("heads-delete")
        app.buttons["Cancel"].firstMatch.tap()
        XCTAssertTrue(element("heads.summary").waitForExistence(timeout: 10))
        XCTAssertTrue(element("heads.summary").label.hasPrefix("2 looks"), element("heads.summary").label)
    }

    func testAPersonWithoutAModelSaysWhatOneIs() throws {
        try launchSignedIn()
        openPerson("Heads Newcomer")
        XCTAssertTrue(element("heads.intro").waitForExistence(timeout: 30), "the 3D tab did not show")
        let blocked = element("heads.blocked")
        XCTAssertTrue(blocked.exists)
        XCTAssertTrue(blocked.label.contains("not installed"), blocked.label)
        XCTAssertFalse(app.buttons["heads.generate"].isEnabled, "Generate is offered on a server without a generator")
        XCTAssertFalse(element("heads.viewer").exists)
        snap("heads-empty")
    }
}
