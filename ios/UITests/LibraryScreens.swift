import XCTest

/// The library area against the seeded server. Its own items (ci/seeds/library.py): eight photos named
/// "library lantern NN.jpg" in January and February 2026, the first three food, three tagged "Library sample". The
/// tests that filter or change something use that tag, so what other areas add to the library changes nothing here.
final class LibraryScreens: ScreenTestCase {
    private var cells: XCUIElementQuery { app.buttons.matching(identifier: "media.cell") }
    private var count: XCUIElement { app.staticTexts["library.count"] }

    private func openLibrary() throws {
        try launchSignedIn()
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 30), "the library did not show its items")
    }

    /// Waits until the count line reads `text`.
    private func waitForCount(_ text: String, timeout: TimeInterval = 20) {
        let predicate = NSPredicate(format: "label == %@", text)
        let found = expectation(for: predicate, evaluatedWith: count)
        wait(for: [found], timeout: timeout)
    }

    /// Filters by the seed's tag in the filter sheet.
    private func filterBySampleTag() {
        app.buttons["library.filters"].tap()
        let tag = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Library sample'")).firstMatch
        XCTAssertTrue(tag.waitForExistence(timeout: 15), "the filter sheet does not list the seed's tag")
        snap("library-filter-sheet")
        tag.tap()
        let done = app.buttons["library.filters.done"]
        done.tap()
        // the count changes behind the sheet as soon as the tag is picked: wait for the sheet itself to go
        XCTAssertTrue(done.waitForNonExistence(timeout: 10), "the filter sheet did not close")
        waitForCount("3 items")
    }

    func testTimelineByMonthWithScrubber() throws {
        try openLibrary()
        XCTAssertTrue(count.waitForExistence(timeout: 10))
        XCTAssertTrue(count.label.hasSuffix("items"), "the count line reads \(count.label)")
        let november = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'November 2026'")).firstMatch
        XCTAssertTrue(november.waitForExistence(timeout: 10), "the newest month has no heading")
        // the newest month holds two items: tiles beyond them belong to the next month, which loads by itself
        wait(for: [expectation(for: NSPredicate(format: "count >= 4"), evaluatedWith: cells)], timeout: 20)
        snap("library-timeline")
        let scrubber = app.descendants(matching: .any)["library.scrubber"]
        XCTAssertTrue(scrubber.waitForExistence(timeout: 10), "no scrubber beside the timeline")
        // the bottom of the rail is the oldest month: the timeline jumps there, away from the newest one
        scrubber.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.98)).tap()
        let away = NSPredicate(format: "exists == false OR hittable == false")
        wait(for: [expectation(for: away, evaluatedWith: november)], timeout: 15)
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 15), "no tiles where the scrubber landed")
        snap("library-scrubbed")
    }

    func testFiltersNarrowAndClear() throws {
        try openLibrary()
        filterBySampleTag()
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'February 2026'")).firstMatch
            .waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'January 2026'")).firstMatch
            .exists)
        // The chips scroll sideways and Food starts behind the Filters and map buttons: a tap at its place would open
        // the filters again, so the row is scrolled until Food can be touched.
        let food = app.buttons["library.category.food"]
        let chips = app.scrollViews["library.categories"]
        XCTAssertTrue(chips.waitForExistence(timeout: 10), "no category chips")
        // slow drags towards Food (a quick swipe flings the short row past it)
        for _ in 0..<6 where !food.isHittable {
            let behindLeft = food.frame.midX < chips.frame.midX
            let from = chips.coordinate(withNormalizedOffset: CGVector(dx: behindLeft ? 0.2 : 0.8, dy: 0.5))
            let to = chips.coordinate(withNormalizedOffset: CGVector(dx: behindLeft ? 0.8 : 0.2, dy: 0.5))
            from.press(forDuration: 0.05, thenDragTo: to, withVelocity: .slow, thenHoldForDuration: 0.3)
        }
        XCTAssertTrue(food.isHittable, "the Food chip cannot be reached")
        food.tap()
        waitForCount("1 item")
        snap("library-filter-food")
        app.buttons["library.clearFilters"].tap()
        let back = NSPredicate(format: "label != '1 item' AND label != '3 items' AND label ENDSWITH 'items'")
        wait(for: [expectation(for: back, evaluatedWith: count)], timeout: 20)
    }

    func testSearchWithSnippetsAndModes() throws {
        try openLibrary()
        let field = app.searchFields.firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 10), "no search field")
        field.tap()
        field.typeText("lantern\n")
        let results = NSPredicate(format: "label ENDSWITH 'for \"lantern\"'")
        wait(for: [expectation(for: results, evaluatedWith: count)], timeout: 30)
        XCTAssertTrue(cells.firstMatch.waitForExistence(timeout: 10), "the search found nothing")
        snap("library-search")
        // a word of the seeded text note: its tile shows the matched words
        field.buttons["Clear text"].firstMatch.tap()
        field.typeText("packing\n")
        let note = NSPredicate(format: "label ENDSWITH 'for \"packing\"'")
        wait(for: [expectation(for: note, evaluatedWith: count)], timeout: 30)
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "identifier == 'media.cell' AND label CONTAINS 'Packing list'"))
            .firstMatch.waitForExistence(timeout: 10), "the note's snippet is not on its tile")
        app.segmentedControls["library.mode"].buttons["Text"].tap()
        wait(for: [expectation(for: note, evaluatedWith: count)], timeout: 30)
        snap("library-search-snippet")
    }

    func testSelectionFavouritesTheSeedsItems() throws {
        try openLibrary()
        filterBySampleTag()
        app.navigationBars["Library"].buttons["Select"].tap()
        app.buttons["library.bulk.all"].tap()
        XCTAssertEqual(app.staticTexts["bulk.count"].label, "3 selected")
        snap("library-selection")
        app.buttons["library.bulk.favorite"].tap()
        let toast = app.descendants(matching: .any)["toast"]
        XCTAssertTrue(toast.waitForExistence(timeout: 20), "no message after Favourite")
        XCTAssertTrue(toast.label.contains("3 items added to favourites."), "the message reads \(toast.label)")
        XCTAssertFalse(app.staticTexts["bulk.count"].exists, "the selection did not end")
        snap("library-favourited")
    }
}
