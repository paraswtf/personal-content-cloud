import XCTest

/// The viewer's info panel, toolbar and own pages, on the viewer seed's items (ci/seeds/viewer.py: a boarding pass
/// PDF, a photo received on WhatsApp with the tag "Viewer trip", a video with a face at one second when the machine
/// could make one, a file without a preview), which sit at the end of the library (April 2026). Every change a test
/// makes it takes back.
final class ViewerScreens: ScreenTestCase {
    /// The library tile whose label starts with one of `prefixes` (the library names a parsed document by its type,
    /// "Boarding pass, Apr 20, 2026", where the core's grid says "PDF, Apr 20, 2026").
    private func cell(_ prefixes: String...) -> XCUIElement {
        let any = prefixes.map { _ in "label BEGINSWITH %@" }.joined(separator: " OR ")
        return app.buttons.matching(NSPredicate(format: "identifier == 'media.cell' AND (\(any))", argumentArray: prefixes))
            .firstMatch
    }

    /// Scrolls the library down to the item whose label starts with one of `prefixes` and opens it in the viewer.
    private func open(_ prefixes: String..., file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(app.buttons.matching(identifier: "media.cell").firstMatch.waitForExistence(timeout: 30),
                      "the library is empty", file: file, line: line)
        let prefix = prefixes.first ?? ""
        let any = prefixes.map { _ in "label BEGINSWITH %@" }.joined(separator: " OR ")
        let target = app.buttons.matching(NSPredicate(format: "identifier == 'media.cell' AND (\(any))",
                                                      argumentArray: prefixes)).firstMatch
        var swipes = 0
        while !(target.exists && target.isHittable) && swipes < 40 {
            app.swipeUp()
            swipes += 1
        }
        XCTAssertTrue(target.isHittable, "\(prefix) is not in the library", file: file, line: line)
        target.tap()
        XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 20), "the viewer did not open",
                      file: file, line: line)
    }

    private func text(_ label: String) -> XCUIElement {
        app.staticTexts.matching(NSPredicate(format: "label ==[c] %@", label)).firstMatch
    }

    /// Any element whose label or value holds `words` (a labelled row may read as one element).
    private func anything(_ words: String) -> XCUIElement {
        app.descendants(matching: .any)
            .matching(NSPredicate(format: "label CONTAINS[c] %@ OR value CONTAINS[c] %@", words, words)).firstMatch
    }

    private func openDetails(file: StaticString = #filePath, line: UInt = #line) {
        app.buttons["viewer.info"].tap()
        XCTAssertTrue(app.buttons["Done"].waitForExistence(timeout: 15), "the details did not open", file: file, line: line)
    }

    /// Scrolls the details until `element` shows.
    private func scrollDetails(to element: XCUIElement, file: StaticString = #filePath, line: UInt = #line) {
        let list = app.collectionViews["viewer.info.list"].exists ? app.collectionViews["viewer.info.list"]
                                                                  : app.collectionViews.firstMatch
        var swipes = 0
        while !(element.exists && element.isHittable) && swipes < 12 {
            list.swipeUp()
            swipes += 1
        }
        XCTAssertTrue(element.exists, "\(element) did not show in the details", file: file, line: line)
    }

    func testTheDetailsOfAReceivedPhoto() throws {
        try launchSignedIn()
        open("Photo, Apr 18, 2026")
        openDetails()
        XCTAssertTrue(app.navigationBars["Photo details"].waitForExistence(timeout: 15))
        XCTAssertTrue(anything("Received on WhatsApp").waitForExistence(timeout: 15), "the received section is missing")
        XCTAssertTrue(anything("Kept in the library as a photo.").exists)
        XCTAssertTrue(app.buttons["viewer.info.markForward"].exists)
        XCTAssertTrue(text("Viewer trip").waitForExistence(timeout: 10), "the seeded tag is missing")
        snap("viewer-info-top")
        scrollDetails(to: anything("CI Camera"))
        // the place's row with its map comes after the camera; the list loads rows as they come into view
        let maps = app.buttons["viewer.info.openInMaps"]
        scrollDetails(to: maps)
        XCTAssertTrue(maps.exists, "the place has no map")
        snap("viewer-info-details")
        scrollDetails(to: app.buttons["viewer.info.trash"])
        XCTAssertTrue(app.buttons["viewer.info.favourite"].exists)
        snap("viewer-info-actions")
    }

    func testAddingAndRemovingATag() throws {
        try launchSignedIn()
        open("Photo, Apr 18, 2026")
        openDetails()
        let field = app.textFields["viewer.tags.field"]
        XCTAssertTrue(field.waitForExistence(timeout: 15))
        field.tap()
        field.typeText("Viewer check")
        app.buttons["viewer.tags.add"].tap()
        let chip = text("Viewer check")
        XCTAssertTrue(chip.waitForExistence(timeout: 15), "the new tag did not show")
        snap("viewer-tag-added")
        app.buttons["Remove tag Viewer check"].tap()
        // the server teaches its tag learner on a removal, which takes a while on the test machine
        let gone = NSPredicate(format: "exists == false")
        expectation(for: gone, evaluatedWith: chip)
        waitForExpectations(timeout: 60)
        XCTAssertTrue(text("Viewer trip").exists, "the seeded tag should stay")
    }

    func testFavouriteTheMenuAndTheTrashQuestion() throws {
        try launchSignedIn()
        open("Photo, Apr 18, 2026")
        let star = app.buttons["viewer.favourite"]
        XCTAssertTrue(star.waitForExistence(timeout: 15))
        XCTAssertEqual(star.label, "Add to favourites")
        star.tap()
        XCTAssertTrue(app.staticTexts["Added to favourites"].waitForExistence(timeout: 15), "no message after favouriting")
        let on = NSPredicate(format: "label == 'Remove from favourites'")
        expectation(for: on, evaluatedWith: star)
        waitForExpectations(timeout: 15)
        snap("viewer-favourite")
        star.tap()
        let off = NSPredicate(format: "label == 'Add to favourites'")
        expectation(for: off, evaluatedWith: star)
        waitForExpectations(timeout: 15)

        // the menu's own button sits inside the labelled one: tap the place
        app.buttons["viewer.more"].coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        XCTAssertTrue(app.buttons["Add to album"].waitForExistence(timeout: 10), "the menu did not open")
        XCTAssertTrue(app.buttons["Mark private"].exists)
        XCTAssertTrue(app.buttons["Save to Photos"].exists)
        snap("viewer-more-menu")
        app.buttons["Move to trash"].tap()
        XCTAssertTrue(app.staticTexts["Move this photo to the trash?"].waitForExistence(timeout: 10),
                      "the trash question did not show")
        snap("viewer-trash-question")
        app.buttons["Cancel"].tap()
        XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 10))
    }

    func testADocumentsPagesAndItsCard() throws {
        try launchSignedIn()
        open("Boarding pass, Apr 20, 2026", "PDF, Apr 20, 2026")
        let position = app.staticTexts["viewer.document.position"]
        XCTAssertTrue(position.waitForExistence(timeout: 20), "the document's pages did not show")
        XCTAssertEqual(position.label, "Page 1 of 2")
        app.buttons["viewer.document.next"].tap()
        let second = NSPredicate(format: "label == 'Page 2 of 2'")
        expectation(for: second, evaluatedWith: position)
        waitForExpectations(timeout: 10)
        snap("viewer-document-page")

        openDetails()
        XCTAssertTrue(app.navigationBars["Document details"].waitForExistence(timeout: 15))
        let title = app.staticTexts["viewer.document.title"]
        scrollDetails(to: title)
        XCTAssertEqual(title.label, "Viewer flight to Goa")
        XCTAssertTrue(text("check this").exists, "the hard-to-read seat is not shaded")
        snap("viewer-document-card")
        let body = app.staticTexts["viewer.text.body"]
        scrollDetails(to: body)
        XCTAssertTrue(body.label.contains("BOARDING PASS"), "the read text is missing")
        snap("viewer-document-text")
    }

    func testAFileWithoutAPreview() throws {
        try launchSignedIn()
        open("File, Apr 10, 2026")
        XCTAssertTrue(text("There is no preview for this kind of file. The original is kept exactly as uploaded.")
            .waitForExistence(timeout: 20))
        XCTAssertTrue(app.buttons["viewer.file.download"].exists)
        snap("viewer-file")
    }

    func testTheVideosPeopleChips() throws {
        try launchSignedIn()
        XCTAssertTrue(app.buttons.matching(identifier: "media.cell").firstMatch.waitForExistence(timeout: 30))
        let video = cell("Video, 0:03, Apr 12, 2026")
        var swipes = 0
        while !(video.exists && video.isHittable) && swipes < 40 {
            app.swipeUp()
            swipes += 1
        }
        guard video.exists else {
            throw XCTSkip("the seed made no video here (neither ffmpeg nor OpenCV could write one)")
        }
        video.tap()
        XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 20))
        let chip = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Viewer Guest, 0:01'")).firstMatch
        XCTAssertTrue(chip.waitForExistence(timeout: 20), "the video's people chips did not show")
        chip.tap()
        snap("viewer-video-chip")
        XCTAssertTrue(app.buttons["Edit, tagged as Viewer Guest"].exists)
    }
}
