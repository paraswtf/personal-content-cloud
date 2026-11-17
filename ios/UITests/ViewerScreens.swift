import XCTest

/// The viewer's info panel, toolbar and own pages, on the viewer seed's items (ci/seeds/viewer.py: a boarding pass
/// PDF, a photo received on WhatsApp with the tag "Viewer trip", a video with a face at one second and a video with
/// nobody on it when the machine could make them, a file without a preview, a private photo in the vault with the
/// tag "Viewer private"), which sit at the end of the library (April 2026). Every change a test makes it takes back.
final class ViewerScreens: ScreenTestCase {
    /// The seed's throwaway account on a server that lives for one CI run (ci/seed_server.py).
    private var password: String { ProcessInfo.processInfo.environment["PCC_TEST_PASSWORD"] ?? "ci-only-password" }

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
        XCTAssertTrue(reveal(target), "\(prefix) is not in the library", file: file, line: line)
        target.tap()
        XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 20), "the viewer did not open",
                      file: file, line: line)
    }

    /// Scrolls the grid with short, slow drags until `target` sits wholly below the search field and the month header
    /// pinned under it, and above the tab bar, where a tap reaches it. XCTest calls a tile under the pinned header
    /// hittable (its frame is inside the window), but a tap there lands on the header and opens nothing; a swipe
    /// flings the grid past the row it looks for.
    @discardableResult
    private func reveal(_ target: XCUIElement, drags: Int = 60) -> Bool {
        let window = app.windows.firstMatch
        var last: CGRect?
        for _ in 0..<drags {
            if target.exists {
                let f = target.frame
                let top = visibleTop, bottom = visibleBottom
                if f.minY >= top && f.maxY <= bottom { return true }
                // the grid did not move (its top or its end): take the tile where it is
                if f == last { return target.isHittable }
                last = f
                if f.minY < top {
                    drag(window, by: min(top - f.minY + 30, 300))
                    continue
                }
                if f.minY < bottom + 600 {
                    drag(window, by: -min(f.maxY - bottom + 30, 300))
                    continue
                }
            }
            drag(window, by: -300)
        }
        return false
    }

    /// Where the grid's rows start to show whole: under the search field (or the bar) and the month header pinned
    /// under it.
    private var visibleTop: CGFloat {
        let search = app.searchFields.firstMatch
        let bar = app.navigationBars.firstMatch
        let above = max(search.exists ? search.frame.maxY : 0, bar.exists ? bar.frame.maxY : 0)
        return above + 56
    }

    /// Where the grid's rows stop showing: the tab bar.
    private var visibleBottom: CGFloat {
        let tabs = app.tabBars.firstMatch
        return tabs.exists ? tabs.frame.minY : app.windows.firstMatch.frame.maxY - 90
    }

    /// One slow drag of the grid: `dy` above zero shows what is above, below zero what is below. It holds at the end,
    /// so the grid stops where the finger does.
    private func drag(_ window: XCUIElement, by dy: CGFloat) {
        let start = window.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: dy > 0 ? 0.4 : 0.75))
        start.press(forDuration: 0.05, thenDragTo: start.withOffset(CGVector(dx: 0, dy: dy)), withVelocity: .slow,
                    thenHoldForDuration: 0.2)
    }

    /// The picker's option for a name.
    private func option(_ name: String) -> XCUIElement {
        app.buttons.matching(NSPredicate(format: "identifier == 'picker.option' AND label BEGINSWITH %@", name)).firstMatch
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
        XCTAssertTrue(tagButton("Viewer trip").waitForExistence(timeout: 10), "the seeded tag is missing")
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
        let chip = tagButton("Viewer check")
        XCTAssertTrue(chip.waitForExistence(timeout: 15), "the new tag did not show")
        snap("viewer-tag-added")
        app.buttons["Remove tag Viewer check"].tap()
        // the server teaches its tag learner on a removal, which takes a while on the test machine
        let gone = NSPredicate(format: "exists == false")
        expectation(for: gone, evaluatedWith: chip)
        waitForExpectations(timeout: 60)
        XCTAssertTrue(tagButton("Viewer trip").exists, "the seeded tag should stay")
    }

    /// The owner's tag chip, which opens the library with that tag.
    private func tagButton(_ name: String) -> XCUIElement {
        app.buttons.matching(NSPredicate(format: "label == %@", name)).firstMatch
    }

    func testATagShowsEverythingWithIt() throws {
        try launchSignedIn()
        open("Photo, Apr 18, 2026")
        openDetails()
        let tag = tagButton("Viewer trip")
        XCTAssertTrue(tag.waitForExistence(timeout: 15), "the seeded tag is missing")
        tag.tap()
        // The viewer closes and the library opens with the tag: the tagged photo is on its first screen.
        XCTAssertTrue(app.buttons["viewer.close"].waitForNonExistence(timeout: 15), "the viewer stayed open")
        XCTAssertTrue(cell("Photo, Apr 18, 2026").waitForExistence(timeout: 30), "the tagged photo is not listed")
        XCTAssertFalse(cell("Boarding pass, Apr 20, 2026", "PDF, Apr 20, 2026").exists,
                       "the library shows items without the tag")
        snap("viewer-tag-library")
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
        let shown = reveal(video)
        guard video.exists else {
            throw XCTSkip("the seed made no video here (neither ffmpeg nor OpenCV could write one)")
        }
        XCTAssertTrue(shown, "the video is not where a tap reaches it")
        video.tap()
        XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 20))
        let chip = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Viewer Guest, 0:01'")).firstMatch
        XCTAssertTrue(chip.waitForExistence(timeout: 20), "the video's people chips did not show")
        chip.tap()
        snap("viewer-video-chip")
        XCTAssertTrue(app.buttons["Edit, tagged as Viewer Guest"].exists)
    }

    func testDrawingAPersonBoxOnAVideoFrame() throws {
        try launchSignedIn()
        XCTAssertTrue(app.buttons.matching(identifier: "media.cell").firstMatch.waitForExistence(timeout: 30))
        let video = cell("Video, 0:03, Apr 11, 2026")
        let shown = reveal(video)
        guard video.exists else {
            throw XCTSkip("the seed made no video here (neither ffmpeg nor OpenCV could write one)")
        }
        XCTAssertTrue(shown, "the video is not where a tap reaches it")
        video.tap()
        XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 20))
        let add = app.buttons["boxes.add"]
        XCTAssertTrue(add.waitForExistence(timeout: 20), "the add box button is missing on a video")
        // The neighbouring page (the Apr 12 video) may show the seed's own "Viewer Guest, 0:01" chip: count the chips
        // before drawing, so only a new one counts as the drawn box.
        let chips = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Viewer Guest, '"))
        let chipsBefore = chips.count
        // A menu's button is a wrapper the accessibility tap cannot scroll to: tap its place on the screen.
        add.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let personItem = app.buttons["Add a person box"]
        XCTAssertTrue(personItem.waitForExistence(timeout: 5), "the add box menu offers no person box on a video")
        personItem.tap()

        let hint = app.staticTexts["boxes.draw.hint"]
        XCTAssertTrue(hint.waitForExistence(timeout: 10), "draw mode did not start")
        XCTAssertTrue(hint.label.hasPrefix("Paused."), "the hint does not say the frame is paused: \(hint.label)")
        XCTAssertEqual(app.buttons["boxes.draw.whole"].label, "Whole frame", "Whole frame is missing")
        // The paused frame shows as a picture with the drawing layer over it.
        let layer = app.descendants(matching: .any)["stage.edit"]
        XCTAssertTrue(layer.waitForExistence(timeout: 30), "the paused frame did not show for drawing")
        snap("viewer-video-drawing")
        let from = layer.coordinate(withNormalizedOffset: CGVector(dx: 0.3, dy: 0.3))
        let to = layer.coordinate(withNormalizedOffset: CGVector(dx: 0.65, dy: 0.75))
        from.press(forDuration: 0.3, thenDragTo: to)
        let next = app.buttons["boxes.draw.next"]
        XCTAssertTrue(next.waitForExistence(timeout: 5), "the drawn box's card did not show")
        XCTAssertTrue(text("Person box on this frame").exists, "the card does not say the box is on this frame")
        snap("viewer-video-drawn")
        next.tap()

        // The seed's own person, so the test makes nobody new (a person made here would stay for every later test).
        XCTAssertTrue(app.navigationBars["Who is in the box?"].waitForExistence(timeout: 10), "the picker did not open")
        let search = app.searchFields.firstMatch
        XCTAssertTrue(search.waitForExistence(timeout: 15))
        search.tap()
        search.typeText("Viewer Guest")
        let guest = option("Viewer Guest")
        XCTAssertTrue(guest.waitForExistence(timeout: 20), "the picker did not list Viewer Guest")
        guest.tap()

        // The box is added with the frame's time and shows as a new chip under the video.
        let failed = app.staticTexts["boxes.draw.error"]
        let outcome = NSPredicate { _, _ in chips.count > chipsBefore || failed.exists }
        expectation(for: outcome, evaluatedWith: nil)
        waitForExpectations(timeout: 60)
        if failed.exists {
            // Without ffmpeg (the CI machine) the server cannot cut the frame and answers 409. It got the frame's
            // time all the same: without it the answer is 400 frame_time_needed ("Pause the video on a frame...").
            XCTAssertFalse(failed.label.hasPrefix("Pause the video"), "the box went without the frame's time")
            XCTAssertTrue(failed.label.contains("could not be opened"), "the box was not added: \(failed.label)")
            snap("viewer-video-box-no-ffmpeg")
            app.buttons["boxes.draw.cancelBox"].tap()
            XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 10), "the viewer's bars did not come back")
            return
        }
        XCTAssertTrue(chips.count > chipsBefore, "the drawn person box did not show as a chip")
        snap("viewer-video-box-added")

        // Taken back: the chip's pencil opens the box, and Not a person removes it. Only when that pencil is the only
        // one: with the seed's own Viewer Guest chip in view, a wrong pick would remove the face other tests use.
        let pencils = app.buttons.matching(NSPredicate(format: "label == 'Edit, tagged as Viewer Guest'"))
        guard pencils.count == 1 else { return }
        pencils.firstMatch.tap()
        let remove = app.buttons["boxes.remove"]
        XCTAssertTrue(remove.waitForExistence(timeout: 10), "the person box's editor did not open")
        remove.tap()
        let back = NSPredicate { _, _ in chips.count == chipsBefore }
        expectation(for: back, evaluatedWith: nil)
        waitForExpectations(timeout: 30)
    }

    func testAPrivateItemsTagShowsTheVaultWithIt() throws {
        try launchSignedIn()
        tab("More")
        let row = app.buttons["more.vault"]
        XCTAssertTrue(row.waitForExistence(timeout: 20), "no vault row under More")
        row.tap()
        // The vault stays open for the whole run; when its window has ended meanwhile, open it first.
        let field = app.secureTextFields["vault.password"]
        let photo = cell("Photo, Apr 16, 2026")
        let first = NSPredicate { _, _ in photo.exists || field.exists }
        expectation(for: first, evaluatedWith: nil)
        waitForExpectations(timeout: 30)
        if field.exists && !photo.exists {
            field.tap()
            field.typeText(password + "\n")
        }
        XCTAssertTrue(photo.waitForExistence(timeout: 30), "the seed's private photo is not in the vault")
        XCTAssertTrue(reveal(photo), "the private photo is not where a tap reaches it")
        photo.tap()
        XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 20), "the viewer did not open")
        openDetails()
        let tag = tagButton("Viewer private")
        XCTAssertTrue(tag.waitForExistence(timeout: 15), "the private photo's tag is missing")
        snap("viewer-vault-tag")
        tag.tap()
        // The viewer closes and the vault opens with the tag: only the tagged private photo is listed.
        XCTAssertTrue(app.buttons["viewer.close"].waitForNonExistence(timeout: 15), "the viewer stayed open")
        XCTAssertTrue(app.buttons["vault.lock"].waitForExistence(timeout: 20), "the vault did not open")
        XCTAssertTrue(photo.waitForExistence(timeout: 30), "the tagged private photo is not listed")
        let cells = app.buttons.matching(identifier: "media.cell")
        let onlyTagged = NSPredicate { _, _ in cells.count == 1 }
        expectation(for: onlyTagged, evaluatedWith: nil)
        waitForExpectations(timeout: 30)
        snap("viewer-vault-tag-filtered")
    }
}
