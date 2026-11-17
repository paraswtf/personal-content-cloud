import XCTest

/// The viewer's box editing against the seeded server: naming a face nobody named, answering a guess, moving a
/// box, drawing a person box, naming a pet, and the info panel's people section. Each test has its own photo
/// from ci/seeds/boxes.py (30 to 26 April 2026), so they never change each other's boxes or the core's.
final class BoxesScreens: ScreenTestCase {
    /// Opens the photo taken on `date` ("Apr 30, 2026"): the seed's photos sit after the core's in the library.
    private func openPhoto(_ date: String, file: StaticString = #filePath, line: UInt = #line) {
        let anyCell = app.buttons.matching(identifier: "media.cell").firstMatch
        XCTAssertTrue(anyCell.waitForExistence(timeout: 30), "the library did not show", file: file, line: line)
        let cell = app.buttons.matching(NSPredicate(format: "identifier == 'media.cell' AND label BEGINSWITH %@",
                                                    "Photo, \(date)")).firstMatch
        var swipes = 0
        while !(cell.exists && cell.isHittable) && swipes < 25 {
            app.swipeUp()
            swipes += 1
        }
        XCTAssertTrue(cell.exists && cell.isHittable, "the seed's photo of \(date) is not in the library", file: file, line: line)
        cell.tap()
        XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 15), "the viewer did not open", file: file, line: line)
    }

    /// A box on the stage, tapped in its middle (the stage hit-tests the boxes; a box takes no touches itself).
    private func tapBox(_ label: String, file: StaticString = #filePath, line: UInt = #line) {
        let box = app.buttons[label]
        XCTAssertTrue(box.waitForExistence(timeout: 30), "the box \"\(label)\" did not show", file: file, line: line)
        box.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
    }

    /// The picker's option for a name.
    private func option(_ name: String) -> XCUIElement {
        app.buttons.matching(NSPredicate(format: "identifier == 'picker.option' AND label BEGINSWITH %@", name)).firstMatch
    }

    func testNamingAFaceNobodyNamed() throws {
        try launchSignedIn()
        openPhoto("Apr 30, 2026")
        tapBox("Face, not named")
        XCTAssertTrue(app.navigationBars["Edit face"].waitForExistence(timeout: 10), "the face editor did not open")
        XCTAssertEqual(app.staticTexts["boxes.editor.name"].label, "Who is this?")
        XCTAssertTrue(app.buttons["boxes.remove"].exists, "Not a face is missing")
        XCTAssertTrue(app.buttons["boxes.adjust"].exists, "Move or resize is missing")
        snap("boxes-editor-unnamed")

        app.buttons["viewer.box.who"].tap()
        XCTAssertTrue(app.navigationBars["Who is this?"].waitForExistence(timeout: 10), "the picker did not open")
        let alpha = option("Boxes Alpha")
        XCTAssertTrue(alpha.waitForExistence(timeout: 20), "the picker did not list Boxes Alpha")
        snap("boxes-picker")
        alpha.tap()

        // The editor closes, and the box carries the name once the item is read again.
        XCTAssertTrue(app.buttons["Face of Boxes Alpha"].waitForExistence(timeout: 20), "the face was not named")
        XCTAssertFalse(app.navigationBars["Edit face"].exists)
        snap("boxes-named")
    }

    func testAnsweringAGuess() throws {
        try launchSignedIn()
        openPhoto("Apr 29, 2026")
        tapBox("Face, maybe Boxes Beta")
        XCTAssertTrue(app.navigationBars["Edit face"].waitForExistence(timeout: 10))
        XCTAssertEqual(app.staticTexts["boxes.editor.name"].label, "Boxes Beta?")
        let guessText = app.staticTexts["boxes.guess.text"]
        XCTAssertTrue(guessText.exists, "the guess panel did not show")
        // "Looks like Boxes Beta, 82% sure.": how sure the guess is, as the website says it
        XCTAssertTrue(guessText.label.hasPrefix("Looks like Boxes Beta, "), "the guess does not say how sure: \(guessText.label)")
        snap("boxes-guess")
        app.buttons["boxes.guess.yes"].tap()
        XCTAssertTrue(app.buttons["Face of Boxes Beta"].waitForExistence(timeout: 20), "the guess was not confirmed")
        snap("boxes-guess-confirmed")
    }

    func testMovingABox() throws {
        try launchSignedIn()
        openPhoto("Apr 28, 2026")
        let face = app.buttons["Face of Boxes Alpha"]
        XCTAssertTrue(face.waitForExistence(timeout: 30))
        let before = face.frame
        tapBox("Face of Boxes Alpha")
        XCTAssertTrue(app.buttons["boxes.adjust"].waitForExistence(timeout: 10))
        app.buttons["boxes.adjust"].tap()
        XCTAssertTrue(app.navigationBars["Edit face"].waitForNonExistence(timeout: 10), "the editor did not close")

        // The stage shows the box with handles; the viewer's bars give way to Save and Cancel.
        let save = app.buttons["boxes.adjust.save"]
        XCTAssertTrue(save.waitForExistence(timeout: 10), "the adjust bar did not show")
        XCTAssertFalse(app.buttons["viewer.close"].exists, "the viewer's bars stayed while moving a box")
        snap("boxes-adjusting")

        // One finger drags the box to the right.
        let origin = app.coordinate(withNormalizedOffset: .zero)
        let from = origin.withOffset(CGVector(dx: before.midX, dy: before.midY))
        let to = origin.withOffset(CGVector(dx: before.midX + 70, dy: before.midY))
        from.press(forDuration: 0.3, thenDragTo: to)
        snap("boxes-adjusted")
        save.tap()

        XCTAssertTrue(face.waitForExistence(timeout: 20), "the box did not come back")
        let moved = NSPredicate { _, _ in face.exists && face.frame.midX > before.midX + 30 }
        expectation(for: moved, evaluatedWith: nil)
        waitForExpectations(timeout: 20)
        XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 5), "the viewer's bars did not come back")
        snap("boxes-moved")
    }

    func testDrawingAPersonBox() throws {
        try launchSignedIn()
        openPhoto("Apr 27, 2026")
        let add = app.buttons["boxes.add"]
        XCTAssertTrue(add.waitForExistence(timeout: 15), "the add box button is missing")
        // A menu's button is a wrapper the accessibility tap cannot scroll to: tap its place on the screen.
        add.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
        let personItem = app.buttons["Add a person box"]
        XCTAssertTrue(personItem.waitForExistence(timeout: 5), "the add box menu did not open")
        snap("boxes-add-menu")
        personItem.tap()
        XCTAssertTrue(app.staticTexts["boxes.draw.hint"].waitForExistence(timeout: 5), "draw mode did not start")
        snap("boxes-drawing")

        // The photo fills the width in the middle of the screen: drag over part of it.
        let window = app.windows.firstMatch
        let from = window.coordinate(withNormalizedOffset: CGVector(dx: 0.3, dy: 0.44))
        let to = window.coordinate(withNormalizedOffset: CGVector(dx: 0.62, dy: 0.56))
        from.press(forDuration: 0.3, thenDragTo: to)
        let next = app.buttons["boxes.draw.next"]
        XCTAssertTrue(next.waitForExistence(timeout: 5), "the drawn box's card did not show")
        snap("boxes-drawn")
        next.tap()

        XCTAssertTrue(app.navigationBars["Who is in the box?"].waitForExistence(timeout: 10), "the picker did not open")
        let alpha = option("Boxes Alpha")
        XCTAssertTrue(alpha.waitForExistence(timeout: 20))
        alpha.tap()
        XCTAssertTrue(app.buttons["Person box of Boxes Alpha"].waitForExistence(timeout: 20), "the person box was not added")
        snap("boxes-person-box-added")
    }

    func testNamingAPet() throws {
        try launchSignedIn()
        openPhoto("Apr 26, 2026")
        tapBox("Pet: Dog")
        XCTAssertTrue(app.navigationBars["Edit pet"].waitForExistence(timeout: 10), "the pet editor did not open")
        XCTAssertEqual(app.staticTexts["boxes.editor.name"].label, "Which pet is this?")
        XCTAssertTrue(app.buttons["boxes.pet.species"].exists, "Not a dog is missing")
        snap("boxes-pet-editor")
        app.buttons["viewer.box.who"].tap()
        XCTAssertTrue(app.navigationBars["Which pet is this?"].waitForExistence(timeout: 10))
        let rex = option("Boxes Rex")
        XCTAssertTrue(rex.waitForExistence(timeout: 20), "the picker did not list the pet")
        rex.tap()
        XCTAssertTrue(app.buttons["Pet: Boxes Rex"].waitForExistence(timeout: 20), "the pet was not named")
        snap("boxes-pet-named")
    }

    /// "Show in the photo" turns the boxes back on after the owner hid them, as the website's showPerson does.
    func testShowInThePhotoBringsHiddenBoxesBack() throws {
        try launchSignedIn()
        openPhoto("Apr 28, 2026")
        let face = app.buttons["Face of Boxes Alpha"]
        XCTAssertTrue(face.waitForExistence(timeout: 30), "the face box did not show")
        let toggle = app.buttons["viewer.boxes"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 10), "the boxes button is missing")
        toggle.tap()
        XCTAssertTrue(face.waitForNonExistence(timeout: 10), "the boxes did not hide")
        XCTAssertEqual(toggle.label, "Show boxes")
        app.buttons["viewer.info"].tap()
        let menu = app.buttons["boxes.person.menu"]
        XCTAssertTrue(menu.waitForExistence(timeout: 15), "the people section did not list Boxes Alpha")
        menu.tap()
        let show = app.buttons["Show in the photo"]
        XCTAssertTrue(show.waitForExistence(timeout: 5), "Show in the photo is missing from the menu")
        show.tap()
        XCTAssertTrue(face.waitForExistence(timeout: 15), "Show in the photo did not bring the boxes back")
        XCTAssertEqual(toggle.label, "Hide boxes")
        snap("boxes-show-brings-boxes-back")
    }

    func testTheInfoPanelsPeopleSection() throws {
        try launchSignedIn()
        openPhoto("Apr 28, 2026")
        let info = app.buttons["viewer.info"]
        XCTAssertTrue(info.waitForExistence(timeout: 15))
        info.tap()
        let menu = app.buttons["boxes.person.menu"]
        XCTAssertTrue(menu.waitForExistence(timeout: 15), "the people section did not list Boxes Alpha")
        XCTAssertTrue(app.buttons["boxes.tag"].exists, "Tag someone is missing")
        snap("boxes-info")
        menu.tap()
        let change = app.buttons["Change to..."]
        XCTAssertTrue(change.waitForExistence(timeout: 5), "the person's menu did not open")
        let notAFace = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Not a face'")).firstMatch
        XCTAssertTrue(notAFace.exists, "Not a face is missing from the menu")
        snap("boxes-info-menu")
        change.tap()
        // The picker opens over the panel, ranked by likeness to Boxes Alpha's boxes here; nothing is saved.
        let bar = app.navigationBars["Change Boxes Alpha to..."]
        XCTAssertTrue(bar.waitForExistence(timeout: 10), "the change picker did not open")
        let note = app.staticTexts["picker.note"]
        XCTAssertTrue(note.waitForExistence(timeout: 5), "the change picker does not say what picking does")
        XCTAssertEqual(note.label, "Their faces and tags in this picture go to the one you pick, and look-alike faces may follow.")
        snap("boxes-info-change")
        bar.buttons["Cancel"].tap()
        XCTAssertTrue(menu.waitForExistence(timeout: 10), "the panel did not come back")

        // The person's own page: the viewer closes and the page opens on the tab underneath.
        let open = app.buttons.matching(NSPredicate(format: "identifier == 'boxes.person.open' AND label == %@",
                                                    "Open Boxes Alpha's page")).firstMatch
        XCTAssertTrue(open.waitForExistence(timeout: 5), "the row has no link to the person's page")
        open.tap()
        XCTAssertTrue(app.buttons["viewer.close"].waitForNonExistence(timeout: 15), "the viewer stayed open")
        XCTAssertTrue(app.navigationBars["Boxes Alpha"].waitForExistence(timeout: 20), "the person's page did not open")
        snap("boxes-info-person-page")
    }
}
