import XCTest

/// The People tab and the person pages against the seeded server (ci/seeds/people.py adds its own people: People
/// Alpha with faces, two guesses and a tattoo, People Beta and People Gamma who look alike, People Delta and People
/// Epsilon to merge, People Zeta to rename, the cat People Pet, and two faces nobody named, 333 px wide). The tests
/// change only those.
final class PeopleScreens: ScreenTestCase {
    private func card(_ name: String) -> XCUIElement {
        app.buttons.matching(NSPredicate(format: "identifier == 'people.card' AND label BEGINSWITH %@", name)).firstMatch
    }

    /// Scrolls the screen up until `element` is there (lazy grids make their cells as they come into view).
    @discardableResult
    private func reveal(_ element: XCUIElement, swipes: Int = 6) -> Bool {
        if element.waitForExistence(timeout: 15) { return true }
        for _ in 0..<swipes {
            app.swipeUp()
            if element.waitForExistence(timeout: 2) { return true }
        }
        return element.exists
    }

    private func openPerson(_ name: String) {
        tab("People")
        let c = card(name)
        XCTAssertTrue(reveal(c), "no card for \(name)")
        c.tap()
        XCTAssertTrue(app.staticTexts["person.name"].waitForExistence(timeout: 20), "the page of \(name) did not open")
    }

    private func openTab(_ id: String) {
        let button = app.buttons[id]
        XCTAssertTrue(button.waitForExistence(timeout: 10), "no tab \(id)")
        if !button.isHittable { app.buttons["people.tab.items"].swipeLeft() }
        button.tap()
    }

    func testPeopleListSuggestionsAndPets() throws {
        try launchSignedIn()
        tab("People")
        XCTAssertTrue(card("People Alpha").waitForExistence(timeout: 20), "the seeded people are not listed")
        XCTAssertTrue(app.staticTexts["Could these be the same person?"].waitForExistence(timeout: 10),
                      "People Beta and People Gamma look alike but are not suggested")
        XCTAssertTrue(app.buttons["people.facesToName"].exists)
        snap("people-list")
        XCTAssertTrue(reveal(card("People Pet")), "the pet is not listed")
        snap("people-pets")
    }

    func testPersonPageTabs() throws {
        try launchSignedIn()
        openPerson("People Alpha")
        XCTAssertEqual(app.staticTexts["person.name"].label, "People Alpha")
        for id in ["people.tab.items", "people.tab.faces", "people.tab.guesses", "people.tab.traits",
                   "people.tab.related", "people.tab.3d"] {
            XCTAssertTrue(app.buttons[id].waitForExistence(timeout: 5), "no tab \(id)")
        }
        XCTAssertTrue(app.buttons.matching(identifier: "media.cell").firstMatch.waitForExistence(timeout: 20))
        snap("people-person-items")

        openTab("people.tab.faces")
        XCTAssertTrue(app.buttons.matching(identifier: "people.face").firstMatch.waitForExistence(timeout: 20))
        snap("people-person-faces")

        openTab("people.tab.guesses")
        XCTAssertTrue(app.buttons["person.guesses.confirmAll"].waitForExistence(timeout: 20),
                      "the two guesses the server made for People Alpha are not listed")
        snap("people-person-guesses")

        openTab("people.tab.traits")
        XCTAssertTrue(app.staticTexts["Tattoos"].waitForExistence(timeout: 20), "the tattoo is not listed")
        snap("people-person-traits")

        openTab("people.tab.related")
        XCTAssertTrue(app.staticTexts["Nothing related yet"].waitForExistence(timeout: 20))
        snap("people-person-related")

        openTab("people.tab.3d")
        snap("people-person-3d")
    }

    func testFacesToNameTagsAFace() throws {
        try launchSignedIn()
        tab("People")
        app.buttons["people.facesToName"].tap()
        let seeded = app.buttons.matching(NSPredicate(format: "identifier == 'people.face' AND label CONTAINS '333 px'"))
        XCTAssertTrue(seeded.firstMatch.waitForExistence(timeout: 20), "the faces nobody named are not listed")
        let before = seeded.count
        seeded.firstMatch.tap()
        XCTAssertEqual(app.staticTexts["faces.count"].label, "1 selected")
        snap("people-faces-to-name")
        app.buttons["faces.tag"].tap()
        let search = app.searchFields.firstMatch
        XCTAssertTrue(search.waitForExistence(timeout: 15), "the picker did not open")
        search.tap()
        search.typeText("People Named")
        let create = app.buttons["picker.create"]
        XCTAssertTrue(create.waitForExistence(timeout: 15), "the picker offers no new name")
        create.tap()
        let fewer = NSPredicate { _, _ in seeded.count < before }
        expectation(for: fewer, evaluatedWith: nil)
        waitForExpectations(timeout: 20)
        snap("people-faces-tagged")
    }

    func testMergeTwoPeople() throws {
        try launchSignedIn()
        tab("People")
        XCTAssertTrue(reveal(card("People Epsilon")), "no card for People Epsilon")
        app.buttons["people.merge"].tap()
        card("People Delta").tap()
        card("People Epsilon").tap()
        app.buttons["people.merge.start"].tap()
        let confirm = app.buttons["people.merge.confirm"]
        XCTAssertTrue(confirm.waitForExistence(timeout: 10), "the merge sheet did not open")
        snap("people-merge-sheet")
        confirm.tap()
        let gone = NSPredicate(format: "exists == false")
        expectation(for: gone, evaluatedWith: card("People Epsilon"))
        waitForExpectations(timeout: 20)
        XCTAssertTrue(card("People Delta").exists, "the person kept is gone")
        snap("people-merged")
    }

    func testRenamePerson() throws {
        try launchSignedIn()
        openPerson("People Zeta")
        app.buttons["person.rename"].tap()
        let field = app.alerts.textFields.firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 10), "the rename box did not open")
        field.typeText(" Renamed")
        app.alerts.buttons["Save"].tap()
        let renamed = NSPredicate(format: "label == 'People Zeta Renamed'")
        expectation(for: renamed, evaluatedWith: app.staticTexts["person.name"])
        waitForExpectations(timeout: 20)
        snap("people-renamed")
    }

    func testPetPage() throws {
        try launchSignedIn()
        openPerson("People Pet")
        XCTAssertTrue(app.buttons["people.tab.crops"].waitForExistence(timeout: 10), "a pet has no Crops tab")
        XCTAssertFalse(app.buttons["people.tab.faces"].exists, "a pet has a Faces tab")
        openTab("people.tab.crops")
        XCTAssertTrue(app.buttons.matching(identifier: "people.crop").firstMatch.waitForExistence(timeout: 20))
        snap("people-pet-crops")
    }

    func testPeopleSheetOnASelection() throws {
        try launchSignedIn()
        openPerson("People Alpha")
        let select = app.buttons["person.items.select"]
        XCTAssertTrue(select.waitForExistence(timeout: 20))
        select.tap()
        app.buttons.matching(identifier: "media.cell").firstMatch.tap()
        XCTAssertEqual(app.staticTexts["bulk.count"].label, "1 selected")
        app.buttons["People"].tap()
        XCTAssertTrue(app.staticTexts["Not People Alpha?"].waitForExistence(timeout: 20),
                      "the sheet does not offer to fix People Alpha on the selection")
        XCTAssertTrue(app.buttons["peopleSheet.tag"].exists)
        snap("people-selection-sheet")
        app.buttons["Close"].tap()
    }
}
