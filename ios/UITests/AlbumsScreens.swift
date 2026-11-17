import XCTest

/// The albums area's screens against the seeded server (ci/seeds/albums.py): the album list with its sort and
/// search, an album's page and its actions, making, renaming and deleting an album, Add to album, making a share
/// link, and the list of share links with Edit and Revoke. Tests change only the albums seed's own albums and links
/// (and an album they make and delete), never the core's "Weekend trip", "Hills" or their share.
final class AlbumsScreens: ScreenTestCase {
    /// An album's card on the Albums screen: one button whose label starts with the album's name.
    private func card(_ name: String) -> XCUIElement {
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH %@", "\(name),")).firstMatch
    }

    /// Taps the destructive button of a confirmation dialog (an action sheet on the phone).
    private func confirm(_ label: String) {
        let inSheet = app.sheets.buttons[label]
        if inSheet.waitForExistence(timeout: 5) {
            inSheet.tap()
        } else {
            app.buttons.matching(NSPredicate(format: "label == %@", label)).firstMatch.tap()
        }
    }

    private func waitUntil(_ element: XCUIElement, _ format: String, timeout: TimeInterval = 15) {
        let done = expectation(for: NSPredicate(format: format), evaluatedWith: element)
        wait(for: [done], timeout: timeout)
    }

    /// Opens the album page's menu and taps `item` in it. The page draws its toolbar again when the album reloads
    /// (the cloud change of a new or renamed album can arrive just then), which closes a menu that had just opened,
    /// so the menu is opened again until the item shows.
    private func tapInAlbumMenu(_ item: String) {
        let entry = app.buttons[item]
        for _ in 0..<3 {
            app.buttons["albums.menu"].tap()
            if entry.waitForExistence(timeout: 5) {
                entry.tap()
                return
            }
        }
        XCTFail("the album's menu did not show \(item)")
    }

    func testAlbumListSortAndSearch() throws {
        try launchSignedIn()
        tab("Albums")
        XCTAssertTrue(card("Weekend trip").waitForExistence(timeout: 20), "the core's album is not listed")
        XCTAssertTrue(card("Hills").exists)
        XCTAssertTrue(card("Albums test: Beach").exists)
        XCTAssertTrue(app.staticTexts["albums.count"].exists)
        snap("albums-list")

        app.buttons["albums.sort"].tap()
        app.buttons["Name"].tap()
        let first = app.buttons.matching(identifier: "albums.card").firstMatch
        waitUntil(first, "label BEGINSWITH 'Albums test: Beach,'")
        snap("albums-sorted-by-name")

        let search = app.searchFields.firstMatch
        XCTAssertTrue(search.waitForExistence(timeout: 10))
        search.tap()
        search.typeText("Hill")
        waitUntil(card("Weekend trip"), "exists == false")
        XCTAssertTrue(card("Hills").waitForExistence(timeout: 10))
        snap("albums-search")
    }

    func testAlbumPageAndShareLink() throws {
        try launchSignedIn()
        tab("Albums")
        let beach = card("Albums test: Beach")
        XCTAssertTrue(beach.waitForExistence(timeout: 20))
        beach.tap()
        XCTAssertTrue(app.navigationBars["Albums test: Beach"].waitForExistence(timeout: 15))
        let header = app.descendants(matching: .any)["albums.header"]
        XCTAssertTrue(header.waitForExistence(timeout: 10))
        XCTAssertTrue(header.label.contains("Made by the albums seed"), header.label)
        XCTAssertTrue(header.label.contains("3 items"), header.label)
        XCTAssertTrue(app.buttons.matching(identifier: "media.cell").firstMatch.waitForExistence(timeout: 15),
                      "the album's photos are not shown")
        snap("albums-page")

        app.buttons["albums.menu"].tap()
        XCTAssertTrue(app.buttons["Share album"].waitForExistence(timeout: 5))
        snap("albums-page-menu")
        app.buttons["Share album"].tap()

        XCTAssertTrue(app.navigationBars["Share"].waitForExistence(timeout: 10))
        let title = app.textFields["sharing.title"]
        XCTAssertTrue(title.waitForExistence(timeout: 5))
        XCTAssertEqual(title.value as? String, "Albums test: Beach", "the album's name is the link's title")
        let password = app.secureTextFields["sharing.password"]
        password.tap()
        password.typeText("beach")
        snap("albums-sharing-form")
        app.buttons["sharing.create"].tap()

        let link = app.staticTexts["sharing.link"]
        XCTAssertTrue(link.waitForExistence(timeout: 15), "the new link is not shown")
        XCTAssertTrue((link.value as? String ?? "").contains("/s/"), "\(String(describing: link.value))")
        XCTAssertTrue(app.staticTexts.containing(NSPredicate(format: "label CONTAINS 'after entering the password'"))
            .firstMatch.exists)
        snap("albums-sharing-created")
        app.buttons["sharing.done"].tap()
        XCTAssertTrue(app.navigationBars["Albums test: Beach"].waitForExistence(timeout: 10))
    }

    func testMakeRenameAndDeleteAnAlbum() throws {
        try launchSignedIn()
        tab("Albums")
        XCTAssertTrue(card("Weekend trip").waitForExistence(timeout: 20))
        let stamp = String(Int(Date().timeIntervalSince1970) % 100_000)
        let made = "Albums test new \(stamp)"
        let renamed = "Albums test renamed \(stamp)"

        app.buttons["albums.new"].tap()
        let name = app.textFields["albums.edit.name"]
        XCTAssertTrue(name.waitForExistence(timeout: 10))
        name.tap()
        name.typeText(made)
        snap("albums-new")
        app.buttons["albums.edit.save"].tap()
        XCTAssertTrue(app.navigationBars[made].waitForExistence(timeout: 15), "the new album did not open")
        snap("albums-new-page")

        tapInAlbumMenu("Rename")
        let field = app.textFields["albums.edit.name"]
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        field.tap()
        let current = field.value as? String ?? ""
        field.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: current.count))
        field.typeText(renamed)
        let description = app.descendants(matching: .any)["albums.edit.description"]
        if description.waitForExistence(timeout: 3) {
            description.tap()
            description.typeText("Made by the albums test")
        }
        snap("albums-rename")
        app.buttons["albums.edit.save"].tap()
        XCTAssertTrue(app.navigationBars[renamed].waitForExistence(timeout: 15), "the album was not renamed")

        tapInAlbumMenu("Delete album")
        confirm("Delete album")
        XCTAssertTrue(app.navigationBars["Albums"].waitForExistence(timeout: 15), "deleting did not go back to Albums")
        waitUntil(card(renamed), "exists == false")
        snap("albums-deleted")
    }

    func testAddToAlbum() throws {
        try launchSignedIn(["PCC_TEST_ALBUMS_ADD_FROM": "Albums test: Beach"])
        tab("Albums")
        XCTAssertTrue(app.navigationBars["Add 3 items to album"].waitForExistence(timeout: 20),
                      "Add to album did not open for the seeded album's three photos")
        let shelf = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Albums test: Shelf,'")).firstMatch
        XCTAssertTrue(shelf.waitForExistence(timeout: 15))
        XCTAssertTrue(app.buttons["albums.add.new"].exists)
        snap("albums-add")
        // Create a new album opens the name field; its Cancel folds it back without closing the sheet
        app.buttons["albums.add.new"].tap()
        XCTAssertTrue(app.textFields["albums.add.newName"].waitForExistence(timeout: 10), "no field for the new name")
        snap("albums-add-new-name")
        app.buttons["albums.add.cancelNew"].tap()
        XCTAssertTrue(app.buttons["albums.add.new"].waitForExistence(timeout: 10), "Cancel did not fold the field back")
        XCTAssertTrue(app.navigationBars["Add 3 items to album"].exists, "Cancel closed the whole sheet")
        shelf.tap()
        let toast = app.descendants(matching: .any)["toast"]
        XCTAssertTrue(toast.waitForExistence(timeout: 15), "no message after adding")
        XCTAssertTrue(toast.label.contains("Albums test: Shelf"), toast.label)
        snap("albums-added")
    }

    func testShareLinksEditAndRevoke() throws {
        try launchSignedIn()
        tab("More")
        app.buttons["more.settings"].tap()
        app.buttons["settings.sharing"].tap()
        XCTAssertTrue(app.navigationBars["Share links"].waitForExistence(timeout: 15))
        let core = app.descendants(matching: .any).matching(NSPredicate(format: "label CONTAINS 'Weekend trip'")).firstMatch
        XCTAssertTrue(core.waitForExistence(timeout: 15), "the core's share is not listed")
        snap("albums-sharing-list")

        let editMenu = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Actions for Albums test link'")).firstMatch
        XCTAssertTrue(editMenu.waitForExistence(timeout: 10))
        editMenu.tap()
        app.buttons["Edit"].tap()
        let title = app.textFields["sharing.edit.title"]
        XCTAssertTrue(title.waitForExistence(timeout: 10))
        title.tap()
        let current = title.value as? String ?? ""
        title.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: current.count))
        title.typeText("Albums test link edited")
        snap("albums-sharing-edit")
        app.buttons["sharing.edit.save"].tap()
        let edited = app.buttons["Actions for Albums test link edited"]
        XCTAssertTrue(edited.waitForExistence(timeout: 15), "the edited title is not shown")

        let revokeMenu = app.buttons["Actions for Albums test revoke"]
        if revokeMenu.waitForExistence(timeout: 5) {
            revokeMenu.tap()
            app.buttons["Revoke"].tap()
            confirm("Revoke")
            waitUntil(revokeMenu, "exists == false")
        }
        let revoked = app.descendants(matching: .any)
            .matching(NSPredicate(format: "label CONTAINS 'Albums test revoke' AND label CONTAINS 'Revoked'")).firstMatch
        XCTAssertTrue(revoked.waitForExistence(timeout: 15), "the revoked link does not say Revoked")
        snap("albums-sharing-revoked")
    }
}
