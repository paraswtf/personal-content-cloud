import XCTest

/// The Review tab against the seeded server (ios/ci/seeds/review.py). Every change these tests make is to the review
/// area's own rows: the guesses are narrowed to "Review Person" before any is answered, and the bins' items are
/// picked by reasons only this area's seed has.
final class ReviewScreens: ScreenTestCase {
    private func toast(_ text: String, timeout: TimeInterval = 15) -> Bool {
        let match = NSPredicate(format: "identifier == 'toast' AND label CONTAINS %@", text)
        return app.descendants(matching: .any).matching(match).firstMatch.waitForExistence(timeout: timeout)
    }

    private func button(_ identifier: String) -> XCUIElement {
        app.buttons.matching(identifier: identifier).firstMatch
    }

    private func pickInPicker(_ name: String) {
        let search = app.searchFields.firstMatch
        if search.waitForExistence(timeout: 10) {
            search.tap()
            search.typeText(name)
        }
        let option = app.buttons.matching(NSPredicate(format: "identifier == 'picker.option' AND label BEGINSWITH %@", name))
            .firstMatch
        XCTAssertTrue(option.waitForExistence(timeout: 20), "\(name) is not in the picker")
        option.tap()
    }

    func testReviewGuesses() throws {
        try launchSignedIn()
        tab("Review")
        button("review.tab.guesses").tap()
        let yes = button("review.guess.yes")
        XCTAssertTrue(yes.waitForExistence(timeout: 30), "no guess cards")
        snap("review-guesses")

        // the basis filter: the grouped face reads as the website words it
        button("review.basis.group").tap()
        XCTAssertTrue(app.staticTexts["Grouped, face 0.62"].waitForExistence(timeout: 20), "no grouped guess")
        XCTAssertTrue(app.staticTexts["grouped with the face you named Review Person"].exists)
        snap("review-guesses-grouped")
        button("review.basis.trait").tap()
        XCTAssertTrue(app.staticTexts["By tattoo, strong match"].waitForExistence(timeout: 20), "no tattoo guess")
        snap("review-guesses-tattoo")
        button("review.basis.all").tap()

        // from here on only this area's guesses: the ones of Review Person
        button("review.person").tap()
        pickInPicker("Review Person")
        XCTAssertTrue(button("review.person.clear").waitForExistence(timeout: 10), "the person filter did not apply")
        XCTAssertTrue(yes.waitForExistence(timeout: 20))
        snap("review-guesses-person")

        // the photo of the first card opens in the viewer
        button("review.guess.open").tap()
        XCTAssertTrue(app.buttons["viewer.close"].waitForExistence(timeout: 20), "the viewer did not open")
        snap("review-guess-viewer")
        app.buttons["viewer.close"].tap()

        // Yes, No and Someone else on the first card each time
        XCTAssertTrue(yes.waitForExistence(timeout: 20))
        yes.tap()
        XCTAssertTrue(toast("Tagged as Review Person."), "no message after Yes")
        let no = button("review.guess.no")
        XCTAssertTrue(no.waitForExistence(timeout: 20))
        no.tap()
        XCTAssertTrue(toast("Guess rejected."), "no message after No")
        let other = button("review.guess.other")
        XCTAssertTrue(other.waitForExistence(timeout: 20))
        other.tap()
        XCTAssertTrue(app.navigationBars["Someone else"].waitForExistence(timeout: 20), "the picker did not open")
        snap("review-someone-else")
        pickInPicker("Review Other")
        XCTAssertTrue(toast("Tagged as Review Other."), "no message after Someone else")

        // Confirm all shown, for the guesses of Review Person that are left
        let confirmAll = button("review.confirmAll")
        if confirmAll.waitForExistence(timeout: 10) {
            confirmAll.tap()
            let confirm = app.buttons["Confirm all"]
            XCTAssertTrue(confirm.waitForExistence(timeout: 10))
            confirm.tap()
            XCTAssertTrue(toast("confirmed."), "no message after Confirm all")
        }
        snap("review-guesses-done")
    }

    func testReviewBins() throws {
        try launchSignedIn()
        tab("Review")

        // mis-clicks: keep this area's pocket shot
        button("review.tab.misclicks").tap()
        let pocket = app.descendants(matching: .any)
            .matching(NSPredicate(format: "identifier == 'review.cell' AND label CONTAINS 'Pocket shot'")).firstMatch
        XCTAssertTrue(pocket.waitForExistence(timeout: 30), "the seeded mis-click is not listed")
        snap("review-misclicks")
        pocket.press(forDuration: 0.6)
        XCTAssertTrue(app.staticTexts["bulk.count"].waitForExistence(timeout: 10))
        XCTAssertEqual(app.staticTexts["bulk.count"].label, "1 selected")
        snap("review-misclicks-selected")
        app.buttons["Keep"].tap()
        XCTAssertTrue(toast("1 item kept."), "no message after Keep")

        // forwards: delete this area's Telegram forward
        button("review.tab.forwards").tap()
        let forward = app.descendants(matching: .any)
            .matching(NSPredicate(format: "identifier == 'review.cell' AND label CONTAINS 'Filename from Telegram'")).firstMatch
        XCTAssertTrue(forward.waitForExistence(timeout: 30), "the seeded forward is not listed")
        snap("review-forwards")
        forward.press(forDuration: 0.6)
        app.buttons["Delete"].tap()
        let moveToTrash = app.buttons["Move to trash"]
        XCTAssertTrue(moveToTrash.waitForExistence(timeout: 10), "no confirmation before deleting")
        moveToTrash.tap()
        XCTAssertTrue(toast("1 item moved to the trash."), "no message after Delete")

        // duplicates: make this area's second copy the best one
        button("review.tab.duplicates").tap()
        let makeBest = app.buttons.matching(NSPredicate(format: "identifier == 'review.dup.best' AND label CONTAINS 'REVIEW_DUP'"))
            .firstMatch
        XCTAssertTrue(makeBest.waitForExistence(timeout: 30), "the seeded duplicates are not listed")
        snap("review-duplicates")
        makeBest.tap()
        XCTAssertTrue(toast("Best copy changed."), "no message after Make best")
        snap("review-duplicates-best")
    }
}
