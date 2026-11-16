import XCTest

/// The map and documents screens against the seeded server (ci/seeds/mapdocs.py adds places and documents).
final class MapDocsScreens: ScreenTestCase {
    private func openMore(_ identifier: String, title: String) {
        tab("More")
        let row = app.buttons[identifier]
        XCTAssertTrue(row.waitForExistence(timeout: 15), "no \(identifier) row under More")
        row.tap()
        XCTAssertTrue(app.navigationBars[title].waitForExistence(timeout: 15), "\(title) did not open")
    }

    private var markers: XCUIElementQuery {
        app.descendants(matching: .any)
            .matching(NSPredicate(format: "identifier == 'map.cluster' OR identifier == 'map.point'"))
    }

    private func waitUntilGone(_ element: XCUIElement, timeout: TimeInterval = 10) -> Bool {
        let gone = expectation(for: NSPredicate(format: "exists == false"), evaluatedWith: element)
        return XCTWaiter().wait(for: [gone], timeout: timeout) == .completed
    }

    func testMapClustersOpenTheirItems() throws {
        try launchSignedIn()
        openMore("more.map", title: "Map")
        let count = app.staticTexts["map.count"]
        XCTAssertTrue(count.waitForExistence(timeout: 20))
        let loaded = expectation(for: NSPredicate(format: "label CONTAINS 'location'"), evaluatedWith: count)
        XCTAssertEqual(XCTWaiter().wait(for: [loaded], timeout: 30), .completed, "the count line never said how many have a location")
        XCTAssertTrue(markers.firstMatch.waitForExistence(timeout: 30), "no cluster or point on the map")
        XCTAssertTrue(app.staticTexts["Place names are off: the GeoNames files are missing from the models folder (see the README)."].waitForExistence(timeout: 10),
                      "the status line about place names did not show (the CI server has no gazetteer)")
        snap("mapdocs-map")

        // a big cluster zooms in first; a small one or a point opens the panel
        let title = app.staticTexts["map.panel.title"]
        for _ in 0..<5 where !title.exists {
            let marker = markers.firstMatch
            guard marker.waitForExistence(timeout: 10) else { break }
            marker.tap()
            _ = title.waitForExistence(timeout: 4)
        }
        XCTAssertTrue(title.waitForExistence(timeout: 10), "tapping the markers never opened the items panel")
        XCTAssertTrue(title.label.hasSuffix("here"), "the panel's title is \(title.label)")
        let cell = app.buttons.matching(identifier: "media.cell").firstMatch
        XCTAssertTrue(cell.waitForExistence(timeout: 20), "the panel shows no items")
        snap("mapdocs-map-panel")

        cell.tap()
        let close = app.buttons["viewer.close"]
        XCTAssertTrue(close.waitForExistence(timeout: 20), "the viewer did not open from the panel")
        close.tap()
        XCTAssertTrue(title.waitForExistence(timeout: 10), "the panel did not stay after the viewer closed")
        app.buttons["map.panel.close"].tap()
        XCTAssertTrue(waitUntilGone(title), "the panel did not close")
    }

    func testPlacesFlyTheMap() throws {
        try launchSignedIn()
        openMore("more.map", title: "Map")
        let places = app.buttons["map.places"]
        XCTAssertTrue(places.waitForExistence(timeout: 20))
        let counted = expectation(for: NSPredicate(format: "label BEGINSWITH 'Places ('"), evaluatedWith: places)
        XCTAssertEqual(XCTWaiter().wait(for: [counted], timeout: 30), .completed, "the places never loaded")
        places.tap()
        XCTAssertTrue(app.navigationBars["Places"].waitForExistence(timeout: 10))
        let panaji = app.buttons.matching(NSPredicate(format: "identifier == 'map.place' AND label BEGINSWITH 'Panaji'")).firstMatch
        XCTAssertTrue(panaji.waitForExistence(timeout: 20), "the seeded place is not listed")
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "identifier == 'map.place' AND label BEGINSWITH 'Shimla'")).firstMatch.exists)
        snap("mapdocs-places")
        panaji.tap()
        XCTAssertTrue(waitUntilGone(app.navigationBars["Places"]), "picking a place did not close the list")
        // zoomed in on Panaji: the two photos there show as thumbnails or a small cluster
        XCTAssertTrue(markers.firstMatch.waitForExistence(timeout: 20), "nothing on the map at the chosen place")
        snap("mapdocs-place-picked")
    }

    func testDocumentsFilterAndSearch() throws {
        try launchSignedIn()
        openMore("more.documents", title: "Documents")
        let pan = app.buttons.matching(NSPredicate(format: "identifier == 'documents.card' AND label BEGINSWITH 'Open MapDocs PAN card'")).firstMatch
        let bill = app.buttons.matching(NSPredicate(format: "identifier == 'documents.card' AND label BEGINSWITH 'Open MapDocs electricity bill'")).firstMatch
        // newest first: the bill is on the first screen, the PAN card (the oldest) further down
        XCTAssertTrue(bill.waitForExistence(timeout: 30), "the seeded bill is not listed")
        snap("mapdocs-documents")

        let identity = app.buttons["documents.group.id"]
        XCTAssertTrue(identity.waitForExistence(timeout: 10), "no Identity chip")
        identity.tap()
        XCTAssertTrue(waitUntilGone(bill), "the Identity group still lists the bill")
        XCTAssertTrue(pan.waitForExistence(timeout: 10), "the Identity group does not list the PAN card")
        XCTAssertTrue((pan.value as? String ?? "").contains("XXXXXX234F"), "the identifier is not masked: \(pan.value ?? "")")
        snap("mapdocs-documents-identity")
        app.buttons["documents.all"].tap()
        XCTAssertTrue(bill.waitForExistence(timeout: 15), "All did not bring the bill back")

        let search = app.searchFields.firstMatch
        XCTAssertTrue(search.waitForExistence(timeout: 10), "no search field")
        search.tap()
        search.typeText("electricity\n")
        XCTAssertTrue(waitUntilGone(pan, timeout: 15), "the search still lists the PAN card")
        XCTAssertTrue(bill.waitForExistence(timeout: 10), "the search did not find the bill")
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "identifier == 'documents.count' AND label CONTAINS 'matching'")).firstMatch.exists)
        snap("mapdocs-documents-search")

        bill.tap()
        let close = app.buttons["viewer.close"]
        XCTAssertTrue(close.waitForExistence(timeout: 20), "the viewer did not open from a card")
        snap("mapdocs-documents-viewer")
        close.tap()
        XCTAssertTrue(bill.waitForExistence(timeout: 10))
    }
}
