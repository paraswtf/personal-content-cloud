import XCTest
@testable import PersonalContentCloud

/// The viewer's counter (the website's Viewer position, useViewerParam): the whole list's total where the screen that
/// opened the viewer knows it, else the items loaded with a "+" while more can come.
final class ViewerCounterTests: XCTestCase {
    func testAKnownTotalCountsTheWholeList() {
        let counter = ViewerCounter.make(index: 2, place: nil, loaded: 40, total: 120, moreMayCome: true)
        XCTAssertEqual(counter, ViewerCounter(number: 3, of: 120, more: false))
        XCTAssertEqual(counter.text, "3 of 120")
        XCTAssertEqual(counter.spoken, "Item 3 of 120")
        XCTAssertTrue(counter.shown)
    }

    func testWithoutATotalItCountsWhatIsLoadedWithAPlusWhileMoreCanCome() {
        let more = ViewerCounter.make(index: 2, place: nil, loaded: 40, total: nil, moreMayCome: true)
        XCTAssertEqual(more.text, "3 of 40+")
        XCTAssertEqual(more.spoken, "Item 3 of at least 40")
        let all = ViewerCounter.make(index: 2, place: nil, loaded: 40, total: nil, moreMayCome: false)
        XCTAssertEqual(all.text, "3 of 40")
        XCTAssertEqual(all.spoken, "Item 3 of 40")
    }

    func testThePlaceInTheWholeListWinsOverTheEntries() {
        // the library's months loaded so far leave a gap (the scrubber jumped): the item is the 5,000th, not the 41st
        let counter = ViewerCounter.make(index: 40, place: 4999, loaded: 60, total: 6000, moreMayCome: true)
        XCTAssertEqual(counter, ViewerCounter(number: 5000, of: 6000, more: false))
        XCTAssertEqual(counter.text, "\(5000.formatted()) of \(6000.formatted())")
        XCTAssertEqual(counter.spoken, "Item 5000 of 6000")
        // its place unknown (its month went in a refresh): its place among the entries
        XCTAssertEqual(ViewerCounter.make(index: 40, place: nil, loaded: 60, total: 6000, moreMayCome: true).number, 41)
    }

    func testATotalIsNeverSmallerThanWhatIsLoadedOrTheItemsOwnPlace() {
        XCTAssertEqual(ViewerCounter.make(index: 0, place: nil, loaded: 50, total: 30, moreMayCome: false).of, 50)
        XCTAssertEqual(ViewerCounter.make(index: 0, place: 99, loaded: 10, total: 50, moreMayCome: false).of, 100)
        XCTAssertEqual(ViewerCounter.make(index: 4, place: nil, loaded: 5, total: 0, moreMayCome: false).of, 5)
    }

    func testOneItemShowsNoCounter() {
        XCTAssertFalse(ViewerCounter.make(index: 0, place: nil, loaded: 1, total: nil, moreMayCome: false).shown)
        XCTAssertFalse(ViewerCounter.make(index: 0, place: nil, loaded: 1, total: 1, moreMayCome: false).shown)
        XCTAssertFalse(ViewerCounter.make(index: 0, place: nil, loaded: 1, total: nil, moreMayCome: true).shown,
                       "one item loaded and more may come: as before, no counter until a second one is there")
        // one loaded of a list the screen knows is longer: counted
        XCTAssertEqual(ViewerCounter.make(index: 0, place: nil, loaded: 1, total: 120, moreMayCome: true).text, "1 of 120")
    }

    func testTheRequestCarriesTheTotalAndThePlace() {
        let entries = (0..<3).map { ViewerEntry(uid: "u\($0)") }
        let request = ViewerRequest(entries: entries, start: 1, total: 120, positionOf: { $0 == "u1" ? 41 : nil })
        XCTAssertEqual(request.total, 120)
        XCTAssertEqual(request.positionOf?("u1"), 41)
        XCTAssertNil(ViewerRequest(entries: entries, start: 0).total, "a screen that knows no total passes none")
    }
}
