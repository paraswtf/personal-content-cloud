import XCTest
@testable import PersonalContentCloud

/// The viewer opened from a person's faces, crops or guesses (PeopleViewer): each item once, at the item tapped, a
/// video paused at the tapped face's second, and paging that goes on with the list's next page.
@MainActor
final class PeopleViewerTests: XCTestCase {
    func testOpensAtTheTappedItemAndItsSecond() {
        let router = AppRouter()
        let entries = [ViewerEntry(uid: "a"), ViewerEntry(uid: "b", kind: "video"), ViewerEntry(uid: "a"),
                       ViewerEntry(uid: "c")]
        PeopleViewer.open(router, entries: entries, start: "b", startAt: 12.5) { [] }
        guard let request = router.viewer else { return XCTFail("the viewer did not open") }
        XCTAssertEqual(request.entries.map(\.uid), ["a", "b", "c"])
        XCTAssertEqual(request.start, 1)
        XCTAssertEqual(request.entries[1].startAt, 12.5)
        XCTAssertNil(request.entries[0].startAt)
        XCTAssertNil(request.entries[2].startAt)
        XCTAssertNotNil(request.loadMore, "paging stops at the rows loaded")
    }

    func testAPhotoOrAStartOfZeroOpensWithoutASecond() {
        let router = AppRouter()
        PeopleViewer.open(router, entries: [ViewerEntry(uid: "a"), ViewerEntry(uid: "b")], start: "b", startAt: nil)
        XCTAssertNil(router.viewer?.entries[1].startAt)
        XCTAssertNil(router.viewer?.loadMore)
        PeopleViewer.open(router, entries: [ViewerEntry(uid: "a")], start: "a", startAt: 0)
        XCTAssertNil(router.viewer?.entries[0].startAt)
    }

    func testNothingToShowOpensNothing() {
        let router = AppRouter()
        PeopleViewer.open(router, entries: [], start: "a", startAt: 3)
        XCTAssertNil(router.viewer)
    }

    func testMoreLoadsUntilANewItemComes() async {
        // three pages: the second holds only faces of an item already shown (a video with many faces)
        let pages: [String?: ([PeopleViewerTestRow], String?)] = [
            nil: ([.init(id: 1, uid: "a"), .init(id: 2, uid: "b")], "p2"),
            "p2": ([.init(id: 3, uid: "b"), .init(id: 4, uid: "a")], "p3"),
            "p3": ([.init(id: 5, uid: "c")], nil),
        ]
        let pager = Pager<PeopleViewerTestRow> { cursor in
            let page = pages[cursor] ?? ([], nil)
            return .init(items: page.0, next: page.1)
        }
        await pager.refresh()
        let all = await PeopleViewer.more(pager) { ViewerEntry(uid: $0.uid) }
        XCTAssertEqual(all.map(\.uid), ["a", "b", "c"])
        XCTAssertTrue(pager.finished)
    }
}

struct PeopleViewerTestRow: Identifiable {
    let id: Int
    let uid: String
}
