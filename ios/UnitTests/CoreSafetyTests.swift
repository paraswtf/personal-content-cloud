import XCTest
@testable import PersonalContentCloud

/// The core's rules that keep the token on the server's host and keep lists from stopping silently.
@MainActor
final class CoreSafetyTests: XCTestCase {
    private struct Row: Identifiable {
        let id: Int
    }

    func testRedirectsStayOnTheServersHost() throws {
        let from = try XCTUnwrap(URL(string: "https://cloud.example/api/items"))
        func allowed(_ to: String) throws -> Bool {
            SameHostRedirects.allowed(from: from, to: try XCTUnwrap(URL(string: to)))
        }
        XCTAssertTrue(try allowed("https://cloud.example/api/items/"))
        XCTAssertTrue(try allowed("https://CLOUD.example:443/files/thumb/a1"))
        XCTAssertFalse(try allowed("https://login.example/cdn-cgi/access/login"))
        XCTAssertFalse(try allowed("https://cloud.example.other.test/api/items"))
        XCTAssertFalse(try allowed("https://cloud.example:8443/api/items"))
        XCTAssertFalse(try allowed("http://cloud.example/api/items"), "a redirect to plain http would send the token in the clear")
        let plain = try XCTUnwrap(URL(string: "http://cloud.example/api/items"))
        XCTAssertTrue(SameHostRedirects.allowed(from: plain, to: try XCTUnwrap(URL(string: "https://cloud.example/api/items"))))
        XCTAssertFalse(SameHostRedirects.allowed(from: plain, to: try XCTUnwrap(URL(string: "https://elsewhere.example/"))))
    }

    func testAFailedLaterPageIsTriedAgainFromItsCursor() async {
        var asked: [String?] = []
        var failSecond = true
        let pager = Pager<Row> { cursor in
            asked.append(cursor)
            if cursor == nil { return .init(items: [Row(id: 0), Row(id: 1)], next: "p2") }
            if failSecond {
                failSecond = false
                throw URLError(.timedOut)
            }
            return .init(items: [Row(id: 2)], next: nil)
        }
        await pager.refresh()
        await pager.loadMore()
        XCTAssertNotNil(pager.error, "the failed page is kept as the pager's error, which the list shows")
        XCTAssertEqual(pager.items.map(\.id), [0, 1])
        await pager.retry()
        XCTAssertNil(pager.error)
        XCTAssertEqual(pager.items.map(\.id), [0, 1, 2])
        XCTAssertTrue(pager.finished)
        XCTAssertEqual(asked, [nil, "p2", "p2"])
    }

    func testAFailedRefreshIsTriedAgainFromTheFirstPage() async {
        var asked: [String?] = []
        var failFirst = false
        let pager = Pager<Row> { cursor in
            asked.append(cursor)
            if cursor == nil && failFirst {
                failFirst = false
                throw URLError(.notConnectedToInternet)
            }
            return cursor == nil ? .init(items: [Row(id: 0), Row(id: 1)], next: "p2") : .init(items: [Row(id: 9)], next: nil)
        }
        await pager.refresh()
        failFirst = true
        await pager.refresh()
        XCTAssertNotNil(pager.error)
        XCTAssertEqual(pager.items.map(\.id), [0, 1], "the rows from before stay while the refresh failed")
        await pager.retry()
        XCTAssertNil(pager.error)
        XCTAssertEqual(asked, [nil, nil, nil], "Try again asks for the first page again, not the next one")
    }
}
