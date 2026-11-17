import XCTest
@testable import PersonalContentCloud

/// The library's answers from the seeded server decode into the generated types, with the filters the app sends:
/// the month skeleton and a month's page (section 23), searches in each mode, and the seed's tag
/// (ci/seeds/library.py). Skipped without the workflow's server and token.
final class LibraryDecodingTests: XCTestCase {
    private func client() throws -> APIClient {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["PCC_TEST_SERVER"], let base = URL(string: address), let token = env["PCC_TEST_TOKEN"],
              !token.isEmpty else {
            throw XCTSkip("PCC_TEST_SERVER and PCC_TEST_TOKEN are not set (the iOS workflow sets them)")
        }
        return APIClient(base: base, token: token)
    }

    func testTimelineDecodesWithTheLibrarysFilters() async throws {
        let api = try client()
        let all = try await api.buckets()
        XCTAssertFalse(all.buckets.isEmpty)
        XCTAssertEqual(all.total, all.buckets.reduce(0) { $0 + $1.count })
        let page = try await api.bucket(key: all.buckets[0].key, limit: 200)
        XCTAssertFalse(page.items.isEmpty)

        var filters = LibraryFilters()
        filters.tag = ["Library sample"]
        let f = filters.normalized(for: .all)
        let tagged = try await api.buckets(category: f.category, kind: f.kind, person: f.person, tag: f.tag,
                                           live: f.live, from: f.from, to: f.to, favorite: f.favorite,
                                           misclicks: f.misclicks, forwards: f.forwards, received: f.received,
                                           dups: f.dups, sort: f.sort)
        XCTAssertEqual(tagged.total, 3, "the seed tags three of its photos")
        XCTAssertEqual(tagged.buckets.map(\.key), ["2026-02", "2026-01"])

        let withMisclicks = try await api.buckets(misclicks: .include)
        XCTAssertGreaterThan(withMisclicks.total, all.total, "the seed's possible mis-click is left out by default")
    }

    func testSearchDecodes() async throws {
        let api = try client()
        let auto = try await api.searchItems(q: "lantern", limit: 60)
        XCTAssertGreaterThanOrEqual(auto.total, 1)
        let text = try await api.searchItems(q: "packing", mode: .text)
        XCTAssertTrue(text.items.contains { ($0.snippet ?? "").contains("<b>") }, "a text match carries its snippet")
        do {
            _ = try await api.searchItems(q: "lantern", mode: .semantic)
        } catch let error as APIError {
            // without the search model (the CI server) a semantic search is refused, as the website expects
            XCTAssertEqual(error.status, 503)
        }
    }
}
