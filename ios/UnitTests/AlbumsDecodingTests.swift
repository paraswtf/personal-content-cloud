import XCTest
@testable import PersonalContentCloud

/// The albums area's answers from the seeded server (ci/seeds/albums.py) decode into the generated types: the album
/// list in every sort, each album, the albums' items, and the share links. Skipped without the workflow's server.
final class AlbumsDecodingTests: XCTestCase {
    func testAlbumsAndSharesDecode() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["PCC_TEST_SERVER"], let base = URL(string: address), let token = env["PCC_TEST_TOKEN"],
              !token.isEmpty else {
            throw XCTSkip("PCC_TEST_SERVER and PCC_TEST_TOKEN are not set (the iOS workflow sets them)")
        }
        let api = APIClient(base: base, token: token)

        for sort in API.ListAlbumsSort.allCases {
            let page = try await api.listAlbums(sort: sort, limit: 60)
            XCTAssertGreaterThanOrEqual(page.total, 4, "the core's two albums and the albums seed's two")
        }
        let page = try await api.listAlbums(sort: .name, limit: 60)
        let names = page.items.map(\.name)
        for name in ["Weekend trip", "Hills", "Albums test: Beach", "Albums test: Shelf"] {
            XCTAssertTrue(names.contains(name), "\(name) is missing from \(names)")
        }
        let found = try await api.listAlbums(q: "Albums test: Bea", limit: 10)
        XCTAssertEqual(found.items.map(\.name), ["Albums test: Beach"])

        for album in page.items {
            let one = try await api.getAlbum(albumId: album.id)
            XCTAssertEqual(one.id, album.id)
            _ = try await api.listItems(album: album.id, limit: 50)
        }
        if let beach = page.items.first(where: { $0.name == "Albums test: Beach" }) {
            XCTAssertEqual(beach.count, 3)
            XCTAssertNotNil(beach.coverUrl)
            XCTAssertNotNil(beach.oldest)
        }

        let shares = try await api.listShares()
        let titles = shares.map(\.title)
        XCTAssertTrue(titles.contains("Weekend trip"), "\(titles)")
        XCTAssertTrue(titles.contains(where: { $0.hasPrefix("Albums test link") }), "\(titles)")
        for share in shares {
            let one = try await api.getShare(shareId: share.id)
            XCTAssertEqual(one.path, share.path)
            XCTAssertTrue(share.path.hasPrefix("/s/"))
        }
    }
}
