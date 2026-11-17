import XCTest
@testable import PersonalContentCloud

/// The albums area's wording and rules (Features/Albums): date ranges, facts, the add message, share link states,
/// expiry text, descriptions and the share API's refusals.
final class AlbumsTests: XCTestCase {
    private func share(_ json: String) throws -> API.ShareOut {
        try JSONDecoder().decode(API.ShareOut.self, from: Data(json.utf8))
    }

    private func album(count: Int = 3, vaultCount: Int = 0, oldest: String?, newest: String?) throws -> API.AlbumOut {
        var fields = [
            "\"id\": 7", "\"name\": \"Beach\"", "\"description\": \"\"", "\"count\": \(count)",
            "\"vault_count\": \(vaultCount)", "\"cover_url\": null", "\"created_at\": \"2026-01-01T10:00:00.000Z\"",
            "\"updated_at\": \"2026-01-02T10:00:00.000Z\"",
        ]
        fields.append(oldest.map { "\"oldest\": \"\($0)\"" } ?? "\"oldest\": null")
        fields.append(newest.map { "\"newest\": \"\($0)\"" } ?? "\"newest\": null")
        return try JSONDecoder().decode(API.AlbumOut.self, from: Data("{\(fields.joined(separator: ", "))}".utf8))
    }

    func testDateRangeIsOneMonthOrAFromToRange() {
        let one = AlbumsText.dateRange(oldest: "2026-02-10T09:30:00.000Z", newest: "2026-02-12T09:30:00.000Z")
        XCTAssertTrue(one.contains("2026"))
        XCTAssertFalse(one.contains(" to "), one)
        let span = AlbumsText.dateRange(oldest: "2026-03-01T09:30:00.000Z", newest: "2026-11-02T09:30:00.000Z")
        XCTAssertTrue(span.contains(" to "), span)
        XCTAssertEqual(AlbumsText.dateRange(oldest: nil, newest: "2026-11-02T09:30:00.000Z"), "")
    }

    func testCardLineAndFacts() throws {
        let empty = try album(count: 0, oldest: nil, newest: nil)
        XCTAssertEqual(AlbumsText.cardLine(empty), "0 items")
        XCTAssertEqual(AlbumsText.facts(empty), "0 items")
        let full = try album(count: 1, vaultCount: 2, oldest: "2026-02-10T09:30:00.000Z", newest: "2026-02-12T09:30:00.000Z")
        XCTAssertTrue(AlbumsText.cardLine(full).hasPrefix("1 item, "))
        XCTAssertTrue(AlbumsText.facts(full).hasSuffix(", 2 private items in the vault"), AlbumsText.facts(full))
        XCTAssertTrue(AlbumsText.cardLabel(full).hasPrefix("Beach, 1 item"))
    }

    func testAddedMessage() {
        XCTAssertEqual(AlbumsText.added(3, to: "Goa"), "3 items added to Goa.")
        XCTAssertEqual(AlbumsText.added(1, to: "Goa"), "1 item added to Goa.")
        XCTAssertEqual(AlbumsText.added(0, to: "Goa"), "No items were added to Goa. They are in it already.")
    }

    func testAlbumRefusal() {
        XCTAssertEqual(AlbumsText.error(APIError(status: 409, detail: "album_exists"), doing: "create the album"),
                       "An album with this name exists already.")
        XCTAssertTrue(AlbumsText.error(APIError(status: 500, detail: "boom"), doing: "create the album")
            .hasPrefix("Could not create the album. "))
    }

    func testShareStatusAndExpiry() throws {
        let now = ISODate.parse("2026-11-10T12:00:00.000Z")!
        let base = """
        "id": 1, "token": "t", "path": "/s/t", "kind": "items", "album": null, "title": "", "count": 2,
        "has_password": false, "allow_download": true, "created_at": "2026-11-01T10:00:00.000Z", "view_count": 0,
        "last_viewed_at": null
        """
        let live = try share("{\(base), \"expires_at\": null, \"revoked_at\": null}")
        XCTAssertEqual(SharingText.status(live, now: now), .active)
        let past = try share("{\(base), \"expires_at\": \"2026-11-09T12:00:00.000Z\", \"revoked_at\": null}")
        XCTAssertEqual(SharingText.status(past, now: now), .expired)
        let gone = try share("{\(base), \"expires_at\": null, \"revoked_at\": \"2026-11-05T12:00:00.000Z\"}")
        XCTAssertEqual(SharingText.status(gone, now: now), .revoked)

        XCTAssertEqual(SharingText.expiry(nil, now: now), "Never expires")
        XCTAssertEqual(SharingText.expiry("2026-11-10T11:00:00.000Z", now: now), "Expired")
        XCTAssertEqual(SharingText.expiry("2026-11-10T13:00:00.000Z", now: now), "Expires in 1 hour")
        XCTAssertEqual(SharingText.expiry("2026-11-10T17:00:00.000Z", now: now), "Expires in 5 hours")
        XCTAssertEqual(SharingText.expiry("2026-11-17T12:00:00.000Z", now: now), "Expires in 7 days")
    }

    func testShareDescriptions() throws {
        let now = ISODate.parse("2026-11-10T12:00:00.000Z")!
        let album = try share("""
        {"id": 2, "token": "t", "path": "/s/t", "kind": "album", "album": {"id": 4, "name": "Goa"}, "title": "Trip",
         "count": 12, "has_password": true, "expires_at": null, "allow_download": false,
         "created_at": "2026-11-01T10:00:00.000Z", "revoked_at": null, "view_count": 1, "last_viewed_at": null}
        """)
        XCTAssertEqual(SharingText.describe(album), "Album: Goa (12 items)")
        let line = SharingText.details(album, now: now)
        XCTAssertTrue(line.hasPrefix("Album: Goa (12 items). Created "), line)
        XCTAssertTrue(line.contains(". 1 view. Never expires."), line)
        XCTAssertTrue(line.hasSuffix(" Downloads off."), line)

        let items = try share("""
        {"id": 3, "token": "u", "path": "/s/u", "kind": "items", "album": null, "title": "", "count": 1,
         "has_password": false, "expires_at": null, "allow_download": true, "created_at": "2026-11-01T10:00:00.000Z",
         "revoked_at": "2026-11-02T10:00:00.000Z", "view_count": 3, "last_viewed_at": "2026-11-10T11:00:00.000Z"}
        """)
        XCTAssertEqual(SharingText.describe(items), "1 item")
        let revoked = SharingText.details(items, now: now)
        XCTAssertTrue(revoked.contains("3 views, last 1 h ago. Taken down "), revoked)
    }

    func testWhatALinkShows() {
        XCTAssertEqual(SharingText.what(.album(id: 1, title: "Goa")), "the album")
        XCTAssertEqual(SharingText.what(.items(uids: ["a"])), "this item")
        XCTAssertEqual(SharingText.what(.items(uids: ["a", "b", "c"])), "these 3 items")
    }

    func testShareRefusals() {
        XCTAssertEqual(SharingText.error(APIError(status: 422, detail: "vault_item"), fallback: "x"),
                       "Private items cannot be shared. Take them out of the selection first.")
        XCTAssertEqual(SharingText.error(APIError(status: 404, detail: "share_not_found"), fallback: "x"),
                       "This link does not exist or was taken down.")
        XCTAssertTrue(SharingText.error(APIError(status: 500, detail: "boom"), fallback: "The link could not be created.")
            .hasPrefix("The link could not be created. "))
    }

    func testExpiryChoices() {
        XCTAssertEqual(SharingExpiry.allCases.map(\.label), ["Never", "After 1 day", "After 7 days", "After 30 days"])
        XCTAssertNil(SharingExpiry.never.days)
        XCTAssertEqual(SharingExpiry.sevenDays.days, 7)
    }
}
