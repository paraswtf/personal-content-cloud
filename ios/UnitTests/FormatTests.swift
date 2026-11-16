import XCTest
@testable import PersonalContentCloud

/// The app's formatting matches the website's lib/format.ts, and the server's timestamps are read in every shape.
final class FormatTests: XCTestCase {
    func testTimestampsInTheServersShapes() throws {
        let utc = try XCTUnwrap(ISODate.parse("2026-11-16T19:18:17.000Z"))
        XCTAssertEqual(utc.timeIntervalSince1970, 1_794_856_697, accuracy: 0.001)
        let micro = try XCTUnwrap(ISODate.parse("2026-11-16T19:18:17.123456+00:00"))
        XCTAssertEqual(micro.timeIntervalSince(utc), 0.123, accuracy: 0.001)
        let offset = try XCTUnwrap(ISODate.parse("2026-11-17T00:48:17+05:30"))
        XCTAssertEqual(offset.timeIntervalSince(utc), 0, accuracy: 0.001)
        // Without a zone: local time, as the website's Date reads it.
        let local = try XCTUnwrap(ISODate.parse("2026-11-16T19:18:17"))
        let parts = Calendar.current.dateComponents([.hour, .minute], from: local)
        XCTAssertEqual(parts.hour, 19)
        XCTAssertEqual(parts.minute, 18)
        XCTAssertNotNil(ISODate.parse("2026-11-16"))
        XCTAssertNil(ISODate.parse(""))
        XCTAssertNil(ISODate.parse("not a date"))
    }

    func testDurationsCountsAndSizes() {
        XCTAssertEqual(Fmt.duration(75), "1:15")
        XCTAssertEqual(Fmt.duration(3725), "1:02:05")
        XCTAssertEqual(Fmt.duration(nil), "")
        XCTAssertEqual(Fmt.bytes(Int64(512)), "512 B")
        XCTAssertEqual(Fmt.bytes(Int64(1536)), "1.5 KB")
        XCTAssertEqual(Fmt.bytes(Int64(250 * 1024 * 1024)), "250 MB")
        XCTAssertEqual(Fmt.minutes(65), "1 h 5 min")
        XCTAssertEqual(Fmt.minutes(3), "3 min")
        XCTAssertEqual(Fmt.monthHeading("2026-11").isEmpty, false)
        XCTAssertEqual(Fmt.initials("Test Person"), "TP")
        XCTAssertEqual(Fmt.initials(nil), "?")
        XCTAssertEqual(Fmt.speciesLabel("guinea_pig"), "Guinea pig")
    }

    func testNamingMessagesHaveTheWebsitesTails() {
        XCTAssertEqual(Fmt.guessedTail(0), "")
        XCTAssertEqual(Fmt.guessedTail(nil), "")
        XCTAssertEqual(Fmt.guessedTail(1), " 1 face that looks alike waits in Review.")
        XCTAssertEqual(Fmt.guessedTail(2), " 2 faces that look alike wait in Review.")
        XCTAssertEqual(Fmt.followedTail(3), " 3 look-alikes came along.")
    }

    func testOnlyRealNamesCountAsNamed() {
        XCTAssertTrue(Fmt.isNamed("Ann"))
        XCTAssertFalse(Fmt.isNamed(nil))
        XCTAssertFalse(Fmt.isNamed("   "))
        XCTAssertEqual(Fmt.personName(nil, kind: "pet", species: "dog"), "Unnamed dog")
        XCTAssertNil(PersonOption(id: 1, name: " ", coverRef: nil, kind: "person", species: nil))
        XCTAssertNotNil(PersonOption(id: 2, name: nil, coverRef: nil, kind: "pet", species: "cat"))
    }

    func testErrorsInPlainWords() {
        XCTAssertEqual(ErrorText.sentence("it worked"), "It worked.")
        XCTAssertEqual(ErrorText.sentence("done. "), "Done.")
        XCTAssertEqual(ErrorText.describe(APIError(status: 401, detail: "invalid_credentials")), "The username or password is wrong.")
        XCTAssertEqual(ErrorText.describe(APIError(status: 403, detail: "vault_locked")), "Unlock the vault to see this.")
        XCTAssertEqual(UndoCenter.describe(method: "DELETE", path: "/api/items/abc"), "Moved an item to the trash")
        XCTAssertEqual(UndoCenter.describe(method: "POST", path: "/api/faces/assign"), "Changed who is in a photo")
    }

    func testMediaAddressesStayOnTheServer() throws {
        let connection = APIConnection(base: try XCTUnwrap(URL(string: "https://cloud.example")), token: "t")
        XCTAssertEqual(connection.mediaURL("/files/thumb/a1?v=2")?.absoluteString, "https://cloud.example/files/thumb/a1?v=2")
        XCTAssertEqual(connection.mediaURL("/files/thumb/a1?v=2", vault: true)?.absoluteString,
                       "https://cloud.example/files/thumb/a1?v=2&vault=1")
        XCTAssertEqual(connection.mediaURL("https://cloud.example/files/face/3")?.absoluteString, "https://cloud.example/files/face/3")
        XCTAssertNil(connection.mediaURL("https://elsewhere.example/files/face/3"), "the token never goes to another host")
        let request = connection.request(try XCTUnwrap(URL(string: "https://elsewhere.example/x")))
        XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
    }
}
