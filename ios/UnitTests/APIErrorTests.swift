import XCTest
@testable import PersonalContentCloud

/// Unit tests run inside the app (hosted). Decoding tests for the generated API models go next to this file.
final class APIErrorTests: XCTestCase {
    func testServerDetailAndOffsetAreRead() {
        let body = Data(#"{"detail": "offset_mismatch", "received_bytes": 42}"#.utf8)
        let error = APIError.from(status: 409, body: body)
        XCTAssertEqual(error.status, 409)
        XCTAssertEqual(error.detail, "offset_mismatch")
        XCTAssertEqual(error.receivedBytes, 42)
    }

    func testABodyOfAnotherShapeFallsBackToItsText() {
        let error = APIError.from(status: 502, body: Data("Bad gateway".utf8))
        XCTAssertEqual(error.detail, "Bad gateway")
        XCTAssertNil(error.receivedBytes)
    }
}
