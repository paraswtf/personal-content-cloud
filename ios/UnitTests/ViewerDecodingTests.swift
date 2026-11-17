import XCTest
@testable import PersonalContentCloud

/// The answers the viewer reads decode into the generated types: the viewer seed's items in full (the boarding pass
/// with its pages, text and document, the received photo with its tag, the video with a face at one second, the
/// file without a preview), the text, the related suggestions, the document types and the tags.
final class ViewerDecodingTests: XCTestCase {
    func testTheViewersAnswersDecode() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["PCC_TEST_SERVER"], let base = URL(string: address), let token = env["PCC_TEST_TOKEN"],
              !token.isEmpty else {
            throw XCTSkip("PCC_TEST_SERVER and PCC_TEST_TOKEN are not set (the iOS workflow sets them)")
        }
        let api = APIClient(base: base, token: token)
        let page = try await api.listItems(from: "2026-04-01", to: "2026-04-30", limit: 200)
        let byName = Dictionary(page.items.map { ($0.filename, $0.uid) }, uniquingKeysWith: { a, _ in a })

        let pdfUid = try XCTUnwrap(byName["Viewer boarding pass.pdf"], "the viewer seed's PDF is in the library")
        let pdf = try await api.getItem(uid: pdfUid)
        XCTAssertEqual(pdf.urls.pages.count, 2)
        XCTAssertEqual(pdf.textStatus, .done)
        let document = try XCTUnwrap(pdf.document)
        XCTAssertEqual(document.docType, "boarding_pass")
        XCTAssertEqual(document.fields["seat"]?.value, "14C")
        XCTAssertLessThan(document.fields["seat"]?.confidence ?? 1, ViewerFormat.lowConfidence)
        let text = try await api.itemText(uid: pdfUid)
        XCTAssertTrue(text.text.contains("BOARDING PASS"))
        _ = try await api.itemRelatedSuggestions(uid: pdfUid)

        let receivedUid = try XCTUnwrap(byName["VIEWER_RECEIVED.JPG"])
        let received = try await api.getItem(uid: receivedUid)
        XCTAssertEqual(ViewerFormat.receivedLabel(received.receivedVia), "Received on WhatsApp")
        XCTAssertEqual(received.userTags.map(\.name), ["Viewer trip"])
        XCTAssertNotNil(received.gps)
        XCTAssertFalse(ViewerFormat.qualityLines(received.qualityScores).isEmpty)

        let otherUid = try XCTUnwrap(byName["VIEWER_NOTES.BIN"])
        let other = try await api.getItem(uid: otherUid)
        XCTAssertEqual(MediaKind(other.kind.rawValue), .other)

        if let clipUid = byName["VIEWER_CLIP.MP4"] {
            let clip = try await api.getItem(uid: clipUid)
            XCTAssertNotNil(clip.urls.video)
            XCTAssertEqual(ViewerFormat.videoTracks(clip).map(\.name), ["Viewer Guest"])
        }

        _ = try await api.documentTypes()
        _ = try await api.listTags(q: "Viewer")
        _ = try await api.getConfig()
    }
}
