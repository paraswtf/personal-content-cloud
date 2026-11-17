import XCTest
@testable import PersonalContentCloud

/// The viewer's words and rules match the website's (components/library/misclick.ts and forwards.ts, the info
/// panel's quality rows, components/documents/docMeta.ts, components/viewer/boxes.ts, DocumentStage's strip).
final class ViewerFormatTests: XCTestCase {
    func testMisclickReasonsInTheWebsitesOrder() {
        let reasons: [String: API.JSONValue] = [
            "burst": .object(["best_uid": .string("abc123")]),
            "pocket": .object([:]),
            "blurry": .number(21),
            "odd_angle": .bool(true),
        ]
        XCTAssertEqual(ViewerFormat.reasonKeys(reasons), ["blurry", "pocket", "burst", "odd_angle"])
        XCTAssertEqual(ViewerFormat.reasonKeys(["reasons": .array([.string("dark"), .string("blurry")])]), ["dark", "blurry"])
        XCTAssertEqual(ViewerFormat.reasonLabel("blown_out"), "Blown out")
        XCTAssertEqual(ViewerFormat.reasonLabel("odd_angle"), "Odd angle")
        XCTAssertEqual(ViewerFormat.burstBestUid(reasons), "abc123")
        XCTAssertEqual(ViewerFormat.burstBestUid(["best_uid": .string("xyz")]), "xyz")
        XCTAssertNil(ViewerFormat.burstBestUid(nil))
    }

    func testForwardReasonsAndTheMessagingApp() {
        let reasons: [String: API.JSONValue] = [
            "meme_p": .number(0.8123), "text": .bool(true), "small": .bool(true), "stripped": .bool(true),
            "jpeg_quality": .number(74.6), "name": .string("whatsapp"),
        ]
        XCTAssertEqual(ViewerFormat.forwardReasons(reasons),
                       ["Meme 0.81", "Text", "Small, no camera data", "JPEG quality 75", "Filename from WhatsApp"])
        XCTAssertEqual(ViewerFormat.forwardReasons(["stripped": .bool(true), "video_small": .number(1)]),
                       ["No camera data", "Small video"])
        XCTAssertEqual(ViewerFormat.receivedLabel("whatsapp"), "Received on WhatsApp")
        XCTAssertEqual(ViewerFormat.receivedLabel("app"), "Received through a messaging app")
        XCTAssertNil(ViewerFormat.receivedLabel(nil))
    }

    func testQualityRowsAndCategories() {
        let lines = ViewerFormat.qualityLines(["blur_var": .number(182.4), "accident_p": .number(0.04),
                                               "mean_lum": .number(141), "skipped": .bool(false)])
        XCTAssertEqual(lines.map(\.label), ["Sharpness", "Brightness", "Accidental shot"])
        XCTAssertEqual(lines.map(\.value), ["182 (blurry under 40)", "141 of 255", "4%"])
        XCTAssertFalse(ViewerFormat.qualitySkipped(["skipped": .bool(false)]))
        XCTAssertTrue(ViewerFormat.qualitySkipped(["skipped": .string("too small")]))
        XCTAssertEqual(ViewerFormat.topScores(["places": 0.62, "misc": 0.21, "food": 0.04, "pets": 0.05, "objects": 0.03,
                                               "documents": 0.01]).map(\.key),
                       ["places", "misc", "pets", "food", "objects"])
        XCTAssertEqual(ViewerFormat.categoryLabel("personal"), "People")
        XCTAssertEqual(ViewerFormat.categoryLabel("objects"), "Things")
        XCTAssertFalse(ViewerFormat.categories.contains(.nsfw))
    }

    func testDocumentFieldsAndLabels() {
        XCTAssertEqual(ViewerFormat.orderedKeys(fieldOrder: [], fieldKeys: ["amount", "zone", "challan_number", "date"],
                                                docType: "traffic_challan"),
                       ["challan_number", "date", "amount", "zone"])
        XCTAssertEqual(ViewerFormat.orderedKeys(fieldOrder: ["pnr", "seat"], fieldKeys: ["seat", "flight", "pnr"],
                                                docType: "boarding_pass"),
                       ["pnr", "seat", "flight"])
        XCTAssertEqual(ViewerFormat.fieldLabel(key: "in_time", serverLabel: nil, docType: "parking_receipt"), "In")
        XCTAssertEqual(ViewerFormat.fieldLabel(key: "seat", serverLabel: "Seat", docType: "boarding_pass"), "Seat")
        XCTAssertEqual(ViewerFormat.fieldLabel(key: "gate", serverLabel: "", docType: "boarding_pass"), "gate")
        XCTAssertEqual(ViewerFormat.docTypeLabel("boarding_pass"), "Boarding pass")
        XCTAssertEqual(ViewerFormat.docTypeLabel("ration_card"), "Ration card")
        XCTAssertEqual(ViewerFormat.groupLabel("pass"), "Passes and tickets")
        XCTAssertTrue(ViewerFormat.isMonospaced("pnr"))
        XCTAssertTrue(ViewerFormat.isMonospaced("challan_number"))
        XCTAssertFalse(ViewerFormat.isMonospaced("passenger"))
        XCTAssertTrue(ViewerFormat.isUnpaid("Payment pending"))
        XCTAssertFalse(ViewerFormat.isUnpaid("Paid"))
    }

    func testFilesAndTitles() {
        XCTAssertEqual(ViewerFormat.extLabel("notes.bin"), "BIN")
        XCTAssertEqual(ViewerFormat.extLabel("archive"), "FILE")
        XCTAssertEqual(ViewerFormat.extLabel("x.toolongext"), "FILE")
        XCTAssertEqual(ViewerFormat.detailsTitle(.pdf), "Document details")
        XCTAssertEqual(ViewerFormat.trashNoun(.image), "photo")
        XCTAssertEqual(ViewerFormat.trashNoun(.pdf), "file")
        XCTAssertEqual(ViewerFormat.videoLine(codec: "h264", fps: 25, bitrate: 4_200_000, hasAudio: false),
                       "H264, 25 fps, 4.2 Mbit/s, no sound")
        XCTAssertEqual(ViewerFormat.videoLine(codec: nil, fps: 29.97, bitrate: nil, hasAudio: true), "29.97 fps")
        XCTAssertEqual(ViewerToolbar.clipName("IMG_0001.HEIC"), "IMG_0001.MOV")
        XCTAssertEqual(ViewerToolbar.pageName("Ticket.pdf", 2), "Ticket page 2.webp")
        XCTAssertTrue(ViewerFormat.isLongText(String(repeating: "a\n", count: 13)))
        XCTAssertFalse(ViewerFormat.isLongText("short"))
    }

    func testThePageStripKeepsAWindowAndTheEnds() {
        XCTAssertEqual(ViewerDocumentStage.stripPages(current: 0, count: 3), [0, 1, 2])
        XCTAssertEqual(ViewerDocumentStage.stripPages(current: 50, count: 100, span: 2), [0, 48, 49, 50, 51, 52, 99])
        XCTAssertEqual(ViewerDocumentStage.stripPages(current: 0, count: 0), [])
    }

    func testThePageStripMarksThePagesLeftOut() {
        let strip = ViewerDocumentStage.stripPages(current: 50, count: 100, span: 2)   // [0, 48, ..., 52, 99]
        XCTAssertEqual(strip.indices.filter { ViewerDocumentStage.gapBefore(strip, at: $0) }, [1, 6])
        let near = ViewerDocumentStage.stripPages(current: 2, count: 6, span: 1)       // [0, 1, 2, 3, 5]
        XCTAssertEqual(near.indices.filter { ViewerDocumentStage.gapBefore(near, at: $0) }, [4],
                       "no marker when the first page joins the window")
    }

    /// A video's tracks gathered by person, only named people named, guesses with a question mark.
    func testVideoTracksAreGatheredAndNamedOnlyWhenNamed() throws {
        let json = """
        {"uid": "v1", "kind": "video", "version": 1, "filename": "clip.mp4", "created_at": "2026-04-12T11:00:00Z",
         "category": "personal", "mime": "video/mp4", "size_bytes": 1000, "updated_at": "2026-04-12T11:00:00Z",
         "urls": {"thumb": "/t", "preview": "/p", "preview_small": "/ps", "video": "/v"},
         "faces": [
          {"id": 1, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2, "det_score": 0.9, "size_px": 90, "quality": 0.8,
           "frame_time_s": 2.0, "group": 7, "url": "/f/1", "assigned_by": "user",
           "person": {"id": 3, "name": "Viewer Guest"}},
          {"id": 2, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2, "det_score": 0.9, "size_px": 90, "quality": 0.8,
           "frame_time_s": 0.5, "group": 7, "url": "/f/2", "person": {"id": 3, "name": "Viewer Guest"}},
          {"id": 3, "x": 0.5, "y": 0.1, "w": 0.2, "h": 0.2, "det_score": 0.9, "size_px": 90, "quality": 0.8,
           "frame_time_s": 1.0, "url": "/f/3", "guess": {"person": {"id": 4, "name": "Test Visitor"}, "score": 0.7}},
          {"id": 4, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "det_score": 0.9, "size_px": 90, "quality": 0.8,
           "frame_time_s": 3.0, "url": "/f/4", "person": {"id": 9, "name": null}},
          {"id": 5, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "det_score": 0.9, "size_px": 90, "quality": 0.8,
           "url": "/f/5"}
         ]}
        """
        let item = try JSONDecoder().decode(API.ItemDetail.self, from: Data(json.utf8))
        let tracks = ViewerFormat.videoTracks(item)
        XCTAssertEqual(tracks.map(\.box.boxId), [2, 3, 1, 4], "by time, stills left out")
        XCTAssertEqual(tracks.map(\.name), ["Viewer Guest", "Test Visitor?", "Viewer Guest", "Not tagged"])
        XCTAssertTrue(tracks[1].isGuess)
        XCTAssertFalse(tracks[3].isNamed, "an unnamed person is never named")
        let groups = ViewerFormat.groupTracks(tracks)
        XCTAssertEqual(groups.map(\.key), ["group:7", "solo:face:3", "solo:face:4"])
        XCTAssertEqual(groups[0].tracks.map(\.box.boxId), [2, 1])
        XCTAssertEqual(groups[0].origin, .you, "the owner's origin when any track is the owner's")
        XCTAssertEqual(groups[1].first.origin, .guess)
    }
}
