import XCTest
import CoreLocation
@testable import PersonalContentCloud

/// The map's and the documents' rules, and the seeded server's map and documents answers
/// (ci/seeds/mapdocs.py) read into the generated types.
final class MapDocsTests: XCTestCase {
    // MARK: map rules

    func testClusterTextAndSize() {
        XCTAssertEqual(MapRules.clusterText(7), "7")
        XCTAssertEqual(MapRules.clusterText(999), "999")
        XCTAssertEqual(MapRules.clusterText(1234), "1.2k")
        XCTAssertEqual(MapRules.clusterText(2000), "2k")
        XCTAssertEqual(MapRules.clusterSize(5), 30)
        XCTAssertEqual(MapRules.clusterSize(10), 36)
        XCTAssertEqual(MapRules.clusterSize(100), 44)
        XCTAssertEqual(MapRules.clusterSize(1000), 52)
    }

    func testSmallClustersOpenThePanelAndBigOnesZoom() {
        let spread = (0..<20).map { CLLocationCoordinate2D(latitude: 15 + Double($0) * 0.01, longitude: 73.8) }
        XCTAssertTrue(MapRules.zoomsIn(count: 20, coordinates: spread))
        XCTAssertFalse(MapRules.zoomsIn(count: 12, coordinates: Array(spread.prefix(12))), "12 or fewer open the panel")
        let stacked = (0..<20).map { _ in CLLocationCoordinate2D(latitude: 15.5, longitude: 73.8) }
        XCTAssertFalse(MapRules.zoomsIn(count: 20, coordinates: stacked), "photos on one spot cannot split by zooming")
    }

    func testPickedAreaBox() throws {
        let area = try XCTUnwrap(MapPickedArea.around([
            CLLocationCoordinate2D(latitude: 15.4989, longitude: 73.8278),
            CLLocationCoordinate2D(latitude: 15.5003, longitude: 73.8301),
        ], count: 2, pad: MapRules.clusterPad))
        XCTAssertEqual(area.count, 2)
        XCTAssertEqual(area.bboxParam, "73.8276,15.4987,73.8303,15.5005")
        XCTAssertNil(MapPickedArea.around([], count: 0, pad: 0))
    }

    func testCountLine() throws {
        XCTAssertEqual(MapRules.countLine(nil), "Loading")
        let none = try decode(API.MapPoints.self, #"{"points": [], "total": 0, "with_gps": 0}"#)
        XCTAssertEqual(MapRules.countLine(none), "No items match these filters.")
        let some = try decode(API.MapPoints.self, #"{"points": [[15.5, 73.8, "abc", 0]], "total": 41, "with_gps": 27}"#)
        XCTAssertEqual(MapRules.countLine(some), "27 items of 41 have a location.")
        XCTAssertEqual(some.points.first?.v2, "abc")
    }

    // MARK: documents

    func testMaskKeepsTheLastFour() {
        XCTAssertEqual(DocumentsMeta.mask("1234 5678 9012"), "XXXX XXXX 9012")
        XCTAssertEqual(DocumentsMeta.mask("ABCDE1234F"), "XXXXXX234F")
        XCTAssertEqual(DocumentsMeta.mask("12AB"), "12AB")
    }

    func testNumberFieldsAndLabels() {
        XCTAssertTrue(DocumentsMeta.isNumberField("number"))
        XCTAssertTrue(DocumentsMeta.isNumberField("invoice_number"))
        XCTAssertTrue(DocumentsMeta.isNumberField("pnr"))
        XCTAssertFalse(DocumentsMeta.isNumberField("name"))
        XCTAssertEqual(DocumentsMeta.typeLabel("pan"), "PAN card")
        XCTAssertEqual(DocumentsMeta.typeLabel("rent_agreement"), "Rent agreement")
        XCTAssertEqual(DocumentsMeta.typeLabel(nil), "Document")
        XCTAssertEqual(DocumentsMeta.group("pass").label, "Passes and tickets")
        XCTAssertEqual(DocumentsMeta.group("unknown").label, "Other")
    }

    func testKeyFieldsPutTheMaskedIdentifierFirst() throws {
        let doc = try decode(API.DocumentOut.self, """
        {"doc_type": "pan", "doc_group": "id", "label": "PAN card", "confidence": 0.9,
         "fields": {"name": {"value": "A Holder", "label": "Name"}, "number": {"value": "ABCDE1234F", "label": "PAN"},
                    "dob": {"value": "1990-01-01", "label": "Date of birth"}},
         "field_order": ["name", "number", "dob"], "updated_at": "2026-01-01T00:00:00.000Z"}
        """)
        let fields = DocumentsMeta.keyFields(doc)
        XCTAssertEqual(fields.map(\.key), ["number", "dob"], "the name is left out of a card, the identifier leads")
        XCTAssertEqual(fields.first?.value, "XXXXXX234F")
        XCTAssertEqual(fields.first?.label, "PAN")
        XCTAssertEqual(DocumentsMeta.title(doc), "PAN card")
    }

    func testKeyFieldsOfATypeWithItsOwnOrder() throws {
        let doc = try decode(API.DocumentOut.self, """
        {"doc_type": "parking_receipt", "doc_group": "money", "label": "Parking receipt", "confidence": 0.9,
         "fields": {"location": {"value": "Mall", "label": ""}, "amount": {"value": "Rs 40", "label": ""},
                    "receipt_number": {"value": "P-77", "label": ""}, "in_time": {"value": "2026-03-01T10:00:00.000Z", "label": ""}},
         "updated_at": "2026-01-01T00:00:00.000Z"}
        """)
        let fields = DocumentsMeta.keyFields(doc)
        XCTAssertEqual(fields.map(\.key), ["receipt_number", "amount", "in_time"])
        XCTAssertEqual(fields.first?.label, "Receipt number")
        XCTAssertEqual(fields.first?.value, "P-77", "only identity documents are masked")
    }

    // MARK: the seeded server

    func testTheSeededMapAndDocumentsAnswers() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["PCC_TEST_SERVER"], let base = URL(string: address), let token = env["PCC_TEST_TOKEN"],
              !token.isEmpty else {
            throw XCTSkip("PCC_TEST_SERVER and PCC_TEST_TOKEN are not set (the iOS workflow sets them)")
        }
        let api = APIClient(base: base, token: token)

        let points = try await api.points()
        XCTAssertGreaterThan(points.withGps, 0)
        XCTAssertEqual(points.points.count, points.withGps)
        let places = try await api.placeList()
        let panaji = try XCTUnwrap(places.first { $0.place == "Panaji, Goa, India" }, "the seeded place is listed")
        XCTAssertGreaterThanOrEqual(panaji.count, 2)
        XCTAssertEqual(panaji.placeCity, "Panaji")
        let here = try await api.items(bbox: "73.82,15.49,73.84,15.51", limit: 20)
        XCTAssertGreaterThanOrEqual(here.total, 2)
        _ = try await api.mapStatus()

        let types = try await api.documentTypes()
        XCTAssertGreaterThanOrEqual(types.first { $0.type == "pan" }?.count ?? 0, 1)
        let documents = try await api.listDocuments(limit: 40)
        let pan = try XCTUnwrap(documents.items.first { $0.document.title == "MapDocs PAN card" })
        XCTAssertEqual(pan.document.docGroup, "id")
        XCTAssertEqual(pan.document.fields["number"]?.value, "ABCDE1234F")
        let found = try await api.listDocuments(q: "electricity")
        XCTAssertTrue(found.items.contains { $0.document.title == "MapDocs electricity bill" })
        let identity = try await api.listDocuments(group: "id")
        XCTAssertFalse(identity.items.contains { $0.document.docGroup != "id" })
    }

    private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(json.utf8))
    }
}
