import XCTest
@testable import PersonalContentCloud

/// The person picker's rows, as the website builds them (components/people/PersonPicker.tsx): the most alike named
/// people first, typing that keeps the order of likeness, no unnamed people, and "Create" for a new name.
final class PickerOptionsTests: XCTestCase {
    private func people(_ json: String) throws -> [API.PersonOut] {
        try JSONDecoder().decode(API.PersonPage.self, from: Data(#"{"items": \#(json), "total": 0}"#.utf8)).items
    }

    private func similar(_ json: String) throws -> API.SimilarOut {
        try JSONDecoder().decode(API.SimilarOut.self, from: Data(json.utf8))
    }

    /// Five people as `GET /api/people` lists them by item count: one has no name (it must never show).
    private func listed() throws -> [API.PersonOut] {
        try people("""
        [{"id": 1, "name": "Ann", "item_count": 50, "face_count": 80, "created_at": "2026-01-01T00:00:00Z", "kind": "person"},
         {"id": 2, "name": "Ben", "item_count": 40, "face_count": 60, "created_at": "2026-01-01T00:00:00Z", "kind": "person"},
         {"id": 3, "name": null, "item_count": 30, "face_count": 30, "created_at": "2026-01-01T00:00:00Z", "kind": "person"},
         {"id": 4, "name": "Cara", "item_count": 20, "face_count": 25, "created_at": "2026-01-01T00:00:00Z", "kind": "person"},
         {"id": 5, "name": "Dan", "item_count": 10, "face_count": 12, "created_at": "2026-01-01T00:00:00Z", "kind": "person"}]
        """)
    }

    /// The likeness answer: Cara first (likely), then Dan; scores for everyone it looked at.
    private func ranking() throws -> API.SimilarOut {
        try similar("""
        {"items": [
          {"person": {"id": 4, "name": "Cara", "kind": "person"}, "score": 0.81, "likely": true, "basis": "face", "reason": "Looks like them", "first": true},
          {"person": {"id": 5, "name": "Dan", "kind": "person"}, "score": 0.55, "likely": false, "basis": "face", "reason": "Looks like them"}
         ],
         "scores": {"4": 0.81, "5": 0.55, "2": 0.3, "1": 0.1}}
        """)
    }

    private func ids(_ rows: [PickerRow]) -> [String] { rows.map(\.id) }

    func testWithNothingTypedTheMostAlikeComeFirstThenEveryoneElse() throws {
        let rows = PickerOptions.build(listed: try listed(), ranking: try ranking(), typed: "", kind: .person,
                                       excluding: [], species: nil, speciesOnly: false, allowCreate: true)
        XCTAssertEqual(ids(rows), ["p4", "p5", "p1", "p2"])
        XCTAssertEqual(rows.map(\.section), [.alike, .alike, .rest, .rest])
        guard case .person(let cara, _) = rows[0] else { return XCTFail("the first row is not a person") }
        XCTAssertTrue(cara.likely)
        XCTAssertEqual(cara.detail, "Looks like them, 20 items")
        guard case .person(let ann, _) = rows[2] else { return XCTFail("the third row is not a person") }
        XCTAssertFalse(ann.likely)
        XCTAssertEqual(ann.detail, "80 faces, 50 items")
    }

    func testTypingFiltersByNameAndKeepsTheOrderOfLikeness() throws {
        // The server found Ann, Ben and Dan for what was typed (in name order); likeness puts Dan first, then Ben.
        let found = try listed().filter { [1, 2, 5].contains($0.id) }
        let rows = PickerOptions.build(listed: found, ranking: try ranking(), typed: "n", kind: .person,
                                       excluding: [], species: nil, speciesOnly: false, allowCreate: true)
        XCTAssertEqual(ids(rows), ["p5", "p2", "p1", "c-person-n"])
    }

    func testWithoutALikenessAnswerThePlainListShows() throws {
        let rows = PickerOptions.build(listed: try listed(), ranking: nil, typed: "", kind: .person,
                                       excluding: [2], species: nil, speciesOnly: false, allowCreate: true)
        XCTAssertEqual(ids(rows), ["p1", "p4", "p5"], "no unnamed person, no excluded one, no create row without text")
    }

    func testCreateIsOfferedOnlyForANameNobodyHas() throws {
        let list = try listed()
        let exact = PickerOptions.build(listed: list, ranking: nil, typed: " ann ", kind: .person,
                                        excluding: [], species: nil, speciesOnly: false, allowCreate: true)
        XCTAssertFalse(ids(exact).contains { $0.hasPrefix("c-") })
        let fresh = PickerOptions.build(listed: list, ranking: nil, typed: "Eve", kind: .person,
                                        excluding: [], species: nil, speciesOnly: false, allowCreate: true)
        XCTAssertEqual(ids(fresh).last, "c-person-Eve")
        let off = PickerOptions.build(listed: list, ranking: nil, typed: "Eve", kind: .person,
                                      excluding: [], species: nil, speciesOnly: false, allowCreate: false)
        XCTAssertFalse(ids(off).contains { $0.hasPrefix("c-") })
    }

    func testBothKindsListPeopleThenPetsAndANewNameCanBeEither() throws {
        let list = try people("""
        [{"id": 7, "name": null, "item_count": 9, "created_at": "2026-01-01T00:00:00Z", "kind": "pet", "species": "dog"},
         {"id": 1, "name": "Ann", "item_count": 50, "created_at": "2026-01-01T00:00:00Z", "kind": "person"},
         {"id": 8, "name": "Rex", "item_count": 5, "created_at": "2026-01-01T00:00:00Z", "kind": "pet", "species": "dog"}]
        """)
        let rows = PickerOptions.build(listed: list, ranking: nil, typed: "Zed", kind: .all,
                                       excluding: [], species: nil, speciesOnly: false, allowCreate: true)
        XCTAssertEqual(ids(rows), ["p1", "p7", "p8", "c-person-Zed", "c-pet-Zed"])
        XCTAssertEqual(rows.map(\.section), [.rest, .pets, .pets, .create, .create])
        guard case .person(let unnamedDog, _) = rows[1] else { return XCTFail("not a pet row") }
        XCTAssertEqual(unnamedDog.name, "Unnamed dog", "a pet without a name is still listed")
    }

    func testPetsOfTheBoxSpeciesComeFirstOrAlone() throws {
        let list = try people("""
        [{"id": 1, "name": "Tom", "item_count": 9, "created_at": "2026-01-01T00:00:00Z", "kind": "pet", "species": "cat"},
         {"id": 2, "name": "Rex", "item_count": 5, "created_at": "2026-01-01T00:00:00Z", "kind": "pet", "species": "dog"}]
        """)
        let first = PickerOptions.build(listed: list, ranking: nil, typed: "", kind: .pet,
                                        excluding: [], species: "dog", speciesOnly: false, allowCreate: true)
        XCTAssertEqual(ids(first), ["p2", "p1"])
        let only = PickerOptions.build(listed: list, ranking: nil, typed: "", kind: .pet,
                                       excluding: [], species: "dog", speciesOnly: true, allowCreate: true)
        XCTAssertEqual(ids(only), ["p2"])
    }

    func testTheLikenessRequestIsTheSameForTheSameSource() {
        let a = LikenessSource(faceIds: [3, 1, 3], uids: ["b", "a"]).request(limit: 6)
        XCTAssertEqual(a.faceIds, [1, 3])
        XCTAssertNil(a.uids, "items count only with a person")
        XCTAssertEqual(a.limit, 6)
        let b = LikenessSource.person(9, on: ["b", "a", "b"]).request(limit: 6)
        XCTAssertEqual(b.uids, ["a", "b"])
        XCTAssertEqual(b.personId, 9)
        XCTAssertTrue(LikenessSource().isEmpty)
        XCTAssertFalse(LikenessSource.item("u1").isEmpty)
        XCTAssertEqual(LikenessSource.box(StageBox(kind: .body, boxId: 4, x: 0, y: 0, w: 1, h: 1)), .bodies([4]))
    }
}
