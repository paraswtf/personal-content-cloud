import XCTest
@testable import PersonalContentCloud

/// Every answer of the seeded server (ios/ci/seed_server.py) decodes into the generated types. On the CI machine the
/// workflow passes the server's address and a device token (TEST_RUNNER_PCC_TEST_SERVER, TEST_RUNNER_PCC_TEST_TOKEN);
/// without them the test is skipped. An answer the server refuses (a 404 for something the seed does not have) is
/// noted, not failed: this test is about reading what the server sends.
final class APIDecodingTests: XCTestCase {
    private var failures: [String] = []
    private var notes: [String] = []

    private func check<T>(_ name: String, _ call: () async throws -> T) async -> T? {
        do {
            return try await call()
        } catch let error as APIError where error.status == 0 {
            failures.append("\(name): \(error.detail)")
        } catch let error as APIError {
            notes.append("\(name): the server answered \(error.status) \(error.detail)")
        } catch {
            failures.append("\(name): \(error)")
        }
        return nil
    }

    func testTheSeededServersAnswersDecode() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["PCC_TEST_SERVER"], let base = URL(string: address), let token = env["PCC_TEST_TOKEN"],
              !token.isEmpty else {
            throw XCTSkip("PCC_TEST_SERVER and PCC_TEST_TOKEN are not set (the iOS workflow sets them)")
        }
        let api = APIClient(base: base, token: token)

        _ = await check("getState") { try await api.getState() }
        _ = await check("getConfig") { try await api.getConfig() }
        _ = await check("stats") { try await api.stats() }
        _ = await check("system") { try await api.system() }
        _ = await check("learningStatus") { try await api.learningStatus() }
        _ = await check("throughput") { try await api.throughput() }
        _ = await check("listSessions") { try await api.listSessions() }
        _ = await check("listDevices") { try await api.listDevices() }
        _ = await check("listPasskeys") { try await api.listPasskeys() }
        _ = await check("relyingParty") { try await api.relyingParty() }
        _ = await check("mapStatus") { try await api.mapStatus() }
        _ = await check("points") { try await api.points() }
        _ = await check("placeList") { try await api.placeList() }
        _ = await check("items") { try await api.items() }
        _ = await check("duplicates") { try await api.duplicates() }
        _ = await check("unassignedFaces") { try await api.unassignedFaces() }
        _ = await check("reviewGuesses") { try await api.reviewGuesses() }
        _ = await check("reviewMisclicks") { try await api.reviewMisclicks() }
        _ = await check("reviewForwards") { try await api.reviewForwards() }
        _ = await check("suggestions") { try await api.suggestions() }
        _ = await check("listTags") { try await api.listTags() }
        _ = await check("listTrash") { try await api.listTrash() }
        _ = await check("documentTypes") { try await api.documentTypes() }
        _ = await check("searchItems") { try await api.searchItems(q: "hills") }
        _ = await check("manyUploads") { try await api.manyUploads() }

        if let page = await check("listItems", { try await api.listItems(limit: 200) }) {
            XCTAssertFalse(page.items.isEmpty, "the seeded library has items")
            for card in page.items.prefix(12) {
                _ = await check("getItem \(card.uid)") { try await api.getItem(uid: card.uid) }
            }
            if let uid = page.items.first?.uid {
                _ = await check("itemPages") { try await api.itemPages(uid: uid) }
                _ = await check("itemRelatedSuggestions") { try await api.itemRelatedSuggestions(uid: uid) }
                _ = await check("itemText") { try await api.itemText(uid: uid) }
            }
        }
        if let buckets = await check("buckets", { try await api.buckets() }), let key = buckets.buckets.first?.key {
            _ = await check("bucket \(key)") { try await api.bucket(key: key) }
        }
        if let people = await check("listPeople", { try await api.listPeople() }) {
            XCTAssertFalse(people.items.isEmpty, "the seed has named people")
            XCTAssertTrue(people.items.allSatisfy { Fmt.isNamed($0.name) || $0.kind == .pet },
                          "the people list holds no unnamed person")
            for person in people.items.prefix(3) {
                let id = person.id
                _ = await check("getPerson \(id)") { try await api.getPerson(personId: id) }
                _ = await check("personItems \(id)") { try await api.personItems(personId: id) }
                _ = await check("personFaces \(id)") { try await api.personFaces(personId: id) }
                _ = await check("personAnimals \(id)") { try await api.personAnimals(personId: id) }
                _ = await check("personTraits \(id)") { try await api.personTraits(personId: id) }
                _ = await check("personRecognition \(id)") { try await api.personRecognition(personId: id) }
                _ = await check("recognitionElsewhere \(id)") { try await api.recognitionElsewhere(personId: id) }
                _ = await check("relatedSuggestions \(id)") { try await api.relatedSuggestions(personId: id) }
                _ = await check("getModel3d \(id)") { try await api.getModel3d(personId: id) }
                _ = await check("similarPeople \(id)") {
                    try await api.similarPeople(body: LikenessSource.person(id).request(limit: 6))
                }
            }
        }
        if let albums = await check("listAlbums", { try await api.listAlbums() }) {
            for album in albums.items.prefix(2) {
                let id = album.id
                _ = await check("getAlbum \(id)") { try await api.getAlbum(albumId: id) }
            }
        }
        if let shares = await check("listShares", { try await api.listShares() }) {
            for share in shares.prefix(2) {
                let id = share.id
                _ = await check("getShare \(id)") { try await api.getShare(shareId: id) }
            }
        }
        if let documents = await check("listDocuments", { try await api.listDocuments() }) {
            for document in documents.items.prefix(2) {
                let uid = document.item.uid
                _ = await check("getDocument \(uid)") { try await api.getDocument(uid: uid) }
            }
        }

        if !notes.isEmpty { print("Answers the server refused:\n" + notes.joined(separator: "\n")) }
        XCTAssertTrue(failures.isEmpty, "answers that did not decode:\n" + failures.joined(separator: "\n"))
    }
}
