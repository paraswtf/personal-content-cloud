import XCTest
@testable import PersonalContentCloud

/// The people screens' answers from the seeded server (ci/seeds/people.py) decode into the generated types: a person,
/// a pet, their faces, crops, items, guesses, traits, recognition, related suggestions, merge suggestions and the
/// people on a selection. Skipped without the workflow's server and token, like APIDecodingTests.
final class PeopleDecodingTests: XCTestCase {
    private var failures: [String] = []

    private func check<T>(_ name: String, _ call: () async throws -> T) async -> T? {
        do {
            return try await call()
        } catch {
            failures.append("\(name): \(error)")
            return nil
        }
    }

    func testThePeopleAnswersDecode() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["PCC_TEST_SERVER"], let base = URL(string: address), let token = env["PCC_TEST_TOKEN"],
              !token.isEmpty else {
            throw XCTSkip("PCC_TEST_SERVER and PCC_TEST_TOKEN are not set (the iOS workflow sets them)")
        }
        let api = APIClient(base: base, token: token)

        let people = await check("listPeople") { try await api.listPeople(q: "People Alpha", kind: .person) }
        let alpha = people?.items.first { $0.name == "People Alpha" }
        XCTAssertNotNil(alpha, "the people seed's People Alpha is not listed")
        if let alpha {
            let person = await check("getPerson") { try await api.getPerson(personId: alpha.id) }
            XCTAssertEqual(person?.traitCount, 1, "People Alpha has one tattoo")
            let faces = await check("personFaces") { try await api.personFaces(personId: alpha.id) }
            XCTAssertFalse(faces?.items.isEmpty ?? true, "People Alpha has faces")
            _ = await check("personItems") { try await api.personItems(personId: alpha.id) }
            _ = await check("personItems related") { try await api.personItems(personId: alpha.id, relation: .related) }
            let guesses = await check("reviewGuesses") { try await api.reviewGuesses(person: alpha.id) }
            XCTAssertEqual(guesses?.items.filter { $0.guess.person.id == alpha.id }.count, 2,
                           "the server made two guesses for People Alpha")
            let traits = await check("personTraits") { try await api.personTraits(personId: alpha.id) }
            XCTAssertEqual(traits?.tattoos.count, 1)
            _ = await check("personRecognition") { try await api.personRecognition(personId: alpha.id) }
            _ = await check("recognitionConflicts") { try await api.recognitionConflicts(personId: alpha.id) }
            _ = await check("recognitionElsewhere") { try await api.recognitionElsewhere(personId: alpha.id) }
            _ = await check("relatedSuggestions") { try await api.relatedSuggestions(personId: alpha.id) }
            if let uid = faces?.items.first?.itemUid {
                let onItems = await check("peopleOnItems") { try await api.peopleOnItems(body: API.ItemsPeopleIn(uids: [uid])) }
                XCTAssertTrue(onItems?.people.contains { $0.id == alpha.id } ?? false, "People Alpha is on their own photo")
            }
        }

        let pets = await check("listPeople pets") { try await api.listPeople(kind: .pet) }
        if let pet = pets?.items.first(where: { $0.name == "People Pet" }) {
            let crops = await check("personAnimals") { try await api.personAnimals(personId: pet.id) }
            XCTAssertEqual(crops?.items.count, 2, "People Pet has two crops")
        } else {
            XCTFail("the people seed's pet is not listed")
        }

        let suggestions = await check("suggestions") { try await api.suggestions(limit: 12) }
        XCTAssertTrue(suggestions?.items.contains { [$0.a.name, $0.b.name].contains("People Beta") } ?? false,
                      "People Beta and People Gamma are not suggested as the same person")
        _ = await check("unassignedFaces") { try await api.unassignedFaces(limit: 120) }

        XCTAssertTrue(failures.isEmpty, failures.joined(separator: "\n"))
    }
}
