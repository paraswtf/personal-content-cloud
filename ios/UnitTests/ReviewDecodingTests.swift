import XCTest
@testable import PersonalContentCloud

/// The review bins of the seeded server (ios/ci/seeds/review.py) decode into the generated types, every guess names
/// a named person (a guess for an unnamed group never reaches the app), and the evidence of this area's guesses
/// reads as the cards show it. The screen tests may answer some of this area's guesses before or after this test
/// runs, so it checks the kinds that are left, not their number.
final class ReviewDecodingTests: XCTestCase {
    func testTheReviewBinsDecode() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["PCC_TEST_SERVER"], let base = URL(string: address), let token = env["PCC_TEST_TOKEN"],
              !token.isEmpty else {
            throw XCTSkip("PCC_TEST_SERVER and PCC_TEST_TOKEN are not set (the iOS workflow sets them)")
        }
        let api = APIClient(base: base, token: token)

        var guesses: [API.GuessItem] = []
        var cursor: String?
        repeat {
            let page = try await api.reviewGuesses(cursor: cursor, limit: 200)
            guesses += page.items
            cursor = page.nextCursor
        } while cursor != nil
        for g in guesses {
            XCTAssertTrue(Fmt.isNamed(g.guess.person.name), "guess \(g.kind.rawValue) \(g.id) names an unnamed person")
        }

        let mine = guesses.filter { $0.guess.person.name == "Review Person" }
        for g in mine {
            let isFace = g.kind == .face
            XCTAssertFalse(ReviewEvidence.label(isFace: isFace, guess: g.guess).isEmpty)
            XCTAssertFalse(ReviewEvidence.sure(g.guess).isEmpty)
            switch ReviewEvidence.basis(g.guess.evidence) {
            case .place:
                let reason = try XCTUnwrap(ReviewEvidence.placeReason(g.guess))
                XCTAssertEqual(reason.text, "In the same spot as Review Person 8 s earlier, and 1 other person stayed in place")
                XCTAssertNotNil(reason.refCropUrl, "the server adds the crop of the box the person was followed from")
            case .group:
                XCTAssertEqual(ReviewEvidence.clothesLine(g.guess), "grouped with the face you named Review Person")
                XCTAssertEqual(ReviewEvidence.sure(g.guess), "face 0.62")
            case .trait:
                XCTAssertEqual(ReviewEvidence.traitReasons(g.guess).first?.kind, .tattoo)
                XCTAssertEqual(ReviewEvidence.againstReasons(g.guess).count, 1)
            default:
                break
            }
        }

        // the filters the screen offers answer too
        for basis in [API.Basis.clothes, .trait, .face, .place, .group] {
            _ = try await api.reviewGuesses(limit: 5, basis: basis)
        }
        if let person = mine.first?.guess.person.id {
            let page = try await api.reviewGuesses(limit: 50, person: person)
            XCTAssertTrue(page.items.allSatisfy { $0.guess.person.id == person })
        }

        let misclicks = try await api.reviewMisclicks(limit: 200)
        let reasons = misclicks.items.filter { $0.filename.hasPrefix("REVIEW_MISCLICK") }
            .flatMap { ReviewItemReasons.misclick($0.misclickReasons) }
        XCTAssertFalse(reasons.isEmpty, "the seeded mis-clicks carry their reasons")

        let forwards = try await api.reviewForwards(limit: 200)
        let forwardReasons = forwards.items.filter { $0.filename.hasPrefix("REVIEW_FORWARD") }
            .flatMap { ReviewItemReasons.forward($0.forwardReasons) }
        XCTAssertTrue(forwardReasons.contains { $0.hasPrefix("Filename from ") }, "the seeded forwards carry their reasons")

        let duplicates = try await api.duplicates(limit: 100)
        XCTAssertTrue(duplicates.groups.contains { g in ([g.best] + g.copies).contains { $0.filename.hasPrefix("REVIEW_DUP") } },
                      "the seeded duplicates form a group")

        let stats = try await api.stats()
        XCTAssertGreaterThanOrEqual(stats.guesses, mine.count)
    }
}
