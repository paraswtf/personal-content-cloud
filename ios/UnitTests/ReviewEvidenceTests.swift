import XCTest
@testable import PersonalContentCloud

/// The guess evidence in words, as the website's evidence.ts words it, from the JSON the server sends.
final class ReviewEvidenceTests: XCTestCase {
    private func guess(_ evidence: String, score: Double = 0.85, name: String? = "Review Person") throws -> API.GuessOut {
        let person = name.map { "\"name\": \"\($0)\"" } ?? "\"name\": null"
        let json = "{\"person\": {\"id\": 7, \(person)}, \"score\": \(score), \"evidence\": \(evidence)}"
        return try JSONDecoder().decode(API.GuessOut.self, from: Data(json.utf8))
    }

    func testClothesGuess() throws {
        let g = try guess("{\"minutes_apart\": 25, \"scene\": 0.86, \"clothes\": 0.91, \"face\": 0.42, \"gps_m\": 12.4}")
        XCTAssertEqual(ReviewEvidence.basis(g.evidence), .clothes)
        XCTAssertEqual(ReviewEvidence.label(isFace: true, guess: g), "Face")
        XCTAssertEqual(ReviewEvidence.label(isFace: false, guess: g), "By clothes")
        XCTAssertEqual(ReviewEvidence.sure(g), "85% sure")
        XCTAssertEqual(ReviewEvidence.clothesLine(g), "25 min apart, same scene 0.86, 12 m apart, same clothes 0.91, face 0.42")
        XCTAssertEqual(ReviewEvidence.icons(isFace: false, guess: g), [.clothes])
        XCTAssertEqual(ReviewEvidence.icons(isFace: true, guess: g), [])
    }

    func testOlderClothesGuessAndLowFace() throws {
        let g = try guess("{\"same_day\": true, \"face\": 0.1, \"p\": 0.92}")
        XCTAssertEqual(ReviewEvidence.clothesLine(g), "same day, trained face 92%")
        let h = try guess("{\"minutes_apart\": 0.4}")
        XCTAssertEqual(ReviewEvidence.clothesLine(h), "same minute")
    }

    func testTrainedFaceGuess() throws {
        let g = try guess("{\"basis\": \"face\", \"face\": 0.61, \"also\": \"clothes\"}", score: 0.92)
        XCTAssertEqual(ReviewEvidence.label(isFace: true, guess: g), "By face")
        XCTAssertEqual(ReviewEvidence.sure(g), "92% sure")
        XCTAssertEqual(ReviewEvidence.clothesLine(g), "Looks like Review Person's photos, 0.61, and the same clothes")
        XCTAssertEqual(ReviewEvidence.icons(isFace: true, guess: g), [.face, .clothes])
    }

    func testGroupedGuess() throws {
        let g = try guess("{\"basis\": \"group\", \"face\": 0.62}", score: 0.62)
        XCTAssertEqual(ReviewEvidence.label(isFace: true, guess: g), "Grouped")
        XCTAssertEqual(ReviewEvidence.sure(g), "face 0.62")
        XCTAssertEqual(ReviewEvidence.clothesLine(g), "grouped with the face you named Review Person")
        XCTAssertEqual(ReviewEvidence.icons(isFace: true, guess: g), [.face])
    }

    func testTattooGuess() throws {
        let g = try guess("""
        {"basis": "trait", "traits": [{"kind": "tattoo", "part": "forearm_l", "days_apart": 20, "support": 1,
          "ref_item_uid": "abc", "crop_url": "/files/trait/3", "ref_crop_url": "//elsewhere/x.png"}],
         "against": [{"kind": "nails", "part": "hand_r", "why": "bare"}, {"kind": "tattoo", "part": "calf"}]}
        """, score: 0.95)
        XCTAssertEqual(ReviewEvidence.label(isFace: true, guess: g), "By tattoo")
        XCTAssertEqual(ReviewEvidence.sure(g), "strong match")
        XCTAssertEqual(ReviewEvidence.clothesLine(g), "")
        let reasons = ReviewEvidence.traitReasons(g)
        XCTAssertEqual(reasons.count, 1)
        XCTAssertEqual(reasons[0].text, "Same tattoo on the forearm as Review Person's photo and 1 other photo")
        XCTAssertEqual(reasons[0].cropUrl, "/files/trait/3")
        XCTAssertNil(reasons[0].refCropUrl, "an address on another host is not a server path")
        XCTAssertEqual(reasons[0].refItemUid, "abc")
        XCTAssertEqual(ReviewEvidence.againstReasons(g), [
            "No nail polish on the hand here, where Review Person has one",
            "No matching tattoo on the calf",
        ])
        XCTAssertEqual(ReviewEvidence.icons(isFace: true, guess: g), [.tattoo])
    }

    func testTattooTakenTheSameDayAndMirrored() throws {
        let g = try guess("{\"basis\": \"trait\", \"traits\": [{\"kind\": \"tattoo\", \"part\": \"upper_arm_r\", \"days_apart\": 0.2, \"mirrored\": true}]}")
        XCTAssertEqual(ReviewEvidence.traitReasons(g).first?.text,
                       "Same tattoo on the upper arm as in Review Person's photo taken the same day, probably a mirror photo")
    }

    func testNailsAndMehndiWithClothes() throws {
        let g = try guess("""
        {"basis": "clothes+trait", "minutes_apart": 5, "traits": [
          {"kind": "nails", "part": "hand_l", "color": "Blue", "finish": "art", "length": "long", "days_apart": 2,
           "ref_before": true, "hex": "#3366ff"},
          {"kind": "mehndi", "part": "hand_r", "days_apart": 1, "ref_before": false}]}
        """)
        XCTAssertEqual(ReviewEvidence.label(isFace: false, guess: g), "By clothes, mehndi and nails")
        let reasons = ReviewEvidence.traitReasons(g)
        XCTAssertEqual(reasons.map(\.text), ["Same blue nail art, long, 2 days earlier", "Same mehndi on the hand, 1 day later"])
        XCTAssertEqual(reasons[0].hex, "#3366ff")
        XCTAssertEqual(ReviewEvidence.icons(isFace: false, guess: g), [.clothes, .hand])
        XCTAssertEqual(ReviewEvidence.clothesLine(g), "5 min apart")
    }

    func testPositionGuess() throws {
        let g = try guess("{\"basis\": \"place\", \"place\": {\"seconds_apart\": 8, \"ref_before\": true, \"others\": 1, \"ref_crop_url\": \"/files/face/2\", \"ref_item_uid\": \"u2\"}}", score: 0.77)
        XCTAssertEqual(ReviewEvidence.label(isFace: true, guess: g), "By position")
        XCTAssertEqual(ReviewEvidence.sure(g), "77% sure")
        let reason = try XCTUnwrap(ReviewEvidence.placeReason(g))
        XCTAssertEqual(reason.text, "In the same spot as Review Person 8 s earlier, and 1 other person stayed in place")
        XCTAssertEqual(reason.when, "8 s earlier")
        XCTAssertEqual(reason.refCropUrl, "/files/face/2")
        XCTAssertEqual(reason.refItemUid, "u2")
        XCTAssertEqual(ReviewEvidence.icons(isFace: true, guess: g), [.place])

        let followed = try guess("{\"basis\": \"clothes+place\", \"place\": {\"seconds_apart\": 150, \"hops\": 3, \"others\": 2}}")
        XCTAssertEqual(ReviewEvidence.placeReason(followed)?.text,
                       "Followed across 3 photos to Review Person, 3 min apart, and 2 other people stayed in place")
        XCTAssertEqual(ReviewEvidence.label(isFace: false, guess: followed), "By clothes and position")
    }

    func testTrainedFaceWithPositionInOneLine() throws {
        let g = try guess("{\"basis\": \"face\", \"face\": 0.7, \"also\": \"clothes+place\", \"place\": {\"seconds_apart\": 30, \"ref_before\": false}}")
        XCTAssertEqual(ReviewEvidence.clothesLine(g), "Looks like Review Person's photos, 0.70, and the same clothes")
        XCTAssertEqual(ReviewEvidence.evidenceLine(g), "Looks like Review Person's photos, 0.70, and the same clothes and spot 30 s later")
        XCTAssertEqual(ReviewEvidence.icons(isFace: true, guess: g), [.face, .clothes, .place])
    }

    func testPhrases() {
        XCTAssertEqual(ReviewEvidence.gapPhrase(0.4, before: true), "in the same second")
        XCTAssertEqual(ReviewEvidence.gapPhrase(45, before: nil), "45 s apart")
        XCTAssertEqual(ReviewEvidence.gapPhrase(120, before: false), "2 min later")
        XCTAssertEqual(ReviewEvidence.gapPhrase(nil, before: true), "")
        XCTAssertEqual(ReviewEvidence.daysPhrase(0.3, before: true), "the same day")
        XCTAssertEqual(ReviewEvidence.daysPhrase(1, before: true), "1 day earlier")
        XCTAssertEqual(ReviewEvidence.daysPhrase(-3, before: nil), "3 days apart")
        XCTAssertEqual(ReviewEvidence.partWords("forearm_l"), "forearm")
        XCTAssertEqual(ReviewEvidence.partWords("lower_back"), "lower back")
        XCTAssertEqual(ReviewEvidence.joinWords(["clothes", "tattoo", "position"]), "clothes, tattoo and position")
        XCTAssertEqual(ReviewEvidence.safeHex(.string("#a1B2c3")), "#a1B2c3")
        XCTAssertNil(ReviewEvidence.safeHex(.string("red")))
        XCTAssertNil(ReviewEvidence.safeHex(.string("#a1b2c3;x")))
        XCTAssertNil(ReviewEvidence.path(.string("//other.host/a.png")))
        XCTAssertNil(ReviewEvidence.path(.string("https://other.host/a.png")))
        XCTAssertEqual(ReviewEvidence.path(.string("/files/face/2")), "/files/face/2")
    }

    func testMalformedEvidenceIsLeftOut() throws {
        let g = try guess("{\"traits\": [{\"kind\": \"scar\", \"part\": \"arm\"}, {\"kind\": \"tattoo\"}, 3], \"place\": [1, 2]}")
        XCTAssertTrue(ReviewEvidence.traitReasons(g).isEmpty)
        XCTAssertNil(ReviewEvidence.placeReason(g))
        XCTAssertEqual(ReviewEvidence.label(isFace: false, guess: g), "By clothes")
    }

    /// A number no measurement could be is left out instead of stopping the app when it is made a whole number.
    func testAbsurdNumbersAreLeftOut() throws {
        XCTAssertNil(ReviewEvidence.num(.number(1e300)))
        XCTAssertNil(ReviewEvidence.int(.number(-1e20)))
        XCTAssertEqual(ReviewEvidence.int(.number(41.6)), 42)
        let g = try guess("{\"minutes_apart\": 1e300, \"scene\": 0.86, \"clothes\": 0.91, \"gps_m\": 1e299}")
        XCTAssertEqual(ReviewEvidence.clothesLine(g), "same scene 0.86, same clothes 0.91")
    }

    func testItemReasons() {
        XCTAssertEqual(ReviewItemReasons.forward(["meme_p": .number(0.81), "text": .bool(true), "small": .bool(true),
                                                  "stripped": .bool(true), "jpeg_quality": .number(75),
                                                  "name": .string("whatsapp")]),
                       ["Meme 0.81", "Text", "Small, no camera data", "JPEG quality 75", "Filename from WhatsApp"])
        XCTAssertEqual(ReviewItemReasons.forward(["stripped": .number(1), "video_small": .bool(true), "name": .string("app")]),
                       ["No camera data", "Small video", "Filename from a messaging app"])
        XCTAssertEqual(ReviewItemReasons.forward(nil), [])
        XCTAssertEqual(ReviewItemReasons.misclick(["dark": .number(0.1), "blurry": .number(0.3), "burst_best": .string("x"),
                                                   "odd_light": .bool(true)]),
                       ["Blurry", "Too dark", "Odd light"])
        XCTAssertEqual(ReviewItemReasons.misclick(["reasons": .array([.string("pocket"), .string("burst")])]),
                       ["Pocket shot", "Burst copy"])
    }

    func testFrameAddress() throws {
        let json = """
        {"uid": "abc", "kind": "video", "version": 3, "filename": "a.mp4", "created_at": "2026-01-01T00:00:00Z",
         "category": "misc", "nsfw": true, "urls": {"thumb": "/files/thumb/abc", "preview": "/p", "preview_small": "/ps"}}
        """
        let item = try JSONDecoder().decode(API.ItemCard.self, from: Data(json.utf8))
        XCTAssertEqual(ReviewBox.frameRef(item, t: 12.3456), "/files/frame/abc?t=12.345&size=preview&v=3n")
        XCTAssertEqual(ReviewBox.frameRef(item, t: -1), "/files/frame/abc?t=0.000&size=preview&v=3n")
    }
}
