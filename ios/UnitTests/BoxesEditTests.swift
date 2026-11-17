import XCTest
@testable import PersonalContentCloud

/// The box editing arithmetic (the website's BoxAdjuster and PhotoStage draw mode), the editors' words, and what the
/// editors read from the API's item answer: who put a person on a box, open guesses, a video's tracks.
final class BoxesEditTests: XCTestCase {
    private let accuracy: CGFloat = 1e-9

    private func assertRect(_ r: CGRect, _ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat,
                            file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(r.minX, x, accuracy: accuracy, "x", file: file, line: line)
        XCTAssertEqual(r.minY, y, accuracy: accuracy, "y", file: file, line: line)
        XCTAssertEqual(r.width, w, accuracy: accuracy, "w", file: file, line: line)
        XCTAssertEqual(r.height, h, accuracy: accuracy, "h", file: file, line: line)
    }

    // MARK: moving and resizing

    func testMovingKeepsTheBoxInsideThePicture() {
        let b = CGRect(x: 0.8, y: 0.1, width: 0.15, height: 0.2)
        let r = StageEditGeometry.moved(.move, b, dx: 0.2, dy: -0.3, minW: 0.015, minH: 0.015)
        assertRect(r, 0.85, 0, 0.15, 0.2)
    }

    func testACornerResizesTheBoxAndTheOppositeCornerStays() {
        let b = CGRect(x: 0.2, y: 0.2, width: 0.2, height: 0.2)
        let r = StageEditGeometry.moved(.bottomRight, b, dx: 0.1, dy: 0.05, minW: 0.015, minH: 0.015)
        assertRect(r, 0.2, 0.2, 0.3, 0.25)
        let l = StageEditGeometry.moved(.topLeft, b, dx: -0.1, dy: -0.05, minW: 0.015, minH: 0.015)
        assertRect(l, 0.1, 0.15, 0.3, 0.25)
    }

    func testACornerNeverPassesTheOppositeOneOrLeavesThePicture() {
        let b = CGRect(x: 0.2, y: 0.2, width: 0.2, height: 0.2)
        let r = StageEditGeometry.moved(.topRight, b, dx: -0.5, dy: -0.5, minW: 0.015, minH: 0.015)
        assertRect(r, 0.2, 0, 0.015, 0.4)
        let out = StageEditGeometry.moved(.bottomLeft, b, dx: -0.9, dy: 0.9, minW: 0.015, minH: 0.015)
        assertRect(out, 0, 0.2, 0.4, 0.8)
    }

    func testTheSmallestSideIsTheBoxsOwnWhenItWasAlreadySmaller() {
        let sides = StageEditGeometry.minSides(for: CGRect(x: 0, y: 0, width: 0.01, height: 0.3))
        XCTAssertEqual(sides.w, 0.01, accuracy: accuracy)
        XCTAssertEqual(sides.h, StageEditGeometry.minSide, accuracy: accuracy)
    }

    func testATouchTakesTheNearestCornerElseTheBoxElseNothing() {
        let b = CGRect(x: 0.2, y: 0.2, width: 0.4, height: 0.4)
        XCTAssertEqual(StageEditGeometry.handle(at: CGPoint(x: 0.21, y: 0.19), in: b, reachX: 0.05, reachY: 0.05), .topLeft)
        XCTAssertEqual(StageEditGeometry.handle(at: CGPoint(x: 0.62, y: 0.61), in: b, reachX: 0.05, reachY: 0.05), .bottomRight)
        XCTAssertEqual(StageEditGeometry.handle(at: CGPoint(x: 0.4, y: 0.4), in: b, reachX: 0.05, reachY: 0.05), .move)
        XCTAssertNil(StageEditGeometry.handle(at: CGPoint(x: 0.9, y: 0.9), in: b, reachX: 0.05, reachY: 0.05))
    }

    func testOnATinyBoxTheNearestCornerWins() {
        // Every corner of a box smaller than the reach is within reach: the closest one takes the touch.
        let b = CGRect(x: 0.5, y: 0.5, width: 0.02, height: 0.02)
        XCTAssertEqual(StageEditGeometry.handle(at: CGPoint(x: 0.521, y: 0.522), in: b, reachX: 0.05, reachY: 0.05),
                       .bottomRight)
    }

    func testGrowingKeepsTheMiddleAndTheShape() {
        let b = CGRect(x: 0.4, y: 0.4, width: 0.2, height: 0.1)
        let r = StageEditGeometry.grown(b, by: 0.02, minW: 0.015, minH: 0.015)
        XCTAssertEqual(r.midX, b.midX, accuracy: 1e-9)
        XCTAssertEqual(r.midY, b.midY, accuracy: 1e-9)
        XCTAssertEqual(r.width / r.height, b.width / b.height, accuracy: 1e-9)
        let small = StageEditGeometry.grown(b, by: -1, minW: 0.015, minH: 0.015)
        XCTAssertEqual(small.width, 0.015, accuracy: 1e-9)
    }

    func testAMoveOfLessThanTheWebsitesThresholdIsNoMove() {
        let a = CGRect(x: 0.3, y: 0.3, width: 0.2, height: 0.2)
        XCTAssertFalse(StageEditGeometry.changed(a, a.offsetBy(dx: 0.00002, dy: 0.00002)))
        XCTAssertTrue(StageEditGeometry.changed(a, a.offsetBy(dx: 0.01, dy: 0)))
    }

    // MARK: drawing

    func testADrawnBoxGoesFromCornerToCornerInsideThePicture() {
        let r = StageEditGeometry.drawn(from: CGPoint(x: 0.6, y: 0.7), to: CGPoint(x: 0.2, y: 1.3), minW: 0.01, minH: 0.01)
        XCTAssertNotNil(r)
        assertRect(r!, 0.2, 0.7, 0.4, 0.3)
    }

    func testADrawingSmallerThanTheSmallestBoxIsNoBox() {
        XCTAssertNil(StageEditGeometry.drawn(from: CGPoint(x: 0.5, y: 0.5), to: CGPoint(x: 0.52, y: 0.6), minW: 0.05, minH: 0.05))
    }

    func testWholePhotoLeavesTheWebsitesMargin() {
        assertRect(StageEditGeometry.wholePhoto, 0.03, 0.03, 0.94, 0.94)
    }

    // MARK: words

    func testANamingMessageSaysHowManyFacesNowWaitInReview() {
        XCTAssertEqual(BoxesWords.withGuessed("Tagged as Sam", 2), "Tagged as Sam. 2 faces that look alike wait in Review.")
        XCTAssertEqual(BoxesWords.withGuessed("Tagged as Sam", 1), "Tagged as Sam. 1 face that looks alike waits in Review.")
        XCTAssertEqual(BoxesWords.withGuessed("Tagged as Sam", 0), "Tagged as Sam")
        XCTAssertEqual(BoxesWords.withGuessed("Tagged as Sam", nil), "Tagged as Sam")
    }

    func testSpeciesTakeTheirArticle() {
        XCTAssertEqual(BoxesWords.aSpecies("dog"), "a dog")
        XCTAssertEqual(BoxesWords.aSpecies("elephant"), "an elephant")
        XCTAssertEqual(BoxesWords.aSpecies("guinea_pig"), "a guinea pig")
        XCTAssertEqual(BoxesWords.aSpecies(nil), "a pet")
    }

    func testTracksAndLookAlikesInTheMessages() {
        XCTAssertEqual(BoxesWords.trackTail(1), "")
        XCTAssertEqual(BoxesWords.trackTail(3), " (3 tracks)")
        XCTAssertEqual(BoxesWords.followedTail(followed: 2, pulled: 1, pet: false),
                       " 2 look-alike faces followed. 1 look-alike group joined them.")
        XCTAssertEqual(BoxesWords.followedTail(followed: 1, pulled: 0, pet: true), " 1 look-alike crop followed.")
        XCTAssertEqual(BoxesWords.displayName(nil, pet: true), "Unnamed pet")
        XCTAssertEqual(BoxesWords.displayName("  ", pet: false), "Unnamed person")
    }

    func testANewBoxsFailuresInTheWebsitesWords() {
        XCTAssertEqual(BoxesWords.addFailure(APIError(status: 422, detail: "no_face")),
                       "No face found in that box. Draw it closer around the face, or add a person box when the face does not show.")
        XCTAssertEqual(BoxesWords.addFailure(APIError(status: 422, detail: "box_too_small")), "Draw a larger box, inside the picture.")
        XCTAssertEqual(BoxesWords.addFailure(APIError(status: 409, detail: "unreadable")), "The photo or frame could not be opened.")
    }

    // MARK: what the editors read from the item

    private func item(faces: String = "[]", bodies: String = "[]", animals: String = "[]", tags: String = "[]",
                      kind: String = "image") throws -> API.ItemDetail {
        let json = """
        {"uid": "u1", "kind": "\(kind)", "version": 1, "filename": "IMG_1.JPG", "created_at": "2026-04-01T10:00:00Z",
         "category": "personal", "mime": "image/jpeg", "size_bytes": 100, "updated_at": "2026-04-01T10:00:00Z",
         "urls": {"thumb": "/t", "preview": "/p", "preview_small": "/ps"},
         "faces": \(faces), "bodies": \(bodies), "animals": \(animals), "tags": \(tags)}
        """
        return try JSONDecoder().decode(API.ItemDetail.self, from: Data(json.utf8))
    }

    private func face(_ id: Int, person: String = "null", assignedBy: String = "auto", byGuess: Bool = false,
                      guess: String = "null", source: String = "detector", t: String = "null", group: String = "null") -> String {
        """
        {"id": \(id), "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2, "det_score": 0.9, "size_px": 100, "quality": 0.8,
         "url": "/api/faces/\(id)/crop", "person": \(person), "assigned_by": "\(assignedBy)", "by_guess": \(byGuess),
         "guess": \(guess), "source": "\(source)", "frame_time_s": \(t), "group": \(group)}
        """
    }

    func testWhoPutThePersonOnABoxFollowsTheWebsite() throws {
        let sam = #"{"id": 7, "name": "Sam"}"#
        let guess = #"{"person": {"id": 7, "name": "Sam"}, "score": 0.8}"#
        let i = try item(faces: "[\(face(1, person: sam, assignedBy: "user")), \(face(2, person: sam, assignedBy: "user", byGuess: true)), \(face(3, person: sam)), \(face(4, assignedBy: "guess", guess: guess)), \(face(5)), \(face(6, person: sam, source: "user"))]")
        let origins = i.faces.map { BoxesTarget.face($0).origin }
        XCTAssertEqual(origins, [.you, .confirmed, .auto, .guess, nil, .drawn])
    }

    func testAGuessForSomeoneWithoutANameIsNoGuess() throws {
        let unnamed = #"{"person": {"id": 9, "name": null}, "score": 0.9}"#
        let named = #"{"person": {"id": 7, "name": "Sam"}, "score": 0.9}"#
        let i = try item(faces: "[\(face(1, assignedBy: "guess", guess: unnamed)), \(face(2, assignedBy: "guess", guess: named))]")
        XCTAssertNil(BoxesTarget.face(i.faces[0]).guess)
        XCTAssertEqual(BoxesTarget.face(i.faces[1]).guessName, "Sam")
        XCTAssertEqual(BoxesLookup.guessTargets(i).map(\.id), ["face:2"])
    }

    func testTheStageBoxOfATargetCarriesItsNameOrGuess() throws {
        let named = #"{"person": {"id": 7, "name": "Sam"}, "score": 0.9}"#
        let i = try item(faces: "[\(face(1, assignedBy: "guess", guess: named))]")
        let box = BoxesTarget.face(i.faces[0]).stageBox
        XCTAssertEqual(box.id, "face:1")
        XCTAssertNil(box.name)
        XCTAssertEqual(box.guessName, "Sam")
        XCTAssertEqual(BoxesTarget.face(i.faces[0]).adjustLabel, "this face box, maybe Sam")
    }

    func testAVideosTracksAreTheBoxesOfOneGroupInTimeOrder() throws {
        let sam = #"{"id": 7, "name": "Sam"}"#
        let i = try item(faces: "[\(face(1, person: sam, t: "4.0", group: "3")), \(face(2, person: sam, t: "1.0", group: "3")), \(face(3, person: sam, t: "2.0", group: "8")), \(face(4))]",
                         kind: "video")
        let target = BoxesTarget.face(i.faces[0])
        XCTAssertEqual(BoxesLookup.tracks(of: target, in: i).map(\.boxId), [2, 1])
        // A photo's box is alone.
        XCTAssertEqual(BoxesLookup.tracks(of: BoxesTarget.face(i.faces[3]), in: i).map(\.boxId), [4])
    }

    func testInTheFrameAreTheNamedPeopleWhoseFacesSpanTheTime() throws {
        let sam = #"{"id": 7, "name": "Sam"}"#
        let ann = #"{"id": 8, "name": "Ann"}"#
        let unnamed = #"{"id": 9, "name": null}"#
        func spanned(_ id: Int, _ person: String, _ spans: String) -> String {
            face(id, person: person, t: "1.0").replacingOccurrences(of: "\"group\": null", with: "\"group\": null, \"spans\": \(spans)")
        }
        let i = try item(faces: "[\(spanned(1, sam, "[[0.0, 2.0]]")), \(spanned(2, sam, "[[1.5, 3.0]]")), \(spanned(3, ann, "[[2.5, 4.0]]")), \(spanned(4, unnamed, "[[0.0, 9.0]]"))]",
                         kind: "video")
        XCTAssertEqual(BoxesLookup.inFrame(i, at: 1.8).map(\.id), [7], "each person once, an unnamed one never")
        XCTAssertEqual(BoxesLookup.inFrame(i, at: 2.7).map(\.id), [7, 8])
        XCTAssertEqual(BoxesLookup.inFrame(i, at: 5.0).map(\.id), [])
    }

    func testInTheFramesShowPicksThePersonsFaceOnScreenNow() throws {
        let sam = #"{"id": 7, "name": "Sam"}"#
        func spanned(_ id: Int, _ person: String, _ t: String, _ spans: String) -> String {
            face(id, person: person, t: t).replacingOccurrences(of: "\"group\": null", with: "\"group\": null, \"spans\": \(spans)")
        }
        let i = try item(faces: "[\(spanned(1, sam, "0.5", "[[0.0, 1.0]]")), \(spanned(2, sam, "2.5", "[[2.0, 3.0]]"))]",
                         kind: "video")
        XCTAssertEqual(BoxesLookup.faceOnScreen(of: 7, in: i, at: 2.4)?.id, 2, "the face on screen, not the first one")
        XCTAssertEqual(BoxesLookup.faceOnScreen(of: 7, in: i, at: 0.2)?.id, 1)
        XCTAssertNil(BoxesLookup.faceOnScreen(of: 7, in: i, at: 1.5), "between the spans no face is on screen")
        XCTAssertNil(BoxesLookup.faceOnScreen(of: 8, in: i, at: 2.4), "someone else's face is not theirs")
    }

    @MainActor
    func testShowingABoxOrMovingOneAsksTheViewerToShowTheBoxes() throws {
        let model = ViewerBoxesModel()
        let start = model.stage.showBoxesRequest
        model.show("face:1")
        XCTAssertEqual(model.stage.showBoxesRequest, start + 1, "Show in the photo leaves hidden boxes hidden")
        let i = try item(faces: "[\(face(3, t: "1.5"))]", kind: "video")
        model.showInVideo(BoxesTarget.face(i.faces[0]), uid: "u1")
        XCTAssertEqual(model.stage.showBoxesRequest, start + 2, "Show in the video leaves hidden boxes hidden")
        XCTAssertEqual(model.videoFocus?.time, 1.5)
        model.startAdjusting(BoxesTarget.face(i.faces[0]).stageBox, uid: "u1", frameTime: 1.5)
        XCTAssertEqual(model.stage.showBoxesRequest, start + 3, "moving a box leaves hidden boxes hidden")
        model.cancelEdit()
    }

    @MainActor
    func testAVideosEditKeepsItsFrameTimeUntilItStops() {
        let state = StageEditState()
        state.startDrawing(.person, on: "v1", frameTime: 1.5)
        XCTAssertEqual(state.frameTime, 1.5)
        state.stop()
        XCTAssertNil(state.frameTime)
        state.startDrawing(.face, on: "p1")
        XCTAssertNil(state.frameTime, "a photo's drawing has no frame time")
        let box = StageBox(kind: .face, boxId: 1, x: 0.1, y: 0.1, w: 0.2, h: 0.2)
        state.startAdjusting(box, on: "v1", frameTime: 2.25)
        XCTAssertEqual(state.frameTime, 2.25)
    }

    func testOnAnItemTheOwnersTagOrBoxWinsOverTheSystems() throws {
        let sam = #"{"id": 7, "name": "Sam"}"#
        let ann = #"{"id": 8, "name": "Ann"}"#
        let i = try item(faces: "[\(face(1, person: sam)), \(face(2, person: sam, assignedBy: "user")), \(face(3, person: ann))]",
                         tags: #"[{"id": 9, "name": "Tia"}]"#)
        let origins = BoxesLookup.itemOrigins(i)
        XCTAssertEqual(origins[7], .you)
        XCTAssertEqual(origins[8], .auto)
        XCTAssertEqual(origins[9], .you)
    }
}
