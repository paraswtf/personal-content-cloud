import XCTest
@testable import PersonalContentCloud

/// The owner's rule for overlapping boxes (docs/ARCHITECTURE.md section 14): the smaller box is drawn on top and gets
/// the tap. The same cases as the website's frontend/scripts/check-boxes.mjs, plus the tap itself.
final class BoxStackTests: XCTestCase {
    private func face(_ id: Int, _ x: Double, _ y: Double, _ w: Double, _ h: Double) -> StageBox {
        StageBox(kind: .face, boxId: id, x: x, y: y, w: w, h: h)
    }

    private func body(_ id: Int, _ x: Double, _ y: Double, _ w: Double, _ h: Double) -> StageBox {
        StageBox(kind: .body, boxId: id, x: x, y: y, w: w, h: h)
    }

    private func pet(_ id: Int, _ x: Double, _ y: Double, _ w: Double, _ h: Double) -> StageBox {
        StageBox(kind: .animal, boxId: id, x: x, y: y, w: w, h: h)
    }

    /// Every order of `list`.
    private func orders(_ list: [StageBox]) -> [[StageBox]] {
        if list.count <= 1 { return [list] }
        var out: [[StageBox]] = []
        for (i, first) in list.enumerated() {
            var rest = list
            rest.remove(at: i)
            for tail in orders(rest) { out.append([first] + tail) }
        }
        return out
    }

    func testAFaceInsideAPersonBoxIsAboveItThoughThePersonBoxComesLater() {
        let f = face(1, 0.4, 0.2, 0.1, 0.12)
        let b = body(2, 0.3, 0.1, 0.4, 0.8)
        let layers = BoxStack.layers([f, b])
        XCTAssertEqual(layers[b.id], 1)
        XCTAssertEqual(layers[f.id], 2)
        XCTAssertEqual(BoxStack.paintOrder([f, b]).map(\.id), [b.id, f.id])
    }

    func testASmallPersonBoxIsAboveALargePetAndAFaceInThePetIsAboveThePet() {
        let small = body(5, 0.45, 0.45, 0.1, 0.2)
        let dog = pet(9, 0.1, 0.3, 0.8, 0.6)
        let f = face(3, 0.2, 0.35, 0.06, 0.08)
        let layers = BoxStack.layers([f, small, dog])
        XCTAssertGreaterThan(layers[small.id]!, layers[dog.id]!)
        XCTAssertGreaterThan(layers[f.id]!, layers[dog.id]!)
        XCTAssertGreaterThan(layers[f.id]!, layers[small.id]!)
    }

    func testTheLayersDoNotDependOnTheOrderTheBoxesComeIn() {
        let boxes = [face(1, 0.1, 0.1, 0.1, 0.1), face(2, 0.12, 0.12, 0.05, 0.06), body(3, 0.05, 0.05, 0.5, 0.9),
                     body(4, 0.3, 0.2, 0.2, 0.7), pet(5, 0.0, 0.5, 0.9, 0.45)]
        let want = BoxStack.layers(boxes)
        let all = orders(boxes)
        XCTAssertEqual(all.count, 120)
        for order in all {
            XCTAssertEqual(BoxStack.layers(order), want, order.map(\.id).joined(separator: " "))
        }
    }

    func testLayersRunFromOneForTheLargestToTheCountForTheSmallest() {
        let boxes = [body(1, 0, 0, 0.9, 0.9), face(2, 0, 0, 0.2, 0.2), face(3, 0, 0, 0.1, 0.1), pet(4, 0, 0, 0.5, 0.5)]
        let layers = BoxStack.layers(boxes)
        XCTAssertEqual(layers.values.sorted(), [1, 2, 3, 4])
        XCTAssertEqual(layers.sorted { $0.value < $1.value }.map { $0.key }, ["body:1", "animal:4", "face:2", "face:3"])
    }

    func testForEveryPairTheSmallerBoxHasTheHigherLayer() {
        var boxes: [StageBox] = []
        for i in 0..<40 {
            let w = 0.02 + Double((i * 37) % 23) / 40
            let h = 0.03 + Double((i * 53) % 19) / 30
            boxes.append(i % 3 == 0 ? face(i, 0.1, 0.1, w, h) : i % 3 == 1 ? body(i, 0.1, 0.1, w, h) : pet(i, 0.1, 0.1, w, h))
        }
        let layers = BoxStack.layers(boxes)
        for a in boxes {
            for b in boxes where BoxStack.area(a) < BoxStack.area(b) {
                XCTAssertGreaterThan(layers[a.id]!, layers[b.id]!, "\(a.id) over \(b.id)")
            }
        }
    }

    func testEqualAreasKeepTheListOrderTheLaterBoxAbove() {
        let a = face(1, 0.1, 0.1, 0.2, 0.3)
        let b = body(1, 0.15, 0.1, 0.3, 0.2)
        XCTAssertEqual(BoxStack.area(a), BoxStack.area(b), accuracy: 1e-12)
        XCTAssertEqual(BoxStack.layers([a, b])[a.id], 1)
        XCTAssertEqual(BoxStack.layers([a, b])[b.id], 2)
        XCTAssertEqual(BoxStack.layers([b, a])[b.id], 1)
        XCTAssertEqual(BoxStack.layers([b, a])[a.id], 2)
    }

    func testABoxOfNoSizeOrABrokenSizeCountsAsTheSmallest() {
        let big = body(1, 0, 0, 0.8, 0.8)
        let mid = face(2, 0, 0, 0.3, 0.3)
        let broken = [face(3, 0, 0, .nan, 0.2), face(4, 0, 0, -0.1, 0.2), face(5, 0, 0, 0, 0.2), face(6, 0, 0, .infinity, 0.1)]
        for box in broken { XCTAssertEqual(BoxStack.area(box), 0) }
        let layers = BoxStack.layers([big] + broken + [mid])
        XCTAssertEqual(layers[big.id], 1)
        XCTAssertEqual(layers[mid.id], 2)
        for box in broken { XCTAssertGreaterThan(layers[box.id]!, 2) }
        XCTAssertTrue(BoxStack.layers([]).isEmpty)
    }

    func testATapInsideBothBoxesGoesToTheSmallerOne() {
        let b = body(2, 0.3, 0.1, 0.4, 0.8)
        let f = face(1, 0.4, 0.2, 0.1, 0.12)
        // In both orders: the face's middle is inside the person box too.
        for boxes in [[f, b], [b, f]] {
            XCTAssertEqual(BoxStack.hit(boxes, at: CGPoint(x: 0.45, y: 0.26))?.id, f.id)
            XCTAssertEqual(BoxStack.hit(boxes, at: CGPoint(x: 0.65, y: 0.8))?.id, b.id)
            XCTAssertNil(BoxStack.hit(boxes, at: CGPoint(x: 0.1, y: 0.1)))
        }
    }

    func testATinyBoxIsWidenedForTheTapAndStillWinsOverTheLargeOne() {
        let b = body(2, 0.0, 0.0, 1.0, 1.0)
        let tiny = face(1, 0.5, 0.5, 0.01, 0.01)
        // Just outside the tiny box: without slop the person box gets it, with slop the face does.
        let near = CGPoint(x: 0.5 + 0.012, y: 0.505)
        XCTAssertEqual(BoxStack.hit([tiny, b], at: near)?.id, b.id)
        XCTAssertEqual(BoxStack.hit([tiny, b], at: near, slopX: 0.04, slopY: 0.04)?.id, tiny.id)
    }

    func testTheStillBoxesOfAPhotoAreFacesFacelessPersonBoxesAndPets() throws {
        let json = """
        {"uid": "a1", "kind": "image", "version": 1, "filename": "IMG_1.JPG", "created_at": "2026-11-01T10:00:00.000Z",
         "urls": {"thumb": "/files/thumb/a1?v=1", "preview": "/files/preview/a1?v=1", "preview_small": "/files/preview-small/a1?v=1"},
         "faces": [
           {"id": 11, "x": 0.42, "y": 0.2, "w": 0.08, "h": 0.1, "frame_time_s": null, "person": {"id": 3, "name": "Test Person"}},
           {"id": 12, "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "frame_time_s": 3.2},
           {"id": 13, "x": 0.6, "y": 0.2, "w": 0.08, "h": 0.1, "person": null, "guess": {"person": {"id": 4, "name": "Test Visitor"}, "score": 0.7}},
           {"id": 14, "x": 0.7, "y": 0.2, "w": 0.08, "h": 0.1, "person": null, "guess": {"person": {"id": 5, "name": null}, "score": 0.7}}
         ],
         "bodies": [
           {"id": 21, "x": 0.3, "y": 0.1, "w": 0.4, "h": 0.8, "face_id": 11},
           {"id": 22, "x": 0.05, "y": 0.1, "w": 0.2, "h": 0.8, "face_id": null, "person": {"id": 3, "name": "Test Person"}}
         ],
         "animals": [{"id": 31, "x": 0.5, "y": 0.6, "w": 0.3, "h": 0.3, "species": "dog"}]}
        """
        let detail = try CoreHTTP.decoder.decode(CoreItemDetail.self, from: Data(json.utf8))
        let boxes = detail.stillBoxes
        XCTAssertEqual(boxes.map(\.id), ["face:11", "face:13", "face:14", "body:22", "animal:31"])
        XCTAssertEqual(boxes[0].name, "Test Person")
        XCTAssertEqual(boxes[1].guessName, "Test Visitor")
        XCTAssertNil(boxes[2].guessName, "a guess for someone without a name is never shown")
        XCTAssertNil(boxes[2].name)
        XCTAssertEqual(boxes[4].name, "Dog")
        XCTAssertEqual(boxes[2].accessibilityText, "Face, not named")
    }
}
