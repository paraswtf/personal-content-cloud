import XCTest
import UIKit
@testable import PersonalContentCloud

/// The origin on the photo's box labels (the website's PhotoStage with OriginIcon, docs/ARCHITECTURE.md section 10):
/// who put the person, the pet or the guess on each still box, the icon after its label and the words VoiceOver reads.
@MainActor
final class StageBoxOriginTests: XCTestCase {
    private func stillBoxes(faces: String = "[]", bodies: String = "[]", animals: String = "[]") throws -> [StageBox] {
        let json = """
        {"uid": "o1", "kind": "image", "version": 1, "filename": "IMG_2.JPG", "created_at": "2026-11-01T10:00:00.000Z",
         "urls": {"thumb": "/files/thumb/o1?v=1", "preview": "/files/preview/o1?v=1", "preview_small": "/files/preview-small/o1?v=1"},
         "faces": \(faces), "bodies": \(bodies), "animals": \(animals)}
        """
        return try CoreHTTP.decoder.decode(CoreItemDetail.self, from: Data(json.utf8)).stillBoxes
    }

    private let font = UIFont.systemFont(ofSize: 12, weight: .semibold)

    private func hasIcon(_ text: NSAttributedString?) -> Bool {
        guard let text else { return false }
        var found = false
        text.enumerateAttribute(.attachment, in: NSRange(location: 0, length: text.length), options: []) { value, _, _ in
            if value is NSTextAttachment { found = true }
        }
        return found
    }

    func testEachFaceCarriesWhoPutThePersonThere() throws {
        let boxes = try stillBoxes(faces: """
        [
          {"id": 1, "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "person": {"id": 3, "name": "Test Person"}, "assigned_by": "user", "source": "detector", "by_guess": false},
          {"id": 2, "x": 0.2, "y": 0.1, "w": 0.1, "h": 0.1, "person": {"id": 3, "name": "Test Person"}, "assigned_by": "auto", "source": "detector", "by_guess": false},
          {"id": 3, "x": 0.3, "y": 0.1, "w": 0.1, "h": 0.1, "person": {"id": 3, "name": "Test Person"}, "assigned_by": "user", "source": "detector", "by_guess": true},
          {"id": 4, "x": 0.4, "y": 0.1, "w": 0.1, "h": 0.1, "person": {"id": 3, "name": "Test Person"}, "assigned_by": "auto", "source": "user", "by_guess": false},
          {"id": 5, "x": 0.5, "y": 0.1, "w": 0.1, "h": 0.1, "person": null, "assigned_by": "guess", "guess": {"person": {"id": 4, "name": "Test Visitor"}, "score": 0.7}},
          {"id": 6, "x": 0.6, "y": 0.1, "w": 0.1, "h": 0.1, "person": null, "assigned_by": "auto", "source": "detector"},
          {"id": 7, "x": 0.7, "y": 0.1, "w": 0.1, "h": 0.1, "person": null, "assigned_by": "auto", "source": "user"},
          {"id": 8, "x": 0.8, "y": 0.1, "w": 0.1, "h": 0.1, "person": null, "guess": {"person": {"id": 5, "name": null}, "score": 0.7}},
          {"id": 9, "x": 0.9, "y": 0.1, "w": 0.05, "h": 0.1, "person": {"id": 6, "name": null}, "assigned_by": "auto"},
          {"id": 10, "x": 0.1, "y": 0.5, "w": 0.1, "h": 0.1, "person": {"id": 3, "name": "Test Person"}}
        ]
        """)
        // tagged by the owner, found by the system, a guess the owner confirmed, a box the owner drew with someone the
        // system matched on it, a guess waiting, nobody, a drawn box with nobody, a guess for someone without a name,
        // a person without a name, and an older server that sends no origin fields
        XCTAssertEqual(boxes.map(\.origin), [.you, .auto, .confirmed, .drawn, .guess, nil, .drawn, nil, nil, .auto])
        XCTAssertEqual(boxes.map(\.labelOrigin), [.you, .auto, .confirmed, .drawn, .guess, nil, nil, nil, nil, .auto],
                       "a box without a label draws no icon")
        XCTAssertEqual(boxes[8].accessibilityText, "Face, not named", "a person without a name is never shown")
    }

    func testPersonBoxesAndPetsCarryTheirOriginToo() throws {
        let boxes = try stillBoxes(bodies: """
        [
          {"id": 21, "x": 0.1, "y": 0.1, "w": 0.3, "h": 0.8, "face_id": null, "person": {"id": 3, "name": "Test Person"}, "assigned_by": "user", "source": "user", "by_guess": false},
          {"id": 22, "x": 0.5, "y": 0.1, "w": 0.3, "h": 0.8, "face_id": null, "person": null, "guess": {"person": {"id": 4, "name": "Test Visitor"}, "score": 0.6}, "source": "detector"}
        ]
        """, animals: """
        [
          {"id": 31, "x": 0.1, "y": 0.6, "w": 0.2, "h": 0.2, "species": "dog", "person": {"id": 7, "name": "Test Pet", "kind": "pet"}, "assigned_by": "user", "source": "detector"},
          {"id": 32, "x": 0.3, "y": 0.6, "w": 0.2, "h": 0.2, "species": "dog", "person": {"id": 8, "name": null, "kind": "pet"}, "assigned_by": "auto", "source": "detector"},
          {"id": 33, "x": 0.5, "y": 0.6, "w": 0.2, "h": 0.2, "species": "cat", "person": null, "assigned_by": "auto", "source": "detector"},
          {"id": 34, "x": 0.7, "y": 0.6, "w": 0.2, "h": 0.2, "species": "cat", "person": null, "assigned_by": "auto", "source": "user"}
        ]
        """)
        XCTAssertEqual(boxes.map(\.id), ["body:21", "body:22", "animal:31", "animal:32", "animal:33", "animal:34"])
        // the owner tagged the box they drew: theirs, not "drawn"; a pet without a name still has who put it there
        XCTAssertEqual(boxes.map(\.origin), [.you, .guess, .you, .auto, nil, .drawn])
        XCTAssertEqual(boxes.map(\.label), ["Test Person", "Test Visitor?", "Test Pet", "Dog", "Cat", "Cat"])
        XCTAssertEqual(boxes.map(\.labelOrigin), [.you, .guess, .you, .auto, nil, .drawn])
    }

    func testTheOwnersOriginsShowAPersonAndTheRestTheSystemsIcon() {
        XCTAssertEqual(BoxOrigin.allCases.map(\.symbolName), ["person.fill", "person.fill", "cpu", "cpu", "cpu"])
        XCTAssertEqual(BoxOrigin.allCases, [.you, .drawn, .confirmed, .auto, .guess])
        for origin in BoxOrigin.allCases {
            XCTAssertNotNil(UIImage(systemName: origin.symbolName), "no symbol for \(origin)")
        }
    }

    func testTheLabelEndsWithTheIconAndVoiceOverReadsTheWords() throws {
        let boxes = try stillBoxes(faces: """
        [
          {"id": 1, "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "person": {"id": 3, "name": "Test Person"}, "assigned_by": "user", "source": "detector", "by_guess": false},
          {"id": 5, "x": 0.5, "y": 0.1, "w": 0.1, "h": 0.1, "person": null, "guess": {"person": {"id": 4, "name": "Test Visitor"}, "score": 0.7}},
          {"id": 6, "x": 0.6, "y": 0.1, "w": 0.1, "h": 0.1, "person": null, "assigned_by": "auto", "source": "detector"},
          {"id": 7, "x": 0.7, "y": 0.1, "w": 0.1, "h": 0.1, "person": null, "assigned_by": "auto", "source": "user"}
        ]
        """)
        let named = BoxView.labelText(boxes[0], font: font)
        XCTAssertTrue(named?.string.hasPrefix("Test Person") ?? false)
        XCTAssertTrue(hasIcon(named), "the name has no origin icon")
        XCTAssertTrue(hasIcon(BoxView.labelText(boxes[1], font: font)), "the guess has no origin icon")
        XCTAssertNil(BoxView.labelText(boxes[2], font: font), "a box with nobody draws no label")
        XCTAssertFalse(hasIcon(BoxView.labelText(StageBox(kind: .face, boxId: 9, x: 0, y: 0, w: 1, h: 1, name: "Test Person"),
                                                 font: font)), "a box without an origin draws no icon")

        // The label stays what it was (the screen tests find boxes by it); the origin is the value VoiceOver reads.
        let tagged = BoxView(box: boxes[0])
        XCTAssertEqual(tagged.accessibilityLabel, "Face of Test Person")
        XCTAssertEqual(tagged.accessibilityValue, "Tagged by you")
        let guess = BoxView(box: boxes[1])
        XCTAssertEqual(guess.accessibilityLabel, "Face, maybe Test Visitor")
        XCTAssertEqual(guess.accessibilityValue, "Suggested automatically")
        XCTAssertNil(BoxView(box: boxes[2]).accessibilityValue)
        XCTAssertEqual(BoxView(box: boxes[3]).accessibilityValue, "Drawn by you", "a drawn box with nobody says so")
    }

    func testTheOriginDoesNotChangeWhichBoxIsOnTop() throws {
        // A small face inside a larger person box, each with another origin: the face is still drawn last and hit
        // first.
        let face = StageBox(kind: .face, boxId: 1, x: 0.4, y: 0.2, w: 0.1, h: 0.12, name: "Test Visitor", origin: .auto)
        let body = StageBox(kind: .body, boxId: 2, x: 0.3, y: 0.1, w: 0.4, h: 0.8, name: "Test Person", origin: .you)
        for boxes in [[face, body], [body, face]] {
            XCTAssertEqual(BoxStack.paintOrder(boxes).last?.id, face.id)
            XCTAssertEqual(BoxStack.hit(boxes, at: CGPoint(x: 0.45, y: 0.26))?.id, face.id)
        }
    }
}
