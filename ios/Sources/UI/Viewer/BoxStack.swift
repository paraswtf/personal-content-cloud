import Foundation
import CoreGraphics

/// One box drawn over a photo: a face, a person (body) or a pet. Coordinates are shares of the picture's width
/// and height (0 to 1), as the server stores them.
struct StageBox: Identifiable, Hashable {
    enum Kind: String { case face, body, animal }

    let kind: Kind
    let boxId: Int
    let x: Double
    let y: Double
    let w: Double
    let h: Double
    /// The named person or pet, or the species of an unnamed pet; nil for a face nobody has named.
    var name: String?
    /// A guess's named person, for a box without a person (never an unnamed one).
    var guessName: String?
    /// The person the box belongs to, when it belongs to one.
    var personId: Int?

    var id: String { "\(kind.rawValue):\(boxId)" }

    /// What VoiceOver reads: "Face of Sam", "Person box, maybe Sam", "Face, not named".
    var accessibilityText: String {
        let noun: String
        switch kind {
        case .face: noun = "Face"
        case .body: noun = "Person box"
        case .animal: noun = "Pet"
        }
        if let name { return kind == .animal ? "\(noun): \(name)" : "\(noun) of \(name)" }
        if let guessName { return "\(noun), maybe \(guessName)" }
        return "\(noun), not named"
    }

    /// The label drawn with the box, if any.
    var label: String? {
        if let name { return name }
        if let guessName { return "\(guessName)?" }
        return nil
    }
}

/// Overlapping boxes (the owner's rule, docs/ARCHITECTURE.md section 14): the smaller box is drawn above a larger
/// one and gets the tap, so a face inside a person box, or a child in front of a grown-up, can always be tapped.
/// The layer comes from the area alone: being selected or edited never raises a box. Equal areas keep the
/// list's order, the later one above. The same rule as the website's `stackLayers` (viewer/boxes.ts).
enum BoxStack {
    /// A box's area in the picture's own units; every box of one picture has the same scale.
    static func area(_ box: StageBox) -> Double {
        func side(_ v: Double) -> Double { v.isFinite && v > 0 ? v : 0 }
        return side(box.w) * side(box.h)
    }

    /// Each box's layer by id: 1 for the largest, `boxes.count` for the smallest.
    static func layers(_ boxes: [StageBox]) -> [String: Int] {
        var out: [String: Int] = [:]
        for (n, box) in paintOrder(boxes).enumerated() { out[box.id] = n + 1 }
        return out
    }

    /// The boxes in painting order, largest first: the last one is on top and is hit first.
    static func paintOrder(_ boxes: [StageBox]) -> [StageBox] {
        let indexed = Array(boxes.enumerated())
        let sorted = indexed.sorted { a, b in
            let areaA = area(a.element), areaB = area(b.element)
            if areaA != areaB { return areaA > areaB }
            return a.offset < b.offset
        }
        return sorted.map { $0.element }
    }

    /// The box a tap at `point` (in the picture's 0 to 1 units) lands on: the topmost one containing it.
    /// `slop` widens small boxes (in the same units) so they stay easy to hit.
    static func hit(_ boxes: [StageBox], at point: CGPoint, slopX: Double = 0, slopY: Double = 0) -> StageBox? {
        for box in paintOrder(boxes).reversed() {
            let padX = max(0, (slopX - box.w) / 2), padY = max(0, (slopY - box.h) / 2)
            let rect = CGRect(x: box.x - padX, y: box.y - padY, width: box.w + 2 * padX, height: box.h + 2 * padY)
            if rect.contains(point) { return box }
        }
        return nil
    }
}
