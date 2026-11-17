import SwiftUI
import Observation

/// What a new box is drawn for (the website's DrawKind): a face the detector missed, a person (a back, a helmet,
/// someone far away) or a pet (docs/ARCHITECTURE.md sections 10 and 22).
enum StageDrawKind: String, Hashable {
    case face, person, pet

    /// "face", "person", "pet": what the owner drags over.
    var noun: String { rawValue }
}

/// The viewer's editing state, shared by the boxes area and the stage. The boxes area owns one and puts it in the
/// environment (`.viewerBoxEditors`). While a box is being moved or resized, or a new box drawn, on the item `uid`:
/// the viewer hides its bars, stops paging and closes its info panel; the stage of that item shows handles or a
/// drawing layer, where one finger edits while two fingers pan and zoom. `rect` is the edited rectangle in the
/// picture's 0 to 1 units. The viewer also keeps each page's latest context here, so the editors show the item as
/// it is after a save.
@MainActor
@Observable
final class StageEditState {
    enum Mode: Hashable {
        case adjust(StageBox)
        case draw(StageDrawKind)
    }

    private(set) var uid: String?
    private(set) var mode: Mode?
    /// The edited rectangle: the box's while adjusting, the drawn one while drawing (nil until something is drawn).
    private(set) var rect: CGRect?
    /// Bumped whenever `rect` is set from outside the stage (starting, "Whole photo", "Draw again"), so the stage
    /// takes it; the stage's own drags do not bump it.
    private(set) var revision = 0
    /// True while a finger draws or drags on the stage.
    var dragging = false
    /// A video's edit: the time of the paused frame the box is drawn or moved on (nil on a photo). The frame
    /// shown and the time sent with a new box are both this one.
    private(set) var frameTime: Double?
    /// The latest context of each page the viewer loaded, by uid (the viewer core keeps it current).
    var contexts: [String: ViewerPageContext] = [:]

    var isActive: Bool { mode != nil }

    /// Whether the item `uid` is being edited.
    func isEditing(_ uid: String) -> Bool { mode != nil && self.uid == uid }

    /// The box being moved or resized, if any.
    var adjustingBox: StageBox? {
        if case .adjust(let box) = mode { return box }
        return nil
    }

    /// What is being drawn, if anything.
    var drawingKind: StageDrawKind? {
        if case .draw(let kind) = mode { return kind }
        return nil
    }

    func startAdjusting(_ box: StageBox, on uid: String, frameTime: Double? = nil) {
        self.uid = uid
        self.frameTime = frameTime
        mode = .adjust(box)
        rect = CGRect(x: box.x, y: box.y, width: box.w, height: box.h)
        dragging = false
        revision += 1
    }

    func startDrawing(_ kind: StageDrawKind, on uid: String, frameTime: Double? = nil) {
        self.uid = uid
        self.frameTime = frameTime
        mode = .draw(kind)
        rect = nil
        dragging = false
        revision += 1
    }

    /// Sets the rectangle from outside the stage (the stage shows it).
    func setRect(_ newRect: CGRect?) {
        rect = newRect
        revision += 1
    }

    /// The stage's own change while a finger moves (the stage already shows it).
    func stageChanged(_ newRect: CGRect?) {
        rect = newRect
    }

    func stop() {
        uid = nil
        frameTime = nil
        mode = nil
        rect = nil
        dragging = false
        revision += 1
    }
}

/// The arithmetic of moving, resizing and drawing a box, in the picture's 0 to 1 units (the website's
/// BoxAdjuster and PhotoStage draw mode).
enum StageEditGeometry {
    enum Handle: Hashable {
        case move, topLeft, topRight, bottomLeft, bottomRight
    }

    /// The smallest side the owner can make (the website's MIN_SIDE; the server says the same), unless the box
    /// already was smaller.
    static let minSide: CGFloat = 0.015
    /// The margin "Whole photo" leaves around a drawn box (the website's WHOLE_MARGIN).
    static let wholeMargin: CGFloat = 0.03
    /// The smallest box the owner can draw, in points on the screen (the website's MIN_DRAW_PX).
    static let minDrawPoints: CGFloat = 24

    static func clamp(_ v: CGFloat, _ lo: CGFloat, _ hi: CGFloat) -> CGFloat { min(max(v, lo), max(lo, hi)) }

    /// The smallest sides for a box that started as `start`.
    static func minSides(for start: CGRect) -> (w: CGFloat, h: CGFloat) {
        (min(minSide, start.width), min(minSide, start.height))
    }

    /// The box after a drag of (`dx`, `dy`) by `handle`: the box moves inside the picture, or the corner moves and
    /// the opposite corner stays, never smaller than `minW` x `minH`.
    static func moved(_ handle: Handle, _ b: CGRect, dx: CGFloat, dy: CGFloat, minW: CGFloat, minH: CGFloat) -> CGRect {
        if handle == .move {
            return CGRect(x: clamp(b.minX + dx, 0, 1 - b.width), y: clamp(b.minY + dy, 0, 1 - b.height),
                          width: b.width, height: b.height)
        }
        var x1 = b.minX, y1 = b.minY, x2 = b.maxX, y2 = b.maxY
        if handle == .topLeft || handle == .bottomLeft {
            x1 = clamp(x1 + dx, 0, x2 - minW)
        } else {
            x2 = clamp(x2 + dx, x1 + minW, 1)
        }
        if handle == .topLeft || handle == .topRight {
            y1 = clamp(y1 + dy, 0, y2 - minH)
        } else {
            y2 = clamp(y2 + dy, y1 + minH, 1)
        }
        return CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
    }

    /// What a touch at `p` takes: a corner within `reachX` x `reachY` of it (the nearest one), else the box when
    /// the touch is inside it, else nothing.
    static func handle(at p: CGPoint, in b: CGRect, reachX: CGFloat, reachY: CGFloat) -> Handle? {
        let corners: [(Handle, CGPoint)] = [
            (.topLeft, CGPoint(x: b.minX, y: b.minY)), (.topRight, CGPoint(x: b.maxX, y: b.minY)),
            (.bottomLeft, CGPoint(x: b.minX, y: b.maxY)), (.bottomRight, CGPoint(x: b.maxX, y: b.maxY)),
        ]
        var best: (Handle, CGFloat)?
        for (handle, c) in corners {
            let dx = abs(p.x - c.x), dy = abs(p.y - c.y)
            guard dx <= reachX, dy <= reachY else { continue }
            let d = (dx / max(reachX, 1e-6)) + (dy / max(reachY, 1e-6))
            if best == nil || d < best!.1 { best = (handle, d) }
        }
        if let best { return best.0 }
        return b.contains(p) ? .move : nil
    }

    /// The box drawn from one corner to the other, kept inside the picture; nil when it is smaller than
    /// `minW` x `minH`.
    static func drawn(from a: CGPoint, to c: CGPoint, minW: CGFloat, minH: CGFloat) -> CGRect? {
        let x0 = clamp(min(a.x, c.x), 0, 1), x1 = clamp(max(a.x, c.x), 0, 1)
        let y0 = clamp(min(a.y, c.y), 0, 1), y1 = clamp(max(a.y, c.y), 0, 1)
        guard x1 - x0 >= minW, y1 - y0 >= minH else { return nil }
        return CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }

    /// The box grown (or shrunk, with a negative `by`) around its middle, keeping its shape and staying inside the
    /// picture (the website's + and - keys).
    static func grown(_ b: CGRect, by: CGFloat, minW: CGFloat, minH: CGFloat) -> CGRect {
        let w = clamp(b.width + by, minW, 1)
        let h = clamp(b.height + by * (b.height / max(b.width, 1e-6)), minH, 1)
        let x = clamp(b.minX + (b.width - w) / 2, 0, 1 - w)
        let y = clamp(b.minY + (b.height - h) / 2, 0, 1 - h)
        return CGRect(x: x, y: y, width: w, height: h)
    }

    /// Whether a box moved or changed size enough to save (the website treats less than 1e-4 in all as unmoved).
    static func changed(_ a: CGRect, _ b: CGRect) -> Bool {
        abs(a.minX - b.minX) + abs(a.minY - b.minY) + abs(a.width - b.width) + abs(a.height - b.height) >= 1e-4
    }

    /// "Whole photo": the picture less a small margin.
    static var wholePhoto: CGRect {
        CGRect(x: wholeMargin, y: wholeMargin, width: 1 - 2 * wholeMargin, height: 1 - 2 * wholeMargin)
    }
}
