import SwiftUI
import UIKit

/// A photo that pinches and double-taps to zoom, with its boxes drawn over it. A tap goes to the topmost box under
/// the finger, which is the smallest one there (BoxStack), else to `onTap` (which shows or hides the controls).
/// While `edit` names this picture, the stage moves or resizes a box, or draws a new one: one finger edits, two
/// fingers pan and zoom, and taps on boxes wait.
struct ZoomableStage: UIViewRepresentable {
    var image: UIImage?
    /// The picture's size in pixels when known before it loads, so the boxes can be placed at once.
    var pixelSize: CGSize?
    var boxes: [StageBox]
    var showBoxes: Bool
    var selectedBoxId: String?
    @Binding var zoomed: Bool
    var onBoxTap: (StageBox) -> Void
    var onTap: () -> Void
    /// The boxes area's edit of this picture, or nil.
    var edit: StageEditState? = nil
    /// Changes whenever the edit starts, stops or takes a rectangle from outside, so the stage updates.
    var editRevision = 0

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    func makeUIView(context: Context) -> StageView {
        let view = StageView(frame: .zero)
        view.coordinator = context.coordinator
        return view
    }

    func updateUIView(_ view: StageView, context: Context) {
        context.coordinator.parent = self
        view.apply(image: image, pixelSize: pixelSize, boxes: boxes, showBoxes: showBoxes, selected: selectedBoxId)
        view.applyEdit(edit, revision: editRevision)
    }

    final class Coordinator {
        var parent: ZoomableStage

        init(parent: ZoomableStage) { self.parent = parent }

        func zoomChanged(_ isZoomed: Bool) {
            guard parent.zoomed != isZoomed else { return }
            DispatchQueue.main.async { self.parent.zoomed = isZoomed }
        }

        func tapped(_ box: StageBox?) {
            if let box { parent.onBoxTap(box) } else { parent.onTap() }
        }
    }
}

/// The zooming surface: a scroll view whose content is the picture with one view per box. Boxes are added
/// largest first, so the smallest is on top; taps are hit-tested in that order by the stage itself.
final class StageView: UIScrollView, UIScrollViewDelegate {
    weak var coordinator: ZoomableStage.Coordinator?

    private let content = UIView()
    private let imageView = UIImageView()
    private var boxViews: [String: BoxView] = [:]
    private var boxes: [StageBox] = []
    private var showBoxes = true
    private var selected: String?
    private var pictureSize: CGSize = .zero
    private var lastBounds: CGSize = .zero
    private let singleTapper: UITapGestureRecognizer
    private let doubleTapper: UITapGestureRecognizer
    private var editOverlay: StageEditOverlay?
    private var editRevision = -1
    /// The id of the box being moved (drawn by the overlay instead), while adjusting.
    private var adjustingId: String?

    override init(frame: CGRect) {
        singleTapper = UITapGestureRecognizer()
        doubleTapper = UITapGestureRecognizer()
        super.init(frame: frame)
        delegate = self
        showsHorizontalScrollIndicator = false
        showsVerticalScrollIndicator = false
        contentInsetAdjustmentBehavior = .never
        decelerationRate = .fast
        bouncesZoom = true
        minimumZoomScale = 1
        maximumZoomScale = 5
        backgroundColor = .clear
        imageView.contentMode = .scaleAspectFit
        imageView.clipsToBounds = true
        imageView.isAccessibilityElement = false
        content.addSubview(imageView)
        content.isAccessibilityElement = false
        addSubview(content)

        singleTapper.addTarget(self, action: #selector(singleTap(_:)))
        doubleTapper.addTarget(self, action: #selector(doubleTap(_:)))
        doubleTapper.numberOfTapsRequired = 2
        singleTapper.require(toFail: doubleTapper)
        addGestureRecognizer(singleTapper)
        addGestureRecognizer(doubleTapper)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    func apply(image: UIImage?, pixelSize: CGSize?, boxes: [StageBox], showBoxes: Bool, selected: String?) {
        var relayout = false
        if imageView.image !== image {
            imageView.image = image
        }
        var size = CGSize.zero
        if let image, image.size.width > 0, image.size.height > 0 {
            size = CGSize(width: image.size.width * image.scale, height: image.size.height * image.scale)
        } else if let pixelSize, pixelSize.width > 0, pixelSize.height > 0 {
            size = pixelSize
        }
        // A thumb and its preview have the same shape: only a new shape lays the picture out again.
        if size.width > 0, !Self.sameShape(size, pictureSize) {
            pictureSize = size
            relayout = true
        }
        if boxes != self.boxes {
            self.boxes = boxes
            rebuildBoxes()
        }
        if showBoxes != self.showBoxes || selected != self.selected {
            self.showBoxes = showBoxes
            self.selected = selected
            styleBoxes()
        }
        if relayout { resetLayout() }
    }

    /// Starts, updates or ends the boxes area's edit of this picture.
    func applyEdit(_ state: StageEditState?, revision: Int) {
        guard let state, let mode = state.mode else {
            if editOverlay != nil { endEdit() }
            return
        }
        let overlay: StageEditOverlay
        if let existing = editOverlay {
            overlay = existing
        } else {
            overlay = StageEditOverlay(frame: content.bounds)
            content.addSubview(overlay)
            editOverlay = overlay
            // One finger edits; two fingers pan and zoom. Taps on boxes wait until the edit ends.
            panGestureRecognizer.minimumNumberOfTouches = 2
            singleTapper.isEnabled = false
            doubleTapper.isEnabled = false
        }
        overlay.state = state
        overlay.frame = content.bounds
        overlay.zoom = zoomScale
        if revision != editRevision {
            editRevision = revision
            overlay.configure(mode: mode, rect: state.rect)
        }
        let newAdjusting: String?
        if case .adjust(let box) = mode { newAdjusting = box.id } else { newAdjusting = nil }
        if newAdjusting != adjustingId {
            adjustingId = newAdjusting
        }
        styleBoxes()
    }

    private func endEdit() {
        editOverlay?.removeFromSuperview()
        editOverlay = nil
        editRevision = -1
        adjustingId = nil
        panGestureRecognizer.minimumNumberOfTouches = 1
        singleTapper.isEnabled = true
        doubleTapper.isEnabled = true
        styleBoxes()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        if bounds.size != lastBounds {
            lastBounds = bounds.size
            resetLayout()
        }
    }

    // MARK: layout

    private static func sameShape(_ a: CGSize, _ b: CGSize) -> Bool {
        guard a.width > 0, a.height > 0, b.width > 0, b.height > 0 else { return false }
        return abs(a.width / a.height - b.width / b.height) < 0.01
    }

    private func fittedSize() -> CGSize {
        guard pictureSize.width > 0, pictureSize.height > 0, bounds.width > 0, bounds.height > 0 else { return bounds.size }
        let scale = min(bounds.width / pictureSize.width, bounds.height / pictureSize.height)
        return CGSize(width: (pictureSize.width * scale).rounded(), height: (pictureSize.height * scale).rounded())
    }

    private func resetLayout() {
        zoomScale = 1
        let size = fittedSize()
        content.transform = .identity
        content.frame = CGRect(origin: .zero, size: size)
        imageView.frame = content.bounds
        contentSize = size
        layoutBoxes()
        centerContent()
        coordinator?.zoomChanged(false)
    }

    private func centerContent() {
        let w = content.frame.width, h = content.frame.height
        let x = max(0, (bounds.width - w) / 2), y = max(0, (bounds.height - h) / 2)
        contentInset = UIEdgeInsets(top: y, left: x, bottom: y, right: x)
    }

    // MARK: boxes

    private func rebuildBoxes() {
        boxViews.values.forEach { $0.removeFromSuperview() }
        boxViews = [:]
        for box in BoxStack.paintOrder(boxes) {
            let view = BoxView(box: box)
            view.onActivate = { [weak self] in self?.coordinator?.tapped(box) }
            if let editOverlay {
                content.insertSubview(view, belowSubview: editOverlay)
            } else {
                content.addSubview(view)
            }
            boxViews[box.id] = view
        }
        layoutBoxes()
        styleBoxes()
    }

    private func layoutBoxes() {
        let size = content.bounds.size
        for box in boxes {
            guard let view = boxViews[box.id] else { continue }
            view.frame = CGRect(x: box.x * size.width, y: box.y * size.height,
                                width: box.w * size.width, height: box.h * size.height)
            view.setZoom(zoomScale)
        }
        if let editOverlay {
            editOverlay.frame = content.bounds
            editOverlay.zoom = zoomScale
        }
    }

    private func styleBoxes() {
        let editing = editOverlay != nil
        for (id, view) in boxViews {
            view.isHidden = !showBoxes || id == adjustingId
            // While a box is edited the others stay in view, faded, and take no part.
            view.alpha = editing ? 0.4 : 1
            view.isAccessibilityElement = showBoxes && !editing
            view.setSelected(id == selected)
            view.setZoom(zoomScale)
        }
        // VoiceOver reads the shown boxes in the item's own order: faces, then person boxes, then pets. Hidden boxes
        // leave the list (a listed element stays readable even when its view is hidden). While editing, VoiceOver
        // reads the edited box with its actions instead.
        if let editOverlay {
            content.accessibilityElements = [editOverlay]
        } else {
            content.accessibilityElements = showBoxes ? boxes.compactMap { boxViews[$0.id] } : []
        }
    }

    // MARK: gestures

    @objc private func singleTap(_ g: UITapGestureRecognizer) {
        let size = content.bounds.size
        guard showBoxes, !boxes.isEmpty, size.width > 0, size.height > 0 else {
            coordinator?.tapped(nil)
            return
        }
        let p = g.location(in: content)
        let unit = CGPoint(x: p.x / size.width, y: p.y / size.height)
        // Boxes smaller than about 36 points on screen are widened to that for the tap.
        let minPoints: CGFloat = 36
        let slopX = Double(minPoints / zoomScale / size.width)
        let slopY = Double(minPoints / zoomScale / size.height)
        coordinator?.tapped(BoxStack.hit(boxes, at: unit, slopX: slopX, slopY: slopY))
    }

    @objc private func doubleTap(_ g: UITapGestureRecognizer) {
        if zoomScale > 1.01 {
            setZoomScale(1, animated: true)
            return
        }
        let p = g.location(in: content)
        let scale = min(maximumZoomScale, 2.5)
        let w = bounds.width / scale, h = bounds.height / scale
        zoom(to: CGRect(x: p.x - w / 2, y: p.y - h / 2, width: w, height: h), animated: true)
    }

    // MARK: UIScrollViewDelegate

    func viewForZooming(in scrollView: UIScrollView) -> UIView? { content }

    func scrollViewDidZoom(_ scrollView: UIScrollView) {
        centerContent()
        for view in boxViews.values { view.setZoom(zoomScale) }
        editOverlay?.zoom = zoomScale
        coordinator?.zoomChanged(zoomScale > 1.01)
    }
}

/// The layer over the picture while a box is moved or resized, or a new one drawn. It lives in the picture's own
/// coordinates (it zooms with it) and keeps its lines and handles the same size on the screen. One finger drags:
/// inside the box it moves the box, at a corner it resizes it; in draw mode it draws a new box from where the
/// finger went down. VoiceOver gets the box as one element with actions to move and resize it.
final class StageEditOverlay: UIView {
    weak var state: StageEditState?
    var zoom: CGFloat = 1 {
        didSet { if zoom != oldValue { setNeedsLayout() } }
    }

    private var mode: StageEditState.Mode?
    private var rect: CGRect?
    /// The box as it was when adjusting started: the smallest sides come from it.
    private var startRect: CGRect = .zero
    private var dragStartRect: CGRect = .zero
    private var dragHandle: StageEditGeometry.Handle?
    private var drawStart: CGPoint?
    private let border = CAShapeLayer()
    private var handles: [CAShapeLayer] = []

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        border.fillColor = UIColor.clear.cgColor
        border.lineJoin = .round
        layer.addSublayer(border)
        for _ in 0..<4 {
            let h = CAShapeLayer()
            h.fillColor = UIColor.white.cgColor
            h.strokeColor = UIColor.systemBlue.cgColor
            layer.addSublayer(h)
            handles.append(h)
        }
        let pan = UIPanGestureRecognizer(target: self, action: #selector(panned(_:)))
        pan.maximumNumberOfTouches = 1
        addGestureRecognizer(pan)
        isAccessibilityElement = true
        accessibilityIdentifier = "stage.edit"
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    func configure(mode: StageEditState.Mode, rect: CGRect?) {
        self.mode = mode
        self.rect = rect
        if case .adjust = mode, let rect { startRect = rect }
        updateAccessibility()
        setNeedsLayout()
    }

    private var color: UIColor {
        switch mode {
        case .draw(.face): return .white
        case .draw(.person): return .systemCyan
        case .draw(.pet): return .systemGreen
        default: return .systemBlue
        }
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        defer { CATransaction.commit() }
        border.frame = bounds
        guard let rect, bounds.width > 0, bounds.height > 0 else {
            border.path = nil
            handles.forEach { $0.isHidden = true }
            return
        }
        let r = CGRect(x: rect.minX * bounds.width, y: rect.minY * bounds.height,
                       width: rect.width * bounds.width, height: rect.height * bounds.height)
        let z = max(zoom, 0.01)
        border.path = UIBezierPath(roundedRect: r, cornerRadius: 4 / z).cgPath
        border.strokeColor = color.cgColor
        border.lineWidth = 3 / z
        let drawingNow = state?.dragging == true && drawStart != nil
        border.lineDashPattern = drawingNow ? [NSNumber(value: Double(6 / z)), NSNumber(value: Double(4 / z))] : nil
        border.fillColor = color.withAlphaComponent(0.12).cgColor
        let adjusting: Bool
        if case .adjust = mode { adjusting = true } else { adjusting = false }
        let corners = [CGPoint(x: r.minX, y: r.minY), CGPoint(x: r.maxX, y: r.minY),
                       CGPoint(x: r.minX, y: r.maxY), CGPoint(x: r.maxX, y: r.maxY)]
        let side = 16 / z
        for (i, h) in handles.enumerated() {
            h.isHidden = !adjusting
            h.frame = bounds
            h.lineWidth = 2 / z
            h.path = UIBezierPath(ovalIn: CGRect(x: corners[i].x - side / 2, y: corners[i].y - side / 2,
                                                 width: side, height: side)).cgPath
        }
    }

    private func unit(_ p: CGPoint) -> CGPoint {
        CGPoint(x: bounds.width > 0 ? p.x / bounds.width : 0, y: bounds.height > 0 ? p.y / bounds.height : 0)
    }

    @objc private func panned(_ g: UIPanGestureRecognizer) {
        guard let mode, bounds.width > 0, bounds.height > 0 else { return }
        let p = unit(g.location(in: self))
        switch mode {
        case .adjust:
            adjustPan(g, at: p)
        case .draw:
            drawPan(g, at: p)
        }
    }

    private func adjustPan(_ g: UIPanGestureRecognizer, at p: CGPoint) {
        switch g.state {
        case .began:
            guard let rect else { return }
            // A corner within about 28 points on the screen takes the touch.
            let reachX = 28 / (bounds.width * max(zoom, 0.01)), reachY = 28 / (bounds.height * max(zoom, 0.01))
            dragHandle = StageEditGeometry.handle(at: p, in: rect, reachX: reachX, reachY: reachY)
            dragStartRect = rect
            state?.dragging = dragHandle != nil
        case .changed:
            guard let handle = dragHandle else { return }
            let t = g.translation(in: self)
            let mins = StageEditGeometry.minSides(for: startRect)
            let next = StageEditGeometry.moved(handle, dragStartRect, dx: t.x / bounds.width, dy: t.y / bounds.height,
                                               minW: mins.w, minH: mins.h)
            rect = next
            setNeedsLayout()
            state?.stageChanged(next)
        default:
            dragHandle = nil
            state?.dragging = false
            updateAccessibility()
        }
    }

    private func drawPan(_ g: UIPanGestureRecognizer, at p: CGPoint) {
        let minW = StageEditGeometry.minDrawPoints / (bounds.width * max(zoom, 0.01))
        let minH = StageEditGeometry.minDrawPoints / (bounds.height * max(zoom, 0.01))
        switch g.state {
        case .began:
            // The finger went down where it was before the pan was recognised.
            let t = g.translation(in: self)
            drawStart = CGPoint(x: p.x - t.x / bounds.width, y: p.y - t.y / bounds.height)
            state?.dragging = true
            rect = nil
            setNeedsLayout()
        case .changed:
            guard let start = drawStart else { return }
            rect = StageEditGeometry.drawn(from: start, to: p, minW: 0, minH: 0)
            setNeedsLayout()
        case .ended:
            guard let start = drawStart else { return }
            drawStart = nil
            let done = StageEditGeometry.drawn(from: start, to: p, minW: minW, minH: minH)
            rect = done ?? state?.rect
            state?.dragging = false
            // Too small: the box drawn before, if any, stays.
            if let done { state?.stageChanged(done) }
            updateAccessibility()
            setNeedsLayout()
        default:
            drawStart = nil
            rect = state?.rect
            state?.dragging = false
            setNeedsLayout()
        }
    }

    // MARK: VoiceOver

    private func updateAccessibility() {
        switch mode {
        case .adjust(let box):
            let text = box.accessibilityText
            let lowered = text.prefix(1).lowercased() + String(text.dropFirst())
            accessibilityLabel = "Moving the box: \(lowered)"
            accessibilityHint = "Use the actions to move it or change its size."
            accessibilityCustomActions = [
                action("Move left") { $0.offsetBy(dx: -0.01, dy: 0) },
                action("Move right") { $0.offsetBy(dx: 0.01, dy: 0) },
                action("Move up") { $0.offsetBy(dx: 0, dy: -0.01) },
                action("Move down") { $0.offsetBy(dx: 0, dy: 0.01) },
                action("Larger") { [weak self] r in
                    let mins = StageEditGeometry.minSides(for: self?.startRect ?? r)
                    return StageEditGeometry.grown(r, by: 0.02, minW: mins.w, minH: mins.h)
                },
                action("Smaller") { [weak self] r in
                    let mins = StageEditGeometry.minSides(for: self?.startRect ?? r)
                    return StageEditGeometry.grown(r, by: -0.02, minW: mins.w, minH: mins.h)
                },
            ]
        case .draw(let kind):
            accessibilityLabel = rect == nil ? "Drag over the \(kind.noun) to draw a box" : "The box you drew over the \(kind.noun)"
            accessibilityHint = nil
            accessibilityCustomActions = nil
        case nil:
            accessibilityLabel = nil
            accessibilityCustomActions = nil
        }
    }

    private func action(_ name: String, _ change: @escaping (CGRect) -> CGRect) -> UIAccessibilityCustomAction {
        UIAccessibilityCustomAction(name: name) { [weak self] _ in
            guard let self, let rect = self.rect else { return false }
            var next = change(rect)
            // A move stays inside the picture.
            next.origin.x = StageEditGeometry.clamp(next.minX, 0, 1 - next.width)
            next.origin.y = StageEditGeometry.clamp(next.minY, 0, 1 - next.height)
            self.rect = next
            self.setNeedsLayout()
            self.state?.stageChanged(next)
            return true
        }
    }
}

/// One box: a border and, for a named person or a guess, a small label above it. It takes no touches itself;
/// the stage hit-tests the boxes smallest first. VoiceOver activates it directly. The border follows the website's
/// PhotoStage: white for a named face, sky for a named person box, green for a pet, amber for a guess, and dashed
/// while nobody is on the box.
final class BoxView: UIView {
    let box: StageBox
    var onActivate: (() -> Void)?

    private let label = PaddedLabel()
    private let outline = CAShapeLayer()
    private var isSelectedBox = false
    private var zoom: CGFloat = 1

    /// Nobody is on the box yet: a face or person box without a person or a guess, or a pet not named.
    private var untagged: Bool {
        box.kind == .animal ? box.personId == nil : (box.name == nil && box.guessName == nil)
    }

    init(box: StageBox) {
        self.box = box
        super.init(frame: .zero)
        isUserInteractionEnabled = false
        backgroundColor = .clear
        outline.fillColor = UIColor.clear.cgColor
        outline.lineJoin = .round
        layer.addSublayer(outline)
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.45
        layer.shadowRadius = 1.5
        layer.shadowOffset = .zero
        label.font = .systemFont(ofSize: 12, weight: .semibold)
        label.textColor = .white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.62)
        label.layer.cornerRadius = 4
        label.clipsToBounds = true
        label.text = box.label
        label.isHidden = box.label == nil
        label.isAccessibilityElement = false
        addSubview(label)
        isAccessibilityElement = true
        accessibilityLabel = box.accessibilityText
        accessibilityTraits = .button
        applyColors()
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    override func accessibilityActivate() -> Bool {
        onActivate?()
        return onActivate != nil
    }

    func setSelected(_ selected: Bool) {
        guard selected != isSelectedBox else { return }
        isSelectedBox = selected
        accessibilityTraits = selected ? [.button, .selected] : .button
        applyColors()
        setZoom(zoom)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        updateOutline()
    }

    /// The border, the same width on screen whatever the zoom.
    private func updateOutline() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        defer { CATransaction.commit() }
        let width = (isSelectedBox ? 3 : 2) / zoom
        outline.frame = bounds
        let radius = (box.kind == .face ? 4 : 6) / zoom
        outline.path = UIBezierPath(roundedRect: bounds.insetBy(dx: width / 2, dy: width / 2), cornerRadius: radius).cgPath
        outline.lineWidth = width
        outline.lineDashPattern = untagged && !isSelectedBox
            ? [NSNumber(value: Double(5 / zoom)), NSNumber(value: Double(3 / zoom))] : nil
    }

    /// Keeps the border and the label the same size on screen whatever the zoom.
    func setZoom(_ z: CGFloat) {
        zoom = max(z, 0.01)
        updateOutline()
        guard !label.isHidden else { return }
        let size = label.intrinsicContentSize
        label.transform = .identity
        label.bounds = CGRect(origin: .zero, size: size)
        let gap = 2 / zoom
        let labelHeight = size.height / zoom
        if box.y < 0.06 || frame.minY < labelHeight + gap {
            // No room above the box: the label goes under it.
            label.layer.anchorPoint = CGPoint(x: 0, y: 0)
            label.layer.position = CGPoint(x: 0, y: bounds.height + gap)
        } else {
            label.layer.anchorPoint = CGPoint(x: 0, y: 1)
            label.layer.position = CGPoint(x: 0, y: -gap)
        }
        label.transform = CGAffineTransform(scaleX: 1 / zoom, y: 1 / zoom)
    }

    private func applyColors() {
        let color: UIColor
        if isSelectedBox {
            color = .systemBlue
        } else if box.kind == .animal {
            color = .systemGreen
        } else if box.name != nil {
            color = box.kind == .body ? .systemCyan : .white
        } else {
            // a guess, and a box nobody is on (dashed)
            color = .systemYellow
        }
        outline.strokeColor = color.cgColor
        updateOutline()
    }
}

/// A label with a little room around its text.
final class PaddedLabel: UILabel {
    var insets = UIEdgeInsets(top: 2, left: 5, bottom: 2, right: 5)

    override func drawText(in rect: CGRect) {
        super.drawText(in: rect.inset(by: insets))
    }

    override var intrinsicContentSize: CGSize {
        let s = super.intrinsicContentSize
        return CGSize(width: s.width + insets.left + insets.right, height: s.height + insets.top + insets.bottom)
    }
}
