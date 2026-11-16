import SwiftUI
import UIKit

/// A photo that pinches and double-taps to zoom, with its boxes drawn over it. A tap goes to the topmost box under
/// the finger, which is the smallest one there (BoxStack), else to `onTap` (which shows or hides the controls).
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

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    func makeUIView(context: Context) -> StageView {
        let view = StageView(frame: .zero)
        view.coordinator = context.coordinator
        return view
    }

    func updateUIView(_ view: StageView, context: Context) {
        context.coordinator.parent = self
        view.apply(image: image, pixelSize: pixelSize, boxes: boxes, showBoxes: showBoxes, selected: selectedBoxId)
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

    override init(frame: CGRect) {
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

        let single = UITapGestureRecognizer(target: self, action: #selector(singleTap(_:)))
        let double = UITapGestureRecognizer(target: self, action: #selector(doubleTap(_:)))
        double.numberOfTapsRequired = 2
        single.require(toFail: double)
        addGestureRecognizer(single)
        addGestureRecognizer(double)
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
            content.addSubview(view)
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
    }

    private func styleBoxes() {
        for (id, view) in boxViews {
            view.isHidden = !showBoxes
            view.isAccessibilityElement = showBoxes
            view.setSelected(id == selected)
            view.setZoom(zoomScale)
        }
        // VoiceOver reads the shown boxes in the item's own order: faces, then person boxes, then pets. Hidden boxes
        // leave the list (a listed element stays readable even when its view is hidden).
        content.accessibilityElements = showBoxes ? boxes.compactMap { boxViews[$0.id] } : []
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
        coordinator?.zoomChanged(zoomScale > 1.01)
    }
}

/// One box: a border and, for a named person or a guess, a small label above it. It takes no touches itself;
/// the stage hit-tests the boxes smallest first. VoiceOver activates it directly.
final class BoxView: UIView {
    let box: StageBox
    var onActivate: (() -> Void)?

    private let label = PaddedLabel()
    private var isSelectedBox = false
    private var zoom: CGFloat = 1

    init(box: StageBox) {
        self.box = box
        super.init(frame: .zero)
        isUserInteractionEnabled = false
        backgroundColor = .clear
        layer.borderWidth = 2
        layer.cornerRadius = 4
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

    /// Keeps the border and the label the same size on screen whatever the zoom.
    func setZoom(_ z: CGFloat) {
        zoom = max(z, 0.01)
        layer.borderWidth = (isSelectedBox ? 3 : 2) / zoom
        layer.cornerRadius = 4 / zoom
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
            color = .systemTeal
        } else if box.name != nil {
            color = .white
        } else if box.guessName != nil {
            color = .systemYellow
        } else {
            color = UIColor.white.withAlphaComponent(0.7)
        }
        layer.borderColor = color.cgColor
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
