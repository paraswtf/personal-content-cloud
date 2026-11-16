import SceneKit
import SwiftUI
import UIKit
import simd

// The heads in SceneKit (the website's Head3dViewer.tsx): a head read from its file becomes a node centred and
// scaled into a box one unit across, with its own colours or, with Colour off, the bare shape in clay; the photo
// layer hangs on it as a node of its own, blended over the head where its weight says, a hair in front of it.

/// How the head is seen (the website's HeadView). `turn`: degrees the head is turned to the viewer's right (0 faces
/// the viewer, 180 shows the back). `tilt`: degrees the viewer looks down from above (negative: up from below).
/// `zoom`: 1 fits the whole head.
struct HeadsView: Equatable {
    var turn: Double = 0
    var tilt: Double = 0
    var zoom: Double = 1

    static let start = HeadsView()
    static let turnRange: ClosedRange<Double> = -180...180
    static let tiltRange: ClosedRange<Double> = -75...75
    static let zoomRange: ClosedRange<Double> = 0.6...2.5

    /// Into -180...180, keeping 180 itself (the back of the head) at the slider's right end.
    static func wrap(_ degrees: Double) -> Double {
        let w = (degrees + 180).truncatingRemainder(dividingBy: 360)
        let r = (w < 0 ? w + 360 : w) - 180
        return r == -180 && degrees > 0 ? 180 : r
    }

    func clamped() -> HeadsView {
        HeadsView(turn: Self.wrap(turn),
                  tilt: min(Self.tiltRange.upperBound, max(Self.tiltRange.lowerBound, tilt)),
                  zoom: min(Self.zoomRange.upperBound, max(Self.zoomRange.lowerBound, zoom)))
    }
}

/// The lamp (the website's HeadLight). `direction`: degrees to the viewer's right of the viewer (0 lights the head
/// from where the viewer stands, 180 from behind it). `height`: degrees above the viewer's eye line. `brightness`:
/// 1 is normal.
struct HeadsLight: Equatable {
    var direction: Double = 30
    var height: Double = 35
    var brightness: Double = 1

    static let start = HeadsLight()
    static let directionRange: ClosedRange<Double> = -180...180
    static let heightRange: ClosedRange<Double> = -90...90
    static let brightnessRange: ClosedRange<Double> = 0.4...1.8
}

/// A head ready for the stage: its container (centred and scaled), the head with its colours and without, and the
/// photo layer over it when the file has one whose picture reads.
final class HeadsPrepared {
    let container: SCNNode
    let head: SCNNode
    let coloured: SCNGeometry
    let bare: SCNGeometry
    let photo: SCNNode?
    let hasColour: Bool

    init(container: SCNNode, head: SCNNode, coloured: SCNGeometry, bare: SCNGeometry, photo: SCNNode?, hasColour: Bool) {
        self.container = container
        self.head = head
        self.coloured = coloured
        self.bare = bare
        self.photo = photo
        self.hasColour = hasColour
    }
}

enum HeadsSceneBuilder {
    /// The bare shape's colour (the website's CLAY).
    static let clay = UIColor(red: 0xCF / 255.0, green: 0xCC / 255.0, blue: 0xC6 / 255.0, alpha: 1)

    static func prepare(_ model: HeadsGLBModel) -> HeadsPrepared {
        let part = model.head
        let normals = part.normals ?? vertexNormals(part.positions, part.indices)
        let points = Self.vertexSource(part.positions)
        let directions = Self.normalSource(normals)
        let element = SCNGeometryElement(indices: part.indices, primitiveType: .triangles)

        let bare = SCNGeometry(sources: [points, directions], elements: [element])
        bare.materials = [material(colour: clay)]
        var coloured = bare
        if let colors = part.colors {
            coloured = SCNGeometry(sources: [points, directions, Self.colorSource(colors)], elements: [element])
            coloured.materials = [material(colour: .white)]
        }
        let head = SCNNode(geometry: part.colors != nil ? coloured : bare)

        // centred and scaled into a box one unit across, whatever the file's own size
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in part.positions {
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
        }
        if part.positions.isEmpty {
            lo = .zero
            hi = .zero
        }
        let size = hi - lo
        let scale = 1 / max(size.x, size.y, size.z, 1e-6)
        head.simdScale = SIMD3<Float>(repeating: scale)
        head.simdPosition = -(lo + hi) / 2 * scale

        var photoNode: SCNNode?
        if let layer = model.photo, let bytes = layer.texture, let picture = UIImage(data: bytes), let uvs = layer.uvs {
            let layerNormals = layer.normals ?? vertexNormals(layer.positions, layer.indices)
            // a hair in front of the head along each corner's normal, so the two surfaces never fight for a pixel
            let lift = 0.0015 / scale
            var lifted: [SIMD3<Float>] = []
            lifted.reserveCapacity(layer.positions.count)
            for (p, n) in zip(layer.positions, layerNormals) { lifted.append(p + n * lift) }
            // the photo's weight at each corner travels as a second set of texture coordinates (u = weight)
            let weights = (layer.colors ?? []).map { SIMD2<Float>(min(1, max(0, $0.w)), 0.5) }
            let ramp = weights.count == layer.positions.count
                ? weights : Array(repeating: SIMD2<Float>(1, 0.5), count: layer.positions.count)
            let geometry = SCNGeometry(
                sources: [vertexSource(lifted), Self.normalSource(layerNormals), texcoordSource(uvs), texcoordSource(ramp)],
                elements: [SCNGeometryElement(indices: layer.indices, primitiveType: .triangles)])
            geometry.materials = [photoMaterial(picture)]
            let node = SCNNode(geometry: geometry)
            node.renderingOrder = 1
            head.addChildNode(node)
            photoNode = node
        }

        let container = SCNNode()
        container.addChildNode(head)
        return HeadsPrepared(container: container, head: head, coloured: coloured, bare: bare, photo: photoNode,
                             hasColour: part.colors != nil)
    }

    static func vertexNormals(_ positions: [SIMD3<Float>], _ indices: [UInt32]) -> [SIMD3<Float>] {
        var normals = [SIMD3<Float>](repeating: .zero, count: positions.count)
        var i = 0
        while i + 2 < indices.count {
            let a = Int(indices[i]), b = Int(indices[i + 1]), c = Int(indices[i + 2])
            let n = simd_cross(positions[b] - positions[a], positions[c] - positions[a])
            normals[a] += n
            normals[b] += n
            normals[c] += n
            i += 3
        }
        return normals.map { n in
            let length = simd_length(n)
            return length > 0 ? n / length : SIMD3<Float>(0, 0, 1)
        }
    }

    static func vertexSource(_ points: [SIMD3<Float>]) -> SCNGeometrySource {
        SCNGeometrySource(vertices: points.map { SCNVector3($0.x, $0.y, $0.z) })
    }

    static func normalSource(_ normals: [SIMD3<Float>]) -> SCNGeometrySource {
        SCNGeometrySource(normals: normals.map { SCNVector3($0.x, $0.y, $0.z) })
    }

    static func texcoordSource(_ uvs: [SIMD2<Float>]) -> SCNGeometrySource {
        SCNGeometrySource(textureCoordinates: uvs.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) })
    }

    static func colorSource(_ colors: [SIMD4<Float>]) -> SCNGeometrySource {
        var flat: [Float] = []
        flat.reserveCapacity(colors.count * 4)
        for c in colors {
            flat.append(c.x)
            flat.append(c.y)
            flat.append(c.z)
            flat.append(c.w)
        }
        let data = flat.withUnsafeBufferPointer { Data(buffer: $0) }
        return SCNGeometrySource(data: data, semantic: .color, vectorCount: colors.count, usesFloatComponents: true,
                                 componentsPerVector: 4, bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
                                 dataStride: MemoryLayout<Float>.size * 4)
    }

    /// A rough surface lit from both sides (the head's material on the website: roughness 0.9, double sided).
    static func material(colour: UIColor) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .lambert
        m.diffuse.contents = colour
        m.isDoubleSided = true
        return m
    }

    /// The photo over the front of the head: its picture, shown as much as each corner's weight says, blended over
    /// the head, single sided, without writing depth.
    static func photoMaterial(_ picture: UIImage) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .lambert
        m.diffuse.contents = picture
        m.diffuse.mappingChannel = 0
        m.diffuse.wrapS = .clamp
        m.diffuse.wrapT = .clamp
        m.diffuse.minificationFilter = .linear
        m.diffuse.magnificationFilter = .linear
        m.diffuse.mipFilter = .linear
        m.diffuse.maxAnisotropy = 16
        m.transparent.contents = weightRamp
        m.transparent.mappingChannel = 1
        m.transparent.wrapS = .clamp
        m.transparent.wrapT = .clamp
        m.transparent.minificationFilter = .linear
        m.transparent.magnificationFilter = .linear
        m.transparent.mipFilter = .none
        m.transparencyMode = .aOne
        m.blendMode = .alpha
        m.writesToDepthBuffer = false
        m.isDoubleSided = false
        return m
    }

    /// 256 pixels from transparent to opaque: a corner's weight, read through the second set of texture
    /// coordinates, becomes how much of the photo shows there.
    static let weightRamp: UIImage = {
        let width = 256
        var pixels = [UInt8](repeating: 0, count: width * 4)
        for x in 0..<width {
            let a = UInt8(x)
            // white with premultiplied alpha: every channel equals the alpha
            pixels[x * 4] = a
            pixels[x * 4 + 1] = a
            pixels[x * 4 + 2] = a
            pixels[x * 4 + 3] = a
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let image = CGImage(width: width, height: 1, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        else { return UIImage() }
        return UIImage(cgImage: image)
    }()
}

/// The SceneKit view, which tells when its size changes (the camera's distance depends on its shape).
final class HeadsSceneView: SCNView {
    var onLayout: (() -> Void)?

    override func layoutSubviews() {
        super.layoutSubviews()
        onLayout?()
    }
}

/// The stage: one SceneKit view with the head on screen, a lamp that follows the viewer and a soft light from
/// above. Drag turns the head, pinch zooms; the view is reported back, so the sliders follow. Going from one look
/// to another fades from a picture of the old one.
struct HeadsStage: UIViewRepresentable {
    let prepared: HeadsPrepared?
    let lookKey: String?
    @Binding var view: HeadsView
    let light: HeadsLight
    let colour: Bool
    let photo: Bool
    let spin: Bool
    /// A head of the model carries its photo: the whole model is lit evenly (the website's PHOTO_LAMP rule).
    let evenLight: Bool

    func makeCoordinator() -> Coordinator { Coordinator(view: $view) }

    func makeUIView(context: Context) -> HeadsSceneView {
        context.coordinator.makeView()
    }

    func updateUIView(_ uiView: HeadsSceneView, context: Context) {
        let c = context.coordinator
        c.binding = $view
        c.update(prepared: prepared, key: lookKey, view: view, light: light, colour: colour, photo: photo,
                 spin: spin, even: evenLight)
    }

    static func dismantleUIView(_ uiView: HeadsSceneView, coordinator: Coordinator) {
        coordinator.stop()
    }

    @MainActor
    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        /// The lamp and the soft light as shares of full light at brightness 1, and the soft light's ground, for a
        /// model of plain colours and for one whose heads carry the photo (the website's LAMP, SOFT, GROUND and
        /// PHOTO_LAMP, PHOTO_SOFT, PHOTO_GROUND, the ground as linear light).
        static let plain = (lamp: 0.62, soft: 0.5, ground: 0.17)
        static let even = (lamp: 0.25, soft: 0.9, ground: 0.83)
        static let fov = 30.0
        static let fitHalf = 0.62
        static let spinDegreesASecond = 14.0
        static let fadeSeconds = 0.3

        var binding: Binding<HeadsView>
        let scene = SCNScene()
        let cameraNode = SCNNode()
        let lampNode = SCNNode()
        let ambientNode = SCNNode()
        let skyNode = SCNNode()
        let fadeView = UIImageView()
        weak var sceneView: HeadsSceneView?
        var shown: HeadsPrepared?
        var shownKey: String?
        var current = HeadsView.start
        var lastReported = HeadsView.start
        var dragging = false
        var panStart = HeadsView.start
        var pinchStart = 1.0
        var spinLink: CADisplayLink?
        var lastTick: CFTimeInterval = 0
        var lastReport: CFTimeInterval = 0

        init(view: Binding<HeadsView>) {
            binding = view
        }

        func makeView() -> HeadsSceneView {
            let v = HeadsSceneView(frame: .zero)
            v.scene = scene
            v.backgroundColor = .secondarySystemBackground
            v.antialiasingMode = .multisampling4X
            v.allowsCameraControl = false
            v.autoenablesDefaultLighting = false
            v.isAccessibilityElement = true
            v.accessibilityLabel = "3D head"
            v.accessibilityHint = "Drag to turn the head and pinch to zoom. The sliders below do the same."
            v.accessibilityIdentifier = "heads.viewer"

            let camera = SCNCamera()
            camera.fieldOfView = CGFloat(Self.fov)
            camera.zNear = 0.05
            camera.zFar = 30
            cameraNode.camera = camera
            scene.rootNode.addChildNode(cameraNode)
            v.pointOfView = cameraNode

            let lamp = SCNLight()
            lamp.type = .directional
            lamp.color = UIColor.white
            lampNode.light = lamp
            cameraNode.addChildNode(lampNode)   // the lamp follows the viewer
            let ambient = SCNLight()
            ambient.type = .ambient
            ambient.color = UIColor.white
            ambientNode.light = ambient
            scene.rootNode.addChildNode(ambientNode)
            let sky = SCNLight()
            sky.type = .directional
            sky.color = UIColor.white
            skyNode.light = sky
            skyNode.simdOrientation = simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))   // straight down
            scene.rootNode.addChildNode(skyNode)

            fadeView.frame = v.bounds
            fadeView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
            fadeView.contentMode = .scaleToFill
            fadeView.alpha = 0
            fadeView.isUserInteractionEnabled = false
            v.addSubview(fadeView)

            let pan = UIPanGestureRecognizer(target: self, action: #selector(onPan(_:)))
            pan.maximumNumberOfTouches = 1
            pan.delegate = self
            v.addGestureRecognizer(pan)
            let pinch = UIPinchGestureRecognizer(target: self, action: #selector(onPinch(_:)))
            pinch.delegate = self
            v.addGestureRecognizer(pinch)

            v.onLayout = { [weak self] in self?.place() }
            sceneView = v
            place()
            return v
        }

        func update(prepared: HeadsPrepared?, key: String?, view: HeadsView, light: HeadsLight, colour: Bool,
                    photo: Bool, spin: Bool, even: Bool) {
            // a slider moved (a view the stage did not report itself): take it
            if view != lastReported {
                current = view.clamped()
                lastReported = view
                place()
            }
            show(prepared, key: key)
            if let shown {
                shown.head.geometry = colour && shown.hasColour ? shown.coloured : shown.bare
                shown.photo?.isHidden = !(colour && photo)
            }
            relight(light, even: even && colour)
            setSpin(spin)
        }

        func stop() {
            spinLink?.invalidate()
            spinLink = nil
        }

        /// Puts the camera where the view says.
        func place() {
            guard let v = sceneView else { return }
            let aspect = v.bounds.height > 0 ? Double(v.bounds.width / v.bounds.height) : 1
            let base = Self.fitHalf / tan(Self.fov / 2 * .pi / 180) / min(1, aspect)
            let far = base / current.zoom
            let around = -current.turn * .pi / 180   // the camera goes left when the head is to turn right
            let up = current.tilt * .pi / 180
            cameraNode.simdPosition = SIMD3<Float>(Float(far * sin(around) * cos(up)), Float(far * sin(up)),
                                                   Float(far * cos(around) * cos(up)))
            cameraNode.simdLook(at: SIMD3<Float>(0, 0, 0))
        }

        func relight(_ light: HeadsLight, even: Bool) {
            let a = Float(light.direction * .pi / 180)
            let h = Float(light.height * .pi / 180)
            // the lamp shines from (sin a cos h, sin h, cos a cos h) of the camera towards it
            lampNode.simdOrientation = simd_quatf(angle: a, axis: SIMD3<Float>(0, 1, 0))
                * simd_quatf(angle: -h, axis: SIMD3<Float>(1, 0, 0))
            let rule = even ? Self.even : Self.plain
            let b = light.brightness
            lampNode.light?.intensity = CGFloat(1000 * rule.lamp * b)
            // the website's hemisphere light: (1 + ground) / 2 everywhere, and the rest of the sky from above
            ambientNode.light?.intensity = CGFloat(1000 * rule.soft * (1 + rule.ground) / 2 * b)
            skyNode.light?.intensity = CGFloat(1000 * rule.soft * (1 - rule.ground) / 2 * b)
        }

        /// Shows a head once it is ready; until then the one on screen stays (under the loading words).
        func show(_ prepared: HeadsPrepared?, key: String?) {
            guard let prepared, prepared !== shown else { return }
            if let v = sceneView, shown != nil, v.bounds.width > 0, !UIAccessibility.isReduceMotionEnabled {
                fadeView.layer.removeAllAnimations()
                fadeView.image = v.snapshot()
                fadeView.alpha = 1
                UIView.animate(withDuration: Self.fadeSeconds, delay: 0.05, options: [.curveEaseInOut]) {
                    self.fadeView.alpha = 0
                }
            }
            shown?.container.removeFromParentNode()
            shown = prepared
            shownKey = key
            scene.rootNode.addChildNode(prepared.container)
        }

        func setSpin(_ on: Bool) {
            if on, spinLink == nil {
                lastTick = CACurrentMediaTime()
                let link = CADisplayLink(target: self, selector: #selector(tick(_:)))
                link.add(to: .main, forMode: .common)
                spinLink = link
            } else if !on, let link = spinLink {
                link.invalidate()
                spinLink = nil
            }
        }

        @objc func tick(_ link: CADisplayLink) {
            let now = link.timestamp
            let dt = min(0.1, max(0, now - lastTick))
            lastTick = now
            guard !dragging else { return }
            current.turn = HeadsView.wrap(current.turn + Self.spinDegreesASecond * dt)
            place()
            if now - lastReport > 0.06 {
                lastReport = now
                report()
            }
        }

        func report() {
            lastReported = current
            binding.wrappedValue = current
        }

        @objc func onPan(_ g: UIPanGestureRecognizer) {
            guard let v = sceneView else { return }
            switch g.state {
            case .began:
                dragging = true
                panStart = current
            case .changed:
                let t = g.translation(in: v)
                let height = Double(max(v.bounds.height, 1))
                var next = panStart
                next.turn = panStart.turn + Double(t.x) / height * 360 * 0.9
                next.tilt = panStart.tilt + Double(t.y) / height * 360 * 0.9
                current = next.clamped()
                place()
                report()
            default:
                dragging = false
            }
        }

        @objc func onPinch(_ g: UIPinchGestureRecognizer) {
            switch g.state {
            case .began:
                pinchStart = current.zoom
            case .changed:
                var next = current
                next.zoom = pinchStart * Double(g.scale)
                current = next.clamped()
                place()
                report()
            default:
                break
            }
        }

        // A drag that starts on the head turns the head: a scroll view around the stage waits for it.
        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                               shouldBeRequiredToFailBy otherGestureRecognizer: UIGestureRecognizer) -> Bool {
            otherGestureRecognizer.view is UIScrollView && otherGestureRecognizer is UIPanGestureRecognizer
        }

        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer,
                               shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
            otherGestureRecognizer.view === gestureRecognizer.view
        }
    }
}
