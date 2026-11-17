import XCTest
import UIKit
@testable import PersonalContentCloud

/// The head file reader (Features/Heads/HeadsGLB.swift) on small binary glTF files built here in the layout of the
/// server's writer (gen3d/glb.py): a head triangle with colours and, over it, the photo layer with its texture.
final class HeadsGLBTests: XCTestCase {
    // MARK: - building files

    struct Blob {
        var bytes = Data()
        var views: [[String: Any]] = []
        var accessors: [[String: Any]] = []

        mutating func view(_ data: Data, target: Int? = nil) -> Int {
            while bytes.count % 4 != 0 { bytes.append(0) }
            var entry: [String: Any] = ["buffer": 0, "byteOffset": bytes.count, "byteLength": data.count]
            if let target { entry["target"] = target }
            bytes.append(data)
            views.append(entry)
            return views.count - 1
        }

        mutating func add(_ data: Data, target: Int, _ accessor: [String: Any]) -> Int {
            var a = accessor
            a["bufferView"] = view(data, target: target)
            accessors.append(a)
            return accessors.count - 1
        }
    }

    static func floats(_ values: [Float]) -> Data {
        var data = Data()
        for v in values { withUnsafeBytes(of: v.bitPattern.littleEndian) { data.append(contentsOf: $0) } }
        return data
    }

    static func u16s(_ values: [UInt16]) -> Data {
        var data = Data()
        for v in values { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        return data
    }

    static func u32s(_ values: [UInt32]) -> Data {
        var data = Data()
        for v in values { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        return data
    }

    static func le32(_ v: UInt32) -> Data { u32s([v]) }

    /// A small PNG of three colours (red and green above, blue below).
    static func picture() -> Data {
        let width = 8, height = 8
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                let rgb: (UInt8, UInt8, UInt8) = y < 4 ? (x < 4 ? (255, 0, 0) : (0, 255, 0)) : (0, 0, 255)
                pixels[i] = rgb.0
                pixels[i + 1] = rgb.1
                pixels[i + 2] = rgb.2
            }
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { return Data() }
        return UIImage(cgImage: image).pngData() ?? Data()
    }

    /// The description and the data of a head file: a triangle with normals, colours and indices, and unless
    /// `photo` is false a second part with texture coordinates, weights and a picture inside the file.
    static func parts(photo: Bool = true, indices: [UInt32] = [0, 1, 2]) -> ([String: Any], Data) {
        var blob = Blob()
        let positions: [Float] = [-0.5, -0.5, 0, 0.5, -0.5, 0, 0, 0.6, 0]
        let normals: [Float] = [0, 0, 1, 0, 0, 1, 0, 0, 1]
        let position = blob.add(floats(positions), target: 34962, [
            "componentType": 5126, "count": 3, "type": "VEC3", "min": [-0.5, -0.5, 0], "max": [0.5, 0.6, 0]])
        let normal = blob.add(floats(normals), target: 34962, ["componentType": 5126, "count": 3, "type": "VEC3"])
        let colour = blob.add(u16s([65535, 0, 0, 65535, 0, 65535, 0, 65535, 0, 0, 32768, 65535]), target: 34962, [
            "componentType": 5123, "normalized": true, "count": 3, "type": "VEC4"])
        let corners = blob.add(u32s(indices), target: 34963, ["componentType": 5125, "count": indices.count, "type": "SCALAR"])
        var primitives: [[String: Any]] = [[
            "attributes": ["POSITION": position, "NORMAL": normal, "COLOR_0": colour],
            "indices": corners, "material": 0, "mode": 4,
        ]]
        var doc: [String: Any] = [
            "asset": ["version": "2.0", "generator": "pcc-gen3d"],
            "scene": 0,
            "scenes": [["nodes": [0]]],
            "nodes": [["mesh": 0, "name": "head"]],
            "materials": [[
                "name": "skin", "doubleSided": true,
                "pbrMetallicRoughness": ["baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 0.9],
            ]],
        ]
        if photo {
            let p2 = blob.add(floats(positions), target: 34962, ["componentType": 5126, "count": 3, "type": "VEC3"])
            let n2 = blob.add(floats(normals), target: 34962, ["componentType": 5126, "count": 3, "type": "VEC3"])
            let uv = blob.add(floats([0, 1, 1, 1, 0.5, 0]), target: 34962, ["componentType": 5126, "count": 3, "type": "VEC2"])
            let weight = blob.add(u16s([65535, 65535, 65535, 65535, 65535, 65535, 65535, 0, 65535, 65535, 65535, 32768]),
                                  target: 34962, ["componentType": 5123, "normalized": true, "count": 3, "type": "VEC4"])
            let i2 = blob.add(u32s([0, 1, 2]), target: 34963, ["componentType": 5125, "count": 3, "type": "SCALAR"])
            let image = blob.view(picture())
            primitives.append([
                "attributes": ["POSITION": p2, "NORMAL": n2, "TEXCOORD_0": uv, "COLOR_0": weight],
                "indices": i2, "material": 1, "mode": 4,
            ])
            var materials = doc["materials"] as? [[String: Any]] ?? []
            materials.append([
                "name": "photo", "doubleSided": false, "alphaMode": "BLEND",
                "pbrMetallicRoughness": ["baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 0.9,
                                         "baseColorTexture": ["index": 0]],
            ])
            doc["materials"] = materials
            doc["textures"] = [["source": 0, "sampler": 0]]
            doc["samplers"] = [["magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071]]
            doc["images"] = [["bufferView": image, "mimeType": "image/png"]]
        }
        while blob.bytes.count % 4 != 0 { blob.bytes.append(0) }
        doc["meshes"] = [["name": "head", "primitives": primitives]]
        doc["buffers"] = [["byteLength": blob.bytes.count]]
        doc["bufferViews"] = blob.views
        doc["accessors"] = blob.accessors
        return (doc, blob.bytes)
    }

    /// The whole file: header, the description padded with spaces, the data, and any chunks after it.
    static func file(_ doc: [String: Any], _ bin: Data, magic: UInt32 = 0x4654_6C67, after: [(UInt32, Data)] = []) -> Data {
        var json = (try? JSONSerialization.data(withJSONObject: doc)) ?? Data()
        while json.count % 4 != 0 { json.append(0x20) }
        var chunks = le32(UInt32(json.count)) + le32(0x4E4F_534A) + json
        chunks += le32(UInt32(bin.count)) + le32(0x004E_4942) + bin
        for (type, data) in after { chunks += le32(UInt32(data.count)) + le32(type) + data }
        return le32(magic) + le32(2) + le32(UInt32(12 + chunks.count)) + chunks
    }

    static func glb(photo: Bool = true, indices: [UInt32] = [0, 1, 2], patch: ((inout [String: Any]) -> Void)? = nil) -> Data {
        var (doc, bin) = parts(photo: photo, indices: indices)
        patch?(&doc)
        return file(doc, bin)
    }

    func assertRefused(_ data: Data, saying words: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try HeadsGLB.read(data), file: file, line: line) { error in
            let message = (error as? HeadsGLBError)?.message ?? "\(error)"
            XCTAssertTrue(message.contains(words), "said: \(message)", file: file, line: line)
        }
    }

    // MARK: - what reads

    func testATexturedTriangleLoadsWithTheRightCounts() throws {
        let model = try HeadsGLB.read(Self.glb())
        XCTAssertEqual(model.parts.count, 2)
        let head = model.head
        XCTAssertEqual(head.vertexCount, 3)
        XCTAssertEqual(head.indices, [0, 1, 2])
        XCTAssertEqual(head.triangleCount, 1)
        XCTAssertEqual(head.normals?.count, 3)
        XCTAssertNil(head.texture)
        XCTAssertTrue(head.doubleSided)
        XCTAssertFalse(head.blended)
        let colors = try XCTUnwrap(head.colors)
        XCTAssertEqual(colors.count, 3)
        XCTAssertEqual(colors[0].x, 1, accuracy: 1e-6)
        XCTAssertEqual(colors[2].z, Float(32768) / 65535, accuracy: 1e-6)

        let photo = try XCTUnwrap(model.photo)
        XCTAssertEqual(photo.vertexCount, 3)
        XCTAssertEqual(photo.uvs?.count, 3)
        XCTAssertEqual(photo.uvs?[2], SIMD2<Float>(0.5, 0))
        XCTAssertTrue(photo.blended)
        XCTAssertFalse(photo.doubleSided)
        XCTAssertEqual(photo.colors?[1].w ?? -1, 0, accuracy: 1e-6)            // the weight rides in the alpha
        let picture = try XCTUnwrap(photo.texture)
        XCTAssertNotNil(UIImage(data: picture), "the picture inside the file does not decode")
    }

    func testAHeadWithoutAPhotoReads() throws {
        let model = try HeadsGLB.read(Self.glb(photo: false))
        XCTAssertEqual(model.parts.count, 1)
        XCTAssertNil(model.photo)
        XCTAssertEqual(model.head.vertexCount, 3)
    }

    func testTheSceneHasTheHeadItsBareShapeAndThePhoto() throws {
        let prepared = HeadsSceneBuilder.prepare(try HeadsGLB.read(Self.glb()))
        XCTAssertTrue(prepared.hasColour)
        XCTAssertFalse(prepared.coloured === prepared.bare)
        XCTAssertTrue(prepared.head.geometry === prepared.coloured)
        XCTAssertNotNil(prepared.photo)
        XCTAssertEqual(prepared.coloured.sources(for: .color).count, 1)
        XCTAssertEqual(prepared.bare.sources(for: .color).count, 0)
        let layer = try XCTUnwrap(prepared.photo?.geometry)
        XCTAssertEqual(layer.sources(for: .texcoord).count, 2, "the picture's places and the weights")
        XCTAssertEqual(prepared.photo?.geometry?.firstMaterial?.writesToDepthBuffer, false)
        // scaled into a box one unit across: the triangle is 1.1 tall
        XCTAssertEqual(prepared.head.simdScale.x, Float(1 / 1.1), accuracy: 1e-5)

        let plain = HeadsSceneBuilder.prepare(try HeadsGLB.read(Self.glb(photo: false)))
        XCTAssertNil(plain.photo)
    }

    // MARK: - what is refused

    func testAnExternalBufferIsRefused() {
        let data = Self.glb { doc in doc["buffers"] = [["byteLength": 400, "uri": "https://example.invalid/head.bin"]] }
        assertRefused(data, saying: "address")
    }

    func testAnExternalPictureIsRefused() {
        let data = Self.glb { doc in doc["images"] = [["uri": "face.png", "mimeType": "image/png"]] }
        assertRefused(data, saying: "address")
    }

    func testADataAddressIsRefusedToo() {
        let data = Self.glb { doc in doc["images"] = [["uri": "data:image/png;base64,iVBORw0KGgo=", "mimeType": "image/png"]] }
        assertRefused(data, saying: "address")
    }

    func testAnAddressAnywhereInTheDescriptionIsRefused() {
        let data = Self.glb { doc in doc["extras"] = ["notes": [["uri": "https://example.invalid/"]]] }
        assertRefused(data, saying: "address")
    }

    func testATruncatedFileIsRefused() {
        let data = Self.glb()
        assertRefused(data.prefix(data.count - 16), saying: "not whole")
        assertRefused(data.prefix(10), saying: "too short")
    }

    func testAWrongMagicNumberIsRefused() {
        let (doc, bin) = Self.parts()
        assertRefused(Self.file(doc, bin, magic: 0x1234_5678), saying: "not a head file")
    }

    func testASecondDescriptionAfterTheDataIsRefused() {
        let (doc, bin) = Self.parts()
        let json = (try? JSONSerialization.data(withJSONObject: ["asset": ["version": "2.0"]])) ?? Data()
        var padded = json
        while padded.count % 4 != 0 { padded.append(0x20) }
        assertRefused(Self.file(doc, bin, after: [(0x4E4F_534A, padded)]), saying: "does not read")
    }

    func testACornerPastThePointsIsRefused() {
        assertRefused(Self.glb(indices: [0, 1, 7]), saying: "corner that is not there")
    }

    func testDataRunningPastItsPlaceIsRefused() {
        let data = Self.glb { doc in
            var accessors = doc["accessors"] as? [[String: Any]] ?? []
            accessors[0]["count"] = 300
            doc["accessors"] = accessors
        }
        assertRefused(data, saying: "runs past")
    }

    func testAnUnknownKindOfNumberIsRefused() {
        let data = Self.glb { doc in
            var accessors = doc["accessors"] as? [[String: Any]] ?? []
            accessors[0]["componentType"] = 5130
            doc["accessors"] = accessors
        }
        assertRefused(data, saying: "kind of number")
    }

    func testSparseDataIsRefused() {
        let data = Self.glb { doc in
            var accessors = doc["accessors"] as? [[String: Any]] ?? []
            accessors[0]["sparse"] = ["count": 1]
            doc["accessors"] = accessors
        }
        assertRefused(data, saying: "sparse")
    }

    // MARK: - where a file is

    func testTheFileAddressIsReadNotMade() throws {
        let file = try HeadsFiles.parts(of: "/files/model3d/5/20260415T120000/heads-young.glb", personId: 5)
        XCTAssertEqual(file.rev, "20260415T120000")
        XCTAssertEqual(file.name, "heads-young.glb")
        XCTAssertThrowsError(try HeadsFiles.parts(of: "/files/model3d/6/20260415T120000/heads-young.glb", personId: 5))
        XCTAssertThrowsError(try HeadsFiles.parts(of: "https://example.invalid/files/model3d/5/r/a.glb", personId: 5))
        XCTAssertThrowsError(try HeadsFiles.parts(of: "/files/model3d/5/a.glb", personId: 5))
    }
}

/// The look sliders (Features/Heads/HeadsModel.swift), as the website's lib/model3d.ts has them.
final class HeadsLooksTests: XCTestCase {
    static func look(_ key: String, age: Double?, band: String, hair: String? = "short hair", facial: String?,
                     facialLevel: Double? = nil, photos: Int, textured: Bool = true) throws -> API.Look3dOut {
        var row: [String: Any] = [
            "key": key, "label": key, "age_band": band, "photos": photos, "item_uid": "u\(key)",
            "picture_url": "/files/model3d/1/r/\(key).jpg", "mesh_url": "/files/model3d/1/r/\(key).glb",
            "bytes": 100, "vertices": 3, "textured": textured, "relief": false,
        ]
        if let age { row["age"] = age }
        if let hair {
            row["hair_label"] = hair
            row["hair"] = 0.45
        }
        if let facial { row["facial_hair_label"] = facial }
        if let facialLevel { row["facial_hair"] = facialLevel }
        let data = try JSONSerialization.data(withJSONObject: row)
        return try JSONDecoder().decode(API.Look3dOut.self, from: data)
    }

    func testTheAxesFollowTheLooks() throws {
        let looks = [
            try Self.look("young", age: 29, band: "25 to 34", facial: "little or no facial hair", facialLevel: 0.3, photos: 2),
            try Self.look("older", age: 41, band: "35 to 44", facial: "a beard", facialLevel: 0.7, photos: 3),
        ]
        let axes = HeadsLooks.axes(looks)
        XCTAssertEqual(axes.map(\.label), ["Age", "Hair", "Facial hair"])
        XCTAssertEqual(axes[0].stops.map(\.text), ["about 29", "about 41"])
        XCTAssertEqual(axes[1].stops.map(\.text), ["short hair"], "one hair length: the slider has one stop")
        XCTAssertEqual(axes[2].stops.map(\.text), ["little or no facial hair", "a beard"])
        XCTAssertEqual(HeadsLooks.stop(of: "older", on: axes[0]), 1)
        XCTAssertEqual(HeadsLooks.text(axes[0], for: looks[0]), "about 29")
        XCTAssertEqual(HeadsLooks.pick(axes, from: "older", moved: .age, to: 0), "young")
        XCTAssertEqual(HeadsLooks.pick(axes, from: "older", moved: .age, to: 1), "older")
        XCTAssertEqual(HeadsLooks.first(looks)?.key, "older", "the latest age band first")
        XCTAssertEqual(HeadsLooks.lines(looks[1]).0, "About 41")
        XCTAssertEqual(HeadsLooks.lines(looks[1]).1, "short hair, a beard")
    }

    func testAnAxisNothingWasReadForHasNoSlider() throws {
        let looks = [try Self.look("a", age: nil, band: "18 to 24", hair: nil, facial: nil, photos: 1)]
        let axes = HeadsLooks.axes(looks)
        XCTAssertEqual(axes[1].stops.count, 0)
        XCTAssertEqual(axes[2].stops.count, 0)
        XCTAssertEqual(HeadsLooks.text(axes[0], for: looks[0]), "18 to 24")
    }

    func testTheFirstLookIsTheOneMostPhotosShowAtTheLatestAge() throws {
        let looks = [
            try Self.look("a", age: 30, band: "25 to 34", facial: "a beard", photos: 9),
            try Self.look("b", age: 41, band: "35 to 44", facial: "stubble or a short beard", photos: 2),
            try Self.look("c", age: 43, band: "35 to 44", facial: "a beard", photos: 5),
        ]
        XCTAssertEqual(HeadsLooks.first(looks)?.key, "c")
    }

    func testTheWords() {
        XCTAssertEqual(HeadsWords.degrees(0, zero: "facing you", half: "from behind", positive: "to the right",
                                          negative: "to the left"), "facing you")
        XCTAssertEqual(HeadsWords.degrees(-35, zero: "level", half: "level", positive: "from above",
                                          negative: "from below"), "35° from below")
        XCTAssertEqual(HeadsWords.degrees(180, zero: "x", half: "from behind", positive: "r", negative: "l"), "from behind")
        XCTAssertEqual(HeadsWords.percent(1.25), "125%")
        XCTAssertEqual(HeadsWords.sentence("the photo did not line up"), "The photo did not line up.")
        XCTAssertNil(HeadsWords.sentence("  "))
        XCTAssertEqual(HeadsView.wrap(190), -170)
        XCTAssertEqual(HeadsView.wrap(180), 180)
        XCTAssertEqual(HeadsView(turn: 0, tilt: 99, zoom: 9).clamped(), HeadsView(turn: 0, tilt: 75, zoom: 2.5))
    }

    func testOptionsComeBackInsideTheirRanges() throws {
        let made = try JSONDecoder().decode(API.Model3dOptions.self, from: Data(#"{"max_looks": 20, "detail": 3, "framing": 2.26}"#.utf8))
        let options = HeadsOptions.complete(made)
        XCTAssertEqual(options.maxLooks, 12)
        XCTAssertEqual(options.detail, 3)
        XCTAssertEqual(options.framing, 2.3, accuracy: 1e-9)
        XCTAssertEqual(options.smoothing, 2, "a model from before the option existed starts at the default")
        XCTAssertEqual(options.texture, 2)
        XCTAssertEqual(options.relief, 1)
    }
}
