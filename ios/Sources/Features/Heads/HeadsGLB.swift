import Foundation

// Reading a head file: binary glTF 2.0 as the server's generator writes it (gen3d/glb.py, docs/ARCHITECTURE.md
// section 36). SceneKit cannot read glTF, so this does, and only what is inside the file: a buffer or a picture
// that names an address (a `uri`, anywhere in the description, also a `data:` one) is refused, and so is anything
// it does not understand (a chunk after the data, a second description, a sparse accessor, a kind of number or
// surface it does not know, a corner past the points), each with a plain error. It never fetches anything: the
// bytes come from the server through the signed-in client (HeadsStore).

/// What went wrong reading a head file, in plain words.
struct HeadsGLBError: LocalizedError, Equatable {
    let message: String
    var errorDescription: String? { message }
}

/// One part of a head file (a glTF primitive): the surface, and for the photo layer its picture.
struct HeadsMeshPart: Sendable {
    var positions: [SIMD3<Float>]
    var normals: [SIMD3<Float>]?
    /// Linear RGBA between 0 and 1, as glTF keeps vertex colours.
    var colors: [SIMD4<Float>]?
    /// Texture coordinates, (0, 0) at the picture's top left corner.
    var uvs: [SIMD2<Float>]?
    var indices: [UInt32]
    /// The base colour texture's own bytes (a JPEG or a PNG) when the part's material has one.
    var texture: Data?
    /// The material blends (alphaMode BLEND), as the photo layer's does.
    var blended: Bool
    var doubleSided: Bool

    var vertexCount: Int { positions.count }
    var triangleCount: Int { indices.count / 3 }
}

/// A head file, read: every part, the head (the first part without a picture) and its photo (the first other part
/// with a picture and the coordinates to place it), picked as the website's viewer picks them.
struct HeadsGLBModel: Sendable {
    var parts: [HeadsMeshPart]

    var headIndex: Int { parts.firstIndex(where: { $0.texture == nil }) ?? 0 }
    var head: HeadsMeshPart { parts[headIndex] }
    var photo: HeadsMeshPart? {
        let head = headIndex
        guard let i = parts.indices.first(where: { $0 != head && parts[$0].texture != nil && parts[$0].uvs != nil })
        else { return nil }
        return parts[i]
    }
}

enum HeadsGLB {
    static let magic: UInt32 = 0x4654_6C67       // "glTF"
    static let jsonChunk: UInt32 = 0x4E4F_534A   // "JSON"
    static let binChunk: UInt32 = 0x004E_4942    // "BIN\0"
    /// A head with its photo at full size is a few tens of megabytes; a larger file is refused.
    static let maxBytes = 256 * 1024 * 1024
    /// Points or corners of one part above this are refused.
    static let maxElements = 16_000_000

    static func read(_ data: Data) throws -> HeadsGLBModel {
        var reader = HeadsGLBReader(bytes: [UInt8](data))
        return try reader.read()
    }
}

private struct HeadsGLBReader {
    let bytes: [UInt8]
    var doc: [String: Any] = [:]
    var bin: Range<Int> = 0..<0
    var views: [View] = []

    struct View {
        let start: Int
        let length: Int
        let stride: Int?
    }

    struct Accessor {
        let view: View
        let offset: Int
        let componentType: Int
        let normalized: Bool
        let count: Int
        let width: Int
        let size: Int
        let step: Int
    }

    static let float = 5126, uint = 5125, ushort = 5123, short = 5122, ubyte = 5121, byte = 5120
    static let componentSize: [Int: Int] = [5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4]
    static let typeWidth: [String: Int] = ["SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4]

    static func fail(_ text: String) -> HeadsGLBError { HeadsGLBError(message: text) }

    func u32(_ at: Int) -> UInt32 {
        UInt32(bytes[at]) | UInt32(bytes[at + 1]) << 8 | UInt32(bytes[at + 2]) << 16 | UInt32(bytes[at + 3]) << 24
    }

    func u16(_ at: Int) -> UInt16 {
        UInt16(bytes[at]) | UInt16(bytes[at + 1]) << 8
    }

    /// A whole number from the description (JSON numbers arrive as NSNumber; a boolean or a fraction is none).
    static func int(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        let d = number.doubleValue
        guard d.isFinite, d == d.rounded(), abs(d) <= 9_007_199_254_740_992 else { return nil }
        return Int(d)
    }

    /// True when anything in the description names an address.
    static func namesAnAddress(_ value: Any) -> Bool {
        if let dict = value as? [String: Any] {
            if dict["uri"] != nil { return true }
            return dict.values.contains(where: { namesAnAddress($0) })
        }
        if let list = value as? [Any] {
            return list.contains(where: { namesAnAddress($0) })
        }
        return false
    }

    mutating func read() throws -> HeadsGLBModel {
        guard bytes.count >= 12 else { throw Self.fail("The head file is too short to be one.") }
        guard bytes.count <= HeadsGLB.maxBytes else { throw Self.fail("The head file is too large.") }
        guard u32(0) == HeadsGLB.magic else {
            throw Self.fail("This is not a head file: it does not start as binary glTF does.")
        }
        guard u32(4) == 2 else { throw Self.fail("The head file is not glTF 2.0.") }
        guard Int(u32(8)) == bytes.count else {
            throw Self.fail("The head file is not whole: its length is not the one its header gives.")
        }
        var offset = 12
        var chunk = 0
        var described = false
        while offset < bytes.count {
            guard bytes.count - offset >= 8 else { throw Self.fail("The head file ends in the middle of a chunk.") }
            let length = Int(u32(offset))
            let type = u32(offset + 4)
            let start = offset + 8
            guard length <= bytes.count - start else { throw Self.fail("A chunk of the head file runs past its end.") }
            if chunk == 0 && type == HeadsGLB.jsonChunk {
                let text = Data(bytes[start..<(start + length)])
                guard let object = try? JSONSerialization.jsonObject(with: text), let dict = object as? [String: Any] else {
                    throw Self.fail("The head file's description does not read.")
                }
                doc = dict
                described = true
            } else if chunk == 1 && type == HeadsGLB.binChunk {
                bin = start..<(start + length)
            } else {
                throw Self.fail("The head file has a part this app does not read.")
            }
            offset = start + length
            chunk += 1
        }
        guard described else { throw Self.fail("The head file has no description.") }
        if Self.namesAnAddress(doc) {
            throw Self.fail("The head file names an address outside itself. The app reads only what is inside the file.")
        }
        if let required = doc["extensionsRequired"] as? [Any], !required.isEmpty {
            throw Self.fail("The head file needs extensions this app does not read.")
        }
        if let asset = doc["asset"] as? [String: Any], let version = asset["version"] as? String, !version.hasPrefix("2.") {
            throw Self.fail("The head file is not glTF 2.0.")
        }
        try readBuffer()
        try readViews()
        let parts = try readParts()
        guard !parts.isEmpty else { throw Self.fail("The head file holds no surface.") }
        return HeadsGLBModel(parts: parts)
    }

    /// The one buffer, which is the data chunk.
    mutating func readBuffer() throws {
        let buffers = doc["buffers"] as? [Any] ?? []
        guard buffers.count <= 1 else { throw Self.fail("The head file has more than one buffer.") }
        guard let first = buffers.first else {
            bin = bin.lowerBound..<bin.lowerBound
            return
        }
        guard let buffer = first as? [String: Any], let length = Self.int(buffer["byteLength"]), length >= 0,
              length <= bin.count else {
            throw Self.fail("The head file's buffer does not match its data.")
        }
        bin = bin.lowerBound..<(bin.lowerBound + length)
    }

    mutating func readViews() throws {
        let list = doc["bufferViews"] as? [Any] ?? []
        var out: [View] = []
        for item in list {
            guard let v = item as? [String: Any], Self.int(v["buffer"]) == 0,
                  let offset = Self.int(v["byteOffset"] ?? 0), offset >= 0,
                  let length = Self.int(v["byteLength"]), length >= 0, offset <= bin.count, length <= bin.count - offset
            else {
                throw Self.fail("A part of the head file's data is not where its description says.")
            }
            var stride: Int?
            if v["byteStride"] != nil {
                guard let s = Self.int(v["byteStride"]), s >= 4, s <= 252, s % 4 == 0 else {
                    throw Self.fail("The head file lays out its data in a way this app does not read.")
                }
                stride = s
            }
            out.append(View(start: bin.lowerBound + offset, length: length, stride: stride))
        }
        views = out
    }

    func accessor(_ index: Any?) throws -> Accessor {
        guard let i = Self.int(index), let list = doc["accessors"] as? [Any], list.indices.contains(i),
              let a = list[i] as? [String: Any] else {
            throw Self.fail("The head file names a part of its data that is not there.")
        }
        if a["sparse"] != nil { throw Self.fail("The head file uses sparse data, which this app does not read.") }
        guard let viewIndex = Self.int(a["bufferView"]), views.indices.contains(viewIndex) else {
            throw Self.fail("The head file names data without a place in the file.")
        }
        guard let type = a["type"] as? String, let width = Self.typeWidth[type] else {
            throw Self.fail("The head file uses a kind of value this app does not read.")
        }
        guard let componentType = Self.int(a["componentType"]), let size = Self.componentSize[componentType] else {
            throw Self.fail("The head file uses a kind of number this app does not read.")
        }
        guard let count = Self.int(a["count"]), count >= 0, count <= HeadsGLB.maxElements else {
            throw Self.fail("The head file holds more points than this app reads.")
        }
        guard let offset = Self.int(a["byteOffset"] ?? 0), offset >= 0 else {
            throw Self.fail("A part of the head file's data is not where its description says.")
        }
        let view = views[viewIndex]
        let element = size * width
        let step = view.stride ?? element
        guard step >= element else { throw Self.fail("The head file lays out its data in a way this app does not read.") }
        if count > 0 {
            let end = offset + step * (count - 1) + element
            guard end <= view.length else { throw Self.fail("A part of the head file's data runs past its place.") }
        }
        let normalized = (a["normalized"] as? Bool) ?? false
        return Accessor(view: view, offset: offset, componentType: componentType, normalized: normalized, count: count,
                        width: width, size: size, step: step)
    }

    /// The accessor's values as floats, normalised integers scaled into 0...1 (or -1...1).
    func floats(_ acc: Accessor) throws -> [Float] {
        var out = [Float](repeating: 0, count: acc.count * acc.width)
        var k = 0
        for i in 0..<acc.count {
            let base = acc.view.start + acc.offset + i * acc.step
            for c in 0..<acc.width {
                let p = base + c * acc.size
                let value: Float
                switch acc.componentType {
                case Self.float:
                    value = Float(bitPattern: u32(p))
                case Self.ubyte:
                    value = acc.normalized ? Float(bytes[p]) / 255 : Float(bytes[p])
                case Self.ushort:
                    value = acc.normalized ? Float(u16(p)) / 65535 : Float(u16(p))
                case Self.byte:
                    let s = Float(Int8(bitPattern: bytes[p]))
                    value = acc.normalized ? max(s / 127, -1) : s
                case Self.short:
                    let s = Float(Int16(bitPattern: u16(p)))
                    value = acc.normalized ? max(s / 32767, -1) : s
                default:
                    throw Self.fail("The head file uses a kind of number this app does not read.")
                }
                out[k] = value
                k += 1
            }
        }
        return out
    }

    func indices(_ acc: Accessor, vertexCount: Int) throws -> [UInt32] {
        guard acc.width == 1 else { throw Self.fail("The head file's corners are not single numbers.") }
        var out = [UInt32](repeating: 0, count: acc.count)
        for i in 0..<acc.count {
            let p = acc.view.start + acc.offset + i * acc.step
            let v: UInt32
            switch acc.componentType {
            case Self.uint: v = u32(p)
            case Self.ushort: v = UInt32(u16(p))
            case Self.ubyte: v = UInt32(bytes[p])
            default: throw Self.fail("The head file's corners use a kind of number this app does not read.")
            }
            guard Int(v) < vertexCount else { throw Self.fail("The head file names a corner that is not there.") }
            out[i] = v
        }
        return out
    }

    func readParts() throws -> [HeadsMeshPart] {
        var parts: [HeadsMeshPart] = []
        for case let mesh as [String: Any] in doc["meshes"] as? [Any] ?? [] {
            for case let primitive as [String: Any] in mesh["primitives"] as? [Any] ?? [] {
                parts.append(try part(primitive))
            }
        }
        return parts
    }

    func vec3(_ acc: Accessor, count n: Int, what: String) throws -> [SIMD3<Float>] {
        guard acc.width == 3, acc.count == n, acc.componentType == Self.float || acc.normalized else {
            throw Self.fail("The head file's \(what) are not stored as this app reads them.")
        }
        let f = try floats(acc)
        guard f.allSatisfy({ $0.isFinite }) else { throw Self.fail("The head file holds \(what) that are no numbers.") }
        return (0..<n).map { SIMD3<Float>(f[$0 * 3], f[$0 * 3 + 1], f[$0 * 3 + 2]) }
    }

    func part(_ primitive: [String: Any]) throws -> HeadsMeshPart {
        guard Self.int(primitive["mode"] ?? 4) == 4 else {
            throw Self.fail("The head file uses a kind of surface this app does not read.")
        }
        guard let attributes = primitive["attributes"] as? [String: Any], attributes["POSITION"] != nil else {
            throw Self.fail("A part of the head file has no points.")
        }
        let positionAccessor = try accessor(attributes["POSITION"])
        guard positionAccessor.width == 3, positionAccessor.componentType == Self.float else {
            throw Self.fail("The head file's points are not stored as this app reads them.")
        }
        let positions = try vec3(positionAccessor, count: positionAccessor.count, what: "points")
        let n = positions.count

        var normals: [SIMD3<Float>]?
        if attributes["NORMAL"] != nil {
            normals = try vec3(try accessor(attributes["NORMAL"]), count: n, what: "normals")
        }
        var colors: [SIMD4<Float>]?
        if attributes["COLOR_0"] != nil {
            let acc = try accessor(attributes["COLOR_0"])
            guard acc.width == 3 || acc.width == 4, acc.count == n, acc.componentType == Self.float || acc.normalized else {
                throw Self.fail("The head file's colours are not stored as this app reads them.")
            }
            let f = try floats(acc)
            guard f.allSatisfy({ $0.isFinite }) else { throw Self.fail("The head file holds colours that are no numbers.") }
            colors = (0..<n).map { i in
                let b = i * acc.width
                return SIMD4<Float>(f[b], f[b + 1], f[b + 2], acc.width == 4 ? f[b + 3] : 1)
            }
        }
        var uvs: [SIMD2<Float>]?
        if attributes["TEXCOORD_0"] != nil {
            let acc = try accessor(attributes["TEXCOORD_0"])
            guard acc.width == 2, acc.count == n, acc.componentType == Self.float || acc.normalized else {
                throw Self.fail("The head file's picture places are not stored as this app reads them.")
            }
            let f = try floats(acc)
            guard f.allSatisfy({ $0.isFinite }) else { throw Self.fail("The head file holds picture places that are no numbers.") }
            uvs = (0..<n).map { SIMD2<Float>(f[$0 * 2], f[$0 * 2 + 1]) }
        }

        let corners: [UInt32]
        if primitive["indices"] != nil {
            corners = try indices(try accessor(primitive["indices"]), vertexCount: n)
        } else {
            corners = (0..<n).map { UInt32($0) }
        }
        guard corners.count % 3 == 0 else { throw Self.fail("The head file has a triangle with a corner missing.") }

        var texture: Data?
        var blended = false
        var doubleSided = false
        if primitive["material"] != nil {
            guard let mi = Self.int(primitive["material"]), let materials = doc["materials"] as? [Any],
                  materials.indices.contains(mi), let material = materials[mi] as? [String: Any] else {
                throw Self.fail("The head file names a material that is not there.")
            }
            blended = (material["alphaMode"] as? String) == "BLEND"
            doubleSided = (material["doubleSided"] as? Bool) ?? false
            if let pbr = material["pbrMetallicRoughness"] as? [String: Any],
               let base = pbr["baseColorTexture"] as? [String: Any] {
                texture = try picture(forTexture: base["index"])
            }
        }
        return HeadsMeshPart(positions: positions, normals: normals, colors: colors, uvs: uvs, indices: corners,
                             texture: texture, blended: blended, doubleSided: doubleSided)
    }

    /// The picture a texture shows: bytes inside the file, a JPEG or a PNG.
    func picture(forTexture index: Any?) throws -> Data {
        guard let ti = Self.int(index), let textures = doc["textures"] as? [Any], textures.indices.contains(ti),
              let texture = textures[ti] as? [String: Any], let si = Self.int(texture["source"]),
              let images = doc["images"] as? [Any], images.indices.contains(si),
              let image = images[si] as? [String: Any] else {
            throw Self.fail("The head file names a picture that is not there.")
        }
        guard let vi = Self.int(image["bufferView"]), views.indices.contains(vi) else {
            throw Self.fail("The head file's picture is not inside the file.")
        }
        let view = views[vi]
        let data = Data(bytes[view.start..<(view.start + view.length)])
        let jpeg = data.starts(with: [0xFF, 0xD8, 0xFF])
        let png = data.starts(with: [0x89, 0x50, 0x4E, 0x47])
        guard jpeg || png else { throw Self.fail("The head file's picture is neither a JPEG nor a PNG.") }
        if let mime = image["mimeType"] as? String, mime != (jpeg ? "image/jpeg" : "image/png") {
            throw Self.fail("The head file's picture is not of the kind it says.")
        }
        return data
    }
}
