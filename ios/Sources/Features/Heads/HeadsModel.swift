import SwiftUI
import Observation

// What the 3D tab knows besides the views: the heads it has read (HeadsStore), the look sliders' logic
// (HeadsLooks, the website's lib/model3d.ts), the options of a generation (HeadsOptions) and the tab's words
// (HeadsWords).

/// The heads of one person's model that have been read: fetched through the signed-in client when first wanted,
/// read off the main thread, and kept for the head on screen and four more (the website's KEEP_MESHES).
@MainActor
@Observable
final class HeadsStore {
    enum Load: Equatable {
        case loading
        case ready
        case failed(String)
    }

    static let keep = 5

    let personId: Int
    /// By mesh address.
    private(set) var states: [String: Load] = [:]
    @ObservationIgnored private var heads: [String: HeadsPrepared] = [:]
    @ObservationIgnored private var recent: [String] = []
    @ObservationIgnored private var tasks: [String: Task<Void, Never>] = [:]

    init(personId: Int) {
        self.personId = personId
    }

    func prepared(for address: String) -> HeadsPrepared? { heads[address] }

    /// Show this look: fetch its head unless it is read or on its way.
    func want(_ look: API.Look3dOut) {
        let address = look.meshUrl
        touch(address)
        switch states[address] {
        case .ready?, .loading?:
            return
        default:
            load(address)
        }
    }

    /// Fetch a head again after it failed.
    func retry(_ look: API.Look3dOut) {
        states[look.meshUrl] = nil
        want(look)
    }

    private func touch(_ address: String) {
        recent.removeAll { $0 == address }
        recent.append(address)
    }

    private func load(_ address: String) {
        states[address] = .loading
        let person = personId
        tasks[address] = Task { [weak self] in
            do {
                let file = try HeadsFiles.parts(of: address, personId: person)
                let data = try await Cloud.api().model3dFile(personId: person, rev: file.rev, name: file.name)
                let model = try await Task.detached(priority: .userInitiated) { try HeadsGLB.read(data) }.value
                guard let self else { return }
                self.heads[address] = HeadsSceneBuilder.prepare(model)
                self.states[address] = .ready
                self.tidy()
            } catch {
                guard let self, !(error is CancellationError) else { return }
                self.states[address] = .failed(Self.describe(error))
            }
            self?.tasks[address] = nil
        }
    }

    /// Lets go of the heads that were on screen longest ago.
    private func tidy() {
        let kept = Set(recent.suffix(Self.keep))
        for address in heads.keys where !kept.contains(address) {
            heads[address] = nil
            states[address] = nil
        }
        recent = recent.filter { kept.contains($0) }
    }

    static func describe(_ error: Error) -> String {
        if let e = error as? HeadsGLBError { return e.message }
        return ErrorText.describe(error, doing: "load the head")
    }
}

/// Where a look's file is: the rev and the name in the address the server gave for it
/// ("/files/model3d/<person>/<rev>/<name>"). The address is read, never made up here.
enum HeadsFiles {
    static func parts(of address: String, personId: Int) throws -> (rev: String, name: String) {
        guard let url = URL(string: address), url.scheme == nil, url.host == nil else {
            throw HeadsGLBError(message: "The head's address is not one this app reads.")
        }
        let parts = url.path.split(separator: "/").map(String.init)
        guard parts.count == 5, parts[0] == "files", parts[1] == "model3d", parts[2] == String(personId),
              !parts[3].isEmpty, !parts[4].isEmpty else {
            throw HeadsGLBError(message: "The head's address is not one this app reads.")
        }
        return (parts[3], parts[4])
    }
}

/// One slider of the looks: an axis (age, hair, facial hair) and its stops, the values the looks have, in order.
struct HeadsLookAxis: Equatable {
    enum Key: String, CaseIterable {
        case age, hair, facialHair
    }

    struct Stop: Equatable {
        /// The words for this stop ("about 24", "short hair", "a beard").
        let text: String
        /// The looks at this stop, in the server's order.
        let keys: [String]
    }

    let key: Key
    let label: String
    /// Empty when no look has a value on this axis: the slider is left out.
    let stops: [Stop]
}

/// The look sliders (the website's lookAxes, stopOf, lookAxisText, pickLook and firstLook). Every look sits at one
/// stop of each axis. Moving one slider decides that axis first; among the looks at the new stop, the one nearest
/// to the head on screen on the other axes is shown, and the other sliders follow it.
enum HeadsLooks {
    static let notRead = "not read"

    static func label(_ key: HeadsLookAxis.Key) -> String {
        switch key {
        case .age: return "Age"
        case .hair: return "Hair"
        case .facialHair: return "Facial hair"
        }
    }

    static func group(_ look: API.Look3dOut, _ key: HeadsLookAxis.Key) -> String? {
        switch key {
        case .age: return look.ageBand
        case .hair: return look.hairLabel
        case .facialHair: return look.facialHairLabel
        }
    }

    static func level(_ look: API.Look3dOut, _ key: HeadsLookAxis.Key) -> Double? {
        switch key {
        case .age: return look.age
        case .hair: return look.hair
        case .facialHair: return look.facialHair
        }
    }

    /// The three axes of a model's looks, each with its stops in order (low to high).
    static func axes(_ looks: [API.Look3dOut]) -> [HeadsLookAxis] {
        HeadsLookAxis.Key.allCases.map { key -> HeadsLookAxis in
            var order: [String] = []
            var members: [String: [API.Look3dOut]] = [:]
            for look in looks {
                let g = group(look, key) ?? ""
                if members[g] == nil { order.append(g) }
                members[g, default: []].append(look)
            }
            if order == [""] { return HeadsLookAxis(key: key, label: label(key), stops: []) }
            var rows: [(text: String, keys: [String], at: Double, index: Int)] = []
            for (index, g) in order.enumerated() {
                let list = members[g] ?? []
                let levels = list.compactMap { level($0, key) }
                let mean: Double? = levels.isEmpty ? nil : levels.reduce(0, +) / Double(levels.count)
                var text = g
                if g.isEmpty {
                    text = notRead
                } else if key == .age, let mean {
                    text = "about \(Int(mean.rounded()))"
                }
                // a group nothing was read for sorts first; one without a level keeps the server's order
                let at = g.isEmpty ? -Double.infinity : (mean ?? Double.infinity)
                rows.append((text, list.map(\.key), at, index))
            }
            rows.sort { a, b in a.at == b.at ? a.index < b.index : a.at < b.at }
            return HeadsLookAxis(key: key, label: label(key),
                                 stops: rows.map { HeadsLookAxis.Stop(text: $0.text, keys: $0.keys) })
        }
    }

    /// Which stop of the axis the look sits at (-1 when the axis has no stops or the look is unknown).
    static func stop(of lookKey: String, on axis: HeadsLookAxis) -> Int {
        axis.stops.firstIndex(where: { $0.keys.contains(lookKey) }) ?? -1
    }

    /// The words for the look's value on an axis: its own age when the axis is age, else the stop's words.
    static func text(_ axis: HeadsLookAxis, for look: API.Look3dOut) -> String {
        if axis.key == .age, let age = look.age { return "about \(Int(age.rounded()))" }
        let i = stop(of: look.key, on: axis)
        return i < 0 ? notRead : axis.stops[i].text
    }

    /// The look to show after the owner moved one slider to a stop, starting from the look on screen.
    static func pick(_ axes: [HeadsLookAxis], from current: String, moved: HeadsLookAxis.Key, to index: Int) -> String {
        guard let axis = axes.first(where: { $0.key == moved }), axis.stops.indices.contains(index) else { return current }
        let target = axis.stops[index]
        if target.keys.isEmpty || target.keys.contains(current) { return current }
        var best = target.keys[0]
        var bestDistance = Double.infinity
        for key in target.keys {
            var distance = 0.0
            for other in axes where other.key != moved && other.stops.count >= 2 {
                let from = stop(of: current, on: other)
                let to = stop(of: key, on: other)
                distance += from < 0 || to < 0 ? 0.5 : Double(abs(from - to)) / Double(other.stops.count - 1)
            }
            if distance < bestDistance {
                best = key
                bestDistance = distance
            }
        }
        return best
    }

    /// The look a model opens on: the person as they are now, so the latest age, and there the look most photos
    /// show.
    static func first(_ looks: [API.Look3dOut]) -> API.Look3dOut? {
        guard let last = looks.last else { return nil }
        let same = looks.filter { $0.ageBand == last.ageBand }
        return same.dropFirst().reduce(same[0]) { best, look in look.photos > best.photos ? look : best }
    }

    /// "About 33" and "short hair, a beard": a look's label in two lines.
    static func lines(_ look: API.Look3dOut) -> (String, String) {
        let first = look.age.map { "About \(Int($0.rounded()))" } ?? look.ageBand
        let rest = [look.hairLabel, look.facialHairLabel].compactMap { $0 }.filter { !$0.isEmpty }
        return (first, rest.joined(separator: ", "))
    }
}

/// The choices of a generation and their ranges (the website's MODEL3D_DEFAULTS and MODEL3D_LIMITS).
struct HeadsOptions: Equatable {
    var maxLooks = 6
    var detail = 2
    var framing = 2.6
    var smoothing = 2
    var texture = 2
    var relief = 1

    static let defaults = HeadsOptions()
    static let detailWords = ["coarse", "normal", "fine"]
    static let smoothingWords = ["off", "light", "normal", "strong"]
    static let textureWords = ["off", "standard", "fine", "full"]

    /// The options a model was made with, complete and inside the ranges; what a model from before an option
    /// existed does not say starts at its default.
    static func complete(_ made: API.Model3dOptions?) -> HeadsOptions {
        var out = HeadsOptions()
        guard let made else { return out }
        out.maxLooks = min(12, max(1, made.maxLooks))
        out.detail = min(3, max(1, made.detail))
        if made.framing.isFinite { out.framing = min(3.2, max(1.8, (made.framing * 10).rounded() / 10)) }
        if let v = made.smoothing { out.smoothing = min(3, max(0, v)) }
        if let v = made.texture { out.texture = min(3, max(0, v)) }
        if let v = made.relief { out.relief = min(1, max(0, v)) }
        return out
    }

    var body: API.Model3dIn {
        API.Model3dIn(maxLooks: maxLooks, detail: detail, framing: (framing * 10).rounded() / 10, smoothing: smoothing,
                      texture: texture, relief: relief)
    }

    static func word(_ words: [String], _ value: Int, offset: Int = 0) -> String {
        let i = value - offset
        return words.indices.contains(i) ? words[i] : String(value)
    }
}

/// The 3D tab's words, as the website says them.
enum HeadsWords {
    /// Why nothing can be started right now, in plain words; nil when it can.
    static func blocked(_ status: API.Model3dOut) -> String? {
        if !status.available { return status.reason ?? "A model cannot be generated right now." }
        if status.busy { return "Another person's model is being generated now. One is made at a time." }
        return nil
    }

    /// The model is from before heads carried their photo.
    static func beforePhotoTextures(_ status: API.Model3dOut) -> Bool {
        !status.looks.isEmpty && !status.looks.contains(where: { $0.textured }) && status.options?.texture == nil
    }

    static func startError(_ error: Error) -> String {
        if let e = error as? APIError {
            switch e.detail {
            case "busy": return "Another model is being generated. Try again when it is done."
            case "generator_missing": return "The 3D generator is not installed on the server."
            case "not_enough_photos": return "There are too few clear, large face photos of this person."
            default: break
            }
        }
        return ErrorText.describe(error, doing: "start")
    }

    static func degrees(_ value: Double, zero: String, half: String, positive: String, negative: String) -> String {
        let d = Int(value.rounded())
        if d == 0 { return zero }
        if abs(d) >= 180 { return half }
        return "\(abs(d))° \(d > 0 ? positive : negative)"
    }

    static func percent(_ value: Double) -> String { "\(Int((value * 100).rounded()))%" }

    /// A note of the generator as a sentence of its own: a capital first, a full stop last.
    static func sentence(_ text: String?) -> String? {
        guard let t = text?.trimmingCharacters(in: .whitespacesAndNewlines), let first = t.first else { return nil }
        let s = first.uppercased() + t.dropFirst()
        return [".", "!", "?"].contains(where: { s.hasSuffix($0) }) ? s : s + "."
    }
}
