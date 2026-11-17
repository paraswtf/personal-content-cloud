import Foundation
import CoreGraphics

/// One box of an item as the editors see it, from the item's whole detail: a face, a person box or a pet box
/// (the website's BoxTarget in viewer/boxes.ts).
enum BoxesTarget: Identifiable, Hashable {
    case face(API.FaceOut)
    case body(API.BodyOut)
    case animal(API.AnimalOut)

    var kind: StageBox.Kind {
        switch self {
        case .face: return .face
        case .body: return .body
        case .animal: return .animal
        }
    }

    var boxId: Int {
        switch self {
        case .face(let f): return f.id
        case .body(let b): return b.id
        case .animal(let a): return a.id
        }
    }

    /// "face:3", the same key as StageBox.id.
    var id: String { "\(kind.rawValue):\(boxId)" }

    var rect: CGRect {
        switch self {
        case .face(let f): return CGRect(x: f.x, y: f.y, width: f.w, height: f.h)
        case .body(let b): return CGRect(x: b.x, y: b.y, width: b.w, height: b.h)
        case .animal(let a): return CGRect(x: a.x, y: a.y, width: a.w, height: a.h)
        }
    }

    /// The person or pet on the box, when it has one.
    var person: API.PersonRef? {
        switch self {
        case .face(let f): return f.person
        case .body(let b): return b.person
        case .animal(let a): return a.person
        }
    }

    /// An open guess: a named person the system suggests for a box with nobody on it (never an unnamed one).
    var guess: API.GuessOut? {
        switch self {
        case .face(let f): return f.person == nil && Fmt.isNamed(f.guess?.person.name) ? f.guess : nil
        case .body(let b): return b.person == nil && Fmt.isNamed(b.guess?.person.name) ? b.guess : nil
        case .animal: return nil
        }
    }

    /// The box's crop.
    var cropRef: String {
        switch self {
        case .face(let f): return f.url
        case .body(let b): return b.url
        case .animal(let a): return a.url
        }
    }

    /// Videos: the time of the frame the box was found on; nil on a photo.
    var frameTimeS: Double? {
        switch self {
        case .face(let f): return f.frameTimeS
        case .body(let b): return b.frameTimeS
        case .animal(let a): return a.frameTimeS
        }
    }

    /// The video track the box belongs to (boxes of one track share it).
    var group: Int? {
        switch self {
        case .face(let f): return f.group
        case .body(let b): return b.group
        case .animal(let a): return a.group
        }
    }

    /// A pet box's species.
    var species: String? {
        if case .animal(let a) = self { return a.species }
        return nil
    }

    /// The name the box shows: the person's or pet's when named.
    var personName: String? {
        guard let p = person, Fmt.isNamed(p.name) else { return nil }
        return p.name?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The guessed person's name, for a box with an open guess.
    var guessName: String? {
        guess.map { Fmt.personName($0.person.name) }
    }

    /// Who put the person (or the guess) on the box (the website's boxOrigin).
    var origin: BoxOrigin? {
        switch self {
        case .face(let f):
            return BoxOrigin.of(hasPerson: f.person != nil, assignedByUser: f.assignedBy == .user,
                                drawnByUser: f.source == .user, byGuess: f.byGuess, hasGuess: guess != nil)
        case .body(let b):
            return BoxOrigin.of(hasPerson: b.person != nil, assignedByUser: b.assignedBy == .user,
                                drawnByUser: b.source == .user, byGuess: b.byGuess, hasGuess: guess != nil)
        case .animal(let a):
            return BoxOrigin.of(hasPerson: a.person != nil, assignedByUser: a.assignedBy == .user,
                                drawnByUser: a.source == .user)
        }
    }

    /// The box as the stage draws it.
    var stageBox: StageBox {
        let r = rect
        let name = kind == .animal ? (personName ?? Fmt.speciesLabel(species)) : personName
        return StageBox(kind: kind, boxId: boxId, x: Double(r.minX), y: Double(r.minY), w: Double(r.width),
                        h: Double(r.height), name: name, guessName: kind == .animal ? nil : guessName,
                        personId: person?.id)
    }

    /// "face", "person box" or "pet box".
    var noun: String {
        switch kind {
        case .face: return "face box"
        case .body: return "person box"
        case .animal: return "pet box"
        }
    }

    /// Whose box it is, for the adjust bar (the website's adjustLabel): "the face box of Sam", "this pet box".
    var adjustLabel: String {
        if kind == .animal {
            return person != nil ? "the \(noun) of \(personName ?? Fmt.speciesLabel(species))" : "this \(noun)"
        }
        if let name = personName { return "the \(noun) of \(name)" }
        if let g = guessName { return "this \(noun), maybe \(g)" }
        return "this \(noun)"
    }

    /// A person box's body traits (tattoos, nails, mehndi).
    var traits: [API.TraitOut] {
        if case .body(let b) = self { return b.traits ?? [] }
        return []
    }
}

/// Finding boxes in an item's detail.
enum BoxesLookup {
    /// A box by kind and id, wherever it is (a photo's or a video frame's).
    static func target(in item: API.ItemDetail, kind: StageBox.Kind, id: Int) -> BoxesTarget? {
        switch kind {
        case .face: return item.faces.first(where: { $0.id == id }).map { BoxesTarget.face($0) }
        case .body: return item.bodies.first(where: { $0.id == id }).map { BoxesTarget.body($0) }
        case .animal: return item.animals.first(where: { $0.id == id }).map { BoxesTarget.animal($0) }
        }
    }

    /// The person box a face belongs to (its traits are on the body).
    static func body(of faceId: Int, in item: API.ItemDetail) -> API.BodyOut? {
        item.bodies.first { $0.faceId == faceId }
    }

    /// The boxes a still picture draws: faces, person boxes whose face was not found, and pets.
    static func stillTargets(_ item: API.ItemDetail) -> [BoxesTarget] {
        item.faces.filter { $0.frameTimeS == nil }.map { BoxesTarget.face($0) }
            + item.bodies.filter { $0.frameTimeS == nil && $0.faceId == nil }.map { BoxesTarget.body($0) }
            + item.animals.filter { $0.frameTimeS == nil }.map { BoxesTarget.animal($0) }
    }

    /// The open guesses of an item, faces first, then person boxes ("Guesses and why").
    static func guessTargets(_ item: API.ItemDetail) -> [BoxesTarget] {
        item.faces.map { BoxesTarget.face($0) }.filter { $0.guess != nil }
            + item.bodies.map { BoxesTarget.body($0) }.filter { $0.guess != nil }
    }

    /// Whole-video editing (the website's groupVideoBoxes): the boxes of the target's track in the video, the
    /// target among them, in time order. A photo's box, or a box without a track, is alone.
    static func tracks(of target: BoxesTarget, in item: API.ItemDetail) -> [BoxesTarget] {
        guard target.frameTimeS != nil, let group = target.group else { return [target] }
        var all: [BoxesTarget] = []
        if target.kind == .animal {
            all = item.animals.filter { $0.frameTimeS != nil && $0.group == group }.map { BoxesTarget.animal($0) }
        } else {
            all = item.faces.filter { $0.frameTimeS != nil && $0.group == group }.map { BoxesTarget.face($0) }
                + item.bodies.filter { $0.frameTimeS != nil && $0.group == group }.map { BoxesTarget.body($0) }
        }
        let sorted = all.sorted { ($0.frameTimeS ?? 0) < ($1.frameTimeS ?? 0) }
        return sorted.contains(target) ? sorted : [target]
    }

    /// Who put each person or pet on the item (the website's itemOrigins): the owner's tag or box wins over the
    /// system's.
    static func itemOrigins(_ item: API.ItemDetail) -> [Int: BoxOrigin] {
        let rank: [BoxOrigin: Int] = [.auto: 0, .confirmed: 1, .drawn: 2, .you: 3]
        var out: [Int: BoxOrigin] = [:]
        func take(_ id: Int, _ origin: BoxOrigin) {
            if let cur = out[id], (rank[cur] ?? 0) >= (rank[origin] ?? 0) { return }
            out[id] = origin
        }
        for t in item.tags { take(t.id, (t.byGuess ?? false) ? .confirmed : .you) }
        let boxes = item.faces.map { BoxesTarget.face($0) } + item.bodies.map { BoxesTarget.body($0) }
            + item.animals.map { BoxesTarget.animal($0) }
        for b in boxes {
            guard let id = b.person?.id, let o = b.origin else { continue }
            take(id, [BoxOrigin.you, .drawn, .confirmed].contains(o) ? o : .auto)
        }
        return out
    }

    /// A person's thumb on this item: their first box's crop, else their cover.
    static func thumb(of person: API.PersonRef, in item: API.ItemDetail) -> String? {
        if let f = item.faces.first(where: { $0.person?.id == person.id }) { return f.url }
        if let b = item.bodies.first(where: { $0.person?.id == person.id }) { return b.url }
        if let a = item.animals.first(where: { $0.person?.id == person.id }) { return a.url }
        return person.coverUrl
    }
}

/// Words the box editors share (the website's lib/format.ts and the editors' own).
enum BoxesWords {
    /// "a dog", "an elephant".
    static func aSpecies(_ species: String?) -> String {
        let word = Fmt.speciesLabel(species).lowercased()
        let vowel = word.first.map { "aeiou".contains($0) } ?? false
        return (vowel ? "an " : "a ") + word
    }

    /// A naming message with the guesses naming made: "Tagged as Sam. 2 faces that look alike wait in Review."
    static func withGuessed(_ message: String, _ guessed: Int?) -> String {
        guard let guessed, guessed > 0 else { return message }
        return message + "." + Fmt.guessedTail(guessed)
    }

    /// " (3 tracks)" when several tracks of a video were changed at once.
    static func trackTail(_ count: Int) -> String {
        count > 1 ? " (\(count) tracks)" : ""
    }

    /// The name of a person from an answer.
    static func name(_ p: API.PersonOut?) -> String? {
        guard let p else { return nil }
        return Fmt.personName(p.name, kind: p.kind == .pet ? "pet" : "person", species: p.species)
    }

    /// The website's displayName: the name, or "Unnamed person" / "Unnamed pet".
    static func displayName(_ name: String?, pet: Bool) -> String {
        if let n = name?.trimmingCharacters(in: .whitespacesAndNewlines), !n.isEmpty { return n }
        return pet ? "Unnamed pet" : "Unnamed person"
    }

    /// The tail of a change on an item (the website's followedTail in PeopleBulkAction).
    static func followedTail(followed: Int, pulled: Int, pet: Bool) -> String {
        var out = ""
        if followed > 0 { out += " \(Fmt.count(followed, pet ? "look-alike crop" : "look-alike face")) followed." }
        if pulled > 0 { out += " \(Fmt.count(pulled, "look-alike group")) joined them." }
        return out
    }

    /// "Not a tattoo", "Not nail polish", "Not mehndi".
    static func notTraitLabel(_ kind: API.TraitGroupOutKind) -> String {
        switch kind {
        case .tattoo: return "Not a tattoo"
        case .mehndi: return "Not mehndi"
        default: return "Not nail polish"
        }
    }

    /// "tattoo", "nail polish", "mehndi".
    static func traitWord(_ kind: API.TraitGroupOutKind) -> String {
        switch kind {
        case .tattoo: return "tattoo"
        case .mehndi: return "mehndi"
        default: return "nail polish"
        }
    }

    /// What a failed new box means, in words (the website's PersonBoxCard failure()).
    static func addFailure(_ error: Error) -> String {
        if let api = error as? APIError {
            switch api.detail {
            case "frame_time_needed": return "Pause the video on a frame, then draw the box again."
            case "person_needed": return "Pick someone, or type a name."
            case "no_face":
                return "No face found in that box. Draw it closer around the face, or add a person box when the face does not show."
            case "models_unavailable": return "The face model is not loaded right now. Try again in a while."
            default: break
            }
            if api.status == 409 { return "The photo or frame could not be opened." }
            if api.status == 422 { return "Draw a larger box, inside the picture." }
        }
        return ErrorText.describe(error, doing: "add the box")
    }
}
