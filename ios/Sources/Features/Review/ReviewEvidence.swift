import Foundation

/// The evidence of a person guess in words (the website's components/people/evidence.ts). A clothes guess reads as
/// one line ("25 min apart, same scene 0.86, same clothes 0.91"); body traits (section 34) add their own reasons
/// ("Same tattoo on the forearm as Sam's photo from 14 Mar 2026"), always first. A guess from traits alone (basis
/// "trait") shows no percentage: it reads "strong match". A guess of recognition trained on one person (basis
/// "face", section 10) reads "By face" with its probability and one line ("Looks like Sam's photos, 0.61, and the
/// same clothes"). Position (section 35) adds its own reason ("In the same spot as Sam 8 s earlier"). A guess made
/// when the owner named a face the app had grouped with this one (basis "group") reads "Grouped" with its
/// similarity ("face 0.62") and "grouped with the face you named Sam".
enum ReviewEvidence {
    typealias Evidence = [String: API.JSONValue]

    enum TraitKind: String, CaseIterable {
        case tattoo, nails, mehndi
    }

    /// What a guess rests on (`evidence.basis`; none means clothes, as before traits).
    enum Basis: String {
        case clothes
        case trait
        case clothesTrait = "clothes+trait"
        case face
        case place
        case clothesPlace = "clothes+place"
        case group
    }

    /// The icons that go with the label.
    enum Icon: String {
        case clothes, tattoo, hand, face, place

        var systemImage: String {
            switch self {
            case .clothes: return "tshirt"
            case .tattoo: return "paintbrush.pointed"
            case .hand: return "hand.raised"
            case .face: return "faceid"
            case .place: return "figure.walk"
            }
        }
    }

    /// One positive trait entry of the evidence.
    struct Trait: Hashable {
        var kind: TraitKind
        var part: String
        var refPart: String?
        var mirrored = false
        var traitId: Int?
        var refTraitId: Int?
        var refBodyId: Int?
        var refItemUid: String?
        var refTakenAt: String?
        var daysApart: Double?
        var refBefore: Bool?
        var inliers: Double?
        var support: Double?
        var strong = false
        var llr: Double?
        var color: String?
        var hex: String?
        var finish: String?
        var length: String?
        var cropUrl: String?
        var refCropUrl: String?
    }

    /// A trait the guessed person has that this box shows bare (it counts against, softly).
    struct Against: Hashable {
        var kind: TraitKind
        var part: String
        var llr: Double?
        var why: String?
    }

    /// The position part of the evidence (section 35): the box the person was followed from.
    struct Place: Hashable {
        var refItemUid: String?
        var refFaceId: Int?
        var refBodyId: Int?
        var secondsApart: Double?
        var refBefore: Bool?
        var hops: Double?
        var offset: Double?
        var kind: String?
        var others: Double?
        var clothes: Double?
        var llr: Double?
        var refCropUrl: String?
    }

    struct TraitReason: Hashable, Identifiable {
        /// The reason as a sentence without a full stop.
        var text: String
        var kind: TraitKind
        /// The part of this photo's box, as stored ("forearm_l").
        var part: String
        var cropUrl: String?
        var refCropUrl: String?
        var refItemUid: String?
        var refTakenAt: String?
        var mirrored: Bool
        /// Nails: the polish colour, for a swatch.
        var hex: String?

        var id: String { "\(kind.rawValue):\(part):\(text)" }
    }

    struct PlaceReason: Hashable {
        /// The reason as a sentence without a full stop.
        var text: String
        var refCropUrl: String?
        var refItemUid: String?
        /// "8 s earlier", for the reference crop's caption.
        var when: String
    }

    // MARK: Reading the JSON

    /// A number of the evidence; one no measurement could be (not finite, or past a thousand million million) is none,
    /// so the words built from it never turn an absurd value into a whole number that stops the app.
    static func num(_ v: API.JSONValue?) -> Double? {
        guard let d = v?.doubleValue, d.isFinite, abs(d) < 1e15 else { return nil }
        return d
    }

    static func int(_ v: API.JSONValue?) -> Int? {
        guard let d = num(v) else { return nil }
        // a number past what an Int holds is no answer (Int(_:) would stop the app)
        return Int(exactly: d.rounded())
    }

    static func str(_ v: API.JSONValue?) -> String? {
        guard let s = v?.stringValue, !s.isEmpty else { return nil }
        return s
    }

    /// A server path for an image, nothing else.
    static func path(_ v: API.JSONValue?) -> String? {
        guard let s = v?.stringValue, s.hasPrefix("/"), !s.hasPrefix("//") else { return nil }
        return s
    }

    /// "#a1b2c3" only, so a colour from the server is used as nothing else.
    static func safeHex(_ v: API.JSONValue?) -> String? {
        guard let s = v?.stringValue, s.count == 7, s.hasPrefix("#") else { return nil }
        let digits = "0123456789abcdefABCDEF"
        return s.dropFirst().allSatisfy({ digits.contains($0) }) ? s : nil
    }

    // MARK: People

    /// The name to show for the guessed person ("Unnamed" never happens for a guess the server sends).
    static func displayName(_ person: API.PersonRef) -> String {
        if let n = person.name?.trimmingCharacters(in: .whitespacesAndNewlines), !n.isEmpty { return n }
        return person.kind == .pet ? "Unnamed pet" : "Unnamed"
    }

    static func isNamed(_ person: API.PersonRef) -> Bool { Fmt.isNamed(person.name) }

    /// "Sam's", or "their" when the guess names no one.
    static func whose(_ person: API.PersonRef) -> String {
        isNamed(person) ? "\(displayName(person))'s" : "their"
    }

    // MARK: The parts of the evidence

    static func basis(_ e: Evidence) -> Basis {
        switch e["basis"]?.stringValue {
        case "trait": return .trait
        case "clothes+trait": return .clothesTrait
        case "face": return .face
        case "place": return .place
        case "clothes+place": return .clothesPlace
        case "group": return .group
        default: return .clothes
        }
    }

    static func place(_ e: Evidence) -> Place? {
        guard let o = e["place"]?.objectValue else { return nil }
        var p = Place()
        p.refItemUid = str(o["ref_item_uid"])
        p.refFaceId = int(o["ref_face_id"])
        p.refBodyId = int(o["ref_body_id"])
        p.secondsApart = num(o["seconds_apart"])
        p.refBefore = o["ref_before"]?.boolValue
        p.hops = num(o["hops"])
        p.offset = num(o["offset"])
        p.kind = str(o["kind"])
        p.others = num(o["others"])
        p.clothes = num(o["clothes"])
        p.llr = num(o["llr"])
        p.refCropUrl = path(o["ref_crop_url"])
        return p
    }

    /// The positive trait entries, strongest first as the server sorted them; malformed ones dropped.
    static func traits(_ e: Evidence) -> [Trait] {
        guard let raw = e["traits"]?.arrayValue else { return [] }
        var out: [Trait] = []
        for entry in raw {
            guard let o = entry.objectValue, let kindText = o["kind"]?.stringValue,
                  let kind = TraitKind(rawValue: kindText), let part = o["part"]?.stringValue else { continue }
            var t = Trait(kind: kind, part: part)
            t.refPart = str(o["ref_part"])
            t.mirrored = o["mirrored"]?.boolValue == true
            t.traitId = int(o["trait_id"])
            t.refTraitId = int(o["ref_trait_id"])
            t.refBodyId = int(o["ref_body_id"])
            t.refItemUid = str(o["ref_item_uid"])
            t.refTakenAt = str(o["ref_taken_at"])
            t.daysApart = num(o["days_apart"])
            t.refBefore = o["ref_before"]?.boolValue
            t.inliers = num(o["inliers"])
            t.support = num(o["support"])
            t.strong = o["strong"]?.boolValue == true
            t.llr = num(o["llr"])
            t.color = str(o["color"])
            t.hex = safeHex(o["hex"])
            t.finish = str(o["finish"])
            t.length = str(o["length"])
            t.cropUrl = path(o["crop_url"])
            t.refCropUrl = path(o["ref_crop_url"])
            out.append(t)
        }
        return out
    }

    static func against(_ e: Evidence) -> [Against] {
        guard let raw = e["against"]?.arrayValue else { return [] }
        var out: [Against] = []
        for entry in raw {
            guard let o = entry.objectValue, let kindText = o["kind"]?.stringValue,
                  let kind = TraitKind(rawValue: kindText), let part = o["part"]?.stringValue else { continue }
            out.append(Against(kind: kind, part: part, llr: num(o["llr"]), why: str(o["why"])))
        }
        return out
    }

    // MARK: Words

    /// How far apart two photos of one sequence were, from the reference's side: "8 s earlier", "2 min later",
    /// "in the same second". Empty when the gap is unknown.
    static func gapPhrase(_ seconds: Double?, before: Bool?) -> String {
        guard let seconds, seconds.isFinite else { return "" }
        let s = abs(seconds)
        if s < 1 { return "in the same second" }
        let gap = s < 90 ? "\(Int(s.rounded())) s" : "\(Int((s / 60).rounded())) min"
        if before == true { return "\(gap) earlier" }
        if before == false { return "\(gap) later" }
        return "\(gap) apart"
    }

    /// How far apart two photos were in days: "the same day", "2 days earlier", "1 day later".
    static func daysPhrase(_ days: Double?, before: Bool?) -> String {
        guard let days, days.isFinite else { return "" }
        let n = Int(abs(days).rounded())
        if n == 0 { return "the same day" }
        let unit = n == 1 ? "day" : "days"
        if before == true { return "\(n) \(unit) earlier" }
        if before == false { return "\(n) \(unit) later" }
        return "\(n) \(unit) apart"
    }

    private static let partNames = ["forearm": "forearm", "upper_arm": "upper arm", "calf": "calf", "hand": "hand"]

    /// "forearm_l" becomes "forearm": matching never looks at the side (a mirror photo swaps it).
    static func partWords(_ part: String) -> String {
        var family = part
        if part.hasSuffix("_l") || part.hasSuffix("_r") { family = String(part.dropLast(2)) }
        return partNames[family] ?? family.replacingOccurrences(of: "_", with: " ")
    }

    /// "Tattoo", "Nails", "Mehndi".
    static func traitNoun(_ kind: TraitKind) -> String {
        switch kind {
        case .tattoo: return "Tattoo"
        case .mehndi: return "Mehndi"
        case .nails: return "Nails"
        }
    }

    /// "blue nail art", "red glitter nails", "French nails", "nail polish".
    static func nailWords(_ t: Trait) -> String {
        let color = t.color?.lowercased() ?? ""
        switch t.finish {
        case "art": return color.isEmpty ? "nail art" : "\(color) nail art"
        case "glitter": return color.isEmpty ? "glitter nails" : "\(color) glitter nails"
        case "french": return "French nails"
        case "multi": return "multicoloured nails"
        default: return color.isEmpty ? "nail polish" : "\(color) nails"
        }
    }

    static func joinWords(_ words: [String]) -> String {
        guard words.count > 1, let last = words.last else { return words.joined() }
        return words.dropLast().joined(separator: ", ") + " and " + last
    }

    private static func roundedCount(_ v: Double?) -> Int {
        guard let v, v != 0 else { return 0 }
        return Int(v.rounded())
    }

    // MARK: Reasons

    /// The position reason: "In the same spot as Sam 8 s earlier, and 2 other people stayed in place".
    static func placeReason(_ guess: API.GuessOut) -> PlaceReason? {
        guard let p = place(guess.evidence) else { return nil }
        let who = isNamed(guess.person) ? displayName(guess.person) : "this person"
        let when = gapPhrase(p.secondsApart, before: p.refBefore)
        let hops = max(1, roundedCount(p.hops) == 0 ? 1 : roundedCount(p.hops))
        var text = hops > 1
            ? "Followed across \(hops) photos to \(who)\(when.isEmpty ? "" : ", \(when)")"
            : "In the same spot as \(who)\(when.isEmpty ? "" : " \(when)")"
        let others = roundedCount(p.others)
        if others > 0 {
            text += others == 1 ? ", and 1 other person stayed in place" : ", and \(others) other people stayed in place"
        }
        return PlaceReason(text: text, refCropUrl: p.refCropUrl, refItemUid: p.refItemUid, when: when)
    }

    /// One reason per positive trait entry, strongest first.
    static func traitReasons(_ guess: API.GuessOut) -> [TraitReason] {
        traits(guess.evidence).map { t in
            let spot = partWords(t.part)
            var text: String
            switch t.kind {
            case .tattoo:
                let sameDay = (t.daysApart ?? 99) < 1
                let when = t.refTakenAt.map { Fmt.date($0) } ?? ""
                text = sameDay
                    ? "Same tattoo on the \(spot) as in \(whose(guess.person)) photo taken the same day"
                    : "Same tattoo on the \(spot) as \(whose(guess.person)) photo\(when.isEmpty ? "" : " from \(when)")"
                let more = roundedCount(t.support)
                if more > 0 { text += more == 1 ? " and 1 other photo" : " and \(more) other photos" }
                if t.mirrored { text += ", probably a mirror photo" }
            case .mehndi:
                let days = daysPhrase(t.daysApart, before: t.refBefore)
                text = "Same mehndi on the \(spot)\(days.isEmpty ? "" : ", \(days)")"
                if t.mirrored { text += ", probably a mirror photo" }
            case .nails:
                let days = daysPhrase(t.daysApart, before: t.refBefore)
                let length = t.length.map { ", \($0)" } ?? ""
                text = "Same \(nailWords(t))\(length)\(days.isEmpty ? "" : ", \(days)")"
            }
            return TraitReason(text: text, kind: t.kind, part: t.part, cropUrl: t.cropUrl, refCropUrl: t.refCropUrl,
                               refItemUid: t.refItemUid, refTakenAt: t.refTakenAt, mirrored: t.mirrored, hex: t.hex)
        }
    }

    /// What counts against the guess: "No tattoo on the upper arm here, where Sam has one".
    static func againstReasons(_ guess: API.GuessOut) -> [String] {
        let who = displayName(guess.person)
        return against(guess.evidence).map { a in
            let spot = partWords(a.part)
            let noun: String
            switch a.kind {
            case .tattoo: noun = "tattoo"
            case .mehndi: noun = "mehndi"
            case .nails: noun = "nail polish"
            }
            return a.why == "bare"
                ? "No \(noun) on the \(spot) here, where \(who) has one"
                : "No matching \(noun) on the \(spot)"
        }
    }

    /// The trait kinds a guess shows, the leading kind first: tattoo, then mehndi; nails never lead.
    static func traitKinds(_ e: Evidence) -> [TraitKind] {
        let seen = Set(traits(e).map(\.kind))
        return [TraitKind.tattoo, .mehndi, .nails].filter { seen.contains($0) }
    }

    /// "Face", "By clothes", "By tattoo", "By mehndi and nails", "By clothes and tattoo", "By face", "By position",
    /// "By clothes and position", "Grouped".
    static func label(isFace: Bool, guess: API.GuessOut) -> String {
        let e = guess.evidence
        let b = basis(e)
        if b == .face { return "By face" }
        if b == .group { return "Grouped" }
        let kinds = traitKinds(e).map(\.rawValue)
        let position = place(e) != nil ? ["position"] : []
        if b == .place { return "By position" }
        if b == .clothesPlace { return "By clothes and position" }
        if b == .trait && !kinds.isEmpty { return "By \(joinWords(kinds + position))" }
        if b == .clothesTrait && !kinds.isEmpty { return "By \(joinWords(["clothes"] + kinds + position))" }
        if b == .trait { return "By tattoo" }
        return isFace ? "Face" : "By clothes"
    }

    private static func alsoParts(_ e: Evidence) -> (clothes: Bool, trait: Bool, place: Bool) {
        let parts = (e["also"]?.stringValue ?? "").split(separator: "+").map(String.init)
        return (parts.contains("clothes"), parts.contains("trait"), parts.contains("place"))
    }

    /// The icons that go with the label: clothes, a tattoo, a hand (mehndi or nails), a face (trained recognition),
    /// footprints (position).
    static func icons(isFace: Bool, guess: API.GuessOut) -> [Icon] {
        let e = guess.evidence
        let b = basis(e)
        if b == .group { return [.face] }
        if b == .face {
            let also = alsoParts(e)
            var out: [Icon] = [.face]
            if also.clothes { out.append(.clothes) }
            if also.place || place(e) != nil { out.append(.place) }
            return out
        }
        let kinds = traitKinds(e)
        var out: [Icon] = []
        let clothes = b == .clothes ? !isFace : (b == .clothesTrait || b == .clothesPlace)
        if clothes { out.append(.clothes) }
        if kinds.contains(.tattoo) { out.append(.tattoo) } else if !kinds.isEmpty { out.append(.hand) }
        if place(e) != nil { out.append(.place) }
        return out
    }

    /// "85% sure" for a clothes guess, trained recognition and the position alone; "strong match" for traits
    /// alone (no percentage); "face 0.62" for a grouped face (its similarity, not a probability).
    static func sure(_ guess: API.GuessOut) -> String {
        switch basis(guess.evidence) {
        case .trait: return "strong match"
        case .group: return "face \(twoDigits(guess.score.isFinite ? guess.score : 0))"
        default: return "\(Fmt.percent(guess.score)) sure"
        }
    }

    static func twoDigits(_ v: Double) -> String { String(format: "%.2f", v) }

    private static func placeWords(_ p: Place) -> (hops: Int, when: String) {
        let hops = roundedCount(p.hops) == 0 ? 1 : roundedCount(p.hops)
        return (hops, gapPhrase(p.secondsApart, before: p.refBefore))
    }

    /// "same spot 8 s earlier", "followed across 3 photos".
    private static func placePhrase(_ p: Place) -> String {
        let w = placeWords(p)
        return "\(w.hops > 1 ? "followed across \(w.hops) photos" : "same spot")\(w.when.isEmpty ? "" : " \(w.when)")"
    }

    /// A guess of recognition trained on the person: "Looks like Sam's photos, 0.61" and what else names them.
    /// With the position part, the position has a reason of its own, so it is left out unless `withPlace`.
    static func faceLine(_ guess: API.GuessOut, withPlace: Bool = false) -> String {
        let e = guess.evidence
        let face = num(e["face"])
        let also = alsoParts(e)
        var same: [String] = []
        if also.clothes { same.append("clothes") }
        if also.trait { same.append("tattoo or nails") }
        var followed = ""
        if let p = place(e) {
            if withPlace {
                let w = placeWords(p)
                if w.hops > 1 { followed = ", \(placePhrase(p))" } else { same.append("spot\(w.when.isEmpty ? "" : " \(w.when)")") }
            }
        } else if also.place {
            same.append("spot")
        }
        var text = "Looks like \(whose(guess.person)) photos\(face.map { ", \(twoDigits($0))" } ?? "")"
        if !same.isEmpty { text += ", and the same \(joinWords(same))" }
        return text + followed
    }

    /// The line under a guess: the time window, scene, place, clothes and face of a clothes guess; faceLine for
    /// trained recognition; "trained face 92%" or nothing for traits alone or the position alone; "grouped with
    /// the face you named Sam" for a grouped face.
    static func clothesLine(_ guess: API.GuessOut) -> String {
        let e = guess.evidence
        switch basis(e) {
        case .group:
            return isNamed(guess.person) ? "grouped with the face you named \(displayName(guess.person))"
                : "grouped with a face you named"
        case .trait, .place:
            if let agrees = num(e["p"]) { return "trained face \(Fmt.percent(agrees))" }
            return ""
        case .face:
            return faceLine(guess)
        default:
            var parts: [String] = []
            if let minutes = num(e["minutes_apart"]) {
                parts.append(minutes < 1 ? "same minute" : "\(Int(minutes.rounded())) min apart")
            } else if e["same_day"]?.boolValue == true {
                parts.append("same day")
            } else if e["same_day"]?.boolValue == false {
                parts.append("within hours")
            }
            if let scene = num(e["scene"]) { parts.append("same scene \(twoDigits(scene))") }
            if let gps = num(e["gps_m"]) { parts.append("\(Int(gps.rounded())) m apart") }
            if let clothes = num(e["clothes"]) { parts.append("same clothes \(twoDigits(clothes))") }
            if let face = num(e["face"]), face >= 0.2 { parts.append("face \(twoDigits(face))") }
            if let trained = num(e["p"]) { parts.append("trained face \(Fmt.percent(trained))") }
            return parts.joined(separator: ", ")
        }
    }

    /// "same tattoo on the forearm", "same red nails 2 days earlier".
    private static func traitPhrases(_ e: Evidence) -> [String] {
        traits(e).map { t in
            if t.kind == .tattoo { return "same tattoo on the \(partWords(t.part))" }
            let days = daysPhrase(t.daysApart, before: t.refBefore)
            let what = t.kind == .mehndi ? "same mehndi on the \(partWords(t.part))" : "same \(nailWords(t))"
            return days.isEmpty ? what : "\(what) \(days)"
        }
    }

    /// The whole evidence as one line: trait phrases first, then the position, then the clothes parts.
    static func evidenceLine(_ guess: API.GuessOut) -> String {
        var parts = traitPhrases(guess.evidence)
        if basis(guess.evidence) == .face {
            parts.append(faceLine(guess, withPlace: true))
            return parts.joined(separator: ", ")
        }
        if let p = place(guess.evidence) { parts.append(placePhrase(p)) }
        let clothes = clothesLine(guess)
        if !clothes.isEmpty { parts.append(clothes) }
        return parts.joined(separator: ", ")
    }

    /// The item the guess was matched against, when the evidence names it.
    static func evidenceItemUid(_ guess: API.GuessOut) -> String? {
        guess.evidence["ref_item_uid"]?.stringValue
    }
}

/// Why an item waits in a review bin, as the tiles show it (the website's components/library/misclick.ts and
/// forwards.ts).
enum ReviewItemReasons {
    /// Mis-click reasons in a stable order.
    static let misclickLabels: [(key: String, label: String)] = [
        ("blurry", "Blurry"), ("dark", "Too dark"), ("blown_out", "Blown out"), ("pocket", "Pocket shot"),
        ("burst", "Burst copy"),
    ]

    /// The labels of an item's mis-click reasons (`{reason: detail}`, or `{reasons: [...]}`).
    static func misclick(_ reasons: [String: API.JSONValue]?) -> [String] {
        guard let reasons else { return [] }
        let keys: [String]
        if let list = reasons["reasons"]?.arrayValue {
            keys = list.compactMap { $0.stringValue }
        } else {
            let order = misclickLabels.map { $0.key }
            keys = reasons.keys.filter { $0 != "burst_best" && $0 != "best_uid" }.sorted { a, b in
                let ia = order.firstIndex(of: a) ?? 99, ib = order.firstIndex(of: b) ?? 99
                return ia == ib ? a < b : ia < ib
            }
        }
        return keys.map(misclickLabel)
    }

    static func misclickLabel(_ key: String) -> String {
        if let known = misclickLabels.first(where: { $0.key == key }) { return known.label }
        let words = key.replacingOccurrences(of: "_", with: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }

    static let appLabels = ["whatsapp": "WhatsApp", "telegram": "Telegram", "facebook": "Facebook",
                            "snapchat": "Snapchat", "messenger": "Messenger", "app": "a messaging app"]

    static func appLabel(_ via: String) -> String { appLabels[via] ?? via }

    /// The reasons an item was set apart as a forward, in a stable order: "Meme 0.81", "Text",
    /// "Small, no camera data", "JPEG quality 75", "Filename from WhatsApp".
    static func forward(_ reasons: [String: API.JSONValue]?) -> [String] {
        guard let r = reasons else { return [] }
        var out: [String] = []
        if let meme = ReviewEvidence.num(r["meme_p"]) { out.append("Meme \(ReviewEvidence.twoDigits(meme))") }
        if truthy(r["text"]) { out.append("Text") }
        let small = truthy(r["small"]), stripped = truthy(r["stripped"])
        if small && stripped {
            out.append("Small, no camera data")
        } else if small {
            out.append("Small")
        } else if stripped {
            out.append("No camera data")
        }
        if truthy(r["video_small"]) { out.append("Small video") }
        if let q = ReviewEvidence.num(r["jpeg_quality"]) { out.append("JPEG quality \(Int(q.rounded()))") }
        if let name = r["name"]?.stringValue, !name.isEmpty { out.append("Filename from \(appLabel(name))") }
        return out
    }

    /// JavaScript's truthiness for a JSON value.
    static func truthy(_ v: API.JSONValue?) -> Bool {
        guard let v else { return false }
        switch v {
        case .null: return false
        case .bool(let b): return b
        case .number(let n): return n != 0 && !n.isNaN
        case .string(let s): return !s.isEmpty
        case .array, .object: return true
        }
    }
}
