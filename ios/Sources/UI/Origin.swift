import SwiftUI

/// Who put a person or pet on a box, an item or a selection (docs/ARCHITECTURE.md section 10, the website's
/// lib/origin.ts): the owner (a tag, a box they tagged or drew) or the system (recognition, grouping, a guess).
/// Naming a group or merging people does not make its faces the owner's.
enum BoxOrigin: String, CaseIterable {
    /// Named by the owner.
    case you
    /// A box the owner drew, with someone the system matched on it.
    case drawn
    /// A guess the owner said yes to: still the system's find.
    case confirmed
    /// Found by the system.
    case auto
    /// The system suggests someone and waits for an answer.
    case guess

    /// The website's words for it (ORIGIN_LABEL).
    var label: String {
        switch self {
        case .you: return "Tagged by you"
        case .drawn: return "Drawn by you"
        case .confirmed: return "Guessed automatically, confirmed by you"
        case .auto: return "Tagged automatically"
        case .guess: return "Suggested automatically"
        }
    }

    /// The owner's origins show a person icon, the rest the system's icon.
    var isMine: Bool { self == .you || self == .drawn }

    /// Who put the person (or the guess) on one box; nil for a box the detector found with nobody on it (the
    /// website's boxOrigin). `assignedByUser`: `assigned_by == "user"`; `drawnByUser`: `source == "user"`;
    /// `byGuess`: the person came from a guess the owner confirmed.
    static func of(hasPerson: Bool, assignedByUser: Bool, drawnByUser: Bool = false, byGuess: Bool = false,
                   hasGuess: Bool = false) -> BoxOrigin? {
        if hasPerson && assignedByUser && !byGuess { return .you }
        if hasPerson && drawnByUser { return .drawn }
        if hasPerson && assignedByUser { return .confirmed }
        if hasPerson { return .auto }
        if hasGuess { return .guess }
        if drawnByUser { return .drawn }
        return nil
    }
}

/// A person icon for the owner's origins and the system's icon for the rest (the website's OriginIcon), read by
/// VoiceOver as the origin's words unless `decorative` (inside a control whose label already says them).
struct OriginIcon: View {
    let origin: BoxOrigin
    var label: String? = nil
    var decorative = false

    var body: some View {
        Image(systemName: origin.isMine ? "person.fill" : "cpu")
            .imageScale(.small)
            .accessibilityLabel(label ?? origin.label)
            .accessibilityHidden(decorative)
    }
}
