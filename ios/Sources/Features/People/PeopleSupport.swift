import SwiftUI

// What the people screens share: rows for the lists (the API types are not Identifiable), the website's wording
// (lib/format.ts, components/people/PeopleBulkAction.tsx) and the way an owner action is run and announced.

/// A person or pet in a list.
struct PeopleRow: Identifiable, Hashable {
    let person: API.PersonOut
    var id: Int { person.id }
}

/// A face in a grid.
struct PeopleFaceRow: Identifiable, Hashable {
    let face: API.FaceListItem
    var id: Int { face.id }
}

/// A pet's crop in a grid.
struct PeopleCropRow: Identifiable, Hashable {
    let crop: API.AnimalListItem
    var id: Int { crop.id }
}

/// A guess in a list (a face and a person box can have the same id).
struct PeopleGuessRow: Identifiable, Hashable {
    let guess: API.GuessItem
    var id: String { "\(guess.kind.rawValue)-\(guess.id)" }
}

/// The people the merge sheet chooses from.
struct PeopleMergeRequest: Identifiable {
    let id = UUID()
    let people: [API.PersonOut]
}

/// A picked person as the API's fields: an existing person, or a new name (with its kind).
struct PeopleChoice {
    var personId: Int?
    var newName: String?
    var newKind: String?

    init(_ pick: PersonPick) {
        switch pick {
        case .existing(let option):
            personId = option.id
        case .new(let name, let kind):
            newName = name
            newKind = kind
        }
    }
}

enum PeopleText {
    /// A person's or pet's name for display ("Unnamed cat" for a pet nobody named).
    static func name(_ p: API.PersonOut) -> String {
        Fmt.personName(p.name, kind: p.kind.rawValue, species: p.species)
    }

    static func isPet(_ p: API.PersonOut) -> Bool { p.kind == .pet }

    /// "Asha", or "Unnamed person" / "Unnamed pet" (the selection sheet's wording).
    static func display(_ name: String?, kind: String) -> String {
        if Fmt.isNamed(name), let name { return name }
        return kind == "pet" ? "Unnamed pet" : "Unnamed person"
    }

    /// "Not Asha", or "Not this person" / "Not this pet".
    static func notLabel(_ name: String?, kind: String) -> String {
        if Fmt.isNamed(name), let name { return "Not \(name)" }
        return kind == "pet" ? "Not this pet" : "Not this person"
    }

    /// Who a "Not ..." is about inside a sentence: "Asha", or "this person" / "this pet".
    static func notWho(_ name: String?, kind: String) -> String {
        if Fmt.isNamed(name), let name { return name }
        return kind == "pet" ? "this pet" : "this person"
    }

    /// "this item" or "these 3 items".
    static func itemsTail(_ n: Int) -> String {
        n == 1 ? "this item" : "these \(n.formatted()) items"
    }

    /// " 3 look-alike faces followed. 2 look-alike groups joined them." after a move or a removal on items.
    static func bulkTail(_ r: API.PeopleBulkOut, pet: Bool) -> String {
        let followed = r.followed > 0
            ? " \(Fmt.count(r.followed, pet ? "look-alike crop" : "look-alike face")) followed." : ""
        let pulled = r.pulled > 0 ? " \(Fmt.count(r.pulled, "look-alike group")) joined them." : ""
        return followed + pulled
    }

    /// "cat" -> "cats" for "These are cats".
    static func speciesPlural(_ species: String?) -> String {
        "\(Fmt.speciesLabel(species).lowercased())s"
    }

    /// The website's words for who put a box or a crop there, in a sentence ("tagged by you").
    static func originPhrase(_ origin: BoxOrigin) -> String {
        let label = origin.label
        return label.prefix(1).lowercased() + label.dropFirst()
    }
}

extension BoxOrigin {
    /// Who put the person (or the guess) on a face of a list.
    static func of(face f: API.FaceListItem) -> BoxOrigin? {
        BoxOrigin.of(hasPerson: f.personId != nil, assignedByUser: f.assignedBy == .user,
                     drawnByUser: f.source == .user, byGuess: f.byGuess, hasGuess: f.guess != nil)
    }

    /// Who put the pet on a crop of a list.
    static func of(crop a: API.AnimalListItem) -> BoxOrigin? {
        BoxOrigin.of(hasPerson: a.personId != nil, assignedByUser: a.assignedBy == .user, drawnByUser: a.source == .user)
    }
}

/// Runs the people screens' owner actions. The undo history keeps `label`; the message, which often depends on
/// the server's answer (how many faces, who they went to, the guesses it made), shows with Undo when the server
/// changed something. Errors show as "Could not <doing>." Screens that show people load again afterwards.
@MainActor
enum PeopleAction {
    @discardableResult
    static func run<T>(_ label: String, doing: String, items: [String] = [], vault: Bool = false,
                       _ body: () async throws -> T, message: (T) -> String?) async -> T? {
        let before = UndoCenter.shared.entries.first?.id
        do {
            let value: T
            if vault {
                value = try await APIContext.$vault.withValue(true) {
                    try await UndoCenter.shared.perform(label, toast: false, body)
                }
            } else {
                value = try await UndoCenter.shared.perform(label, toast: false, body)
            }
            let after = UndoCenter.shared.entries.first?.id
            if let text = message(value) {
                if let after, after != before {
                    Toaster.shared.ok(text, action: Toaster.ToastAction(label: "Undo") {
                        Task { await UndoCenter.shared.undo(after) }
                    })
                } else {
                    Toaster.shared.ok(text)
                }
            }
            CloudEvents.post(items: items)
            return value
        } catch {
            Toaster.shared.error(error, doing: doing)
            return nil
        }
    }
}

/// Opens items in the viewer from a list that is not of items (faces, crops, guesses): each item once, in order.
@MainActor
enum PeopleViewer {
    static func open(_ router: AppRouter, uids: [String], thumbs: [String: String] = [:], start: String) {
        var seen = Set<String>()
        let unique = uids.filter { seen.insert($0).inserted }
        guard !unique.isEmpty else { return }
        let entries = unique.map { ViewerEntry(uid: $0, thumb: thumbs[$0]) }
        router.viewer = ViewerRequest(entries: entries, start: unique.firstIndex(of: start) ?? 0)
    }
}

/// A small heading with a line under it, as the website's section headings.
struct PeopleSectionHeader: View {
    let title: String
    var systemImage: String?
    var note: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let systemImage {
                Label(title, systemImage: systemImage).font(.headline)
            } else {
                Text(title).font(.headline)
            }
            if let note {
                Text(note).font(.caption).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }
}
