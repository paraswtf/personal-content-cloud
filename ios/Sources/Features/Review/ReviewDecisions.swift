import SwiftUI

/// Answers to guesses and the review bins' actions (the website's useGuessReview and the bins' bulk actions). Each
/// is one owner action: it joins the undo history, and the message it shows offers Undo.
@MainActor
enum ReviewDecisions {
    enum Action: Hashable {
        case confirm, reject, correct
    }

    /// Runs one owner action and shows `message(answer)`, with Undo when the server changed something.
    @discardableResult
    static func run<T>(_ label: String, message: (T) -> String, _ body: () async throws -> T) async throws -> T {
        let center = UndoCenter.shared
        let before = center.entries.first?.id
        let value = try await center.perform(label, toast: false, body)
        let text = message(value)
        if let entry = center.entries.first, entry.id != before {
            let id = entry.id
            Toaster.shared.ok(text, action: .init(label: "Undo") { Task { await UndoCenter.shared.undo(id) } })
        } else {
            Toaster.shared.ok(text)
        }
        return value
    }

    private static func boxIds(_ items: [API.GuessItem]) -> (faces: [Int], bodies: [Int]) {
        (items.filter { $0.kind == .face }.map(\.id), items.filter { $0.kind == .body }.map(\.id))
    }

    /// Yes: tags each box as its guessed person.
    @discardableResult
    static func confirm(_ items: [API.GuessItem]) async throws -> API.BulkOut {
        let ids = boxIds(items)
        let api = try Cloud.api()
        let single = items.count == 1 ? ReviewEvidence.displayName(items[0].guess.person) : nil
        let label = single.map { "Tagged as \($0)." } ?? "Confirmed \(Fmt.count(items.count, "guess", "guesses"))."
        return try await run(label, message: { (out: API.BulkOut) in
            single.map { "Tagged as \($0)." } ?? "\(Fmt.count(out.updated, "guess", "guesses")) confirmed."
        }) {
            try await api.confirmGuesses(body: API.GuessIdsIn(faces: ids.faces, bodies: ids.bodies))
        }
    }

    /// No: the guess stops, and the app learns it was wrong.
    @discardableResult
    static func reject(_ items: [API.GuessItem]) async throws -> API.BulkOut {
        let ids = boxIds(items)
        let api = try Cloud.api()
        let label = items.count == 1 ? "Guess rejected." : "Rejected \(Fmt.count(items.count, "guess", "guesses"))."
        return try await run(label, message: { (out: API.BulkOut) in
            items.count == 1 ? "Guess rejected." : "\(Fmt.count(out.updated, "guess", "guesses")) rejected."
        }) {
            try await api.rejectGuesses(body: API.GuessIdsIn(faces: ids.faces, bodies: ids.bodies))
        }
    }

    /// Someone else: the guess is learned as wrong and the box goes to the person picked, in one step.
    @discardableResult
    static func correct(_ items: [API.GuessItem], to pick: PersonPick) async throws -> API.AssignOut {
        let ids = boxIds(items)
        let api = try Cloud.api()
        var body = API.GuessCorrectIn(faces: ids.faces, bodies: ids.bodies)
        let picked: String
        switch pick {
        case .existing(let option):
            body.personId = option.id
            picked = option.name
        case .new(let name, _):
            body.newPersonName = name
            picked = name
        }
        return try await run("Tagged as \(picked).", message: { (out: API.AssignOut) in
            let name = out.person.map { Fmt.personName($0.name) } ?? picked
            return "Tagged as \(name).\(Fmt.guessedTail(out.guessed))"
        }) {
            try await api.correctGuesses(body: body)
        }
    }

    /// Keep (mis-clicks), keep in the library (forwards) or move to the trash, for the selected items.
    @discardableResult
    static func bulk(_ uids: [String], action: API.Action, message: @escaping (Int) -> String) async throws -> API.BulkOut {
        let api = try Cloud.api()
        return try await run(message(uids.count), message: { (out: API.BulkOut) in message(out.updated) }) {
            try await api.bulk(body: API.BulkIn(uids: uids, action: action))
        }
    }
}
