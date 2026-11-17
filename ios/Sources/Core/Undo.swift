import Foundation
import Observation

/// One owner action in progress (docs/ARCHITECTURE.md section 33). Every write sent while it is the task's
/// current scope carries its id in `X-Action-Id` (the core's own requests read `ActionScope.current`, the generated
/// calls read `APIContext.actionID`, and `UndoCenter.perform` sets both), so a loop over many items undoes as one.
/// The server answers with the same header when the action changed something.
final class ActionScope: @unchecked Sendable {
    @TaskLocal static var current: ActionScope?

    let id: String
    let label: String
    private let lock = NSLock()
    private var changedFlag = false

    private static let registryLock = NSLock()
    private static var active: [String: ActionScope] = [:]

    init(label: String) {
        // 32 characters of [0-9a-f]: inside the server's [A-Za-z0-9_-]{8,64}.
        id = APIContext.newActionID()
        self.label = label
    }

    /// True once the server said this action changed something.
    var changed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return changedFlag
    }

    /// Reads the answer of one of this action's requests (the core's own requests).
    func note(_ response: HTTPURLResponse) {
        if response.value(forHTTPHeaderField: "X-Action-Id") == id { markChanged() }
    }

    func markChanged() {
        lock.lock()
        changedFlag = true
        lock.unlock()
    }

    /// While registered, answers of the generated calls that name this action mark it changed.
    func register() {
        Self.registryLock.lock()
        Self.active[id] = self
        Self.registryLock.unlock()
    }

    func unregister() {
        Self.registryLock.lock()
        Self.active[id] = nil
        Self.registryLock.unlock()
    }

    /// A generated call's answer named action `id` (`.apiActionRecorded`): true when it belongs to a running scope.
    @discardableResult
    static func noteRecorded(_ id: String) -> Bool {
        registryLock.lock()
        let scope = active[id]
        registryLock.unlock()
        scope?.markChanged()
        return scope != nil
    }
}

/// What `POST /api/actions/{id}/undo` answers.
struct UndoResult: Decodable {
    let undoId: String
    let changes: Int
    let partial: Bool
    let items: [String]
}

/// The owner's undo history (docs/ARCHITECTURE.md sections 14 and 33). The server keeps what each action
/// changed; this phone keeps the list of actions it made, newest first, at most 200, labelled by the message
/// the app showed. An undo is an action of its own, so undoing it again is a redo.
@MainActor
@Observable
final class UndoCenter {
    static let shared = UndoCenter()

    struct Entry: Codable, Identifiable, Hashable {
        enum State: String, Codable { case done, undone }
        /// The action id the change was made with.
        let id: String
        let label: String
        let at: Date
        var state: State
        /// The server action whose effect is live now: undo this one.
        var activeId: String
        /// After an undo: the undo's own action id, undone again for a redo.
        var undoId: String?
        var undoneAt: Date?
        /// The server no longer keeps it (older than 30 days, or unknown there).
        var expired: Bool = false
        /// The last undo or redo left some of it as it had become since.
        var partial: Bool = false
    }

    static let maxEntries = 200
    private(set) var entries: [Entry] = []
    @ObservationIgnored private var busy: Set<String> = []
    @ObservationIgnored private let storageKey = "pcc.undo.v1"

    init() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let saved = try? JSONDecoder().decode([Entry].self, from: data) {
            entries = Array(saved.prefix(Self.maxEntries))
        }
        // Writes of the generated calls: inside a running action they mark it changed; outside any action (a
        // screen that called the API directly) they still join the history, without a message.
        _ = NotificationCenter.default.addObserver(forName: .apiActionRecorded, object: nil, queue: nil) { note in
            guard let id = note.userInfo?["id"] as? String else { return }
            if ActionScope.noteRecorded(id) { return }
            let method = note.userInfo?["method"] as? String ?? ""
            let path = note.userInfo?["path"] as? String ?? ""
            Task { @MainActor in UndoCenter.shared.recordLoose(id: id, method: method, path: path) }
        }
    }

    /// Runs one owner action. Every write inside carries one action id; when the server says it changed
    /// something, the action joins the history and, with `toast`, a message offers Undo. Errors are rethrown
    /// for the caller to show.
    @discardableResult
    func perform<T>(_ label: String, toast: Bool = true, _ body: () async throws -> T) async throws -> T {
        let scope = ActionScope(label: label)
        scope.register()
        defer { scope.unregister() }
        do {
            let value = try await ActionScope.$current.withValue(scope) {
                try await APIContext.$actionID.withValue(scope.id) { try await body() }
            }
            if scope.changed { record(scope, toast: toast) }
            return value
        } catch {
            if scope.changed { record(scope, toast: false) }
            throw error
        }
    }

    /// A write made outside `perform` that the server recorded: it joins the history with a plain label.
    func recordLoose(id: String, method: String, path: String) {
        guard !entries.contains(where: { $0.id == id || $0.activeId == id || $0.undoId == id }) else { return }
        entries.insert(Entry(id: id, label: Self.describe(method: method, path: path), at: Date(), state: .done,
                             activeId: id), at: 0)
        trimAndSave()
    }

    /// A plain label for a write nobody named (the website's describe() in lib/undo.ts, shortened).
    nonisolated static func describe(method: String, path: String) -> String {
        let parts = path.split(separator: "/").map(String.init)
        let area = parts.count > 1 ? parts[1] : ""
        switch (method, area) {
        case ("DELETE", "items"): return "Moved an item to the trash"
        case (_, "items"): return "Changed an item"
        case (_, "faces"), (_, "bodies"): return "Changed who is in a photo"
        case (_, "animals"): return "Changed a pet"
        case ("DELETE", "people"): return "Removed a person"
        case (_, "people"): return "Changed a person"
        case (_, "guesses"): return "Answered a guess"
        case (_, "albums"): return "Changed an album"
        case (_, "shares"): return "Changed a share"
        case (_, "tags"): return "Changed a tag"
        case (_, "documents"): return "Changed a document"
        default: return "Made a change"
        }
    }

    /// Adds a finished action that changed something, for code that sent its own X-Action-Id.
    func record(_ scope: ActionScope, toast: Bool) {
        entries.removeAll { $0.id == scope.id }
        entries.insert(Entry(id: scope.id, label: scope.label, at: Date(), state: .done, activeId: scope.id), at: 0)
        trimAndSave()
        if toast {
            let id = scope.id
            Toaster.shared.ok(scope.label, action: .init(label: "Undo") { Task { await UndoCenter.shared.undo(id) } })
        }
    }

    /// Undoes an entry and offers Redo.
    func undo(_ entryId: String) async {
        guard let e = entries.first(where: { $0.id == entryId }), e.state == .done, !e.expired,
              !busy.contains(entryId), let server = APIConnection.current else { return }
        busy.insert(entryId)
        defer { busy.remove(entryId) }
        do {
            let out = try await CoreHTTP.send(UndoResult.self, server, "POST", "/api/actions/\(e.activeId)/undo")
            update(entryId) {
                $0.state = .undone
                $0.undoId = out.undoId
                $0.undoneAt = Date()
                $0.partial = out.partial
            }
            CloudEvents.post(items: out.items)
            let tail = out.partial ? " Some of it had changed again since and stays as it is." : ""
            Toaster.shared.ok("Undone: \(ErrorText.sentence(e.label))\(tail)",
                              action: .init(label: "Redo") { Task { await UndoCenter.shared.redo(entryId) } })
        } catch {
            explain(error, entryId: entryId, redo: false)
        }
    }

    /// Redoes an undone entry (undoes its undo) and offers Undo again.
    func redo(_ entryId: String) async {
        guard let e = entries.first(where: { $0.id == entryId }), e.state == .undone, let undoId = e.undoId,
              !e.expired, !busy.contains(entryId), let server = APIConnection.current else { return }
        busy.insert(entryId)
        defer { busy.remove(entryId) }
        do {
            let out = try await CoreHTTP.send(UndoResult.self, server, "POST", "/api/actions/\(undoId)/undo")
            update(entryId) {
                $0.state = .done
                $0.activeId = out.undoId
                $0.undoId = nil
                $0.undoneAt = nil
                $0.partial = out.partial
            }
            CloudEvents.post(items: out.items)
            let tail = out.partial ? " Some of it had changed again since and stays as it is." : ""
            Toaster.shared.ok("Redone: \(ErrorText.sentence(e.label))\(tail)",
                              action: .init(label: "Undo") { Task { await UndoCenter.shared.undo(entryId) } })
        } catch {
            explain(error, entryId: entryId, redo: true)
        }
    }

    /// Forgets the whole history (sign-out).
    func clear() {
        entries = []
        UserDefaults.standard.removeObject(forKey: storageKey)
    }

    private func update(_ id: String, _ change: (inout Entry) -> Void) {
        guard let i = entries.firstIndex(where: { $0.id == id }) else { return }
        change(&entries[i])
        trimAndSave()
    }

    private func trimAndSave() {
        if entries.count > Self.maxEntries { entries = Array(entries.prefix(Self.maxEntries)) }
        if let data = try? JSONEncoder().encode(entries) {
            UserDefaults.standard.set(data, forKey: storageKey)
        }
    }

    /// The server's refusal as a plain sentence, with the same wording as the website (lib/undoRun.ts).
    private func explain(_ error: Error, entryId: String, redo: Bool) {
        if error is CancellationError { return }
        guard let api = error as? APIError, api.status != 0 else {
            Toaster.shared.error(redo ? "Could not redo it. Check the connection and try again."
                                      : "Could not undo it. Check the connection and try again.")
            return
        }
        switch (api.status, api.detail) {
        case (409, "later_changes"):
            Toaster.shared.error("Changes made since touch the same things. Undo those first.")
        case (409, "already_undone"):
            update(entryId) { $0.expired = redo }
            if !redo { update(entryId) { $0.state = .undone } }
            Toaster.shared.error(redo ? "That was redone already, maybe on another device."
                                      : "That was undone already, maybe on another device.")
        case (410, _), (404, _):
            update(entryId) { $0.expired = true }
            Toaster.shared.error("That change is too old to undo. Changes can be undone for 30 days.")
        case (403, _):
            Toaster.shared.error("Unlock the vault to undo this.")
        case (401, _):
            Toaster.shared.error("Sign in again to undo this.")
        case (409, _):
            Toaster.shared.error("It cannot be undone now: something it would bring back is in the way, for example a name used since.")
        default:
            Toaster.shared.error(ErrorText.sentence(api.detail).isEmpty ? "Could not undo it. Try again." : ErrorText.sentence(api.detail))
        }
    }
}
