import SwiftUI
import Observation

/// Whether the vault is open for this app, and until when (the website's useAuth vault state, from the auth state
/// the server answers). The app signs in with a session of its own, so the vault opened in a browser does not open
/// it here. Every vault screen reads this; unlocking and locking go through it.
@MainActor
@Observable
final class VaultSession {
    static let shared = VaultSession()

    /// When the vault closes again; nil while it is locked.
    private(set) var until: Date?
    /// Passkeys on the account: the website offers them for unlocking.
    private(set) var passkeys = 0
    /// The auth state was asked for at least once.
    private(set) var loaded = false
    /// The last time asking failed, while nothing is known yet.
    private(set) var loadError: Error?

    /// Open now, on this phone's clock.
    func isOpen(at now: Date = Date()) -> Bool {
        VaultClock.secondsLeft(until: until, now: now) > 0
    }

    /// Asks the server whether the vault is open for this app.
    func refresh() async {
        do {
            apply(try await Cloud.api().getState())
            loadError = nil
        } catch {
            if error is CancellationError || Task.isCancelled { return }
            if !loaded { loadError = error }
        }
        loaded = true
    }

    func apply(_ state: API.AuthState) {
        until = ISODate.parse(state.vaultUntil)
        passkeys = state.passkeys
        loaded = true
        loadError = nil
        // A locked vault leaves no picture of it in memory either.
        if !isOpen() { ImageLoader.shared.forgetVault() }
        scheduleExpiry()
    }

    @ObservationIgnored private var expiry: Task<Void, Never>?

    /// When the window ends on this phone's clock, the vault reads as locked and its pictures leave memory, wherever
    /// the owner is in the app (the server sets the window once, at unlocking, and never extends it).
    private func scheduleExpiry() {
        expiry?.cancel()
        expiry = nil
        guard let end = until, end > Date() else { return }
        expiry = Task { [weak self] in
            try? await Task.sleep(for: .seconds(end.timeIntervalSinceNow))
            guard !Task.isCancelled, let self, self.until == end else { return }
            self.expire()
        }
    }

    /// Opens the vault with the account's password. Throws the server's answer (401 for a wrong password).
    func unlock(password: String) async throws {
        var body = API.VaultUnlockIn()
        body.password = password
        apply(try await Cloud.api().vaultUnlock(body: body))
        CloudEvents.post()
    }

    /// Locks the vault at once.
    func lock() async throws {
        apply(try await Cloud.api().vaultLock())
        CloudEvents.post()
    }

    /// The window ran out on this phone's clock and the server did not say otherwise: read it as locked.
    func expire() {
        until = nil
        expiry?.cancel()
        expiry = nil
        ImageLoader.shared.forgetVault()
    }
}

/// The vault window's arithmetic and words, apart from the screens so they can be tested.
enum VaultClock {
    /// Whole seconds left in the window; 0 once it has ended, or when there is none.
    static func secondsLeft(until: Date?, now: Date) -> Int {
        guard let until else { return 0 }
        return max(0, Int(until.timeIntervalSince(now).rounded(.down)))
    }

    /// "Unlocked for 14 min" (the website's vault page); "Unlocked" when the end is not known.
    static func openText(secondsLeft: Int?) -> String {
        guard let s = secondsLeft, s > 0 else { return "Unlocked" }
        return "Unlocked for \(Fmt.minutes(Double(s) / 60))"
    }
}

/// The vault's words that more than one screen shows.
enum VaultText {
    static let lockedTitle = "The vault is locked"
    static let lockedText = "Private items are kept out of the library, search and people. Confirm it is you to see them "
        + "for a while."
    static let sheetText = "This needs the vault open. Confirm it is you with your password."

    /// What the unlock form says when the server refuses: a wrong password, or anything else.
    static func unlockError(_ error: Error) -> String {
        if let api = error as? APIError, api.status == 401 { return "That password is not right." }
        return ErrorText.describe(error, doing: "unlock")
    }
}
