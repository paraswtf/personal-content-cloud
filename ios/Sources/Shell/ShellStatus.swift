import SwiftUI
import Observation

/// What the shell shows on every screen: the cloud's name, how many items wait in Review (the website's Nav count)
/// and, through VaultSession, whether the vault is open (the website's header badge). Read when the app opens,
/// when it comes back to the front, after a change on the server and when the owner changes tabs.
@MainActor
@Observable
final class ShellStatus {
    static let shared = ShellStatus()

    /// The server's site title ("Content Cloud" until it is read).
    private(set) var siteTitle: String?
    /// Mis-clicks plus guesses waiting in Review (the website's useReviewCount).
    private(set) var reviewWaiting = 0

    @ObservationIgnored private var statsAt: Date?

    /// The auth state (the vault's window and the cloud's name) and the Review count. Failures stay quiet: the
    /// screens that need these say what went wrong themselves.
    func refresh(force: Bool = true) async {
        await refreshState()
        await refreshStats(force: force)
    }

    private func refreshState() async {
        guard let state = try? await Cloud.api().getState() else { return }
        VaultSession.shared.apply(state)
        siteTitle = state.siteTitle
    }

    /// The Review count, at most once in 15 seconds unless `force` (the website keeps it for a minute).
    func refreshStats(force: Bool) async {
        if !force, let at = statsAt, Date().timeIntervalSince(at) < 15 { return }
        statsAt = Date()
        guard let stats = try? await Cloud.api().stats() else { return }
        reviewWaiting = max(0, stats.misclicks) + max(0, stats.guesses)
    }

    /// Signed out: nothing of the last session is shown.
    func clear() {
        siteTitle = nil
        reviewWaiting = 0
        statsAt = nil
    }

    /// The tab's badge: nothing for 0, "999+" above 999 (the website's CountBadge).
    nonisolated static func badge(_ count: Int) -> String? {
        guard count > 0 else { return nil }
        return count > 999 ? "999+" : String(count)
    }
}

/// The open vault on every screen (the website's VaultBadge in the header): the time left as m:ss, which opens the
/// vault, and a button that locks it at once. Hidden while the vault is locked. VaultSession ends the window on the
/// phone's clock, which hides the bar and drops the vault's pictures wherever the owner is.
struct ShellVaultBar: View {
    @Environment(AppRouter.self) private var router
    @State private var locking = false

    private var vault: VaultSession { VaultSession.shared }

    var body: some View {
        if let until = vault.until {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                let left = VaultClock.secondsLeft(until: until, now: context.date)
                if left > 0 {
                    bar(left: left)
                }
            }
        }
    }

    private func bar(left: Int) -> some View {
        HStack(spacing: 0) {
            Button(action: openVault) {
                HStack(spacing: 6) {
                    Image(systemName: "lock.open.fill")
                        .accessibilityHidden(true)
                    Text("Vault open")
                    Text(Fmt.clock(left))
                        .monospacedDigit()
                }
                .font(.footnote.weight(.semibold))
                .padding(.leading, 12)
                .padding(.trailing, 8)
                .frame(height: 32)
                .contentShape(Rectangle())
            }
            // In whole minutes, so the label changes once a minute rather than every second: VoiceOver does not
            // read a ticking clock, and UI tests do not see the screen's elements change under them every second.
            .accessibilityLabel(Self.spoken(left))
            .accessibilityHint("Opens the vault")
            .accessibilityIdentifier("shell.vault")
            Divider().frame(height: 18)
            Button(action: lock) {
                Group {
                    if locking { ProgressView().controlSize(.small) } else { Image(systemName: "lock.fill") }
                }
                .font(.footnote.weight(.semibold))
                .frame(width: 40, height: 32)
                .contentShape(Rectangle())
            }
            .disabled(locking)
            .accessibilityLabel("Lock the vault now")
            .accessibilityIdentifier("shell.vaultLock")
        }
        .foregroundStyle(.orange)
        .background(Color.orange.opacity(0.12), in: Capsule())
        .overlay(Capsule().strokeBorder(Color.orange.opacity(0.3)))
        .frame(maxWidth: .infinity)
        .padding(.vertical, 4)
        .background(.bar)
    }

    /// "Vault unlocked, 12 minutes left" (rounded up, so the last seconds read as 1 minute).
    nonisolated static func spoken(_ secondsLeft: Int) -> String {
        let minutes = max(1, (max(0, secondsLeft) + 59) / 60)
        return "Vault unlocked, \(minutes) \(minutes == 1 ? "minute" : "minutes") left"
    }

    private func openVault() {
        router.tab = .more
        router.popToRoot(.more)
        router.open(.more(.vault))
    }

    private func lock() {
        guard !locking else { return }
        locking = true
        Task {
            defer { locking = false }
            do {
                try await vault.lock()
                Toaster.shared.info("The vault is locked.")
            } catch {
                if error is CancellationError { return }
                Toaster.shared.error("Could not lock the vault. Try again.")
            }
        }
    }
}
