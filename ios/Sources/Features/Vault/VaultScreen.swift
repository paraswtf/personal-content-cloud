import SwiftUI

/// The vault (the website's VaultPage): private items. Locked, it shows only the unlock form and asks for nothing
/// else (no lists, no pictures). Open, it shows the library of private items (LibraryBrowser with the vault scope,
/// every request with vault=1) under a bar with the time left and Lock now. When the window ends, on this phone's
/// clock or because the server says so, it asks for the password again.
struct VaultScreen: View {
    @Environment(\.scenePhase) private var scenePhase
    @Environment(AppRouter.self) private var router
    @State private var locking = false

    private var vault: VaultSession { VaultSession.shared }

    var body: some View {
        content
            .task { await vault.refresh() }
            // Sleeps until the window ends, then asks the server again (the website's page does the same).
            .task(id: vault.until) {
                guard let until = vault.until else { return }
                let wait = until.timeIntervalSinceNow
                if wait > 0 { try? await Task.sleep(for: .seconds(wait + 0.5)) }
                guard !Task.isCancelled else { return }
                await vault.refresh()
                if !vault.isOpen() { vault.expire() }
            }
            .onChange(of: scenePhase) { _, phase in
                if phase == .active { Task { await vault.refresh() } }
            }
            // a vault change elsewhere (an undo, a lock in Settings); a vault list request answered 403 vault_locked
            // locks the screen itself (LibraryModel.call, VaultSession.foundLocked)
            .onCloudChange { Task { await vault.refresh() } }
            // Leaving the vault drops its pictures from memory (the website's VaultPage drops the vault data); opening
            // an item in the viewer over it does not.
            .onDisappear {
                if router.viewer == nil { ImageLoader.shared.forgetVault() }
            }
    }

    @ViewBuilder private var content: some View {
        // the open vault's browser sets the title itself ("Search results" while searching)
        if !vault.loaded {
            LoadingState()
                .navigationTitle("Vault")
        } else if let until = vault.until, vault.isOpen() {
            LibraryBrowser(scope: .vault,
                           header: AnyView(VaultOpenBar(until: until, locking: locking, onLock: lock)),
                           title: "Vault")
        } else if let error = vault.loadError {
            ErrorState(error: error) { Task { await vault.refresh() } }
                .navigationTitle("Vault")
        } else {
            ScrollView {
                VaultUnlockForm()
                    .padding(.horizontal, 24)
                    .padding(.vertical, 40)
                    .frame(maxWidth: .infinity)
            }
            .refreshable { await vault.refresh() }
            .navigationTitle("Vault")
        }
    }

    private func lock() {
        guard !locking else { return }
        locking = true
        Task {
            defer { locking = false }
            do {
                try await vault.lock()
                Toaster.shared.ok("Vault locked")
            } catch {
                Toaster.shared.error(error, doing: "lock the vault")
            }
        }
    }
}

/// The bar over the open vault: how long it stays open (the website's "Unlocked for 14 min", counted every second
/// like its header badge) and Lock now.
struct VaultOpenBar: View {
    let until: Date
    let locking: Bool
    let onLock: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                let left = VaultClock.secondsLeft(until: until, now: context.date)
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.shield.fill")
                        .foregroundStyle(.green)
                        .accessibilityHidden(true)
                    Text(VaultClock.openText(secondsLeft: left))
                        .font(.subheadline.weight(.medium).monospacedDigit())
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                }
                .padding(.horizontal, 12)
                .frame(height: 36)
                .background(Color(.secondarySystemBackground), in: Capsule())
                .accessibilityElement(children: .combine)
                .accessibilityLabel(VaultClock.openText(secondsLeft: left))
                .accessibilityIdentifier("vault.status")
            }
            Spacer(minLength: 0)
            Button(action: onLock) {
                HStack(spacing: 6) {
                    if locking { ProgressView() } else { Image(systemName: "lock") }
                    Text("Lock now")
                }
            }
            .buttonStyle(.bordered)
            .disabled(locking)
            .accessibilityLabel("Lock now")
            .accessibilityIdentifier("vault.lock")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(.bar)
    }
}
