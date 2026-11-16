import SwiftUI

/// Opens the vault (the website's components/vault/UnlockForm.tsx). The vault area builds it; Settings presents it
/// when the server answers 403 `vault_locked` (adding a device or a passkey needs the vault open), and runs
/// `onUnlocked` once the vault is open. Its name and parameters stay. Placeholder until then.
struct VaultUnlockSheet: View {
    var onUnlocked: (() -> Void)? = nil

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            PlaceholderScreen(title: "Open the vault", systemImage: "lock.shield")
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                }
        }
    }
}
