import SwiftUI

/// Opens the vault (the website's UnlockForm in its compact dialog). Settings presents it when the server answers
/// 403 `vault_locked` (adding a device or a passkey needs the vault open) and runs `onUnlocked` once the vault is
/// open, to try its action again. Like the website's dialog it can say what it is for: `title` ("Confirm it is you")
/// and `reason` ("To add a device, unlock the vault with your password."); without them it reads as before.
struct VaultUnlockSheet: View {
    var title: String? = nil
    var reason: String? = nil
    var onUnlocked: (() -> Void)? = nil

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VaultUnlockForm(compact: true, description: reason ?? VaultText.sheetText) {
                    onUnlocked?()
                    dismiss()
                }
                .padding(20)
            }
            .navigationTitle(title ?? "Open the vault")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                        .accessibilityIdentifier("vault.sheet.cancel")
                }
            }
        }
        .presentationDetents([.medium, .large])
        .task { if !VaultSession.shared.loaded { await VaultSession.shared.refresh() } }
    }
}
