import SwiftUI

/// Opens the vault (the website's UnlockForm in its compact dialog). Settings presents it when the server answers
/// 403 `vault_locked` (adding a device or a passkey needs the vault open) and runs `onUnlocked` once the vault is
/// open, to try its action again.
struct VaultUnlockSheet: View {
    var onUnlocked: (() -> Void)? = nil

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VaultUnlockForm(compact: true, description: VaultText.sheetText) {
                    onUnlocked?()
                    dismiss()
                }
                .padding(20)
            }
            .navigationTitle("Open the vault")
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
