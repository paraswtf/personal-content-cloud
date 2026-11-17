import SwiftUI

/// Settings > Passkeys (the website's PasskeysTab): the account's passkeys, to rename and remove. Removing one needs
/// the vault open: when the server answers 403 `vault_locked`, the vault's unlock sheet opens and the removal runs
/// again once. The app cannot make a passkey for a server it has no associated domain for (and an app for any
/// self-hosted address cannot have one), so adding one happens on the website, as the website's own note says for
/// browsers that cannot use passkeys on an address.
struct SettingsPasskeysScreen: View {
    @EnvironmentObject private var session: AppSession

    @State private var keys: [API.PasskeyOut] = []
    @State private var loaded = false
    @State private var loadError: Error?
    @State private var home: URL?
    @State private var renaming: API.PasskeyOut?
    @State private var newName = ""
    @State private var removing: API.PasskeyOut?
    /// The removal waiting for the vault, and what to run once the unlock sheet has gone.
    @State private var pending: SettingsAccountPending?
    @State private var resume: SettingsAccountPending?

    var body: some View {
        List {
            Section {
                addNote
            } header: {
                Text("Add a passkey")
            } footer: {
                Text("A passkey signs you in with a device's fingerprint, face or screen lock, without typing the password. It also unlocks the vault. Adding or removing a passkey asks for the password or a passkey again.")
            }

            Section {
                if !loaded && loadError == nil {
                    HStack {
                        Spacer()
                        ProgressView("Loading passkeys")
                        Spacer()
                    }
                } else if let loadError, keys.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(ErrorText.describe(loadError, doing: "load the passkeys"))
                            .foregroundStyle(.red)
                        Button("Try again") { Task { await load() } }
                            .buttonStyle(.bordered)
                    }
                } else if keys.isEmpty {
                    HStack(spacing: 8) {
                        Image(systemName: "key")
                            .accessibilityHidden(true)
                        Text("No passkeys yet.")
                    }
                    .foregroundStyle(.secondary)
                } else {
                    ForEach(keys, id: \.id) { key in
                        row(key)
                    }
                }
            } header: {
                Text("Your passkeys")
            } footer: {
                Text("The password always works too, so removing every passkey never locks you out.")
            }
        }
        .navigationTitle("Passkeys")
        .task { await load() }
        .refreshable { await load() }
        .alert("Rename passkey", isPresented: renameShown, presenting: renaming) { key in
            TextField("Name", text: $newName)
                .accessibilityIdentifier("passkeys.renameField")
            Button("Cancel", role: .cancel) {}
            Button("Save") {
                let name = newName
                Task { await rename(key, to: name) }
            }
        }
        .confirmationDialog("Remove passkey", isPresented: removeShown, titleVisibility: .visible,
                            presenting: removing) { key in
            Button("Remove", role: .destructive) { Task { await remove(key, retried: false) } }
            Button("Cancel", role: .cancel) {}
        } message: { key in
            Text("\(key.name) will no longer sign in here. Also remove it from the device or password manager that holds it, so it is not offered again.")
        }
        .sheet(item: $pending, onDismiss: unlockSheetGone) { action in
            VaultUnlockSheet(title: SettingsAccountText.unlockTitle, reason: SettingsAccountText.unlockReason(action),
                             onUnlocked: {
                                 resume = action
                                 pending = nil
                             })
        }
    }

    /// Where passkeys are added: the server's website (its own address, where passkeys work).
    @ViewBuilder
    private var addNote: some View {
        let host = home?.host() ?? session.serverURL?.host() ?? "your server"
        VStack(alignment: .leading, spacing: 10) {
            Label {
                Text("This app can't make passkeys. To add one, sign in on the website at \(host).")
            } icon: {
                Image(systemName: "info.circle")
            }
            .font(.subheadline)
            .foregroundStyle(.secondary)
            if let url = addURL {
                Link(destination: url) {
                    Label("Open the website", systemImage: "safari")
                }
                .accessibilityIdentifier("passkeys.addOnWebsite")
            }
        }
        .padding(.vertical, 4)
    }

    /// The website's passkeys settings at the server's own address.
    private var addURL: URL? {
        guard let base = home ?? session.serverURL,
              var parts = URLComponents(url: base, resolvingAgainstBaseURL: false) else { return nil }
        parts.path = "/settings"
        parts.queryItems = [URLQueryItem(name: "tab", value: "passkeys")]
        return parts.url
    }

    private func row(_ key: API.PasskeyOut) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "person.badge.key")
                .foregroundStyle(.secondary)
                .frame(width: 28)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(key.name)
                        .fontWeight(.semibold)
                        .lineLimit(1)
                    if key.backupEligible {
                        Chip(text: "Synced")
                    }
                }
                Text(SettingsAccountText.passkeyLine(key))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            Button {
                newName = key.name
                renaming = key
            } label: {
                Image(systemName: "pencil")
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.borderless)
            .accessibilityLabel("Rename \(key.name)")
            .accessibilityIdentifier("passkeys.rename")
            Button(role: .destructive) {
                removing = key
            } label: {
                Image(systemName: "trash")
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.borderless)
            .accessibilityLabel("Remove \(key.name)")
            .accessibilityIdentifier("passkeys.remove")
        }
        .padding(.vertical, 2)
    }

    private var renameShown: Binding<Bool> {
        Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })
    }

    private var removeShown: Binding<Bool> {
        Binding(get: { removing != nil }, set: { if !$0 { removing = nil } })
    }

    private func load() async {
        do {
            let api = try Cloud.api()
            async let list = api.listPasskeys()
            async let place = api.relyingParty()
            keys = try await list
            if let rp = try? await place, let url = URL(string: rp.homeUrl), url.host() != nil {
                home = url
            }
            loadError = nil
        } catch {
            if error is CancellationError { return }
            loadError = error
        }
        loaded = true
    }

    private func rename(_ key: API.PasskeyOut, to text: String) async {
        let clean = SettingsAccountText.cleanName(text, max: SettingsAccountText.passkeyNameMax)
        guard !clean.isEmpty else { return }
        do {
            try await UndoCenter.shared.perform("Passkey renamed", toast: false) {
                _ = try await Cloud.api().passkeyRename(passkeyId: key.id, body: API.PasskeyRenameIn(name: clean))
            }
            await load()
        } catch {
            Toaster.shared.error(error, doing: "rename the passkey")
        }
    }

    /// `retried` is true for the one run after the vault was opened: a second refusal is shown, not asked again.
    private func remove(_ key: API.PasskeyOut, retried: Bool) async {
        do {
            try await UndoCenter.shared.perform("Passkey removed", toast: false) {
                _ = try await Cloud.api().passkeyDelete(passkeyId: key.id)
            }
            Toaster.shared.ok("Passkey \"\(key.name)\" removed.")
            await load()
        } catch {
            if !retried && SettingsAccountText.isVaultLocked(error) {
                pending = .removePasskey(key)
            } else if SettingsAccountText.isVaultLocked(error) {
                Toaster.shared.error(SettingsAccountText.vaultStillLocked)
            } else {
                Toaster.shared.error(error, doing: "remove the passkey")
            }
        }
    }

    /// The unlock sheet has gone: run the removal again if the vault was opened.
    private func unlockSheetGone() {
        guard let action = resume else { return }
        resume = nil
        if case .removePasskey(let key) = action {
            Task { await remove(key, retried: true) }
        }
    }
}
