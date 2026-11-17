import SwiftUI

/// Settings, in the website's sections (pages/SettingsPage.tsx). This list is shared by two areas: the account
/// screens are in Features/Settings/Account/, the system screens in Features/Settings/System/, and Share links
/// belongs to the albums area. The areas fill the screens; this list stays as it is.
struct SettingsScreen: View {
    @EnvironmentObject private var session: AppSession
    @State private var confirmSignOut = false
    @State private var signingOut = false
    @State private var offerPhoneSignOut = false

    var body: some View {
        List {
            Section {
                link("Account", "person.crop.circle", "settings.account") { SettingsAccountScreen() }
                link("Passkeys", "person.badge.key", "settings.passkeys") { SettingsPasskeysScreen() }
                link("Sessions and devices", "laptopcomputer.and.iphone", "settings.sessions") { SettingsSessionsScreen() }
            }
            Section {
                link("System", "server.rack", "settings.system") { SettingsSystemScreen() }
                link("Learning", "brain", "settings.learning") { SettingsLearningScreen() }
                link("Undo history", "arrow.uturn.backward.circle", "settings.undo") { SettingsUndoHistoryScreen() }
            }
            Section {
                link("Share links", "link", "settings.sharing") { SharingLinksScreen() }
                link("Wallet", "wallet.pass", "settings.wallet") { SettingsWalletScreen() }
                link("Appearance", "circle.lefthalf.filled", "settings.appearance") { SettingsAppearanceScreen() }
            }
            Section("Server") {
                if let name = ShellStatus.shared.siteTitle, !name.isEmpty {
                    LabeledContent("Name", value: name)
                        .accessibilityIdentifier("settings.siteTitle")
                }
                LabeledContent("Address", value: session.serverURL?.absoluteString ?? "")
                Button(role: .destructive) {
                    confirmSignOut = true
                } label: {
                    HStack(spacing: 8) {
                        Text("Sign out")
                        if signingOut { ProgressView() }
                    }
                }
                .disabled(signingOut)
                .accessibilityIdentifier("settings.signOut")
            }
        }
        .navigationTitle("Settings")
        .confirmAction("Sign out of this server?", isPresented: $confirmSignOut, actionTitle: "Sign out") {
            Task { await signOut() }
        }
        .alert("The server could not be reached", isPresented: $offerPhoneSignOut) {
            Button("Sign out on this phone", role: .destructive) { session.signOut() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The server still accepts this phone until you remove it under Sessions and devices on the website. "
                 + "Sign out on this phone anyway?")
        }
    }

    /// Ends the session on the server first, as the website's Sign out does, then forgets the token.
    private func signOut() async {
        guard !signingOut else { return }
        signingOut = true
        defer { signingOut = false }
        switch await session.signOutOnServer() {
        case .done, .cancelled: break
        case .unreachable: offerPhoneSignOut = true
        case .failed: Toaster.shared.error("Could not sign out. Try again.")
        }
    }

    private func link<Destination: View>(_ title: String, _ systemImage: String, _ identifier: String,
                                         @ViewBuilder destination: @escaping () -> Destination) -> some View {
        NavigationLink {
            destination()
        } label: {
            Label(title, systemImage: systemImage)
        }
        .accessibilityIdentifier(identifier)
    }
}
