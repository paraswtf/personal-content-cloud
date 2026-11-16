import SwiftUI

/// Settings, in the website's sections (pages/SettingsPage.tsx). This list is shared by two areas: the account
/// screens are in Features/Settings/Account/, the system screens in Features/Settings/System/, and Share links
/// belongs to the albums area. The areas fill the screens; this list stays as it is.
struct SettingsScreen: View {
    @EnvironmentObject private var session: AppSession
    @State private var confirmSignOut = false

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
                LabeledContent("Address", value: session.serverURL?.absoluteString ?? "")
                Button("Sign out", role: .destructive) { confirmSignOut = true }
                    .accessibilityIdentifier("settings.signOut")
            }
        }
        .navigationTitle("Settings")
        .confirmAction("Sign out of this server?", isPresented: $confirmSignOut, actionTitle: "Sign out") {
            session.signOut()
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
