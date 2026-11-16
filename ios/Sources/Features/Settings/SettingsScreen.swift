import SwiftUI

/// Settings. Placeholder of the core (the settings area replaces this file): the server and signing out.
struct SettingsScreen: View {
    @EnvironmentObject private var session: AppSession
    @State private var confirmSignOut = false

    var body: some View {
        List {
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
}
