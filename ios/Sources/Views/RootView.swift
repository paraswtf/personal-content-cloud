import SwiftUI

struct RootView: View {
    @EnvironmentObject private var session: AppSession

    var body: some View {
        if session.signedIn {
            TabView {
                SyncView().tabItem { Label("Sync", systemImage: "arrow.triangle.2.circlepath") }
                DeleteSyncedView().tabItem { Label("Delete synced", systemImage: "trash") }
                SettingsView().tabItem { Label("Settings", systemImage: "gearshape") }
            }
        } else {
            SignInView()
        }
    }
}

struct SignInView: View {
    @EnvironmentObject private var session: AppSession
    @State private var server = AppSettings.serverURL?.absoluteString ?? "https://"
    @State private var token = ""
    @State private var busy = false
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("Server") {
                    TextField("https://your-cloud.example", text: $server)
                        .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                }
                Section("Device token") {
                    SecureField("Paste the token from Settings > Sessions > Devices", text: $token)
                    Text("Create one on the website under Settings, Sessions, Devices. It is shown once.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                if let error { Text(error).foregroundStyle(.red) }
                Button(busy ? "Checking" : "Sign in") { Task { await signIn() } }
                    .disabled(busy || token.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .navigationTitle("Content Cloud")
        }
    }

    private func signIn() async {
        guard let url = URL(string: server.trimmingCharacters(in: .whitespaces)), url.scheme != nil, url.host != nil else {
            error = "Enter the server address, including https://"; return
        }
        busy = true; error = nil
        let api = APIClient(base: url, token: token.trimmingCharacters(in: .whitespacesAndNewlines))
        do {
            let state = try await api.state()
            guard state.authenticated else { error = "The token was not accepted"; busy = false; return }
            session.signIn(server: url, token: token.trimmingCharacters(in: .whitespacesAndNewlines))
            _ = await PhotoLibrary.requestAccess()
            BackgroundSync.schedule()
        } catch {
            self.error = error.localizedDescription
        }
        busy = false
    }
}
