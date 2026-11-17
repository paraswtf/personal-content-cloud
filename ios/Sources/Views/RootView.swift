import SwiftUI

/// Signed in: the app's tabs. Signed out: the sign-in screen. Toasts show over both.
struct RootView: View {
    @EnvironmentObject private var session: AppSession
    @State private var lastUnauthorized: Date?

    var body: some View {
        Group {
            if session.signedIn {
                MainShell()
            } else {
                SignInView()
            }
        }
        .toastHost()
        .onChange(of: session.signedIn) { _, signedIn in
            ConnectionCache.shared.reset()
            // A signed-out phone keeps nothing of the library: no cached pictures, no undo history, and no open
            // vault window (the next session's vault starts locked, whatever the last one's said).
            if !signedIn {
                ImageLoader.shared.clear()
                UndoCenter.shared.clear()
                VaultSession.shared.expire()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .apiUnauthorized)) { _ in
            guard session.signedIn else { return }
            if let last = lastUnauthorized, Date().timeIntervalSince(last) < 60 { return }
            lastUnauthorized = Date()
            Toaster.shared.error("The server no longer accepts this device.",
                                 action: .init(label: "Sign in") { session.signOut() })
        }
        .task {
            // Keep the picture cache under its limit, away from the main thread.
            Task.detached(priority: .utility) { ImageLoader.shared.trimDisk() }
        }
    }
}

/// Sign-in: the server's address, then a device token (Settings > Sessions > Devices on the website) or the
/// account's username and password, which create a device token for this phone (docs/ARCHITECTURE.md section 4).
struct SignInView: View {
    enum Method: String, CaseIterable, Identifiable {
        case token = "Device token"
        case password = "Password"
        var id: String { rawValue }
    }

    @EnvironmentObject private var session: AppSession
    @State private var method: Method = .token
    @State private var server = AppSettings.serverURL?.absoluteString ?? "https://"
    @State private var token = ""
    @State private var username = ""
    @State private var password = ""
    @State private var deviceName = ""
    @State private var busy = false
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("Server") {
                    TextField("https://your-cloud.example", text: $server)
                        .keyboardType(.URL)
                        .textContentType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .accessibilityIdentifier("signin.server")
                }
                Section {
                    Picker("Sign in with", selection: $method) {
                        ForEach(Method.allCases) { m in Text(m.rawValue).tag(m) }
                    }
                    .pickerStyle(.segmented)
                    .accessibilityIdentifier("signin.method")
                }
                if method == .token {
                    Section {
                        SecureField("Paste the device token", text: $token)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .accessibilityIdentifier("signin.token")
                    } header: {
                        Text("Device token")
                    } footer: {
                        Text("On the website, open Settings, Sessions, Devices and add this phone. The token is shown once.")
                    }
                } else {
                    Section {
                        TextField("Username", text: $username)
                            .textContentType(.username)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .accessibilityIdentifier("signin.username")
                        SecureField("Password", text: $password)
                            .textContentType(.password)
                            .accessibilityIdentifier("signin.password")
                        TextField("This phone's name", text: $deviceName)
                            .accessibilityIdentifier("signin.device")
                    } header: {
                        Text("Account")
                    } footer: {
                        Text("Signing in adds this phone to Settings, Sessions, Devices on the website, where you can remove it.")
                    }
                }
                if let error {
                    Text(error)
                        .foregroundStyle(.red)
                        .accessibilityIdentifier("signin.error")
                }
                Button {
                    Task { await signIn() }
                } label: {
                    if busy { ProgressView() } else { Text("Sign in") }
                }
                .disabled(busy || !canSubmit)
                .accessibilityIdentifier("signin.submit")
            }
            .navigationTitle("Content Cloud")
            .task {
                if deviceName.isEmpty { deviceName = UIDevice.current.name }
            }
        }
    }

    private var canSubmit: Bool {
        switch method {
        case .token: return !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .password: return !username.trimmingCharacters(in: .whitespaces).isEmpty && !password.isEmpty
        }
    }

    private func serverURL() -> URL? {
        let text = server.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: text), let scheme = url.scheme?.lowercased(), scheme == "https" || scheme == "http",
              url.host != nil else { return nil }
        return url
    }

    private func signIn() async {
        guard let url = serverURL() else {
            error = "Enter the server's address, including https://"
            return
        }
        busy = true
        error = nil
        defer { busy = false }
        do {
            let newToken: String
            switch method {
            case .token:
                newToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
            case .password:
                let name = deviceName.trimmingCharacters(in: .whitespacesAndNewlines)
                let created = try await DeviceSignIn.login(base: url, username: username.trimmingCharacters(in: .whitespaces),
                                                           password: password, deviceName: name.isEmpty ? "iPhone" : name)
                newToken = created.token
            }
            let state = try await DeviceSignIn.check(base: url, token: newToken)
            guard state.authenticated else {
                error = "The server did not accept the token."
                return
            }
            password = ""
            session.signIn(server: url, token: newToken)
            ConnectionCache.shared.reset()
            // Photos access is asked for on the upload and sync screens, not here.
            BackgroundSync.schedule()
        } catch {
            self.error = ErrorText.describe(error)
        }
    }
}

/// The sign-in requests, made before the phone has a token in the Keychain.
enum DeviceSignIn {
    private struct LoginBody: Encodable {
        let username: String
        let password: String
        let device_name: String
    }

    /// Signs in with the account's password: the server creates a device token for this phone, shown once.
    static func login(base: URL, username: String, password: String, deviceName: String) async throws -> CoreDeviceCreated {
        let connection = APIConnection(base: base, token: "")
        return try await CoreHTTP.send(CoreDeviceCreated.self, connection, "POST", "/api/auth/device-login",
                                       json: LoginBody(username: username, password: password, device_name: deviceName))
    }

    /// Asks the server whether a token works.
    static func check(base: URL, token: String) async throws -> CoreAuthState {
        try await CoreHTTP.get(CoreAuthState.self, APIConnection(base: base, token: token), "/api/auth/state")
    }
}
