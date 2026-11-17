import SwiftUI

/// Settings > Wallet (the website's WalletTab): whether Apple Wallet passes can be made from event tickets and
/// boarding passes, and if not, exactly what to put where on the server. Secrets never go through this screen: the
/// owner places files in the wallet folder and sets environment variables on the server.
struct SettingsWalletScreen: View {
    @State private var status: SettingsWalletStatus?
    @State private var loadError: Error?

    var body: some View {
        List {
            if let status {
                statusSection(status)
                if !status.configured {
                    setupSection(status)
                }
            } else if let loadError {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(ErrorText.describe(loadError, doing: "load the wallet status"))
                            .foregroundStyle(.red)
                        Button("Try again") { Task { await load() } }
                            .buttonStyle(.bordered)
                    }
                }
            } else {
                Section {
                    HStack {
                        Spacer()
                        ProgressView("Loading wallet status")
                        Spacer()
                    }
                }
            }
        }
        .navigationTitle("Wallet")
        .task { await load() }
        .refreshable { await load() }
    }

    private func statusSection(_ status: SettingsWalletStatus) -> some View {
        Section {
            HStack {
                Label("Apple Wallet", systemImage: "wallet.pass")
                    .font(.headline)
                Spacer()
                Chip(text: status.configured ? "Ready" : "Not set up", tint: status.configured ? .green : .orange)
                    .accessibilityIdentifier("wallet.status")
            }
            Text("Event tickets and boarding passes found among your documents can be added to Apple Wallet. Passes must be signed with your own Apple Developer certificate.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            if status.configured {
                Label("Wallet passes are set up. Open a pass in Documents and choose Add to Apple Wallet.",
                      systemImage: "checkmark.circle.fill")
                    .font(.subheadline)
                    .foregroundStyle(.green)
            } else {
                Label("Still missing:", systemImage: "exclamationmark.circle")
                    .font(.subheadline)
                    .foregroundStyle(.orange)
                ForEach(status.missing, id: \.self) { item in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(item)
                            .font(.system(.subheadline, design: .monospaced))
                            .textSelection(.enabled)
                        if let help = SettingsWalletStatus.help[item] {
                            Text(help)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }

    private func setupSection(_ status: SettingsWalletStatus) -> some View {
        Section {
            step(1, "In the Apple Developer portal, open Certificates, Identifiers and Profiles, then Identifiers, and register a Pass Type ID (for example pass.com.example.cloud).")
            step(2, "Create a Pass Type ID certificate for it, download it, open it in Keychain Access on a Mac and export it with its private key as pass.p12.")
            step(3, "Convert it to PEM files (OpenSSL 3 may need -legacy for files exported by Keychain):",
                 code: "openssl pkcs12 -in pass.p12 -clcerts -nokeys -out cert.pem\nopenssl pkcs12 -in pass.p12 -nocerts -nodes -out key.pem")
            step(4, "Download the Apple WWDR intermediate certificate (G4) from Apple PKI and convert it:",
                 code: "openssl x509 -inform der -in AppleWWDRCAG4.cer -out wwdr.pem")
            step(5, "Put cert.pem, key.pem and wwdr.pem in the wallet folder on the server and make the key readable only by the server user:",
                 code: "\(status.dir)/cert.pem\n\(status.dir)/key.pem      (chmod 600)\n\(status.dir)/wwdr.pem")
            step(6, "Set these environment variables for the server, then restart it:",
                 code: "WALLET_PASS_TYPE_ID=pass.com.example.cloud\nWALLET_TEAM_ID=ABCDE12345\nWALLET_ORG_NAME=My Cloud            (optional)")
            step(7, "Come back to this screen. It shows Ready once every item is in place.")
        } header: {
            Text("How to set it up")
        } footer: {
            VStack(alignment: .leading, spacing: 8) {
                Text("Everything happens on the server and in your Apple Developer account. This screen never asks for keys or passwords.")
                Text("The pass is built and signed on this server; nothing is sent to Apple until you add the pass on your device.")
            }
        }
    }

    private func step(_ n: Int, _ text: String, code: String? = nil) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text("\(n).")
                .font(.subheadline.weight(.semibold))
                .monospacedDigit()
            VStack(alignment: .leading, spacing: 6) {
                Text(text)
                    .font(.subheadline)
                if let code {
                    Text(code)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(uiColor: .tertiarySystemFill), in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
        .padding(.vertical, 2)
    }

    private func load() async {
        do {
            let system = try await Cloud.api().system()
            status = SettingsWalletStatus(system.wallet)
            loadError = nil
        } catch {
            if error is CancellationError { return }
            loadError = error
        }
    }
}

/// The wallet part of GET /api/system: whether passes can be made and what is still missing.
struct SettingsWalletStatus: Equatable {
    let configured: Bool
    let missing: [String]
    /// The wallet folder on the server (the server does not say it today, so the website's placeholder).
    let dir: String

    init(_ wallet: [String: API.JSONValue]) {
        configured = wallet["configured"]?.boolValue ?? false
        missing = wallet["missing"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let folder = wallet["dir"]?.stringValue ?? ""
        dir = folder.isEmpty ? "DATA_DIR/wallet" : folder
    }

    /// What each missing setting is (the website's ENV_HELP).
    static let help: [String: String] = [
        "WALLET_PASS_TYPE_ID": "the Pass Type ID you registered, for example pass.com.example.cloud",
        "WALLET_TEAM_ID": "your 10-character Apple Developer Team ID (Membership details)",
        "WALLET_ORG_NAME": "the name shown on passes (optional; defaults to the site title)",
    ]
}
