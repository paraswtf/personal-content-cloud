import SwiftUI
import UIKit

/// Settings > Sessions and devices (the website's SessionsTab): every signed-in browser, which can be revoked one by
/// one, and the device tokens the iOS app signs in with, this phone's marked. Adding a device shows its token once
/// and needs the vault open (on 403 `vault_locked` the vault's unlock sheet opens and the add runs again once);
/// ending a device signs that app out, and ending this phone's signs this phone out.
struct SettingsSessionsScreen: View {
    @EnvironmentObject private var session: AppSession

    @State private var sessions: [API.SessionOut] = []
    @State private var devices: [API.SessionOut] = []
    @State private var loaded = false
    @State private var loadError: Error?
    @State private var revoking: API.SessionOut?
    @State private var ending: API.SessionOut?
    @State private var deviceName = "My iPhone"
    @State private var adding = false
    @State private var created: SettingsAccountNewDevice?
    /// The add waiting for the vault, and what to run once the unlock sheet has gone.
    @State private var pending: SettingsAccountPending?
    @State private var resume: SettingsAccountPending?
    @State private var asked: SettingsAccountPending?

    var body: some View {
        List {
            Section {
                if !loaded && loadError == nil {
                    loadingRow("Loading sessions")
                } else if let loadError, sessions.isEmpty, devices.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(ErrorText.describe(loadError, doing: "load the sessions"))
                            .foregroundStyle(.red)
                        Button("Try again") { Task { await load() } }
                            .buttonStyle(.bordered)
                    }
                } else if sessions.isEmpty {
                    Text("No browser is signed in.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(sortedSessions, id: \.id) { s in
                        sessionRow(s)
                    }
                }
            } header: {
                Text("Signed-in sessions")
            } footer: {
                Text("Sessions last 30 days from their last use. End any you do not recognise.")
            }

            Section {
                HStack(spacing: 8) {
                    TextField("Device name", text: $deviceName, prompt: Text("For example, My iPhone"))
                        .textInputAutocapitalization(.words)
                        .submitLabel(.done)
                        .onSubmit { add() }
                        .onChange(of: deviceName) { _, value in
                            if value.count > SettingsAccountText.deviceNameMax {
                                deviceName = String(value.prefix(SettingsAccountText.deviceNameMax))
                            }
                        }
                        .accessibilityLabel("Device name")
                        .accessibilityIdentifier("sessions.deviceName")
                    Button {
                        add()
                    } label: {
                        if adding {
                            ProgressView()
                        } else {
                            Label("Add device", systemImage: "plus")
                        }
                    }
                    .buttonStyle(.borderless)
                    .disabled(adding || deviceName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityIdentifier("sessions.addDevice")
                }
                if loaded && devices.isEmpty && loadError == nil {
                    Text("No devices yet.")
                        .foregroundStyle(.secondary)
                }
                ForEach(sortedDevices, id: \.id) { d in
                    deviceRow(d)
                }
            } header: {
                Text("Devices")
            } footer: {
                Text("A device token lets the iOS app sign in and sync your photos. It keeps working for a year from its last use, and you can end it here at any time.")
            }
        }
        .navigationTitle("Sessions and devices")
        .task { await load() }
        .refreshable { await load() }
        .confirmationDialog(revokeTitle, isPresented: revokeShown, titleVisibility: .visible,
                            presenting: revoking) { s in
            Button(s.current ? "Sign out" : "Revoke", role: s.current ? nil : .destructive) {
                Task { await revoke(s) }
            }
            Button("Cancel", role: .cancel) {}
        } message: { s in
            Text(revokeMessage(s))
        }
        .confirmationDialog(endTitle, isPresented: endShown, titleVisibility: .visible,
                            presenting: ending) { d in
            Button(d.current ? "Sign out" : "End", role: .destructive) {
                Task { await end(d) }
            }
            Button("Cancel", role: .cancel) {}
        } message: { d in
            Text(endMessage(d))
        }
        .sheet(item: $created) { device in
            SettingsAccountTokenSheet(device: device)
        }
        .sheet(item: $pending, onDismiss: unlockSheetGone) { action in
            VaultUnlockSheet(onUnlocked: {
                resume = action
                pending = nil
            })
        }
    }

    /// This browser first (never the app: its session is a device), then the most recently used.
    private var sortedSessions: [API.SessionOut] {
        sessions.sorted { a, b in
            if a.current != b.current { return a.current }
            return a.lastSeenAt > b.lastSeenAt
        }
    }

    /// This phone first, then the most recently used.
    private var sortedDevices: [API.SessionOut] {
        devices.sorted { a, b in
            if a.current != b.current { return a.current }
            return a.lastSeenAt > b.lastSeenAt
        }
    }

    private var revokeTitle: String {
        revoking?.current == true ? "Sign out of this device" : "Revoke session"
    }

    private func revokeMessage(_ s: API.SessionOut) -> String {
        if s.current { return "You will need the password or a passkey to sign in again." }
        return "\(SettingsAccountText.describeAgent(s.userAgent)) will be signed out at once."
    }

    private var endTitle: String {
        ending?.current == true ? "Sign out of this phone" : "End device"
    }

    private func endMessage(_ d: API.SessionOut) -> String {
        if d.current { return "This phone will need the password or a new device token to sign in again." }
        return "\(SettingsAccountText.deviceLabel(d)) will be signed out at once. The app on it stops syncing until you add it again with a new token."
    }

    private var revokeShown: Binding<Bool> {
        Binding(get: { revoking != nil }, set: { if !$0 { revoking = nil } })
    }

    private var endShown: Binding<Bool> {
        Binding(get: { ending != nil }, set: { if !$0 { ending = nil } })
    }

    private func loadingRow(_ text: String) -> some View {
        HStack {
            Spacer()
            ProgressView(text)
            Spacer()
        }
    }

    private func sessionRow(_ s: API.SessionOut) -> some View {
        let agent = SettingsAccountText.describeAgent(s.userAgent)
        return HStack(spacing: 12) {
            Image(systemName: SettingsAccountText.isPhone(s.userAgent) ? "iphone" : "laptopcomputer")
                .foregroundStyle(.secondary)
                .frame(width: 28)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text(agent)
                    .fontWeight(.semibold)
                if s.current {
                    Chip(text: "This device", tint: .green)
                }
                Text(SettingsAccountText.sessionLine(s))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            Button(s.current ? "Sign out" : "Revoke", role: s.current ? nil : .destructive) {
                revoking = s
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(s.current ? "Sign out of this device" : "Revoke \(agent)")
            .accessibilityIdentifier("sessions.session.revoke")
        }
        .padding(.vertical, 2)
    }

    private func deviceRow(_ d: API.SessionOut) -> some View {
        let label = SettingsAccountText.deviceLabel(d)
        return HStack(spacing: 12) {
            Image(systemName: "iphone.gen3")
                .foregroundStyle(.secondary)
                .frame(width: 28)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text(label)
                    .fontWeight(.semibold)
                    .lineLimit(2)
                if d.current {
                    Chip(text: "This device", tint: .green)
                }
                Text(SettingsAccountText.deviceLine(d))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            Button(d.current ? "Sign out" : "End", role: .destructive) {
                ending = d
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(d.current ? "Sign out of this phone" : "End \(label)")
            .accessibilityIdentifier("sessions.device.end")
        }
        .padding(.vertical, 2)
    }

    private func load() async {
        do {
            let api = try Cloud.api()
            async let browsers = api.listSessions()
            async let tokens = api.listDevices()
            sessions = try await browsers
            devices = try await tokens
            loadError = nil
        } catch {
            if error is CancellationError { return }
            loadError = error
        }
        loaded = true
    }

    private func revoke(_ s: API.SessionOut) async {
        let agent = SettingsAccountText.describeAgent(s.userAgent)
        do {
            try await UndoCenter.shared.perform("Session ended", toast: false) {
                _ = try await Cloud.api().revokeSession(sessionId: s.id)
            }
            if s.current {
                session.signOut()
                return
            }
            Toaster.shared.ok("\(agent) was signed out.")
            await load()
        } catch {
            Toaster.shared.error(error, doing: "end that session")
        }
    }

    private func end(_ d: API.SessionOut) async {
        let label = SettingsAccountText.deviceLabel(d)
        do {
            try await UndoCenter.shared.perform("Device ended", toast: false) {
                _ = try await Cloud.api().revokeDevice(sessionId: d.id)
            }
            if d.current {
                session.signOut()
                return
            }
            Toaster.shared.ok("\(label) was signed out.")
            await load()
        } catch {
            Toaster.shared.error(error, doing: "end that device")
        }
    }

    private func add() {
        let clean = SettingsAccountText.cleanName(deviceName, max: SettingsAccountText.deviceNameMax)
        guard !clean.isEmpty, !adding else { return }
        settingsAccountEndEditing()
        Task { await runAdd(clean, retried: false) }
    }

    /// `retried` is true for the one run after the vault was opened: a second refusal is shown, not asked again.
    private func runAdd(_ name: String, retried: Bool) async {
        adding = true
        defer { adding = false }
        do {
            let device = try await UndoCenter.shared.perform("Device added", toast: false) {
                try await Cloud.api().createDevice(body: API.DeviceIn(name: name))
            }
            created = SettingsAccountNewDevice(id: device.id, name: device.name, token: device.token)
            await load()
        } catch {
            if !retried && SettingsAccountText.isVaultLocked(error) {
                asked = .addDevice(name: name)
                pending = .addDevice(name: name)
            } else if SettingsAccountText.isVaultLocked(error) {
                Toaster.shared.error(SettingsAccountText.vaultStillLocked)
            } else {
                Toaster.shared.error(error, doing: "add the device")
            }
        }
    }

    /// The unlock sheet has gone: add the device again if the vault was opened, or say it was not added.
    private func unlockSheetGone() {
        let wasAsked = asked
        asked = nil
        if let action = resume {
            resume = nil
            if case .addDevice(let name) = action {
                Task { await runAdd(name, retried: true) }
            }
        } else if case .addDevice = wasAsked {
            Toaster.shared.error("The device was not added: the vault stayed locked.")
        }
    }
}

/// A device just added, with its token (shown once).
struct SettingsAccountNewDevice: Identifiable, Equatable {
    let id: Int
    let name: String
    let token: String
}

/// A new device token, shown once with a Copy button (the website's TokenModal).
struct SettingsAccountTokenSheet: View {
    let device: SettingsAccountNewDevice

    @Environment(\.dismiss) private var dismiss
    @State private var copied = false

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("Enter this token in the iOS app on \(device.name). It will not be shown again.")
                        .foregroundStyle(.secondary)
                    Text(device.token)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .accessibilityIdentifier("sessions.token")
                    Button {
                        UIPasteboard.general.string = device.token
                        copied = true
                        Task {
                            try? await Task.sleep(nanoseconds: 2_000_000_000)
                            copied = false
                        }
                    } label: {
                        Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                    }
                    .accessibilityIdentifier("sessions.tokenCopy")
                }
            }
            .navigationTitle("Device token")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                        .accessibilityIdentifier("sessions.tokenDone")
                }
            }
        }
    }
}
