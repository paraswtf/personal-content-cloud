import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var session: AppSession
    @State private var wifiOnly = AppSettings.wifiOnly
    @State private var chargingOnly = AppSettings.chargingOnly
    @State private var includeVideos = AppSettings.includeVideos
    @State private var parallel = AppSettings.parallelUploads
    @State private var autoSync = AppSettings.autoSync
    @State private var onlyRecent = AppSettings.syncSince != nil
    @State private var since = AppSettings.syncSince ?? Date()
    @State private var confirmSignOut = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Server") {
                    LabeledContent("Address", value: session.serverURL?.absoluteString ?? "")
                    Button("Sign out", role: .destructive) { confirmSignOut = true }
                }
                Section("Sync") {
                    Toggle("Sync automatically", isOn: $autoSync).onChange(of: autoSync) { _, v in AppSettings.autoSync = v; if v { BackgroundSync.schedule() } }
                    Toggle("Only on Wi-Fi", isOn: $wifiOnly).onChange(of: wifiOnly) { _, v in AppSettings.wifiOnly = v; BackgroundUploader.shared.settingsChanged() }
                    Toggle("Only while charging", isOn: $chargingOnly).onChange(of: chargingOnly) { _, v in AppSettings.chargingOnly = v; BackgroundSync.schedule(); BackgroundUploader.shared.settingsChanged() }
                    Toggle("Include videos", isOn: $includeVideos).onChange(of: includeVideos) { _, v in AppSettings.includeVideos = v }
                    Stepper("Files at a time: \(parallel)", value: $parallel, in: 1...AppSettings.maxParallelUploads)
                        .onChange(of: parallel) { _, v in AppSettings.parallelUploads = v }
                    Toggle("Only items taken after a date", isOn: $onlyRecent).onChange(of: onlyRecent) { _, v in AppSettings.syncSince = v ? since : nil }
                    if onlyRecent {
                        DatePicker("Taken after", selection: $since, displayedComponents: .date)
                            .onChange(of: since) { _, v in AppSettings.syncSince = v }
                    }
                }
                Section {
                    Text("Uploads continue in the background after you leave the app. With Wi-Fi only or charging only on, iOS may hold them until the phone is on Wi-Fi and plugged in.")
                        .font(.footnote).foregroundStyle(.secondary)
                    Text("Originals are uploaded as they are and kept on the server at full quality. Nothing leaves your own server.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .confirmationDialog("Sign out of this server?", isPresented: $confirmSignOut, titleVisibility: .visible) {
                Button("Sign out", role: .destructive) { session.signOut() }
            }
        }
    }
}
